import frams
import numpy as np
from pyemd import emd
from ctypes import cdll
from ctypes.util import find_library

class VoxelsEMD:
    libm = cdll.LoadLibrary(find_library('m'))
    frams = frams
    density = 10 
    steps = 3
    verbose = False 
    reduce = True
    EPSILON = 0.0001
    def __init__(self, FramsPath=None,FramsLib=None, density = 10, steps = 3, reduce=True, verbose=False):
        """ __init__
        Args:
            FramsPath (string): - path to Framstick CLI
            density (int, optional): density of samplings for frams.ModelGeometry . Defaults to 10.
            steps (int, optional): How many steps is used for sampling space of voxels, 
                The higher value the more accurate sampling and the longer calculations. Defaults to 3.
            reduce (bool, optional): If we should use reduction to remove blank samples. Defaults to True.
            verbose (bool, optional): Turning on logging, works only for calculateEMDforGeno. Defaults to False.            
        """
        assert FramsPath != None or FramsLib != None , "You must specify FramsPath or FramsLib"
        if FramsLib != None:
            self.frams = FramsLib
        elif FramsPath != None:
            self.frams.init(FramsPath)
            
        self.density = density
        self.steps = steps
        self.verbose = verbose
        self.reduce = reduce

    def calculateNeighberhood(self,array,mean_coords):
        """ Calculates number of elements for given sample and set ups the center of this sample 
        to the center of mass (calculated by mean of every coordinate)
        Args:
            array ([[float,float,float],...,[float,float,float]]): array of voxels that belong to given sample. 
            mean_coords ([float,float,float]): default coordinates that are the 
                middle of the sample (used when number of voxels in sample is equal to 0) 

        Returns:
            weight [int]: number of voxels in a sample
            coordinates [float,float,float]: center of mass for a sample 
        """
        weight = len(array)
        if weight > 0:
            point = [np.mean(array[:,0]),np.mean(array[:,1]),np.mean(array[:,2])]
            return weight, point
        else:
            return 0,mean_coords

    def calculateDistPoints(self,point1, point2):
        """ Returns euclidean distance between two points
        Args:
            point1 ([float,float,float]) - coordinates of first point
            point2 ([float,float,float]) - coordinates of second point

        Returns:
            [int]: euclidean distance
        """
        return np.sqrt(np.sum(np.square(point1-point2)))

    def calculateDistanceMatrix(self,array1, array2):
        """

        Args:
            array1 ([type]): array of size n with points representing firsts model 
            array2 ([type]): array of size n with points representing second model

        Returns:
            np.array(np.array(,dtype=float)): distance matrix n x n 
        """
        distMatrix = []
        for v1 in array1:
            distRow = []
            for v2 in array2:
                distRow.append(self.calculateDistPoints(v1,v2))
            distMatrix.append(distRow)
        return np.array(distMatrix)
        
    def reduceSignatures(self,s1,s2):
        """Removes samples from signatures if corresponding samples for both models have weight 0. 
        Args:
            s1 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] 
            s2 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights]

        Returns:
            s1new ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] after reduction
            s2new ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] after reduction
        """
        lens = len(s1[0])
        s1featnew=[]
        s1wieghtsnew=[]
        s2featnew=[]
        s2wieghtsnew=[]
        for i in range(lens):
            if s1[1][i]!=0 or s2[1][i]!=0:
                s1featnew.append(s1[0][i])
                s1wieghtsnew.append(s1[1][i])
                s2featnew.append(s2[0][i])
                s2wieghtsnew.append(s2[1][i])
        s1new = [np.array(s1featnew,dtype=np.float64), np.array(s1wieghtsnew,dtype=np.float64)]
        s2new = [np.array(s2featnew,dtype=np.float64), np.array(s2wieghtsnew,dtype=np.float64)]
        return s1new, s2new

    def getSignatures(self,array,steps_all,step_all):
        """Generates signature for array representing model. Signature is composed of list of points [x,y,z] (float) and list of weights (int).

        Args:
            array (np.array(np.array(,dtype=float))): array with voxels representing model
            steps_all ([np.array(,dtype=float),np.array(,dtype=float),np.array(,dtype=float)]): lists with edges for each step for each axis in order x,y,z
            step_all ([float,float,float]): [size of step for x axis, size of step for y axis, size of step for y axis] 

        Returns:
           signature [np.array(,dtype=np.float64),np.array(,dtype=np.float64)]: returns signatuere [np.array of points, np.array of weights]
        """
        x_steps,y_steps,z_steps = steps_all
        x_step,y_step,z_step=step_all
        feature_array = []
        weight_array = []
        for x in range(len(x_steps[:-1])):
            for y in range(len(y_steps[:-1])) :
                for z in range(len(z_steps[:-1])):
                    rows=np.where((array[:,0]> x_steps[x]) &
                                  (array[:,0]<= x_steps[x+1]) &
                                  (array[:,1]> y_steps[y]) & 
                                  (array[:,1]<= y_steps[y+1]) &
                                  (array[:,2]> z_steps[z]) &
                                  (array[:,2]<= z_steps[z+1]))
                    weight, point = self.calculateNeighberhood(array[rows],[x_steps[x]+(x_step/2),y_steps[y]+(y_step/2),z_steps[z]+(z_step/2)])

                    feature_array.append(point)
                    weight_array.append(weight)     

        return [np.array(feature_array,dtype=np.float64), np.array(weight_array,dtype=np.float64)]   

    def getSignaturesForPair(self,array1,array2,steps=None):
        """generates signatures for given pair of models represented by array of voxels.
        We calculate space for given models by taking the extremas for each axis and dividing the space by the number of steps.
        This divided space generate us samples which contains points. Each sample will have new coordinates which are mean of all points from it and weight
        which equals to the number of points.
       
        Args:
            array1 (np.array(np.array(,dtype=float))): array with voxels representing model1
            array2 (np.array(np.array(,dtype=float))): array with voxels representing model2
            steps (int, optional): How many steps is used for sampling space of voxels. Defaults to self.steps (3).

        Returns:
            s1 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights] 
            s2 ([np.array(,dtype=np.float64),np.array(,dtype=np.float64)]): [coordinates of samples, weights]
        """
        if steps ==None:
            steps = self.steps

        self.EPSILON = 0.0001
        min_x = np.min([np.min(array1[:,0]),np.min(array2[:,0])]) - self.EPSILON  # 0.0001 is added and removed to deal with float values 
        max_x = np.max([np.max(array1[:,0]),np.max(array2[:,0])]) + self.EPSILON
        min_y = np.min([np.min(array1[:,1]),np.min(array2[:,1])]) - self.EPSILON
        max_y = np.max([np.max(array1[:,1]),np.max(array2[:,1])]) + self.EPSILON
        min_z = np.min([np.min(array1[:,2]),np.min(array2[:,2])]) - self.EPSILON
        max_z = np.max([np.max(array1[:,2]),np.max(array2[:,2])]) + self.EPSILON

        x_steps,x_step = np.linspace(min_x,max_x,steps,retstep=True)
        y_steps,y_step = np.linspace(min_y,max_y,steps,retstep=True)
        z_steps,z_step = np.linspace(min_z,max_z,steps,retstep=True)
        
        if steps == 1:
            x_steps = [min_x,max_x]
            y_steps = [min_y,max_y]
            z_steps = [min_z,max_z]
            x_step = max_x - min_x
            y_step = max_y - min_y
            z_step = max_z - min_z

        steps_all = (x_steps,y_steps,z_steps)
        step_all = (x_step,y_step,z_step)
        
        s1 = self.getSignatures(array1,steps_all,step_all)
        s2 = self.getSignatures(array2,steps_all,step_all)    
        
        return s1,s2
        
    def getVoxels(self,geno):
        """ Generates voxels for genotype using frams.ModelGeometry

        Args:
            geno (string): representation of model in one of the formats handled by frams http://www.framsticks.com/a/al_genotype.html

        Returns:
            np.array([np.array(,dtype=float)]: list of voxels representing model.
        """
        m = self.frams.ModelGeometry.forModel(self.frams.Model.newFromString(geno));
        m.geom_density = self.density;
        voxels = np.array([np.array([p.x._value(),p.y._value(),p.z._value()]) for p in m.voxels()])
        return voxels

    def calculateEMDforVoxels(self,voxels1, voxels2 ,steps = None):
        """ Calculate EMD for pair of voxels representing models.
        Args:
            voxels1 np.array([np.array(,dtype=float)]: list of voxels representing model1.
            voxels2 np.array([np.array(,dtype=float)]: list of voxels representing model2.
            steps (int, optional): How many steps is used for sampling space of voxels. Defaults to self.steps (3).

        Returns:
            float: EMD for pair of list of voxels
        """
        numvox1 = len(voxels1)
        numvox2 = len(voxels2)    

        s1, s2 = self.getSignaturesForPair(voxels1, voxels2, steps=steps)

        if self.reduce == True:
            s1, s2 = self.reduceSignatures(s1,s2)
            
            if numvox1 != sum(s1[1]) or numvox2 != sum(s2[1]):
                print("Voxel reduction didnt work properly")
                print("Base voxels fig1: ", numvox1, " fig2: ",numvox2)
                print("After reduction voxels fig1: ", sum(s1[1]), " fig2: ",sum(s2[1]))
        
        dist_matrix = self.calculateDistanceMatrix(s1[0],s2[0])

        self.libm.fedisableexcept(0x04) # allowing for operation divide by 0 because pyemd requiers it.

        emd_out = emd(s1[1],s2[1],dist_matrix)

        self.libm.feclearexcept(0x04) # disabling operation divide by 0 because framsticks doesnt like it. 
        self.libm.feenableexcept(0x04)

        return emd_out

    def calculateEMDforGeno(self,geno1, geno2, steps = None):
        """ Calculate EMD for pair of voxels representing models.
        Args:
            geno1 (string): representation of model1 in one of the formats handled by frams http://www.framsticks.com/a/al_genotype.html
            geno2 (string): representation of model2 in one of the formats handled by frams http://www.framsticks.com/a/al_genotype.html
            steps (int, optional): How many steps is used for sampling space of voxels. Defaults to self.steps (3).

        Returns:
            float: EMD for pair of strings representing models. 
        """     

        voxels1 = self.getVoxels(geno1)
        voxels2 = self.getVoxels(geno2)
        
        numvox1 = len(voxels1)
        numvox2 = len(voxels2)    

        s1, s2 = self.getSignaturesForPair(voxels1,voxels2, steps=steps)

        if self.reduce == True:
            s1, s2 = self.reduceSignatures(s1,s2)
            
            if numvox1 != sum(s1[1]) or numvox2 != sum(s2[1]):
                print("Voxel reduction didnt work properly")
                print("Base voxels fig1: ", numvox1, " fig2: ",numvox2)
                print("After reduction voxels fig1: ", sum(s1[1]), " fig2: ",sum(s2[1]))
        
        dist_matrix = self.calculateDistanceMatrix(s1[0],s2[0])

        self.libm.fedisableexcept(0x04)  # allowing for operation divide by 0 because pyemd requiers it.

        emd_out = emd(s1[1],s2[1],dist_matrix)

        self.libm.feclearexcept(0x04) # disabling operation divide by 0 because framsticks doesnt like it.
        self.libm.feenableexcept(0x04)

        if self.verbose == True:
            print("Steps: ", steps)
            print("Geno1:\n",geno1)
            print("Geno2:\n",geno2)
            print("EMD:\n",emd_out)

        return emd_out

    def getDissimilarityMatrix(self,listOfGeno,steps=None):
        """

        Args:
            listOfGeno ([string]): list of strings representing genotypes in one of the formats handled by frams http://www.framsticks.com/a/al_genotype.html
            steps (int, optional): How many steps is used for sampling space of voxels. Defaults to self.steps (3).

        Returns:
            np.array(np.array(,dtype=float)): dissimilarity matrix of EMD for given list of genotypes
        """
        numOfGeno = len(listOfGeno)
        dissimMatrix = np.zeros(shape=[numOfGeno,numOfGeno])
        listOfVoxels = [self.getVoxels(g) for g in listOfGeno]
        for i in range(numOfGeno):
            for j in range(numOfGeno):
                dissimMatrix[i,j] = self.calculateEMDforVoxels(listOfVoxels[i],listOfVoxels[j],steps)
        return dissimMatrix
                

# if __name__ == "__main__":

#     path = '/home/adam/Framsticks/Framsticks50rc19'
#     e = VoxelsEMD(path)
    
#     print("Calculating EMDforGeno")
#     geno1 = "X"
#     geno2 = "X"
#     e.calculateEMDforGeno(geno1, geno2,steps=1)
#     e.calculateEMDforGeno(geno1, geno2,steps=4)
#     e.calculateEMDforGeno(geno1, geno2,steps=6)
#     e.calculateEMDforGeno(geno1, geno2,steps=8)

#     print("Calculating EMDforGeno")
#     list_of_genos = ["X","XX","XXX","XXXXX","XXXXCCX","XCCXXXCCX"]
#     dissim = e.getDissimilarityMatrix(list_of_genos)

#     print(dissim)
