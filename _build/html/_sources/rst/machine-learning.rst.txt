**4. Machine Learning in Diagnosis**
================================================

4.1. Linear Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Machine Learning (ML) is a branch of artificial intelligence that employs a wide range of techniques and tools which are helpful in diagnostic applications in a variety of medical areas like
diagnosis and detection of diseases. These methods are being used in a variety of applications inorder to identify, classify and detect different diseases ranging from cancer and Alzheimer
to ophthalmology. Linear Discriminant Analysis is a supervised classification technique in machine learning for
dimensionality reduction applications. In this method we seek to reduce the dimensions by taking off the unimportant and redundant features by transforming the features from higher dimensional
space to a lower dimensional space where the data achieves maximum class separability

4.2. Mathematical Backend  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the LDA, the derived features are linear combinations of the original features. For implementation we can start off by calculating two matrixes including within-class :math:`S_W`and
between-class :math:`S_B`. The :math:`S_W` and :math:`S_B` are two :math:`n \times n` matrixes indicating the number of features in the data where :math:`n` is the number of the features. 
The :math:`S_W` corresponds to the separability between different classes (i.e. the distance between the mean of different classes). The optimal transformation in classical LDA is achieved by minimizing the :math:`S_W` distance and maximizing the :math:`S_B` distance simultaneously, leading to maximum discrimination. The :math:`S_W` is represented as following: 


.. math:: 
  :name: eq.100

   S_W=  \sum_{i_=1}^{class} S_i \quad Where \quad  S_i = \sum_{j_=1}^{n} (x_j-m_i)(x_j-m_i)^T

In the above equations, the class represent the classes (e.g. Normal, Low-Flattened and High-Flattened globes). :math:`X` is
the number of the samples including the features in each class. The :math:`m` is a vector including the mean of each feature in each class:

  
.. math:: 
  :name: eq.101

   m_i= \frac {1}{n_i} \sum_{k=1}{n} X_k

The :math:`S_B` is computed as following:

.. math:: 
  :name: eq.102

   S_B= \sum_{I=1}^{c} N_i(m_i-\mu)(m_i-\mu)^T


Where :math:`\mu` is a vector containing the mean of all the samples in all of the classes. In the LDA analysis we try to obtain the lower dimensional space in a way that it maximizes the between class variance and minimizes the within class variance. We can transform data using a transformation matrix :math:`\xi` .With that being said, the transformed dataset (Y), could be written as:

.. math:: 
  :name: eq.103

   Y = \xi^T \times X

In order to obtain a good separation between classes we seek to maximize the :math:`\frac{S_B}{S_W}`. By applying this transformation to :math:`S_B` and :math:`S_W` :

.. math:: 
  :name: eq.104


   S_W= \sum_{I=I}^{class} \sum_{j=1}^{class} [(\xi^T(x_j-m_i)(\xi^T(x_j-m_i))^T]_i=\xi^T S_W \xi

.. math:: 
  :name: eq.105

   S_B= \sum_{I=I}^{class} N_i [(\xi^T(m_i-m)(\xi^T(m_i-m))^T]=\xi^T S_B \xi


Then the equation  becomes  :math:`\frac{\xi^T S_B \xi}{\xi^T S_W \xi}`. Now we can find the :math:`\xi` that maximizes this equation. It turns out that :math:`\xi` can be found by calculating the Eigenvectors of :math:`{S_W}^{-1} S_B`.
In other words, we can look at it as a constrained optimization problem with :math:`\xi^T S_W \xi = P`.To this end, we can rewrite this in Lagrangian form:
   
.. math:: 
  :name: eq.106
   

   L=\xi^T S_B \xi - \lambda (\xi^T S_W \xi - P)

Then we can take the derivative of the above equation and set it equal to zero:

.. math:: 
  :name: eq.107

   \frac{\partial L}{\partial \xi} = 2(S_B-\lambda S_W) \xi = 0

Which gives:

.. math:: 
  :name: eq.108

   S_B \xi = \lambda S_W \xi

The above generalized eigenvalue problem can be written as following:

.. math:: 
  :name: eq.109

   {S_W}^{-1} S_B \xi = \lambda \xi

.. math:: 
  :name: eq.110

   {S_W}^{-1} S_B \xi - \lambda \xi = 0 \Rightarrow {S_W}^{-1} S_B \xi - \lambda I \xi = 0

The above equation could be written as following:
   
.. math:: 
  :name: eq.111

   ({S_W}^{-1} S_B - \lambda I) \xi = 0

Solving this equation gives us the eigenvalues (:math:`\lambda_i`) and corresponding eigenvectors of the :math:`S_W^{-1} S_B`
matrix.

4.3. A Real Medical Application   
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.3.1 Disease Background  
""""""""""""""""""""""""""""""""""""""""""

In this section, we use the LDA in a real world application in order to detect an eye-related disease. This eye disorder is called Idiopathic intracranial hypertension (IIH). The IIH occurs most frequently among obese women and astronauts coming back from long-duration space mission. The most common symptoms of this disease include daily and persistent headache , transient visual obscuration and nausea.
Flattening of the posterior eye globe in the magnetic resonance (MR) images is known as one of the most important signs in the patients diagnosed with IIH. 
In this regards, mechanical factors have been proposed for such phenomenon. These mechanical factors include the material properties of the tissue within the optic nerve head and also the pressure loads like ICP and intraocular pressure (IOP). This eye disease has also been studied using numerical model based on finite element method: 

.. note:: 

    Here is the reference: 

	 	`Numerical Investigation on the Role of Mechanical Factors Contributing to Globe Flattening in States of Elevated Intracranial Pressure <https://www.mdpi.com/2075-1729/10/12/316>`_

In general, the degree of flattening of the posterior of the eye globe could represent the severity of this disease. To be more specific, the higher the flattening, the more severe the disease.  
The eye globe could be modeled as an axisymetric finite element model where the flattening of the posterior of the globe is represented by :math:`\alpha` and :math:`\beta`. we introduce a new parameter and call it Cut-Off angle (:math:`\beta`) as an angle (:math:`\theta`) in which the slopes (:math:`\alpha`) up to the
:math:`\beta` are fairly small (−0.5° ≤ :math:`\alpha` ≤ 0.5°). The simplified eye model consisted of different tissues including Sclera (SC), Peripapillary Sclera (PSC), Dura Mater (DM), Retina (RET), Vessel (VES), Lamina Cribrosa (LC), Pia Mater (PM) and Optic Nerve (ON). These parameters are all shown in next figure: 

.. figure:: PNG/14.png
   :align: center

   Illustration of the parameters :math:`\alpha` , :math:`\beta` and :math:`\theta`. The tissues within the eye globe are shown in the left.

In general, the lower value of :math:`\beta` indicates larger flattening area at the posterior of the eye globe that is equal to higher severity of this eye disease. 
For a normal eye where there is no flattening at the back of the globe, the :math:`\beta=90°`. For an eye globe where the area of flattening is relatively small, the :math:`\beta=80°`. Finally the eye globe with largest degree of globe flattening is the eye with the :math:`\beta=70°`.


4.3.2. Python Implementation   
"""""""""""""""""""""""""""""""""""""""""
According to these 3 classes, we are given 1211 eye globes including 360 normal eye globe, 550 eye globes with low globe flattening and 301 eye globe with high globe flattening. The different classes are shown in this table: 
 


.. csv-table:: The criteria defining different levels of globe flattening
   :name: tab.1
   :widths: 10,7,10



   Class Description, :math:`\beta` (Degrees), Number of eye globe
   Normal Globe,90,360
   Low - Flattened Globe,80,550
   High - Flattened Globe,70,301

From the mechanical point of view, we have 9 different mechanical features playing role in the deformation of the eye globe two of which are internal pressures including the IOP and ICP. The remaining 7 features are the material properties of the constitutive tissues as shown previously. 
If want to visualize our classes, this is not feasible to do it on a 9 dimensionals space. Instead, we take advantage of LDA in order to project the results on a 2 dimensional space where the classes are separated from each other. 
Regarding the given eye globes, we have information of the pressures  (IOP and ICP) as well as estimation of the material properties of the all tissues inside each globe. 

.. note:: 

    All values should be used after normalization meaning that for each feature, all data should be divided by the maximum value in the corresponding feature. With that being said, all data are values between 0 and 1. We should write all data in a text file. Each line in the text file correspond to 1 eye globe that has 9 numbers which are the normalized values of the features for that particular eye separated by "," and the last item in the line is the name of the class that the eye globe belongs to. It should be either "A" or "B" or "C" corresponding to the "High Flattened", "Low Flattened" and "Normal" eye globes. 

Here is an example of one line in the text file for an eye globe belonging to the high-flattened class: 

.. code-block:: python

			0.25,0.1,0.2,0.04,1,0.1111111111,0.1111111111,0.6665,0.33325,A

.. note:: 

    The name of the text file is : **LARGE-NORMALIZED.txt** including 1211 lines and is available in the github repository in the folder **EYE-MODELS** 

Here is the Python implementation of the LDA analysis on the given data. We use **Panda** and **Scikitlearn** for reading and preprocessing of the data including the features and classes. In the meantime, we use **Numpy** for mathematical implementation of the LDA analysis:

.. code-block:: python


        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        
        # Here we should define all 9 features including 9 material properties and 2 pressures
        feature_dict = {i:label for i,label in zip(
                        range(9),('SC','PSC','PIA','DURA','LC','RET','ON','IOP','ICP'))}
        
        df = pd.read_csv('LARGE-NORMALIZED.txt', header=None)
        df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
        df.tail()
        X = df.ix[:,0:9].values
        y = df.ix[:,9].values
        label_dict = {1: 'High-Flattened', 2: 'Low-Flattened', 3:'Normal'}
        
        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1
        
        np.set_printoptions(precision=4)
        
        mean_vectors = []
        for cl in range(1,4):
            mean_vectors.append(np.mean(X[y==cl], axis=0))
        
        # S_W is 9x9 matrix
        S_W = np.zeros((9,9))
        for cl,mv in zip(range(1,4), mean_vectors):
            class_sc_mat = np.zeros((9,9))                  # scatter matrix for every class
            for row in X[y == cl]:
                row, mv = row.reshape(9,1), mv.reshape(9,1) # make column vectors
                class_sc_mat += (row-mv).dot((row-mv).T)
            S_W += class_sc_mat                             # sum class scatter matrices
        
        overall_mean = np.mean(X, axis=0)
        
        # S_B is 9x9 matrix
        S_B = np.zeros((9,9))
        for i,mean_vec in enumerate(mean_vectors):
            n = X[y==i+1,:].shape[0]
            mean_vec = mean_vec.reshape(9,1) # make column vector
            overall_mean = overall_mean.reshape(9,1) # make column vector
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        
        for i in range(len(eig_vals)):
            eigvec_sc = eig_vecs[:,i].reshape(9,1)
        
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
        print('Variance explained:\n')
        eigv_sum = sum(eig_vals)
        for i,j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
        
        W = np.hstack((eig_pairs[0][1].reshape(9,1), eig_pairs[1][1].reshape(9,1)))
        X_lda = X.dot(W)
        
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(8, 6))
            for label, marker, color in zip(range(1, 4), ('^', 's', 'o'), ('r', 'b','g')):
                plt.scatter(x=X_lda[:, 0].real[y == label],
                            y=X_lda[:, 1].real[y == label],
                            marker=marker,
                            color=color,
                            alpha=0.6,s=100,
                            label=label_dict[label])
        
            plt.xlabel('LDA 1', fontsize=25)
            plt.ylabel('LDA 2', fontsize=25)
        
            plt.xlim(-1,1)
            plt.ylim(-1,1.0)
            plt.tick_params(axis='both', which='major', labelsize=30)
        
            plt.legend(loc='lower right', fontsize=30)
            plt.tight_layout()
            plt.savefig('Results.png', format='png', dpi=1200)
            plt.show()

Here is the output of the code: 

.. code-block:: python

			Variance explained:

					eigenvalue 1: 80.74%
					eigenvalue 2: 19.26%
					eigenvalue 3: 0.00%
					eigenvalue 4: 0.00%
					eigenvalue 5: 0.00%
					eigenvalue 6: 0.00%
					eigenvalue 7: 0.00%
					eigenvalue 8: 0.00%
					eigenvalue 9: 0.00%

The above shows the sorted Variance explained where it is has become zero in 7 directions and only 2 directions have remained (LDA1 & LDA2) where the variance explained are largest. 

In the below figure, we can see how the classes have been projected on the 2D space and separated from each other: 
 
.. figure:: PNG/15.png
   :align: center

   Separation of the classes including the Normal eyes, Lowe-Flattened eyes and High-Flattened eyes after implementation of the LDA analysis on the data. 



4.3.3. Prediction of an Unknown Case   
"""""""""""""""""""""""""""""""""""""""""

Now, lets say how we can use the LDA as a supervised machine learning technique to make prediction on an eye globe where we do not know which class it belongs to. We have the information about the values of the features including the material properties and also the pressures for that eye globe.

.. note:: 

     We do not know how deformation in an eye globe with these features coulde be. In addition, we do not have any MR image of this particular eye globe to figure out if this is a normal, low flattened  or high flattened eye globe. 

We should normalize the given data from this unknown case and then add a new line to the text file with a new class name like **D** corresponding to the unknown class. With that being said, after adding the new line, we need to implement some minor changes in the body of the previous code to visualize the location of the new data (e.g., Unknown Case): 

.. code-block:: python

        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        
        # Here we should define all 9 features including 9 material properties and 2 pressures
        feature_dict = {i:label for i,label in zip(
                        range(9),('SC','PSC','PIA','DURA','LC','RET','ON','IOP','ICP'))}
        
        df = pd.read_csv('LARGE-NORMALIZED.txt', header=None)
        df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
        df.tail()
        X = df.ix[:,0:9].values
        y = df.ix[:,9].values
        label_dict = {1: 'High-Flattened', 2: 'Low-Flattened', 3:'Normal',4: 'Unknown Class'}
        
        enc = LabelEncoder()
        label_encoder = enc.fit(y)
        y = label_encoder.transform(y) + 1
        
        np.set_printoptions(precision=4)
        
        mean_vectors = []
        for cl in range(1,5):
            mean_vectors.append(np.mean(X[y==cl], axis=0))
        
        # S_W is 9x9 matrix
        S_W = np.zeros((9,9))
        for cl,mv in zip(range(1,5), mean_vectors):
            class_sc_mat = np.zeros((9,9))                  # scatter matrix for every class
            for row in X[y == cl]:
                row, mv = row.reshape(9,1), mv.reshape(9,1) # make column vectors
                class_sc_mat += (row-mv).dot((row-mv).T)
            S_W += class_sc_mat                             # sum class scatter matrices
        
        overall_mean = np.mean(X, axis=0)
        
        # S_B is 9x9 matrix
        S_B = np.zeros((9,9))
        for i,mean_vec in enumerate(mean_vectors):
            n = X[y==i+1,:].shape[0]
            mean_vec = mean_vec.reshape(9,1) # make column vector
            overall_mean = overall_mean.reshape(9,1) # make column vector
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        
        for i in range(len(eig_vals)):
            eigvec_sc = eig_vecs[:,i].reshape(9,1)
        
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
        print('Variance explained:\n')
        eigv_sum = sum(eig_vals)
        for i,j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
        
        W = np.hstack((eig_pairs[0][1].reshape(9,1), eig_pairs[1][1].reshape(9,1)))
        X_lda = X.dot(W)
        
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(8, 6))
            for label, marker, color in zip(range(1, 5), ('^', 's', 'o','*'), ('r', 'b','g','k')):
                plt.scatter(x=X_lda[:, 0].real[y == label],
                            y=X_lda[:, 1].real[y == label],
                            marker=marker,
                            color=color,
                            alpha=0.6,s=100,
                            label=label_dict[label])
        
            plt.xlabel('LDA 1', fontsize=25)
            plt.ylabel('LDA 2', fontsize=25)
        
            plt.xlim(-1,1)
            plt.ylim(-1,1.0)
            plt.tick_params(axis='both', which='major', labelsize=30)
        
            plt.legend(loc='lower right', fontsize=30)
            plt.tight_layout()
            plt.savefig('Results.png', format='png', dpi=1200)
            plt.show()

The new unknow data will be shown on the space as a black ★. This way we can see inside which class the new data will fall in. 

4.3.3.1. Case.1   
##################

Here is the information regarding the first unknow case that should be added as a line to the data (**LARGE-NORMALIZED.txt**). Here is the line:

.. code-block:: python

			0.05,0.25,0.2,0.2,0.1,1,0.3333333333,1,0.33325,D

We can visualize where the data is located in this figure:


.. figure:: PNG/16.png
   :align: center

   Illustration of the new data. The arrow points at the location where the data is located

It could be clearly seen that this case belongs to the high-flattened class. 

4.3.3.2. Case.2   
##################

Here is the information regarding the second unknown case that should be added as a line to the data (**LARGE-NORMALIZED.txt**). Here is the line:


.. code-block:: python

			0.1,0.1,0.04,1,1,0.3333333333,0.1111111111,0.33325,1,D

We can visualize where the second unknown data is located in this figure:

.. figure:: PNG/17.png
   :align: center

   Illustration of the second unknown data. The arrow points at the location where the data is located

It is obvious that this case belongs to the low-flattened class. 

4.3.3.3. Case.3   
##################

Here is the information regarding the third unknow case that should be added as a line to the data (**LARGE-NORMALIZED.txt**). Here is the line:


.. code-block:: python

			1,0.11,1,0.03,1,0.22,0.3333333333,0.1,0.11,D

We can visualize where the third unknown data is located in this figure:


.. figure:: PNG/18.png
   :align: center

   Illustration of the third unknown data. The arrow points at the location where the data is located

This case, clearly falls within the normal class. 

.. note:: 

     The accuracy of the prediction of the LDA analysis to determine the class that they belong to could are verified after doing the FEM simulation based on the defined criteria defined in :ref:`Table.1 <tab.1>`

