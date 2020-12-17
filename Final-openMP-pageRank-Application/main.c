//////////////////////////////
// Name      : Liam McDonnell
// Date      : 12/16/2020
// class     : cpts-411
// assignment: Parallel Pagerank Estimator
//////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

int ThreadNumber;

void pageRank(int N,int **adjMat,int *degreeArray,int *inboundArray,double *pageRankVec, double dampingRatio);

void pageRank(int N, int **adjMat, int *degreeArray, int *inboundArray, double *pageRankVec, double dampingRatio){
    int i, j;
    double error=1, danglingSum, iterationSum;

    //Damping factor
    double alpha = dampingRatio;
    double delta = 1 - alpha;

    //PageRank vector of previous iteration is saved in z
    double *z;
    z = malloc(sizeof(double) * N);
    if (z == NULL){
        printf("Could not allocate memory for the z array");
        exit(-1);
    }

    //PageRank vector initiallization
    for (i = 0; i < N; i++)
        pageRankVec[i] = 1.0 / N;

    int count = 0;

    omp_set_num_threads(ThreadNumber);
    while (error > 1e-6){

        danglingSum=0;

        //Compute dangling node sum
		#pragma omp parallel private(j) shared(degreeArray,N)
        {
			#pragma omp for reduction(+: danglingSum)
			for(j = 0; j < N; j++){
				if(degreeArray[j] == 0){
					danglingSum = danglingSum + pageRankVec[j] / N;
				}
			}
		}

		#pragma omp parallel shared(N, inboundArray, pageRankVec, adjMat, degreeArray, alpha, delta) private(i, j, iterationSum)
		{
			#pragma omp for
			for(i = 0; i < N; i++){
				z[i] = pageRankVec[i];
				iterationSum = 0;
				for (j=0;j<inboundArray[i];j++)
					iterationSum = iterationSum  + pageRankVec[adjMat[i][j]] / degreeArray[adjMat[i][j]];

				pageRankVec[i] = alpha * (danglingSum + iterationSum) + delta / (double)N;

			}
		}
        count++;
    }
}

// argv parameters(in order): fileName, numThreads, walkLength, dampingRatio
int main(int argc , char **argv){
	if (argc != 5) {
        printf("Error with input parameters: should go: fileName, numThreads, walkLength, dampingRatio");
        exit(1);
    }
    // Set input values
    char *filename = argv[1];
    ThreadNumber = atoi(argv[2]);
    int walkLength = atoi(argv[3]);
    double dampingRatio = atof(argv[4]);

    int i, j, N;
    char line[512],s[2] = " ", *token;

    // Open input file for read
    FILE *file = fopen(filename, "r");
    if(file == NULL){
        printf("Error opening file %s\n",filename);
        exit(1);
    }

    // First we need to get our N (total nodes in dataset)
    while (fgets(line, 512, file) != NULL){
        if (line[0] == '#'){
            token = strtok(line,s);
            token = strtok(NULL,s);
            if ( !strcmp(token,"Nodes:") ){
                token = strtok(NULL,s);
                N = atoi(token);
            }
        }
    }
    printf("Number of nodes in data set: %d \n",N);

    // degreeArray that is used to store the degree of all of our nodes
    int *degreeArray;
    degreeArray = malloc(sizeof(int) * N);
    if (degreeArray == NULL){
        printf("Could not allocate memory for the degreeArray \n");
        exit(-1);
    }

    // inboundsArray that is used to store all of the inbounds connections
    // to our graph.
    int *inboundArray;
    inboundArray = malloc(sizeof(int) * N);
    if (inboundArray == NULL){
        printf("Could not allocate memory for the inboundArray \n");
        exit(-1);
    }
    for (i = 0; i < N; i++)
        inboundArray[i] = 0;


    //reset file pointer back to start of file.
    fseek(file, 0, SEEK_SET);
	//Count outbound and inbound links
	int from, to;
    while (fgets(line, 512, file) != NULL){
        token = strtok(line,s);
        if (strcmp(token, "#")){
            sscanf(line, "%d %d \n", &from, &to);
            degreeArray[from]++;
            inboundArray[to]++;
        }
    }

    // adjacentcyMatrix that is used to store the adjacent links
    // in our graph.
	int **adjMat;
    adjMat = malloc(sizeof(int*) * N);
    if (adjMat == NULL){
        printf("Could not allocate memory for the adjMat array\n");
        exit(-1);
    }
    // We set its links based on the inbounds connections of our graph.
    for (i = 0; i <= N;i++)
        adjMat[i] = malloc(sizeof(int) * inboundArray[i]);

    // simple counter that will be used to count connections for each page
    // of our graph.
    int *counter;
    counter = malloc(sizeof(int) * N);
    if (counter == NULL){
        printf("Could not allocate memory for the counter array \n");
        exit(-1);
    }
    for (i = 0; i < N; i++)
        counter[i] = 0;

    fseek(file, 0, SEEK_SET);
    //Now we can create our adjacency matrix
    while (fgets(line, 512, file) != NULL){
        token = strtok(line, s);
        if (strcmp(token,"#")){
            sscanf(line, "%d %d \n", &from, &to);
            adjMat[to][counter[to]] = from;
            counter[to]++;
        }
    }

    //Finally we create our pageRankVector that stores the rank of each node
    // in our graph.
    double *pageRankVec;
    pageRankVec = malloc(sizeof(double) * N);
    if (pageRankVec == NULL){
        printf("Could not allocate memory for the pageRank vector \n");
        exit(-1);
    }

    struct timeval startwtime, endwtime;
    gettimeofday (&startwtime, NULL);
    pageRank(N, adjMat, degreeArray, inboundArray, pageRankVec, dampingRatio);
    gettimeofday (&endwtime, NULL);


    double time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

    for (i=0;i<100;i++){
        printf("page rank %d: %.20f \n",i, pageRankVec[i]);
    }
    printf("PageRank completion time: %f \n", time);

    //Finally we free all of our memory
    free(inboundArray);
    free(degreeArray);
    free(pageRankVec);
    free(counter);
    for (i = 0; i < N; i++)
        free(adjMat[i]);
    free(adjMat);

    return 0;

}
