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