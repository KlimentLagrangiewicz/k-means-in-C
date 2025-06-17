#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "help.h"
#include "kmeans.h"


int main(int argc, char **argv) {
	if (argc < 6) {
		puts("Not enough parameters...");
		exit(1);
	}
	const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
	if (n < 1 || m < 1 || k < 1 || k > n) {
		puts("Values of input parameters are incorrect...");
		exit(1);
	}
	double *x = (double*)malloc(n * m * sizeof(double));
	if (!x) {
		puts("Memory allocation error...");
		exit(1);
	}	
	fscanf_data(argv[1], x, n * m);
	clock_t cl = clock();
	int *y = kmeans(x, n, m, k);	
	cl = clock() - cl;
	if (!y) {
		puts("Memory allocation error...");
		free(x);
		exit(1);
	}
	if (argc > 6) {
		int *ideal = (int*)malloc(n * sizeof(int));
		if (!ideal) fprintf_result(argv[5], y, n);
		else {
			fscanf_splitting(argv[6], ideal, n);
			const double p = get_precision(ideal, y, n);
			printf("Precision of k-means clustering = %.5lf;\n", p);
			fprintf_full_result(argv[5], y, n, p);
			free(ideal);
		}
	} else fprintf_result(argv[5], y, n);
	printf("Time for k-means clustering = %lf s.;\nThe work of the program is completed...\n", (double)cl / CLOCKS_PER_SEC);
	free(x);
	free(y);
	return 0;
}
