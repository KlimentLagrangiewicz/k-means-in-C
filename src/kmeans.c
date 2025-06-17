#include "kmeans.h"

double get_distance(const double *x1, const double *x2, int m) {
	double d, r = 0.0;
	for (; m > 0; --m) {
		d = *x1 - *x2;
		r += d * d;
		++x1;
		++x2;
	}
	return r;
}

void autoscaling(double* const x, const int n, const int m) {
	const double* const end = x + n * m;
	int j;
	for (j = 0; j < m; ++j) {
		double sd, Ex = 0.0, Exx = 0.0, *ptr;
		for (ptr = x + j; ptr < end; ptr += m) {
			sd = *ptr;
			Ex += sd;
			Exx += sd * sd;
		}
		Exx /= n;
		Ex /= n;
		sd = fabs(Exx - Ex * Ex);
		if (sd == 0.0) sd = 1.0;
		sd = 1.0 / sqrt(sd);
		for (ptr = x + j; ptr < end; ptr += m) {
			*ptr = (*ptr - Ex) * sd;
		}
	}
}

char constr(const int *y, const int val, int s) {
	for (; s > 0; --s) {
		if (*y == val) return 1;
		++y;
	}
	return 0;
}

void det_cores(const double* const x, double* const c, const int n, const int m, const int k) {
	int *nums = (int*)malloc(k * sizeof(int));
	srand((unsigned int)time(NULL));
	int i;
	for (i = 0; i < k; ++i) {
		int val = rand() % n;
		while (constr(nums, val, i)) {
			val = rand() % n;
		}
		nums[i] = val;
		memcpy(c + i * m, x + val * m, m * sizeof(double));
	}
	free(nums);
}

int get_cluster(const double* const x, const double* const c, const int m, int k) {
	int res = --k;
	double min_d = get_distance(x, c + k * m, m);
	while (k > 0) {
		--k;
		const double cur_d = get_distance(x, c + k * m, m);
		if (cur_d < min_d) {
			min_d = cur_d;
			res = k;
		}
	}
	return res;
}

int* det_start_partition(const double* const x, const double* const c, int* const nums, int n, const int m, const int k) {
	int *y = (int*)malloc(n * sizeof(int));
	memset(nums, 0, k * sizeof(int));
	while (n > 0) {
		--n;
		const int l = get_cluster(x + n * m, c, m, k);
		y[n] = l;
		++nums[l];
	}
	return y;
}

char check_partition(const double* const x, double* const c, int* const y, int* const nums, const int n, const int m, const int k) {
	memset(c, 0, k * m * sizeof(double));
	int i, j;
	for (i = 0; i < n; ++i) {
		double* const c_yi = c + y[i] * m;
		const double* const x_i = x + i * m;
		for (j = 0; j < m; ++j) {
			c_yi[j] += x_i[j];
		}
	}
	for (i = 0; i < k; ++i) {
		const double f = nums[i];
		double* const c_i = c + i * m;
		for (j = 0; j < m; ++j) {
			c_i[j] /= f;
		}
	}
	memset(nums, 0, k * sizeof(int));
	char flag = 0; 
	for (i = 0; i < n; ++i) {
		const int f = get_cluster(x + i * m, c, m, k);
		if (y[i] != f) flag = 1;
		y[i] = f;
		++nums[f];
	}
	return flag;
}

int* kmeans(const double* const X, const int n, const int m, const int k) {
	double *x = (double*)malloc(n * m * sizeof(double));
	memcpy(x, X, n * m * sizeof(double));
	autoscaling(x, n, m);
	double *c = (double*)malloc(k * m * sizeof(double));
	det_cores(x, c, n, m, k);
	int *nums = (int*)malloc(k * sizeof(int));
	int *y = det_start_partition(x, c, nums, n, m, k);
	if (!y) return NULL;
	while (check_partition(x, c, y, nums, n, m, k));
	free(x);
	free(c);
	free(nums);
	return y;
}

/* k-means without scaling data at the beginning */
int* kmeans_ws(const double* const x, const int n, const int m, const int k) {
	double *c = (double*)malloc(k * m * sizeof(double));
	det_cores(x, c, n, m, k);
	int *nums = (int*)malloc(k * sizeof(int));
	int *y = det_start_partition(x, c, nums, n, m, k);
	if (!y) return NULL;
	while (check_partition(x, c, y, nums, n, m, k));
	free(c);
	free(nums);
	return y;
}
