// RUN: %libomptarget-compile-generic -O3 && %libomptarget-run-generic | %fcheck-generic

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_N 25000

void reset_input(double *a, double *a_h, double *b, double *c) {
  for(int i = 0 ; i < MAX_N ; i++) {
    a[i] = a_h[i] = i;
    b[i] = i*2;
    c[i] = i-3;
  }
}

int main(int argc, char *argv[]) {
  double * a = (double *) malloc(MAX_N * sizeof(double));
  double * a_h = (double *) malloc(MAX_N * sizeof(double));
  double * d = (double *) malloc(MAX_N * sizeof(double));
  double * d_h = (double *) malloc(MAX_N * sizeof(double));
  double * b = (double *) malloc(MAX_N * sizeof(double));
  double * c = (double *) malloc(MAX_N * sizeof(double));

#pragma omp target enter data map(to:a[:MAX_N],b[:MAX_N],c[:MAX_N],d[:MAX_N])

  for (int n = 32 ; n < MAX_N ; n+=5000) {
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n],d[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
        t++;
#pragma omp target
#pragma omp teams num_teams(tms) thread_limit(ths)
        {
#pragma omp distribute parallel for
          for (int i = 0; i < n; ++i) {
            a[i] += b[i] + c[i];
          }
#pragma omp distribute parallel for
          for (int i = 0; i < n; ++i) {
           d[i] -= b[i] + c[i];
          }
        }
      } // loop over 'ths'
    } // loop over 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++) {
      for (int i = 0; i < n; ++i) {
        a_h[i] += b[i] + c[i];
      }
      for (int i = 0; i < n; ++i)
        d_h[i] -= b[i] + c[i];
    }
#pragma omp target update from(a[:n],d[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
        printf("A Error at n = %d, i = %d: host = %f, device = %f\n", n, i, a_h[i], a[i]);
        return 1;
      }
      if (d_h[i] != d[i]) {
        printf("D Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, d_h[i], d[i]);
        return 1;
      }
    }
  } // loop over 'n'

  // CHECK: Succeeded
  printf("Succeeded\n");
  return 0;
}
