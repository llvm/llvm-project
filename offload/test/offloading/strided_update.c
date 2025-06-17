// Checks that "update from" clause in OpenMP is supported when the elements are updated in a non-contiguous manner.
// RUN: %libomptarget-compile-run-and-check-generic
#include <omp.h>  
#include <stdio.h>  
  
int main() {  
  int len = 8;  
  double data[len];  
  #pragma omp target map(tofrom: len, data[0:len])  
  {  
    for (int i = 0; i < len; i++) {  
      data[i] = i;  
    }  
  }  
  // initial values  
  printf("original host array values:\n");  
  for (int i = 0; i < len; i++)  
    printf("%f\n", data[i]);  
  printf("\n");  
  
  #pragma omp target data map(to: len, data[0:len])  
  {  
    #pragma omp target  
    for (int i = 0; i < len; i++) {  
      data[i] += i ;  
    }  
  
    #pragma omp target update from(data[0:8:2])  
  }  
  // from results  
  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 4.000000
  // CHECK: 3.000000
  // CHECK: 8.000000
  // CHECK: 5.000000
  // CHECK: 12.000000
  // CHECK: 7.000000
  // CHECK-NOT: 2.000000
  // CHECK-NOT: 6.000000
  // CHECK-NOT: 10.000000
  // CHECK-NOT: 14.000000

  printf("from target array results:\n");  
  for (int i = 0; i < len; i++)  
    printf("%f\n", data[i]);  
  printf("\n");  
  
  return 0;  
}  

