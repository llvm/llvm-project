// RUN: %libomptarget-compilexx-run-and-check-generic

// amdgcn does not have printf definition
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: amdgcn-amd-amdhsa-newRTL

#include <stdio.h>

void sum(int* input, int size, int* output)
{
#pragma omp target teams distribute parallel for reduction(+:output[0]) \
                                                 map(to:input[0:size])
  for (int i = 0; i < size; i++)
    output[0] += input[i];
}
int main()
{
  const int size = 100;
  int *array = new int[size];
  int result = 0;
  for (int i = 0; i < size; i++)
    array[i] = i + 1;
  sum(array, size, &result);
  // CHECK: Result=5050
  printf("Result=%d\n", result);
  delete[] array;
  return 0;
}

