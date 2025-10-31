// RUN: %libomptarget-compilexx-run-and-check-generic
// This test checks that #pragma omp target update from(data[0:4:3]) correctly
// updates every third element (stride 3) from the device to the host, partially
// across the array

#include <iomanip>
#include <iostream>
#include <omp.h>

using namespace std;

int main() {
  int len = 11;
  double data[len];

#pragma omp target map(tofrom : data[0 : len])
  {
    for (int i = 0; i < len; i++)
      data[i] = i;
  }

  // Initial values
  cout << "original host array values:" << endl;
  for (int i = 0; i < len; i++)
    cout << fixed << setprecision(6) << data[i] << endl;
  cout << endl;

#pragma omp target data map(to : data[0 : len])
  {
// Modify arrays on device
#pragma omp target
    for (int i = 0; i < len; i++)
      data[i] += i;

#pragma omp target update from(data[0 : 4 : 3]) // indices 0,3,6,9
  }

  cout << "device array values after update from:" << endl;
  for (int i = 0; i < len; i++)
    cout << fixed << setprecision(6) << data[i] << endl;
  cout << endl;

  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 2.000000
  // CHECK: 3.000000
  // CHECK: 4.000000
  // CHECK: 5.000000
  // CHECK: 6.000000
  // CHECK: 7.000000
  // CHECK: 8.000000
  // CHECK: 9.000000
  // CHECK: 10.000000

  // CHECK: 0.000000
  // CHECK: 1.000000
  // CHECK: 2.000000
  // CHECK: 6.000000
  // CHECK: 4.000000
  // CHECK: 5.000000
  // CHECK: 12.000000
  // CHECK: 7.000000
  // CHECK: 8.000000
  // CHECK: 18.000000
  // CHECK: 10.000000

  return 0;
}