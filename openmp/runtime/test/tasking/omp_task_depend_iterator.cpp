// RUN: %libomp-cxx-compile-and-run

/*

This test is imported from SOLLVE: 5.0/task/test_task_depend_iterator.cpp
SOLLVE page: https://github.com/SOLLVE/sollve_vv

OpenMP API Version 5.0 Nov 2020

This test is for the iterator modifier when used with the task depend
clause. This modifier should create an iterator that expands to multiple values
inside the clause they appear. In this particular test case the iterator expands into
several values creating several dependencies at the same time.

*/

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include "omp_testsuite.h"

#define N 1024
#define FROM 64
#define LENGTH 128

int test_omp_task_depend_iterator() {
  int ptr[] = {0, 4, 5, 6, 7, 8, 9, 10, 11};
  int cols[] = {1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8};
  std::vector<int> threadOrder;
  bool threadOrderError = false;
#pragma omp parallel num_threads(8)
  {
#pragma omp single
    {
      for (int i = 0; i < 8; ++i) {
        int pos = ptr[i], size = ptr[i + 1] - ptr[i];
#pragma omp task depend(iterator(it = 0 : size), in : ptr[cols[pos + it]]) depend(out : ptr[i])
        {
#pragma omp critical
          {
            threadOrder.push_back(i);
          } // end critical section
        } // end task depend
      }
    } // end single
  } // end parallel

  // store the indices of the execution order of generated tasks in idx[]
  std::vector<int>::iterator idx[8];
  for (int i = 0; i < 8; ++i)
    idx[i] = std::find (threadOrder.begin(), threadOrder.end(), i);

  // verify that dependencies are met in the order
  if (idx[0] != threadOrder.begin())
    threadOrderError |= true;
  if (idx[1] > idx[5] || idx[2] > idx[5])
    threadOrderError |= true;
  if (idx[3] > idx[6] || idx[4] > idx[6])
    threadOrderError |= true;
  if (idx[5] > idx[7] || idx[6] > idx[7])
    threadOrderError |= true;

  std::sort(threadOrder.begin(), threadOrder.end());
  for(int i = 0; i < 8; ++i)
    threadOrderError = (threadOrder[i] != i) || threadOrderError;

  // FALSE If dependencies between tasks were not enforced in the correct order.
  return !threadOrderError; 
}



int main() {
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_depend_iterator()) {
      num_failed++;
    }
  }
  return num_failed;
}
