// clang-format off
// RUN: %clangxx-generic %s %S/Inputs/veccopy-ctor-1.cpp -o %t -Xoffload-linker -lompdevice && %libomptarget-run-generic | %fcheck-generic
// REQUIRES: ompt
// REQUIRES: amdgpu
// clang-format on
/*
 * Test whether OMPT support works for OpenMP directives 
 * in a global constructor without using a shared library. 
 * This currently succeeds.
 */

#include <stdio.h>
#include <omp.h>

extern int status;

int main()
{
  int N = 10;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong varlue: a[%d]=%d\n", i, a[i]);
    }

  if (!rc && !status)
    printf("Success\n");

  return rc;
}

// clang-format off
/// CHECK: Record Target task
