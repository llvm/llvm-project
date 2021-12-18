// RUN: %libomptarget-compile-run-and-check-generic
// REQUIRES: ompt
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-newRTL
// UNSUPPORTED: x86_64-pc-linux-gnu

#include <stdio.h>
#include <assert.h>
#include <omp.h>

#include "callbacks.h"
#include "register_emi_map.h"

int main()
{
  int N = 100000;

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

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

  /// CHECK: 0: Could not register callback 'ompt_callback_target_map_emi'
  /// CHECK: Callback Init:
  /// CHECK: Callback Load:
  /// CHECK: Callback Target EMI: kind=1 endpoint=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback Submit EMI: endpoint=1 req_num_teams=1
  /// CHECK: Callback Submit EMI: endpoint=2 req_num_teams=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback Target EMI: kind=1 endpoint=2
  /// CHECK: Callback Target EMI: kind=1 endpoint=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback Submit EMI: endpoint=1 req_num_teams=0
  /// CHECK: Callback Submit EMI: endpoint=2 req_num_teams=0
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback Target EMI: kind=1 endpoint=2
  /// CHECK: Callback Fini:
