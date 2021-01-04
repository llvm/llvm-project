// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -gline-tables-only && env LIBOMPTARGET_INFO=23 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda -allow-empty -check-prefix=INFO

#include <stdio.h>
#include <omp.h>

#define N 64

int main() {
  int A[N];
  int B[N];
  int C[N];
  int val = 1;

// INFO: CUDA device 0 info: Device supports up to {{.*}} CUDA blocks and {{.*}} threads with a warp size of {{.*}}
// INFO: Libomptarget device 0 info: Entering OpenMP data region at info.c:33:1 with 3 arguments:
// INFO: Libomptarget device 0 info: alloc(A[0:64])[256]
// INFO: Libomptarget device 0 info: tofrom(B[0:64])[256]
// INFO: Libomptarget device 0 info: to(C[0:64])[256]
// INFO: Libomptarget device 0 info: OpenMP Host-Device pointer mappings after block at info.c:33:1:
// INFO: Libomptarget device 0 info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
// INFO: Libomptarget device 0 info: {{.*}} {{.*}} 256      1        C[0:64] at info.c:11:7
// INFO: Libomptarget device 0 info: {{.*}} {{.*}} 256      1        B[0:64] at info.c:10:7
// INFO: Libomptarget device 0 info: {{.*}} {{.*}} 256      1        A[0:64] at info.c:9:7
// INFO: Libomptarget device 0 info: Entering OpenMP kernel at info.c:34:1 with 1 arguments:
// INFO: Libomptarget device 0 info: firstprivate(val)[4]
// INFO: CUDA device 0 info: Launching kernel {{.*}} with {{.*}} and {{.*}} threads in {{.*}} mode
// INFO: Libomptarget device 0 info: OpenMP Host-Device pointer mappings after block at info.c:34:1:
// INFO: Libomptarget device 0 info: Host Ptr           Target Ptr         Size (B) RefCount Declaration
// INFO: Libomptarget device 0 info: 0x{{.*}} 0x{{.*}} 256      1        C[0:64] at info.c:11:7
// INFO: Libomptarget device 0 info: 0x{{.*}} 0x{{.*}} 256      1        B[0:64] at info.c:10:7
// INFO: Libomptarget device 0 info: 0x{{.*}} 0x{{.*}} 256      1        A[0:64] at info.c:9:7
// INFO: Libomptarget device 0 info: Exiting OpenMP data region at info.c:33:1
#pragma omp target data map(alloc:A[0:N]) map(tofrom:B[0:N]) map(to:C[0:N])
#pragma omp target firstprivate(val)
  { val = 1; }

  return 0;
}
