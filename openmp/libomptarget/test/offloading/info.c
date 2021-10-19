// RUN: %libomptarget-compile-nvptx64-nvidia-cuda \
// RUN:     -gline-tables-only -fopenmp-extensions
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda -allow-empty -check-prefix=INFO
// REQUIRES: nvptx64-nvidia-cuda

#include <stdio.h>
#include <omp.h>

#define N 64

#pragma omp declare target
int global;
#pragma omp end declare target

extern void __tgt_set_info_flag(unsigned);

int main() {
  int A[N];
  int B[N];
  int C[N];
  int val = 1;

// INFO: CUDA device 0 info: Device supports up to {{[0-9]+}} CUDA blocks and {{[0-9]+}} threads with a warp size of {{[0-9]+}}
// INFO: Libomptarget device 0 info: Entering OpenMP data region at info.c:{{[0-9]+}}:{{[0-9]+}} with 3 arguments:
// INFO: Libomptarget device 0 info: alloc(A[0:64])[256]
// INFO: Libomptarget device 0 info: tofrom(B[0:64])[256]
// INFO: Libomptarget device 0 info: to(C[0:64])[256]
// INFO: Libomptarget device 0 info: Creating new map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, DynRefCount=1, HoldRefCount=0, Name=A[0:64]
// INFO: Libomptarget device 0 info: Creating new map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, DynRefCount=0, HoldRefCount=1, Name=B[0:64]
// INFO: Libomptarget device 0 info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=256, Name=B[0:64]
// INFO: Libomptarget device 0 info: Creating new map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, DynRefCount=1, HoldRefCount=0, Name=C[0:64]
// INFO: Libomptarget device 0 info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=256, Name=C[0:64]
// INFO: Libomptarget device 0 info: OpenMP Host-Device pointer mappings after block at info.c:{{[0-9]+}}:{{[0-9]+}}:
// INFO: Libomptarget device 0 info: Host Ptr           Target Ptr         Size (B) DynRefCount HoldRefCount Declaration
// INFO: Libomptarget device 0 info: {{.*}}             {{.*}}             256      1           0            C[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: Libomptarget device 0 info: {{.*}}             {{.*}}             256      0           1            B[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: Libomptarget device 0 info: {{.*}}             {{.*}}             256      1           0            A[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: Libomptarget device 0 info: Entering OpenMP kernel at info.c:{{[0-9]+}}:{{[0-9]+}} with 1 arguments:
// INFO: Libomptarget device 0 info: firstprivate(val)[4]
// INFO: CUDA device 0 info: Launching kernel __omp_offloading_{{.*}}main{{.*}} with {{[0-9]+}} blocks and {{[0-9]+}} threads in Generic mode
// INFO: Libomptarget device 0 info: OpenMP Host-Device pointer mappings after block at info.c:{{[0-9]+}}:{{[0-9]+}}:
// INFO: Libomptarget device 0 info: Host Ptr           Target Ptr         Size (B) DynRefCount HoldRefCount Declaration
// INFO: Libomptarget device 0 info: {{.*}}             {{.*}}             256      1           0            C[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: Libomptarget device 0 info: {{.*}}             {{.*}}             256      0           1            B[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: Libomptarget device 0 info: {{.*}}             {{.*}}             256      1           0            A[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: Libomptarget device 0 info: Exiting OpenMP data region at info.c:{{[0-9]+}}:{{[0-9]+}} with 3 arguments:
// INFO: Libomptarget device 0 info: alloc(A[0:64])[256]
// INFO: Libomptarget device 0 info: tofrom(B[0:64])[256]
// INFO: Libomptarget device 0 info: to(C[0:64])[256]
// INFO: Libomptarget device 0 info: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}}, Size=256, Name=B[0:64]
// INFO: Libomptarget device 0 info: Removing map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, Name=C[0:64]
// INFO: Libomptarget device 0 info: Removing map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, Name=B[0:64]
// INFO: Libomptarget device 0 info: Removing map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, Name=A[0:64]
// INFO: Libomptarget device 0 info: OpenMP Host-Device pointer mappings after block at info.c:[[#%u,]]:[[#%u,]]:
// INFO: Libomptarget device 0 info: Host Ptr  Target Ptr Size (B) DynRefCount HoldRefCount Declaration
// INFO: Libomptarget device 0 info: [[#%#x,]] [[#%#x,]]  4        INF         0            unknown at unknown:0:0
#pragma omp target data map(alloc:A[0:N]) map(ompx_hold,tofrom:B[0:N]) map(to:C[0:N])
#pragma omp target firstprivate(val)
  { val = 1; }

  __tgt_set_info_flag(0x0);
// INFO-NOT: Libomptarget device 0 info: {{.*}}
#pragma omp target
  { }

  return 0;
}
