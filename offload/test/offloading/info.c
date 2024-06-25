// RUN: %libomptarget-compile-generic \
// RUN:     -gline-tables-only -fopenmp-extensions
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -allow-empty -check-prefixes=INFO
// RUN: env LIBOMPTARGET_INFO=63 %libomptarget-run-amdgcn-amd-amdhsa 2>&1 | \
// RUN:   %fcheck-amdgcn-amd-amdhsa -allow-empty -check-prefixes=INFO,AMDGPU

// FIXME: Fails due to optimized debugging in 'ptxas'.
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO

#include <omp.h>
#include <stdio.h>

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

// clang-format off
// INFO: info: Entering OpenMP data region with being_mapper at info.c:{{[0-9]+}}:{{[0-9]+}} with 3 arguments:
// INFO: info: alloc(A[0:64])[256]
// INFO: info: tofrom(B[0:64])[256]
// INFO: info: to(C[0:64])[256]
// INFO: info: Creating new map entry with HstPtrBase={{.*}}, HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, DynRefCount=1, HoldRefCount=0, Name=A[0:64]
// INFO: info: Creating new map entry with HstPtrBase={{.*}}, HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, DynRefCount=0, HoldRefCount=1, Name=B[0:64]
// INFO: info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=256, Name=B[0:64]
// INFO: info: Creating new map entry with HstPtrBase={{.*}}, HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, DynRefCount=1, HoldRefCount=0, Name=C[0:64]
// INFO: info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=256, Name=C[0:64]
// INFO: info: OpenMP Host-Device pointer mappings after block at info.c:{{[0-9]+}}:{{[0-9]+}}:
// INFO: info: Host Ptr           Target Ptr         Size (B) DynRefCount HoldRefCount Declaration
// INFO: info: {{.*}}             {{.*}}             256      1           0            C[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: {{.*}}             {{.*}}             256      0           1            B[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: {{.*}}             {{.*}}             256      1           0            A[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: Entering OpenMP kernel at info.c:{{[0-9]+}}:{{[0-9]+}} with 1 arguments:
// INFO: info: firstprivate(val)[4]
// INFO: info: Launching kernel __omp_offloading_{{.*}}main{{.*}} with {{[0-9]+}} blocks and {{[0-9]+}} threads in Generic mode
// AMDGPU: AMDGPU device {{[0-9]}} info: #Args: {{[0-9]}} Teams x Thrds: {{[0-9]+}}x {{[0-9]+}} (MaxFlatWorkGroupSize: {{[0-9]+}}) LDS Usage: {{[0-9]+}}B #SGPRs/VGPRs: {{[0-9]+}}/{{[0-9]+}} #SGPR/VGPR Spills: {{[0-9]+}}/{{[0-9]+}} Tripcount: {{[0-9]+}}
// INFO: info: OpenMP Host-Device pointer mappings after block at info.c:{{[0-9]+}}:{{[0-9]+}}:
// INFO: info: Host Ptr           Target Ptr         Size (B) DynRefCount HoldRefCount Declaration
// INFO: info: {{.*}}             {{.*}}             256      1           0            C[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: {{.*}}             {{.*}}             256      0           1            B[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: {{.*}}             {{.*}}             256      1           0            A[0:64] at info.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: Exiting OpenMP data region with end_mapper at info.c:{{[0-9]+}}:{{[0-9]+}} with 3 arguments:
// INFO: info: alloc(A[0:64])[256]
// INFO: info: tofrom(B[0:64])[256]
// INFO: info: to(C[0:64])[256]
// INFO: info: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}}, Size=256, Name=B[0:64]
// INFO: info: Removing map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, Name=C[0:64]
// INFO: info: Removing map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, Name=B[0:64]
// INFO: info: Removing map entry with HstPtrBegin={{.*}}, TgtPtrBegin={{.*}}, Size=256, Name=A[0:64]
// INFO: info: OpenMP Host-Device pointer mappings after block at info.c:[[#%u,]]:[[#%u,]]:
// INFO: info: Host Ptr  Target Ptr Size (B) DynRefCount HoldRefCount Declaration
// INFO: info: [[#%#x,]] [[#%#x,]]  4        INF         0            global at unknown:0:0
// clang-format on
#pragma omp target data map(alloc : A[0 : N])                                  \
    map(ompx_hold, tofrom : B[0 : N]) map(to : C[0 : N])
#pragma omp target firstprivate(val)
  { val = 1; }

  __tgt_set_info_flag(0x0);
// INFO-NOT: omptarget device 0 info: {{.*}}
#pragma omp target
  {}

  return 0;
}
