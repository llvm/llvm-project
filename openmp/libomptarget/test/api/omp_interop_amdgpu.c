// RUN: %libomptarget-compile-amdgcn-amd-amdhsa -O1
// RUN: %libomptarget-run-amdgcn-amd-amdhsa | %fcheck-amdgcn-amd-amdhsa
// REQUIRES: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16384

void vectorSet(int n, double s, double *x) {
  for (int i = 0; i < n; ++i)
    x[i] = s * (i + 1);
}

void vectorCopy(int n, double *x, double *y) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i];
}

void vectorScale(int n, double s, double *x) {
  for (int i = 0; i < n; ++i)
    x[i] = s * x[i];
}

int main() {
  const double ScaleFactor = 2.0;
  double x[N], y[N];
  omp_interop_t SyncObj = omp_interop_none;
  int DeviceNum = omp_get_default_device();

  // clang-format off
  #pragma omp target nowait depend(out : x [0:N])                                \
          map(from : x [0:N]) device(DeviceNum)
  // clang-format on
  vectorSet(N, 1.0, x);

#pragma omp task depend(out : y [0:N])
  vectorSet(N, -1.0, y);

  // Get SyncObject for synchronization
  // clang-format off
  #pragma omp interop init(targetsync : SyncObj) device(DeviceNum)               \
          depend(in : x [0:N]) depend(inout : y [0:N])
  // clang-format on

  int ForeignContextId = (int)omp_get_interop_int(SyncObj, omp_ipr_fr_id, NULL);
  char *ForeignContextName =
      (char *)omp_get_interop_str(SyncObj, omp_ipr_fr_name, NULL);

  if (SyncObj != omp_interop_none && ForeignContextId == omp_ifr_amdhsa) {
    printf("OpenMP working with %s runtime to execute async memcpy.\n",
           ForeignContextName);
    int Status;
    omp_get_interop_ptr(SyncObj, omp_ipr_targetsync, &Status);

    if (Status != omp_irc_success) {
      fprintf(stderr, "ERROR: Failed to get %s stream, rt error = %d.\n",
              ForeignContextName, Status);
      if (Status == omp_irc_no_value)
        fprintf(stderr, "Parameters valid, but no meaningful value available.");
      exit(1);
    }

    vectorCopy(N, x, y);
  } else {
    // Execute as OpenMP offload
    printf("Notice: Offloading myCopy to perform memcpy.\n");
    // clang-format off
  #pragma omp target depend(in : x [0:N]) depend(inout : y [0:N]) nowait         \
          map(to : x [0:N]) map(tofrom : y [0:N]) device(DeviceNum)
    // clang-format on
    vectorCopy(N, x, y);
  }

  // This also ensures foreign tasks complete
#pragma omp interop destroy(SyncObj) nowait depend(out : y [0:N])

#pragma omp target depend(inout : x [0:N])
  vectorScale(N, ScaleFactor, x);

#pragma omp taskwait

  printf("(1 : 16384) %f:%f\n", y[0], y[N - 1]);
  printf("(2 : 32768) %f:%f\n", x[0], x[N - 1]);

  return 0;
}

// ToDo: Add meaningful checks; the following is a placeholder.

// CHECK: OpenMP working with amdhsa backend runtime to execute async memcpy
