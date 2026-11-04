// Verify the RFC verbosity levels: LIBOMPTARGET_DEBUG=Type:Level selects how
// much detail the user-facing INFO messages print (L1 summary, L2 detailed,
// L3 full), and the legacy LIBOMPTARGET_INFO bitmask prints full detail. In
// OMPTARGET_DEBUG builds the selected messages use the debug stream ("--> ")
// instead of the "info: " prefix, so matches below accept either.
//
// RUN: %libomptarget-compile-generic -gline-tables-only
// RUN: env LIBOMPTARGET_DEBUG=KernelArgs:1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=KA1
// RUN: env LIBOMPTARGET_DEBUG=KernelArgs:2 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=KA2
// RUN: env LIBOMPTARGET_DEBUG=KernelArgs:3 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=KA3
// RUN: env LIBOMPTARGET_DEBUG=DataCopy:1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=DT1
// RUN: env LIBOMPTARGET_DEBUG=DataCopy:2 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=DT2
// RUN: env LIBOMPTARGET_DEBUG=DataCopy:3 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=DT3
// RUN: env LIBOMPTARGET_DEBUG=Info:1,DataCopy:3 %libomptarget-run-generic \
// RUN:   2>&1 | %fcheck-generic -check-prefix=DT3
// RUN: env LIBOMPTARGET_DEBUG=MappingChanged:1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=MC1
// RUN: env LIBOMPTARGET_DEBUG=MappingChanged:2 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=MC2
// RUN: env LIBOMPTARGET_DEBUG=MappingChanged:3 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=MC3
// RUN: env LIBOMPTARGET_DEBUG=PluginKernel:1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=PK1
// RUN: env LIBOMPTARGET_DEBUG=PluginKernel:2 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=PK2
// RUN: env LIBOMPTARGET_INFO=-1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=FULL
//
// Without debug info there are no mapping names.
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=DataCopy:2 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -check-prefix=NONAME

#include <stdio.h>

#define N 64

int main() {
  int A[N];
  int B[4] = {1, 2, 3, 4};
  int s = 3;
  int t = 7;

  fprintf(stderr, "MARKER begin target region\n");
#pragma omp target map(tofrom : A) firstprivate(s, B)
  {
    A[0] = s + t;
    A[1] = B[1];
  }
  fprintf(stderr, "MARKER end target region\n");

  // KernelArgs:1 -- region header only, no per-argument lines.
  // KA1: MARKER begin target region
  // KA1-NOT: firstprivate(
  // KA1: {{.*}}OpenMP kernel{{.*}}with {{[0-9]+}} arguments:
  // KA1-NOT: firstprivate(
  // KA1: MARKER end target region

  // KernelArgs:2 -- header + explicit named args, but not implicit ones.
  // KA2: MARKER begin target region
  // KA2-NOT: (implicit)
  // KA2: firstprivate(s)
  // KA2-NOT: (implicit)
  // KA2: MARKER end target region

  // KernelArgs:3 -- header + explicit args + implicit args.
  // KA3: {{.*}}OpenMP kernel{{.*}}with {{[0-9]+}} arguments:
  // KA3: (implicit)

  // DataCopy:1 -- no per-copy or synchronization lines at all.
  // DT1: MARKER begin target region
  // DT1-NOT: Copying data
  // DT1-NOT: Waiting for asynchronous operations
  // DT1: MARKER end target region

  // DataCopy:2 -- per-copy lines for mapped data, but without pointers
  // (note "device, Size=" adjacency: no HstPtr/TgtPtr elements). The packed
  // firstprivate buffer holding B has no map entry, so it is L3 only, as is
  // the synchronization line.
  // DT2-NOT: Waiting for asynchronous operations
  // DT2: Copying data from host to device, Size=256, Name=A
  // DT2-NOT: Copying data from host to device, Size=16
  // DT2-NOT: Waiting for asynchronous operations
  // DT2: Copying data from device to host, Size=256, Name=A
  // DT2-NOT: Waiting for asynchronous operations
  // DT2: A[0] = 10

  // DataCopy:3 -- per-copy lines including host/target pointers, the
  // runtime-internal packed firstprivate copy, which has no name, and the
  // synchronization line with its source location.
  // DT3: Copying data from host to device, HstPtr={{.*}}, Size=256, Name=A
  // DT3: Copying data from host to device, HstPtr={{.*}}, Size=16, Name=unknown
  // DT3: Waiting for asynchronous operations to complete at {{.*}}info_level.c:{{[0-9]+}}:{{[0-9]+}}

  // Mapped copies are still printed when the mapping name is unknown.
  // NONAME: Copying data from host to device, Size=256, Name=unknown
  // NONAME: Copying data from device to host, Size=256, Name=unknown

  // MappingChanged:1 -- Size and Name only (no refcounts, no addresses).
  // MC1: Creating new map entry Size={{[0-9]+}}, Name=

  // MappingChanged:2 -- adds the reference counts.
  // MC2: Creating new map entry Size={{[0-9]+}}, DynRefCount=

  // MappingChanged:3 -- adds all the addresses.
  // MC3: Creating new map entry with HstPtrBase=

  // PluginKernel:1 -- launch line ends at the execution mode (no tripcount).
  // PK1: Launching kernel {{.*}}mode{{$}}

  // PluginKernel:2 -- launch line adds the tripcount.
  // PK2: Launching kernel {{.*}}mode with tripcount {{[0-9]+}}

  // The legacy numeric bitmask keeps the full (L3) output with the info prefix.
  // FULL: info: {{.*}}OpenMP kernel{{.*}}with {{[0-9]+}} arguments:
  // FULL: info: firstprivate(
  // FULL: info: Creating new map entry with HstPtrBase=
  // FULL: info: Copying data from host to device, HstPtr=
  // FULL: info: Launching kernel {{.*}}mode with tripcount {{[0-9]+}}

  printf("A[0] = %d\n", A[0]);
  // KA1: A[0] = 10
  // KA3: A[0] = 10
  // DT3: A[0] = 10
  // MC1: A[0] = 10
  // PK2: A[0] = 10
  // FULL: A[0] = 10
  // NONAME: A[0] = 10
  return 0;
}
