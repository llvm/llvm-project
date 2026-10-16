// Check that specifying device as omp_get_initial_device():
// - Doesn't cause the runtime to fail.
// - Offloads code to the host.
// - Doesn't transfer data.  In this case, just check that neither host data nor
//   default device data are affected by the specified transfers.
// - Works whether it's specified directly or as the default device.

// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

static void check(char *X, int Dev) {
  printf("  host X = %c\n", *X);
  char DV = -1;
#pragma omp target device(Dev) map(from : DV)
  DV = *X;
  printf("device X = %c\n", DV);
}

#define CHECK_DATA() check(&X, DevDefault)

int main(void) {
  int DevDefault = omp_get_default_device();
  int DevInit = omp_get_initial_device();

  //--------------------------------------------------
  // Initialize data on the host and default device.
  //--------------------------------------------------

  //      CHECK:   host X = h
  // CHECK-NEXT: device X = d
  char X = 'd';
#pragma omp target enter data map(to : X)
  X = 'h';
  CHECK_DATA();

//--------------------------------------------------
// Check behavior when specifying host directly.
//--------------------------------------------------

// CHECK-NEXT: omp_is_initial_device() = 1
// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target device(DevInit) map(always, tofrom : X)
  printf("omp_is_initial_device() = %d\n", omp_is_initial_device());
  CHECK_DATA();

// CHECK-NEXT: omp_is_initial_device() = 1
// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target teams device(DevInit) num_teams(1) map(always, tofrom : X)
  printf("omp_is_initial_device() = %d\n", omp_is_initial_device());
  CHECK_DATA();

// Check that __kmpc_push_target_tripcount_mapper doesn't fail. I'm not sure
// how to check that it actually pushes to the initial device.
#pragma omp target teams device(DevInit) num_teams(1)
#pragma omp distribute
  for (int i = 0; i < 2; ++i)
    ;

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target data device(DevInit) map(always, tofrom : X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target enter data device(DevInit) map(always, to : X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target exit data device(DevInit) map(always, from : X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target update device(DevInit) to(X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target update device(DevInit) from(X)
  ;
  CHECK_DATA();

  //--------------------------------------------------
  // Check behavior when device defaults to host.
  //--------------------------------------------------

  omp_set_default_device(DevInit);

// CHECK-NEXT: omp_is_initial_device() = 1
// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target map(always, tofrom : X)
  printf("omp_is_initial_device() = %d\n", omp_is_initial_device());
  CHECK_DATA();

// CHECK-NEXT: omp_is_initial_device() = 1
// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target teams num_teams(1) map(always, tofrom : X)
  printf("omp_is_initial_device() = %d\n", omp_is_initial_device());
  CHECK_DATA();

// Check that __kmpc_push_target_tripcount_mapper doesn't fail. I'm not sure
// how to check that it actually pushes to the initial device.
#pragma omp target teams num_teams(1)
#pragma omp distribute
  for (int i = 0; i < 2; ++i)
    ;

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target data map(always, tofrom : X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target enter data map(always, to : X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target exit data map(always, from : X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target update to(X)
  ;
  CHECK_DATA();

// CHECK-NEXT:   host X = h
// CHECK-NEXT: device X = d
#pragma omp target update from(X)
  ;
  CHECK_DATA();

  return 0;
}
