// RUN: %libomptarget-compile-run-and-check-generic

// Exercise omp_target_associate_ptr and omp_target_disassociate_ptr: presence
// and mapped-pointer queries, reuse of the same host address, a missing
// association, that device memory from omp_target_alloc is not released, and
// that a mapping created by "target enter data" cannot be disassociated.

#include <omp.h>
#include <stdio.h>

int main() {
  int Dev = omp_get_default_device();
  int HostVal = 42;
  int *DevPtr = (int *)omp_target_alloc(sizeof(int), Dev);
  if (!DevPtr) {
    printf("omp_target_alloc failed\n");
    return 1;
  }

  // CHECK: present before associate: 0
  printf("present before associate: %d\n",
         omp_target_is_present(&HostVal, Dev));

  int Rc = omp_target_associate_ptr(&HostVal, DevPtr, sizeof(int), 0, Dev);
  // CHECK: associate: 0
  printf("associate: %d\n", Rc);

  // CHECK: present after associate: 1
  printf("present after associate: %d\n", omp_target_is_present(&HostVal, Dev));
  // CHECK: mapped matches: 1
  printf("mapped matches: %d\n",
         omp_get_mapped_ptr(&HostVal, Dev) == (void *)DevPtr);

  Rc = omp_target_disassociate_ptr(&HostVal, Dev);
  // CHECK: disassociate: 0
  printf("disassociate: %d\n", Rc);

  // CHECK: present after disassociate: 0
  printf("present after disassociate: %d\n",
         omp_target_is_present(&HostVal, Dev));
  // CHECK: mapped after disassociate is null: 1
  printf("mapped after disassociate is null: %d\n",
         omp_get_mapped_ptr(&HostVal, Dev) == NULL);

  for (int I = 0; I < 8; ++I) {
    if (omp_target_associate_ptr(&HostVal, DevPtr, sizeof(int), 0, Dev)) {
      printf("repeated associate failed at %d\n", I);
      omp_target_free(DevPtr, Dev);
      return 1;
    }
    if (omp_target_disassociate_ptr(&HostVal, Dev)) {
      printf("repeated disassociate failed at %d\n", I);
      omp_target_free(DevPtr, Dev);
      return 1;
    }
  }
  // CHECK: repeated associate/disassociate: ok
  printf("repeated associate/disassociate: ok\n");

  Rc = omp_target_disassociate_ptr(&HostVal, Dev);
  // CHECK: disassociate missing: 1
  printf("disassociate missing: %d\n", Rc != 0);

  // Device storage is independent of the host association.
  int In = 7, Out = 0;
  if (omp_target_memcpy(DevPtr, &In, sizeof(int), 0, 0, Dev,
                        omp_get_initial_device()) ||
      omp_target_memcpy(&Out, DevPtr, sizeof(int), 0, 0,
                        omp_get_initial_device(), Dev)) {
    printf("omp_target_memcpy failed\n");
    omp_target_free(DevPtr, Dev);
    return 1;
  }
  // CHECK: device memory after disassociate: 7
  printf("device memory after disassociate: %d\n", Out);

  int MappedByEnter = 0;
#pragma omp target enter data map(alloc : MappedByEnter)
  Rc = omp_target_disassociate_ptr(&MappedByEnter, Dev);
  // CHECK: disassociate of mapped data: 1
  printf("disassociate of mapped data: %d\n", Rc != 0);
#pragma omp target exit data map(delete : MappedByEnter)

  omp_target_free(DevPtr, Dev);
  return 0;
}
