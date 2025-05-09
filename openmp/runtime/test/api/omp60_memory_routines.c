// RUN: %libomp-compile -Wl,--export-dynamic && %libomp-run

// REQUIRES: linux

// Test OpenMP 6.0 memory management routines.
// Test host runtime's basic support with an emulated offload runtime.

#include <stdlib.h>
#include <omp.h>

#define NUM_DEVICES 4

//
// Required offload runtime interfaces
//
extern int __tgt_get_num_devices(void) { return NUM_DEVICES; }

extern int __tgt_get_mem_resources(int num_devices, const int *devices,
                                   int host, omp_memspace_handle_t memspace,
                                   int *resources) {
  int i;
  // We expect valid inputs within this test.
  int num_resources = num_devices;
  if (resources) {
    // Simple resouce ID mapping example in the backend (=device ID).
    // This does not represent any real backend.
    for (i = 0; i < num_devices; i++)
      resources[i] = devices[i];
  }
  return num_resources;
}

extern void *__tgt_omp_alloc(size_t size, omp_allocator_handle_t allocator) {
  return malloc(size);
}

extern void __tgt_omp_free(void *ptr, omp_allocator_handle_t allocator) {
  free(ptr);
}

// Code above is also used by the corresponding Fortran test

#define CHECK_OR_RET_FAIL(Expr)                                                \
  do {                                                                         \
    if (!(Expr))                                                               \
      return EXIT_FAILURE;                                                     \
  } while (0)

// Test user-initialized allocator with the given memory space
static int test_user_allocator(omp_memspace_handle_t ms) {
  omp_allocator_handle_t al = omp_null_allocator;
  al = omp_init_allocator(ms, 0, NULL);
  CHECK_OR_RET_FAIL(al != omp_null_allocator);
  void *m = omp_alloc(1024, al);
  CHECK_OR_RET_FAIL(m != NULL);
  omp_free(m, al);
  omp_destroy_allocator(al);
  return EXIT_SUCCESS;
}

static int test_allocator(omp_allocator_handle_t al) {
  void *m = omp_alloc(1024, al);
  CHECK_OR_RET_FAIL(m != NULL);
  omp_free(m, al);
  omp_destroy_allocator(al);
  return EXIT_SUCCESS;
}

static int test_mem_space(void) {
  int i, count;
  int num_devices = omp_get_num_devices();
  CHECK_OR_RET_FAIL(num_devices == NUM_DEVICES);

  int *all_devices = (int *)malloc(sizeof(int) * num_devices);
  for (i = 0; i < num_devices; i++)
    all_devices[i] = i;

  omp_memspace_handle_t predef = omp_default_mem_space;
  omp_memspace_handle_t ms1 = omp_null_mem_space;
  omp_memspace_handle_t ms2 = omp_null_mem_space;

  // Test the following API routines.
  // * omp_get_device_memspace
  // * omp_get_device_and_host_memspace
  // * omp_get_devices_memspace
  // * omp_get_devices_and_host_memspace
  // Test if runtime returns the same memory space handle for the same input.
  // Test if we can use the memory space to intialize allocator.
  for (i = 0; i < num_devices; i++) {
    ms1 = omp_get_device_memspace(i, predef);
    CHECK_OR_RET_FAIL(ms1 != omp_null_mem_space);
    ms2 = omp_get_device_memspace(i, predef);
    CHECK_OR_RET_FAIL(ms1 == ms2);
    CHECK_OR_RET_FAIL(test_user_allocator(ms1) == EXIT_SUCCESS);
    ms1 = ms2 = omp_null_mem_space;

    ms1 = omp_get_device_and_host_memspace(i, predef);
    CHECK_OR_RET_FAIL(ms1 != omp_null_mem_space);
    ms2 = omp_get_device_and_host_memspace(i, predef);
    CHECK_OR_RET_FAIL(ms1 == ms2);
    CHECK_OR_RET_FAIL(test_user_allocator(ms1) == EXIT_SUCCESS);
    ms1 = ms2 = omp_null_mem_space;

    for (count = 1; i + count <= num_devices; count++) {
      int *devices = &all_devices[i];
      ms1 = omp_get_devices_memspace(count, devices, predef);
      CHECK_OR_RET_FAIL(ms1 != omp_null_mem_space);
      ms2 = omp_get_devices_memspace(count, devices, predef);
      CHECK_OR_RET_FAIL(ms1 == ms2);
      CHECK_OR_RET_FAIL(test_user_allocator(ms1) == EXIT_SUCCESS);
      ms1 = ms2 = omp_null_mem_space;

      ms1 = omp_get_devices_and_host_memspace(count, devices, predef);
      CHECK_OR_RET_FAIL(ms1 != omp_null_mem_space);
      ms2 = omp_get_devices_and_host_memspace(count, devices, predef);
      CHECK_OR_RET_FAIL(ms1 == ms2);
      CHECK_OR_RET_FAIL(test_user_allocator(ms1) == EXIT_SUCCESS);
      ms1 = ms2 = omp_null_mem_space;
    }
  }

  // Test the following API routines.
  // * omp_get_devices_all_memspace
  // Test if runtime returns the same memory space handle for the same input.
  ms1 = omp_get_devices_all_memspace(predef);
  CHECK_OR_RET_FAIL(ms1 != omp_null_mem_space);
  ms2 = omp_get_devices_all_memspace(predef);
  CHECK_OR_RET_FAIL(ms1 == ms2);

  free(all_devices);

  return EXIT_SUCCESS;
}

static int test_mem_allocator(void) {
  int i, count;
  int num_devices = omp_get_num_devices();
  CHECK_OR_RET_FAIL(num_devices == NUM_DEVICES);

  int *all_devices = (int *)malloc(sizeof(int) * num_devices);
  for (i = 0; i < num_devices; i++)
    all_devices[i] = i;

  omp_memspace_handle_t predef = omp_default_mem_space;
  omp_allocator_handle_t al = omp_null_allocator;

  // Test the following API routines.
  // * omp_get_device_allocator
  // * omp_get_device_and_host_allocator
  // * omp_get_devices_allocator
  // * omp_get_devices_and_host_allocator
  for (i = 0; i < num_devices; i++) {
    al = omp_get_device_allocator(i, predef);
    CHECK_OR_RET_FAIL(al != omp_null_allocator);
    CHECK_OR_RET_FAIL(test_allocator(al) == EXIT_SUCCESS);
    al = omp_null_allocator;

    al = omp_get_device_and_host_allocator(i, predef);
    CHECK_OR_RET_FAIL(al != omp_null_allocator);
    CHECK_OR_RET_FAIL(test_allocator(al) == EXIT_SUCCESS);
    al = omp_null_allocator;

    for (count = 1; i + count <= num_devices; count++) {
      int *devices = &all_devices[i];
      al = omp_get_devices_allocator(count, devices, predef);
      CHECK_OR_RET_FAIL(al != omp_null_allocator);
      CHECK_OR_RET_FAIL(test_allocator(al) == EXIT_SUCCESS);
      al = omp_null_allocator;

      al = omp_get_devices_and_host_allocator(count, devices, predef);
      CHECK_OR_RET_FAIL(al != omp_null_allocator);
      CHECK_OR_RET_FAIL(test_allocator(al) == EXIT_SUCCESS);
      al = omp_null_allocator;
    }
  }

  // Test the following API routines.
  // * omp_get_devices_all_allocator
  al = omp_get_devices_all_allocator(predef);
  CHECK_OR_RET_FAIL(al != omp_null_allocator);
  CHECK_OR_RET_FAIL(test_allocator(al) == EXIT_SUCCESS);

  free(all_devices);

  return EXIT_SUCCESS;
}

// Just test what we can expect from the emulated backend.
static int test_sub_mem_space(void) {
  int i;
  omp_memspace_handle_t ms = omp_null_mem_space;
  ms = omp_get_devices_all_memspace(omp_default_mem_space);
  CHECK_OR_RET_FAIL(ms != omp_null_mem_space);
  int num_resources = omp_get_memspace_num_resources(ms);
  CHECK_OR_RET_FAIL(num_resources == NUM_DEVICES);

  // Check if single-resource sub memspace is correctly returned.
  for (i = 0; i < num_resources; i++) {
    omp_memspace_handle_t sub = omp_get_submemspace(ms, 1, &i);
    CHECK_OR_RET_FAIL(sub != omp_null_mem_space);
    CHECK_OR_RET_FAIL(sub != ms);
    int num_sub_resources = omp_get_memspace_num_resources(sub);
    CHECK_OR_RET_FAIL(num_sub_resources == 1);
  }

  // Check if all-resrouce sub memspace is correctly returned.
  int *resources = (int *)malloc(sizeof(int) * num_resources);
  for (i = 0; i < num_resources; i++)
    resources[i] = i;
  omp_memspace_handle_t sub = omp_get_submemspace(ms, num_resources, resources);
  CHECK_OR_RET_FAIL(sub != omp_null_mem_space);
  CHECK_OR_RET_FAIL(sub == ms);

  return EXIT_SUCCESS;
}

int main() {
  int rc = test_mem_space();
  CHECK_OR_RET_FAIL(rc == EXIT_SUCCESS);

  rc = test_mem_allocator();
  CHECK_OR_RET_FAIL(rc == EXIT_SUCCESS);

  rc = test_sub_mem_space();
  CHECK_OR_RET_FAIL(rc == EXIT_SUCCESS);

  return rc;
}
