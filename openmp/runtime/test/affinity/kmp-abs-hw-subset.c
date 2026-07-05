// RUN: %libomp-compile -D_GNU_SOURCE
// RUN: env OMP_PLACES=threads %libomp-run 1 0
// RUN: env OMP_PLACES=threads %libomp-run 1 1
// RUN: env OMP_PLACES=threads %libomp-run 2 1
// RUN: env OMP_PLACES=threads %libomp-run 2 2
// RUN: env OMP_PLACES=threads %libomp-run 3 1
// RUN: env OMP_PLACES=threads %libomp-run 3 2
// REQUIRES: linux
//
// The test requires topologies with sockets, cores, threads layers where
// the socket layer contains multiple threads.
// The s390x architecture does not produce this topology and seems to have
// one thread per socket.
// UNSUPPORTED: s390x-target-arch

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libomp_test_affinity.h"
#include "libomp_test_topology.h"

// Check openmp place list to make sure it follow KMP_HW_SUBSET restriction
static int compare_abs_hw_subset_places(const place_list_t *openmp_places,
                                        int nthreads, int offset) {
  int i, j, expected_per_place;
  if (openmp_places->num_places != nthreads) {
    fprintf(
        stderr,
        "error: KMP_HW_SUBSET did not restrict the thread resource layer!\n");
    printf("openmp_places places:\n");
    topology_print_places(openmp_places);
    printf("\n");
    return EXIT_FAILURE;
  }
  for (i = 0; i < openmp_places->num_places; ++i) {
    int count = affinity_mask_count(openmp_places->masks[i]);
    if (count != 1) {
      fprintf(stderr, "error: place %d has %d OS procs instead of %d\n", i,
              count, expected_per_place);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

static int check_places(int nthreads, int offset) {
  char buf[100];
  topology_obj_type_t type;
  const char *value;
  int status = EXIT_SUCCESS;
  place_list_t *threads, *openmp_places;
  threads = topology_alloc_type_places(TOPOLOGY_OBJ_THREAD);

  if (threads->num_places <= 1) {
    printf("Only one hardware thread to execute on. Skipping test.\n");
    return status;
  }

  if (nthreads + offset > threads->num_places) {
    printf("Only %d total hardware threads to execute on. Skipping test with "
           "nthreads=%d and offset=%d (too big).\n",
           threads->num_places, nthreads, offset);
    return status;
  }

  value = getenv("OMP_PLACES");
  if (!value) {
    fprintf(stderr, "error: OMP_PLACES must be set to threads!\n");
    return EXIT_FAILURE;
  }

  snprintf(buf, sizeof(buf), ":1s,%dt@%d", nthreads, offset);
  setenv("KMP_HW_SUBSET", buf, 1);

  openmp_places = topology_alloc_openmp_places();
  status = compare_abs_hw_subset_places(openmp_places, nthreads, offset);
  topology_free_places(threads);
  topology_free_places(openmp_places);
  return status;
}

int main(int argc, char **argv) {
  int offset = 0;
  int nthreads = 1;

  if (!topology_using_full_mask()) {
    printf("Thread does not have access to all logical processors. Skipping "
           "test.\n");
    return EXIT_SUCCESS;
  }

  if (argc != 3) {
    fprintf(stderr, "usage: %s <nthreads> <offset>\n", argv[0]);
    return EXIT_FAILURE;
  }

  nthreads = atoi(argv[1]);
  offset = atoi(argv[2]);

  return check_places(nthreads, offset);
}
