// RUN: %libomp-compile -D_GNU_SOURCE
// RUN: env KMP_AFFINITY=granularity=thread,compact %libomp-run
// RUN: env KMP_AFFINITY=granularity=core,compact %libomp-run
// RUN: env KMP_AFFINITY=granularity=socket,compact %libomp-run
// REQUIRES: linux
// UNSUPPORTED: linux

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libomp_test_affinity.h"
#include "libomp_test_topology.h"

// Compare place lists. Make sure every place in p1 is in p2.
static int compare_places(const place_list_t *p1, const place_list_t *p2) {
  int i, j;
  for (i = 0; i < p1->num_places; ++i) {
    int found = 0;
    for (j = 0; j < p2->num_places; ++j) {
      if (affinity_mask_equal(p1->masks[i], p2->masks[j])) {
        found = 1;
        break;
      }
    }
    if (!found) {
      printf("Found place in p1 not in p2!\n");
      printf("p1 places:\n");
      topology_print_places(p1);
      printf("\n");
      printf("p2 places:\n");
      topology_print_places(p2);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

static int check_places() {
  int status;
  const char *value = getenv("KMP_AFFINITY");
  if (!value) {
    fprintf(stderr, "error: must set OMP_PLACES envirable for this test!\n");
    return EXIT_FAILURE;
  }
  place_list_t *places, *openmp_places;
  if (strstr(value, "socket")) {
    places = topology_alloc_type_places(TOPOLOGY_OBJ_SOCKET);
  } else if (strstr(value, "core")) {
    places = topology_alloc_type_places(TOPOLOGY_OBJ_CORE);
  } else if (strstr(value, "thread")) {
    places = topology_alloc_type_places(TOPOLOGY_OBJ_THREAD);
  } else {
    fprintf(
        stderr,
        "error: KMP_AFFINITY granularity must be one of thread,core,socket!\n");
    return EXIT_FAILURE;
  }
  openmp_places = topology_alloc_openmp_places();
  status = compare_places(openmp_places, places);
  topology_free_places(places);
  topology_free_places(openmp_places);
  return status;
}

int main() {
  if (!topology_using_full_mask()) {
    printf("Thread does not have access to all logical processors. Skipping "
           "test.\n");
    return EXIT_SUCCESS;
  }
  return check_places();
}
