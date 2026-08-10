// RUN: %libomp-compile -D_GNU_SOURCE
// RUN: env OMP_PLACES=threads %libomp-run
// RUN: env OMP_PLACES=cores %libomp-run
// RUN: env OMP_PLACES=sockets %libomp-run
// REQUIRES: linux

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libomp_test_affinity.h"
#include "libomp_test_topology.h"

// Compare place lists. The order is not taken into consideration here.
// The OS detection might have the cores/sockets in a different
// order from the runtime.
static int compare_places(const place_list_t *p1, const place_list_t *p2) {
  int i, j;
  if (p1->num_places != p2->num_places) {
    fprintf(stderr, "error: places do not have same number of places! (p1 has "
                    "%d, p2 has %d)\n",
            p1->num_places, p2->num_places);
    printf("p1 places:\n");
    topology_print_places(p1);
    printf("\n");
    printf("p2 places:\n");
    topology_print_places(p2);
    return EXIT_FAILURE;
  }
  for (i = 0; i < p1->num_places; ++i) {
    int found = 0;
    for (j = 0; j < p2->num_places; ++j) {
      if (affinity_mask_equal(p1->masks[i], p2->masks[j])) {
        found = 1;
        break;
      }
    }
    if (!found) {
      printf("Found difference in places!\n");
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
  const char *value = getenv("OMP_PLACES");
  if (!value) {
    fprintf(stderr, "error: must set OMP_PLACES envirable for this test!\n");
    return EXIT_FAILURE;
  }
  place_list_t *places, *openmp_places;
  if (strcmp(value, "sockets") == 0) {
    places = topology_alloc_type_places(TOPOLOGY_OBJ_SOCKET);
  } else if (strcmp(value, "cores") == 0) {
    places = topology_alloc_type_places(TOPOLOGY_OBJ_CORE);
  } else if (strcmp(value, "threads") == 0) {
    places = topology_alloc_type_places(TOPOLOGY_OBJ_THREAD);
  } else {
    fprintf(stderr,
            "error: OMP_PLACES must be one of threads,cores,sockets!\n");
    return EXIT_FAILURE;
  }
  openmp_places = topology_alloc_openmp_places();
  status = compare_places(places, openmp_places);
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
