// RUN: %libomp-compile -D_GNU_SOURCE
// RUN: env OMP_PLACES=threads %libomp-run
// RUN: env OMP_PLACES=cores %libomp-run
// RUN: env OMP_PLACES=sockets %libomp-run
// RUN: env OMP_PLACES=cores RUN_OUT_OF_ORDER=1 %libomp-run
// REQUIRES: linux

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libomp_test_affinity.h"
#include "libomp_test_topology.h"

// Check openmp place list to make sure it follow KMP_HW_SUBSET restriction
static int compare_hw_subset_places(const place_list_t *openmp_places,
                                    topology_obj_type_t type, int nsockets,
                                    int ncores_per_socket,
                                    int nthreads_per_core) {
  int i, j, expected_total, expected_per_place;
  if (type == TOPOLOGY_OBJ_THREAD) {
    expected_total = nsockets * ncores_per_socket * nthreads_per_core;
    expected_per_place = 1;
  } else if (type == TOPOLOGY_OBJ_CORE) {
    expected_total = nsockets * ncores_per_socket;
    expected_per_place = nthreads_per_core;
  } else {
    expected_total = nsockets;
    expected_per_place = ncores_per_socket;
  }
  if (openmp_places->num_places != expected_total) {
    fprintf(stderr, "error: KMP_HW_SUBSET did not half each resource layer!\n");
    printf("openmp_places places:\n");
    topology_print_places(openmp_places);
    printf("\n");
    return EXIT_FAILURE;
  }
  for (i = 0; i < openmp_places->num_places; ++i) {
    int count = affinity_mask_count(openmp_places->masks[i]);
    if (count != expected_per_place) {
      fprintf(stderr, "error: place %d has %d OS procs instead of %d\n", i,
              count, expected_per_place);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

static int check_places() {
  char buf[100];
  topology_obj_type_t type;
  const char *value;
  int status = EXIT_SUCCESS;
  place_list_t *threads, *cores, *sockets, *openmp_places;
  threads = topology_alloc_type_places(TOPOLOGY_OBJ_THREAD);
  cores = topology_alloc_type_places(TOPOLOGY_OBJ_CORE);
  sockets = topology_alloc_type_places(TOPOLOGY_OBJ_SOCKET);

  if (threads->num_places <= 1) {
    printf("Only one hardware thread to execute on. Skipping test.\n");
    return status;
  }

  value = getenv("OMP_PLACES");
  if (!value) {
    fprintf(stderr,
            "error: OMP_PLACES must be set to one of threads,cores,sockets!\n");
    return EXIT_FAILURE;
  }
  if (strcmp(value, "threads") == 0)
    type = TOPOLOGY_OBJ_THREAD;
  else if (strcmp(value, "cores") == 0)
    type = TOPOLOGY_OBJ_CORE;
  else if (strcmp(value, "sockets") == 0)
    type = TOPOLOGY_OBJ_SOCKET;
  else {
    fprintf(stderr,
            "error: OMP_PLACES must be one of threads,cores,sockets!\n");
    return EXIT_FAILURE;
  }

  // Calculate of num threads per core, num cores per socket, & num sockets
  if (cores->num_places <= 0) {
    printf("Invalid number of cores (%d). Skipping test.\n", cores->num_places);
    return status;
  } else if (sockets->num_places <= 0) {
    printf("Invalid number of sockets (%d). Skipping test.\n",
           cores->num_places);
    return status;
  }
  int nthreads_per_core = threads->num_places / cores->num_places;
  int ncores_per_socket = cores->num_places / sockets->num_places;
  int nsockets = sockets->num_places;

  if (nsockets * ncores_per_socket * nthreads_per_core != threads->num_places) {
    printf("Only uniform topologies can be tested. Skipping test.\n");
    return status;
  }

  // Use half the resources of every level
  if (nthreads_per_core > 1)
    nthreads_per_core /= 2;
  if (ncores_per_socket > 1)
    ncores_per_socket /= 2;
  if (nsockets > 1)
    nsockets /= 2;

  if (getenv("RUN_OUT_OF_ORDER")) {
    snprintf(buf, sizeof(buf), "%dt,%ds,%dc", nthreads_per_core, nsockets,
             ncores_per_socket);
  } else {
    snprintf(buf, sizeof(buf), "%ds,%dc,%dt", nsockets, ncores_per_socket,
             nthreads_per_core);
  }
  setenv("KMP_HW_SUBSET", buf, 1);

  openmp_places = topology_alloc_openmp_places();
  status = compare_hw_subset_places(openmp_places, type, nsockets,
                                    ncores_per_socket, nthreads_per_core);
  topology_free_places(threads);
  topology_free_places(cores);
  topology_free_places(sockets);
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
