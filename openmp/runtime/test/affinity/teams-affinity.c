// RUN: %libomp-compile && env OMP_PLACES=cores OMP_TEAMS_THREAD_LIMIT=1 KMP_TEAMS_THREAD_LIMIT=256 %libomp-run
// RUN: %libomp-compile && env OMP_PLACES=cores OMP_TEAMS_THREAD_LIMIT=1 KMP_TEAMS_THREAD_LIMIT=256 KMP_HOT_TEAMS_MAX_LEVEL=2 %libomp-run
// RUN: %libomp-compile && env OMP_PLACES=cores OMP_TEAMS_THREAD_LIMIT=1 KMP_TEAMS_THREAD_LIMIT=256 KMP_TEAMS_PROC_BIND=close %libomp-run
// RUN: %libomp-compile && env OMP_PLACES=cores OMP_TEAMS_THREAD_LIMIT=1 KMP_TEAMS_THREAD_LIMIT=256 KMP_TEAMS_PROC_BIND=close KMP_HOT_TEAMS_MAX_LEVEL=2 %libomp-run
// RUN: %libomp-compile && env OMP_PLACES=cores OMP_TEAMS_THREAD_LIMIT=1 KMP_TEAMS_THREAD_LIMIT=256 KMP_TEAMS_PROC_BIND=primary %libomp-run
// RUN: %libomp-compile && env OMP_PLACES=cores OMP_TEAMS_THREAD_LIMIT=1 KMP_TEAMS_THREAD_LIMIT=256 KMP_TEAMS_PROC_BIND=primary KMP_HOT_TEAMS_MAX_LEVEL=2 %libomp-run
// REQUIRES: linux
// UNSUPPORTED: clang-5, clang-6, clang-7, clang-8, clang-9, clang-10
// UNSUPPORTED: gcc-5, gcc-6, gcc-7, gcc-8
// UNSUPPORTED: icc
//
// KMP_TEAMS_THREAD_LIMIT limits the number of total teams
// OMP_TEAMS_THREAD_LIMIT limits the number of threads per team

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libomp_test_affinity.h"
#include "libomp_test_topology.h"

#define _STR(X) #X
#define STR(X) _STR(X)

#ifndef MAX_NTEAMS
#define MAX_NTEAMS 256
#endif

static void set_default_max_nteams() {
  // Do not overwrite if already in environment
  setenv("KMP_TEAMS_THREAD_LIMIT", STR(MAX_NTEAMS), 0);
}

static int get_max_nteams() {
  int max_nteams;
  const char *value = getenv("KMP_TEAMS_THREAD_LIMIT");
  if (!value) {
    fprintf(stderr, "KMP_TEAMS_THREAD_LIMIT must be set!\n");
    exit(EXIT_FAILURE);
  }
  max_nteams = atoi(value);
  if (max_nteams <= 0)
    max_nteams = 1;
  if (max_nteams > MAX_NTEAMS)
    max_nteams = MAX_NTEAMS;
  return max_nteams;
}

// Return the value in KMP_TEAMS_PROC_BIND
static omp_proc_bind_t get_teams_proc_bind() {
  // defaults to spread
  omp_proc_bind_t proc_bind = omp_proc_bind_spread;
  const char *value = getenv("KMP_TEAMS_PROC_BIND");
  if (value) {
    if (strcmp(value, "spread") == 0) {
      proc_bind = omp_proc_bind_spread;
    } else if (strcmp(value, "close") == 0) {
      proc_bind = omp_proc_bind_close;
    } else if (strcmp(value, "primary") == 0 || strcmp(value, "master") == 0) {
      proc_bind = omp_proc_bind_master;
    } else {
      fprintf(stderr,
              "KMP_TEAMS_PROC_BIND should be one of spread, close, primary");
      exit(EXIT_FAILURE);
    }
  }
  return proc_bind;
}

int main(int argc, char **argv) {
  int i, nteams, max_nteams, factor;
  place_list_t **teams_places;
  place_list_t *place_list;
  omp_proc_bind_t teams_proc_bind;

  // Set a default for the max number of teams if it is not already set
  set_default_max_nteams();
  place_list = topology_alloc_openmp_places();
  max_nteams = get_max_nteams();
  // Further limit the number of teams twice the number of OMP_PLACES
  if (max_nteams > 2 * place_list->num_places)
    max_nteams = 2 * place_list->num_places;
  teams_places = (place_list_t **)malloc(sizeof(place_list_t *) * max_nteams);
  for (i = 0; i < max_nteams; ++i)
    teams_places[i] = NULL;
  teams_proc_bind = get_teams_proc_bind();

  // factor inversely controls the number of test cases.
  // the larger the factor, the more test cases will be performed.
  if (teams_proc_bind == omp_proc_bind_master) {
    factor = 2;
  } else {
    factor = 8;
  }

  for (nteams = 1; nteams <= max_nteams;
       nteams = nteams * factor / (factor - 1) + 1) {
    // Check the same value twice to make sure hot teams are ok
    int j;
    for (j = 0; j < 2; ++j) {
      // Gather the proc bind partitions from each team
      #pragma omp teams num_teams(nteams)
      teams_places[omp_get_team_num()] = topology_alloc_openmp_partition();

      // Check all the partitions with the parent partition
      proc_bind_check(teams_proc_bind, place_list, teams_places, nteams);

      // Free the proc bind partitions
      for (i = 0; i < nteams; ++i)
        topology_free_places(teams_places[i]);
    }
  }

  free(teams_places);
  topology_free_places(place_list);
  return EXIT_SUCCESS;
}
