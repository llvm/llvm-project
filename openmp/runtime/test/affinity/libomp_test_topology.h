#ifndef LIBOMP_TEST_TOPOLOGY_H
#define LIBOMP_TEST_TOPOLOGY_H

#include "libomp_test_affinity.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <ctype.h>
#include <omp.h>

typedef enum topology_obj_type_t {
  TOPOLOGY_OBJ_THREAD,
  TOPOLOGY_OBJ_CORE,
  TOPOLOGY_OBJ_SOCKET,
  TOPOLOGY_OBJ_MAX
} topology_obj_type_t;

typedef struct place_list_t {
  int num_places;
  affinity_mask_t **masks;
} place_list_t;

// Return the first character in file 'f' that is not a whitespace character
// including newlines and carriage returns
static int get_first_nonspace_from_file(FILE *f) {
  int c;
  do {
    c = fgetc(f);
  } while (c != EOF && (isspace(c) || c == '\n' || c == '\r'));
  return c;
}

// Read an integer from file 'f' into 'number'
// Return 1 on successful read of integer,
//        0 on unsuccessful read of integer,
//        EOF on end of file.
static int get_integer_from_file(FILE *f, int *number) {
  int n;
  n = fscanf(f, "%d", number);
  if (feof(f))
    return EOF;
  if (n != 1)
    return 0;
  return 1;
}

// Read a siblings list file from Linux /sys/devices/system/cpu/cpu?/topology/*
static affinity_mask_t *topology_get_mask_from_file(const char *filename) {
  int status = EXIT_SUCCESS;
  FILE *f = fopen(filename, "r");
  if (!f) {
    perror(filename);
    exit(EXIT_FAILURE);
  }
  affinity_mask_t *mask = affinity_mask_alloc();
  while (1) {
    int c, i, n, lower, upper;
    // Read the first integer
    n = get_integer_from_file(f, &lower);
    if (n == EOF) {
      break;
    } else if (n == 0) {
      fprintf(stderr, "syntax error: expected integer\n");
      status = EXIT_FAILURE;
      break;
    }

    // Now either a , or -
    c = get_first_nonspace_from_file(f);
    if (c == EOF || c == ',') {
      affinity_mask_set(mask, lower);
      if (c == EOF)
        break;
    } else if (c == '-') {
      n = get_integer_from_file(f, &upper);
      if (n == EOF || n == 0) {
        fprintf(stderr, "syntax error: expected integer\n");
        status = EXIT_FAILURE;
        break;
      }
      for (i = lower; i <= upper; ++i)
        affinity_mask_set(mask, i);
      c = get_first_nonspace_from_file(f);
      if (c == EOF) {
        break;
      } else if (c == ',') {
        continue;
      } else {
        fprintf(stderr, "syntax error: unexpected character: '%c (%d)'\n", c,
                c);
        status = EXIT_FAILURE;
        break;
      }
    } else {
      fprintf(stderr, "syntax error: unexpected character: '%c (%d)'\n", c, c);
      status = EXIT_FAILURE;
      break;
    }
  }
  fclose(f);
  if (status == EXIT_FAILURE) {
    affinity_mask_free(mask);
    mask = NULL;
  }
  return mask;
}

static int topology_get_num_cpus() {
  char buf[1024];
  // Count the number of cpus
  int cpu = 0;
  while (1) {
    snprintf(buf, sizeof(buf), "/sys/devices/system/cpu/cpu%d", cpu);
    DIR *dir = opendir(buf);
    if (dir) {
      closedir(dir);
      cpu++;
    } else {
      break;
    }
  }
  if (cpu == 0)
    cpu = 1;
  return cpu;
}

// Return whether the current thread has access to all logical processors
static int topology_using_full_mask() {
  int cpu;
  int has_all = 1;
  int num_cpus = topology_get_num_cpus();
  affinity_mask_t *mask = affinity_mask_alloc();
  get_thread_affinity(mask);
  for (cpu = 0; cpu < num_cpus; ++cpu) {
    if (!affinity_mask_isset(mask, cpu)) {
      has_all = 0;
      break;
    }
  }
  affinity_mask_free(mask);
  return has_all;
}

// Return array of masks representing OMP_PLACES keyword (e.g., sockets, cores,
// threads)
static place_list_t *topology_alloc_type_places(topology_obj_type_t type) {
  char buf[1024];
  int i, cpu, num_places, num_unique;
  int num_cpus = topology_get_num_cpus();
  place_list_t *places = (place_list_t *)malloc(sizeof(place_list_t));
  affinity_mask_t **masks =
      (affinity_mask_t **)malloc(sizeof(affinity_mask_t *) * num_cpus);
  num_unique = 0;
  for (cpu = 0; cpu < num_cpus; ++cpu) {
    affinity_mask_t *mask;
    if (type == TOPOLOGY_OBJ_CORE) {
      snprintf(buf, sizeof(buf),
               "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
               cpu);
      mask = topology_get_mask_from_file(buf);
    } else if (type == TOPOLOGY_OBJ_SOCKET) {
      snprintf(buf, sizeof(buf),
               "/sys/devices/system/cpu/cpu%d/topology/core_siblings_list",
               cpu);
      mask = topology_get_mask_from_file(buf);
    } else if (type == TOPOLOGY_OBJ_THREAD) {
      mask = affinity_mask_alloc();
      affinity_mask_set(mask, cpu);
    } else {
      fprintf(stderr, "Unknown topology type (%d)\n", (int)type);
      exit(EXIT_FAILURE);
    }
    // Check for unique topology objects above the thread level
    if (type != TOPOLOGY_OBJ_THREAD) {
      for (i = 0; i < num_unique; ++i) {
        if (affinity_mask_equal(masks[i], mask)) {
          affinity_mask_free(mask);
          mask = NULL;
          break;
        }
      }
    }
    if (mask)
      masks[num_unique++] = mask;
  }
  places->num_places = num_unique;
  places->masks = masks;
  return places;
}

static place_list_t *topology_alloc_openmp_places() {
  int place, i;
  int num_places = omp_get_num_places();
  place_list_t *places = (place_list_t *)malloc(sizeof(place_list_t));
  affinity_mask_t **masks =
      (affinity_mask_t **)malloc(sizeof(affinity_mask_t *) * num_places);
  for (place = 0; place < num_places; ++place) {
    int num_procs = omp_get_place_num_procs(place);
    int *ids = (int *)malloc(sizeof(int) * num_procs);
    omp_get_place_proc_ids(place, ids);
    affinity_mask_t *mask = affinity_mask_alloc();
    for (i = 0; i < num_procs; ++i)
      affinity_mask_set(mask, ids[i]);
    masks[place] = mask;
  }
  places->num_places = num_places;
  places->masks = masks;
  return places;
}

// Free the array of masks from one of: topology_alloc_type_masks()
// or topology_alloc_openmp_masks()
static void topology_free_places(place_list_t *places) {
  int i;
  for (i = 0; i < places->num_places; ++i)
    affinity_mask_free(places->masks[i]);
  free(places->masks);
  free(places);
}

static void topology_print_places(const place_list_t *p) {
  int i;
  char buf[1024];
  for (i = 0; i < p->num_places; ++i) {
    affinity_mask_snprintf(buf, sizeof(buf), p->masks[i]);
    printf("Place %d: %s\n", i, buf);
  }
}

#endif
