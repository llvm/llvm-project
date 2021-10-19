#ifndef LIBOMP_TEST_TOPOLOGY_H
#define LIBOMP_TEST_TOPOLOGY_H

#include "libomp_test_affinity.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <ctype.h>
#include <omp.h>
#include <stdarg.h>

typedef enum topology_obj_type_t {
  TOPOLOGY_OBJ_THREAD,
  TOPOLOGY_OBJ_CORE,
  TOPOLOGY_OBJ_SOCKET,
  TOPOLOGY_OBJ_MAX
} topology_obj_type_t;

typedef struct place_list_t {
  int num_places;
  int current_place;
  int *place_nums;
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
  int *place_nums;
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
  place_nums = (int *)malloc(sizeof(int) * num_unique);
  for (i = 0; i < num_unique; ++i)
    place_nums[i] = i;
  places->num_places = num_unique;
  places->masks = masks;
  places->place_nums = place_nums;
  places->current_place = -1;
  return places;
}

static place_list_t *topology_alloc_openmp_places() {
  int place, i;
  int num_places = omp_get_num_places();
  place_list_t *places = (place_list_t *)malloc(sizeof(place_list_t));
  affinity_mask_t **masks =
      (affinity_mask_t **)malloc(sizeof(affinity_mask_t *) * num_places);
  int *place_nums = (int *)malloc(sizeof(int) * num_places);
  for (place = 0; place < num_places; ++place) {
    int num_procs = omp_get_place_num_procs(place);
    int *ids = (int *)malloc(sizeof(int) * num_procs);
    omp_get_place_proc_ids(place, ids);
    affinity_mask_t *mask = affinity_mask_alloc();
    for (i = 0; i < num_procs; ++i)
      affinity_mask_set(mask, ids[i]);
    masks[place] = mask;
    place_nums[place] = place;
  }
  places->num_places = num_places;
  places->place_nums = place_nums;
  places->masks = masks;
  places->current_place = omp_get_place_num();
  return places;
}

static place_list_t *topology_alloc_openmp_partition() {
  int p, i;
  int num_places = omp_get_partition_num_places();
  place_list_t *places = (place_list_t *)malloc(sizeof(place_list_t));
  int *place_nums = (int *)malloc(sizeof(int) * num_places);
  affinity_mask_t **masks =
      (affinity_mask_t **)malloc(sizeof(affinity_mask_t *) * num_places);
  omp_get_partition_place_nums(place_nums);
  for (p = 0; p < num_places; ++p) {
    int place = place_nums[p];
    int num_procs = omp_get_place_num_procs(place);
    int *ids = (int *)malloc(sizeof(int) * num_procs);
    if (num_procs == 0) {
      fprintf(stderr, "place %d has 0 procs?\n", place);
      exit(EXIT_FAILURE);
    }
    omp_get_place_proc_ids(place, ids);
    affinity_mask_t *mask = affinity_mask_alloc();
    for (i = 0; i < num_procs; ++i)
      affinity_mask_set(mask, ids[i]);
    if (affinity_mask_count(mask) == 0) {
      fprintf(stderr, "place %d has 0 procs set?\n", place);
      exit(EXIT_FAILURE);
    }
    masks[p] = mask;
  }
  places->num_places = num_places;
  places->place_nums = place_nums;
  places->masks = masks;
  places->current_place = omp_get_place_num();
  return places;
}

// Free the array of masks from one of: topology_alloc_type_masks()
// or topology_alloc_openmp_masks()
static void topology_free_places(place_list_t *places) {
  int i;
  for (i = 0; i < places->num_places; ++i)
    affinity_mask_free(places->masks[i]);
  free(places->masks);
  free(places->place_nums);
  free(places);
}

static void topology_print_places(const place_list_t *p) {
  int i;
  char buf[1024];
  for (i = 0; i < p->num_places; ++i) {
    affinity_mask_snprintf(buf, sizeof(buf), p->masks[i]);
    printf("Place %d: %s\n", p->place_nums[i], buf);
  }
}

// Print out an error message, possibly with two problem place lists,
// and then exit with failure
static void proc_bind_die(omp_proc_bind_t proc_bind, int T, int P,
                          const char *format, ...) {
  va_list args;
  va_start(args, format);
  const char *pb;
  switch (proc_bind) {
  case omp_proc_bind_false:
    pb = "False";
    break;
  case omp_proc_bind_true:
    pb = "True";
    break;
  case omp_proc_bind_master:
    pb = "Master (Primary)";
    break;
  case omp_proc_bind_close:
    pb = "Close";
    break;
  case omp_proc_bind_spread:
    pb = "Spread";
    break;
  default:
    pb = "(Unknown Proc Bind Type)";
    break;
  }
  if (proc_bind == omp_proc_bind_spread || proc_bind == omp_proc_bind_close) {
    if (T <= P) {
      fprintf(stderr, "%s : (T(%d) <= P(%d)) : ", pb, T, P);
    } else {
      fprintf(stderr, "%s : (T(%d) > P(%d)) : ", pb, T, P);
    }
  } else {
    fprintf(stderr, "%s : T = %d, P = %d : ", pb, T, P);
  }
  vfprintf(stderr, format, args);
  va_end(args);

  exit(EXIT_FAILURE);
}

// Return 1 on failure, 0 on success.
static void proc_bind_check(omp_proc_bind_t proc_bind,
                            const place_list_t *parent, place_list_t **children,
                            int nchildren) {
  place_list_t *partition;
  int T, i, j, place, low, high, first, last, count, current_place, num_places;
  const int *place_nums;
  int P = parent->num_places;

  // Find the correct T (there could be null entries in children)
  place_list_t **partitions =
      (place_list_t **)malloc(sizeof(place_list_t *) * nchildren);
  T = 0;
  for (i = 0; i < nchildren; ++i)
    if (children[i])
      partitions[T++] = children[i];
  // Only able to check spread, close, master (primary)
  if (proc_bind != omp_proc_bind_spread && proc_bind != omp_proc_bind_close &&
      proc_bind != omp_proc_bind_master)
    proc_bind_die(proc_bind, T, P, NULL, NULL,
                  "Cannot check this proc bind type\n");

  if (proc_bind == omp_proc_bind_spread) {
    if (T <= P) {
      // Run through each subpartition
      for (i = 0; i < T; ++i) {
        partition = partitions[i];
        place_nums = partition->place_nums;
        num_places = partition->num_places;
        current_place = partition->current_place;
        // Correct count?
        low = P / T;
        high = P / T + (P % T ? 1 : 0);
        if (num_places != low && num_places != high) {
          proc_bind_die(proc_bind, T, P,
                        "Incorrect number of places for thread %d: %d. "
                        "Expecting between %d and %d\n",
                        i, num_places, low, high);
        }
        // Consecutive places?
        for (j = 1; j < num_places; ++j) {
          if (place_nums[j] != (place_nums[j - 1] + 1) % P) {
            proc_bind_die(proc_bind, T, P,
                          "Not consecutive places: %d, %d in partition\n",
                          place_nums[j - 1], place_nums[j]);
          }
        }
        first = place_nums[0];
        last = place_nums[num_places - 1];
        // Primary thread executes on place of the parent thread?
        if (i == 0) {
          if (current_place != parent->current_place) {
            proc_bind_die(
                proc_bind, T, P,
                "Primary thread not on same place (%d) as parent thread (%d)\n",
                current_place, parent->current_place);
          }
        } else {
          // Thread's current place is first place within it's partition?
          if (current_place != first) {
            proc_bind_die(proc_bind, T, P,
                          "Thread's current place (%d) is not the first place "
                          "in its partition [%d, %d]\n",
                          current_place, first, last);
          }
        }
        // Partitions don't have intersections?
        int f1 = first;
        int l1 = last;
        for (j = 0; j < i; ++j) {
          int f2 = partitions[j]->place_nums[0];
          int l2 = partitions[j]->place_nums[partitions[j]->num_places - 1];
          if (f1 > l1 && f2 > l2) {
            proc_bind_die(proc_bind, T, P,
                          "partitions intersect. [%d, %d] and [%d, %d]\n", f1,
                          l1, f2, l2);
          }
          if (f1 > l1 && f2 <= l2)
            if (f1 < l2 || l1 > f2) {
              proc_bind_die(proc_bind, T, P,
                            "partitions intersect. [%d, %d] and [%d, %d]\n", f1,
                            l1, f2, l2);
            }
          if (f1 <= l1 && f2 > l2)
            if (f2 < l1 || l2 > f1) {
              proc_bind_die(proc_bind, T, P,
                            "partitions intersect. [%d, %d] and [%d, %d]\n", f1,
                            l1, f2, l2);
            }
          if (f1 <= l1 && f2 <= l2)
            if (!(f2 > l1 || l2 < f1)) {
              proc_bind_die(proc_bind, T, P,
                            "partitions intersect. [%d, %d] and [%d, %d]\n", f1,
                            l1, f2, l2);
            }
        }
      }
    } else {
      // T > P
      // Each partition has only one place?
      for (i = 0; i < T; ++i) {
        if (partitions[i]->num_places != 1) {
          proc_bind_die(
              proc_bind, T, P,
              "Incorrect number of places for thread %d: %d. Expecting 1\n", i,
              partitions[i]->num_places);
        }
      }
      // Correct number of consecutive threads per partition?
      low = T / P;
      high = T / P + (T % P ? 1 : 0);
      for (i = 1, count = 1; i < T; ++i) {
        if (partitions[i]->place_nums[0] == partitions[i - 1]->place_nums[0]) {
          count++;
          if (count > high) {
            proc_bind_die(
                proc_bind, T, P,
                "Too many threads have place %d for their partition\n",
                partitions[i]->place_nums[0]);
          }
        } else {
          if (count < low) {
            proc_bind_die(
                proc_bind, T, P,
                "Not enough threads have place %d for their partition\n",
                partitions[i]->place_nums[0]);
          }
          count = 1;
        }
      }
      // Primary thread executes on place of the parent thread?
      current_place = partitions[0]->place_nums[0];
      if (parent->current_place != -1 &&
          current_place != parent->current_place) {
        proc_bind_die(
            proc_bind, T, P,
            "Primary thread not on same place (%d) as parent thread (%d)\n",
            current_place, parent->current_place);
      }
    }
  } else if (proc_bind == omp_proc_bind_close ||
             proc_bind == omp_proc_bind_master) {
    // Check that each subpartition is the same as the parent
    for (i = 0; i < T; ++i) {
      partition = partitions[i];
      place_nums = partition->place_nums;
      num_places = partition->num_places;
      current_place = partition->current_place;
      if (parent->num_places != num_places) {
        proc_bind_die(proc_bind, T, P,
                      "Number of places in subpartition (%d) does not match "
                      "parent (%d)\n",
                      num_places, parent->num_places);
      }
      for (j = 0; j < num_places; ++j) {
        if (parent->place_nums[j] != place_nums[j]) {
          proc_bind_die(proc_bind, T, P,
                        "Subpartition place (%d) does not match "
                        "parent partition place (%d)\n",
                        place_nums[j], parent->place_nums[j]);
        }
      }
    }
    // Find index into place_nums of current place for parent
    for (j = 0; j < parent->num_places; ++j)
      if (parent->place_nums[j] == parent->current_place)
        break;
    if (proc_bind == omp_proc_bind_close) {
      if (T <= P) {
        // close T <= P
        // check place assignment for each thread
        for (i = 0; i < T; ++i) {
          partition = partitions[i];
          current_place = partition->current_place;
          if (current_place != parent->place_nums[j]) {
            proc_bind_die(
                proc_bind, T, P,
                "Thread %d's current place (%d) is incorrect. expected %d\n", i,
                current_place, parent->place_nums[j]);
          }
          j = (j + 1) % parent->num_places;
        }
      } else {
        // close T > P
        // check place assignment for each thread
        low = T / P;
        high = T / P + (T % P ? 1 : 0);
        count = 1;
        if (partitions[0]->current_place != parent->current_place) {
          proc_bind_die(
              proc_bind, T, P,
              "Primary thread's place (%d) is not parent thread's place (%d)\n",
              partitions[0]->current_place, parent->current_place);
        }
        for (i = 1; i < T; ++i) {
          current_place = partitions[i]->current_place;
          if (current_place == parent->place_nums[j]) {
            count++;
            if (count > high) {
              proc_bind_die(
                  proc_bind, T, P,
                  "Too many threads have place %d for their current place\n",
                  current_place);
            }
          } else {
            if (count < low) {
              proc_bind_die(
                  proc_bind, T, P,
                  "Not enough threads have place %d for their current place\n",
                  parent->place_nums[j]);
            }
            j = (j + 1) % parent->num_places;
            if (current_place != parent->place_nums[j]) {
              proc_bind_die(
                  proc_bind, T, P,
                  "Thread %d's place (%d) is not corret. Expected %d\n", i,
                  partitions[i]->current_place, parent->place_nums[j]);
            }
            count = 1;
          }
        }
      }
    } else {
      // proc_bind_primary
      // Every thread should be assigned to the primary thread's place
      for (i = 0; i < T; ++i) {
        if (partitions[i]->current_place != parent->current_place) {
          proc_bind_die(
              proc_bind, T, P,
              "Thread %d's place (%d) is not the primary thread's place (%d)\n",
              i, partitions[i]->current_place, parent->current_place);
        }
      }
    }
  }

  // Check that each partition's current place is within the partition
  for (i = 0; i < T; ++i) {
    current_place = partitions[i]->current_place;
    num_places = partitions[i]->num_places;
    first = partitions[i]->place_nums[0];
    last = partitions[i]->place_nums[num_places - 1];
    for (j = 0; j < num_places; ++j)
      if (partitions[i]->place_nums[j] == current_place)
        break;
    if (j == num_places) {
      proc_bind_die(proc_bind, T, P,
                    "Thread %d's current place (%d) is not within its "
                    "partition [%d, %d]\n",
                    i, current_place, first, last);
    }
  }

  free(partitions);
}

#endif
