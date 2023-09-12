#ifndef LIBOMP_TEST_AFFINITY_H
#define LIBOMP_TEST_AFFINITY_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct affinity_mask_t {
  size_t setsize;
  cpu_set_t *set;
} affinity_mask_t;

#ifdef __ve__
// VE's sched_getaffinity put garbage if the buffer is too big.  512 is
// a good number to make all tests run correctly.
#define AFFINITY_MAX_CPUS 512
#else
#define AFFINITY_MAX_CPUS (32 * 64)
#endif

// Operating system affinity mask API
static void affinity_mask_zero(affinity_mask_t *mask) {
  CPU_ZERO_S(mask->setsize, mask->set);
}

static affinity_mask_t *affinity_mask_alloc() {
  size_t setsize = CPU_ALLOC_SIZE(AFFINITY_MAX_CPUS);
  cpu_set_t *set = CPU_ALLOC(AFFINITY_MAX_CPUS);
  affinity_mask_t *retval = (affinity_mask_t *)malloc(sizeof(affinity_mask_t));
  retval->setsize = setsize;
  retval->set = set;
  affinity_mask_zero(retval);
  return retval;
}

static void affinity_mask_free(affinity_mask_t *mask) { CPU_FREE(mask->set); }

static void affinity_mask_copy(affinity_mask_t *dest,
                               const affinity_mask_t *src) {
  memcpy(dest->set, src->set, dest->setsize);
}

static void affinity_mask_set(affinity_mask_t *mask, int cpu) {
  CPU_SET_S(cpu, mask->setsize, mask->set);
}

static void affinity_mask_clr(affinity_mask_t *mask, int cpu) {
  CPU_CLR_S(cpu, mask->setsize, mask->set);
}

static int affinity_mask_isset(const affinity_mask_t *mask, int cpu) {
  return CPU_ISSET_S(cpu, mask->setsize, mask->set);
}

static int affinity_mask_count(const affinity_mask_t *mask) {
  return CPU_COUNT_S(mask->setsize, mask->set);
}

static int affinity_mask_equal(const affinity_mask_t *mask1,
                               const affinity_mask_t *mask2) {
  return CPU_EQUAL_S(mask1->setsize, mask1->set, mask2->set);
}

static void get_thread_affinity(affinity_mask_t *mask) {
  if (sched_getaffinity(0, mask->setsize, mask->set) != 0) {
    perror("sched_getaffinity()");
    exit(EXIT_FAILURE);
  }
}

static void set_thread_affinity(const affinity_mask_t *mask) {
  if (sched_setaffinity(0, mask->setsize, mask->set) != 0) {
    perror("sched_setaffinity()");
    exit(EXIT_FAILURE);
  }
}

static void affinity_update_snprintf_values(char **ptr, size_t *remaining,
                                            size_t n, size_t *retval) {
  if (n > *remaining && *remaining > 0) {
    *ptr += *remaining;
    *remaining = 0;
  } else {
    *ptr += n;
    *remaining -= n;
  }
  *retval += n;
}

static size_t affinity_mask_snprintf(char *buf, size_t bufsize,
                                     const affinity_mask_t *mask) {
  int cpu, need_comma, begin, end;
  size_t n;
  char *ptr = buf;
  size_t remaining = bufsize;
  size_t retval = 0;

  n = snprintf(ptr, remaining, "%c", '{');
  affinity_update_snprintf_values(&ptr, &remaining, n, &retval);

  need_comma = 0;
  for (cpu = 0; cpu < AFFINITY_MAX_CPUS; cpu++) {
    if (!affinity_mask_isset(mask, cpu))
      continue;
    if (need_comma) {
      n = snprintf(ptr, remaining, "%c", ',');
      affinity_update_snprintf_values(&ptr, &remaining, n, &retval);
    }
    begin = cpu;
    // Find end of range (inclusive end)
    for (end = begin + 1; end < AFFINITY_MAX_CPUS; ++end) {
      if (!affinity_mask_isset(mask, end))
        break;
    }
    end--;

    if (end - begin >= 2) {
      n = snprintf(ptr, remaining, "%d-%d", begin, end);
      affinity_update_snprintf_values(&ptr, &remaining, n, &retval);
    } else if (end - begin == 1) {
      n = snprintf(ptr, remaining, "%d,%d", begin, end);
      affinity_update_snprintf_values(&ptr, &remaining, n, &retval);
    } else if (end - begin == 0) {
      n = snprintf(ptr, remaining, "%d", begin);
      affinity_update_snprintf_values(&ptr, &remaining, n, &retval);
    }
    need_comma = 1;
    cpu = end;
  }
  n = snprintf(ptr, remaining, "%c", '}');
  affinity_update_snprintf_values(&ptr, &remaining, n, &retval);
  return retval;
}
#endif
