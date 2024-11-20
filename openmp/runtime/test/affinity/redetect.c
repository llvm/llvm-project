// RUN: %libomp-compile
// RUN: %libomp-run
// RUN: env KMP_AFFINITY=none %libomp-run
// REQUIRES: linux

// Check if forked child process resets affinity properly by restricting
// child's affinity to a subset of the parent and then checking it after
// a parallel region

#define _GNU_SOURCE
#include "libomp_test_affinity.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>

// Set the affinity mask of the calling thread to a proper subset of the
// original affinity mask, specifically, one processor less.
void set_subset_affinity(affinity_mask_t *mask) {
  int cpu;
  affinity_mask_t *original_mask = affinity_mask_alloc();
  affinity_mask_copy(original_mask, mask);
  // Find first processor to clear for subset mask
  for (cpu = 0; cpu <= AFFINITY_MAX_CPUS; ++cpu) {
    if (affinity_mask_isset(original_mask, cpu)) {
      affinity_mask_clr(mask, cpu);
      break;
    }
  }
  affinity_mask_free(original_mask);
  set_thread_affinity(mask);
}

int main(int argc, char **argv) {
  char buf[1024] = {0};
  char *other_buf;
  size_t n;
  int child_exit_status, exit_status;
  affinity_mask_t *mask = affinity_mask_alloc();
  get_thread_affinity(mask);
  n = affinity_mask_snprintf(buf, sizeof(buf), mask);
  printf("Orignal Mask: %s\n", buf);

  if (affinity_mask_count(mask) == 1) {
    printf("Only one processor in affinity mask, skipping test.\n");
    exit(EXIT_SUCCESS);
  }

  #pragma omp parallel
  {
    #pragma omp single
    printf("Hello! Thread %d executed single region in parent process\n",
           omp_get_thread_num());
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork()");
    exit(EXIT_FAILURE);
  }

  if (pid == 0) {
    // Let child set a new initial mask
    set_subset_affinity(mask);
    #pragma omp parallel
    {
      #pragma omp single
      printf("Hello! Thread %d executed single region in child process\n",
             omp_get_thread_num());
    }
    affinity_mask_t *new_mask = affinity_mask_alloc();
    get_thread_affinity(new_mask);
    if (!affinity_mask_equal(mask, new_mask)) {
      affinity_mask_snprintf(buf, sizeof(buf), mask);
      fprintf(stderr, "Original Mask = %s\n", buf);
      affinity_mask_snprintf(buf, sizeof(buf), new_mask);
      fprintf(stderr, "New Mask = %s\n", buf);
      affinity_mask_free(new_mask);
      fprintf(stderr, "Child affinity mask did not reset properly\n");
      exit(EXIT_FAILURE);
    }
    affinity_mask_free(new_mask);
    exit_status = EXIT_SUCCESS;
  } else {
    pid_t child_pid = pid;
    pid = wait(&child_exit_status);
    if (pid == -1) {
      perror("wait()");
      exit(EXIT_FAILURE);
    }
    if (WIFEXITED(child_exit_status)) {
      exit_status = WEXITSTATUS(child_exit_status);
    } else {
      exit_status = EXIT_FAILURE;
    }
  }

  affinity_mask_free(mask);
  return exit_status;
}
