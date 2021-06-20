// RUN: %libomp-compile && env LIBOMP_NUM_HIDDEN_HELPER_THREADS=0 OMP_PROC_BIND=close OMP_PLACES=cores KMP_AFFINITY=verbose %libomp-run 8 1 4
// REQUIRED: linux
//
// This test pthread_creates 8 root threads before any OpenMP
// runtime entry is ever called. We have all the root threads
// register with the runtime by calling omp_set_num_threads(),
// but this does not initialize their affinity. The fourth root thread
// then calls a parallel region and we make sure its affinity
// is correct. We also make sure all the other root threads are
// free-floating since they have not called into a parallel region.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include "libomp_test_affinity.h"

volatile int entry_flag = 0;
volatile int flag = 0;
volatile int num_roots_arrived = 0;
int num_roots;
int spawner = 0;
pthread_mutex_t lock;
int register_workers = 0; // boolean
affinity_mask_t *full_mask;

int __kmpc_global_thread_num(void*);

int get_os_thread_id() {
  return (int)syscall(SYS_gettid);
}

int place_and_affinity_match() {
  int i, max_cpu;
  char buf[512];
  affinity_mask_t *mask = affinity_mask_alloc();
  int place = omp_get_place_num();
  int num_procs = omp_get_place_num_procs(place);
  int *ids = (int*)malloc(sizeof(int) * num_procs);
  omp_get_place_proc_ids(place, ids);
  get_thread_affinity(mask);
  affinity_mask_snprintf(buf, sizeof(buf), mask);
  printf("Primary Thread Place: %d\n", place);
  printf("Primary Thread mask: %s\n", buf);

  for (i = 0; i < num_procs; ++i) {
    int cpu = ids[i];
    if (!affinity_mask_isset(mask, cpu))
      return 0;
  }

  max_cpu = AFFINITY_MAX_CPUS;
  for (i = 0; i < max_cpu; ++i) {
    int cpu = i;
    if (affinity_mask_isset(mask, cpu)) {
      int j, found = 0;
      for (j = 0; j < num_procs; ++j) {
        if (ids[j] == cpu) {
          found = 1;
          break;
        }
      }
      if (!found)
        return 0;
    }
  }

  affinity_mask_free(mask);
  free(ids);
  return 1;
}

void* thread_func(void *arg) {
  int place, nplaces;
  int root_id = *((int*)arg);
  int pid = getpid();
  int tid = get_os_thread_id();

  // Order how the root threads are assigned a gtid in the runtime
  // i.e., root_id = gtid
  while (1) {
    int v = entry_flag;
    if (v == root_id)
      break;
  }

  // If main root thread
  if (root_id == spawner) {
    printf("Initial application thread (pid=%d, tid=%d, spawner=%d) reached thread_func (will call OpenMP)\n", pid, tid, spawner);
    omp_set_num_threads(4);
    #pragma omp atomic
    entry_flag++;
    // Wait for the workers to signal their arrival before #pragma omp parallel
    while (num_roots_arrived < num_roots - 1) {}
    // This will trigger the output for KMP_AFFINITY in this case
    #pragma omp parallel
    {
      int gtid = __kmpc_global_thread_num(NULL);
      #pragma omp single
      {
        printf("Exactly %d threads in the #pragma omp parallel\n",
               omp_get_num_threads());
      }
      #pragma omp critical
      {
        printf("OpenMP thread %d: gtid=%d\n", omp_get_thread_num(), gtid);
      }
    }
    flag = 1;
    if (!place_and_affinity_match()) {
      fprintf(stderr, "error: place and affinity mask do not match for primary thread\n");
      exit (EXIT_FAILURE);
    }

  } else { // If worker root thread
    // Worker root threads, register with OpenMP through omp_set_num_threads()
    // if designated to, signal their arrival and then wait for the main root
    // thread to signal them to exit.
    printf("New root pthread (pid=%d, tid=%d) reached thread_func\n", pid, tid);
    if (register_workers)
      omp_set_num_threads(4);
    #pragma omp atomic
    entry_flag++;

    pthread_mutex_lock(&lock);
    num_roots_arrived++;
    pthread_mutex_unlock(&lock);
    while (flag == 0) {}

    // Main check whether root threads' mask is equal to the
    // initial affinity mask
    affinity_mask_t *mask = affinity_mask_alloc();
    get_thread_affinity(mask);
    if (!affinity_mask_equal(mask, full_mask)) {
      char buf[1024];
      printf("root thread %d mask: ", root_id);
      affinity_mask_snprintf(buf, sizeof(buf), mask);
      printf("initial affinity mask: %s\n", buf);
      fprintf(stderr, "error: root thread %d affinity mask not equal"
                      " to initial full mask\n", root_id);
      affinity_mask_free(mask);
      exit(EXIT_FAILURE);
    }
    affinity_mask_free(mask);
  }
  return NULL;
}

int main(int argc, char** argv) {
  int i;
  if (argc != 3 && argc != 4) {
    fprintf(stderr, "usage: %s <num_roots> <register_workers_bool> [<spawn_root_number>]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initialize pthread mutex
  pthread_mutex_init(&lock, NULL);

  // Get initial full mask
  full_mask = affinity_mask_alloc();
  get_thread_affinity(full_mask);

  // Get the number of root pthreads to create and allocate resources for them
  num_roots = atoi(argv[1]);
  pthread_t *roots = (pthread_t*)malloc(sizeof(pthread_t) * num_roots);
  int *root_ids = (int*)malloc(sizeof(int) * num_roots);

  // Get the flag indicating whether to have root pthreads call omp_set_num_threads() or not
  register_workers = atoi(argv[2]);

  if (argc == 4)
    spawner = atoi(argv[3]);

  // Spawn worker root threads
  for (i = 1; i < num_roots; ++i) {
    *(root_ids + i) = i;
    pthread_create(roots + i, NULL, thread_func, root_ids + i);
  }
  // Have main root thread (root 0) go into thread_func
  *root_ids = 0;
  thread_func(root_ids);

  // Cleanup all resources
  for (i = 1; i < num_roots; ++i) {
    void *status;
    pthread_join(roots[i], &status);
  }
  free(roots);
  free(root_ids);
  pthread_mutex_destroy(&lock);
  return EXIT_SUCCESS;
}
