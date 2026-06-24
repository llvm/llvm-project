/**
 * Reproduction: steal-after-finish race via proxy task OOO completion
 *
 * The race: __kmpc_proxy_task_completed_ooo re-enqueues a proxy task into a
 * deque and THEN decrements td_incomplete_child_tasks (ICC) to 0. If all
 * threads mark finished (unfinished→0) before picking up the proxy, the
 * primary deactivates the task team with a task still in a deque.
 *
 * Key: worker deques must be pre-allocated (otherwise __kmpc_give_task
 * falls back to the primary's deque, and the primary picks it up).
 *
 * Requires libomp built with -DLIBOMP_REPRO_DELAY (delays in
 * execute_tasks_template + barrier spin loop to widen race window).
 */
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>

#define NUM_THREADS  4
#define NUM_TRIALS   500

static omp_event_handle_t g_event;
static atomic_int g_event_ready;

static void *fulfiller_fn(void *arg) {
    while (!atomic_load_explicit(&g_event_ready, memory_order_acquire))
        ;
    /* Wait for barrier entry + first inner loop so fulfiller fires
       during the 500us delay window in execute_tasks_template. */
    usleep(1000);
    omp_fulfill_event(g_event);
    return NULL;
}

static void crash_handler(int sig) {
    const char *msg = "\n*** BUG REPRODUCED ***\n"
        "  Steal-after-finish: orphaned proxy task in deactivated task team\n\n";
    write(STDERR_FILENO, msg, strlen(msg));
    _exit(1);
}

int main(int argc, char *argv[]) {
    int trials = NUM_TRIALS;
    if (argc > 1)
        trials = atoi(argv[1]);

    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGABRT, crash_handler);

    omp_set_num_threads(NUM_THREADS);

    printf("steal-after-finish race reproducer\n");
    printf("  Threads: %d, Trials: %d\n", NUM_THREADS, trials);
    printf("  Requires: libomp built with -DLIBOMP_REPRO_DELAY\n\n");

    for (int trial = 0; trial < trials; trial++) {
        atomic_store_explicit(&g_event_ready, 0, memory_order_relaxed);

        pthread_t fulfiller;
        pthread_create(&fulfiller, NULL, fulfiller_fn, NULL);

        #pragma omp parallel num_threads(NUM_THREADS)
        {
            int tid = omp_get_thread_num();

            /* Phase 1: Every thread creates a dummy task to force deque
               allocation for all threads IN THE SAME task team that the
               implicit barrier will use. NO explicit barrier here — an
               explicit barrier toggles th_task_state, causing the implicit
               barrier to use a DIFFERENT task team whose deques are NOT
               allocated, so the proxy ends up in thread 0's own deque. */
            volatile int dummy = 0;
            #pragma omp task firstprivate(dummy)
            {
                dummy = 1;  /* trivial work */
            }

            /* Phase 2: Thread 0 creates the detachable task.
               Same task team as the dummy tasks → all worker deques exist →
               __kmpc_give_task can enqueue proxy to a WORKER's deque. */
            if (tid == 0) {
                omp_event_handle_t evt;
                #pragma omp task detach(evt)
                {
                    g_event = evt;
                    atomic_store_explicit(&g_event_ready, 1,
                                          memory_order_release);
                    /* Body returns WITHOUT fulfilling → proxy task.
                       External pthread fulfills via OOO path:
                       1. Re-enqueues proxy to a worker's deque
                       2. Decrements ICC → 0
                       With delays in libomp:
                       - 500us in execute_tasks_template (all threads)
                       - 1000us in barrier spin loop (workers only)
                       Primary deactivates while proxy sits in worker's deque. */
                }
            }
            /* Implicit barrier at end of parallel region.
               The detachable task is executed during this barrier's
               task-execution phase. */
        }

        pthread_join(fulfiller, NULL);

        if ((trial + 1) % 50 == 0)
            printf("  [%d/%d] trials...\n", trial + 1, trials);
    }

    printf("\nCompleted %d trials.\n", trials);
    return 0;
}
