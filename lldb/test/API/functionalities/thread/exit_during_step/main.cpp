// This test is intended to create a situation in which one thread will exit
// while the debugger is stepping in another thread.

#include "pseudo_barrier.h"
#include <thread>

#define do_nothing()

// A barrier to synchronize thread start.
pseudo_barrier_t g_barrier;

volatile int g_thread_exited = 0;

volatile int g_test = 0;

void *
step_thread_func ()
{
    // Wait until both threads are started.
    pseudo_barrier_wait(g_barrier);

    g_test = 0;         // Set breakpoint here

    while (!g_thread_exited)
        g_test++;

    // One more time to provide a continue point
    g_test++;           // Continue from here

    // Return
    return NULL;
}

void *
exit_thread_func ()
{
    // Wait until both threads are started.
    pseudo_barrier_wait(g_barrier);

    // Wait until the other thread is stepping.
    while (g_test == 0)
      do_nothing();

    // Return
    return NULL;
}

int main ()
{
    // Synchronize thread start so that doesn't happen during stepping.
    pseudo_barrier_init(g_barrier, 2);

    // Create a thread to hit the breakpoint.
    std::thread thread_1(step_thread_func);

    // Create a thread to exit while we're stepping.
    std::thread thread_2(exit_thread_func);

    // Wait for the exit thread to finish.
    thread_2.join();

    // Let the stepping thread know the other thread is gone.
    g_thread_exited = 1;

    // Wait for the stepping thread to finish.
    thread_1.join();

    return 0;
}
