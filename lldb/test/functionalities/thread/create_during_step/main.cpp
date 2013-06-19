//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which one thread will be
// created while a the debugger is stepping in another thread.

#include <pthread.h>
#include <atomic>

// Note that although hogging the CPU while waiting for a variable to change
// would be terrible in production code, it's great for testing since it
// avoids a lot of messy context switching to get multiple threads synchronized.
#define do_nothing()

#define pseudo_barrier_wait(bar) \
    --bar;                       \
    while (bar > 0)              \
        do_nothing();

#define pseudo_barrier_init(bar, count) (bar = count)

std::atomic_int g_barrier;

volatile int g_thread_created = 0;
volatile int g_test = 0;

void *
step_thread_func (void *input)
{
    g_test = 0;         // Set breakpoint here

    while (!g_thread_created)
        g_test++;

    // One more time to provide a continue point
    g_test++;           // Continue from here

    // Return
    return NULL;
}

void *
create_thread_func (void *input)
{
    pthread_t *step_thread = (pthread_t*)input;

    // Wait until the main thread knows this thread is started.
    pseudo_barrier_wait(g_barrier);

    // Wait until the other thread is done.
    pthread_join(*step_thread, NULL);

    // Return
    return NULL;
}

int main ()
{
    pthread_t thread_1;
    pthread_t thread_2;

    // Use a simple count to simulate a barrier.
    pseudo_barrier_init(g_barrier, 2);

    // Create a thread to hit the breakpoint.
    pthread_create (&thread_1, NULL, step_thread_func, NULL);

    // Wait until the step thread is stepping
    while (g_test < 1)
        do_nothing();

    // Create a thread to exit while we're stepping.
    pthread_create (&thread_2, NULL, create_thread_func, &thread_1);

    // Wait until that thread is started
    pseudo_barrier_wait(g_barrier);

    // Let the stepping thread know the other thread is there
    g_thread_created = 1;

    // Wait for the threads to finish.
    pthread_join(thread_2, NULL);
    pthread_join(thread_1, NULL);

    return 0;
}
