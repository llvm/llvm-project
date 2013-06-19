//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which one thread will exit
// while a breakpoint is being handled in another thread.  This may not always
// happen because it's possible that the exiting thread will exit before the
// breakpoint is hit.  The test case should be flexible enough to treat that
// as success.

#include <pthread.h>
#include <unistd.h>
#include <atomic>

volatile int g_test = 0;

// Note that although hogging the CPU while waiting for a variable to change
// would be terrible in production code, it's great for testing since it
// avoids a lot of messy context switching to get multiple threads synchronized.
#define do_nothing()  

#define pseudo_barrier_wait(bar) \
    --bar;                       \
    while (bar > 0)              \
        do_nothing();

#define pseudo_barrier_init(bar, count) (bar = count)

// A barrier to synchronize all the threads except the one that will exit.
std::atomic_int g_barrier1;

// A barrier to synchronize all the threads including the one that will exit.
std::atomic_int g_barrier2;

// A barrier to keep the first group of threads from exiting until after the
// breakpoint has been passed.
std::atomic_int g_barrier3;

void *
break_thread_func (void *input)
{
    // Wait until the entire first group of threads is running
    pseudo_barrier_wait(g_barrier1);

    // Wait for the exiting thread to start
    pseudo_barrier_wait(g_barrier2);

    // Do something
    g_test++;       // Set breakpoint here

    // Synchronize after the breakpoint
    pseudo_barrier_wait(g_barrier3);

    // Return
    return NULL;
}

void *
wait_thread_func (void *input)
{
    // Wait until the entire first group of threads is running
    pseudo_barrier_wait(g_barrier1);

    // Wait for the exiting thread to start
    pseudo_barrier_wait(g_barrier2);

    // Wait until the breakpoint has been passed
    pseudo_barrier_wait(g_barrier3);

    // Return
    return NULL;
}

void *
exit_thread_func (void *input)
{
    // Sync up with the rest of the threads.
    pseudo_barrier_wait(g_barrier2);

    // Try to make sure this thread doesn't exit until the breakpoint is hit.
    usleep(1);

    // Return
    return NULL;
}

int main ()
{
    pthread_t thread_1;
    pthread_t thread_2;
    pthread_t thread_3;
    pthread_t thread_4;
    pthread_t thread_5;

    // The first barrier waits for the non-exiting threads to start.
    // This thread will also participate in that barrier.
    // The idea here is to guarantee that the exiting thread will be
    // last in the internal list maintained by the debugger.
    pseudo_barrier_init(g_barrier1, 5);

    // The second break synchronyizes thread exection with the breakpoint.
    pseudo_barrier_init(g_barrier2, 5);

    // The third barrier keeps the waiting threads around until the breakpoint
    // has been passed.
    pseudo_barrier_init(g_barrier3, 4);

    // Create a thread to hit the breakpoint
    pthread_create (&thread_1, NULL, break_thread_func, NULL);

    // Create more threads to slow the debugger down during processing.
    pthread_create (&thread_2, NULL, wait_thread_func, NULL);
    pthread_create (&thread_3, NULL, wait_thread_func, NULL);
    pthread_create (&thread_4, NULL, wait_thread_func, NULL);

    // Wait for all these threads to get started.
    pseudo_barrier_wait(g_barrier1);

    // Create a thread to exit during the breakpoint
    pthread_create (&thread_5, NULL, exit_thread_func, NULL);

    // Wait for the threads to finish
    pthread_join(thread_5, NULL);
    pthread_join(thread_4, NULL);
    pthread_join(thread_3, NULL);
    pthread_join(thread_2, NULL);
    pthread_join(thread_1, NULL);

    return 0;
}
