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

// A barrier to synchronize all the threads.
std::atomic_int g_barrier1;

// A barrier to keep the threads from exiting until after the breakpoint has
// been passed.
std::atomic_int g_barrier2;

void *
break_thread_func (void *input)
{
    // Wait until all the threads are running
    pseudo_barrier_wait(g_barrier1);

    // Wait for the join thread to join
    usleep(50);

    // Do something
    g_test++;       // Set breakpoint here

    // Synchronize after the breakpoint
    pseudo_barrier_wait(g_barrier2);

    // Return
    return NULL;
}

void *
wait_thread_func (void *input)
{
    // Wait until the entire first group of threads is running
    pseudo_barrier_wait(g_barrier1);

    // Wait until the breakpoint has been passed
    pseudo_barrier_wait(g_barrier2);

    // Return
    return NULL;
}

void *
join_thread_func (void *input)
{
    pthread_t *thread_to_join = (pthread_t*)input;

    // Sync up with the rest of the threads.
    pseudo_barrier_wait(g_barrier1);

    // Join the other thread
    pthread_join(*thread_to_join, NULL);

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

    // The first barrier waits for the non-joining threads to start.
    // This thread will also participate in that barrier.
    // The idea here is to guarantee that the joining thread will be
    // last in the internal list maintained by the debugger.
    pseudo_barrier_init(g_barrier1, 5);

    // The second barrier keeps the waiting threads around until the breakpoint
    // has been passed.
    pseudo_barrier_init(g_barrier2, 4);

    // Create a thread to hit the breakpoint
    pthread_create (&thread_1, NULL, break_thread_func, NULL);

    // Create more threads to slow the debugger down during processing.
    pthread_create (&thread_2, NULL, wait_thread_func, NULL);
    pthread_create (&thread_3, NULL, wait_thread_func, NULL);
    pthread_create (&thread_4, NULL, wait_thread_func, NULL);

    // Create a thread to join the breakpoint thread
    pthread_create (&thread_5, NULL, join_thread_func, &thread_4);

    // Wait for the threads to finish
    pthread_join(thread_5, NULL);
    pthread_join(thread_4, NULL);
    pthread_join(thread_3, NULL);
    pthread_join(thread_2, NULL);
    pthread_join(thread_1, NULL);

    return 0;
}
