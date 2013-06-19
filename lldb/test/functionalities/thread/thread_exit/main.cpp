//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test verifies the correct handling of child thread exits.

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

std::atomic_int g_barrier1;
std::atomic_int g_barrier2;
std::atomic_int g_barrier3;

void *
thread1 (void *input)
{
    // Synchronize with the main thread.
    pseudo_barrier_wait(g_barrier1);

    // Synchronize with the main thread and thread2.
    pseudo_barrier_wait(g_barrier2);

    // Return
    return NULL;                                      // Set second breakpoint here
}

void *
thread2 (void *input)
{
    // Synchronize with thread1 and the main thread.
    pseudo_barrier_wait(g_barrier2);

    // Synchronize with the main thread.
    pseudo_barrier_wait(g_barrier3);

    // Return
    return NULL;
}

int main ()
{
    pthread_t thread_1;
    pthread_t thread_2;
    pthread_t thread_3;

    pseudo_barrier_init(g_barrier1, 2);
    pseudo_barrier_init(g_barrier2, 3);
    pseudo_barrier_init(g_barrier3, 2);

    // Create a thread.
    pthread_create (&thread_1, NULL, thread1, NULL);

    // Wait for thread1 to start.
    pseudo_barrier_wait(g_barrier1);

    // Create another thread.
    pthread_create (&thread_2, NULL, thread2, NULL);  // Set first breakpoint here

    // Wait for thread2 to start.
    pseudo_barrier_wait(g_barrier2);

    // Wait for the first thread to finish
    pthread_join(thread_1, NULL);

    // Synchronize with the remaining thread
    pseudo_barrier_wait(g_barrier3);                  // Set third breakpoint here

    // Wait for the second thread to finish
    pthread_join(thread_2, NULL);

    return 0;                                         // Set fourth breakpoint here
}
