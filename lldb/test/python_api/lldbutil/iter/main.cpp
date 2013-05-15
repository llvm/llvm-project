//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C includes
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

pthread_t g_thread_1 = NULL;
pthread_t g_thread_2 = NULL;
pthread_t g_thread_3 = NULL;

typedef enum {
    eGet,
    eAssign,
    eClearBits
} MaskAction;

uint32_t mask_access (MaskAction action, uint32_t mask = 0);

uint32_t
mask_access (MaskAction action, uint32_t mask)
{
    static pthread_mutex_t g_mask_mutex = PTHREAD_MUTEX_INITIALIZER;
    static uint32_t g_mask = 0;
    ::pthread_mutex_lock (&g_mask_mutex);
    switch (action)
    {
    case eGet:
        break;

    case eAssign:
        g_mask |= mask;
        break;

    case eClearBits:
        g_mask &= ~mask;
        break;
    }
    uint32_t new_mask = g_mask;
    ::pthread_mutex_unlock (&g_mask_mutex);
    return new_mask;
}

void *
thread_func (void *arg)
{
    uint32_t thread_index = *((uint32_t *)arg);
    uint32_t thread_mask = (1u << (thread_index));
    printf ("%s (thread index = %u) startng...\n", __FUNCTION__, thread_index);

    while (mask_access(eGet) & thread_mask)
    {
        // random micro second sleep from zero to 3 seconds
        int usec = ::rand() % 3000000;
        printf ("%s (thread = %u) doing a usleep (%d)...\n", __FUNCTION__, thread_index, usec);
        ::usleep (usec);
        printf ("%s (thread = %u) after usleep ...\n", __FUNCTION__, thread_index); // Set break point at this line.
    }
    printf ("%s (thread index = %u) exiting...\n", __FUNCTION__, thread_index);
    return NULL;
}


int main (int argc, char const *argv[])
{
    int err;
    void *thread_result = NULL;
    uint32_t thread_index_1 = 1;
    uint32_t thread_index_2 = 2;
    uint32_t thread_index_3 = 3;
    uint32_t thread_mask_1 = (1u << thread_index_1);
    uint32_t thread_mask_2 = (1u << thread_index_2);
    uint32_t thread_mask_3 = (1u << thread_index_3);

    // Make a mask that will keep all threads alive
    mask_access (eAssign, thread_mask_1 | thread_mask_2 | thread_mask_3); // And that line.

    // Create 3 threads
    err = ::pthread_create (&g_thread_1, NULL, thread_func, &thread_index_1);
    err = ::pthread_create (&g_thread_2, NULL, thread_func, &thread_index_2);
    err = ::pthread_create (&g_thread_3, NULL, thread_func, &thread_index_3);

    char line[64];
    while (mask_access(eGet) != 0)
    {
        printf ("Enter thread index to kill or ENTER for all:\n");
        fflush (stdout);
        // Kill threads by index, or ENTER for all threads

        if (fgets (line, sizeof(line), stdin))
        {
            if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            {
                printf ("Exiting all threads...\n");
                break;
            }
            int32_t index = strtoul (line, NULL, 0);
            switch (index)
            {
                case 1: mask_access (eClearBits, thread_mask_1); break;
                case 2: mask_access (eClearBits, thread_mask_2); break;
                case 3: mask_access (eClearBits, thread_mask_3); break;
            }
            continue;
        }

        break;
    }

    // Clear all thread bits to they all exit
    mask_access (eClearBits, UINT32_MAX);

    // Join all of our threads
    err = ::pthread_join (g_thread_1, &thread_result);
    err = ::pthread_join (g_thread_2, &thread_result);
    err = ::pthread_join (g_thread_3, &thread_result);

    return 0;
}
