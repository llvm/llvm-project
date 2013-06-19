#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

volatile int g_thread_2_continuing = 0;

void *
thread_1_func (void *input)
{
    // Waiting to be released by the debugger.
    while (!g_thread_2_continuing) // The debugger will change this value
    {
        usleep(1);
    }

    // Return
    return NULL;  // Set third breakpoint here
}

void *
thread_2_func (void *input)
{
    // Waiting to be released by the debugger.
    int child_thread_continue = 0;
    while (!child_thread_continue) // The debugger will change this value
    {
        usleep(1);  // Set second breakpoint here
    }

    // Release thread 1
    g_thread_2_continuing = 1;

    // Return
    return NULL;
}

int main(int argc, char const *argv[])
{
    pthread_t thread_1;
    pthread_t thread_2;

    // Create a new thread
    pthread_create (&thread_1, NULL, thread_1_func, NULL);

    // Waiting to be attached by the debugger.
    int main_thread_continue = 0;
    while (!main_thread_continue) // The debugger will change this value
    {
        usleep(1);  // Set first breakpoint here
    }

    // Create another new thread
    pthread_create (&thread_2, NULL, thread_2_func, NULL);

    // Wait for the threads to finish.
    pthread_join(thread_1, NULL);
    pthread_join(thread_2, NULL);

    printf("Exiting now\n");
}
