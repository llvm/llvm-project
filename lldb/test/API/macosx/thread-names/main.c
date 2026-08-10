#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

int threads_up_and_running = 0;

void *
second_thread (void *in)
{
    pthread_setname_np ("second thread");
    while (1) 
        sleep (1);
    return NULL;
}

void *
third_thread (void *in)
{
    pthread_setname_np ("third thread");
    while (1) 
        sleep (1);
    return NULL;
}

int main ()
{
    pthread_setname_np ("main thread");
    pthread_t other_thread;
    pthread_create (&other_thread, NULL, second_thread, NULL);
    pthread_create (&other_thread, NULL, third_thread, NULL);

    threads_up_and_running = 1;

    while (1)
        sleep (1);
}
