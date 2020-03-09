#include <sys/time.h>  // work around module map issue with iOS sdk, <rdar://problem/35159346> 
#include <sys/select.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void *
select_thread (void *in)
{
    pthread_setname_np ("select thread");
    fd_set fdset;
    FD_SET (STDIN_FILENO, &fdset);
    while (1)
        select (2, &fdset, NULL, NULL, NULL);
    return NULL;
}

void stopper ()
{
    while (1)
        sleep(1); // break here
}

int main ()
{
    pthread_setname_np ("main thread");
    pthread_t other_thread;
    pthread_create (&other_thread, NULL, select_thread, NULL);
    sleep (1);
    stopper();
}
