#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

char *pointer;

void *f1(void *p) {
    pointer[0] = 'x'; // thread1 line
    return NULL;
}

void *f2(void *p) {
    pointer[0] = 'y'; // thread2 line
    return NULL;
}

int main (int argc, char const *argv[])
{
    pointer = (char *)malloc(10); // malloc line

    pthread_t t1, t2;
    pthread_create(&t1, NULL, f1, NULL);
    pthread_create(&t2, NULL, f2, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
