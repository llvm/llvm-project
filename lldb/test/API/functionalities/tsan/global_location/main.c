#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

long my_global_variable;  // global variable

void *f1(void *p) {
    my_global_variable = 42;
    return NULL;
}

void *f2(void *p) {
    my_global_variable = 43;
    return NULL;
}

int main (int argc, char const *argv[])
{
    pthread_t t1;
    pthread_create(&t1, NULL, f1, NULL);

    pthread_t t2;
    pthread_create(&t2, NULL, f2, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
