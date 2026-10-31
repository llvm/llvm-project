#include <pthread.h>
#include <stdlib.h>

void *f1(void *p) { return NULL; }

int main (int argc, char const *argv[])
{
    pthread_t t1;
    pthread_create(&t1, NULL, f1, NULL);

    return 0;
}
