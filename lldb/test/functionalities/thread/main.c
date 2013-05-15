#include <pthread.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex3 = PTHREAD_MUTEX_INITIALIZER;

void *
thread3 (void *input)
{
    pthread_mutex_unlock(&mutex2); // Set break point at this line.
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void *
thread2 (void *input)
{
    pthread_mutex_unlock(&mutex3);
    pthread_mutex_lock(&mutex2);
    pthread_mutex_unlock(&mutex2);

    return NULL;
}

void *
thread1 (void *input)
{
    pthread_t thread_2;
    pthread_create (&thread_2, NULL, thread2, NULL);

    pthread_mutex_lock(&mutex1);
    pthread_mutex_unlock(&mutex1);

    pthread_join(thread_2, NULL);

    return NULL;
}

int main ()
{
  pthread_t thread_1;
  pthread_t thread_3;

  pthread_mutex_lock (&mutex1);
  pthread_mutex_lock (&mutex2);
  pthread_mutex_lock (&mutex3);

  pthread_create (&thread_1, NULL, thread1, NULL);

  pthread_mutex_lock(&mutex3);
  pthread_create (&thread_3, NULL, thread3, NULL);

  pthread_join (thread_1, NULL);
  pthread_join (thread_3, NULL);

  return 0;

}
