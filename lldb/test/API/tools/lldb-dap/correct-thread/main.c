#include <pthread.h>
#include <stdio.h>

int state_var;

void *thread(void *in) {
  state_var++; // break here
  return NULL;
}

int main(int argc, char **argv) {
  pthread_t t1, t2;

  pthread_create(&t1, NULL, *thread, NULL);
  pthread_join(t1, NULL);
  pthread_create(&t2, NULL, *thread, NULL);
  pthread_join(t2, NULL);

  printf("state_var is %d\n", state_var);
  return 0;
}
