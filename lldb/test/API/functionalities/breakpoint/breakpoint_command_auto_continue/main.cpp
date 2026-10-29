#include <pthread.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

__attribute__((noinline)) void break_here(char *buf) {
  // Memory barrier to keep the function body non-empty.
  asm volatile("" ::: "memory");
}

void *test_thread(void *) {
  char buf[1024];
  for (int i = 0; i < 200; i++) {
    memset(buf, 0, sizeof(buf));
    break_here(buf);
    if (memcmp(buf, "OK", 2) != 0) {
      printf("FAILED at iteration %d\n", i);
      _exit(1);
    }
    usleep(50);
  }
  return nullptr;
}

int main() {
  pthread_t thread;
  if (pthread_create(&thread, nullptr, test_thread, nullptr) != 0) {
    perror("pthread_create");
    return 1;
  }
  pthread_join(thread, nullptr);
  printf("PASSED\n");
  return 0;
}
