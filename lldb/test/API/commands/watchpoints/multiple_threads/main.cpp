#include "pseudo_barrier.h"
#include <cstdio>
#include <thread>

volatile uint32_t g_val = 0;
pseudo_barrier_t g_barrier, g_barrier2;

void thread_func() {
  pseudo_barrier_wait(g_barrier);
  pseudo_barrier_wait(g_barrier2);
  printf("%s starting...\n", __FUNCTION__);
  for (uint32_t i = 0; i < 10; ++i)
    g_val = i;
}

int main(int argc, char const *argv[]) {
  printf("Before running the thread\n");
  pseudo_barrier_init(g_barrier, 2);
  pseudo_barrier_init(g_barrier2, 2);
  std::thread thread(thread_func);

  printf("After launching the thread\n");
  pseudo_barrier_wait(g_barrier);

  printf("After running the thread\n");
  pseudo_barrier_wait(g_barrier2);

  thread.join();

  return 0;
}
