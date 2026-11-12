#include "pseudo_barrier.h"
#include <thread>

pseudo_barrier_t g_barrier;

static int my_add(int a, int b) { // breakpoint
  return a + b;
}

static void do_test() {
  // Don't let either thread do anything until they're both ready.
  pseudo_barrier_init(g_barrier, 2);

  std::thread t1([] {
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);
    my_add(1, 2);
  });
  std::thread t2([] {
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);
    my_add(4, 5);
  });

  t1.join();
  t2.join();
}

int main(int argc, char const *argv[]) {
  do_test();
  return 0;
}
