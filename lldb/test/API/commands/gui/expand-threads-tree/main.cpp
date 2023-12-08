#include "pseudo_barrier.h"
#include <thread>


pseudo_barrier_t barrier_before;
pseudo_barrier_t barrier_after;

void break_here() {}

void thread_func() {
    pseudo_barrier_wait(barrier_before);
    break_here();
    pseudo_barrier_wait(barrier_after);
}

int main() {
  pseudo_barrier_init(barrier_before, 2);
  pseudo_barrier_init(barrier_after, 2);
  std::thread thread(thread_func);
  pseudo_barrier_wait(barrier_before);
  pseudo_barrier_wait(barrier_after);
  thread.join();
  return 0;
}
