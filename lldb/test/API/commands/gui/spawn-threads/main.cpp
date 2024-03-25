#include <iostream>
#include <thread>
#include <vector>

#include "pseudo_barrier.h"

pseudo_barrier_t barrier_inside;

void thread_func() { pseudo_barrier_wait(barrier_inside); }

void test_thread() {
  std::vector<std::thread> thrs;
  for (int i = 0; i < 5; i++)
    thrs.push_back(std::thread(thread_func)); // break here

  pseudo_barrier_wait(barrier_inside); // break before join
  for (auto &t : thrs)
    t.join();
}

int main() {
  pseudo_barrier_init(barrier_inside, 6);
  test_thread();
  return 0;
}
