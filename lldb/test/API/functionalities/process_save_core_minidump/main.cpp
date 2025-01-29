#include <cassert>
#include <iostream>
#include <thread>
thread_local size_t lf = 42;

void g() { assert(false); }

void f() { g(); }

size_t h() {
  size_t sum = 0;
  for (size_t i = 0; i < 1000000; ++i)
    for (size_t j = 0; j < 1000000; ++j)
      if ((i * j) % 2 == 0) {
        sum += 1;
      }
  return sum;
}

int main() {
  std::thread t1(f);

  size_t x = h();

  t1.join();

  std::cout << "X is " << x << "\n";
  return 0;
}
