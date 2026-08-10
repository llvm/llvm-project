#include <chrono>
#include <thread>

void f3() {
  int m;
  m = 2; // thread 3
}

void f2() {
  int n;
  n = 1; // thread 2
  std::thread t3(f3);
  t3.join();
}

int main() { // main
  std::thread t2(f2);
  t2.join();
  return 0;
}
