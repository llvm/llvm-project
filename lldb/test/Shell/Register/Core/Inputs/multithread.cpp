// This program is used to generate a core dump for testing multithreading
// support.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

std::atomic_int start_counter{3};

void pseudobarrier_wait() {
  start_counter--;
  while (start_counter > 0);
}

void fcommon(uint32_t a, uint32_t b, uint32_t c, uint32_t d, double fa, double fb, double fc, double fd, bool segv) {
  if (segv) {
    int *foo = nullptr;
    *foo = 0;
  }
  while (1);
}

void f1() {
  volatile uint32_t a = 0x01010101;
  volatile uint32_t b = 0x02020202;
  volatile uint32_t c = 0x03030303;
  volatile uint32_t d = 0x04040404;
  volatile double fa = 2.0;
  volatile double fb = 4.0;
  volatile double fc = 8.0;
  volatile double fd = 16.0;

  pseudobarrier_wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fcommon(a, b, c, d, fa, fb, fc, fd, true);
}

void f2() {
  volatile uint32_t a = 0x11111111;
  volatile uint32_t b = 0x12121212;
  volatile uint32_t c = 0x13131313;
  volatile uint32_t d = 0x14141414;
  volatile double fa = 3.0;
  volatile double fb = 6.0;
  volatile double fc = 9.0;
  volatile double fd = 12.0;

  pseudobarrier_wait();
  fcommon(a, b, c, d, fa, fb, fc, fd, false);
}

void f3() {
  volatile uint32_t a = 0x21212121;
  volatile uint32_t b = 0x22222222;
  volatile uint32_t c = 0x23232323;
  volatile uint32_t d = 0x24242424;
  volatile double fa = 5.0;
  volatile double fb = 10.0;
  volatile double fc = 15.0;
  volatile double fd = 20.0;

  pseudobarrier_wait();
  fcommon(a, b, c, d, fa, fb, fc, fd, false);
}

int main() {
  std::thread t1{f1};
  std::thread t2{f2};
  std::thread t3{f3};

  t3.join();
  t2.join();
  t1.join();

  return 0;
}
