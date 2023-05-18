#include <cstdint>
#include <mutex>
#include <thread>

std::mutex t1_mutex, t2_mutex;

struct test_data {
  uint32_t eax;
  uint32_t ebx;

  struct alignas(16) {
    uint64_t mantissa;
    uint16_t sign_exp;
  } st0;
};

void t_func(std::mutex &t_mutex, const test_data &t_data) {
  std::lock_guard<std::mutex> t_lock(t_mutex);

  asm volatile(
    "finit\t\n"
    "fldt %2\t\n"
    "int3\n\t"
    :
    : "a"(t_data.eax), "b"(t_data.ebx), "m"(t_data.st0)
    : "st"
  );
}

int main() {
  test_data t1_data = {
    .eax = 0x05060708,
    .ebx = 0x15161718,
    .st0 = {0x8070605040302010, 0x4000},
  };
  test_data t2_data = {
    .eax = 0x25262728,
    .ebx = 0x35363738,
    .st0 = {0x8171615141312111, 0xc000},
  };

  // block both threads from proceeding
  std::unique_lock<std::mutex> m1_lock(t1_mutex);
  std::unique_lock<std::mutex> m2_lock(t2_mutex);

  // start both threads
  std::thread t1(t_func, std::ref(t1_mutex), std::ref(t1_data));
  std::thread t2(t_func, std::ref(t2_mutex), std::ref(t2_data));

  // release lock on thread 1 to make it interrupt the program
  m1_lock.unlock();
  // wait for thread 1 to finish
  t1.join();

  // release lock on thread 2
  m2_lock.unlock();
  // wait for thread 2 to finish
  t2.join();

  return 0;
}
