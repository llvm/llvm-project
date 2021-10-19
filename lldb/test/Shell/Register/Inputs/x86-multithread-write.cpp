#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>

std::mutex t1_mutex, t2_mutex;

struct test_data {
  uint32_t eax;
  uint32_t ebx;

  struct alignas(16) {
    uint8_t data[10];
  } st0;
};

constexpr test_data filler = {
  .eax = 0xffffffff,
  .ebx = 0xffffffff,
  .st0 = {{0x1f, 0x2f, 0x3f, 0x4f, 0x5f, 0x6f, 0x7f, 0x8f, 0x80, 0x40}},
};

void t_func(std::mutex &t_mutex) {
  std::lock_guard<std::mutex> t_lock(t_mutex);
  test_data out = filler;

  asm volatile(
    "finit\t\n"
    "fldt %2\t\n"
    "int3\n\t"
    "fstpt %2\t\n"
    : "+a"(out.eax), "+b"(out.ebx)
    : "m"(out.st0)
    : "memory", "st"
  );

  printf("eax = 0x%08" PRIx32 "\n", out.eax);
  printf("ebx = 0x%08" PRIx32 "\n", out.ebx);
  printf("st0 = { ");
  for (int i = 0; i < sizeof(out.st0.data); ++i)
    printf("0x%02" PRIx8 " ", out.st0.data[i]);
  printf("}\n");
}

int main() {
  // block both threads from proceeding
  std::unique_lock<std::mutex> m1_lock(t1_mutex);
  std::unique_lock<std::mutex> m2_lock(t2_mutex);

  // start both threads
  std::thread t1(t_func, std::ref(t1_mutex));
  std::thread t2(t_func, std::ref(t2_mutex));

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
