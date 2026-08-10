#include <stdlib.h>
#include <thread>
#include <unistd.h>
#include <vector>

void *sleep_worker(void *in) {
  sleep(30);
  sleep(30);
  return nullptr;
}

void *crash_worker(void *in) {
  sleep(1);
  volatile int *p = nullptr; // break here
  return (void *)*p;
}

int main() {
  std::vector<std::thread> threads;
  threads.push_back(std::move(std::thread(crash_worker, nullptr)));
  for (int i = 0; i < 15; i++)
    threads.push_back(std::move(std::thread(sleep_worker, nullptr)));
  sleep(10);
}
