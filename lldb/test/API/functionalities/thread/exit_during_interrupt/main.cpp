#include <atomic>
#include <chrono>
#include <thread>

std::atomic<bool> done(false);

int main() {
  std::thread churner([] {
    while (!done)
      std::thread([] {}).join();
  });
  while (!done)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  churner.join();
  return 0;
}
