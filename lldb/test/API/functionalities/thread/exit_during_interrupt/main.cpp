#include <chrono>
#include <thread>

int main() {
  std::thread churner([] {
    while (true)
      std::thread([] {}).join();
  });
  while (true)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  churner.join();
  return 0;
}
