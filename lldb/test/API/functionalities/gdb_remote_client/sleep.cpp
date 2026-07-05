#include <thread>

int main() {
  std::this_thread::sleep_for(std::chrono::minutes(1));
  return 0;
}
