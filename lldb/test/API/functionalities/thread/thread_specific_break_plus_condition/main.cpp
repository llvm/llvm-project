#include <chrono>
#include <thread>
#include <vector>

void thread_function(int my_value) {
  int counter = 0;
  while (counter < 20) {
    counter++; // Break here in thread body.
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

int main() {
  std::vector<std::thread> threads;

  for (int i = 0; i < 10; i++)
    threads.push_back(std::thread(thread_function, threads.size() + 1));

  for (std::thread &t : threads)
    t.join();

  return 0;
}
