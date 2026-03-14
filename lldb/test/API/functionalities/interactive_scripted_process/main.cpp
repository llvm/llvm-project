#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

void spawn_thread(int index) {
  std::string name = "I'm thread " + std::to_string(index) + " !";
  bool done = false;
  std::string state = "Started execution!";
  while (true) {
    if (done) // also break here
      break;
  }

  state = "Stopped execution!";
}

int main() {
  constexpr size_t num_threads = 10;
  std::vector<std::thread> threads;

  for (size_t i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(spawn_thread, i));
  }

  std::cout << "Spawned " << threads.size() << " threads!" << std::endl; // Break here

  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }

  return 0;
}
