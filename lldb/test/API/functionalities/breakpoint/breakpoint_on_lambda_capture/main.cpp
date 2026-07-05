#include <chrono>
#include <cstdio>
#include <thread>

struct Foo {
  bool enable = true;
  uint32_t offset = 0;

  void usleep_helper(uint32_t usec) {
    [this, &usec] {
      puts("Break here in the helper");
      std::this_thread::sleep_for(
          std::chrono::duration<unsigned int, std::milli>(offset + usec));
    }();
  }
};

void *background_thread(void *) {
  Foo f;
  for (;;) {
    f.usleep_helper(2);
  }
}

int main() {
  std::puts("First break");
  std::thread main_thread(background_thread, nullptr);
  Foo f;
  for (;;) {
    f.usleep_helper(1);
  }
}
