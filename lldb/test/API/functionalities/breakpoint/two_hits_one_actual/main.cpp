#include <thread>
#include <chrono>

void usleep_helper(unsigned int usec) {
  // Break here in the helper
  std::this_thread::sleep_for(std::chrono::duration<unsigned int, std::milli>(usec));
}

void *background_thread(void *arg) {
    (void) arg;
    for (;;) {
        usleep_helper(2);
    }
}

int main(void) {
  unsigned int main_usec = 1;
  std::thread main_thread(background_thread, nullptr); // Set bkpt here to get started
  for (;;) {
    usleep_helper(main_usec);
  }
}
