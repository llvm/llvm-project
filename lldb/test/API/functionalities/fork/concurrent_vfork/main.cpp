#include <thread>
#include <unistd.h>
#include <iostream>
#include <vector>

int call_vfork() {
  printf("Before vfork\n");

  pid_t child_pid = vfork();

  if (child_pid == -1) {
      // Error handling
      perror("vfork");
      return 1;
  } else if (child_pid == 0) {
      // This code is executed by the child process
      printf("Child process\n");
      _exit(0); // Exit the child process
  } else {
      // This code is executed by the parent process
      printf("Parent process\n");
  }

  printf("After vfork\n");
  return 0;
}

void worker_thread() {
  call_vfork();
}

void create_threads(int num_threads) {
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(std::thread(worker_thread));
  }
  printf("Created %d threads, joining...\n", num_threads); // end_of_create_threads
  for (auto &thread: threads) {
    thread.join();
  }
}

int main() {
  int num_threads = 5; // break here
  create_threads(num_threads);
}
