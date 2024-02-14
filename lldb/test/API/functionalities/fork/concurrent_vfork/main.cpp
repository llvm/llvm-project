#include <iostream>
#include <mutex>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

pid_t g_pid = 0;
std::mutex g_child_pids_mutex;
std::vector<pid_t> g_child_pids;

int call_vfork(int index) {
  pid_t child_pid = vfork();

  if (child_pid == -1) {
    // Error handling
    perror("vfork");
    return 1;
  } else if (child_pid == 0) {
    // This code is executed by the child process
    g_pid = getpid();
    printf("Child process: %d\n", g_pid);
    _exit(index + 10); // Exit the child process
  } else {
    // This code is executed by the parent process
    printf("[Parent] Forked process id: %d\n", child_pid);
  }
  return 0;
}

void wait_all_children_to_exit() {
  std::lock_guard<std::mutex> Lock(g_child_pids_mutex);
  for (pid_t child_pid : g_child_pids) {
    int child_status = 0;
    pid_t pid = waitpid(child_pid, &child_status, 0);
    if (child_status != 0) {
      int exit_code = WEXITSTATUS(child_status);
      if (exit_code > 15 || exit_code < 10) {
        printf("Error: child process exits with unexpected code %d\n",
               exit_code);
        _exit(1); // This will let our program know that some child processes
                  // didn't exist with an expected exit status.
      }
    }
    if (pid != child_pid)
      _exit(2); // This will let our program know it didn't succeed
  }
}

void create_threads(int num_threads) {
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(std::thread(call_vfork, i));
  }
  printf("Created %d threads, joining...\n",
         num_threads); // end_of_create_threads
  for (auto &thread : threads) {
    thread.join();
  }
  wait_all_children_to_exit();
}

int main() {
  g_pid = getpid();
  printf("Entering main() pid: %d\n", g_pid);

  int num_threads = 5; // break here
  create_threads(num_threads);
  printf("Exiting main() pid: %d\n", g_pid);
}
