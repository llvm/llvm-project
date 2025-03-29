#include <assert.h>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

pid_t g_pid = 0;
std::mutex g_child_pids_mutex;
std::vector<pid_t> g_child_pids;

const char *g_program = nullptr;
bool g_use_vfork = true;  // Use vfork by default.
bool g_call_exec = false; // Does not call exec by default.

int call_vfork(int index) {
  pid_t child_pid = 0;
  if (g_use_vfork) {
    child_pid = vfork();
  } else {
    child_pid = fork();
  }

  if (child_pid == -1) {
    // Error handling
    perror("vfork");
    return 1;
  } else if (child_pid == 0) {
    // This code is executed by the child process
    g_pid = getpid();
    printf("Child process: %d\n", g_pid);

    if (g_call_exec) {
      std::string child_exit_code = std::to_string(index + 10);
      execl(g_program, g_program, "--child", child_exit_code.c_str(), NULL);
    } else {
      _exit(index + 10);
    }
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

// Can be called in various ways:
// 1. [program]: use vfork and not call exec
// 2. [program] --fork: use fork and not call exec
// 3. [program] --fork --exec: use fork and call exec
// 4. [program] --exec: use vfork and call exec
// 5. [program] --child [exit_code]: child process
int main(int argc, char *argv[]) {
  g_pid = getpid();
  g_program = argv[0];

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--child") == 0) {
      assert(i + 1 < argc);
      int child_exit_code = std::stoi(argv[i + 1]);
      printf("Child process: %d, exiting with code %d\n", g_pid,
             child_exit_code);
      _exit(child_exit_code);
    } else if (strcmp(argv[i], "--fork") == 0)
      g_use_vfork = false;
    else if (strcmp(argv[i], "--exec") == 0)
      g_call_exec = true;
  }

  int num_threads = 5; // break here
  create_threads(num_threads);
  return 0;
}
