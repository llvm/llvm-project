#include "pseudo_barrier.h"
#include "thread.h"
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

pseudo_barrier_t barrier;
std::mutex print_mutex;
std::atomic<bool> can_work = ATOMIC_VAR_INIT(false);
thread_local volatile sig_atomic_t can_exit_now = false;

static void sigint_handler(int signo) {}

static void sigusr1_handler(int signo) {
  std::lock_guard<std::mutex> lock{print_mutex};
  std::printf("received SIGUSR1 on thread id: %" PRIx64 "\n", get_thread_id());
  can_exit_now = true;
}

static void thread_func() {
  // this ensures that all threads start before we stop
  pseudo_barrier_wait(barrier);

  // wait till the main thread indicates that we can go
  // (note: using a mutex here causes hang on FreeBSD when another thread
  // is suspended)
  while (!can_work.load())
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // the mutex guarantees that two writes don't get interspersed
  {
    std::lock_guard<std::mutex> lock{print_mutex};
    std::printf("thread %" PRIx64 " running\n", get_thread_id());
  }

  // give other threads a fair chance to run
  for (int i = 0; i < 5; ++i) {
    std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (can_exit_now)
      return;
  }

  // if we didn't get signaled, terminate the program explicitly.
  _exit(0);
}

int main(int argc, char **argv) {
  int num = atoi(argv[1]);

  pseudo_barrier_init(barrier, num + 1);

  signal(SIGINT, sigint_handler);
  signal(SIGUSR1, sigusr1_handler);

  std::vector<std::thread> threads;
  for (int i = 0; i < num; ++i)
    threads.emplace_back(thread_func);

  // use the barrier to make sure all threads start before we stop
  pseudo_barrier_wait(barrier);
  std::raise(SIGINT);

  // allow the threads to work
  can_work.store(true);

  for (std::thread &thread : threads)
    thread.join();
  return 0;
}
