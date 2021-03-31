#include "pseudo_barrier.h"
#include "thread.h"
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

pseudo_barrier_t barrier;

static void sigusr1_handler(int signo) {
  char buf[100];
  std::snprintf(buf, sizeof(buf),
                "received SIGUSR1 on thread id: %" PRIx64 "\n",
                get_thread_id());
  write(STDOUT_FILENO, buf, strlen(buf));
}

static void thread_func() {
  pseudo_barrier_wait(barrier);
  std::this_thread::sleep_for(std::chrono::minutes(1));
}

int main(int argc, char **argv) {
  int num = atoi(argv[1]);

  pseudo_barrier_init(barrier, num + 1);

  signal(SIGUSR1, sigusr1_handler);

  std::vector<std::thread> threads;
  for(int i = 0; i < num; ++i)
    threads.emplace_back(thread_func);

  pseudo_barrier_wait(barrier);

  std::puts("@started");

  for (std::thread &thread : threads)
    thread.join();
  return 0;
}
