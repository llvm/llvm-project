#include "pseudo_barrier.h"
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <thread>
#include <vector>

pseudo_barrier_t barrier;

constexpr size_t nthreads = 5;
volatile bool wait_for_debugger_flag = true;

void break_here() {}

void tfunc() {
  pseudo_barrier_wait(barrier);

  break_here();
}

int main(int argc, char const *argv[]) {
  lldb_enable_attach();

  if (argc < 3)
    return 1;

  // Create a file to signal that this process has started up.
  std::ofstream(argv[1]).close();

  // And wait for it to attach.
  for (int i = 0; i < 100 && wait_for_debugger_flag; ++i)
    std::this_thread::sleep_for(std::chrono::seconds(1));

  // Fire up the threads and have them call break_here() simultaneously.
  pseudo_barrier_init(barrier, nthreads);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < nthreads; ++i)
    threads.emplace_back(tfunc);

  for (std::thread &t : threads)
    t.join();

  // Create the file to let the debugger know we're running.
  std::ofstream(argv[2]).close();

  return 0;
}
