// Two real threads: main, which the test stops on, and a worker that parks on a
// mutex main holds so it stays in the thread list for the whole test. The OS
// plugin in this directory hides main (core 0) behind a virtual thread, which
// leaves the worker thread as the first entry of the user-facing thread list.

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

int g_watched = 0;

static std::mutex g_mutex;
static std::atomic<bool> g_worker_started(false);

static void worker() {
  g_worker_started = true;
  std::lock_guard<std::mutex> lock(g_mutex);
}

int main(int argc, char *argv[]) {
  std::unique_lock<std::mutex> lock(g_mutex);
  std::thread worker_thread(worker);
  while (!g_worker_started)
    std::this_thread::yield();

  g_watched = 1; // Break here
  g_watched = 2; // Second stop here

  lock.unlock();
  worker_thread.join();

  if (argc > 1) {
    if (FILE *marker = fopen(argv[1], "w")) {
      fputs("done\n", marker);
      fclose(marker);
    }
  }
  return 0;
}
