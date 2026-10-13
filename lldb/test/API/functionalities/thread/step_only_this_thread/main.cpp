#include <atomic>
#include <chrono>
#include <thread>

static std::atomic<bool> g_bystander_started{false};
static std::atomic<bool> g_done{false};
static std::atomic<bool> g_worker_ran{false};

static void bystander() {
  g_bystander_started = true;
  while (!g_done)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

static void worker() {
  g_worker_ran = true; // Break in the worker.
}

static void start_worker(std::thread &worker_thread) {
  worker_thread = std::thread(worker);
  // If the worker gets to run, give it time to reach its breakpoint before
  // this function returns.
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

// Starts a thread and waits for it. The test also calls this in an expression.
int run_and_join_thread() {
  std::thread thread([] {});
  thread.join();
  return 42;
}

int main() {
  std::thread bystander_thread(bystander);
  while (!g_bystander_started)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

  std::thread worker_thread;
  start_worker(worker_thread); // Start the worker.
  worker_thread.join();        // Stepped to here.

  g_done = true;
  bystander_thread.join();
  return run_and_join_thread() == 42 ? 0 : 1;
}
