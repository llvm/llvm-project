#include <thread>

void thread_func() {
  // Set a breakpoint here
}

int
main()
{
  // Set a breakpoint here
  std::thread stopped_thread(thread_func);
  stopped_thread.join();
  return 0;
}
