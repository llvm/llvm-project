#include <condition_variable>
#include <mutex>
#include <thread>

std::mutex mux;
std::condition_variable cv;
bool ready = false;

static int my_add(int a, int b) { // breakpoint 1
  std::unique_lock<std::mutex> lk(mux);
  cv.wait(lk, [] { return ready; });
  return a + b;
}

int main(int argc, char const *argv[]) {
  std::thread t1(my_add, 1, 2);
  std::thread t2(my_add, 4, 5);

  {
    std::lock_guard<std::mutex> lk(mux);
    ready = true;
    cv.notify_all();
  }
  t1.join();
  t2.join();
  return 0;
}
