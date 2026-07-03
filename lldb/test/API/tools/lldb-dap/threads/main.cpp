#include <cstdio>
#include <thread>

int state_var;

void thread() {
  state_var++; // break here
}

int main(int argc, char **argv) {
  std::thread t1(thread);
  t1.join();
  std::thread t2(thread);
  t2.join();

  printf("state_var is %d\n", state_var);
  return 0;
}
