#include <thread>

void t_func() {
  asm volatile(
    "int3\n\t"
  );
}

int main() {
  std::thread t(t_func);
  t.join();
  return 0;
}
