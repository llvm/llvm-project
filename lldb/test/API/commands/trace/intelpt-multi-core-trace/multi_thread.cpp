#include <thread>
#include <cstring>
#include <unistd.h>
using namespace std;

bool done = false;
void foo() {
  int x = 0;
  for (int i = 0; i < 10000; i++)
    x++;
  sleep(1);
  for (int i = 0; i < 10000; i++)
    x++;
  done = true;
}

void bar() {
  int y = 0;
  while (!done) {
    y++;
  }
  printf("bar %d\n", y);
}

int main() {
  std::thread first(foo);
  std::thread second(bar);
  first.join();
  second.join();

  printf("complete\n");
  return 0;

}
