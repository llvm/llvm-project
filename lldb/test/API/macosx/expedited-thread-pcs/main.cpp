#include <dlfcn.h>
#include <stdio.h>
#include <thread>
#include <unistd.h>

void f1() {
  while (1)
    sleep(1);
}
void f2() {
  while (1)
    sleep(1);
}
void f3() {
  while (1)
    sleep(1);
}

int main() {
  std::thread t1{f1};
  std::thread t2{f2};
  std::thread t3{f3};

  puts("break here");

  void *handle = dlopen("libfoo.dylib", RTLD_LAZY);
  int (*foo_ptr)() = (int (*)())dlsym(handle, "foo");
  int c = foo_ptr();

  // clang-format off
  // multiple function calls on a single source line so 'step'
  // and 'next' need to do multiple steps of work.
  puts("1"); puts("2"); puts("3"); puts("4"); puts("5");
  puts("6"); puts("7"); puts("8"); puts("9"); puts("10");
  puts("11"); puts("12"); puts("13"); puts("14"); puts("15");
  puts("16"); puts("17"); puts("18"); puts("19"); puts("20");
  puts("21"); puts("22"); puts("23"); puts("24"); puts("24");
  // clang-format on
  puts("one");
  puts("two");
  puts("three");
  puts("four");
  puts("five");
  puts("six");
  puts("seven");
  puts("eight");
  puts("nine");
  puts("ten");
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  c++;
  return c;
}
