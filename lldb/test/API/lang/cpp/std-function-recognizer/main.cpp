#include <functional>

int foo(int x, int y) {
  return x * y; // break here
}

int main(int argc, char *argv[]) {
  std::function<int(int, int)> fn = foo;
  return fn(argc, 1);
}
