#include <functional>

int g = 0;
void greet() {
  g++; // BREAK HERE
}

int main() {
  std::function<void()> func{greet};
  func();
  return 0;
}
