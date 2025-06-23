#include <functional>
#include <iostream>

void greet() {
  // BREAK HERE
  std::cout << "Hello\n";
}

int main() {
  std::function<void()> func{greet};
  func();
  return 0;
}
