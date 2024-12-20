#include <iostream>

int sum(int a, int b) {
  return a + b; // Find the line number of function sum here.
}

int main() {

  int a = 2;
  int b = 3;
  int c = sum(a, b);

  std::cout << "c is " << c << std::endl;
  return 0;
}
