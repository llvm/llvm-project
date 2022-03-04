
#include <iostream>



int next() {
  static int i = 0;
  std::cout << "incrementing " << i << std::endl;
  return ++i;
}

int main() {
  int i = 0;
  while (i < 5)
    i = next();
  return 0;
}
