
#include <iostream>

int my_next() {
  static int i = 0;
  std::cout << "incrementing " << i << std::endl;
  return ++i;
}

int main() {
  int i = 0;
  while (i < 5)
    i = my_next();
  return 0;
}
