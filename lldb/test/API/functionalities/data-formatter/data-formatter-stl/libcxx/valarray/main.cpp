#include <iostream>
#include <valarray>

int main() {

  std::valarray<int> va_int(4);
  std::cout << "break here";

  va_int[0] = 1;
  va_int[1] = 12;
  va_int[2] = 123;
  va_int[3] = 1234;

  std::valarray<double> va_double({1.0, 0.5, 0.25, 0.125});

  std::valarray<int> va({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::slice_array<int> sa = va[std::slice(1, 4, 2)];

  std::cout << "break here\n";
}
