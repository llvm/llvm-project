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

  std::valarray<int> va({10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
  std::slice_array<int> sa = va[std::slice(1, 4, 2)];
  std::gslice_array<int> ga = va[std::gslice(
      3, std::valarray<std::size_t>(3, 1), std::valarray<std::size_t>(1, 1))];
  std::mask_array<int> ma = va[std::valarray<bool>{false, true, true}];
  std::indirect_array<int> ia = va[std::valarray<size_t>{3, 6, 9}];

  std::cout << "break here\n";
}
