// Layout approximation of MSVC STL std::valarray.

#include <stddef.h>

namespace std {
template <class T> class valarray {
public:
  valarray() : _Myptr(nullptr), _Mysize(0) {}
  valarray(T *ptr, size_t size) : _Myptr(ptr), _Mysize(size) {}
  T *_Myptr;
  size_t _Mysize;
};
} // namespace std

int main() {
  int va_vals[] = {1, 12, 123, 1234};
  std::valarray<int> va(va_vals, 4);
  std::valarray<int> va_empty;
  std::valarray<int> &va_ref = va;
  return 0; // break here
}
