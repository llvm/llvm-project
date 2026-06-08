#include <stddef.h>

namespace std {
inline namespace __LegacyLayout {
template <typename T> class vector {
public:
  typedef T *pointer;

  vector(pointer begin, size_t size) : __begin_(begin), __end_(begin + size) {}

private:
  pointer __begin_;
  pointer __end_;
  // __cap_ and __alloc_ aren't used, so they've been removed for simplicity.
};
} // namespace __LegacyLayout

inline namespace __PointerBasedLayout {
template <typename T> struct __vector_layout {
  T *__begin_;
  T *__end_;
};

template <typename T> class vector {
public:
  vector(T *begin, size_t size) : __layout_{begin, begin + size} {}

private:
  __vector_layout<T> __layout_;
};
} // namespace __PointerBasedLayout

inline namespace __SizeBasedLayout {
template <typename T> struct __vector_layout {
  T *__begin_;
  size_t __size_;
};

template <typename T> class vector {
public:
  vector(T *begin, size_t size) : __layout_{begin, size} {}

private:
  __vector_layout<T> __layout_;
};
} // namespace __SizeBasedLayout
} // namespace std

int main() {
  int a1[] = {10};
  int a2[] = {-10, -20};
  int a3[] = {56, 10, 87};

  std::__LegacyLayout::vector<int> legacy_layout0(a1, 0);
  std::__LegacyLayout::vector<int> legacy_layout1(a1, 1);
  std::__LegacyLayout::vector<int> legacy_layout2(a2, 2);
  std::__LegacyLayout::vector<int> legacy_layout3(a3, 3);

  std::__PointerBasedLayout::vector<int> pointer_based_layout0(a1, 0);
  std::__PointerBasedLayout::vector<int> pointer_based_layout1(a1, 1);
  std::__PointerBasedLayout::vector<int> pointer_based_layout2(a2, 2);
  std::__PointerBasedLayout::vector<int> pointer_based_layout3(a3, 3);

  std::__SizeBasedLayout::vector<int> size_based_layout0(a1, 0);
  std::__SizeBasedLayout::vector<int> size_based_layout1(a1, 1);
  std::__SizeBasedLayout::vector<int> size_based_layout2(a2, 2);
  std::__SizeBasedLayout::vector<int> size_based_layout3(a3, 3);

  return 0;
}
