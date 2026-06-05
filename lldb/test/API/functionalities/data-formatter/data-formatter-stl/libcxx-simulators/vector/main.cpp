#include <stddef.h>

#define LLDB_TEST_VECTOR_WITHOUT_LAYOUT_DATA_MEMBER 0
#define LLDB_TEST_VECTOR_WITH_POINTER_LAYOUT 1
#define LLDB_TEST_VECTOR_WITH_SIZE_LAYOUT 2
#define LLDB_TEST_VECTOR_WITH_LAYOUT_MISSING_DATA_MEMBERS 3

#ifndef LLDB_TEST_CASE
#error LLDB_TEST_CASE must be defined as an integer
#endif

namespace std {
namespace __lldb {

#if LLDB_TEST_CASE == LLDB_TEST_VECTOR_WITHOUT_LAYOUT_DATA_MEMBER
template <typename T> class vector {
public:
  typedef T *pointer;

  vector(pointer begin, size_t size)
      : __begin_(begin), __end_(begin + size) {}

private:
  pointer __begin_;
  pointer __end_;
  // __cap_ and __alloc_ aren't used, so they've been removed for simplicity.
};
#elif LLDB_TEST_CASE == LLDB_TEST_VECTOR_WITH_POINTER_LAYOUT
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

#elif LLDB_TEST_CASE == LLDB_TEST_VECTOR_WITH_SIZE_LAYOUT
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

#else
#error LLDB_TEST_CASE defined out-of-range
#undef LLDB_TEST_CASE
#endif

} // namespace __lldb
} // namespace std

int main() {
#ifdef LLDB_TEST_CASE
  int a1[] = {10};
  std::__lldb::vector<int> v0(a1, 0);
  std::__lldb::vector<int> v1(a1, 1);

  int a2[] = {-10, -20};
  std::__lldb::vector<int> v2(a2, 2);

  int a3[] = {56, 10, 87};
  std::__lldb::vector<int> v3(a3, 3);

  return 0; // break here
#endif
}
