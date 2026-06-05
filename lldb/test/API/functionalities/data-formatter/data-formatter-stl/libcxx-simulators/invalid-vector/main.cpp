#define COMPRESSED_PAIR_REV 4
#include <libcxx-simulators-common/compressed_pair.h>
#include <stddef.h>

namespace std {
inline namespace __1 {
template <typename T> struct vector {
  T *__begin_;
  T *__end_;
  _LLDB_COMPRESSED_PAIR(T *, __cap_ = nullptr, void *, __alloc_);
};
} // namespace __1

inline namespace __2 {
template <typename T> struct vector {};
} // namespace __2

inline namespace __3 {
template <typename T> struct vector {
  short *__begin_;
  short *__end_;
  _LLDB_COMPRESSED_PAIR(short *, __cap_ = nullptr, void *, __alloc_);
};
} // namespace __3

struct DummyStruct { int x; int y; };

inline namespace __4 {
template <typename T> struct __vector_layout {
  int __begin_;
  T *__end_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __4

inline namespace __5 {
template <typename T> struct __vector_layout {
  T *__begin_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __5

inline namespace __6 {
template <typename T> struct __vector_layout {
  T *__begin_;
  DummyStruct __end_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __6

inline namespace __7 {
template <typename T> struct __vector_layout {
  T *__begin_;
  T *__end_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __7

inline namespace __8 {
template <typename T> struct __vector_layout {
  T *__begin_;
  size_t __size_;
};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __8

inline namespace __9 {
template <typename T> struct __vector_layout {};
template <typename T> struct vector {
  __vector_layout<T> __layout_;
};
} // namespace __9
} // namespace std

int main() {
  int arr[] = {1, 2, 3};
  std::__1::vector<int> v1{.__begin_ = arr, .__end_ = nullptr};
  std::__1::vector<int> v2{.__begin_ = nullptr, .__end_ = arr};
  std::__1::vector<int> v3{.__begin_ = &arr[2], .__end_ = arr};
  std::__2::vector<int> v4;

  char carr[] = {'a'};
  std::__3::vector<char> v5{.__begin_ = reinterpret_cast<short*>(carr), .__end_ = reinterpret_cast<short*>(carr + 1)};

  std::__4::vector<int> v6{.__layout_ = { .__begin_ = 0, .__end_ = arr }};
  std::__5::vector<int> v7{.__layout_ = { .__begin_ = arr }};
  std::__6::vector<int> v8{.__layout_ = { .__begin_ = arr, .__end_ = {1, 2} }};
  std::__7::vector<int> v9{.__layout_ = { .__begin_ = arr, .__end_ = nullptr }};
  std::__7::vector<int> v10{.__layout_ = { .__begin_ = nullptr, .__end_ = arr }};
  std::__8::vector<int> v11{.__layout_ = { .__begin_ = nullptr, .__size_ = 3 }};
  std::__8::vector<int> v12{.__layout_ = { .__begin_ = arr, .__size_ = static_cast<size_t>(-2) }};
  std::__9::vector<int> v13;

  return 0;
}
