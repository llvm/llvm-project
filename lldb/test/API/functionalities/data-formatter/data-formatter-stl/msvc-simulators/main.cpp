//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Layout approximations of MSVC STL types so the data formatters can be
// exercised without a Windows MSVC toolchain.

#include <stddef.h>
#include <stdint.h>

namespace std {

template <size_t N> class bitset {
public:
  using _Ty = unsigned long;
  static constexpr size_t _Bitsperword = sizeof(_Ty) * 8;
  static constexpr size_t _Words = N == 0 ? 0 : (N - 1) / _Bitsperword;
  _Ty _Array[_Words + 1]{};
};

template <class T> class initializer_list {
public:
  initializer_list(const T *first, const T *last)
      : _First(first), _Last(last) {}
  const T *_First;
  const T *_Last;
};

template <class T> struct _Vector_val {
  T *_Myfirst;
  T *_Mylast;
  T *_Myend;
};

template <class T> struct _Compressed_pair {
  _Vector_val<T> _Myval2;
};

template <class T> class vector {
public:
  vector(T *begin, T *end) : _Mypair{{begin, end, end}} {}
  _Compressed_pair<T> _Mypair;
};

template <class T, class C = vector<T>> class queue {
public:
  explicit queue(C container) : c(container) {}
  C c;
};

template <class T, class C = vector<T>> class stack {
public:
  explicit stack(C container) : c(container) {}
  C c;
};

template <class T, class C = vector<T>> class priority_queue {
public:
  explicit priority_queue(C container) : c(container) {}
  C c;
};

template <class T> class valarray {
public:
  valarray(T *ptr, size_t size) : _Myptr(ptr), _Mysize(size) {}
  T *_Myptr;
  size_t _Mysize;
};

template <class T, class E> class expected {
public:
  expected(const T &value) : _Value(value), _Has_value(true) {}
  expected(const E &error, bool) : _Unexpected(error), _Has_value(false) {}
  union {
    T _Value;
    E _Unexpected;
  };
  bool _Has_value;
};

struct source_location {
  uint32_t _Line;
  uint32_t _Column;
  const char *_File;
  const char *_Function;
};

struct error_code {
  int _Myval;
  const void *_Mycat;
};

template <class T> class _Vector_iterator {
public:
  explicit _Vector_iterator(T *ptr) : _Ptr(ptr) {}
  T *_Ptr;
};

namespace chrono {
struct nanoseconds {
  long long _MyRep;
};
struct seconds {
  long long _MyRep;
};
} // namespace chrono

namespace filesystem {
struct path {
  const char *_Text;
};
} // namespace filesystem

} // namespace std

int main() {
  std::bitset<0> empty_bitset;
  std::bitset<13> small_bitset;
  small_bitset._Array[0] = 0b00011111111100UL; // bits 2..9 set

  int init_vals[] = {1, 2, 3, 4, 5};
  std::initializer_list<int> ili(init_vals, init_vals + 5);

  int vec_vals[] = {10, 20, 30};
  std::vector<int> vec(vec_vals, vec_vals + 3);
  std::queue<int> q(vec);
  std::stack<int> st(vec);
  std::priority_queue<int> pq(vec);

  int va_vals[] = {1, 12, 123, 1234};
  std::valarray<int> va(va_vals, 4);

  std::expected<int, const char *> ok(7);
  std::expected<int, const char *> err("boom", true);

  std::source_location loc{6, 1, "main.cpp", "int main()"};
  std::source_location loc_empty{0, 0, "", ""};

  std::chrono::nanoseconds ns{1};
  std::chrono::seconds s{1234};

  std::error_code ec{2, nullptr};
  std::filesystem::path p{"C:\\tmp\\file.txt"};

  int item = 3;
  std::_Vector_iterator<int> it(&item);

  return 0; // break here
}
