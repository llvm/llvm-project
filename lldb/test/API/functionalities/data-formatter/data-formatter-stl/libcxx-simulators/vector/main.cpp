// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// libc++ has changed how `std::vector` is represented over time. This file
// provides approximations of these representations for testing the vector
// pretty-printer. This lets us test that LLDB is able to handle all iterations
// of libc++'s `std::vector`.

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

  // libc++ changed how the capacity member and allocator were stored in
  // 27c8338. LLDB only relies on `__begin_` and `__end_`. Adding the capacity
  // and allocator members in their different formats doesn't add test coverage,
  // but may convince the reader that it does. As such, we don't provide two
  // legacy layouts.
  //
  // Before 27c83382d83dce0f33ae67abb3bc94977cb3031f:
  //   __compressed_pair<size_type, __storage_allocator> __cap_alloc_;
  //
  // Since 27c83382d83dce0f33ae67abb3bc94977cb3031f:
  //   _LIBCPP_COMPRESSED_PAIR(pointer, __cap_ = nullptr, allocator_type,
  //   __alloc_);
};
} // namespace __LegacyLayout

inline namespace __PointerBasedLayout {
// `__PointerBasedLayout::__vector_layout` is structurally equal to
// `__LegacyLayout::vector`.
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
