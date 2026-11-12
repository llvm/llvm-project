/*
 * kmp_utils.h -- Utilities that used internally
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __KMP_UTILS_H__
#define __KMP_UTILS_H__

#include <cstddef>

#include "kmp.h"

/// A simple pure header implementation of VLA that aims to replace uses of
/// actual VLA, which can cause compile warning. This class by default creates a
/// stack buffer that can accomodate \p N elements. If the number of elements is
/// greater than \p N, then a heap buffer will be allocated and used to
/// accomodate the elements. Similar to the actual VLA, we don't check boundary
/// (for now), so we will not store the number of elements. We can always revise
/// it later.
template <typename T, unsigned N = 8> class SimpleVLA final {
  T StackBuffer[N];
  T *HeapBuffer = nullptr;
  T *Ptr = StackBuffer;

public:
  SimpleVLA() = delete;
  SimpleVLA(const SimpleVLA &) = delete;
  SimpleVLA(SimpleVLA &&) = delete;
  SimpleVLA &operator=(const SimpleVLA &) = delete;
  SimpleVLA &operator=(SimpleVLA &&) = delete;

  explicit SimpleVLA(unsigned NumOfElements) noexcept {
    if (NumOfElements > N) {
      HeapBuffer =
          reinterpret_cast<T *>(__kmp_allocate(NumOfElements * sizeof(T)));
      Ptr = HeapBuffer;
    }
  }

  ~SimpleVLA() {
    if (HeapBuffer)
      __kmp_free(HeapBuffer);
  }

  operator T *() noexcept { return Ptr; }
  operator const T *() const noexcept { return Ptr; }
};

#endif
