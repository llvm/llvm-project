/*
 * kmp_adt.cpp -- Advanced Data Types used internally
 *
 * FIXME: This is in intermediate solution until we agree and implement some
 * common resource according to
 * https://discourse.llvm.org/t/meta-rfc-adts-without-c-runtime-dependency/90317.
 * As soon as we will have this common resource that can be used for runtimes
 * such as openmp that want to avoid the link dependency to the C++ STL, this
 * shall be refactored.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_adt.h"
#include "kmp.h"
#include "kmp_str.h"

#include <cctype>
#include <cstring>

bool kmp_str_ref::consume_integer(int &value, bool allow_zero,
                                  bool allow_negative) {
  kmp_str_ref orig = *this; // save state
  bool is_negative = consume_front("-");
  if (is_negative && !allow_negative) {
    *this = orig;
    return false;
  }
  size_t num_digits = count_while(
      [](char c) { return isdigit(static_cast<unsigned char>(c)) != 0; });
  if (!num_digits) {
    *this = orig;
    return false;
  }
  value = __kmp_basic_str_to_int(sv.data(), num_digits);
  if (value == INT_MAX) {
    *this = orig;
    return false;
  }
  drop_front(num_digits);
  if (is_negative)
    value = -value;
  if (!allow_zero && value == 0) {
    *this = orig;
    return false;
  }
  return true;
}

char *kmp_str_ref::copy() const {
  char *copy_str = static_cast<char *>(KMP_INTERNAL_MALLOC(length() + 1));
  if (!copy_str)
    KMP_FATAL(MemoryAllocFailed);
  memcpy(copy_str, sv.data(), length());
  copy_str[length()] = '\0';
  return copy_str;
}
