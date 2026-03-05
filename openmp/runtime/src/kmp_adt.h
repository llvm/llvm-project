/*
 * kmp_adt.h -- Advanced Data Types used internally
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __KMP_ADT_H__
#define __KMP_ADT_H__

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <string_view>

#include "kmp.h"

/// kmp_str_ref is a non-owning string class (similar to llvm::StringRef).
class kmp_str_ref final {
  std::string_view sv;

public:
  kmp_str_ref(const char *str) : sv(str) {}
  kmp_str_ref(std::string_view sv) : sv(sv) {}

  kmp_str_ref(const kmp_str_ref &other) = default;
  kmp_str_ref &operator=(const kmp_str_ref &other) = default;

  // Check if the string starts with the given prefix and remove it from the
  // string afterwards.
  bool consume_front(kmp_str_ref prefix) {
    if (length() < prefix.length())
      return false;
    if (sv.compare(0, prefix.length(), prefix.sv) != 0)
      return false;
    drop_front(prefix.length());
    return true;
  }

  // Start consuming an integer from the start of the string and remove it from
  // the string afterwards.
  // The maximum integer value that can currently be parsed is INT_MAX - 1.
  bool consume_integer(int &value, bool allow_zero = true,
                       bool allow_negative = false) {
    kmp_str_ref orig = *this; // save state
    bool is_negative = consume_front("-");
    if (is_negative && !allow_negative) {
      *this = orig;
      return false;
    }
    size_t num_digits = count_while([](char c) {
      return static_cast<bool>(isdigit(static_cast<unsigned char>(c)));
    });
    if (!num_digits) {
      *this = orig;
      return false;
    }
    assert(num_digits <= INT_MAX);
    value = __kmp_basic_str_to_int(sv.data(), static_cast<int>(num_digits));
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

  // Get an own duplicate of the string.
  // Must be freed with KMP_INTERNAL_FREE().
  char *copy() const {
    char *copy_str = static_cast<char *>(KMP_INTERNAL_MALLOC(length() + 1));
    assert(copy_str && "copy() failed to allocate memory");
    memcpy(copy_str, sv.data(), length());
    copy_str[length()] = '\0';
    return copy_str;
  }

  // Count the number of characters in the string while the predicate returns
  // true.
  size_t count_while(bool (*predicate)(char)) const {
    size_t i = 0;
    while (i < length() && predicate(sv[i]))
      ++i;
    return i;
  }

  // Drop the first n characters from the string.
  // (Limit n to the length of the string.)
  void drop_front(size_t n) { sv.remove_prefix(std::min(n, length())); }

  // Drop characters from the string while the predicate returns true.
  void drop_while(bool (*predicate)(char)) {
    drop_front(count_while(predicate));
  }

  // Check if the string is empty.
  bool empty() const { return sv.empty(); }

  // Get the length of the string.
  size_t length() const { return sv.length(); }

  // Drop space from the start of the string.
  void skip_space() {
    drop_while([](char c) {
      return static_cast<bool>(isspace(static_cast<unsigned char>(c)));
    });
  }

  // Construct a new string with the longest prefix of the original string that
  // satisfies the predicate. Doesn't modify the original string.
  kmp_str_ref take_while(bool (*predicate)(char)) const {
    return kmp_str_ref(sv.substr(0, count_while(predicate)));
  }

  // Iterator support (raw pointers work as iterators for contiguous storage)
  const char *begin() const { return sv.data(); }
  const char *end() const { return sv.data() + length(); }
};

#endif // __KMP_ADT_H__
