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

#include "kmp.h"

/// kmp_str_ref is a non-owning string class (similar to llvm::StringRef).
class kmp_str_ref final {
  const char *data;
  size_t len;

public:
  kmp_str_ref(const char *data) : data(data), len(data ? strlen(data) : 0) {
    assert(data && "kmp_str_ref cannot be constructed from nullptr");
  }
  kmp_str_ref(const char *data, size_t len) : data(data), len(len) {
    assert(data && "kmp_str_ref cannot be constructed from nullptr");
  }

  kmp_str_ref(const kmp_str_ref &other) = default;
  kmp_str_ref &operator=(const kmp_str_ref &other) = default;

  // Check if the string starts with the given prefix and remove it from the
  // string afterwards.
  bool consume_front(kmp_str_ref prefix) {
    if (len < prefix.len)
      return false;
    if (memcmp(data, prefix.data, prefix.len) != 0)
      return false;
    data += prefix.len;
    len -= prefix.len;
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
    value = __kmp_basic_str_to_int(data, static_cast<int>(num_digits));
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
    char *copy_str = static_cast<char *>(KMP_INTERNAL_MALLOC(len + 1));
    assert(copy_str && "copy() failed to allocate memory");
    memcpy(copy_str, data, len);
    copy_str[len] = '\0';
    return copy_str;
  }

  // Count the number of characters in the string while the predicate returns
  // true.
  size_t count_while(bool (*predicate)(char)) const {
    size_t i = 0;
    while (i < len && predicate(data[i]))
      ++i;
    return i;
  }

  // Drop the first n characters from the string.
  void drop_front(size_t n) {
    if (n > len)
      n = len;
    data += n;
    len -= n;
  }

  // Drop characters from the string while the predicate returns true.
  void drop_while(bool (*predicate)(char)) {
    drop_front(count_while(predicate));
  }

  // Check if the string is empty.
  bool empty() const { return len == 0; }

  // Get the length of the string.
  size_t length() const { return len; }

  // Drop space from the start of the string.
  void skip_space() {
    drop_while([](char c) {
      return static_cast<bool>(isspace(static_cast<unsigned char>(c)));
    });
  }

  // Construct a new string with the longest prefix of the original string that
  // satisfies the predicate. Doesn't modify the original string.
  kmp_str_ref take_while(bool (*predicate)(char)) const {
    return kmp_str_ref(data, count_while(predicate));
  }

  // Iterator support (raw pointers work as iterators for contiguous storage)
  const char *begin() const { return data; }
  const char *end() const { return data + len; }
};

#endif // __KMP_ADT_H__
