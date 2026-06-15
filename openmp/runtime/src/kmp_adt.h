/*
 * kmp_adt.h -- Advanced Data Types used internally
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

#ifndef KMP_ADT_H
#define KMP_ADT_H

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

/// kmp_str_ref is a non-owning string class (similar to llvm::StringRef).
class kmp_str_ref final {
  const char *data;
  size_t len;

public:
  static constexpr size_t npos = SIZE_MAX;

  kmp_str_ref(const char *str) : data(str), len(str ? strlen(str) : 0) {}
  kmp_str_ref(const char *str, size_t len) : data(str), len(len) {
    assert((data || !len) && "len must be 0 for nullptr data");
  }

  kmp_str_ref(const kmp_str_ref &other) = default;
  kmp_str_ref &operator=(const kmp_str_ref &other) = default;

  /// Check if the string starts with the given prefix and remove it from the
  /// string afterwards.
  bool consume_front(kmp_str_ref prefix) {
    if (len < prefix.len)
      return false;
    if (empty() || prefix.empty()) // avoid calling memcmp on potential nullptr
      return true;
    if (memcmp(data, prefix.data, prefix.len) != 0)
      return false;
    drop_front(prefix.len);
    return true;
  }

  /// Start consuming an integer from the start of the string and remove it from
  /// the string afterwards.
  /// The maximum integer value that can currently be parsed is INT_MAX - 1.
  bool consume_integer(int &value, bool allow_zero = true,
                       bool allow_negative = false);

  /// Get an own duplicate of the string.
  /// Must be freed with KMP_INTERNAL_FREE().
  char *copy() const;

  /// Count the number of characters in the string while the predicate returns
  /// true.
  template <typename Fn> size_t count_while(const Fn &predicate) const {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    size_t n = find_if_not(predicate);
    return n == npos ? len : n;
  }

  /// Drop the first n characters from the string.
  /// (Limit n to the length of the string.)
  void drop_front(size_t n) {
    if (n > len)
      n = len;
    data += n;
    len -= n;
  }

  /// Drop characters from the string while the predicate returns true.
  template <typename Fn> void drop_while(const Fn &predicate) {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    drop_front(count_while(predicate));
  }

  /// Check if the string is empty.
  bool empty() const { return len == 0; }

  /// Return the index of the first character in the string for which the
  /// predicate returns true.
  /// Returns npos if no match is found.
  template <typename Fn> size_t find_if(const Fn &predicate) const {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    size_t i = 0;
    while (i < len && !predicate(data[i]))
      ++i;
    return i < len ? i : npos;
  }

  /// Return the index of the first character in the string for which the
  /// predicate returns false.
  /// Returns npos if no match is found.
  template <typename Fn> size_t find_if_not(const Fn &predicate) const {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    return find_if([predicate](char c) { return !predicate(c); });
  }

  /// Get the length of the string.
  size_t length() const { return len; }
  size_t size() const { return length(); }

  /// Drop space from the start of the string.
  void skip_space() {
    drop_while([](char c) {
      return static_cast<bool>(isspace(static_cast<unsigned char>(c)));
    });
  }

  /// Construct a new string with the longest prefix of the original string that
  /// satisfies the predicate. Doesn't modify the original string.
  template <typename Fn> kmp_str_ref take_while(const Fn &predicate) const {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    return kmp_str_ref(data, count_while(predicate));
  }

  /// Iterator support (raw pointers work as iterators for contiguous storage)
  const char *begin() const { return data; }
  const char *end() const { return data + len; }
};

#endif // KMP_ADT_H
