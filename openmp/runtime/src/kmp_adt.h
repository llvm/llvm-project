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

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

/// kmp_str_ref is a non-owning string class (similar to llvm::StringRef).
class kmp_str_ref final {
  std::string_view sv;

public:
  static constexpr size_t npos = SIZE_MAX;

  kmp_str_ref(const char *str)
      : sv(str ? std::string_view(str) : std::string_view()) {}
  kmp_str_ref(std::string_view sv) : sv(sv) {}

  kmp_str_ref(const kmp_str_ref &other) = default;
  kmp_str_ref &operator=(const kmp_str_ref &other) = default;

  /// Check if the string starts with the given prefix and remove it from the
  /// string afterwards.
  bool consume_front(kmp_str_ref prefix) {
    if (length() < prefix.length())
      return false;
    if (sv.compare(0, prefix.length(), prefix.sv) != 0)
      return false;
    drop_front(prefix.length());
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
    size_t len = find_if_not(predicate);
    return len == npos ? length() : len;
  }

  /// Drop the first n characters from the string.
  /// (Limit n to the length of the string.)
  void drop_front(size_t n) { sv.remove_prefix(std::min(n, length())); }

  /// Drop characters from the string while the predicate returns true.
  template <typename Fn> void drop_while(const Fn &predicate) {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    drop_front(count_while(predicate));
  }

  /// Check if the string is empty.
  bool empty() const { return sv.empty(); }

  /// Return the index of the first character in the string for which the
  /// predicate returns true.
  /// Returns npos if no match is found.
  template <typename Fn> size_t find_if(const Fn &predicate) const {
    static_assert(std::is_invocable_r_v<bool, Fn, char>,
                  "predicate must be callable as bool(char)");
    size_t i = 0;
    size_t len = length();
    while (i < len && !predicate(sv[i]))
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
  size_t length() const { return sv.length(); }
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
    return kmp_str_ref(sv.substr(0, count_while(predicate)));
  }

  /// Iterator support (raw pointers work as iterators for contiguous storage)
  const char *begin() const { return sv.data(); }
  const char *end() const { return sv.data() + length(); }
};

#endif // KMP_ADT_H
