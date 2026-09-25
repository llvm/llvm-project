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
#include <memory>
#include <type_traits>

#include "kmp.h"

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

/// kmp_vector is a vector class for managing small vectors.
/// INLINE_THRESHOLD: Number of elements in the inline array. If exceeded, the
/// vector will grow dynamically.
template <typename T, size_t INLINE_THRESHOLD = 8> class kmp_vector final {
  static_assert(std::is_copy_constructible_v<T>,
                "T must be copy constructible");
  static_assert(std::is_destructible_v<T>, "T must be destructible");

  struct default_eq {
    bool operator()(const T &a, const T &b) const { return a == b; }
  };

  T inline_data[INLINE_THRESHOLD];
  T *data = inline_data;
  size_t count = 0;
  size_t capacity = INLINE_THRESHOLD;

  void copy_data(T *dst, const T *src, size_t num_elements) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      memcpy(dst, src, num_elements * sizeof(T));
    } else {
      for (size_t i = 0; i < num_elements; i++)
        new (&dst[i]) T(src[i]); // copy-construct to memory
    }
  }

  /// Grow by ~1.5x / at least by +1 element.
  /// If MinSize > 0, grow only if necessary to guarantee space
  /// for at least MinSize elements.
  void grow(size_t MinSize = 0) {
    if (MinSize) {
      if (MinSize <= capacity)
        return;
      capacity = MinSize;
    } else {
      capacity = capacity + (capacity / 2) + 1;
    }
    T *old_data = data != inline_data ? data : nullptr;
    data =
        static_cast<T *>(KMP_INTERNAL_REALLOC(old_data, capacity * sizeof(T)));
    if (!data)
      KMP_FATAL(MemoryAllocFailed);
    // Copy the data to the new array if we didn't use a dynamic array before.
    if (!old_data)
      copy_data(data, inline_data, count);
  }

  void init(size_t new_capacity, const T *init_data, size_t new_count) {
    assert(new_capacity >= new_count &&
           "more elements requested than capacity");
    if (new_capacity > capacity)
      grow(new_capacity);
    if (init_data)
      copy_data(data, init_data, new_count);
    count = new_count;
  }

  /// Move data from other vector to this vector (which must be emptied before)
  void move_from(kmp_vector &&other) {
    assert(empty() && "must be empty before overwriting");
    if (other.data == other.inline_data) {
      // Cannot move inline data, must copy.
      init(other.capacity, other.data, other.count);
    } else {
      // Steal dynamic data.
      data = other.data;
      count = other.count;
      capacity = other.capacity;
    }
    other.reset(/*free_data=*/false);
  }

  void reset(bool free_data) {
    if (free_data && data != inline_data) {
      clear();
      KMP_INTERNAL_FREE(data);
    }
    data = inline_data;
    count = 0;
    capacity = INLINE_THRESHOLD;
  }

public:
  ~kmp_vector() { reset(/*free_data=*/true); }

  explicit kmp_vector(size_t capacity = 0) { init(capacity, nullptr, 0); }

  kmp_vector(size_t capacity, const T *init_data, size_t count) {
    init(capacity, init_data, count);
  }

  kmp_vector(const kmp_vector &other) {
    init(other.capacity, other.data, other.count);
  }

  kmp_vector(kmp_vector &&other) noexcept { move_from(std::move(other)); }

  kmp_vector &operator=(const kmp_vector &other) {
    if (this != &other) {
      reset(/*free_data=*/true);
      init(other.capacity, other.data, other.count);
    }
    return *this;
  }

  kmp_vector &operator=(kmp_vector &&other) noexcept {
    if (this != &other) {
      reset(/*free_data=*/true);
      move_from(std::move(other));
    }
    return *this;
  }

  /// Destroy all elements in the vector. Doesn't free the memory.
  void clear() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_t i = 0; i < count; i++)
        data[i].~T();
    }
    count = 0;
  }

  /// Check if the vector contains the given value.
  /// If a comparator is provided, it will be used to compare the values.
  /// Otherwise, the equality operator will be used.
  template <typename Fn = default_eq>
  bool contains(const T &value, const Fn &comp = Fn{}) const {
    static_assert(std::is_invocable_r_v<bool, Fn, const T &, const T &>,
                  "predicate must be callable as bool(const T &, const T &)");
    for (size_t i = 0; i < count; i++) {
      if (comp(data[i], value))
        return true;
    }
    return false;
  }

  bool empty() const { return !count; }

  /// Check if the two vectors are equal with set semantics.
  /// Current implementation is naive O(n^2) and not optimized for performance.
  /// Handles duplicates correctly.
  template <typename Fn = default_eq>
  bool is_set_equal(const kmp_vector &other, const Fn &comp = Fn{}) const {
    static_assert(std::is_invocable_r_v<bool, Fn, const T &, const T &>,
                  "predicate must be callable as bool(const T &, const T &)");
    for (const T &val : *this) {
      if (!other.contains(val, comp))
        return false;
    }
    for (const T &val : other) {
      if (!contains(val, comp))
        return false;
    }
    return true;
  }

  /// Add a new element to the end of the vector.
  void push_back(const T &value) {
    if (count == capacity)
      grow();
    if constexpr (std::is_trivially_copyable_v<T>)
      data[count++] = value;
    else
      new (&data[count++]) T(value);
  }

  /// Reserve space for the given number of elements.
  /// (Note: does not shrink the vector.)
  void reserve(size_t new_capacity) {
    if (new_capacity > capacity)
      grow(new_capacity);
  }

  size_t size() const { return count; }

  T &operator[](size_t index) {
    assert(index < count && "Index out of bounds");
    return data[index];
  }
  const T &operator[](size_t index) const {
    assert(index < count && "Index out of bounds");
    return data[index];
  }

  /// Iterator support (raw pointers work as iterators for contiguous storage)
  T *begin() { return data; }
  T *end() { return data + count; }
  const T *begin() const { return data; }
  const T *end() const { return data + count; }
};

#endif // KMP_ADT_H
