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
#include <utility>

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

/// kmp_vector is a vector class for managing small vectors.
/// inline_threshold: Number of elements in the inline array. If exceeded, the
/// vector will grow dynamically.
template <typename T, size_t inline_threshold = 8> class kmp_vector final {
  static_assert(std::is_copy_constructible_v<T>,
                "T must be copy constructible");
  static_assert(std::is_destructible_v<T>, "T must be destructible");

  T inline_data[inline_threshold];
  T *data = inline_data;
  size_t count = 0;
  size_t capacity = inline_threshold;

  void copy_data(T *dst, const T *src, size_t num_elements) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      memcpy(dst, src, num_elements * sizeof(T));
    } else {
      for (size_t i = 0; i < num_elements; i++)
        new (&dst[i]) T(src[i]); // copy-construct to memory
    }
  }

  void grow() {
    size_t new_capacity = capacity + (capacity / 2) + 1;
    resize(new_capacity);
  }

  void init(size_t new_capacity, const T *init_data, size_t new_count) {
    assert(new_capacity >= new_count);
    if (new_capacity > inline_threshold)
      resize(new_capacity);
    if (init_data)
      copy_data(data, init_data, new_count);
    count = new_count;
  }

  void move_from(kmp_vector &&other) {
    if (other.data == other.inline_data) {
      // Cannot move inline data, must copy.
      init(other.capacity, other.data, other.count);
    } else {
      // Steal dynamic data.
      data = other.data;
      count = other.count;
      capacity = other.capacity;
    }
    other.reset(false);
  }

  void reset(bool free_data) {
    if (free_data && data != inline_data) {
      clear();
      KMP_INTERNAL_FREE(data);
    }
    data = inline_data;
    count = 0;
    capacity = inline_threshold;
  }

  // resize only changes the capacity, not the size (i.e., the number of
  // actually used elements)
  void resize(size_t new_capacity) {
    // Currently only supports growing the capacity. (Consequently, doesn't need
    // to worry about going from a dynamic array back to an inline array.)
    assert(new_capacity > capacity && "resize() only supports growing");
    capacity = new_capacity;
    T *old_data = data != inline_data ? data : nullptr;
    data =
        static_cast<T *>(KMP_INTERNAL_REALLOC(old_data, capacity * sizeof(T)));
    assert(data);
    // Copy the data to the new array if we didn't use a dynamic array before.
    if (!old_data)
      copy_data(data, inline_data, count);
  }

public:
  ~kmp_vector() { reset(true); }

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
      reset(true);
      init(other.capacity, other.data, other.count);
    }
    return *this;
  }

  kmp_vector &operator=(kmp_vector &&other) noexcept {
    if (this != &other) {
      reset(true);
      move_from(std::move(other));
    }
    return *this;
  }

  // Destroy all elements in the vector. Doesn't free the memory.
  void clear() {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (size_t i = 0; i < count; i++)
        data[i].~T();
    }
    count = 0;
  }

  // Check if the vector contains the given value.
  // If a comparator is provided, it will be used to compare the values.
  // Otherwise, the equality operator will be used.
  bool contains(const T &value,
                bool (*comp)(const T &, const T &) = nullptr) const {
    for (size_t i = 0; i < count; i++) {
      if (comp ? comp(data[i], value) : data[i] == value)
        return true;
    }
    return false;
  }

  bool empty() const { return !count; }

  // Check if the two vectors are equal with set semantics.
  // Current implementation is naive O(n^2) and not optimized for performance.
  // Handles duplicates correctly.
  bool is_set_equal(const kmp_vector &other,
                    bool (*comp)(const T &, const T &) = nullptr) const {
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

  // Add a new element to the end of the vector.
  void push_back(const T &value) {
    if (count == capacity)
      grow();
    if constexpr (std::is_trivially_copyable_v<T>) {
      data[count++] = value;
    } else {
      new (&data[count++]) T(value);
    }
  }

  // Reserve space for the given number of elements.
  // (Note: does not shrink the vector.)
  void reserve(size_t new_capacity) {
    if (new_capacity > capacity)
      resize(new_capacity);
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

  // Iterator support (raw pointers work as iterators for contiguous storage)
  T *begin() { return data; }
  T *end() { return data + count; }
  const T *begin() const { return data; }
  const T *end() const { return data + count; }
};

#endif // __KMP_ADT_H__
