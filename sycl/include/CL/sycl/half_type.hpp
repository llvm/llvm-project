//==-------------- half_type.hpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <functional>

namespace cl {
namespace sycl {
namespace detail {
namespace half_impl {

class half {
public:
  half() = default;
  half(const half &) = default;
  half(half &&) = default;

  half(const float &rhs);

  half &operator=(const half &rhs) = default;

  // Operator +=, -=, *=, /=
  half &operator+=(const half &rhs);

  half &operator-=(const half &rhs);

  half &operator*=(const half &rhs);

  half &operator/=(const half &rhs);

  // Operator ++, --
  half &operator++() {
    *this += 1;
    return *this;
  }

  half operator++(int) {
    half ret(*this);
    operator++();
    return ret;
  }

  half &operator--() {
    *this -= 1;
    return *this;
  }

  half operator--(int) {
    half ret(*this);
    operator--();
    return ret;
  }

  // Operator float
  operator float() const;

  template <typename Key> friend struct std::hash;

private:
  uint16_t Buf;
};
} // namespace half_impl
} // namespace detail

} // namespace sycl
} // namespace cl

namespace std {

template <> struct hash<cl::sycl::detail::half_impl::half> {
  size_t operator()(cl::sycl::detail::half_impl::half const &key) const
      noexcept {
    return hash<uint16_t>()(key.Buf);
  }
};

} // namespace std
