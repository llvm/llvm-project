//==----------- range.hpp --- SYCL iteration range -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/array.hpp>
#include <stdexcept>
#include <type_traits>

namespace cl {
namespace sycl {
template <int dimensions> struct id;
template <int dimensions = 1>
class range : public detail::array<dimensions> {
  static_assert(dimensions >= 1 && dimensions <= 3,
                "range can only be 1, 2, or 3 dimentional.");
  using base = detail::array<dimensions>;
public:
  /* The following constructor is only available in the range class
  specialization where: dimensions==1 */
  template <int N = dimensions>
  range(typename std::enable_if<(N == 1), size_t>::type dim0) : base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: dimensions==2 */
  template <int N = dimensions>
  range(typename std::enable_if<(N == 2), size_t>::type dim0, size_t dim1)
      : base(dim0, dim1) {}

  /* The following constructor is only available in the range class
  specialization where: dimensions==3 */
  template <int N = dimensions>
  range(typename std::enable_if<(N == 3), size_t>::type dim0, size_t dim1,
        size_t dim2) : base(dim0, dim1, dim2) {}

  explicit operator id<dimensions>() const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = this->get(i);
    }
    return result;
  }

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < dimensions; ++i) {
      size *= this->get(i);
    }
    return size;
  }

  range(const range<dimensions> &rhs) = default;
  range(range<dimensions> &&rhs) = default;
  range<dimensions> &operator=(const range<dimensions> &rhs) = default;
  range<dimensions> &operator=(range<dimensions> &&rhs) = default;
  range() = default;

  // OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
  #define __SYCL_GEN_OPT(op)                                                   \
    range<dimensions> operator op(const range<dimensions> &rhs) const {        \
      range<dimensions> result;                                                \
      for (int i = 0; i < dimensions; ++i) {                                   \
        result.common_array[i] = this->common_array[i] op rhs.common_array[i]; \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
    range<dimensions> operator op(const size_t &rhs) const {                   \
      range<dimensions> result;                                                \
      for (int i = 0; i < dimensions; ++i) {                                   \
        result.common_array[i] = this->common_array[i] op rhs;                 \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
    friend range<dimensions> operator op(const size_t &lhs,                    \
                                       const range<dimensions> &rhs) {         \
      range<dimensions> result;                                                \
      for (int i = 0; i < dimensions; ++i) {                                   \
        result.common_array[i] = lhs op rhs.common_array[i];                   \
      }                                                                        \
      return result;                                                           \
    }                                                                          \

  __SYCL_GEN_OPT(+)
  __SYCL_GEN_OPT(-)
  __SYCL_GEN_OPT(*)
  __SYCL_GEN_OPT(/)
  __SYCL_GEN_OPT(%)
  __SYCL_GEN_OPT(<<)
  __SYCL_GEN_OPT(>>)
  __SYCL_GEN_OPT(&)
  __SYCL_GEN_OPT(|)
  __SYCL_GEN_OPT(^)
  __SYCL_GEN_OPT(&&)
  __SYCL_GEN_OPT(||)
  __SYCL_GEN_OPT(<)
  __SYCL_GEN_OPT(>)
  __SYCL_GEN_OPT(<=)
  __SYCL_GEN_OPT(>=)

  #undef __SYCL_GEN_OPT

  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
  #define __SYCL_GEN_OPT(op)                                                   \
    range<dimensions> &operator op(const range<dimensions> &rhs) {             \
      for (int i = 0; i < dimensions; ++i) {                                   \
        this->common_array[i] op rhs[i];                                       \
      }                                                                        \
      return *this;                                                            \
    }                                                                          \
    range<dimensions> &operator op(const size_t &rhs) {                        \
      for (int i = 0; i < dimensions; ++i) {                                   \
        this->common_array[i] op rhs;                                          \
      }                                                                        \
      return *this;                                                            \
    }                                                                          \


  __SYCL_GEN_OPT(+=)
  __SYCL_GEN_OPT(-=)
  __SYCL_GEN_OPT(*=)
  __SYCL_GEN_OPT(/=)
  __SYCL_GEN_OPT(%=)
  __SYCL_GEN_OPT(<<=)
  __SYCL_GEN_OPT(>>=)
  __SYCL_GEN_OPT(&=)
  __SYCL_GEN_OPT(|=)
  __SYCL_GEN_OPT(^=)

  #undef __SYCL_GEN_OPT

};
} // namespace sycl
} // namespace cl
