//==----------- id.hpp --- SYCL iteration id -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/array.hpp>
#include <CL/sycl/item.hpp>
#include <CL/sycl/range.hpp>

namespace cl {
namespace sycl {
template <int dimensions> class range;
template <int dimensions = 1> struct id : public detail::array<dimensions> {
private:
  using base = detail::array<dimensions>;
  static_assert(dimensions >= 1 && dimensions <= 3,
                "id can only be 1, 2, or 3 dimentional.");
public:
  id() = default;

  /* The following constructor is only available in the id struct
   * specialization where: dimensions==1 */
  template <int N = dimensions>
  id(typename std::enable_if<(N == 1), size_t>::type dim0) : base(dim0) {}

  template <int N = dimensions>
  id(typename std::enable_if<(N == 1), const range<dimensions> &>::type
         range_size)
      : base(range_size.get(0)) {}

  template <int N = dimensions>
  id(typename std::enable_if<(N == 1), const item<dimensions> &>::type item)
      : base(item.get_id(0)) {}

  /* The following constructor is only available in the id struct
   * specialization where: dimensions==2 */
  template <int N = dimensions>
  id(typename std::enable_if<(N == 2), size_t>::type dim0, size_t dim1)
      : base(dim0, dim1) {}

  template <int N = dimensions>
  id(typename std::enable_if<(N == 2), const range<dimensions> &>::type
         range_size)
      : base(range_size.get(0), range_size.get(1)) {}

  template <int N = dimensions>
  id(typename std::enable_if<(N == 2), const item<dimensions> &>::type item)
      : base(item.get_id(0), item.get_id(1)) {}

  /* The following constructor is only available in the id struct
   * specialization where: dimensions==3 */
  template <int N = dimensions>
  id(typename std::enable_if<(N == 3), size_t>::type dim0, size_t dim1,
     size_t dim2)
      : base(dim0, dim1, dim2) {}

  template <int N = dimensions>
  id(typename std::enable_if<(N == 3), const range<dimensions> &>::type
         range_size)
      : base(range_size.get(0), range_size.get(1), range_size.get(2)) {}

  template <int N = dimensions>
  id(typename std::enable_if<(N == 3), const item<dimensions> &>::type item)
      : base(item.get_id(0), item.get_id(1), item.get_id(2)) {}

  explicit operator range<dimensions>() const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = this->get(i);
    }
    return result;
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
  #define __SYCL_GEN_OPT(op)                                                   \
    id<dimensions> operator op(const id<dimensions> &rhs) const {              \
      id<dimensions> result;                                                   \
      for (int i = 0; i < dimensions; ++i) {                                   \
        result.common_array[i] = this->common_array[i] op rhs.common_array[i]; \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
    id<dimensions> operator op(const size_t &rhs) const {                      \
      id<dimensions> result;                                                   \
      for (int i = 0; i < dimensions; ++i) {                                   \
        result.common_array[i] = this->common_array[i] op rhs;                 \
      }                                                                        \
      return result;                                                           \
    }                                                                          \
    friend id<dimensions> operator op(const size_t &lhs,                       \
                                      const id<dimensions> &rhs) {             \
      id<dimensions> result;                                                   \
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
    id<dimensions> &operator op(const id<dimensions> &rhs) {                   \
      for (int i = 0; i < dimensions; ++i) {                                   \
        this->common_array[i] op rhs.common_array[i];                          \
      }                                                                        \
      return *this;                                                            \
    }                                                                          \
    id<dimensions> &operator op(const size_t &rhs) {                           \
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

namespace detail {
template <int dimensions>
size_t getOffsetForId(range<dimensions> Range, id<dimensions> Id,
                      id<dimensions> Offset) {
  size_t offset = 0;
  for (int i = 0; i < dimensions; ++i)
    offset = offset * Range[i] + Offset[i] + Id[i];
  return offset;
}
} // namespace detail
} // namespace sycl
} // namespace cl
