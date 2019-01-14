//==----------- id.hpp --- SYCL iteration id -------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
public:
  using base = detail::array<dimensions>;
  INLINE_IF_DEVICE id() = default;

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
  id<dimensions> operator+(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] + rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator-(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] - rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator*(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] * rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator/(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] / rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator%(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] % rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator<<(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] << rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator>>(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >> rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator&(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] & rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator|(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] | rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator^(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] ^ rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator&&(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] && rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator||(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] || rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator<(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] < rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator>(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] > rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator<=(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] <= rhs.common_array[i];
    }
    return result;
  }
  id<dimensions> operator>=(const id<dimensions> &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >= rhs.common_array[i];
    }
    return result;
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
  id<dimensions> operator+(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] + rhs;
    }
    return result;
  }
  id<dimensions> operator-(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] - rhs;
    }
    return result;
  }
  id<dimensions> operator*(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] * rhs;
    }
    return result;
  }
  id<dimensions> operator/(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] / rhs;
    }
    return result;
  }
  id<dimensions> operator%(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] % rhs;
    }
    return result;
  }
  id<dimensions> operator<<(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] << rhs;
    }
    return result;
  }
  id<dimensions> operator>>(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >> rhs;
    }
    return result;
  }
  id<dimensions> operator&(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] & rhs;
    }
    return result;
  }
  id<dimensions> operator|(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] | rhs;
    }
    return result;
  }
  id<dimensions> operator^(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] ^ rhs;
    }
    return result;
  }
  id<dimensions> operator&&(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] && rhs;
    }
    return result;
  }
  id<dimensions> operator||(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] || rhs;
    }
    return result;
  }
  id<dimensions> operator<(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] < rhs;
    }
    return result;
  }
  id<dimensions> operator>(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] > rhs;
    }
    return result;
  }
  id<dimensions> operator<=(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] <= rhs;
    }
    return result;
  }
  id<dimensions> operator>=(const size_t &rhs) const {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >= rhs;
    }
    return result;
  }

  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
  id<dimensions> &operator+=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] += rhs[i];
    }
    return *this;
  }
  id<dimensions> &operator-=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] -= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator*=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] *= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator/=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] /= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator%=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] %= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator<<=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] <<= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator>>=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] >>= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator&=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] &= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator|=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] |= rhs.common_array[i];
    }
    return *this;
  }
  id<dimensions> &operator^=(const id<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] ^= rhs.common_array[i];
    }
    return *this;
  }

  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
  id<dimensions> &operator+=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] += rhs;
    }
    return *this;
  }
  id<dimensions> &operator-=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] -= rhs;
    }
    return *this;
  }
  id<dimensions> &operator*=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] *= rhs;
    }
    return *this;
  }
  id<dimensions> &operator/=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] /= rhs;
    }
    return *this;
  }
  id<dimensions> &operator%=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] %= rhs;
    }
    return *this;
  }
  id<dimensions> &operator<<=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] <<= rhs;
    }
    return *this;
  }
  id<dimensions> &operator>>=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] >>= rhs;
    }
    return *this;
  }
  id<dimensions> &operator&=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] &= rhs;
    }
    return *this;
  }
  id<dimensions> &operator|=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] |= rhs;
    }
    return *this;
  }
  id<dimensions> &operator^=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] ^= rhs;
    }
    return *this;
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ^, <, >, <=, >=, &&, ||
  friend id<dimensions> operator+(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs + rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator-(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs - rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator*(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs * rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator/(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs / rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator%(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs % rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator<<(const size_t &lhs,
                                   const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs << rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator>>(const size_t &lhs,
                                   const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs >> rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator&(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs & rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator|(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs | rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator^(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs ^ rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator<(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs < rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator>(const size_t &lhs,
                                  const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs > rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator<=(const size_t &lhs,
                                   const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs <= rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator>=(const size_t &lhs,
                                   const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs >= rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator&&(const size_t &lhs,
                                   const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs && rhs.common_array[i];
    }
    return result;
  }
  friend id<dimensions> operator||(const size_t &lhs,
                                   const id<dimensions> &rhs) {
    id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs || rhs.common_array[i];
    }
    return result;
  }
};

namespace detail {
template <int dimensions> INLINE_IF_DEVICE
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
