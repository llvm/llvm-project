//==----------- range.hpp --- SYCL iteration range -------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
public:
  using base = detail::array<dimensions>;
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
  range<dimensions> operator+(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] + rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator-(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] - rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator*(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] * rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator/(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] / rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator%(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] % rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator<<(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] << rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator>>(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >> rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator&(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] & rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator|(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] | rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator^(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] ^ rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator&&(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] && rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator||(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] || rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator<(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] < rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator>(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] > rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator<=(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] <= rhs.common_array[i];
    }
    return result;
  }
  range<dimensions> operator>=(const range<dimensions> &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >= rhs.common_array[i];
    }
    return result;
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
  range<dimensions> operator+(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] + rhs;
    }
    return result;
  }
  range<dimensions> operator-(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] - rhs;
    }
    return result;
  }
  range<dimensions> operator*(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] * rhs;
    }
    return result;
  }
  range<dimensions> operator/(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] / rhs;
    }
    return result;
  }
  range<dimensions> operator%(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] % rhs;
    }
    return result;
  }
  range<dimensions> operator<<(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] << rhs;
    }
    return result;
  }
  range<dimensions> operator>>(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >> rhs;
    }
    return result;
  }
  range<dimensions> operator&(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] & rhs;
    }
    return result;
  }
  range<dimensions> operator|(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] | rhs;
    }
    return result;
  }
  range<dimensions> operator^(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] ^ rhs;
    }
    return result;
  }
  range<dimensions> operator&&(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] && rhs;
    }
    return result;
  }
  range<dimensions> operator||(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] || rhs;
    }
    return result;
  }
  range<dimensions> operator<(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] < rhs;
    }
    return result;
  }
  range<dimensions> operator>(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] > rhs;
    }
    return result;
  }
  range<dimensions> operator<=(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] <= rhs;
    }
    return result;
  }
  range<dimensions> operator>=(const size_t &rhs) const {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = this->common_array[i] >= rhs;
    }
    return result;
  }

  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
  range<dimensions> &operator+=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] += rhs[i];
    }
    return *this;
  }
  range<dimensions> &operator-=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] -= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator*=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] *= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator/=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] /= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator%=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] %= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator<<=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] <<= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator>>=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] >>= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator&=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] &= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator|=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] |= rhs.common_array[i];
    }
    return *this;
  }
  range<dimensions> &operator^=(const range<dimensions> &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] ^= rhs.common_array[i];
    }
    return *this;
  }

  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
  range<dimensions> &operator+=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] += rhs;
    }
    return *this;
  }
  range<dimensions> &operator-=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] -= rhs;
    }
    return *this;
  }
  range<dimensions> &operator*=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] *= rhs;
    }
    return *this;
  }
  range<dimensions> &operator/=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] /= rhs;
    }
    return *this;
  }
  range<dimensions> &operator%=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] %= rhs;
    }
    return *this;
  }
  range<dimensions> &operator<<=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] <<= rhs;
    }
    return *this;
  }
  range<dimensions> &operator>>=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] >>= rhs;
    }
    return *this;
  }
  range<dimensions> &operator&=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] &= rhs;
    }
    return *this;
  }
  range<dimensions> &operator|=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] |= rhs;
    }
    return *this;
  }
  range<dimensions> &operator^=(const size_t &rhs) {
    for (int i = 0; i < dimensions; ++i) {
      this->common_array[i] ^= rhs;
    }
    return *this;
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ^, <, >, <=, >=, &&, ||
  friend range<dimensions> operator+(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs + rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator-(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs - rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator*(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs * rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator/(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs / rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator%(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs % rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator<<(const size_t &lhs,
                                      const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs << rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator>>(const size_t &lhs,
                                      const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs >> rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator&(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs & rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator|(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs | rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator^(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs ^ rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator<(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs < rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator>(const size_t &lhs,
                                     const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs > rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator<=(const size_t &lhs,
                                      const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs <= rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator>=(const size_t &lhs,
                                      const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs >= rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator&&(const size_t &lhs,
                                      const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs && rhs.common_array[i];
    }
    return result;
  }
  friend range<dimensions> operator||(const size_t &lhs,
                                      const range<dimensions> &rhs) {
    range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result.common_array[i] = lhs || rhs.common_array[i];
    }
    return result;
  }
};
} // namespace sycl
} // namespace cl
