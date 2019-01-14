//==-------- array.hpp --- SYCL common iteration object ---------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/exception.hpp>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace cl {
namespace sycl {
template <int dimensions> struct id;
template <int dimensions> class range;
namespace detail {

template <int dimensions = 1> class array {
public:
  INLINE_IF_DEVICE array() : common_array{0} {}

  /* The following constructor is only available in the array struct
   * specialization where: dimensions==1 */
  template <int N = dimensions> INLINE_IF_DEVICE
  array(typename std::enable_if<(N == 1), size_t>::type dim0)
      : common_array{dim0} {}

  /* The following constructor is only available in the array struct
   * specialization where: dimensions==2 */
  template <int N = dimensions> INLINE_IF_DEVICE
  array(typename std::enable_if<(N == 2), size_t>::type dim0, size_t dim1)
      : common_array{dim0, dim1} {}

  /* The following constructor is only available in the array struct
   * specialization where: dimensions==3 */
  template <int N = dimensions> INLINE_IF_DEVICE
  array(typename std::enable_if<(N == 3), size_t>::type dim0, size_t dim1,
        size_t dim2)
      : common_array{dim0, dim1, dim2} {}

  // Conversion operators to derived classes
  INLINE_IF_DEVICE operator cl::sycl::id<dimensions>() const {
    cl::sycl::id<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = common_array[i];
    }
    return result;
  }

  INLINE_IF_DEVICE operator cl::sycl::range<dimensions>() const {
    cl::sycl::range<dimensions> result;
    for (int i = 0; i < dimensions; ++i) {
      result[i] = common_array[i];
    }
    return result;
  }

  INLINE_IF_DEVICE size_t get(int dimension) const {
    check_dimension(dimension);
    return common_array[dimension];
  }

  INLINE_IF_DEVICE size_t &operator[](int dimension) {
    check_dimension(dimension);
    return common_array[dimension];
  }

  INLINE_IF_DEVICE size_t operator[](int dimension) const {
    check_dimension(dimension);
    return common_array[dimension];
  }

  INLINE_IF_DEVICE array(const array<dimensions> &rhs) = default;
  INLINE_IF_DEVICE array(array<dimensions> &&rhs) = default;
  INLINE_IF_DEVICE array<dimensions> &operator=(const array<dimensions> &rhs) = default;
  INLINE_IF_DEVICE array<dimensions> &operator=(array<dimensions> &&rhs) = default;

  // Returns true iff all elements in 'this' are equal to
  // the corresponding elements in 'rhs'.
  INLINE_IF_DEVICE bool operator==(const array<dimensions> &rhs) const {
    for (int i = 0; i < dimensions; ++i) {
      if (this->common_array[i] != rhs.common_array[i]) {
        return false;
      }
    }
    return true;
  }

  // Returns true iff there is at least one element in 'this'
  // which is not equal to the corresponding element in 'rhs'.
  INLINE_IF_DEVICE bool operator!=(const array<dimensions> &rhs) const {
    for (int i = 0; i < dimensions; ++i) {
      if (this->common_array[i] != rhs.common_array[i]) {
        return true;
      }
    }
    return false;
  }

protected:
  size_t common_array[dimensions];
  ALWAYS_INLINE void check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension >= dimensions || dimension < 0) {
      throw cl::sycl::invalid_parameter_error("Index out of range");
    }
#endif
  }
};

} // namespace detail
} // namespace sycl
} // namespace cl
