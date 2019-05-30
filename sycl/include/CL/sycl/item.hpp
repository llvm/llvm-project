//==------------ item.hpp --- SYCL iteration item --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/id.hpp>
#include <CL/sycl/range.hpp>
#include <stdexcept>
#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {
struct Builder;
}
template <int dimensions> struct id;
template <int dimensions> class range;
template <int dimensions = 1, bool with_offset = true> struct item {

  item() = delete;

  id<dimensions> get_id() const { return index; }

  size_t get_id(int dimension) const { return index[dimension]; }

  size_t operator[](int dimension) const { return index[dimension]; }

  range<dimensions> get_range() const { return extent; }

  size_t get_range(int dimension) const { return extent.get(dimension); }

  // only available if with_offset is true;
  template <bool W = with_offset,
            typename = typename std::enable_if<(W == true)>::type>
  id<dimensions> get_offset() const {
    return offset;
  }

  template <bool W = with_offset>
  operator typename std::enable_if<W == false, item<dimensions, true>>::type()
      const {
    return item<dimensions, true>(extent, index, offset);
  }

  /* The following member function is only available in the id class
   * specialization where: dimensions>0 and dimensions<4 */
  template <int N = dimensions,
            typename = typename std::enable_if<((N > 0) && (N < 4))>::type>
  size_t get_linear_id() const {
    if (1 == dimensions) {
      return index[0] - offset[0];
    }
    if (2 == dimensions) {
      return (index[0] - offset[0]) * extent[1] + (index[1] - offset[1]);
    }
    return ((index[0] - offset[0]) * extent[1] * extent[2]) +
           ((index[1] - offset[1]) * extent[2]) + (index[2] - offset[2]);
  }

  item<dimensions, with_offset>(const item<dimensions, with_offset> &rhs) =
      default;

  item<dimensions, with_offset>(item<dimensions, with_offset> &&rhs) = default;

  item<dimensions, with_offset> &
  operator=(const item<dimensions, with_offset> &rhs) = default;

  item<dimensions, with_offset> &
  operator=(item<dimensions, with_offset> &&rhs) = default;

  bool operator==(const item<dimensions, with_offset> &rhs) const {
    return (rhs.index == this->index) && (rhs.extent == this->extent) &&
           (rhs.offset == this->offset);
  }

  bool operator!=(const item<dimensions, with_offset> &rhs) const {
    return !((*this) == rhs);
  }

protected:
  // For call constructor inside conversion operator
  friend struct item<dimensions, false>;
  friend struct item<dimensions, true>;
  friend struct detail::Builder;

  template <size_t W = with_offset>
  item(typename std::enable_if<(W == true), const range<dimensions>>::type &R,
       const id<dimensions> &I, const id<dimensions> &O)
      : extent(R), index(I), offset(O) {}

  template <size_t W = with_offset>
  item(typename std::enable_if<(W == false), const range<dimensions>>::type &R,
       const id<dimensions> &I)
      : extent(R), index(I), offset() {}

private:
  range<dimensions> extent;
  id<dimensions> index;
  id<dimensions> offset;
};

} // namespace sycl
} // namespace cl
