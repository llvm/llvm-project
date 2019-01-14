//==---------------- helpers.hpp - SYCL helpers ----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cl {
namespace sycl {
class context;
class event;
template <int dimensions, bool with_offset> class item;
template <int dimensions> class group;
template <int dimensions> class range;
template <int dimensions> class id;
template <int dimensions> class nd_item;
namespace detail {

// The function returns list of events that can be passed to OpenCL API as
// dependency list and waits for others.
std::vector<cl_event>
getOrWaitEvents(std::vector<cl::sycl::event> DepEvents,
                cl::sycl::context Context);

void waitEvents(std::vector<cl::sycl::event> DepEvents);

struct Builder {
  Builder() = delete;
  template <int dimensions>
  static group<dimensions> createGroup(const cl::sycl::range<dimensions> &G,
                                       const cl::sycl::range<dimensions> &L,
                                       const cl::sycl::id<dimensions> &I) {
    return cl::sycl::group<dimensions>(G, L, I);
  }

  template <int dimensions, bool with_offset>
  static item<dimensions, with_offset> createItem(
      typename std::enable_if<(with_offset == true),
                              const cl::sycl::range<dimensions>>::type &R,
      const cl::sycl::id<dimensions> &I, const cl::sycl::id<dimensions> &O) {
    return cl::sycl::item<dimensions, with_offset>(R, I, O);
  }

  template <int dimensions, bool with_offset>
  static item<dimensions, with_offset> createItem(
      typename std::enable_if<(with_offset == false),
                              const cl::sycl::range<dimensions>>::type &R,
      const cl::sycl::id<dimensions> &I) {
    return cl::sycl::item<dimensions, with_offset>(R, I);
  }

  template <int dimensions>
  static nd_item<dimensions>
  createNDItem(const cl::sycl::item<dimensions, true> &GL,
               const cl::sycl::item<dimensions, false> &L,
               const cl::sycl::group<dimensions> &GR) {
    return cl::sycl::nd_item<dimensions>(GL, L, GR);
  }
};

} // namespace detail
} // namespace sycl
} // namespace cl
