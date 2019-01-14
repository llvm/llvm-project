//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/event.hpp>

namespace cl {
namespace sycl {
namespace detail {

std::vector<cl_event>
getOrWaitEvents(std::vector<cl::sycl::event> DepEvents,
                cl::sycl::context Context) {
  std::vector<cl_event> CLEvents;
  for (auto SyclEvent : DepEvents) {
    auto SyclEventImplPtr = detail::getSyclObjImpl(SyclEvent);
    // TODO: Add check that contexts are equal.
    if (SyclEventImplPtr->is_host()) {
      SyclEventImplPtr->waitInternal();
    } else {
      CLEvents.push_back(SyclEventImplPtr->getHandleRef());
    }
  }
  return CLEvents;
}

void waitEvents(std::vector<cl::sycl::event> DepEvents) {
  for (auto SyclEvent : DepEvents) {
    detail::getSyclObjImpl(SyclEvent)->waitInternal();
  }
}

} // namespace detail
} // namespace sycl
} // namespace cl
