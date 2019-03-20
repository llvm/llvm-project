//==---------------- helpers.cpp - SYCL helpers ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/event.hpp>

#include <memory>

namespace cl {
namespace sycl {
using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;
namespace detail {
std::vector<cl_event> getOrWaitEvents(std::vector<cl::sycl::event> DepEvents,
                                      ContextImplPtr Context) {
  std::vector<cl_event> CLEvents;
  for (auto SyclEvent : DepEvents) {
    auto SyclEventImplPtr = detail::getSyclObjImpl(SyclEvent);
    if (SyclEventImplPtr->is_host() ||
        SyclEventImplPtr->getContextImpl() != Context) {
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
