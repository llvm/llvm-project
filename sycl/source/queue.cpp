//==-------------- queue.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/queue.hpp>
#include <algorithm>
namespace cl {
namespace sycl {
queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList) {

  const vector_class<device> Devs = syclContext.get_devices();

  auto Comp = [&deviceSelector](const device &d1, const device &d2) {
    return deviceSelector(d1) < deviceSelector(d2);
  };

  const device &syclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);
  impl = std::make_shared<detail::queue_impl>(syclDevice, syclContext,
                                              asyncHandler, propList);
}

queue::queue(const device &syclDevice, const async_handler &asyncHandler,
             const property_list &propList) {
  impl =
      std::make_shared<detail::queue_impl>(syclDevice, asyncHandler, propList);
}

queue::queue(cl_command_queue clQueue, const context &syclContext,
             const async_handler &asyncHandler) {
  impl =
      std::make_shared<detail::queue_impl>(clQueue, syclContext, asyncHandler);
}

} // namespace sycl
} // namespace cl
