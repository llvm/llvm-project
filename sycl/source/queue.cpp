//==-------------- queue.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  *this = queue(*std::max_element(Devs.begin(), Devs.end(), Comp), asyncHandler,
                propList);
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
