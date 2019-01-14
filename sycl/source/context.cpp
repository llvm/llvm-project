//==---------------- context.cpp - SYCL context ----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>

namespace cl {
namespace sycl {
  context::context(const vector_class<device> &deviceList,
          async_handler asyncHandler) {
    if (deviceList.empty())
      throw invalid_parameter_error("First argument deviceList is empty.");

    if (deviceList[0].is_host()) {
      impl =
          std::make_shared<detail::context_host>(deviceList[0], asyncHandler);
    } else {
      impl = std::make_shared<detail::context_opencl>(deviceList, asyncHandler);
    }
  }

  context::context(cl_context clContext, async_handler asyncHandler) {
    impl = std::make_shared<detail::context_opencl>(clContext, asyncHandler);
  }
} // namespace sycl
} // namespace cl
