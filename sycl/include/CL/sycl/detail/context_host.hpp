//==------------- context_host.hpp - SYCL host context ---------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/context_impl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>
#include <memory>
// 4.6.2 Context class

namespace cl {
namespace sycl {
namespace detail {
class context_host : public context_impl {
public:
  context_host(const device &rhs, async_handler asyncHandler)
      : context_impl(asyncHandler), dev(rhs) {}

  cl_context get() const override {
    throw invalid_object_error("This instance of context is a host instance");
  }

  bool is_host() const override { return true; }

  platform get_platform() const override { return platform(); }

  vector_class<device> get_devices() const override {
    return vector_class<device>(1, dev);
  }

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type get_info() const;
private:
  device dev;
};
} // namespace detail
} // namespace sycl
} // namespace cl
