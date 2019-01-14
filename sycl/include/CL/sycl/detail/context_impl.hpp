//==---------------- context.hpp - SYCL context ----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/stl.hpp>
// 4.6.2 Context class

namespace cl {
namespace sycl {
// Forward declaration
class platform;
class device;
namespace detail {
template <info::context param> struct get_context_info_cl {
  using RetType =
      typename info::param_traits<info::context, param>::return_type;

  static RetType _(cl_context ctx) {
    RetType Result = 0;
    // TODO catch an exception and put it to list of asynchronous exceptions
    CHECK_OCL_CODE(clGetContextInfo(ctx, cl_context_info(param), sizeof(Result),
                                    &Result, nullptr));
    return Result;
  }
};

class context_impl {
public:
  context_impl(async_handler asyncHandler) : m_AsyncHandler(asyncHandler) {}

  template <info::context param>
  inline typename info::param_traits<info::context, param>::return_type
  get_info() const;

  const async_handler& get_async_handler() const { return m_AsyncHandler; }

  virtual cl_context get() const = 0;

  virtual bool is_host() const = 0;

  virtual platform get_platform() const = 0;

  virtual vector_class<device> get_devices() const = 0;

  virtual ~context_impl() = default;

private:
  async_handler m_AsyncHandler;
};
template <>
inline typename info::param_traits<info::context,
                                   info::context::reference_count>::return_type
context_impl::get_info<info::context::reference_count>() const {
  if (is_host()) {
    return 0;
  }
  return get_context_info_cl<info::context::reference_count>::_(this->get());
}
template <>
inline typename info::param_traits<info::context,
                                   info::context::platform>::return_type
context_impl::get_info<info::context::platform>() const {
  return get_platform();
}
template <>
inline typename info::param_traits<info::context,
                                   info::context::devices>::return_type
context_impl::get_info<info::context::devices>() const {
  return get_devices();
}

} // namespace detail
} // namespace sycl
} // namespace cl
