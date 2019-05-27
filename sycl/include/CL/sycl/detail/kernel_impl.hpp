//==------- kernel_impl.hpp --- SYCL kernel implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/kernel_info.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/info/info_desc.hpp>

#include <cassert>
#include <memory>

namespace cl {
namespace sycl {
// Forward declaration
class program;

namespace detail {
class program_impl;

class kernel_impl {
public:
  kernel_impl(cl_kernel ClKernel, const context &SyclContext);

  kernel_impl(cl_kernel ClKernel, const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl,
              bool IsCreatedFromSource)
      : ClKernel(ClKernel), Context(SyclContext), ProgramImpl(ProgramImpl),
        IsCreatedFromSource(IsCreatedFromSource) {}

  // Host kernel constructor
  kernel_impl(const context &SyclContext,
              std::shared_ptr<program_impl> ProgramImpl)
      : Context(SyclContext), ProgramImpl(ProgramImpl) {}

  ~kernel_impl() {
    // TODO replace CHECK_OCL_CODE_NO_EXC to CHECK_OCL_CODE and
    // TODO catch an exception and put it to list of asynchronous exceptions
    if (!is_host()) {
      CHECK_OCL_CODE_NO_EXC(clReleaseKernel(ClKernel));
    }
  }

  cl_kernel get() const {
    if (is_host()) {
      throw invalid_object_error("This instance of kernel is a host instance");
    }
    CHECK_OCL_CODE(clRetainKernel(ClKernel));
    return ClKernel;
  }

  bool is_host() const { return Context.is_host(); }

  context get_context() const { return Context; }

  program get_program() const;

  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const {
    if (is_host()) {
      // TODO implement
      assert(0 && "Not implemented");
    }
    return get_kernel_info_cl<
        typename info::param_traits<info::kernel, param>::return_type,
        param>::_(this->get());
  }

  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &Device) const {
    if (is_host()) {
      return get_kernel_work_group_info_host<param>(Device);
    }
    return get_kernel_work_group_info_cl<
        typename info::param_traits<info::kernel_work_group,
                                    param>::return_type,
        param>::_(this->get(), Device.get());
  }

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &Device) const {
    if (is_host()) {
      throw runtime_error("Sub-group feature is not supported on HOST device.");
    }
    return get_kernel_sub_group_info_cl<
        typename info::param_traits<info::kernel_sub_group, param>::return_type,
        param>::_(this->get(), Device.get());
  }

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const device &Device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          Value) const {
    if (is_host()) {
      throw runtime_error("Sub-group feature is not supported on HOST device.");
    }
    return get_kernel_sub_group_info_with_input_cl<
        typename info::param_traits<info::kernel_sub_group, param>::return_type,
        param,
        typename info::param_traits<info::kernel_sub_group,
                                    param>::input_type>::_(this->get(),
                                                           Device.get(), Value);
  }

  cl_kernel &getHandleRef() { return ClKernel; }

  bool isCreatedFromSource() const;

private:
  cl_kernel ClKernel;
  context Context;
  std::shared_ptr<program_impl> ProgramImpl;
  bool IsCreatedFromSource = true;
};

template <> context kernel_impl::get_info<info::kernel::context>() const;

template <> program kernel_impl::get_info<info::kernel::program>() const;

} // namespace detail
} // namespace sycl
} // namespace cl
