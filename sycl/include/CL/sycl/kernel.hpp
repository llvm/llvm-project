//==--------------- kernel.hpp --- SYCL kernel -----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/kernel_impl.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

namespace cl {
namespace sycl {
// Forward declaration
class program;
class context;

class kernel {
  template <class T>
  friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  kernel(cl_kernel clKernel, const context &syclContext)
      : impl(std::make_shared<detail::kernel_impl>(clKernel, syclContext)) {}

  kernel(const kernel &rhs) = default;

  kernel(kernel &&rhs) = default;

  kernel &operator=(const kernel &rhs) = default;

  kernel &operator=(kernel &&rhs) = default;

  bool operator==(const kernel &rhs) const { return impl == rhs.impl; }

  bool operator!=(const kernel &rhs) const { return !operator==(rhs); }

  cl_kernel get() const { return impl->get(); }

  bool is_host() const { return impl->is_host(); }

  context get_context() const { return impl->get_context(); }

  program get_program() const;

  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type
  get_info() const {
    return impl->get_info<param>();
  }

  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const device &dev) const {
    return impl->get_work_group_info<param>(dev);
  }

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &dev) const {
    return impl->get_sub_group_info<param>(dev);
  }

  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const device &dev,
                     typename info::param_traits<info::kernel_sub_group,
                                                 param>::input_type val) const {
    return impl->get_sub_group_info<param>(dev, val);
  }

private:
  kernel(std::shared_ptr<detail::kernel_impl> impl) : impl(impl) {}

  std::shared_ptr<detail::kernel_impl> impl;
};
} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::kernel> {
  size_t operator()(const cl::sycl::kernel &k) const {
    return hash<std::shared_ptr<cl::sycl::detail::kernel_impl>>()(
        cl::sycl::detail::getSyclObjImpl(k));
  }
};
} // namespace std
