//==------- kernel_impl.cpp --- SYCL kernel implementation -----------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/kernel_impl.hpp>

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/program.hpp>
#include <memory>

namespace cl {
namespace sycl {
namespace detail {

kernel_impl::kernel_impl(cl_kernel ClKernel, const context &SyclContext)
    : kernel_impl(ClKernel, SyclContext,
                  std::make_shared<program_impl>(SyclContext, ClKernel)) {}

program kernel_impl::get_program() const {
  return createSyclObjFromImpl<program>(ProgramImpl);
}

template <> context kernel_impl::get_info<info::kernel::context>() const {
  return get_context();
}

template <> program kernel_impl::get_info<info::kernel::program>() const {
  return get_program();
}

} // namespace detail
} // namespace sycl
} // namespace cl
