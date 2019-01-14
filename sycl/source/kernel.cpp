//==--------------- kernel.cpp --- SYCL kernel -----------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/kernel.hpp>

#include <CL/sycl/program.hpp>

namespace cl {
namespace sycl {

program kernel::get_program() const { return impl->get_program(); }

} // namespace sycl
} // namespace cl
