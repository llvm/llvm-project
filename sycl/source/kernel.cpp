//==--------------- kernel.cpp --- SYCL kernel -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/kernel.hpp>

#include <CL/sycl/program.hpp>

namespace cl {
namespace sycl {

program kernel::get_program() const { return impl->get_program(); }

} // namespace sycl
} // namespace cl
