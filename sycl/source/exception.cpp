//==---------------- exception.cpp - SYCL exception ------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// 4.9.2 Exception Class Interface
#include <CL/sycl/context.hpp>
#include <CL/sycl/exception.hpp>
#include <exception>

namespace cl {
namespace sycl {

bool exception::has_context() const { return (Context != nullptr); }

context exception::get_context() const {
  if (!has_context())
    throw invalid_object_error();

  return *Context;
}

cl_int exception::get_cl_code() const { return cl_err; }
} // namespace sycl
} // namespace cl
