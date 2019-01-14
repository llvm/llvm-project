//==------- common_info.hpp ----- Common SYCL info methods------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/stl.hpp>

namespace cl {
namespace sycl {
namespace detail {

vector_class<string_class> split_string(const string_class &str,
                                        char delimeter);

} // namespace detail
} // namespace sycl
} // namespace cl
