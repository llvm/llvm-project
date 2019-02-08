//==------- common_info.hpp ----- Common SYCL info methods------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
