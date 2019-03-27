//===-- util.cpp - Shared SYCL runtime utilities impl ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/util.hpp>

namespace cl {
namespace sycl {
namespace detail {

Sync &Sync::getInstance() {
  // Use C++11 "magic static" idiom to implement the singleton concept
  static Sync Instance;
  return Instance;
}

} // namespace detail
} // namespace sycl
} // namespace cl
