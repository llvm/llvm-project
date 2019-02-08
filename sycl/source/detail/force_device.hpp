//==---------- force_device.hpp - Forcing SYCL device ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/info/info_desc.hpp>

namespace cl {
namespace sycl {
namespace detail {

bool match_types(const info::device_type &l, const info::device_type &r);

info::device_type get_forced_type();

} // namespace detail
} // namespace sycl
} // namespace cl
