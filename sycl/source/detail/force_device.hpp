//==---------- force_device.hpp - Forcing SYCL device ----------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
