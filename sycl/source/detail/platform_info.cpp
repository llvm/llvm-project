//==----------- platform_info.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/platform_info.hpp>

namespace cl {
namespace sycl {
namespace detail {

template <> string_class get_platform_info_host<info::platform::profile>() {
  return "FULL PROFILE";
}

template <> string_class get_platform_info_host<info::platform::version>() {
  return "1.2";
}

template <> string_class get_platform_info_host<info::platform::name>() {
  return "SYCL host platform";
}

template <> string_class get_platform_info_host<info::platform::vendor>() {
  return "";
}

template <>
vector_class<string_class>
get_platform_info_host<info::platform::extensions>() {
  // TODO update when appropriate
  return {};
}

} // namespace detail
} // namespace sycl
} // namespace cl
