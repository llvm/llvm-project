//==----------- platform_info.cpp -----------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
