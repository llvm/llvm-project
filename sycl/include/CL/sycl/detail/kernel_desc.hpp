//==----------------------- kernel_desc.hpp --------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===////

#pragma once

#include <CL/sycl/access/access.hpp>

namespace cl {
namespace sycl {
namespace detail {

// kernel parameter kinds
enum class kernel_param_kind_t {
  kind_accessor,
  kind_std_layout, // standard layout object parameters
  kind_sampler
};

// describes a kernel parameter
struct kernel_param_desc_t {
  // parameter kind
  kernel_param_kind_t kind;
  // kind == kind_std_layout
  //   parameter size in bytes (includes padding for structs)
  // kind == kind_accessor
  //   access target; possible access targets are defined in access/access.hpp
  int info;
  // offset of the captured value of the parameter in the lambda or function
  // object
  int offset;
};

template <class KernelNameType> struct KernelInfo;

} // namespace detail
} // namespace sycl
} // namespace cl
