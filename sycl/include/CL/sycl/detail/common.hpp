//==---------- common.hpp ----- Common declarations ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Suppress a compiler warning about undefined CL_TARGET_OPENCL_VERSION
// Khronos ICD supports only latest OpenCL version
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_intel.h>
#include <string>

#include <type_traits>

const char *stringifyErrorCode(cl_int error);

#define OCL_CODE_TO_STR(code)                                                  \
  std::string(std::to_string(code) + " (" + stringifyErrorCode(code) + ")")

#define STRINGIFY_LINE_HELP(s) #s
#define STRINGIFY_LINE(s) STRINGIFY_LINE_HELP(s)

#define OCL_ERROR_REPORT                                                       \
  "OpenCL API failed. " __FILE__                                               \
  ":" STRINGIFY_LINE(__LINE__) ": "                                            \
                               "OpenCL API returns: "

#ifndef SYCL_SUPPRESS_OCL_ERROR_REPORT
#include <iostream>
#define REPORT_OCL_ERR_TO_STREAM(code)                                         \
  if (code != CL_SUCCESS) {                                                    \
    std::cerr << OCL_ERROR_REPORT << OCL_CODE_TO_STR(code) << std::endl;       \
  }
#endif

#ifndef SYCL_SUPPRESS_EXCEPTIONS
#include <CL/sycl/exception.hpp>

#define REPORT_OCL_ERR_TO_EXC(code, exc)                                       \
  if (code != CL_SUCCESS) {                                                    \
    std::string errorMessage(OCL_ERROR_REPORT + OCL_CODE_TO_STR(code));        \
    std::cerr << errorMessage << std::endl;                                    \
    throw exc(errorMessage.c_str(), (code));                                   \
  }
#define REPORT_OCL_ERR_TO_EXC_THROW(code, exc) REPORT_OCL_ERR_TO_EXC(code, exc)
#define REPORT_OCL_ERR_TO_EXC_BASE(code)                                       \
  REPORT_OCL_ERR_TO_EXC(code, cl::sycl::runtime_error)
#else
#define REPORT_OCL_ERR_TO_EXC_BASE(code) REPORT_OCL_ERR_TO_STREAM(code)
#endif

#ifdef SYCL_SUPPRESS_OCL_ERROR_REPORT
#define CHECK_OCL_CODE(X) (void)(X)
#define CHECK_OCL_CODE_THROW(X, EXC) (void)(X)
#define CHECK_OCL_CODE_NO_EXC(X) (void)(X)
#else
#define CHECK_OCL_CODE(X) REPORT_OCL_ERR_TO_EXC_BASE(X)
#define CHECK_OCL_CODE_THROW(X, EXC) REPORT_OCL_ERR_TO_EXC_THROW(X, EXC)
#define CHECK_OCL_CODE_NO_EXC(X) REPORT_OCL_ERR_TO_STREAM(X)
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define ALWAYS_INLINE
#endif

namespace cl {
namespace sycl {
namespace detail {
// Helper function for extracting implementation from SYCL's interface objects.
// Note! This function relies on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from this function.
template <class T> decltype(T::impl) getSyclObjImpl(const T &SyclObject) {
  return SyclObject.impl;
}

// Returns the raw pointer to the impl object of given face object. The caller
// must make sure the returned pointer is not captured in a field or otherwise
// stored - i.e. must live only as on-stack value.
template <class T>
typename std::add_pointer<typename decltype(T::impl)::element_type>::type
getRawSyclObjImpl(const T &SyclObject) {
  return SyclObject.impl.get();
}

// Helper function for creation SYCL interface objects from implementations.
// Note! This function relies on the fact that all SYCL interface classes
// contain "impl" field that points to implementation object. "impl" field
// should be accessible from this function.
template <class T> T createSyclObjFromImpl(decltype(T::impl) ImplObj) {
  return T(ImplObj);
}

#ifdef __SYCL_DEVICE_ONLY__
// The flag type for passing flag arguments to barrier(), mem_fence(),
// read_mem_fence(), and write_mem_fence() functions.
typedef uint cl_mem_fence_flags;

const cl_mem_fence_flags CLK_LOCAL_MEM_FENCE   = 0x01;
const cl_mem_fence_flags CLK_GLOBAL_MEM_FENCE  = 0x02;
const cl_mem_fence_flags CLK_CHANNEL_MEM_FENCE = 0x04;
#endif // __SYCL_DEVICE_ONLY__

} // namespace detail
} // namespace sycl
} // namespace cl
