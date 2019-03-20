//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.9.2 Exception Class Interface

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/stl.hpp>
#include <exception>

namespace cl {
namespace sycl {

class context;

struct exception {
  exception() = default;

  const char *what() const noexcept { return msg.c_str(); }
  bool has_context() const;
  context get_context() const;
  cl_int get_cl_code() const;

private:
  std::string msg = "Message not specified";
  cl_int cl_err = CL_SUCCESS;
  shared_ptr_class<context> Context;

protected:
  exception(const char *msg, int cl_err = CL_SUCCESS,
            shared_ptr_class<context> Context = nullptr)
      : msg(std::string(msg) + " " +
            ((cl_err == CL_SUCCESS) ? "" : OCL_CODE_TO_STR(cl_err))),
        cl_err(cl_err), Context(Context) {}
};

// Forward declaration
namespace detail { class queue_impl; }

class exception_list : private vector_class<exception_ptr_class> {
  using list_t = vector_class<exception_ptr_class>;

public:
  using value_type = exception_ptr_class;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = ::size_t;
  using iterator = list_t::const_iterator;
  using const_iterator = list_t::const_iterator;

  using vector_class<exception_ptr_class>::size;

  /** first asynchronous exception */
  using vector_class<exception_ptr_class>::begin;

  /** refer to past-the-end last asynchronous exception */
  using vector_class<exception_ptr_class>::end;

  friend class detail::queue_impl;
};

using async_handler = function_class<void(cl::sycl::exception_list)>;

class runtime_error : public exception {
public:
  runtime_error(const char *str, cl_int err = CL_SUCCESS)
      : exception(str, err) {}
};
class kernel_error : public runtime_error {
  using runtime_error::runtime_error;
};
class accessor_error : public runtime_error {};
class nd_range_error : public runtime_error {};
class event_error : public runtime_error {};
class invalid_parameter_error : public runtime_error {
  using runtime_error::runtime_error;
};
class device_error : public exception {
public:
  device_error(const char *str, cl_int err = CL_SUCCESS)
      : exception(str, err) {}
  device_error() : device_error("") {}
};
class compile_program_error : public device_error {
  using device_error::device_error;
};
class link_program_error : public device_error {};
class invalid_object_error : public device_error {
  using device_error::device_error;
};
class memory_allocation_error : public device_error {};
class platform_error : public device_error {};
class profiling_error : public device_error {};
class feature_not_supported : public device_error {};

} // namespace sycl
} // namespace cl
