//==---------------- exception.hpp - SYCL exception ------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

class exception_list {
  using list_t = vector_class<exception_ptr_class>;
  list_t list;

public:
  using value_type = exception_ptr_class;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = ::size_t;
  using iterator = list_t::const_iterator;
  using const_iterator = list_t::const_iterator;

  ::size_t size() const { return list.size(); }

  void clear() noexcept {
    list.clear();
  }

  void push_back(const_reference value) {
    list.push_back(value);
  }

  void push_back(value_type&& value) {
    list.push_back(std::move(value));
  }

  /** first asynchronous exception */
  iterator begin() const { return list.begin(); }
  /** refer to past-the-end last asynchronous exception */
  iterator end() const { return list.end(); }

  bool operator==(const exception_list &rhs) const { return list == rhs.list; }

  bool operator!=(const exception_list &rhs) const { return !(*this == rhs); }
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
