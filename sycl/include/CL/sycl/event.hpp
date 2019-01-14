//==---------------- event.hpp --- SYCL event ------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

namespace cl {
namespace sycl {
// Forward declaration
class context;
class event {
public:
  event();

  event(cl_event clEvent, const context &syclContext);

  event(const event &rhs) = default;

  event(event &&rhs) = default;

  event &operator=(const event &rhs) = default;

  event &operator=(event &&rhs) = default;

  bool operator==(const event &rhs) const;

  bool operator!=(const event &rhs) const;

  cl_event get();

  bool is_host() const;

  void wait() const;

  // vector_class<event> get_wait_list();

  // static void wait(const vector_class<event> &eventList);

  // void wait_and_throw();

  // static void wait_and_throw(const vector_class<event> &eventList);

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type
  get_profiling_info() const;

private:
  event(std::shared_ptr<detail::event_impl> event_impl);

  std::shared_ptr<detail::event_impl> impl;

  template <class T>
  friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);

  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};

} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::event> {
  size_t operator()(const cl::sycl::event &e) const {
    return hash<std::shared_ptr<cl::sycl::detail::event_impl>>()(
        cl::sycl::detail::getSyclObjImpl(e));
  }
};
} // namespace std
