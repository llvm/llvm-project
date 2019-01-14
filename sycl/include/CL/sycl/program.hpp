//==--------------- program.hpp --- SYCL program ---------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/program_impl.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

namespace cl {
namespace sycl {

class context;
class device;
class kernel;

class program {
  template <class T>
  friend decltype(T::impl) detail::getSyclObjImpl(const T &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  program() = delete;

  explicit program(const context &context)
      : impl(std::make_shared<detail::program_impl>(context)) {}

  program(const context &context, vector_class<device> deviceList)
      : impl(std::make_shared<detail::program_impl>(context, deviceList)) {}

  program(vector_class<program> programList, string_class linkOptions = "") {
    std::vector<std::shared_ptr<detail::program_impl>> impls;
    for (auto &x : programList) {
      impls.push_back(detail::getSyclObjImpl(x));
    }
    impl = std::make_shared<detail::program_impl>(impls, linkOptions);
  }

  program(const context &context, cl_program clProgram)
      : impl(std::make_shared<detail::program_impl>(context, clProgram)) {}

  program(const program &rhs) = default;

  program(program &&rhs) = default;

  program &operator=(const program &rhs) = default;

  program &operator=(program &&rhs) = default;

  bool operator==(const program &rhs) const { return impl == rhs.impl; }

  bool operator!=(const program &rhs) const { return !operator==(rhs); }

  cl_program get() const { return impl->get(); }

  bool is_host() const { return impl->is_host(); }

  template <typename kernelT>
  void compile_with_kernel_type(string_class compileOptions = "") {
    impl->compile_with_kernel_type<kernelT>(compileOptions);
  }

  void compile_with_source(string_class kernelSource,
                           string_class compileOptions = "") {
    impl->compile_with_source(kernelSource, compileOptions);
  }

  template <typename kernelT>
  void build_with_kernel_type(string_class buildOptions = "") {
    impl->build_with_kernel_type<kernelT>(buildOptions);
  }

  void build_with_source(string_class kernelSource,
                         string_class buildOptions = "") {
    impl->build_with_source(kernelSource, buildOptions);
  }

  void link(string_class linkOptions = "") { impl->link(linkOptions); }

  template <typename kernelT> bool has_kernel() const {
    return impl->has_kernel<kernelT>();
  }

  bool has_kernel(string_class kernelName) const {
    return impl->has_kernel(kernelName);
  }

  template <typename kernelT> kernel get_kernel() const {
    return impl->get_kernel<kernelT>(impl);
  }

  kernel get_kernel(string_class kernelName) const {
    return impl->get_kernel(kernelName, impl);
  }

  template <info::program param>
  typename info::param_traits<info::program, param>::return_type
  get_info() const {
    return impl->get_info<param>();
  }

  vector_class<vector_class<char>> get_binaries() const {
    return impl->get_binaries();
  }

  context get_context() const { return impl->get_context(); }

  vector_class<device> get_devices() const { return impl->get_devices(); }

  string_class get_compile_options() const {
    return impl->get_compile_options();
  }

  string_class get_link_options() const { return impl->get_link_options(); }

  string_class get_build_options() const { return impl->get_build_options(); }

  program_state get_state() const { return impl->get_state(); }

private:
  program(std::shared_ptr<detail::program_impl> impl) : impl(impl) {}

  std::shared_ptr<detail::program_impl> impl;
};
} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::program> {
  size_t operator()(const cl::sycl::program &prg) const {
    return hash<std::shared_ptr<cl::sycl::detail::program_impl>>()(
        cl::sycl::detail::getSyclObjImpl(prg));
  }
};
} // namespace std
