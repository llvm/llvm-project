//==-------- platform_opencl.hpp - SYCL OpenCL platform --------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/platform_impl.hpp>

// 4.6.2 Platform class for opencl platform
namespace cl {
namespace sycl {

// Forward declaration
class device_selector;
class device;

namespace detail {
// TODO: implement parameters treatment
class platform_opencl : public platform_impl {
public:
  platform_opencl(cl_platform_id platform_id) : id(platform_id) {}

  vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all) const override;

  bool has_extension(const string_class &extension_name) const override {
    string_class all_extension_names =
        get_platform_info_cl<string_class, info::platform::extensions>::_(id);
    return (all_extension_names.find(extension_name) != std::string::npos);
  }

  cl_platform_id get() const override { return id; }

  bool is_host() const override { return false; }

private:
  cl_platform_id id = 0;
}; // class platform_opencl
} // namespace detail
} // namespace sycl
} // namespace cl
