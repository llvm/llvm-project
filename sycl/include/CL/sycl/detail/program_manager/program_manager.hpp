//==------ program_manager.hpp --- SYCL program manager---------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/stl.hpp>

#include <map>
#include <vector>

/// This struct is a record of the device image information
struct __tgt_device_image {
  void *ImageStart;                  // Pointer to the target code start
  void *ImageEnd;                    // Pointer to the target code end
};

/// This struct is a record of all the host code that may be offloaded to a
/// target.
struct __tgt_bin_desc {
  int32_t NumDeviceImages;           // Number of device types supported
  __tgt_device_image *DeviceImages;  // Array of device images (1 per dev. type)
};

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" void __tgt_register_lib(__tgt_bin_desc *desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" void __tgt_unregister_lib(__tgt_bin_desc *desc);

// +++ }

namespace cl {
namespace sycl {
class context;
namespace detail {

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  static ProgramManager &getInstance();
  cl_program getBuiltOpenCLProgram(const context &Context);
  cl_kernel getOrCreateKernel(const context &Context, const char *KernelName);
  cl_program getClProgramFromClKernel(cl_kernel ClKernel);

  void setDeviceImages(__tgt_bin_desc *_DeviceImages) {
    // TODO thread-unsafe, see comments in __tgt_register_lib
    DeviceImages = _DeviceImages;
  }

private:
  const vector_class<char> getSpirvSource();
  void build(cl_program &ClProgram, const string_class &Options = "",
             std::vector<cl_device_id> ClDevices = std::vector<cl_device_id>());

  struct ContextLess {
    bool operator()(const context &LHS, const context &RHS) const;
  };

  ProgramManager() : DeviceImages(nullptr) {}
  ~ProgramManager() = default;
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  unique_ptr_class<vector_class<char>> m_SpirvSource;
  std::map<context, cl_program, ContextLess> m_CachedSpirvPrograms;
  std::map<cl_program, std::map<string_class, cl_kernel>> m_CachedKernels;

  /// Device executable images available in this module (.exe or .dll).
  __tgt_bin_desc *DeviceImages;
};
} // namespace detail
} // namespace sycl
} // namespace cl
