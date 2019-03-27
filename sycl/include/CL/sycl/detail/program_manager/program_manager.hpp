//==------ program_manager.hpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/cnri.h>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/stl.hpp>

#include <map>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" void __tgt_register_lib(cnri_bin_desc *desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" void __tgt_unregister_lib(cnri_bin_desc *desc);

// +++ }

namespace cl {
namespace sycl {
class context;
namespace detail {

using DeviceImage = cnri_device_image;

// Custom deleter for the DeviceImage. Must only be called for "orphan" images
// allocated by the runtime. Those Images which are part of binaries must not
// be attempted to de-allocate.
struct ImageDeleter;

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // Returns the single instance of the program manager for the entire process.
  // Can only be called after staticInit is done.
  static ProgramManager &getInstance();
  cl_program createOpenCLProgram(OSModuleHandle M, const context &Context,
                                 DeviceImage **I = nullptr) {
    return loadProgram(M, Context, I);
  }
  cl_program getBuiltOpenCLProgram(OSModuleHandle M, const context &Context);
  cl_kernel getOrCreateKernel(OSModuleHandle M, const context &Context,
                              const string_class &KernelName);
  cl_program getClProgramFromClKernel(cl_kernel ClKernel);

  void addImages(cnri_bin_desc *DeviceImages);
  void debugDumpBinaryImages() const;
  void debugDumpBinaryImage(const DeviceImage *Img) const;

private:
  cnri_program loadProgram(OSModuleHandle M, const context &Context,
                           DeviceImage **I = nullptr);
  void build(cl_program &ClProgram, const string_class &Options = "",
             std::vector<cl_device_id> ClDevices = std::vector<cl_device_id>());

  struct ContextAndModuleLess {
    bool operator()(const std::pair<context, OSModuleHandle> &LHS,
                    const std::pair<context, OSModuleHandle> &RHS) const;
  };

  ProgramManager() = default;
  ~ProgramManager() = default;
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  std::map<std::pair<context, OSModuleHandle>, cl_program, ContextAndModuleLess>
      m_CachedSpirvPrograms;
  std::map<cl_program, std::map<string_class, cl_kernel>> m_CachedKernels;

  /// Keeps all available device executable images added via \ref addImages.
  /// Organizes the images as a map from a module handle (.exe .dll) to the
  /// vector of images coming from the module.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::map<OSModuleHandle, std::unique_ptr<std::vector<DeviceImage *>>>
      m_DeviceImages;
  /// Keeps device images not bound to a particular module. Program manager
  /// allocated memory for these images, so they are auto-freed in destructor.
  /// No image can out-live the Program manager.
  std::vector<std::unique_ptr<DeviceImage, ImageDeleter>> m_OrphanDeviceImages;
};
} // namespace detail
} // namespace sycl
} // namespace cl
