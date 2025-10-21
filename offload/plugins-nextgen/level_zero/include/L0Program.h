//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Program abstraction
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PROGRAM_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PROGRAM_H

#include "L0Kernel.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;

/// Program data to be initialized by plugin
struct ProgramDataTy {
  int Initialized = 0;
  int NumDevices = 0;
  int DeviceNum = -1;
  uint32_t TotalEUs = 0;
  uint32_t HWThreadsPerEU = 0;
  uintptr_t DynamicMemoryLB = 0;
  uintptr_t DynamicMemoryUB = 0;
  int DeviceType = 0;
  void *DynamicMemPool = nullptr;
  int TeamsThreadLimit = 0;
};

/// Level Zero program that can contain multiple modules.
class L0ProgramTy : public DeviceImageTy {
  /// Handle multiple modules within a single target image
  llvm::SmallVector<ze_module_handle_t> Modules;

  /// Map of kernel names to Modules
  std::unordered_map<std::string, ze_module_handle_t> KernelsToModuleMap;

  /// List of kernels built for this image
  /// We need to delete them ourselves as the main library is not doing
  /// that right now
  std::list<L0KernelTy *> Kernels;

  /// Module that contains global data including device RTL
  ze_module_handle_t GlobalModule = nullptr;

  /// Requires module link
  bool RequiresModuleLink = false;

  /// Is this module library
  bool IsLibModule = false;

  /// Build a single module with the given image, build option, and format.
  int32_t addModule(const size_t Size, const uint8_t *Image,
                    const std::string_view BuildOption,
                    ze_module_format_t Format);
  /// Read file and return the size of the binary if successful.
  size_t readFile(const char *FileName, std::vector<uint8_t> &OutFile) const;
  void replaceDriverOptsWithBackendOpts(const L0DeviceTy &Device,
                                        std::string &Options) const;

  /// Check if the image should be handled as a library module
  void setLibModule();

  L0DeviceTy &getL0Device() const;

public:
  L0ProgramTy() = delete;

  L0ProgramTy(int32_t ImageId, GenericDeviceTy &Device,
              std::unique_ptr<MemoryBuffer> Image)
      : DeviceImageTy(ImageId, Device, std::move(Image)) {}
  ~L0ProgramTy() {}

  L0ProgramTy(const L0ProgramTy &other) = delete;
  L0ProgramTy(L0ProgramTy &&) = delete;
  L0ProgramTy &operator=(const L0ProgramTy &) = delete;
  L0ProgramTy &operator=(const L0ProgramTy &&) = delete;

  Error deinit();

  static L0ProgramTy &makeL0Program(DeviceImageTy &Device) {
    return static_cast<L0ProgramTy &>(Device);
  }

  /// Build modules from the target image description
  int32_t buildModules(const std::string_view BuildOptions);

  /// Link modules stored in \p Modules.
  int32_t linkModules();

  /// Loads the kernels names from all modules
  int32_t loadModuleKernels();

  /// Read data from the location in the device image which corresponds to the
  /// specified global variable name.
  int32_t readGlobalVariable(const char *Name, size_t Size, void *HostPtr);

  /// Write data to the location in the device image which corresponds to the
  /// specified global variable name.
  int32_t writeGlobalVariable(const char *Name, size_t Size,
                              const void *HostPtr);

  /// Looks up an OpenMP declare target global variable with the given
  /// \p Name and \p Size in the device environment for the current device.
  /// The lookup is first done via the device offload table. If it fails,
  /// then the lookup falls back to non-OpenMP specific lookup on the device.
  void *getOffloadVarDeviceAddr(const char *Name) const;

  /// Returns the handle of a module that contains a given Kernel name
  ze_module_handle_t findModuleFromKernelName(const char *KernelName) const {
    auto K = KernelsToModuleMap.find(std::string(KernelName));
    if (K == KernelsToModuleMap.end())
      return nullptr;

    return K->second;
  }

  void addKernel(L0KernelTy *Kernel) { Kernels.push_back(Kernel); }
};

struct L0GlobalHandlerTy final : public GenericGlobalHandlerTy {
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override;
};

bool isValidOneOmpImage(StringRef Image, uint64_t &MajorVer,
                        uint64_t &MinorVer);
} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PROGRAM_H
