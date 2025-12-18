//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Program abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PROGRAM_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PROGRAM_H

#include "L0Kernel.h"

namespace llvm::omp::target::plugin {

class L0DeviceTy;

/// Program data to be initialized by plugin.
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

class L0ProgramBuilderTy {
  L0DeviceTy &Device;
  std::unique_ptr<MemoryBuffer> Image;
  /// Handle multiple modules within a single target image.
  llvm::SmallVector<ze_module_handle_t> Modules;

  /// Module that contains global data including device RTL.
  ze_module_handle_t GlobalModule = nullptr;

  /// Requires module link.
  bool RequiresModuleLink = false;

  /// Build a single module with the given image, build option, and format.
  Error addModule(const size_t Size, const uint8_t *Image,
                  const std::string_view BuildOption,
                  ze_module_format_t Format);

  Error linkModules();

public:
  L0ProgramBuilderTy(L0DeviceTy &Device, std::unique_ptr<MemoryBuffer> &&Image)
      : Device(Device), Image(std::move(Image)) {}
  ~L0ProgramBuilderTy() = default;

  L0DeviceTy &getL0Device() const { return Device; }
  ze_module_handle_t getGlobalModule() const { return GlobalModule; }
  llvm::SmallVector<ze_module_handle_t> &getModules() { return Modules; }

  const void *getStart() const { return Image->getBufferStart(); }
  size_t getSize() const { return Image->getBufferSize(); }

  MemoryBufferRef getMemoryBuffer() const {
    return MemoryBufferRef(*Image);
  }
  Error buildModules(const std::string_view BuildOptions);

  /// Retrieve the ELF binary for the program.
  Expected<std::unique_ptr<MemoryBuffer>> getELF();
};

/// Level Zero program that can contain multiple modules.
class L0ProgramTy : public DeviceImageTy {
  /// Handle multiple modules within a single target image.
  llvm::SmallVector<ze_module_handle_t> Modules;

  /// Map of kernel names to Modules
  std::unordered_map<std::string, ze_module_handle_t> KernelsToModuleMap;

  /// List of kernels built for this image.
  /// We need to delete them ourselves as the main library is not doing
  /// that right now.
  std::list<L0KernelTy *> Kernels;

  /// Module that contains global data including device RTL.
  ze_module_handle_t GlobalModule = nullptr;

  L0DeviceTy &getL0Device() const;

public:
  L0ProgramTy() = delete;

  L0ProgramTy(int32_t ImageId, GenericDeviceTy &Device,
              std::unique_ptr<MemoryBuffer> Image,
              ze_module_handle_t GlobalModule,
              llvm::SmallVector<ze_module_handle_t> &&Modules)
      : DeviceImageTy(ImageId, Device, std::move(Image)),
        Modules(std::move(Modules)), GlobalModule(GlobalModule) {}
  ~L0ProgramTy() = default;

  L0ProgramTy(const L0ProgramTy &other) = delete;
  L0ProgramTy(L0ProgramTy &&) = delete;
  L0ProgramTy &operator=(const L0ProgramTy &) = delete;
  L0ProgramTy &operator=(const L0ProgramTy &&) = delete;

  Error deinit();

  static L0ProgramTy &makeL0Program(DeviceImageTy &Device) {
    return static_cast<L0ProgramTy &>(Device);
  }

  /// Loads the kernels names from all modules.
  Error loadModuleKernels();

  /// Read data from the location in the device image which corresponds to the
  /// specified global variable name.
  Error readGlobalVariable(const char *Name, size_t Size, void *HostPtr);

  /// Write data to the location in the device image which corresponds to the
  /// specified global variable name.
  Error writeGlobalVariable(const char *Name, size_t Size, const void *HostPtr);

  /// Looks up an OpenMP declare target global variable with the given
  /// \p Name and \p Size in the device environment for the current device.
  /// The lookup is first done via the device offload table. If it fails,
  /// then the lookup falls back to non-OpenMP specific lookup on the device.
  Expected<void *> getOffloadVarDeviceAddr(const char *Name) const;

  /// Returns the handle of a module that contains a given Kernel name.
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
