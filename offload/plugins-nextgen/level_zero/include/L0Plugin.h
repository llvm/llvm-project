//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Plugin interface for SPIR-V/Xe machine.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PLUGIN_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PLUGIN_H

#include "L0Context.h"
#include "L0Device.h"
#include "L0Memory.h"
#include "L0Options.h"
#include "L0Program.h"
#include "L0Queue.h"

namespace llvm::omp::target::plugin {

/// Class implementing the LevelZero specific functionalities of the plugin.
class LevelZeroPluginTy final : public GenericPluginTy {
private:
  struct DeviceInfoTy {
    L0DeviceIdTy Id;
    L0DriverTy *Driver;
    bool isRoot() const { return Id.SubId < 0 && Id.CCSId < 0; }
  };
  llvm::SmallVector<DeviceInfoTy> DetectedDevices;

  /// Driver-specific data.
  std::list<L0DriverTy> DriverList;

  /// L0 plugin options.
  L0OptionsTy Options;

  /// Find L0 devices and initialize device properties.
  /// Returns number of devices reported to omptarget.
  Expected<int32_t> findDevices();

public:
  LevelZeroPluginTy() : GenericPluginTy(getTripleArch()) {}
  virtual ~LevelZeroPluginTy() = default;

  const L0OptionsTy &getOptions() { return Options; }

  const L0DeviceTy &getDeviceFromId(int32_t DeviceId) const {
    return static_cast<const L0DeviceTy &>(getDevice(DeviceId));
  }
  L0DeviceTy &getDeviceFromId(int32_t DeviceId) {
    return static_cast<L0DeviceTy &>(getDevice(DeviceId));
  }

  // Plugin interface.
  Expected<int32_t> initImpl() override;
  Error deinitImpl() override;
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override;
  GenericGlobalHandlerTy *createGlobalHandler() override;

  Expected<std::unique_ptr<PluginContextTy>>
  createPluginContext(llvm::ArrayRef<GenericDeviceTy *> Devices) override;

  uint16_t getMagicElfBits() const override { return ELF::EM_INTELGT; }
  Triple::ArchType getTripleArch() const override { return Triple::spirv64; }
  const char *getName() const override { return GETNAME(TARGET_NAME); }

  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    // ELF images are not supported. Images must be SPIR-V or OffloadBinary.
    return false;
  }

  Error flushQueueImpl(omp_interop_val_t *Interop) override;
  Error syncBarrierImpl(omp_interop_val_t *Interop) override;
  Error asyncBarrierImpl(omp_interop_val_t *Interop) override;

  Expected<bool> isImageCompatible(StringRef Image) const override;
};

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0PLUGIN_H
