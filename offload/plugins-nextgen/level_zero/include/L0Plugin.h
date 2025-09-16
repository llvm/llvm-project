//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Plugin interface for SPIR-V/Xe machine
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AsyncQueue.h"
#include "L0Defs.h"
#include "L0Device.h"
#include "L0Memory.h"
#include "L0Options.h"
#include "L0Program.h"
#include "TLS.h"

namespace llvm::omp::target::plugin {

/// Class implementing the LevelZero specific functionalities of the plugin.
class LevelZeroPluginTy final : public GenericPluginTy {
private:
  /// Number of devices available including subdevices
  uint32_t NumDevices = 0;

  /// Context (and Driver) specific data
  std::list<L0ContextTy> ContextList;

  /// L0 device used by each OpenMP device
  using DeviceContainerTy = llvm::SmallVector<L0DeviceTy *>;
  DeviceContainerTy L0Devices;

  // Table containing per-thread information using TLS
  L0ThreadTblTy ThreadTLSTable;
  // Table containing per-thread information for each device using TLS
  L0DeviceTLSTableTy DeviceTLSTable;
  // Table containing per-thread information for each Context using TLS
  L0ContextTLSTableTy ContextTLSTable;

  /// L0 plugin global options
  static L0OptionsTy Options;

  /// Global mutex
  std::mutex GlobalMutex;

  /// Common pool of AsyncQueue
  AsyncQueuePoolTy AsyncQueuePool;

  auto &getTLS() { return ThreadTLSTable.get(); }

public:
  LevelZeroPluginTy() : GenericPluginTy(getTripleArch()) {}
  virtual ~LevelZeroPluginTy() {}

  auto &getDeviceTLS(int32_t DeviceId) { return DeviceTLSTable.get(DeviceId); }
  auto &getContextTLS(ze_context_handle_t Context) {
    return ContextTLSTable.get(Context);
  }

  static const auto &getOptions() { return Options; }

  auto &getGlobalMutex() { return GlobalMutex; }

  struct DevicesRangeTy {
    using iterator = DeviceContainerTy::iterator;

    iterator BeginIt;
    iterator EndIt;

    DevicesRangeTy(iterator BeginIt, iterator EndIt)
        : BeginIt(BeginIt), EndIt(EndIt) {}

    auto &begin() { return BeginIt; }
    auto &end() { return EndIt; }
  };

  auto getDevicesRange() {
    return DevicesRangeTy(L0Devices.begin(), L0Devices.end());
  }

  /// Clean-up routine to be invoked by the destructor or
  /// LevelZeroPluginTy::deinit.
  void closeRTL();

  /// Find L0 devices and initialize device properties.
  /// Returns number of devices reported to omptarget.
  int32_t findDevices();

  L0DeviceTy &getDeviceFromId(int32_t DeviceId) const {
    assert("Invalid device ID" && DeviceId >= 0 &&
           DeviceId < static_cast<int32_t>(L0Devices.size()));
    return *L0Devices[DeviceId];
  }

  uint32_t getNumRootDevices() const { return NumDevices; }

  AsyncQueueTy *getAsyncQueue() {
    auto *Queue = getTLS().getAsyncQueue();
    if (!Queue)
      Queue = AsyncQueuePool.get();
    return Queue;
  }

  void releaseAsyncQueue(AsyncQueueTy *Queue) {
    if (!Queue)
      return;
    Queue->reset();
    Queue->InUse = false;
    if (!getTLS().releaseAsyncQueue(Queue))
      AsyncQueuePool.release(Queue);
  }

  // Plugin interface

  Expected<int32_t> initImpl() override;
  Error deinitImpl() override;
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override;
  GenericGlobalHandlerTy *createGlobalHandler() override;
  uint16_t getMagicElfBits() const override;
  Triple::ArchType getTripleArch() const override;
  const char *getName() const override;
  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override;

  Error flushQueueImpl(omp_interop_val_t *Interop) override;
  Error syncBarrierImpl(omp_interop_val_t *Interop) override;
  Error asyncBarrierImpl(omp_interop_val_t *Interop) override;
};

} // namespace llvm::omp::target::plugin
