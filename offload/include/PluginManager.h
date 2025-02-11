//===-- PluginManager.h - Plugin loading and communication API --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_PLUGIN_MANAGER_H
#define OMPTARGET_PLUGIN_MANAGER_H

#include "PluginInterface.h"

#include "DeviceImage.h"
#include "ExclusiveAccess.h"
#include "Shared/APITypes.h"
#include "Shared/Requirements.h"

#include "device.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>

using GenericPluginTy = llvm::omp::target::plugin::GenericPluginTy;

/// Struct for the data required to handle plugins
struct PluginManager {
  /// Type of the devices container. We hand out DeviceTy& to queries which are
  /// stable addresses regardless if the container changes.
  using DeviceContainerTy = llvm::SmallVector<std::unique_ptr<DeviceTy>>;

  /// Exclusive accessor type for the device container.
  using ExclusiveDevicesAccessorTy = Accessor<DeviceContainerTy>;

  PluginManager() {}

  void init();

  void deinit();

  // Register a shared library with all (compatible) RTLs.
  void registerLib(__tgt_bin_desc *Desc);

  // Unregister a shared library from all RTLs.
  void unregisterLib(__tgt_bin_desc *Desc);

  void addDeviceImage(__tgt_bin_desc &TgtBinDesc,
                      __tgt_device_image &TgtDeviceImage) {
    DeviceImages.emplace_back(
        std::make_unique<DeviceImageTy>(TgtBinDesc, TgtDeviceImage));
  }

  /// Return the device presented to the user as device \p DeviceNo if it is
  /// initialized and ready. Otherwise return an error explaining the problem.
  llvm::Expected<DeviceTy &> getDevice(uint32_t DeviceNo);

  /// Iterate over all initialized and ready devices registered with this
  /// plugin.
  auto devices(ExclusiveDevicesAccessorTy &DevicesAccessor) {
    return llvm::make_pointee_range(*DevicesAccessor);
  }

  /// Iterate over all device images registered with this plugin.
  auto deviceImages() { return llvm::make_pointee_range(DeviceImages); }

  /// Translation table retrieved from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  llvm::SmallVector<llvm::offloading::EntryTy *>
      HostEntriesBeginRegistrationOrder;

  /// Map from ptrs on the host to an entry in the Translation Table
  HostPtrToTableMapTy HostPtrToTableMap;
  std::mutex TblMapMtx; ///< For HostPtrToTableMap

  // Work around for plugins that call dlopen on shared libraries that call
  // tgt_register_lib during their initialisation. Stash the pointers in a
  // vector until the plugins are all initialised and then register them.
  bool delayRegisterLib(__tgt_bin_desc *Desc) {
    if (RTLsLoaded)
      return false;
    DelayedBinDesc.push_back(Desc);
    return true;
  }

  void registerDelayedLibraries() {
    // Only called by libomptarget constructor
    RTLsLoaded = true;
    for (auto *Desc : DelayedBinDesc)
      __tgt_register_lib(Desc);
    DelayedBinDesc.clear();
  }

  /// Return the number of usable devices.
  int getNumDevices() { return getExclusiveDevicesAccessor()->size(); }

  /// Return an exclusive handle to access the devices container.
  ExclusiveDevicesAccessorTy getExclusiveDevicesAccessor() {
    return Devices.getExclusiveAccessor();
  }

  /// Initialize \p Plugin. Returns true on success.
  bool initializePlugin(GenericPluginTy &Plugin);

  /// Initialize device \p DeviceNo of \p Plugin. Returns true on success.
  bool initializeDevice(GenericPluginTy &Plugin, int32_t DeviceId);

  /// Eagerly initialize all plugins and their devices.
  void initializeAllDevices();

  /// Iterator range for all plugins (in use or not, but always valid).
  auto plugins() { return llvm::make_pointee_range(Plugins); }

  /// Iterator range for all plugins (in use or not, but always valid).
  auto plugins() const { return llvm::make_pointee_range(Plugins); }

  /// Return the user provided requirements.
  int64_t getRequirements() const { return Requirements.getRequirements(); }

  /// Add \p Flags to the user provided requirements.
  void addRequirements(int64_t Flags) { Requirements.addRequirements(Flags); }

  /// Returns the number of plugins that are active.
  int getNumActivePlugins() const {
    int count = 0;
    for (auto &R : plugins())
      if (R.is_initialized())
        ++count;

    return count;
  }

private:
  bool RTLsLoaded = false;
  llvm::SmallVector<__tgt_bin_desc *> DelayedBinDesc;

  // List of all plugins, in use or not.
  llvm::SmallVector<std::unique_ptr<GenericPluginTy>> Plugins;

  // Mapping of plugins to the OpenMP device identifier.
  llvm::DenseMap<std::pair<const GenericPluginTy *, int32_t>, int32_t>
      DeviceIds;

  // Set of all device images currently in use.
  llvm::DenseSet<const __tgt_device_image *> UsedImages;

  /// Executable images and information extracted from the input images passed
  /// to the runtime.
  llvm::SmallVector<std::unique_ptr<DeviceImageTy>> DeviceImages;

  /// The user provided requirements.
  RequirementCollection Requirements;

  std::mutex RTLsMtx; ///< For RTLs

  /// Devices associated with plugins, accesses to the container are exclusive.
  ProtectedObj<DeviceContainerTy> Devices;

  /// References to upgraded legacy offloading entries.
  std::list<llvm::SmallVector<llvm::offloading::EntryTy, 0>> LegacyEntries;
  std::list<llvm::SmallVector<__tgt_device_image, 0>> LegacyImages;
  llvm::DenseMap<__tgt_bin_desc *, __tgt_bin_desc> UpgradedDescriptors;
  __tgt_bin_desc *upgradeLegacyEntries(__tgt_bin_desc *Desc);
};

/// Initialize the plugin manager and OpenMP runtime.
void initRuntime();

/// Deinitialize the plugin and delete it.
void deinitRuntime();

extern PluginManager *PM;

#endif // OMPTARGET_PLUGIN_MANAGER_H
