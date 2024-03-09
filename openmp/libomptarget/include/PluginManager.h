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

#include "DeviceImage.h"
#include "ExclusiveAccess.h"
#include "Shared/APITypes.h"
#include "Shared/PluginAPI.h"
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

struct PluginManager;

/// Plugin adaptors should be created via `PluginAdaptorTy::create` which will
/// invoke the constructor and call `PluginAdaptorTy::init`. Eventual errors are
/// reported back to the caller, otherwise a valid and initialized adaptor is
/// returned.
struct PluginAdaptorTy {
  /// Try to create a plugin adaptor from a filename.
  static llvm::Expected<std::unique_ptr<PluginAdaptorTy>>
  create(const std::string &Name);

  /// Initialize as many devices as possible for this plugin adaptor. Devices
  /// that fail to initialize are ignored.
  void initDevices(PluginManager &PM);

  bool isUsed() const { return DeviceOffset >= 0; }

  /// Return the number of devices visible to the underlying plugin.
  int32_t getNumberOfPluginDevices() const { return NumberOfPluginDevices; }

  /// Return the number of devices successfully initialized and visible to the
  /// user.
  int32_t getNumberOfUserDevices() const { return NumberOfUserDevices; }

  /// RTL index, index is the number of devices of other RTLs that were
  /// registered before, i.e. the OpenMP index of the first device to be
  /// registered with this RTL.
  int32_t DeviceOffset = -1;

  /// Name of the shared object file representing the plugin.
  std::string Name;

  /// Access to the shared object file representing the plugin.
  std::unique_ptr<llvm::sys::DynamicLibrary> LibraryHandler;

#define PLUGIN_API_HANDLE(NAME)                                                \
  using NAME##_ty = decltype(__tgt_rtl_##NAME);                                \
  NAME##_ty *NAME = nullptr;

#include "Shared/PluginAPI.inc"
#undef PLUGIN_API_HANDLE

  llvm::DenseSet<const __tgt_device_image *> UsedImages;

private:
  /// Number of devices the underling plugins sees.
  int32_t NumberOfPluginDevices = -1;

  /// Number of devices exposed to the user. This can be less than the number of
  /// devices for the plugin if some failed to initialize.
  int32_t NumberOfUserDevices = 0;

  /// Create a plugin adaptor for filename \p Name with a dynamic library \p DL.
  PluginAdaptorTy(const std::string &Name,
                  std::unique_ptr<llvm::sys::DynamicLibrary> DL);

  /// Initialize the plugin adaptor, this can fail in which case the adaptor is
  /// useless.
  llvm::Error init();
};

/// Struct for the data required to handle plugins
struct PluginManager {
  /// Type of the devices container. We hand out DeviceTy& to queries which are
  /// stable addresses regardless if the container changes.
  using DeviceContainerTy = llvm::SmallVector<std::unique_ptr<DeviceTy>>;

  /// Exclusive accessor type for the device container.
  using ExclusiveDevicesAccessorTy = Accessor<DeviceContainerTy>;

  PluginManager() {}

  void init();

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

  /// Translation table retreived from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  llvm::SmallVector<__tgt_offload_entry *> HostEntriesBeginRegistrationOrder;

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

  int getNumUsedPlugins() const {
    int NCI = 0;
    for (auto &P : PluginAdaptors)
      NCI += P->isUsed();
    return NCI;
  }

  // Initialize all plugins.
  void initAllPlugins();

  /// Iterator range for all plugin adaptors (in use or not, but always valid).
  auto pluginAdaptors() { return llvm::make_pointee_range(PluginAdaptors); }

  /// Return the user provided requirements.
  int64_t getRequirements() const { return Requirements.getRequirements(); }

  /// Add \p Flags to the user provided requirements.
  void addRequirements(int64_t Flags) { Requirements.addRequirements(Flags); }

private:
  bool RTLsLoaded = false;
  llvm::SmallVector<__tgt_bin_desc *> DelayedBinDesc;

  // List of all plugin adaptors, in use or not.
  llvm::SmallVector<std::unique_ptr<PluginAdaptorTy>> PluginAdaptors;

  /// Executable images and information extracted from the input images passed
  /// to the runtime.
  llvm::SmallVector<std::unique_ptr<DeviceImageTy>> DeviceImages;

  /// The user provided requirements.
  RequirementCollection Requirements;

  std::mutex RTLsMtx; ///< For RTLs

  /// Devices associated with plugins, accesses to the container are exclusive.
  ProtectedObj<DeviceContainerTy> Devices;
};

/// Initialize the plugin manager and OpenMP runtime.
void initRuntime();

/// Deinitialize the plugin and delete it.
void deinitRuntime();

extern PluginManager *PM;

#endif // OMPTARGET_PLUGIN_MANAGER_H
