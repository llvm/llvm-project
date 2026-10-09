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

#include "OffloadAPI.h"
#include "PluginInterface.h"

#include "DeviceImage.h"
#include "ExclusiveAccess.h"
#include "OmpAccError.h"
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

  /// Initialize device \p DeviceHandle as on OpenMP device. Returns true on
  /// success.
  bool initializeDevice(ol_device_handle_t DeviceHandle);

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
  llvm::SmallVector<GenericPluginTy *> Plugins;

  // Mapping of device handles to the OpenMP device identifier.
  llvm::DenseMap<ol_device_handle_t, int32_t> DeviceIds;

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

  /// Map global data and execute pending ctors.
  int loadImagesOntoDevice(DeviceTy &Device);

  /// Register the image \p Img from \p Desc on the compatible device
  /// \p DeviceHandle, unless the device is already in \p UsedDevices. Returns
  /// true if the image was registered.
  bool
  registerImageOnDevice(ol_device_handle_t DeviceHandle, __tgt_bin_desc *Desc,
                        __tgt_device_image *Img,
                        llvm::SmallVectorImpl<ol_device_handle_t> &UsedDevices);
};

namespace llvm::omp::target::helpers {
// Helper functions to iterate over different elements provided by liboffload.
template <typename ElemTy, typename IterateFn, typename CallbackTy>
ol_result_t iterate(IterateFn Func, CallbackTy Callback, void *UserData) {
  struct {
    CallbackTy *Callback;
    void *UserData;
  } WrapperData = {&Callback, UserData};
  auto Wrapper = [](ElemTy Elem, void *UserData) -> bool {
    auto *Unwrapped = static_cast<decltype(WrapperData) *>(UserData);
    (*Unwrapped->Callback)(Elem, Unwrapped->UserData);
    return true;
  };
  return Func(Wrapper, &WrapperData);
}

template <typename ElemTy, typename IterateFn, typename CallbackTy>
ol_result_t iterate(IterateFn Func, CallbackTy Callback) {
  auto Wrapper = [](ElemTy Elem, void *UserData) -> bool {
    auto *Unwrapped = static_cast<CallbackTy *>(UserData);
    (*Unwrapped)(Elem);
    return true;
  };
  return Func(Wrapper, reinterpret_cast<void *>(&Callback));
}

inline llvm::Error iterateCheck(ol_result_t Result, llvm::StringRef Message) {
  if (Result)
    return llvm::omp::target::error::createError(
        llvm::omp::target::error::ErrorCode::BackendFailure, "%s : %s",
        Message.str().c_str(), Result->Details);
  return llvm::Error::success();
}

// Iterate platforms
template <typename CallbackTy>
llvm::Error iteratePlatforms(CallbackTy Callback, void *UserData) {
  return iterateCheck(
      iterate<ol_platform_handle_t>(olIteratePlatforms, Callback, UserData),
      "Failed to iterate platforms");
}
template <typename CallbackTy>
llvm::Error iteratePlatforms(CallbackTy Callback) {
  return iterateCheck(
      iterate<ol_platform_handle_t>(olIteratePlatforms, Callback),
      "Failed to iterate platforms");
}

// Iterate devices
template <typename CallbackTy>
llvm::Error iterateDevices(CallbackTy Callback, void *UserData) {
  return iterateCheck(
      iterate<ol_device_handle_t>(olIterateDevices, Callback, UserData),
      "Failed to iterate devices");
}
template <typename CallbackTy> llvm::Error iterateDevices(CallbackTy Callback) {
  return iterateCheck(iterate<ol_device_handle_t>(olIterateDevices, Callback),
                      "Failed to iterate devices");
}

} // namespace llvm::omp::target::helpers

#endif // OMPTARGET_PLUGIN_MANAGER_H
