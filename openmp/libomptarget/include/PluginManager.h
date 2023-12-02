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

#include "Shared/APITypes.h"
#include "Shared/PluginAPI.h"

#include "device.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"

#include <list>
#include <mutex>

// Forward declarations.
struct DeviceTy;
struct __tgt_bin_desc;

struct PluginAdaptorTy {
  int32_t Idx = -1;             // RTL index, index is the number of devices
                                // of other RTLs that were registered before,
                                // i.e. the OpenMP index of the first device
                                // to be registered with this RTL.
  int32_t NumberOfDevices = -1; // Number of devices this RTL deals with.

  std::unique_ptr<llvm::sys::DynamicLibrary> LibraryHandler;

#ifdef OMPTARGET_DEBUG
  std::string RTLName;
#endif

#define PLUGIN_API_HANDLE(NAME, MANDATORY)                                     \
  using NAME##_ty = decltype(__tgt_rtl_##NAME);                                \
  NAME##_ty *NAME = nullptr;

#include "Shared/PluginAPI.inc"
#undef PLUGIN_API_HANDLE

  // Are there images associated with this RTL.
  bool IsUsed = false;

  llvm::DenseSet<const __tgt_device_image *> UsedImages;

  // Mutex for thread-safety when calling RTL interface functions.
  // It is easier to enforce thread-safety at the libomptarget level,
  // so that developers of new RTLs do not have to worry about it.
  std::mutex Mtx;
};

/// RTLs identified in the system.
struct PluginAdaptorManagerTy {
  // List of the detected runtime libraries.
  std::list<PluginAdaptorTy> AllRTLs;

  // Array of pointers to the detected runtime libraries that have compatible
  // binaries.
  llvm::SmallVector<PluginAdaptorTy *> UsedRTLs;

  int64_t RequiresFlags = OMP_REQ_UNDEFINED;

  explicit PluginAdaptorManagerTy() = default;

  // Register the clauses of the requires directive.
  void registerRequires(int64_t Flags);

  // Initialize RTL if it has not been initialized
  void initRTLonce(PluginAdaptorTy &RTL);

  // Initialize all RTLs
  void initAllRTLs();

  // Register a shared library with all (compatible) RTLs.
  void registerLib(__tgt_bin_desc *Desc);

  // Unregister a shared library from all RTLs.
  void unregisterLib(__tgt_bin_desc *Desc);

  // not thread-safe, called from global constructor (i.e. once)
  void loadRTLs();

private:
  static bool attemptLoadRTL(const std::string &RTLName, PluginAdaptorTy &RTL);
};

/// Struct for the data required to handle plugins
struct PluginManager {
  /// RTLs identified on the host
  PluginAdaptorManagerTy RTLs;

  /// Executable images and information extracted from the input images passed
  /// to the runtime.
  std::list<std::pair<__tgt_device_image, __tgt_image_info>> Images;

  /// Devices associated with RTLs
  std::vector<std::unique_ptr<DeviceTy>> Devices;
  std::mutex RTLsMtx; ///< For RTLs and Devices

  /// Translation table retreived from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  std::vector<__tgt_offload_entry *> HostEntriesBeginRegistrationOrder;

  /// Map from ptrs on the host to an entry in the Translation Table
  HostPtrToTableMapTy HostPtrToTableMap;
  std::mutex TblMapMtx; ///< For HostPtrToTableMap

  // Store target policy (disabled, mandatory, default)
  kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
  std::mutex TargetOffloadMtx; ///< For TargetOffloadPolicy

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

  int getNumDevices() {
    std::lock_guard<decltype(RTLsMtx)> Lock(RTLsMtx);
    return Devices.size();
  }

private:
  bool RTLsLoaded = false;
  llvm::SmallVector<__tgt_bin_desc *> DelayedBinDesc;
};

extern PluginManager *PM;

#endif // OMPTARGET_PLUGIN_MANAGER_H
