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

#include "device.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"

#include <list>
#include <mutex>

struct PluginAdaptorTy {
  typedef int32_t(init_plugin_ty)();
  typedef int32_t(is_valid_binary_ty)(void *);
  typedef int32_t(is_valid_binary_info_ty)(void *, void *);
  typedef int32_t(is_data_exchangable_ty)(int32_t, int32_t);
  typedef int32_t(number_of_devices_ty)();
  typedef int32_t(init_device_ty)(int32_t);
  typedef __tgt_target_table *(load_binary_ty)(int32_t, void *);
  typedef void *(data_alloc_ty)(int32_t, int64_t, void *, int32_t);
  typedef int32_t(data_submit_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_submit_async_ty)(int32_t, void *, void *, int64_t,
                                        __tgt_async_info *);
  typedef int32_t(data_retrieve_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_retrieve_async_ty)(int32_t, void *, void *, int64_t,
                                          __tgt_async_info *);
  typedef int32_t(data_exchange_ty)(int32_t, void *, int32_t, void *, int64_t);
  typedef int32_t(data_exchange_async_ty)(int32_t, void *, int32_t, void *,
                                          int64_t, __tgt_async_info *);
  typedef int32_t(data_delete_ty)(int32_t, void *, int32_t);
  typedef int32_t(launch_kernel_ty)(int32_t, void *, void **, ptrdiff_t *,
                                    const KernelArgsTy *, __tgt_async_info *);
  typedef int64_t(init_requires_ty)(int64_t);
  typedef int32_t(synchronize_ty)(int32_t, __tgt_async_info *);
  typedef int32_t(query_async_ty)(int32_t, __tgt_async_info *);
  typedef int32_t(supports_empty_images_ty)();
  typedef void(print_device_info_ty)(int32_t);
  typedef void(set_info_flag_ty)(uint32_t);
  typedef int32_t(create_event_ty)(int32_t, void **);
  typedef int32_t(record_event_ty)(int32_t, void *, __tgt_async_info *);
  typedef int32_t(wait_event_ty)(int32_t, void *, __tgt_async_info *);
  typedef int32_t(sync_event_ty)(int32_t, void *);
  typedef int32_t(destroy_event_ty)(int32_t, void *);
  typedef int32_t(release_async_info_ty)(int32_t, __tgt_async_info *);
  typedef int32_t(init_async_info_ty)(int32_t, __tgt_async_info **);
  typedef int64_t(init_device_into_ty)(int64_t, __tgt_device_info *,
                                       const char **);
  typedef int32_t(data_lock_ty)(int32_t, void *, int64_t, void **);
  typedef int32_t(data_unlock_ty)(int32_t, void *);
  typedef int32_t(data_notify_mapped_ty)(int32_t, void *, int64_t);
  typedef int32_t(data_notify_unmapped_ty)(int32_t, void *);
  typedef int32_t(set_device_offset_ty)(int32_t);
  typedef int32_t(activate_record_replay_ty)(int32_t, uint64_t, void *, bool,
                                             bool, uint64_t &);

  int32_t Idx = -1;             // RTL index, index is the number of devices
                                // of other RTLs that were registered before,
                                // i.e. the OpenMP index of the first device
                                // to be registered with this RTL.
  int32_t NumberOfDevices = -1; // Number of devices this RTL deals with.

  std::unique_ptr<llvm::sys::DynamicLibrary> LibraryHandler;

#ifdef OMPTARGET_DEBUG
  std::string RTLName;
#endif

  // Functions implemented in the RTL.
  init_plugin_ty *init_plugin = nullptr;
  is_valid_binary_ty *is_valid_binary = nullptr;
  is_valid_binary_info_ty *is_valid_binary_info = nullptr;
  is_data_exchangable_ty *is_data_exchangable = nullptr;
  number_of_devices_ty *number_of_devices = nullptr;
  init_device_ty *init_device = nullptr;
  load_binary_ty *load_binary = nullptr;
  data_alloc_ty *data_alloc = nullptr;
  data_submit_ty *data_submit = nullptr;
  data_submit_async_ty *data_submit_async = nullptr;
  data_retrieve_ty *data_retrieve = nullptr;
  data_retrieve_async_ty *data_retrieve_async = nullptr;
  data_exchange_ty *data_exchange = nullptr;
  data_exchange_async_ty *data_exchange_async = nullptr;
  data_delete_ty *data_delete = nullptr;
  launch_kernel_ty *launch_kernel = nullptr;
  init_requires_ty *init_requires = nullptr;
  synchronize_ty *synchronize = nullptr;
  query_async_ty *query_async = nullptr;
  supports_empty_images_ty *supports_empty_images = nullptr;
  set_info_flag_ty *set_info_flag = nullptr;
  print_device_info_ty *print_device_info = nullptr;
  create_event_ty *create_event = nullptr;
  record_event_ty *record_event = nullptr;
  wait_event_ty *wait_event = nullptr;
  sync_event_ty *sync_event = nullptr;
  destroy_event_ty *destroy_event = nullptr;
  init_async_info_ty *init_async_info = nullptr;
  init_device_into_ty *init_device_info = nullptr;
  release_async_info_ty *release_async_info = nullptr;
  data_lock_ty *data_lock = nullptr;
  data_unlock_ty *data_unlock = nullptr;
  data_notify_mapped_ty *data_notify_mapped = nullptr;
  data_notify_unmapped_ty *data_notify_unmapped = nullptr;
  set_device_offset_ty *set_device_offset = nullptr;
  activate_record_replay_ty *activate_record_replay = nullptr;

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
  PluginManager(bool UseEventsForAtomicTransfers)
      : UseEventsForAtomicTransfers(UseEventsForAtomicTransfers) {}

  /// RTLs identified on the host
  PluginAdaptorManagerTy RTLs;

  /// Executable images and information extracted from the input images passed
  /// to the runtime.
  std::list<std::pair<__tgt_device_image, __tgt_image_info>> Images;

  /// Devices associated with RTLs
  llvm::SmallVector<std::unique_ptr<DeviceTy>> Devices;
  std::mutex RTLsMtx; ///< For RTLs and Devices

  /// Translation table retreived from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  llvm::SmallVector<__tgt_offload_entry *> HostEntriesBeginRegistrationOrder;

  /// Map from ptrs on the host to an entry in the Translation Table
  HostPtrToTableMapTy HostPtrToTableMap;
  std::mutex TblMapMtx; ///< For HostPtrToTableMap

  // Store target policy (disabled, mandatory, default)
  kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
  std::mutex TargetOffloadMtx; ///< For TargetOffloadPolicy

  /// Flag to indicate if we use events to ensure the atomicity of
  /// map clauses or not. Can be modified with an environment variable.
  const bool UseEventsForAtomicTransfers;

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

private:
  bool RTLsLoaded = false;
  llvm::SmallVector<__tgt_bin_desc *> DelayedBinDesc;
};

extern PluginManager *PM;

#endif // OMPTARGET_PLUGIN_MANAGER_H
