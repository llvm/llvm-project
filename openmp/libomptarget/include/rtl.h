//===------------ rtl.h - Target independent OpenMP target RTL ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for handling RTL plugins.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_RTL_H
#define _OMPTARGET_RTL_H

#include "omptarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"

#include "omptarget.h"

#include <list>
#include <map>
#include <mutex>
#include <string>

// Forward declarations.
struct DeviceTy;
struct __tgt_bin_desc;

struct RTLInfoTy {
  typedef int32_t(init_plugin_ty)();
  typedef int32_t(deinit_plugin_ty)();
  typedef int32_t(is_valid_binary_ty)(void *);
  typedef int32_t(is_valid_binary_info_ty)(void *, void *);
  typedef int32_t(is_data_exchangable_ty)(int32_t, int32_t);
  typedef int32_t(number_of_devices_ty)();
  typedef int32_t(init_device_ty)(int32_t);
  typedef int32_t(deinit_device_ty)(int32_t);
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
  typedef int32_t (*register_lib_ty)(__tgt_bin_desc *);
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
  deinit_plugin_ty *deinit_plugin = nullptr;
  is_valid_binary_ty *is_valid_binary = nullptr;
  is_valid_binary_info_ty *is_valid_binary_info = nullptr;
  is_data_exchangable_ty *is_data_exchangable = nullptr;
  number_of_devices_ty *number_of_devices = nullptr;
  init_device_ty *init_device = nullptr;
  deinit_device_ty *deinit_device = nullptr;
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
  register_lib_ty register_lib = nullptr;
  register_lib_ty unregister_lib = nullptr;
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

  // Are there images associated with this RTL.
  bool IsUsed = false;

  llvm::DenseSet<const __tgt_device_image *> UsedImages;

  // Mutex for thread-safety when calling RTL interface functions.
  // It is easier to enforce thread-safety at the libomptarget level,
  // so that developers of new RTLs do not have to worry about it.
  std::mutex Mtx;
};

/// RTLs identified in the system.
struct RTLsTy {
  // List of the detected runtime libraries.
  std::list<RTLInfoTy> AllRTLs;

  // Array of pointers to the detected runtime libraries that have compatible
  // binaries.
  llvm::SmallVector<RTLInfoTy *> UsedRTLs;

  int64_t RequiresFlags = OMP_REQ_UNDEFINED;

  explicit RTLsTy() = default;

  // Register the clauses of the requires directive.
  void registerRequires(int64_t Flags);

  // Initialize RTL if it has not been initialized
  void initRTLonce(RTLInfoTy &RTL);

  // Initialize all RTLs
  void initAllRTLs();

  // Register a shared library with all (compatible) RTLs.
  void registerLib(__tgt_bin_desc *Desc);

  // Unregister a shared library from all RTLs.
  void unregisterLib(__tgt_bin_desc *Desc);

  // not thread-safe, called from global constructor (i.e. once)
  void loadRTLs();

private:
  static bool attemptLoadRTL(const std::string &RTLName, RTLInfoTy &RTL);
};

/// Map between the host entry begin and the translation table. Each
/// registered library gets one TranslationTable. Use the map from
/// __tgt_offload_entry so that we may quickly determine whether we
/// are trying to (re)register an existing lib or really have a new one.
struct TranslationTable {
  __tgt_target_table HostTable;

  // Image assigned to a given device.
  llvm::SmallVector<__tgt_device_image *>
      TargetsImages; // One image per device ID.

  // Table of entry points or NULL if it was not already computed.
  llvm::SmallVector<__tgt_target_table *>
      TargetsTable; // One table per device ID.
};
typedef std::map<__tgt_offload_entry *, TranslationTable>
    HostEntriesBeginToTransTableTy;

/// Map between the host ptr and a table index
struct TableMap {
  TranslationTable *Table = nullptr; // table associated with the host ptr.
  uint32_t Index = 0; // index in which the host ptr translated entry is found.
  TableMap() = default;
  TableMap(TranslationTable *Table, uint32_t Index)
      : Table(Table), Index(Index) {}
};
typedef std::map<void *, TableMap> HostPtrToTableMapTy;

#endif
