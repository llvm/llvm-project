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

#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations.
struct DeviceTy;
struct __tgt_bin_desc;

struct RTLInfoTy {
  typedef int32_t(is_valid_binary_ty)(void *);
  typedef int32_t(number_of_devices_ty)();
  typedef int32_t(init_device_ty)(int32_t);
  typedef __tgt_target_table *(load_binary_ty)(int32_t, void *);
  typedef void *(data_alloc_ty)(int32_t, int64_t, void *);
  typedef int32_t(data_submit_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_retrieve_ty)(int32_t, void *, void *, int64_t);
  typedef int32_t(data_delete_ty)(int32_t, void *);
  typedef int32_t(run_region_ty)(int32_t, void *, void **, ptrdiff_t *,
                                 int32_t);
  typedef int32_t(run_team_region_ty)(int32_t, void *, void **, ptrdiff_t *,
                                      int32_t, int32_t, int32_t, uint64_t);

  int32_t Idx;                     // RTL index, index is the number of devices
                                   // of other RTLs that were registered before,
                                   // i.e. the OpenMP index of the first device
                                   // to be registered with this RTL.
  int32_t NumberOfDevices;         // Number of devices this RTL deals with.

  void *LibraryHandler;

#ifdef OMPTARGET_DEBUG
  std::string RTLName;
#endif

  // Functions implemented in the RTL.
  is_valid_binary_ty *is_valid_binary;
  number_of_devices_ty *number_of_devices;
  init_device_ty *init_device;
  load_binary_ty *load_binary;
  data_alloc_ty *data_alloc;
  data_submit_ty *data_submit;
  data_retrieve_ty *data_retrieve;
  data_delete_ty *data_delete;
  run_region_ty *run_region;
  run_team_region_ty *run_team_region;

  // Are there images associated with this RTL.
  bool isUsed;

  // Mutex for thread-safety when calling RTL interface functions.
  // It is easier to enforce thread-safety at the libomptarget level,
  // so that developers of new RTLs do not have to worry about it.
  std::mutex Mtx;

  // The existence of the mutex above makes RTLInfoTy non-copyable.
  // We need to provide a copy constructor explicitly.
  RTLInfoTy()
      : Idx(-1), NumberOfDevices(-1), LibraryHandler(0),
#ifdef OMPTARGET_DEBUG
        RTLName(),
#endif
        is_valid_binary(0), number_of_devices(0), init_device(0),
        load_binary(0), data_alloc(0), data_submit(0), data_retrieve(0),
        data_delete(0), run_region(0), run_team_region(0), isUsed(false),
        Mtx() {}

  RTLInfoTy(const RTLInfoTy &r) : Mtx() {
    Idx = r.Idx;
    NumberOfDevices = r.NumberOfDevices;
    LibraryHandler = r.LibraryHandler;
#ifdef OMPTARGET_DEBUG
    RTLName = r.RTLName;
#endif
    is_valid_binary = r.is_valid_binary;
    number_of_devices = r.number_of_devices;
    init_device = r.init_device;
    load_binary = r.load_binary;
    data_alloc = r.data_alloc;
    data_submit = r.data_submit;
    data_retrieve = r.data_retrieve;
    data_delete = r.data_delete;
    run_region = r.run_region;
    run_team_region = r.run_team_region;
    isUsed = r.isUsed;
  }
};

/// RTLs identified in the system.
class RTLsTy {
private:
  // Mutex-like object to guarantee thread-safety and unique initialization
  // (i.e. the library attempts to load the RTLs (plugins) only once).
  std::once_flag initFlag;
  void LoadRTLs(); // not thread-safe

public:
  // List of the detected runtime libraries.
  std::list<RTLInfoTy> AllRTLs;

  // Array of pointers to the detected runtime libraries that have compatible
  // binaries.
  std::vector<RTLInfoTy *> UsedRTLs;

  explicit RTLsTy() {}

  // Register a shared library with all (compatible) RTLs.
  void RegisterLib(__tgt_bin_desc *desc);

  // Unregister a shared library from all RTLs.
  void UnregisterLib(__tgt_bin_desc *desc);
};
extern RTLsTy RTLs;
extern std::mutex RTLsMtx;


/// Map between the host entry begin and the translation table. Each
/// registered library gets one TranslationTable. Use the map from
/// __tgt_offload_entry so that we may quickly determine whether we
/// are trying to (re)register an existing lib or really have a new one.
struct TranslationTable {
  __tgt_target_table HostTable;

  // Image assigned to a given device.
  std::vector<__tgt_device_image *> TargetsImages; // One image per device ID.

  // Table of entry points or NULL if it was not already computed.
  std::vector<__tgt_target_table *> TargetsTable; // One table per device ID.
};
typedef std::map<__tgt_offload_entry *, TranslationTable>
    HostEntriesBeginToTransTableTy;
extern HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
extern std::mutex TrlTblMtx;

/// Map between the host ptr and a table index
struct TableMap {
  TranslationTable *Table; // table associated with the host ptr.
  uint32_t Index; // index in which the host ptr translated entry is found.
  TableMap() : Table(0), Index(0) {}
  TableMap(TranslationTable *table, uint32_t index)
      : Table(table), Index(index) {}
};
typedef std::map<void *, TableMap> HostPtrToTableMapTy;
extern HostPtrToTableMapTy HostPtrToTableMap;
extern std::mutex TblMapMtx;

#endif
