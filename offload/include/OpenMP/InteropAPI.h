//===-- OpenMP/InteropAPI.h - OpenMP interoperability types and API - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_INTEROP_API_H
#define OMPTARGET_OPENMP_INTEROP_API_H

#include "omp.h"

#include "PerThreadTable.h"
#include "omptarget.h"

extern "C" {

typedef enum kmp_interop_type_t {
  kmp_interop_type_unknown = -1,
  kmp_interop_type_target,
  kmp_interop_type_targetsync,
} kmp_interop_type_t;

struct interop_attrs_t {
  uint32_t inorder : 1;
  uint32_t reserved : 31;

  /// Check if the supported attributes are compatible with the current
  ///   attributes. Only if an attribute is supported can the value be true,
  ///   otherwise it needs to be false
  bool checkSupportedOnly(interop_attrs_t supported) const {
    return supported.inorder || (!supported.inorder && !inorder);
  }
};

struct interop_spec_t {
  int32_t fr_id;
  interop_attrs_t attrs; // Common attributes
  int64_t impl_attrs; // Implementation specific attributes (recognized by each
                      // plugin)
};

struct interop_flags_t {
  uint32_t implicit : 1; // dispatch (true) or interop (false)
  uint32_t nowait : 1;   // has nowait flag
  uint32_t reserved : 30;
};

struct interop_ctx_t {
  uint32_t version; // version of the interface (current is 0)
  interop_flags_t flags;
  int32_t gtid;
};

struct dep_pack_t {
  int32_t ndeps;
  int32_t ndeps_noalias;
  kmp_depend_info_t *deplist;
  kmp_depend_info_t *noalias_deplist;
};

struct omp_interop_val_t;

typedef void ompx_interop_cb_t(omp_interop_val_t *interop, void *data);

struct omp_interop_cb_instance_t {
  ompx_interop_cb_t *cb;
  void *data;

  omp_interop_cb_instance_t(ompx_interop_cb_t *cb, void *data)
      : cb(cb), data(data) {}

  void operator()(omp_interop_val_t *interop) { cb(interop, data); }
};

/// The interop value type, aka. the interop object.
typedef struct omp_interop_val_t {
  /// Device and interop-type are determined at construction time and fix.
  omp_interop_val_t(intptr_t device_id, kmp_interop_type_t interop_type)
      : interop_type(interop_type), device_id(device_id) {}
  const char *err_str = nullptr;
  __tgt_async_info *async_info = nullptr;
  __tgt_device_info device_info;
  const kmp_interop_type_t interop_type;
  const intptr_t device_id;
  omp_vendor_id_t vendor_id = omp_vendor_llvm;
  tgt_foreign_runtime_id_t fr_id = tgt_fr_none;
  interop_attrs_t attrs{false, 0}; // Common prefer specification attributes
  int64_t impl_attrs = 0; // Implementation prefer specification attributes

  // Constants
  static constexpr int no_owner = -1; // This interop has no current owner

  void *rtl_property = nullptr; // Plugin dependent information
  // For implicitly created Interop objects (e.g., from a dispatch construct)
  // who owns the object
  int owner_gtid = no_owner;
  // Marks whether the object was requested since the last time it was synced
  bool clean = true;

  typedef llvm::SmallVector<omp_interop_cb_instance_t> callback_list_t;

  callback_list_t completion_cbs;

  void reset() {
    owner_gtid = no_owner;
    markClean();
    clearCompletionCbs();
  }

  llvm::Expected<DeviceTy &> getDevice() const;

  bool hasOwner() const { return owner_gtid != no_owner; }

  void setOwner(int gtid) { owner_gtid = gtid; }
  bool isOwnedBy(int gtid) { return owner_gtid == gtid; }
  bool isCompatibleWith(int32_t InteropType, const interop_spec_t &Spec);
  bool isCompatibleWith(int32_t InteropType, const interop_spec_t &Spec,
                        int64_t DeviceNum, int gtid);
  void markClean() { clean = true; }
  void markDirty() { clean = false; }
  bool isClean() const { return clean; }

  int32_t flush(DeviceTy &Device);
  int32_t sync_barrier(DeviceTy &Device);
  int32_t async_barrier(DeviceTy &Device);
  int32_t release(DeviceTy &Device);

  void addCompletionCb(ompx_interop_cb_t *cb, void *data) {
    completion_cbs.push_back(omp_interop_cb_instance_t(cb, data));
  }

  int numCompletionCbs() const { return completion_cbs.size(); }
  void clearCompletionCbs() { completion_cbs.clear(); }

  void runCompletionCbs() {
    for (auto &cbInstance : completion_cbs)
      cbInstance(this);
    clearCompletionCbs();
  }
} omp_interop_val_t;

} // extern "C"

struct InteropTableEntry {
  using ContainerTy = typename std::vector<omp_interop_val_t *>;
  using iterator = typename ContainerTy::iterator;

  ContainerTy Interops;

  static constexpr int reservedEntriesPerThread =
      20; // reserve some entries to avoid reallocation

  void add(omp_interop_val_t *obj) {
    if (Interops.capacity() == 0)
      Interops.reserve(reservedEntriesPerThread);
    Interops.push_back(obj);
  }

  template <class ClearFuncTy> void clear(ClearFuncTy f) {
    for (auto &Obj : Interops) {
      f(Obj);
    }
  }

  /// vector interface
  int size() const { return Interops.size(); }
  iterator begin() { return Interops.begin(); }
  iterator end() { return Interops.end(); }
  iterator erase(iterator it) { return Interops.erase(it); }
};

struct InteropTblTy
    : public PerThreadTable<InteropTableEntry, omp_interop_val_t *> {
  void clear();
};

void syncImplicitInterops(int gtid, void *event);

#endif // OMPTARGET_OPENMP_INTEROP_API_H
