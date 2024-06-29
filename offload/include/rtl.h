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

#include "llvm/ADT/SmallVector.h"

#include "omptarget.h"

#include <cstdint>
#include <map>

/// Map between the host entry begin and the translation table. Each
/// registered library gets one TranslationTable. Use the map from
/// __tgt_offload_entry so that we may quickly determine whether we
/// are trying to (re)register an existing lib or really have a new one.
struct TranslationTable {
  __tgt_target_table HostTable;
  llvm::SmallVector<__tgt_target_table> DeviceTables;

  // Image assigned to a given device.
  llvm::SmallVector<__tgt_device_image *>
      TargetsImages; // One image per device ID.

  // Arrays of entries active on the device.
  llvm::SmallVector<llvm::SmallVector<__tgt_offload_entry>>
      TargetsEntries; // One table per device ID.

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
