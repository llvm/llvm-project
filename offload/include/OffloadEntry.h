//===-- OffloadEntry.h - Representation of offload entries ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OFFLOAD_ENTRY_H
#define OMPTARGET_OFFLOAD_ENTRY_H

#include "Shared/APITypes.h"

#include "omptarget.h"

#include "llvm/ADT/StringRef.h"

class DeviceImageTy;

class OffloadEntryTy {
  DeviceImageTy &DeviceImage;
  llvm::offloading::EntryTy &OffloadEntry;

public:
  OffloadEntryTy(DeviceImageTy &DeviceImage,
                 llvm::offloading::EntryTy &OffloadEntry)
      : DeviceImage(DeviceImage), OffloadEntry(OffloadEntry) {}

  bool isGlobal() const { return getSize() != 0; }
  size_t getSize() const { return OffloadEntry.Size; }

  void *getnAddress() const { return OffloadEntry.Address; }
  llvm::StringRef getName() const { return OffloadEntry.SymbolName; }
  const char *getNameAsCStr() const { return OffloadEntry.SymbolName; }
  __tgt_bin_desc *getBinaryDescription() const;

  bool isLink() const { return hasFlags(OMP_DECLARE_TARGET_LINK); }

  bool hasFlags(OpenMPOffloadingDeclareTargetFlags Flags) const {
    return Flags & OffloadEntry.Flags;
  }
};

#endif // OMPTARGET_OFFLOAD_ENTRY_H
