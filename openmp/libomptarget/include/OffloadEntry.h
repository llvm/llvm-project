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
  __tgt_offload_entry &OffloadEntry;

public:
  OffloadEntryTy(DeviceImageTy &DeviceImage, __tgt_offload_entry &OffloadEntry)
      : DeviceImage(DeviceImage), OffloadEntry(OffloadEntry) {}

  bool isGlobal() const { return getSize() != 0; }
  size_t getSize() const { return OffloadEntry.size; }

  void *getAddress() const { return OffloadEntry.addr; }
  llvm::StringRef getName() const { return OffloadEntry.name; }
  const char *getNameAsCStr() const { return OffloadEntry.name; }
  __tgt_bin_desc *getBinaryDescription() const;

  bool isCTor() const { return hasFlags(OMP_DECLARE_TARGET_CTOR); }
  bool isDTor() const { return hasFlags(OMP_DECLARE_TARGET_DTOR); }
  bool isLink() const { return hasFlags(OMP_DECLARE_TARGET_LINK); }

  bool hasFlags(OpenMPOffloadingDeclareTargetFlags Flags) const {
    return Flags & OffloadEntry.flags;
  }
};

#endif // OMPTARGET_OFFLOAD_ENTRY_H
