//===- llvm/Support/FileSystem/UniqueID.h - UniqueID for files --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is cut out of llvm/Support/FileSystem.h to allow UniqueID to be
// reused without bloating the includes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILESYSTEM_UNIQUEID_H
#define LLVM_SUPPORT_FILESYSTEM_UNIQUEID_H

#include <cstdint>

namespace llvm {
namespace sys {
namespace fs {

class UniqueID {
  uint64_t Device;
  uint64_t File;

public:
  UniqueID() = default;
  UniqueID(uint64_t Device, uint64_t File) : Device(Device), File(File) {}

  bool operator==(const UniqueID &Other) const {
    return Device == Other.Device && File == Other.File;
  }
  bool operator!=(const UniqueID &Other) const { return !(*this == Other); }
  bool operator<(const UniqueID &Other) const {
    /// Don't use std::tie since it bloats the compile time of this header.
    if (Device < Other.Device)
      return true;
    if (Other.Device < Device)
      return false;
    return File < Other.File;
  }

  uint64_t getDevice() const { return Device; }
  uint64_t getFile() const { return File; }
};

} // end namespace fs
} // end namespace sys
} // end namespace llvm

#endif // LLVM_SUPPORT_FILESYSTEM_UNIQUEID_H
