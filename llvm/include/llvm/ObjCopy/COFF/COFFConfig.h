//===- COFFConfig.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_COFF_COFFCONFIG_H
#define LLVM_OBJCOPY_COFF_COFFCONFIG_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {
namespace objcopy {

// Identifies a resource of a PE image by its integer type and name IDs and,
// optionally, its language ID.
struct COFFResourceIdentifier {
  uint32_t Type = 0;
  uint32_t Name = 0;
  std::optional<uint16_t> Language;
};

// A resource to dump from a PE image into a file.
struct COFFResourceDump {
  COFFResourceIdentifier Resource;
  StringRef FileName;
};

// A resource to add to a PE image or to replace in it.
struct COFFResourceUpdate {
  COFFResourceIdentifier Resource;
  std::shared_ptr<MemoryBuffer> Data;
};

// Coff specific configuration for copying/stripping a single file.
struct COFFConfig {
  std::optional<unsigned> Subsystem;
  std::optional<unsigned> MajorSubsystemVersion;
  std::optional<unsigned> MinorSubsystemVersion;
  SmallVector<COFFResourceDump, 0> DumpResource;
  SmallVector<COFFResourceUpdate, 0> UpdateResource;
};

} // namespace objcopy
} // namespace llvm

#endif // LLVM_OBJCOPY_COFF_COFFCONFIG_H
