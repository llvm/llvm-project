//===- Options.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_DWARFUTIL_OPTIONS_H
#define LLVM_TOOLS_LLVM_DWARFUTIL_OPTIONS_H

#include "llvm/Support/Compression.h"
#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
namespace dwarfutil {

/// The kind of tombstone value.
enum class TombstoneKind {
  BFD,       /// 0/[1:1]. Bfd default.
  MaxPC,     /// -1/-2. Assumed to match with
             /// http://www.dwarfstd.org/ShowIssue.php?issue=200609.1.
  Universal, /// both: BFD + MaxPC
  Exec,      /// match with address range of executable sections.
};

/// The kind of accelerator table.
enum class DwarfUtilAccelKind : uint8_t {
  None,
  DWARF // DWARFv5: .debug_names
};

struct Options {
  std::string InputFileName;
  std::string OutputFileName;
  bool DoGarbageCollection = false;
  bool DoODRDeduplication = false;
  bool BuildSeparateDebugFile = false;
  TombstoneKind Tombstone = TombstoneKind::Universal;
  bool Verbose = false;
  int NumThreads = 0;
  bool Verify = false;
  bool UseDWARFLinkerParallel = false;
  DwarfUtilAccelKind AccelTableKind = DwarfUtilAccelKind::None;
  /// Compression type of the output debug sections. If not specified, the
  /// output debug sections are compressed the same way as the input ones.
  std::optional<DebugCompressionType> CompressionType;

  std::string getSeparateDebugFileName() const {
    return OutputFileName + ".debug";
  }
};

} // namespace dwarfutil
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_DWARFUTIL_OPTIONS_H
