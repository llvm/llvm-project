//===- MCTargetOptions.h - MC Target Options --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETOPTIONS_H
#define LLVM_MC_MCTARGETOPTIONS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Compression.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

enum class EmitDwarfUnwindType {
  Always,          // Always emit dwarf unwind
  NoCompactUnwind, // Only emit if compact unwind isn't available
  DwarfOnly,       // Force compact unwind to reference DWARF
  Default,         // Default behavior is based on the target
};

// For ELF targets, whether to adjust relocations referencing eligible local
// symbols to use section symbols.
enum class RelocSectionSymType {
  All,      // For all eligible local symbols (default)
  Internal, // For .L symbols
  None,     // Never use section symbols
};

class StringRef;

class MCTargetOptions {
public:
  enum AsmInstrumentation { AsmInstrumentationNone, AsmInstrumentationAddress };

  enum DwarfDirectory {
    // Force disable.
    DisableDwarfDirectory,
    // Force enable for assemblers that support the
    // `.file fileno directory filename' syntax.
    EnableDwarfDirectory,
    // Default is based on the target.
    DefaultDwarfDirectory
  };

#define MC_TARGET_OPTION_TYPE(...) __VA_ARGS__
#define MC_TARGET_OPTION_DECLARE_BITFIELD(Type, Name, Bits, Default)           \
  MC_TARGET_OPTION_TYPE Type Name : Bits;
#define MC_TARGET_OPTION_DECLARE_BOOL(Type, Name, Bits, Default)               \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION_DECLARE_ENUM(Type, Name, Bits, Default)               \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION_DECLARE_OPTIONAL_UINT(Type, Name, Bits, Default)      \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION_DECLARE_INT(Type, Name, Bits, Default)                \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION_DECLARE_PAIR(Type, Name, Bits, Default)               \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION_DECLARE_STRING(Type, Name, Bits, Default)             \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION_DECLARE_STRING_LIST(Type, Name, Bits, Default)        \
  MC_TARGET_OPTION_TYPE Type Name = Default;
#define MC_TARGET_OPTION(Type, Name, Bits, Default, Kind)                      \
  MC_TARGET_OPTION_DECLARE_##Kind(Type, Name, Bits, Default)
#include "llvm/MC/MCTargetOptions.def"
#undef MC_TARGET_OPTION_TYPE
#undef MC_TARGET_OPTION_DECLARE_BITFIELD
#undef MC_TARGET_OPTION_DECLARE_BOOL
#undef MC_TARGET_OPTION_DECLARE_ENUM
#undef MC_TARGET_OPTION_DECLARE_OPTIONAL_UINT
#undef MC_TARGET_OPTION_DECLARE_INT
#undef MC_TARGET_OPTION_DECLARE_PAIR
#undef MC_TARGET_OPTION_DECLARE_STRING
#undef MC_TARGET_OPTION_DECLARE_STRING_LIST

  LLVM_ABI MCTargetOptions();

  /// Parse a binutils version string ("major[.minor]" or "none") into a
  /// (major, minor) pair. "none" maps to {INT_MAX, INT_MAX}.
  LLVM_ABI static std::pair<int, int> parseBinutilsVersion(StringRef Version);

  /// getABIName - If this returns a non-empty string this represents the
  /// textual name of the ABI that we want the backend to use, e.g. o32, or
  /// aapcs-linux.
  LLVM_ABI StringRef getABIName() const;

  /// getAssemblyLanguage - If this returns a non-empty string this represents
  /// the textual name of the assembly language that we will use for this
  /// target, e.g. masm.
  LLVM_ABI StringRef getAssemblyLanguage() const;
};

} // end namespace llvm

#endif // LLVM_MC_MCTARGETOPTIONS_H
