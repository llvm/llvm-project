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
namespace cl {
class LibraryOptionsContext;
} // namespace cl

class MCTargetOptions {
public:
  enum AsmInstrumentation {
    AsmInstrumentationNone,
    AsmInstrumentationAddress
  };

  enum DwarfDirectory {
    // Force disable
    DisableDwarfDirectory,
    // Force enable, for assemblers that support
    // `.file fileno directory filename' syntax
    EnableDwarfDirectory,
    // Default is based on the target
    DefaultDwarfDirectory
  };

#define LLVM_MC_OPTION_FIELD(Type, Name, Initializer) Type Name Initializer;
#define LLVM_MC_OPTION_BITFIELD(Name) bool Name : 1;
#define LLVM_MC_OPTION_BINUTILS_FIELD(Name) std::pair<int, int> Name = {0, 0};
#include "llvm/MC/MCTargetOptionsFields.def"
#undef LLVM_MC_OPTION_BINUTILS_FIELD
#undef LLVM_MC_OPTION_BITFIELD
#undef LLVM_MC_OPTION_FIELD

  // The library options of the compilation, usually its LLVMContext's.
  // TODO: Make it non-null once every tool sets it. Null means the global
  // values.
  const cl::LibraryOptionsContext *LibraryOptions = nullptr;

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
