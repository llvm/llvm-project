//===- llvm/TextAPI/Architecture.h - Architecture ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the architecture enum and helper methods.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_ARCHITECTURE_H
#define LLVM_TEXTAPI_MACHO_ARCHITECTURE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace MachO {

/// Defines the architecture slices that are supported by Text-based Stub files.
enum class Architecture : uint8_t {
#define ARCHINFO(Arch, Type, SubType) Arch,
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO
  unknown, // this has to go last.
};

/// Convert a CPU Type and Subtype pair to an architecture slice.
Architecture getArchitectureFromCpuType(uint32_t CPUType, uint32_t CPUSubType);

/// Convert a name to an architecture slice.
Architecture getArchitectureFromName(StringRef Name);

/// Convert an architecture slice to a string.
StringRef getArchitectureName(Architecture Arch);

/// Convert an architecture slice to a CPU Type and Subtype pair.
std::pair<uint32_t, uint32_t> getCPUTypeFromArchitecture(Architecture Arch);

raw_ostream &operator<<(raw_ostream &OS, Architecture Arch);

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_ARCHITECTURE_H
