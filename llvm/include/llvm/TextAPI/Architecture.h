//===- llvm/TextAPI/Architecture.h - Architecture ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the architecture enum and helper methods.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_ARCHITECTURE_H
#define LLVM_TEXTAPI_ARCHITECTURE_H

#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <utility>

namespace llvm {
class raw_ostream;
class StringRef;
class Triple;

namespace MachO {

/// Defines the architecture slices that are supported by Text-based Stub files.
enum Architecture : uint8_t {
  AK_i386,
  AK_x86_64,
  AK_x86_64h,
  AK_armv4t,
  AK_armv6,
  AK_armv5,
  AK_armv7,
  AK_armv7s,
  AK_armv7k,
  AK_armv6m,
  AK_armv7m,
  AK_armv7em,
  AK_armv8m_main,
  AK_armv8m_base,
  AK_armv8_1m_main,
  AK_arm64,
  AK_arm64e,
  AK_arm64_32,
  AK_riscv32,
  AK_unknown,
  AK_arm64e_x1,
};

/// Convert a CPU Type and Subtype pair to an architecture slice.
LLVM_ABI Architecture getArchitectureFromCpuType(uint32_t CPUType,
                                                 uint32_t CPUSubType);

/// Convert a name to an architecture slice.
LLVM_ABI Architecture getArchitectureFromName(StringRef Name);

/// Convert an architecture slice to a string.
LLVM_ABI StringRef getArchitectureName(Architecture Arch);

/// Convert an architecture slice to a CPU Type and Subtype pair.
LLVM_ABI std::pair<uint32_t, uint32_t>
getCPUTypeFromArchitecture(Architecture Arch);

/// Convert a target to an architecture slice.
LLVM_ABI Architecture mapToArchitecture(const llvm::Triple &Target);

/// Check if architecture is 64 bit.
LLVM_ABI bool is64Bit(Architecture);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, Architecture Arch);

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_ARCHITECTURE_H
