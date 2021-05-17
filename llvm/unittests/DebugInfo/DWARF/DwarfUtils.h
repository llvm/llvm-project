//===--- unittests/DebugInfo/DWARF/DwarfUtils.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFUTILS_H
#define LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFUTILS_H

#include <cstdint>

namespace llvm {

class Triple;

namespace dwarf {
namespace utils {

Triple getDefaultTargetTripleForAddrSize(uint8_t AddrSize);
Triple getNormalizedDefaultTargetTriple();
bool isConfigurationSupported(Triple &T);
bool isObjectEmissionSupported(Triple &T);

} // end namespace utils
} // end namespace dwarf
} // end namespace llvm

#endif // LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFUTILS_H
