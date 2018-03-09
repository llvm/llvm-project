//===--- unittests/DebugInfo/DWARF/DwarfUtils.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFUTILS_H
#define LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFUTILS_H

#include <cstdint>

namespace llvm {

class Triple;

namespace dwarf {
namespace utils {

Triple getHostTripleForAddrSize(uint8_t AddrSize);
bool isConfigurationSupported(Triple &T);

} // end namespace utils
} // end namespace dwarf
} // end namespace llvm

#endif // LLVM_UNITTESTS_DEBUG_INFO_DWARF_DWARFUTILS_H
