//===- llvm/Support/Disassembler.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the necessary glue to call external disassembler
// libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_DISASSEMBLER_H
#define LLVM_SYSTEM_DISASSEMBLER_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
namespace sys {

/// This function returns true, if there is possible to use some external
/// disassembler library. False otherwise.
bool hasDisassembler();

/// This function provides some "glue" code to call external disassembler
/// libraries.
std::string disassembleBuffer(uint8_t* start, size_t length, uint64_t pc = 0);

}
}

#endif // LLVM_SYSTEM_DISASSEMBLER_H
