//===-- llvm/Target/TargetOpcodes.h - Target Indep Opcodes ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target independent instruction opcodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPCODES_H
#define LLVM_TARGET_TARGETOPCODES_H

namespace llvm {

/// Invariant opcodes: All instruction sets have these as their low opcodes.
///
namespace TargetOpcode {
enum {
#define HANDLE_TARGET_OPCODE(OPC, NUM) OPC = NUM,
#define HANDLE_TARGET_OPCODE_MARKER(IDENT, OPC) IDENT = OPC,
#include "llvm/Target/TargetOpcodes.def"
};
} // end namespace TargetOpcode
} // end namespace llvm

#endif
