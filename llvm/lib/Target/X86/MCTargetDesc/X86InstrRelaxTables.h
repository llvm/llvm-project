//===-- X86InstrRelaxTables.h - X86 Instruction Relaxation Tables -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the interface to query the X86 instruction relaxation
// tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86INSTRRELAXTABLES_H
#define LLVM_LIB_TARGET_X86_X86INSTRRELAXTABLES_H

#include <cstdint>

namespace llvm {

// This struct is used for both the relaxed and short tables. The KeyOp is used
// to determine the sorting order.
struct X86InstrRelaxTableEntry {
  uint16_t KeyOp;
  uint16_t DstOp;

  bool operator<(const X86InstrRelaxTableEntry &RHS) const {
    return KeyOp < RHS.KeyOp;
  }
  bool operator==(const X86InstrRelaxTableEntry &RHS) const {
    return KeyOp == RHS.KeyOp;
  }
  friend bool operator<(const X86InstrRelaxTableEntry &TE, unsigned Opcode) {
    return TE.KeyOp < Opcode;
  }
};

/// Look up the relaxed form table entry for a given \p ShortOp.
const X86InstrRelaxTableEntry *lookupRelaxTable(unsigned ShortOp);

/// Look up the short form table entry for a given \p RelaxOp.
const X86InstrRelaxTableEntry *lookupShortTable(unsigned RelaxOp);

namespace X86 {

/// Get the short instruction opcode for a given relaxed opcode.
unsigned getShortOpcodeArith(unsigned RelaxOp);

/// Get the relaxed instruction opcode for a given short opcode.
unsigned getRelaxedOpcodeArith(unsigned ShortOp);
} // namespace X86
} // namespace llvm

#endif
