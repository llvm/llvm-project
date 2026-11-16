//===- MemCacheHint.h - Cache hint metadata helpers ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MEMCACHEHINT_H
#define LLVM_IR_MEMCACHEHINT_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class Instruction;
class MDNode;

/// Return the hint payload for operand \p OperandNo of \p I, or nullptr if
/// absent. Operand numbers are argument indices for intrinsic calls and
/// instruction operand indices otherwise.
LLVM_ABI MDNode *getMemCacheHint(const Instruction &I, unsigned OperandNo = 0);

/// Set the hint payload for operand \p OperandNo of \p I, preserving other
/// operands' hints. A null \p Hint removes this operand's entry. Entries are
/// kept in increasing operand order; removing the last entry clears the
/// attachment. Operand numbering follows getMemCacheHint().
LLVM_ABI void setMemCacheHint(Instruction &I, unsigned OperandNo, MDNode *Hint);

} // namespace llvm

#endif // LLVM_IR_MEMCACHEHINT_H
