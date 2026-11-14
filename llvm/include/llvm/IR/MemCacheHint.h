//===- MemCacheHint.h - Cache hint metadata helpers ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MEMCACHEHINT_H
#define LLVM_IR_MEMCACHEHINT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Instruction;
class MDNode;
class Value;

/// Return the hint payload for operand \p OperandNo of \p I, or nullptr if
/// absent. Operand numbers are argument indices for intrinsic calls and
/// instruction operand indices otherwise.
LLVM_ABI MDNode *getMemCacheHint(const Instruction &I, unsigned OperandNo = 0);

/// Set the hint payload for operand \p OperandNo of \p I, preserving other
/// operands' hints. A null \p Hint removes this operand's entry. Entries are
/// kept in increasing operand order; removing the last entry clears the
/// attachment. Operand numbering follows getMemCacheHint().
LLVM_ABI void setMemCacheHint(Instruction &I, unsigned OperandNo, MDNode *Hint);

/// Copy identical source cache-hint attachments to \p Dest, or clear its
/// attachment if propagation fails. All accesses must be supported reads (loads
/// or masked loads/gathers) or writes (stores or masked stores/scatters), with
/// matching pointer operand positions.
LLVM_ABI void propagateMemCacheHint(Instruction &Dest,
                                    ArrayRef<Value *> Sources);

} // namespace llvm

#endif // LLVM_IR_MEMCACHEHINT_H
