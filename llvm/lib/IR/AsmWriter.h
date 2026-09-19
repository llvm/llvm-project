//===- AsmWriter.h - Internal LLVM assembly printing helpers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_IR_ASMWRITER_H
#define LLVM_LIB_IR_ASMWRITER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Instruction;
class ModuleSlotTracker;
class Type;
class Value;
class raw_ostream;

namespace detail {

using ValueNameCallback = function_ref<void(raw_ostream &, const Value &)>;
using TypeNameCallback = function_ref<void(raw_ostream &, const Type &)>;

LLVM_ABI void printInstructionWithCustomNames(raw_ostream &OS,
                                              ModuleSlotTracker &MST,
                                              const Instruction &I,
                                              ValueNameCallback PrintValueName,
                                              TypeNameCallback PrintTypeName,
                                              bool PrintCallAttributesInline);

} // namespace detail
} // namespace llvm

#endif // LLVM_LIB_IR_ASMWRITER_H
