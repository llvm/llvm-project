//===- ReplaceConstant.h - Replacing LLVM constant expressions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the utility function for replacing LLVM constant
// expressions by instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_REPLACECONSTANT_H
#define LLVM_IR_REPLACECONSTANT_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"

namespace llvm {

/// Create a replacement instruction for constant expression \p CE and insert
/// it before \p Instr.
Instruction *createReplacementInstr(ConstantExpr *CE, Instruction *Instr);

} // end namespace llvm

#endif // LLVM_IR_REPLACECONSTANT_H
