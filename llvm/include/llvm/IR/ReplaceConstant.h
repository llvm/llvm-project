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

#include <map>
#include <vector>

namespace llvm {

class ConstantExpr;
class Instruction;
class Use;
template <typename PtrType> class SmallPtrSetImpl;

/// The given instruction \p I contains given constant expression \p CE as one
/// of its operands, possibly nested within constant expression trees. Convert
/// all reachable paths from contant expression operands of \p I to \p CE into
/// corresponding instructions, insert them before \p I, update operands of \p I
/// accordingly, and if required, return all such converted instructions at
/// \p Insts.
void convertConstantExprsToInstructions(
    Instruction *I, ConstantExpr *CE,
    SmallPtrSetImpl<Instruction *> *Insts = nullptr);

/// The given instruction \p I contains constant expression CE within the
/// constant expression trees of it`s constant expression operands, and
/// \p CEPaths holds all the reachable paths (to CE) from such constant
/// expression trees of \p I. Convert constant expressions within these paths
/// into corresponding instructions, insert them before \p I, update operands of
/// \p I accordingly, and if required, return all such converted instructions at
/// \p Insts.
void convertConstantExprsToInstructions(
    Instruction *I,
    std::map<Use *, std::vector<std::vector<ConstantExpr *>>> &CEPaths,
    SmallPtrSetImpl<Instruction *> *Insts = nullptr);

/// Given an instruction \p I which uses given constant expression \p CE as
/// operand, either directly or nested within other constant expressions, return
/// all reachable paths from the constant expression operands of \p I to \p CE,
/// and return collected paths at \p CEPaths.
void collectConstantExprPaths(
    Instruction *I, ConstantExpr *CE,
    std::map<Use *, std::vector<std::vector<ConstantExpr *>>> &CEPaths);

} // end namespace llvm

#endif // LLVM_IR_REPLACECONSTANT_H
