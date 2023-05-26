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

template <typename T> class ArrayRef;
class Constant;

/// Replace constant expressions users of the given constants with
/// instructions. Return whether anything was changed.
bool convertUsersOfConstantsToInstructions(ArrayRef<Constant *> Consts);

} // end namespace llvm

#endif // LLVM_IR_REPLACECONSTANT_H
