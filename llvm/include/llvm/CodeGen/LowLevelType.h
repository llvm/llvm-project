//== llvm/CodeGen/LowLevelType.h ------------------------------- -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Implement a low-level type suitable for MachineInstr level instruction
/// selection.
///
/// This provides the CodeGen aspects of LowLevelType, such as Type conversion.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LOWLEVELTYPE_H
#define LLVM_CODEGEN_LOWLEVELTYPE_H

#include "llvm/Support/LowLevelTypeImpl.h"

namespace llvm {

class DataLayout;
class Type;

/// Construct a low-level type based on an LLVM type.
LLT getLLTForType(Type &Ty, const DataLayout &DL);

}

#endif // LLVM_CODEGEN_LOWLEVELTYPE_H
