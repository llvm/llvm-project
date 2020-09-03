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
#include "llvm/Support/MachineValueType.h"

namespace llvm {

class DataLayout;
class Type;

/// Construct a low-level type based on an LLVM type.
LLT getLLTForType(Type &Ty, const DataLayout &DL);

/// Get a rough equivalent of an MVT for a given LLT. MVT can't distinguish
/// pointers, so these will convert to a plain integer.
MVT getMVTForLLT(LLT Ty);

/// Get a rough equivalent of an LLT for a given MVT. LLT does not yet support
/// scalarable vector types, and will assert if used.
LLT getLLTForMVT(MVT Ty);

}

#endif // LLVM_CODEGEN_LOWLEVELTYPE_H
