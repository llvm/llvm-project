//===- InlineSpiller.h - Inline spiller helpers -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_INLINESPILLER_H
#define LLVM_LIB_CODEGEN_INLINESPILLER_H

#include "llvm/CodeGen/Register.h"

namespace llvm {

class MachineInstr;

/// Whether rematerializing VReg for MI can produce an assignable register.
bool canGuaranteeAssignmentAfterRemat(Register VReg, MachineInstr &MI);

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_INLINESPILLER_H
