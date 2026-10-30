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

/// Return whether we can guarantee physical register assignment if VReg's value
/// is rematerialized immediately before its use in MI. The new live range must
/// be assignable without further spilling. For example, if MI is a pseudo
/// instruction that has more register operands than available physical
/// registers, rematerializing VReg may leave the new live range unassignable.
bool canGuaranteeAssignmentAfterRemat(Register VReg, const MachineInstr &MI);

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_INLINESPILLER_H
