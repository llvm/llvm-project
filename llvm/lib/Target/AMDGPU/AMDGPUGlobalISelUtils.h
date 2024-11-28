//===- AMDGPUGlobalISelUtils -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H

#include "llvm/CodeGen/Register.h"
#include <utility>

namespace llvm {

#if LLPC_BUILD_NPI
class AllocaInst;
#endif /* LLPC_BUILD_NPI */
class MachineRegisterInfo;
class GCNSubtarget;
class GISelKnownBits;
class LLT;
#if LLPC_BUILD_NPI
class MachineMemOperand;
#endif /* LLPC_BUILD_NPI */

namespace AMDGPU {

/// Returns base register and constant offset.
std::pair<Register, unsigned>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg,
                          GISelKnownBits *KnownBits = nullptr,
                          bool CheckNUW = false);
#if LLPC_BUILD_NPI

bool IsLaneSharedInVGPR(const MachineMemOperand *MemOpnd);

bool IsPromotablePrivate(const AllocaInst &Alloca);
bool IsPromotablePrivate(const MachineMemOperand *MemOpnd);

} // namespace AMDGPU
} // namespace llvm
#else /* LLPC_BUILD_NPI */
}
}
#endif /* LLPC_BUILD_NPI */

#endif
