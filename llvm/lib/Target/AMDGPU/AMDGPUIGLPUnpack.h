//===- AMDGPUIGLPUnpack.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \c llvm.amdgcn.iglp.opt documents mask 0 (small gemm). In
// AMDGPUIGroupLP.cpp, \c IGLPStrategyID and \c createIGLPStrategy() only
// implement strategy ids 0–3 (MFMASmallGemmOptID … MFMAExpSimpleInterleaveID);
// any other immediate reaches \c llvm_unreachable("Unknown IGLPStrategyID").
//
// So \p AMDGPUIGLPOptMaskExperimental (4) is the first mask value not handled
// by IGroupLP scheduling. It is safe for experimental MIR only if those
// IGLP_OPT pseudos are removed before the machine scheduler (see
// kRemoveExperimentalIGLPOptPseudo in AMDGPUIGLPUnpack.cpp), or once IGroupLP
// gains a strategy for that id.
//
// By default (AMDGPUIGLP_UNPACK_POLICY in AMDGPUIGLPUnpack.cpp) V_PK unpack
// only runs in a schedule sub-region that also contains IGLP_OPT with immediate
// AMDGPUIGLPOptMaskExperimental (4).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUIGLPUNPACK_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUIGLPUNPACK_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

inline constexpr unsigned AMDGPUIGLPOptMaskExperimental = 4;

class AMDGPUIGLPUnpackPass
    : public PassInfoMixin<AMDGPUIGLPUnpackPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUIGLPUNPACK_H
