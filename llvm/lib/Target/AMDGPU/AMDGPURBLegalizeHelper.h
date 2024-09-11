//===- AMDGPURBLegalizeHelper ------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPURBLEGALIZEHELPER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPURBLEGALIZEHELPER_H

#include "AMDGPURBLegalizeRules.h"
#include "AMDGPURegisterBankInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

namespace llvm {
namespace AMDGPU {

// Receives list of RegBankLLTMapingApplyID and applies register banks on all
// operands. It is user's responsibility to provide RegBankLLTMapingApplyIDs for
// all register operands, there is no need to specify NonReg for trailing imm
// operands. This finishes selection of register banks if there is no need to
// replace instruction. In other case InstApplyMethod will create new
// instruction(s).
class RegBankLegalizeHelper {
  MachineIRBuilder &B;
  MachineRegisterInfo &MRI;
  const MachineUniformityInfo &MUI;
  const RegisterBankInfo &RBI;
  const RegBankLegalizeRules &RBLRules;
  const RegisterBank *SgprRB;
  const RegisterBank *VgprRB;
  const RegisterBank *VccRB;

  LLT S1 = LLT::scalar(1);
  LLT S16 = LLT::scalar(16);
  LLT S32 = LLT::scalar(32);
  LLT S64 = LLT::scalar(64);
  LLT V2S16 = LLT::fixed_vector(2, 16);
  LLT V2S32 = LLT::fixed_vector(2, 32);
  LLT V3S32 = LLT::fixed_vector(3, 32);
  LLT V4S32 = LLT::fixed_vector(4, 32);
  LLT V6S32 = LLT::fixed_vector(6, 32);
  LLT V7S32 = LLT::fixed_vector(7, 32);
  LLT V8S32 = LLT::fixed_vector(8, 32);

  LLT V3S64 = LLT::fixed_vector(3, 64);
  LLT V4S64 = LLT::fixed_vector(4, 64);
  LLT V16S64 = LLT::fixed_vector(16, 64);

  LLT P1 = LLT::pointer(1, 64);
  LLT P4 = LLT::pointer(4, 64);
  LLT P6 = LLT::pointer(6, 32);

public:
  RegBankLegalizeHelper(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                        const MachineUniformityInfo &MUI,
                        const RegisterBankInfo &RBI,
                        const RegBankLegalizeRules &RBLRules)
      : B(B), MRI(MRI), MUI(MUI), RBI(RBI), RBLRules(RBLRules),
        SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
        VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
        VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)){};

  bool findRuleAndApplyMapping(MachineInstr &MI);

  // Manual apply helpers.
  void applyMappingPHI(MachineInstr &MI);
  void applyMappingTrivial(MachineInstr &MI);

private:
  Register createVgpr(LLT Ty) {
    return MRI.createVirtualRegister({VgprRB, Ty});
  }
  Register createSgpr(LLT Ty) {
    return MRI.createVirtualRegister({SgprRB, Ty});
  }
  Register createVcc() { return MRI.createVirtualRegister({VccRB, S1}); }

  const RegisterBank *getRegBank(Register Reg) {
    const RegisterBank *RB = MRI.getRegBankOrNull(Reg);
    // This assert is not guaranteed by default. RB-select ensures that all
    // instructions that we want to RB-legalize have reg banks on all registers.
    // There might be a few exceptions. Workaround for them is to not write
    // 'mapping' for register operand that is expected to have reg class.
    assert(RB);
    return RB;
  }

  bool executeInWaterfallLoop(MachineIRBuilder &B,
                              iterator_range<MachineBasicBlock::iterator> Range,
                              SmallSet<Register, 4> &SGPROperandRegs);

  LLT getTyFromID(RegBankLLTMapingApplyID ID);

  const RegisterBank *getRBFromID(RegBankLLTMapingApplyID ID);

  void
  applyMappingDst(MachineInstr &MI, unsigned &OpIdx,
                  const SmallVectorImpl<RegBankLLTMapingApplyID> &MethodIDs);

  void
  applyMappingSrc(MachineInstr &MI, unsigned &OpIdx,
                  const SmallVectorImpl<RegBankLLTMapingApplyID> &MethodIDs,
                  SmallSet<Register, 4> &SGPRWaterfallOperandRegs);

  unsigned setBufferOffsets(MachineIRBuilder &B, Register CombinedOffset,
                            Register &VOffsetReg, Register &SOffsetReg,
                            int64_t &InstOffsetVal, Align Alignment);

  void lower(MachineInstr &MI, const RegBankLLTMapping &Mapping,
             SmallSet<Register, 4> &SGPRWaterfallOperandRegs);
};

} // end namespace AMDGPU
} // end namespace llvm

#endif
