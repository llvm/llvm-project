//===- AMDGPURegBankLegalizeHelper ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREGBANKLEGALIZEHELPER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREGBANKLEGALIZEHELPER_H

#include "AMDGPURegBankLegalizeRules.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {

class MachineIRBuilder;

namespace AMDGPU {

// Receives list of RegBankLLTMappingApplyID and applies register banks on all
// operands. It is user's responsibility to provide RegBankLLTMappingApplyIDs
// for all register operands, there is no need to specify NonReg for trailing
// imm operands. This finishes selection of register banks if there is no need
// to replace instruction. In other case InstApplyMethod will create new
// instruction(s).
class RegBankLegalizeHelper {
  const GCNSubtarget &ST;
  MachineIRBuilder &B;
  MachineRegisterInfo &MRI;
  const MachineUniformityInfo &MUI;
  const RegisterBankInfo &RBI;
  const RegBankLegalizeRules &RBLRules;
  const RegisterBank *SgprRB;
  const RegisterBank *VgprRB;
  const RegisterBank *VccRB;

  static constexpr LLT S1 = LLT::scalar(1);
  static constexpr LLT S16 = LLT::scalar(16);
  static constexpr LLT S32 = LLT::scalar(32);
  static constexpr LLT S64 = LLT::scalar(64);
  static constexpr LLT S96 = LLT::scalar(96);
  static constexpr LLT S128 = LLT::scalar(128);
  static constexpr LLT S256 = LLT::scalar(256);

  static constexpr LLT V2S16 = LLT::fixed_vector(2, 16);
  static constexpr LLT V4S16 = LLT::fixed_vector(4, 16);
  static constexpr LLT V6S16 = LLT::fixed_vector(6, 16);
  static constexpr LLT V8S16 = LLT::fixed_vector(8, 16);
  static constexpr LLT V16S16 = LLT::fixed_vector(16, 16);
  static constexpr LLT V32S16 = LLT::fixed_vector(32, 16);

  static constexpr LLT V2S32 = LLT::fixed_vector(2, 32);
  static constexpr LLT V3S32 = LLT::fixed_vector(3, 32);
  static constexpr LLT V4S32 = LLT::fixed_vector(4, 32);
  static constexpr LLT V6S32 = LLT::fixed_vector(6, 32);
  static constexpr LLT V7S32 = LLT::fixed_vector(7, 32);
  static constexpr LLT V8S32 = LLT::fixed_vector(8, 32);
  static constexpr LLT V16S32 = LLT::fixed_vector(16, 32);

  static constexpr LLT V2S64 = LLT::fixed_vector(2, 64);
  static constexpr LLT V3S64 = LLT::fixed_vector(3, 64);
  static constexpr LLT V4S64 = LLT::fixed_vector(4, 64);
  static constexpr LLT V8S64 = LLT::fixed_vector(8, 64);
  static constexpr LLT V16S64 = LLT::fixed_vector(16, 64);

  static constexpr LLT P1 = LLT::pointer(1, 64);
  static constexpr LLT P4 = LLT::pointer(4, 64);
  static constexpr LLT P6 = LLT::pointer(6, 32);

  MachineRegisterInfo::VRegAttrs SgprRB_S32 = {SgprRB, S32};
  MachineRegisterInfo::VRegAttrs VgprRB_S32 = {VgprRB, S32};
  MachineRegisterInfo::VRegAttrs VccRB_S1 = {VccRB, S1};

public:
  RegBankLegalizeHelper(MachineIRBuilder &B, const MachineUniformityInfo &MUI,
                        const RegisterBankInfo &RBI,
                        const RegBankLegalizeRules &RBLRules);

  void findRuleAndApplyMapping(MachineInstr &MI);

  // Manual apply helpers.
  void applyMappingPHI(MachineInstr &MI);
  void applyMappingTrivial(MachineInstr &MI);

private:
  bool executeInWaterfallLoop(MachineIRBuilder &B,
                              iterator_range<MachineBasicBlock::iterator> Range,
                              SmallSet<Register, 4> &SgprOperandRegs);

  LLT getTyFromID(RegBankLLTMappingApplyID ID);
  LLT getBTyFromID(RegBankLLTMappingApplyID ID, LLT Ty);

  const RegisterBank *getRegBankFromID(RegBankLLTMappingApplyID ID);

  void
  applyMappingDst(MachineInstr &MI, unsigned &OpIdx,
                  const SmallVectorImpl<RegBankLLTMappingApplyID> &MethodIDs);

  void
  applyMappingSrc(MachineInstr &MI, unsigned &OpIdx,
                  const SmallVectorImpl<RegBankLLTMappingApplyID> &MethodIDs,
                  SmallSet<Register, 4> &SgprWaterfallOperandRegs);

  void splitLoad(MachineInstr &MI, ArrayRef<LLT> LLTBreakdown,
                 LLT MergeTy = LLT());
  void widenLoad(MachineInstr &MI, LLT WideTy, LLT MergeTy = LLT());

  void lower(MachineInstr &MI, const RegBankLLTMapping &Mapping,
             SmallSet<Register, 4> &SgprWaterfallOperandRegs);

  void lowerVccExtToSel(MachineInstr &MI);
  void lowerV_BFE(MachineInstr &MI);
  void lowerS_BFE(MachineInstr &MI);
  void lowerSplitTo32(MachineInstr &MI);
};

} // end namespace AMDGPU
} // end namespace llvm

#endif
