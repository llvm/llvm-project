//===- AMDGPUResourceUsageAnalysis.h ---- analysis of resources -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Analyzes how many registers and other resources are used by
/// functions.
///
/// The results of this analysis are used to fill the register usage, flat
/// usage, etc. into hardware registers.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUResourceUsageAnalysis.h"
#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "amdgpu-resource-usage"

char llvm::AMDGPUResourceUsageAnalysisWrapperPass::ID = 0;
char &llvm::AMDGPUResourceUsageAnalysisID =
    AMDGPUResourceUsageAnalysisWrapperPass::ID;

INITIALIZE_PASS(AMDGPUResourceUsageAnalysisWrapperPass, DEBUG_TYPE,
                "Function register usage analysis", true, true)

static const Function *getCalleeFunction(const MachineOperand &Op) {
  if (Op.isImm()) {
    assert(Op.getImm() == 0);
    return nullptr;
  }
  return cast<Function>(Op.getGlobal()->stripPointerCastsAndAliases());
}

static bool hasAnyNonFlatUseOfReg(const MachineRegisterInfo &MRI,
                                  const SIInstrInfo &TII, unsigned Reg) {
  for (const MachineOperand &UseOp : MRI.reg_operands(Reg)) {
    if (!UseOp.isImplicit() || !TII.isFLAT(*UseOp.getParent()))
      return true;
  }

  return false;
}

bool AMDGPUResourceUsageAnalysisWrapperPass::runOnMachineFunction(
    MachineFunction &MF) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  const auto &TM = TPC->getTM<GCNTargetMachine>();
  const MCSubtargetInfo &STI = TM.getMCSubtargetInfo();

  // By default, for code object v5 and later, track only the minimum scratch
  // size
  const AMDGPUOptions &CLOpts = TM.getCLOpts();
  bool MinScratch = AMDGPU::getAMDHSACodeObjectVersion(
                        *MF.getFunction().getParent()) >= AMDGPU::AMDHSA_COV5 ||
                    STI.getTargetTriple().getOS() == Triple::AMDPAL;
  uint32_t AssumedStackSizeForDynamicSizeObjects =
      CLOpts.assume_dynamic_stack_object_size.value_or(MinScratch ? 0 : 4096);
  uint32_t AssumedStackSizeForExternalCall =
      CLOpts.assume_external_call_stack_size.value_or(MinScratch ? 0 : 16384);

  ResourceInfo = AMDGPUResourceUsageAnalysisImpl().analyzeResourceUsage(
      MF, AssumedStackSizeForDynamicSizeObjects,
      AssumedStackSizeForExternalCall);

  return false;
}

AnalysisKey AMDGPUResourceUsageAnalysis::Key;
AMDGPUResourceUsageAnalysis::Result
AMDGPUResourceUsageAnalysis::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  const MCSubtargetInfo &STI = TM.getMCSubtargetInfo();

  // By default, for code object v5 and later, track only the minimum scratch
  // size
  const AMDGPUOptions &CLOpts = TM.getCLOpts();
  bool MinScratch = AMDGPU::getAMDHSACodeObjectVersion(
                        *MF.getFunction().getParent()) >= AMDGPU::AMDHSA_COV5 ||
                    STI.getTargetTriple().getOS() == Triple::AMDPAL;
  uint32_t AssumedStackSizeForDynamicSizeObjects =
      CLOpts.assume_dynamic_stack_object_size.value_or(MinScratch ? 0 : 4096);
  uint32_t AssumedStackSizeForExternalCall =
      CLOpts.assume_external_call_stack_size.value_or(MinScratch ? 0 : 16384);

  return AMDGPUResourceUsageAnalysisImpl().analyzeResourceUsage(
      MF, AssumedStackSizeForDynamicSizeObjects,
      AssumedStackSizeForExternalCall);
}

AMDGPUResourceUsageAnalysisImpl::SIFunctionResourceInfo
AMDGPUResourceUsageAnalysisImpl::analyzeResourceUsage(
    const MachineFunction &MF, uint32_t AssumedStackSizeForDynamicSizeObjects,
    uint32_t AssumedStackSizeForExternalCall) const {
  SIFunctionResourceInfo Info;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  Info.UsesFlatScratch = MRI.isPhysRegUsed(AMDGPU::FLAT_SCR_LO) ||
                         MRI.isPhysRegUsed(AMDGPU::FLAT_SCR_HI) ||
                         MRI.isLiveIn(MFI->getPreloadedReg(
                             AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT));

  Info.NumNamedBarrier = MFI->getNumNamedBarriers();

  // Even if FLAT_SCRATCH is implicitly used, it has no effect if flat
  // instructions aren't used to access the scratch buffer. Inline assembly may
  // need it though.
  //
  // If we only have implicit uses of flat_scr on flat instructions, it is not
  // really needed.
  if (Info.UsesFlatScratch && !MFI->getUserSGPRInfo().hasFlatScratchInit() &&
      (!hasAnyNonFlatUseOfReg(MRI, *TII, AMDGPU::FLAT_SCR) &&
       !hasAnyNonFlatUseOfReg(MRI, *TII, AMDGPU::FLAT_SCR_LO) &&
       !hasAnyNonFlatUseOfReg(MRI, *TII, AMDGPU::FLAT_SCR_HI))) {
    Info.UsesFlatScratch = false;
  }

  Info.PrivateSegmentSize = FrameInfo.getStackSize();

  // Assume a big number if there are any unknown sized objects.
  Info.HasDynamicallySizedStack = FrameInfo.hasVarSizedObjects();
  if (Info.HasDynamicallySizedStack)
    Info.PrivateSegmentSize += AssumedStackSizeForDynamicSizeObjects;

  if (MFI->isStackRealigned())
    Info.PrivateSegmentSize += FrameInfo.getMaxAlign().value();

  Info.UsesVCC =
      MRI.isPhysRegUsed(AMDGPU::VCC_LO) || MRI.isPhysRegUsed(AMDGPU::VCC_HI);
  Info.NumExplicitSGPR = TRI.getNumUsedPhysRegs(MRI, AMDGPU::SGPR_32RegClass,
                                                /*IncludeCalls=*/false);
  if (ST.hasMAIInsts())
    Info.NumAGPR = TRI.getNumUsedPhysRegs(MRI, AMDGPU::AGPR_32RegClass,
                                          /*IncludeCalls=*/false);

  Info.NumVGPR = TRI.getNumReportedVGPRs(MF);

  // A tail call isn't considered a call for MachineFrameInfo's purposes.
  if (!FrameInfo.hasCalls() && !FrameInfo.hasTailCall())
    return Info;

  Info.CalleeSegmentSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isCall()) {
        // Pseudo used just to encode the underlying global. Is there a better
        // way to track this?

        // TODO: Some of the generic call-like pseudos do not encode the callee,
        // so we overly conservatively treat this as an indirect call.
        const MachineOperand *CalleeOp =
            TII->getNamedOperand(MI, AMDGPU::OpName::callee);

        const Function *Callee =
            CalleeOp ? getCalleeFunction(*CalleeOp) : nullptr;

        auto isSameFunction = [](const MachineFunction &MF, const Function *F) {
          return F == &MF.getFunction();
        };

        if (Callee && !isSameFunction(MF, Callee))
          Info.Callees.push_back(Callee);

        bool IsIndirect = !Callee || Callee->isDeclaration();
        Info.HasIndirectCall |= IsIndirect;

        bool IsChainCall = MI.getOpcode() == AMDGPU::SI_TCRETURN_CHAIN;
        Info.HasNonChainIndirectCall |= (!IsChainCall && IsIndirect);

        // In object linking mode the linker has the full cross-TU view. It
        // propagates resource usage across both direct calls to external
        // declarations and true indirect calls. Skip the compile-time
        // conservative assumptions so that the locally emitted metadata
        // describes this function's own usage only.
        if (AMDGPUTargetMachine::EnableObjectLinking)
          continue;

        // FIXME: Call site could have norecurse on it
        if (!Callee || !Callee->doesNotRecurse()) {
          Info.HasRecursion = true;

          // TODO: If we happen to know there is no stack usage in the
          // callgraph, we don't need to assume an infinitely growing stack.
          if (!MI.isReturn()) {
            // We don't need to assume an unknown stack size for tail calls.

            // FIXME: This only benefits in the case where the kernel does not
            // directly call the tail called function. If a kernel directly
            // calls a tail recursive function, we'll assume maximum stack size
            // based on the regular call instruction.
            Info.CalleeSegmentSize = std::max(
                Info.CalleeSegmentSize,
                static_cast<uint64_t>(AssumedStackSizeForExternalCall));
          }
        }

        if (IsIndirect) {
          Info.CalleeSegmentSize =
              std::max(Info.CalleeSegmentSize,
                       static_cast<uint64_t>(AssumedStackSizeForExternalCall));

          // Register usage of indirect calls gets handled later
          Info.UsesVCC = true;
          Info.UsesFlatScratch = ST.hasFlatAddressSpace();
          Info.HasDynamicallySizedStack = true;
        }
      }
    }
  }

  return Info;
}
