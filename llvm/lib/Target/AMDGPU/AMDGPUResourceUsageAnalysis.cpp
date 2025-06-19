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

char llvm::AMDGPUResourceUsageAnalysis::ID = 0;
char &llvm::AMDGPUResourceUsageAnalysisID = AMDGPUResourceUsageAnalysis::ID;

// In code object v4 and older, we need to tell the runtime some amount ahead of
// time if we don't know the true stack size. Assume a smaller number if this is
// only due to dynamic / non-entry block allocas.
static cl::opt<uint32_t> clAssumedStackSizeForExternalCall(
    "amdgpu-assume-external-call-stack-size",
    cl::desc("Assumed stack use of any external call (in bytes)"), cl::Hidden,
    cl::init(16384));

static cl::opt<uint32_t> clAssumedStackSizeForDynamicSizeObjects(
    "amdgpu-assume-dynamic-stack-object-size",
    cl::desc("Assumed extra stack use if there are any "
             "variable sized objects (in bytes)"),
    cl::Hidden, cl::init(4096));

INITIALIZE_PASS(AMDGPUResourceUsageAnalysis, DEBUG_TYPE,
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

bool AMDGPUResourceUsageAnalysis::runOnMachineFunction(MachineFunction &MF) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  const TargetMachine &TM = TPC->getTM<TargetMachine>();
  const MCSubtargetInfo &STI = *TM.getMCSubtargetInfo();

  // By default, for code object v5 and later, track only the minimum scratch
  // size
  uint32_t AssumedStackSizeForDynamicSizeObjects =
      clAssumedStackSizeForDynamicSizeObjects;
  uint32_t AssumedStackSizeForExternalCall = clAssumedStackSizeForExternalCall;
  if (AMDGPU::getAMDHSACodeObjectVersion(*MF.getFunction().getParent()) >=
          AMDGPU::AMDHSA_COV5 ||
      STI.getTargetTriple().getOS() == Triple::AMDPAL) {
    if (!clAssumedStackSizeForDynamicSizeObjects.getNumOccurrences())
      AssumedStackSizeForDynamicSizeObjects = 0;
    if (!clAssumedStackSizeForExternalCall.getNumOccurrences())
      AssumedStackSizeForExternalCall = 0;
  }

  ResourceInfo = analyzeResourceUsage(MF, AssumedStackSizeForDynamicSizeObjects,
                                      AssumedStackSizeForExternalCall);

  return false;
}

AMDGPUResourceUsageAnalysis::SIFunctionResourceInfo
AMDGPUResourceUsageAnalysis::analyzeResourceUsage(
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

  // A tail call isn't considered a call for MachineFrameInfo's purposes.
  bool HasCalls = FrameInfo.hasCalls() || FrameInfo.hasTailCall();
  // Functions that use the llvm.amdgcn.init.whole.wave intrinsic often have
  // VGPR arguments that are only added for the purpose of preserving the
  // inactive lanes. These should not be included in the number of used VGPRs.
  bool NeedsExplicitVGPRCount = MFI->hasInitWholeWave();
  if (!HasCalls && !NeedsExplicitVGPRCount) {

    Info.NumVGPR = TRI.getNumUsedPhysRegs(MRI, AMDGPU::VGPR_32RegClass,
                                          /*IncludeCalls=*/false);
    return Info;
  }

  int32_t MaxVGPR = -1;
  Info.CalleeSegmentSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (NeedsExplicitVGPRCount) {
        for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
          const MachineOperand &MO = MI.getOperand(i);

          if (!MO.isReg())
            continue;
          Register Reg = MO.getReg();
          const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);

          if (!RC || !TRI.isVGPRClass(RC))
            continue;

          // Skip inactive VGPRs in chain functions with the init.whole.wave
          // intrinsic. These will only appear as implicit use operands on the
          // chain call, and as the def of an IMPLICIT_DEF. We're going to skip
          // implicit defs unconditionally though because if they're important
          // in a different context then they will be counted when they are
          // used.
          bool IsChainCall =
              MFI->isChainFunction() && MI.getOpcode() == AMDGPU::SI_TCRETURN;
          if (IsChainCall || MI.isImplicitDef())
            continue;

          unsigned Width = TRI.getRegSizeInBits(*RC) / 32;
          unsigned HWReg = TRI.getHWRegIndex(Reg);
          int MaxUsed = HWReg + Width - 1;
          MaxVGPR = MaxUsed > MaxVGPR ? MaxUsed : MaxVGPR;
        }
      }

      if (MI.isCall()) {
        // Pseudo used just to encode the underlying global. Is there a better
        // way to track this?

        const MachineOperand *CalleeOp =
            TII->getNamedOperand(MI, AMDGPU::OpName::callee);

        const Function *Callee = getCalleeFunction(*CalleeOp);

        // Avoid crashing on undefined behavior with an illegal call to a
        // kernel. If a callsite's calling convention doesn't match the
        // function's, it's undefined behavior. If the callsite calling
        // convention does match, that would have errored earlier.
        if (Callee && AMDGPU::isEntryFunctionCC(Callee->getCallingConv()))
          report_fatal_error("invalid call to entry function");

        auto isSameFunction = [](const MachineFunction &MF, const Function *F) {
          return F == &MF.getFunction();
        };

        if (Callee && !isSameFunction(MF, Callee))
          Info.Callees.push_back(Callee);

        bool IsIndirect = !Callee || Callee->isDeclaration();

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
          Info.HasIndirectCall = true;
        }
      }
    }
  }

  if (NeedsExplicitVGPRCount)
    Info.NumVGPR = MaxVGPR + 1;
  else
    Info.NumVGPR = TRI.getNumUsedPhysRegs(MRI, AMDGPU::VGPR_32RegClass, false);

  return Info;
}
