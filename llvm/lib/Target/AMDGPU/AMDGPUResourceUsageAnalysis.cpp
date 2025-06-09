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

  Info.UsesVCC = MRI.isPhysRegUsed(AMDGPU::VCC);

  Info.NumVGPR = TRI.getNumDefinedPhysRegs(MRI, AMDGPU::VGPR_32RegClass);
  Info.NumExplicitSGPR =
      TRI.getNumDefinedPhysRegs(MRI, AMDGPU::SGPR_32RegClass);
  if (ST.hasMAIInsts())
    Info.NumAGPR = TRI.getNumDefinedPhysRegs(MRI, AMDGPU::AGPR_32RegClass);

  // Preloaded registers are written by the hardware, not defined in the
  // function body, so they need special handling.
  if (MFI->isEntryFunction()) {
    Info.NumExplicitSGPR =
        std::max<int32_t>(Info.NumExplicitSGPR, MFI->getNumPreloadedSGPRs());
    Info.NumVGPR = std::max<int32_t>(Info.NumVGPR, MFI->getNumPreloadedVGPRs());
  }

  if (!FrameInfo.hasCalls() && !FrameInfo.hasTailCall())
    return Info;

  Info.CalleeSegmentSize = 0;

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
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

  return Info;
}
