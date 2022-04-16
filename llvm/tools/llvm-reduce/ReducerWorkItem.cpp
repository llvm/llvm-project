//===- ReducerWorkItem.cpp - Wrapper for Module and MachineFunction -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReducerWorkItem.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

static void cloneFrameInfo(
    MachineFrameInfo &DstMFI, const MachineFrameInfo &SrcMFI,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB) {
  DstMFI.setFrameAddressIsTaken(SrcMFI.isFrameAddressTaken());
  DstMFI.setReturnAddressIsTaken(SrcMFI.isReturnAddressTaken());
  DstMFI.setHasStackMap(SrcMFI.hasStackMap());
  DstMFI.setHasPatchPoint(SrcMFI.hasPatchPoint());
  DstMFI.setUseLocalStackAllocationBlock(
      SrcMFI.getUseLocalStackAllocationBlock());
  DstMFI.setOffsetAdjustment(SrcMFI.getOffsetAdjustment());

  DstMFI.ensureMaxAlignment(SrcMFI.getMaxAlign());
  assert(DstMFI.getMaxAlign() == SrcMFI.getMaxAlign() &&
         "we need to set exact alignment");

  DstMFI.setAdjustsStack(SrcMFI.adjustsStack());
  DstMFI.setHasCalls(SrcMFI.hasCalls());
  DstMFI.setHasOpaqueSPAdjustment(SrcMFI.hasOpaqueSPAdjustment());
  DstMFI.setHasCopyImplyingStackAdjustment(
      SrcMFI.hasCopyImplyingStackAdjustment());
  DstMFI.setHasVAStart(SrcMFI.hasVAStart());
  DstMFI.setHasMustTailInVarArgFunc(SrcMFI.hasMustTailInVarArgFunc());
  DstMFI.setHasTailCall(SrcMFI.hasTailCall());

  if (SrcMFI.isMaxCallFrameSizeComputed())
    DstMFI.setMaxCallFrameSize(SrcMFI.getMaxCallFrameSize());

  DstMFI.setCVBytesOfCalleeSavedRegisters(
      SrcMFI.getCVBytesOfCalleeSavedRegisters());

  if (MachineBasicBlock *SavePt = SrcMFI.getSavePoint())
    DstMFI.setSavePoint(Src2DstMBB.find(SavePt)->second);
  if (MachineBasicBlock *RestorePt = SrcMFI.getRestorePoint())
    DstMFI.setRestorePoint(Src2DstMBB.find(RestorePt)->second);


  auto CopyObjectProperties = [](MachineFrameInfo &DstMFI,
                                 const MachineFrameInfo &SrcMFI, int FI) {
    if (SrcMFI.isStatepointSpillSlotObjectIndex(FI))
      DstMFI.markAsStatepointSpillSlotObjectIndex(FI);
    DstMFI.setObjectSSPLayout(FI, SrcMFI.getObjectSSPLayout(FI));
    DstMFI.setObjectZExt(FI, SrcMFI.isObjectZExt(FI));
    DstMFI.setObjectSExt(FI, SrcMFI.isObjectSExt(FI));
  };

  for (int i = 0, e = SrcMFI.getNumObjects() - SrcMFI.getNumFixedObjects();
       i != e; ++i) {
    int NewFI;

    assert(!SrcMFI.isFixedObjectIndex(i));
    if (SrcMFI.isVariableSizedObjectIndex(i)) {
      NewFI = DstMFI.CreateVariableSizedObject(SrcMFI.getObjectAlign(i),
                                               SrcMFI.getObjectAllocation(i));
    } else {
      NewFI = DstMFI.CreateStackObject(
          SrcMFI.getObjectSize(i), SrcMFI.getObjectAlign(i),
          SrcMFI.isSpillSlotObjectIndex(i), SrcMFI.getObjectAllocation(i),
          SrcMFI.getStackID(i));
      DstMFI.setObjectOffset(NewFI, SrcMFI.getObjectOffset(i));
    }

    CopyObjectProperties(DstMFI, SrcMFI, i);

    (void)NewFI;
    assert(i == NewFI && "expected to keep stable frame index numbering");
  }

  // Copy the fixed frame objects backwards to preserve frame index numbers,
  // since CreateFixedObject uses front insertion.
  for (int i = -1; i >= (int)-SrcMFI.getNumFixedObjects(); --i) {
    assert(SrcMFI.isFixedObjectIndex(i));
    int NewFI = DstMFI.CreateFixedObject(
      SrcMFI.getObjectSize(i), SrcMFI.getObjectOffset(i),
      SrcMFI.isImmutableObjectIndex(i), SrcMFI.isAliasedObjectIndex(i));
    CopyObjectProperties(DstMFI, SrcMFI, i);

    (void)NewFI;
    assert(i == NewFI && "expected to keep stable frame index numbering");
  }

  for (unsigned I = 0, E = SrcMFI.getLocalFrameObjectCount(); I < E; ++I) {
    auto LocalObject = SrcMFI.getLocalFrameObjectMap(I);
    DstMFI.mapLocalFrameObject(LocalObject.first, LocalObject.second);
  }

  DstMFI.setCalleeSavedInfo(SrcMFI.getCalleeSavedInfo());

  if (SrcMFI.hasStackProtectorIndex()) {
    DstMFI.setStackProtectorIndex(SrcMFI.getStackProtectorIndex());
  }

  // FIXME: Needs test, missing MIR serialization.
  if (SrcMFI.hasFunctionContextIndex()) {
    DstMFI.setFunctionContextIndex(SrcMFI.getFunctionContextIndex());
  }
}

static std::unique_ptr<MachineFunction> cloneMF(MachineFunction *SrcMF) {
  auto DstMF = std::make_unique<MachineFunction>(
      SrcMF->getFunction(), SrcMF->getTarget(), SrcMF->getSubtarget(),
      SrcMF->getFunctionNumber(), SrcMF->getMMI());
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> Src2DstMBB;
  DenseMap<Register, Register> Src2DstReg;

  auto *SrcMRI = &SrcMF->getRegInfo();
  auto *DstMRI = &DstMF->getRegInfo();

  // Clone blocks.
  for (MachineBasicBlock &SrcMBB : *SrcMF) {
    MachineBasicBlock *DstMBB =
        DstMF->CreateMachineBasicBlock(SrcMBB.getBasicBlock());
    Src2DstMBB[&SrcMBB] = DstMBB;

    if (SrcMBB.hasAddressTaken())
      DstMBB->setHasAddressTaken();

    // FIXME: This is not serialized
    if (SrcMBB.hasLabelMustBeEmitted())
      DstMBB->setLabelMustBeEmitted();

    DstMBB->setAlignment(SrcMBB.getAlignment());

    // FIXME: This is not serialized
    DstMBB->setMaxBytesForAlignment(SrcMBB.getMaxBytesForAlignment());

    DstMBB->setIsEHPad(SrcMBB.isEHPad());
    DstMBB->setIsEHScopeEntry(SrcMBB.isEHScopeEntry());
    DstMBB->setIsEHCatchretTarget(SrcMBB.isEHCatchretTarget());
    DstMBB->setIsEHFuncletEntry(SrcMBB.isEHFuncletEntry());

    // FIXME: These are not serialized
    DstMBB->setIsCleanupFuncletEntry(SrcMBB.isCleanupFuncletEntry());
    DstMBB->setIsBeginSection(SrcMBB.isBeginSection());
    DstMBB->setIsEndSection(SrcMBB.isEndSection());

    DstMBB->setSectionID(SrcMBB.getSectionID());
    DstMBB->setIsInlineAsmBrIndirectTarget(
        SrcMBB.isInlineAsmBrIndirectTarget());

    // FIXME: This is not serialized
    if (Optional<uint64_t> Weight = SrcMBB.getIrrLoopHeaderWeight())
      DstMBB->setIrrLoopHeaderWeight(*Weight);
  }

  const MachineFrameInfo &SrcMFI = SrcMF->getFrameInfo();
  MachineFrameInfo &DstMFI = DstMF->getFrameInfo();

  // Copy stack objects and other info
  cloneFrameInfo(DstMFI, SrcMFI, Src2DstMBB);

  // Remap the debug info frame index references.
  DstMF->VariableDbgInfos = SrcMF->VariableDbgInfos;

  // FIXME: Need to clone MachineFunctionInfo, which may also depend on frame
  // index and block mapping.

  // Create vregs.
  for (auto &SrcMBB : *SrcMF) {
    for (auto &SrcMI : SrcMBB) {
      for (unsigned I = 0, E = SrcMI.getNumOperands(); I < E; ++I) {
        auto &DMO = SrcMI.getOperand(I);
        if (DMO.isRegMask()) {
          DstMRI->addPhysRegsUsedFromRegMask(DMO.getRegMask());
          continue;
        }

        if (!DMO.isReg())
          continue;
        Register SrcReg = DMO.getReg();
        if (Register::isPhysicalRegister(SrcReg))
          continue;

        if (Src2DstReg.find(SrcReg) != Src2DstReg.end())
          continue;

        Register DstReg = DstMRI->createIncompleteVirtualRegister(
            SrcMRI->getVRegName(SrcReg));
        DstMRI->setRegClassOrRegBank(DstReg,
                                     SrcMRI->getRegClassOrRegBank(SrcReg));

        LLT RegTy = SrcMRI->getType(SrcReg);
        if (RegTy.isValid())
          DstMRI->setType(DstReg, RegTy);
        Src2DstReg[SrcReg] = DstReg;
      }
    }
  }

  // Copy register allocation hints.
  for (std::pair<Register, Register> RegMapEntry : Src2DstReg) {
    const auto &Hints = SrcMRI->getRegAllocationHints(RegMapEntry.first);
    for (Register PrefReg : Hints.second) {
      if (PrefReg.isVirtual()) {
        auto PrefRegEntry = Src2DstReg.find(PrefReg);
        assert(PrefRegEntry !=Src2DstReg.end());
        DstMRI->addRegAllocationHint(RegMapEntry.second, PrefRegEntry->second);
      } else
        DstMRI->addRegAllocationHint(RegMapEntry.second, PrefReg);
    }
  }

  const TargetSubtargetInfo &STI = DstMF->getSubtarget();
  const TargetInstrInfo *TII = STI.getInstrInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  // Link blocks.
  for (auto &SrcMBB : *SrcMF) {
    auto *DstMBB = Src2DstMBB[&SrcMBB];
    DstMF->push_back(DstMBB);

    for (auto It = SrcMBB.succ_begin(), IterEnd = SrcMBB.succ_end();
         It != IterEnd; ++It) {
      auto *SrcSuccMBB = *It;
      auto *DstSuccMBB = Src2DstMBB[SrcSuccMBB];
      DstMBB->addSuccessor(DstSuccMBB, SrcMBB.getSuccProbability(It));
    }
    for (auto &LI : SrcMBB.liveins())
      DstMBB->addLiveIn(LI);

    // Make sure MRI knows about registers clobbered by unwinder.
    if (DstMBB->isEHPad()) {
      if (auto *RegMask = TRI->getCustomEHPadPreservedMask(*DstMF))
        DstMRI->addPhysRegsUsedFromRegMask(RegMask);
    }
  }

  // Clone instructions.
  for (auto &SrcMBB : *SrcMF) {
    auto *DstMBB = Src2DstMBB[&SrcMBB];
    for (auto &SrcMI : SrcMBB) {
      const auto &MCID = TII->get(SrcMI.getOpcode());
      auto *DstMI = DstMF->CreateMachineInstr(MCID, SrcMI.getDebugLoc(),
                                              /*NoImplicit=*/true);
      DstMBB->push_back(DstMI);
      for (auto &SrcMO : SrcMI.operands()) {
        MachineOperand DstMO(SrcMO);
        DstMO.clearParent();
        // Update vreg.
        if (DstMO.isReg() && Src2DstReg.count(DstMO.getReg())) {
          DstMO.setReg(Src2DstReg[DstMO.getReg()]);
        }
        // Update MBB.
        if (DstMO.isMBB()) {
          DstMO.setMBB(Src2DstMBB[DstMO.getMBB()]);
        }

        DstMI->addOperand(DstMO);
      }
      DstMI->setMemRefs(*DstMF, SrcMI.memoperands());
    }
  }

  DstMF->setAlignment(SrcMF->getAlignment());
  DstMF->setExposesReturnsTwice(SrcMF->exposesReturnsTwice());
  DstMF->setHasInlineAsm(SrcMF->hasInlineAsm());
  DstMF->setHasWinCFI(SrcMF->hasWinCFI());

  DstMF->getProperties().reset().set(SrcMF->getProperties());

  if (!SrcMF->getFrameInstructions().empty() ||
      !SrcMF->getLongjmpTargets().empty() ||
      !SrcMF->getCatchretTargets().empty())
    report_fatal_error("cloning not implemented for machine function property");

  DstMF->setCallsEHReturn(SrcMF->callsEHReturn());
  DstMF->setCallsUnwindInit(SrcMF->callsUnwindInit());
  DstMF->setHasEHCatchret(SrcMF->hasEHCatchret());
  DstMF->setHasEHScopes(SrcMF->hasEHScopes());
  DstMF->setHasEHFunclets(SrcMF->hasEHFunclets());

  if (!SrcMF->getLandingPads().empty() ||
      !SrcMF->getCodeViewAnnotations().empty() ||
      !SrcMF->getTypeInfos().empty() ||
      !SrcMF->getFilterIds().empty() ||
      SrcMF->hasAnyWasmLandingPadIndex() ||
      SrcMF->hasAnyCallSiteLandingPad() ||
      SrcMF->hasAnyCallSiteLabel() ||
      !SrcMF->getCallSitesInfo().empty())
    report_fatal_error("cloning not implemented for machine function property");

  DstMF->setDebugInstrNumberingCount(SrcMF->DebugInstrNumberingCount);

  DstMF->verify(nullptr, "", /*AbortOnError=*/true);
  return DstMF;
}

std::unique_ptr<ReducerWorkItem> parseReducerWorkItem(StringRef Filename,
                                                      LLVMContext &Ctxt,
                                                      MachineModuleInfo *MMI) {
  auto MMM = std::make_unique<ReducerWorkItem>();
  if (MMI) {
    auto FileOrErr = MemoryBuffer::getFileOrSTDIN(Filename, /*IsText=*/true);
    std::unique_ptr<MIRParser> MParser =
        createMIRParser(std::move(FileOrErr.get()), Ctxt);

    auto SetDataLayout =
        [&](StringRef DataLayoutTargetTriple) -> Optional<std::string> {
      return MMI->getTarget().createDataLayout().getStringRepresentation();
    };

    std::unique_ptr<Module> M = MParser->parseIRModule(SetDataLayout);
    MParser->parseMachineFunctions(*M, *MMI);
    MachineFunction *MF = nullptr;
    for (auto &F : *M) {
      if (auto *MF4F = MMI->getMachineFunction(F)) {
        // XXX: Maybe it would not be a lot of effort to handle multiple MFs by
        // simply storing them in a ReducerWorkItem::SmallVector or similar. The
        // single MF use-case seems a lot more common though so that will do for
        // now.
        assert(!MF && "Only single MF supported!");
        MF = MF4F;
      }
    }
    assert(MF && "No MF found!");

    MMM->M = std::move(M);
    MMM->MF = cloneMF(MF);
  } else {
    SMDiagnostic Err;
    std::unique_ptr<Module> Result = parseIRFile(Filename, Err, Ctxt);
    if (!Result) {
      Err.print("llvm-reduce", errs());
      return std::unique_ptr<ReducerWorkItem>();
    }
    MMM->M = std::move(Result);
  }
  if (verifyReducerWorkItem(*MMM, &errs())) {
    errs() << "Error: " << Filename << " - input module is broken!\n";
    return std::unique_ptr<ReducerWorkItem>();
  }
  return MMM;
}

std::unique_ptr<ReducerWorkItem>
cloneReducerWorkItem(const ReducerWorkItem &MMM) {
  auto CloneMMM = std::make_unique<ReducerWorkItem>();
  if (MMM.MF) {
    // Note that we cannot clone the Module as then we would need a way to
    // updated the cloned MachineFunction's IR references.
    // XXX: Actually have a look at
    // std::unique_ptr<Module> CloneModule(const Module &M, ValueToValueMapTy
    // &VMap);
    CloneMMM->M = MMM.M;
    CloneMMM->MF = cloneMF(MMM.MF.get());
  } else {
    CloneMMM->M = CloneModule(*MMM.M);
  }
  return CloneMMM;
}

bool verifyReducerWorkItem(const ReducerWorkItem &MMM, raw_fd_ostream *OS) {
  if (verifyModule(*MMM.M, OS))
    return true;
  if (MMM.MF && !MMM.MF->verify(nullptr, "", /*AbortOnError=*/false))
    return true;
  return false;
}

void ReducerWorkItem::print(raw_ostream &ROS, void *p) const {
  if (MF) {
    printMIR(ROS, *M);
    printMIR(ROS, *MF);
  } else {
    M->print(ROS, /*AssemblyAnnotationWriter=*/nullptr,
             /*ShouldPreserveUseListOrder=*/true);
  }
}
