//===- ReducerWorkItem.cpp - Wrapper for Module and MachineFunction -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReducerWorkItem.h"
#include "llvm/CodeGen/CommandFlags.h"
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

extern cl::OptionCategory LLVMReduceOptions;
static cl::opt<std::string> TargetTriple("mtriple",
                                         cl::desc("Set the target triple"),
                                         cl::cat(LLVMReduceOptions));

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

static void cloneMemOperands(MachineInstr &DstMI, MachineInstr &SrcMI,
                             MachineFunction &SrcMF, MachineFunction &DstMF) {
  // The new MachineMemOperands should be owned by the new function's
  // Allocator.
  PseudoSourceValueManager &PSVMgr = DstMF.getPSVManager();

  // We also need to remap the PseudoSourceValues from the new function's
  // PseudoSourceValueManager.
  SmallVector<MachineMemOperand *, 2> NewMMOs;
  for (MachineMemOperand *OldMMO : SrcMI.memoperands()) {
    MachinePointerInfo NewPtrInfo(OldMMO->getPointerInfo());
    if (const PseudoSourceValue *PSV =
            NewPtrInfo.V.dyn_cast<const PseudoSourceValue *>()) {
      switch (PSV->kind()) {
      case PseudoSourceValue::Stack:
        NewPtrInfo.V = PSVMgr.getStack();
        break;
      case PseudoSourceValue::GOT:
        NewPtrInfo.V = PSVMgr.getGOT();
        break;
      case PseudoSourceValue::JumpTable:
        NewPtrInfo.V = PSVMgr.getJumpTable();
        break;
      case PseudoSourceValue::ConstantPool:
        NewPtrInfo.V = PSVMgr.getConstantPool();
        break;
      case PseudoSourceValue::FixedStack:
        NewPtrInfo.V = PSVMgr.getFixedStack(
            cast<FixedStackPseudoSourceValue>(PSV)->getFrameIndex());
        break;
      case PseudoSourceValue::GlobalValueCallEntry:
        NewPtrInfo.V = PSVMgr.getGlobalValueCallEntry(
            cast<GlobalValuePseudoSourceValue>(PSV)->getValue());
        break;
      case PseudoSourceValue::ExternalSymbolCallEntry:
        NewPtrInfo.V = PSVMgr.getExternalSymbolCallEntry(
            cast<ExternalSymbolPseudoSourceValue>(PSV)->getSymbol());
        break;
      case PseudoSourceValue::TargetCustom:
      default:
        // FIXME: We have no generic interface for allocating custom PSVs.
        report_fatal_error("Cloning TargetCustom PSV not handled");
      }
    }

    MachineMemOperand *NewMMO = DstMF.getMachineMemOperand(
        NewPtrInfo, OldMMO->getFlags(), OldMMO->getMemoryType(),
        OldMMO->getBaseAlign(), OldMMO->getAAInfo(), OldMMO->getRanges(),
        OldMMO->getSyncScopeID(), OldMMO->getSuccessOrdering(),
        OldMMO->getFailureOrdering());
    NewMMOs.push_back(NewMMO);
  }

  DstMI.setMemRefs(DstMF, NewMMOs);
}

static std::unique_ptr<MachineFunction> cloneMF(MachineFunction *SrcMF,
                                                MachineModuleInfo &DestMMI) {
  auto DstMF = std::make_unique<MachineFunction>(
      SrcMF->getFunction(), SrcMF->getTarget(), SrcMF->getSubtarget(),
      SrcMF->getFunctionNumber(), DestMMI);
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> Src2DstMBB;

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
  // Clone virtual registers
  for (unsigned I = 0, E = SrcMRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    Register NewReg = DstMRI->createIncompleteVirtualRegister(
      SrcMRI->getVRegName(Reg));
    assert(NewReg == Reg && "expected to preserve virtreg number");

    DstMRI->setRegClassOrRegBank(NewReg, SrcMRI->getRegClassOrRegBank(Reg));

    LLT RegTy = SrcMRI->getType(Reg);
    if (RegTy.isValid())
      DstMRI->setType(NewReg, RegTy);

    // Copy register allocation hints.
    const auto &Hints = SrcMRI->getRegAllocationHints(Reg);
    for (Register PrefReg : Hints.second)
      DstMRI->addRegAllocationHint(NewReg, PrefReg);
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
      DstMI->setFlags(SrcMI.getFlags());
      DstMI->setAsmPrinterFlag(SrcMI.getAsmPrinterFlags());

      DstMBB->push_back(DstMI);
      for (auto &SrcMO : SrcMI.operands()) {
        MachineOperand DstMO(SrcMO);
        DstMO.clearParent();

        // Update MBB.
        if (DstMO.isMBB())
          DstMO.setMBB(Src2DstMBB[DstMO.getMBB()]);
        else if (DstMO.isRegMask())
          DstMRI->addPhysRegsUsedFromRegMask(DstMO.getRegMask());

        DstMI->addOperand(DstMO);
      }

      cloneMemOperands(*DstMI, SrcMI, *SrcMF, *DstMF);
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

std::unique_ptr<ReducerWorkItem>
parseReducerWorkItem(const char *ToolName, StringRef Filename,
                     LLVMContext &Ctxt, std::unique_ptr<TargetMachine> &TM,
                     bool IsMIR) {
  Triple TheTriple;

  auto MMM = std::make_unique<ReducerWorkItem>();

  if (IsMIR) {
    auto FileOrErr = MemoryBuffer::getFileOrSTDIN(Filename, /*IsText=*/true);
    if (std::error_code EC = FileOrErr.getError()) {
      WithColor::error(errs(), ToolName) << EC.message() << '\n';
      return nullptr;
    }

    std::unique_ptr<MIRParser> MParser =
        createMIRParser(std::move(FileOrErr.get()), Ctxt);

    auto SetDataLayout =
        [&](StringRef DataLayoutTargetTriple) -> Optional<std::string> {
      // If we are supposed to override the target triple, do so now.
      std::string IRTargetTriple = DataLayoutTargetTriple.str();
      if (!TargetTriple.empty())
        IRTargetTriple = Triple::normalize(TargetTriple);
      TheTriple = Triple(IRTargetTriple);
      if (TheTriple.getTriple().empty())
        TheTriple.setTriple(sys::getDefaultTargetTriple());

      std::string Error;
      const Target *TheTarget =
          TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
      if (!TheTarget) {
        WithColor::error(errs(), ToolName) << Error;
        exit(1);
      }

      // Hopefully the MIR parsing doesn't depend on any options.
      TargetOptions Options;
      Optional<Reloc::Model> RM = codegen::getExplicitRelocModel();
      std::string CPUStr = codegen::getCPUStr();
      std::string FeaturesStr = codegen::getFeaturesStr();
      TM = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
          TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM,
          codegen::getExplicitCodeModel(), CodeGenOpt::Default));
      assert(TM && "Could not allocate target machine!");

      return TM->createDataLayout().getStringRepresentation();
    };

    std::unique_ptr<Module> M = MParser->parseIRModule(SetDataLayout);
    LLVMTargetMachine *LLVMTM = static_cast<LLVMTargetMachine *>(TM.get());

    MMM->MMI = std::make_unique<MachineModuleInfo>(LLVMTM);
    MParser->parseMachineFunctions(*M, *MMM->MMI);
    MMM->M = std::move(M);
  } else {
    SMDiagnostic Err;
    std::unique_ptr<Module> Result = parseIRFile(Filename, Err, Ctxt);
    if (!Result) {
      Err.print(ToolName, errs());
      return std::unique_ptr<ReducerWorkItem>();
    }
    MMM->M = std::move(Result);
  }
  if (verifyReducerWorkItem(*MMM, &errs())) {
    WithColor::error(errs(), ToolName)
        << Filename << " - input module is broken!\n";
    return std::unique_ptr<ReducerWorkItem>();
  }
  return MMM;
}

std::unique_ptr<ReducerWorkItem>
cloneReducerWorkItem(const ReducerWorkItem &MMM, const TargetMachine *TM) {
  auto CloneMMM = std::make_unique<ReducerWorkItem>();
  if (TM) {
    // We're assuming the Module IR contents are always unchanged by MIR
    // reductions, and can share it as a constant.
    CloneMMM->M = MMM.M;

    // MachineModuleInfo contains a lot of other state used during codegen which
    // we won't be using here, but we should be able to ignore it (although this
    // is pretty ugly).
    const LLVMTargetMachine *LLVMTM =
        static_cast<const LLVMTargetMachine *>(TM);
    CloneMMM->MMI = std::make_unique<MachineModuleInfo>(LLVMTM);

    for (const Function &F : MMM.getModule()) {
      if (auto *MF = MMM.MMI->getMachineFunction(F))
        CloneMMM->MMI->insertFunction(F, cloneMF(MF, *CloneMMM->MMI));
    }
  } else {
    CloneMMM->M = CloneModule(*MMM.M);
  }
  return CloneMMM;
}

bool verifyReducerWorkItem(const ReducerWorkItem &MMM, raw_fd_ostream *OS) {
  if (verifyModule(*MMM.M, OS))
    return true;

  if (!MMM.MMI)
    return false;

  for (const Function &F : MMM.getModule()) {
    if (const MachineFunction *MF = MMM.MMI->getMachineFunction(F)) {
      if (!MF->verify(nullptr, "", /*AbortOnError=*/false))
        return true;
    }
  }

  return false;
}

void ReducerWorkItem::print(raw_ostream &ROS, void *p) const {
  if (MMI) {
    printMIR(ROS, *M);
    for (Function &F : *M) {
      if (auto *MF = MMI->getMachineFunction(F))
        printMIR(ROS, *MF);
    }
  } else {
    M->print(ROS, /*AssemblyAnnotationWriter=*/nullptr,
             /*ShouldPreserveUseListOrder=*/true);
  }
}

// FIXME: We might want to use a different metric than "number of
// bytes in serialized IR" to detect non-progress of the main delta
// loop
uint64_t ReducerWorkItem::getIRSize() const {
  std::string Str;
  raw_string_ostream SS(Str);
  print(SS, /*AnnotationWriter=*/nullptr);
  return Str.length();
}

/// Try to produce some number that indicates a function is getting smaller /
/// simpler.
static uint64_t computeMIRComplexityScoreImpl(const MachineFunction &MF) {
  uint64_t Score = 0;
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Add for stack objects
  Score += MFI.getNumObjects();

  // Add in the block count.
  Score += 2 * MF.size();

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      const unsigned Opc = MI.getOpcode();

      // Reductions may want or need to introduce implicit_defs, so don't count
      // them.
      // TODO: These probably should count in some way.
      if (Opc == TargetOpcode::IMPLICIT_DEF ||
          Opc == TargetOpcode::G_IMPLICIT_DEF)
        continue;

      // Each instruction adds to the score
      Score += 4;

      if (Opc == TargetOpcode::PHI || Opc == TargetOpcode::G_PHI ||
          Opc == TargetOpcode::INLINEASM || Opc == TargetOpcode::INLINEASM_BR)
        ++Score;

      if (MI.getFlags() != 0)
        ++Score;

      // Increase weight for more operands.
      for (const MachineOperand &MO : MI.operands()) {
        ++Score;

        // Treat registers as more complex.
        if (MO.isReg()) {
          ++Score;

          // And subregisters as even more complex.
          if (MO.getSubReg()) {
            ++Score;
            if (MO.isDef())
              ++Score;
          }
        } else if (MO.isRegMask())
          ++Score;
      }
    }
  }

  return Score;
}

uint64_t ReducerWorkItem::computeMIRComplexityScore() const {
  uint64_t Score = 0;

  for (const Function &F : getModule()) {
    if (auto *MF = MMI->getMachineFunction(F))
      Score += computeMIRComplexityScoreImpl(*MF);
  }

  return Score;
}
