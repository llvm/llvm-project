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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"

static std::unique_ptr<MachineFunction> cloneMF(MachineFunction *SrcMF) {
  auto DstMF = std::make_unique<MachineFunction>(
      SrcMF->getFunction(), SrcMF->getTarget(), SrcMF->getSubtarget(),
      SrcMF->getFunctionNumber(), SrcMF->getMMI());
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> Src2DstMBB;
  DenseMap<Register, Register> Src2DstReg;

  auto *SrcMRI = &SrcMF->getRegInfo();
  auto *DstMRI = &DstMF->getRegInfo();

  // Create vregs.
  for (auto &SrcMBB : *SrcMF) {
    for (auto &SrcMI : SrcMBB) {
      for (unsigned I = 0, E = SrcMI.getNumOperands(); I < E; ++I) {
        auto &DMO = SrcMI.getOperand(I);
        if (!DMO.isReg() || !DMO.isDef())
          continue;
        Register SrcReg = DMO.getReg();
        if (Register::isPhysicalRegister(SrcReg))
          continue;
        auto SrcRC = SrcMRI->getRegClass(SrcReg);
        auto DstReg = DstMRI->createVirtualRegister(SrcRC);
        Src2DstReg[SrcReg] = DstReg;
      }
    }
  }

  // Clone blocks.
  for (auto &SrcMBB : *SrcMF)
    Src2DstMBB[&SrcMBB] = DstMF->CreateMachineBasicBlock();
  // Link blocks.
  for (auto &SrcMBB : *SrcMF) {
    auto *DstMBB = Src2DstMBB[&SrcMBB];
    DstMF->push_back(DstMBB);
    for (auto It = SrcMBB.succ_begin(), IterEnd = SrcMBB.succ_end();
         It != IterEnd; ++It) {
      auto *SrcSuccMBB = *It;
      auto *DstSuccMBB = Src2DstMBB[SrcSuccMBB];
      DstMBB->addSuccessor(DstSuccMBB);
    }
    for (auto &LI : SrcMBB.liveins())
      DstMBB->addLiveIn(LI);
  }
  // Clone instructions.
  for (auto &SrcMBB : *SrcMF) {
    auto *DstMBB = Src2DstMBB[&SrcMBB];
    for (auto &SrcMI : SrcMBB) {
      const auto &MCID =
          DstMF->getSubtarget().getInstrInfo()->get(SrcMI.getOpcode());
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
