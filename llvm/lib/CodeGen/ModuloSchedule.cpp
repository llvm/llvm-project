//===- ModuloSchedule.cpp - Software pipeline schedule expansion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ModuloSchedule.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pipeliner"
using namespace llvm;

/// Return the register values for  the operands of a Phi instruction.
/// This function assume the instruction is a Phi.
static void getPhiRegs(MachineInstr &Phi, MachineBasicBlock *Loop,
                       unsigned &InitVal, unsigned &LoopVal) {
  assert(Phi.isPHI() && "Expecting a Phi.");

  InitVal = 0;
  LoopVal = 0;
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() != Loop)
      InitVal = Phi.getOperand(i).getReg();
    else
      LoopVal = Phi.getOperand(i).getReg();

  assert(InitVal != 0 && LoopVal != 0 && "Unexpected Phi structure.");
}

/// Return the Phi register value that comes from the incoming block.
static unsigned getInitPhiReg(MachineInstr &Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() != LoopBB)
      return Phi.getOperand(i).getReg();
  return 0;
}

/// Return the Phi register value that comes the loop block.
static unsigned getLoopPhiReg(MachineInstr &Phi, MachineBasicBlock *LoopBB) {
  for (unsigned i = 1, e = Phi.getNumOperands(); i != e; i += 2)
    if (Phi.getOperand(i + 1).getMBB() == LoopBB)
      return Phi.getOperand(i).getReg();
  return 0;
}

void ModuloScheduleExpander::expand() {
  BB = Schedule.getLoop()->getTopBlock();
  Preheader = *BB->pred_begin();
  if (Preheader == BB)
    Preheader = *std::next(BB->pred_begin());

  // Iterate over the definitions in each instruction, and compute the
  // stage difference for each use.  Keep the maximum value.
  for (MachineInstr *MI : Schedule.getInstructions()) {
    int DefStage = Schedule.getStage(MI);
    for (unsigned i = 0, e = MI->getNumOperands(); i < e; ++i) {
      MachineOperand &Op = MI->getOperand(i);
      if (!Op.isReg() || !Op.isDef())
        continue;

      Register Reg = Op.getReg();
      unsigned MaxDiff = 0;
      bool PhiIsSwapped = false;
      for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(Reg),
                                             EI = MRI.use_end();
           UI != EI; ++UI) {
        MachineOperand &UseOp = *UI;
        MachineInstr *UseMI = UseOp.getParent();
        int UseStage = Schedule.getStage(UseMI);
        unsigned Diff = 0;
        if (UseStage != -1 && UseStage >= DefStage)
          Diff = UseStage - DefStage;
        if (MI->isPHI()) {
          if (isLoopCarried(*MI))
            ++Diff;
          else
            PhiIsSwapped = true;
        }
        MaxDiff = std::max(Diff, MaxDiff);
      }
      RegToStageDiff[Reg] = std::make_pair(MaxDiff, PhiIsSwapped);
    }
  }

  generatePipelinedLoop();
}

void ModuloScheduleExpander::generatePipelinedLoop() {
  // Create a new basic block for the kernel and add it to the CFG.
  MachineBasicBlock *KernelBB = MF.CreateMachineBasicBlock(BB->getBasicBlock());

  unsigned MaxStageCount = Schedule.getNumStages() - 1;

  // Remember the registers that are used in different stages. The index is
  // the iteration, or stage, that the instruction is scheduled in.  This is
  // a map between register names in the original block and the names created
  // in each stage of the pipelined loop.
  ValueMapTy *VRMap = new ValueMapTy[(MaxStageCount + 1) * 2];
  InstrMapTy InstrMap;

  SmallVector<MachineBasicBlock *, 4> PrologBBs;

  // Generate the prolog instructions that set up the pipeline.
  generateProlog(MaxStageCount, KernelBB, VRMap, PrologBBs);
  MF.insert(BB->getIterator(), KernelBB);

  // Rearrange the instructions to generate the new, pipelined loop,
  // and update register names as needed.
  for (MachineInstr *CI : Schedule.getInstructions()) {
    if (CI->isPHI())
      continue;
    unsigned StageNum = Schedule.getStage(CI);
    MachineInstr *NewMI = cloneInstr(CI, MaxStageCount, StageNum);
    updateInstruction(NewMI, false, MaxStageCount, StageNum, VRMap);
    KernelBB->push_back(NewMI);
    InstrMap[NewMI] = CI;
  }

  // Copy any terminator instructions to the new kernel, and update
  // names as needed.
  for (MachineBasicBlock::iterator I = BB->getFirstTerminator(),
                                   E = BB->instr_end();
       I != E; ++I) {
    MachineInstr *NewMI = MF.CloneMachineInstr(&*I);
    updateInstruction(NewMI, false, MaxStageCount, 0, VRMap);
    KernelBB->push_back(NewMI);
    InstrMap[NewMI] = &*I;
  }

  KernelBB->transferSuccessors(BB);
  KernelBB->replaceSuccessor(BB, KernelBB);

  generateExistingPhis(KernelBB, PrologBBs.back(), KernelBB, KernelBB, VRMap,
                       InstrMap, MaxStageCount, MaxStageCount, false);
  generatePhis(KernelBB, PrologBBs.back(), KernelBB, KernelBB, VRMap, InstrMap,
               MaxStageCount, MaxStageCount, false);

  LLVM_DEBUG(dbgs() << "New block\n"; KernelBB->dump(););

  SmallVector<MachineBasicBlock *, 4> EpilogBBs;
  // Generate the epilog instructions to complete the pipeline.
  generateEpilog(MaxStageCount, KernelBB, VRMap, EpilogBBs, PrologBBs);

  // We need this step because the register allocation doesn't handle some
  // situations well, so we insert copies to help out.
  splitLifetimes(KernelBB, EpilogBBs);

  // Remove dead instructions due to loop induction variables.
  removeDeadInstructions(KernelBB, EpilogBBs);

  // Add branches between prolog and epilog blocks.
  addBranches(*Preheader, PrologBBs, KernelBB, EpilogBBs, VRMap);

  // Remove the original loop since it's no longer referenced.
  for (auto &I : *BB)
    LIS.RemoveMachineInstrFromMaps(I);
  BB->clear();
  BB->eraseFromParent();

  delete[] VRMap;
}

/// Generate the pipeline prolog code.
void ModuloScheduleExpander::generateProlog(unsigned LastStage,
                                            MachineBasicBlock *KernelBB,
                                            ValueMapTy *VRMap,
                                            MBBVectorTy &PrologBBs) {
  MachineBasicBlock *PredBB = Preheader;
  InstrMapTy InstrMap;

  // Generate a basic block for each stage, not including the last stage,
  // which will be generated in the kernel. Each basic block may contain
  // instructions from multiple stages/iterations.
  for (unsigned i = 0; i < LastStage; ++i) {
    // Create and insert the prolog basic block prior to the original loop
    // basic block.  The original loop is removed later.
    MachineBasicBlock *NewBB = MF.CreateMachineBasicBlock(BB->getBasicBlock());
    PrologBBs.push_back(NewBB);
    MF.insert(BB->getIterator(), NewBB);
    NewBB->transferSuccessors(PredBB);
    PredBB->addSuccessor(NewBB);
    PredBB = NewBB;

    // Generate instructions for each appropriate stage. Process instructions
    // in original program order.
    for (int StageNum = i; StageNum >= 0; --StageNum) {
      for (MachineBasicBlock::iterator BBI = BB->instr_begin(),
                                       BBE = BB->getFirstTerminator();
           BBI != BBE; ++BBI) {
        if (Schedule.getStage(&*BBI) == StageNum) {
          if (BBI->isPHI())
            continue;
          MachineInstr *NewMI =
              cloneAndChangeInstr(&*BBI, i, (unsigned)StageNum);
          updateInstruction(NewMI, false, i, (unsigned)StageNum, VRMap);
          NewBB->push_back(NewMI);
          InstrMap[NewMI] = &*BBI;
        }
      }
    }
    rewritePhiValues(NewBB, i, VRMap, InstrMap);
    LLVM_DEBUG({
      dbgs() << "prolog:\n";
      NewBB->dump();
    });
  }

  PredBB->replaceSuccessor(BB, KernelBB);

  // Check if we need to remove the branch from the preheader to the original
  // loop, and replace it with a branch to the new loop.
  unsigned numBranches = TII->removeBranch(*Preheader);
  if (numBranches) {
    SmallVector<MachineOperand, 0> Cond;
    TII->insertBranch(*Preheader, PrologBBs[0], nullptr, Cond, DebugLoc());
  }
}

/// Generate the pipeline epilog code. The epilog code finishes the iterations
/// that were started in either the prolog or the kernel.  We create a basic
/// block for each stage that needs to complete.
void ModuloScheduleExpander::generateEpilog(unsigned LastStage,
                                            MachineBasicBlock *KernelBB,
                                            ValueMapTy *VRMap,
                                            MBBVectorTy &EpilogBBs,
                                            MBBVectorTy &PrologBBs) {
  // We need to change the branch from the kernel to the first epilog block, so
  // this call to analyze branch uses the kernel rather than the original BB.
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  bool checkBranch = TII->analyzeBranch(*KernelBB, TBB, FBB, Cond);
  assert(!checkBranch && "generateEpilog must be able to analyze the branch");
  if (checkBranch)
    return;

  MachineBasicBlock::succ_iterator LoopExitI = KernelBB->succ_begin();
  if (*LoopExitI == KernelBB)
    ++LoopExitI;
  assert(LoopExitI != KernelBB->succ_end() && "Expecting a successor");
  MachineBasicBlock *LoopExitBB = *LoopExitI;

  MachineBasicBlock *PredBB = KernelBB;
  MachineBasicBlock *EpilogStart = LoopExitBB;
  InstrMapTy InstrMap;

  // Generate a basic block for each stage, not including the last stage,
  // which was generated for the kernel.  Each basic block may contain
  // instructions from multiple stages/iterations.
  int EpilogStage = LastStage + 1;
  for (unsigned i = LastStage; i >= 1; --i, ++EpilogStage) {
    MachineBasicBlock *NewBB = MF.CreateMachineBasicBlock();
    EpilogBBs.push_back(NewBB);
    MF.insert(BB->getIterator(), NewBB);

    PredBB->replaceSuccessor(LoopExitBB, NewBB);
    NewBB->addSuccessor(LoopExitBB);

    if (EpilogStart == LoopExitBB)
      EpilogStart = NewBB;

    // Add instructions to the epilog depending on the current block.
    // Process instructions in original program order.
    for (unsigned StageNum = i; StageNum <= LastStage; ++StageNum) {
      for (auto &BBI : *BB) {
        if (BBI.isPHI())
          continue;
        MachineInstr *In = &BBI;
        if ((unsigned)Schedule.getStage(In) == StageNum) {
          // Instructions with memoperands in the epilog are updated with
          // conservative values.
          MachineInstr *NewMI = cloneInstr(In, UINT_MAX, 0);
          updateInstruction(NewMI, i == 1, EpilogStage, 0, VRMap);
          NewBB->push_back(NewMI);
          InstrMap[NewMI] = In;
        }
      }
    }
    generateExistingPhis(NewBB, PrologBBs[i - 1], PredBB, KernelBB, VRMap,
                         InstrMap, LastStage, EpilogStage, i == 1);
    generatePhis(NewBB, PrologBBs[i - 1], PredBB, KernelBB, VRMap, InstrMap,
                 LastStage, EpilogStage, i == 1);
    PredBB = NewBB;

    LLVM_DEBUG({
      dbgs() << "epilog:\n";
      NewBB->dump();
    });
  }

  // Fix any Phi nodes in the loop exit block.
  LoopExitBB->replacePhiUsesWith(BB, PredBB);

  // Create a branch to the new epilog from the kernel.
  // Remove the original branch and add a new branch to the epilog.
  TII->removeBranch(*KernelBB);
  TII->insertBranch(*KernelBB, KernelBB, EpilogStart, Cond, DebugLoc());
  // Add a branch to the loop exit.
  if (EpilogBBs.size() > 0) {
    MachineBasicBlock *LastEpilogBB = EpilogBBs.back();
    SmallVector<MachineOperand, 4> Cond1;
    TII->insertBranch(*LastEpilogBB, LoopExitBB, nullptr, Cond1, DebugLoc());
  }
}

/// Replace all uses of FromReg that appear outside the specified
/// basic block with ToReg.
static void replaceRegUsesAfterLoop(unsigned FromReg, unsigned ToReg,
                                    MachineBasicBlock *MBB,
                                    MachineRegisterInfo &MRI,
                                    LiveIntervals &LIS) {
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(FromReg),
                                         E = MRI.use_end();
       I != E;) {
    MachineOperand &O = *I;
    ++I;
    if (O.getParent()->getParent() != MBB)
      O.setReg(ToReg);
  }
  if (!LIS.hasInterval(ToReg))
    LIS.createEmptyInterval(ToReg);
}

/// Return true if the register has a use that occurs outside the
/// specified loop.
static bool hasUseAfterLoop(unsigned Reg, MachineBasicBlock *BB,
                            MachineRegisterInfo &MRI) {
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(Reg),
                                         E = MRI.use_end();
       I != E; ++I)
    if (I->getParent()->getParent() != BB)
      return true;
  return false;
}

/// Generate Phis for the specific block in the generated pipelined code.
/// This function looks at the Phis from the original code to guide the
/// creation of new Phis.
void ModuloScheduleExpander::generateExistingPhis(
    MachineBasicBlock *NewBB, MachineBasicBlock *BB1, MachineBasicBlock *BB2,
    MachineBasicBlock *KernelBB, ValueMapTy *VRMap, InstrMapTy &InstrMap,
    unsigned LastStageNum, unsigned CurStageNum, bool IsLast) {
  // Compute the stage number for the initial value of the Phi, which
  // comes from the prolog. The prolog to use depends on to which kernel/
  // epilog that we're adding the Phi.
  unsigned PrologStage = 0;
  unsigned PrevStage = 0;
  bool InKernel = (LastStageNum == CurStageNum);
  if (InKernel) {
    PrologStage = LastStageNum - 1;
    PrevStage = CurStageNum;
  } else {
    PrologStage = LastStageNum - (CurStageNum - LastStageNum);
    PrevStage = LastStageNum + (CurStageNum - LastStageNum) - 1;
  }

  for (MachineBasicBlock::iterator BBI = BB->instr_begin(),
                                   BBE = BB->getFirstNonPHI();
       BBI != BBE; ++BBI) {
    Register Def = BBI->getOperand(0).getReg();

    unsigned InitVal = 0;
    unsigned LoopVal = 0;
    getPhiRegs(*BBI, BB, InitVal, LoopVal);

    unsigned PhiOp1 = 0;
    // The Phi value from the loop body typically is defined in the loop, but
    // not always. So, we need to check if the value is defined in the loop.
    unsigned PhiOp2 = LoopVal;
    if (VRMap[LastStageNum].count(LoopVal))
      PhiOp2 = VRMap[LastStageNum][LoopVal];

    int StageScheduled = Schedule.getStage(&*BBI);
    int LoopValStage = Schedule.getStage(MRI.getVRegDef(LoopVal));
    unsigned NumStages = getStagesForReg(Def, CurStageNum);
    if (NumStages == 0) {
      // We don't need to generate a Phi anymore, but we need to rename any uses
      // of the Phi value.
      unsigned NewReg = VRMap[PrevStage][LoopVal];
      rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, 0, &*BBI, Def,
                            InitVal, NewReg);
      if (VRMap[CurStageNum].count(LoopVal))
        VRMap[CurStageNum][Def] = VRMap[CurStageNum][LoopVal];
    }
    // Adjust the number of Phis needed depending on the number of prologs left,
    // and the distance from where the Phi is first scheduled. The number of
    // Phis cannot exceed the number of prolog stages. Each stage can
    // potentially define two values.
    unsigned MaxPhis = PrologStage + 2;
    if (!InKernel && (int)PrologStage <= LoopValStage)
      MaxPhis = std::max((int)MaxPhis - (int)LoopValStage, 1);
    unsigned NumPhis = std::min(NumStages, MaxPhis);

    unsigned NewReg = 0;
    unsigned AccessStage = (LoopValStage != -1) ? LoopValStage : StageScheduled;
    // In the epilog, we may need to look back one stage to get the correct
    // Phi name because the epilog and prolog blocks execute the same stage.
    // The correct name is from the previous block only when the Phi has
    // been completely scheduled prior to the epilog, and Phi value is not
    // needed in multiple stages.
    int StageDiff = 0;
    if (!InKernel && StageScheduled >= LoopValStage && AccessStage == 0 &&
        NumPhis == 1)
      StageDiff = 1;
    // Adjust the computations below when the phi and the loop definition
    // are scheduled in different stages.
    if (InKernel && LoopValStage != -1 && StageScheduled > LoopValStage)
      StageDiff = StageScheduled - LoopValStage;
    for (unsigned np = 0; np < NumPhis; ++np) {
      // If the Phi hasn't been scheduled, then use the initial Phi operand
      // value. Otherwise, use the scheduled version of the instruction. This
      // is a little complicated when a Phi references another Phi.
      if (np > PrologStage || StageScheduled >= (int)LastStageNum)
        PhiOp1 = InitVal;
      // Check if the Phi has already been scheduled in a prolog stage.
      else if (PrologStage >= AccessStage + StageDiff + np &&
               VRMap[PrologStage - StageDiff - np].count(LoopVal) != 0)
        PhiOp1 = VRMap[PrologStage - StageDiff - np][LoopVal];
      // Check if the Phi has already been scheduled, but the loop instruction
      // is either another Phi, or doesn't occur in the loop.
      else if (PrologStage >= AccessStage + StageDiff + np) {
        // If the Phi references another Phi, we need to examine the other
        // Phi to get the correct value.
        PhiOp1 = LoopVal;
        MachineInstr *InstOp1 = MRI.getVRegDef(PhiOp1);
        int Indirects = 1;
        while (InstOp1 && InstOp1->isPHI() && InstOp1->getParent() == BB) {
          int PhiStage = Schedule.getStage(InstOp1);
          if ((int)(PrologStage - StageDiff - np) < PhiStage + Indirects)
            PhiOp1 = getInitPhiReg(*InstOp1, BB);
          else
            PhiOp1 = getLoopPhiReg(*InstOp1, BB);
          InstOp1 = MRI.getVRegDef(PhiOp1);
          int PhiOpStage = Schedule.getStage(InstOp1);
          int StageAdj = (PhiOpStage != -1 ? PhiStage - PhiOpStage : 0);
          if (PhiOpStage != -1 && PrologStage - StageAdj >= Indirects + np &&
              VRMap[PrologStage - StageAdj - Indirects - np].count(PhiOp1)) {
            PhiOp1 = VRMap[PrologStage - StageAdj - Indirects - np][PhiOp1];
            break;
          }
          ++Indirects;
        }
      } else
        PhiOp1 = InitVal;
      // If this references a generated Phi in the kernel, get the Phi operand
      // from the incoming block.
      if (MachineInstr *InstOp1 = MRI.getVRegDef(PhiOp1))
        if (InstOp1->isPHI() && InstOp1->getParent() == KernelBB)
          PhiOp1 = getInitPhiReg(*InstOp1, KernelBB);

      MachineInstr *PhiInst = MRI.getVRegDef(LoopVal);
      bool LoopDefIsPhi = PhiInst && PhiInst->isPHI();
      // In the epilog, a map lookup is needed to get the value from the kernel,
      // or previous epilog block. How is does this depends on if the
      // instruction is scheduled in the previous block.
      if (!InKernel) {
        int StageDiffAdj = 0;
        if (LoopValStage != -1 && StageScheduled > LoopValStage)
          StageDiffAdj = StageScheduled - LoopValStage;
        // Use the loop value defined in the kernel, unless the kernel
        // contains the last definition of the Phi.
        if (np == 0 && PrevStage == LastStageNum &&
            (StageScheduled != 0 || LoopValStage != 0) &&
            VRMap[PrevStage - StageDiffAdj].count(LoopVal))
          PhiOp2 = VRMap[PrevStage - StageDiffAdj][LoopVal];
        // Use the value defined by the Phi. We add one because we switch
        // from looking at the loop value to the Phi definition.
        else if (np > 0 && PrevStage == LastStageNum &&
                 VRMap[PrevStage - np + 1].count(Def))
          PhiOp2 = VRMap[PrevStage - np + 1][Def];
        // Use the loop value defined in the kernel.
        else if (static_cast<unsigned>(LoopValStage) > PrologStage + 1 &&
                 VRMap[PrevStage - StageDiffAdj - np].count(LoopVal))
          PhiOp2 = VRMap[PrevStage - StageDiffAdj - np][LoopVal];
        // Use the value defined by the Phi, unless we're generating the first
        // epilog and the Phi refers to a Phi in a different stage.
        else if (VRMap[PrevStage - np].count(Def) &&
                 (!LoopDefIsPhi || (PrevStage != LastStageNum) ||
                  (LoopValStage == StageScheduled)))
          PhiOp2 = VRMap[PrevStage - np][Def];
      }

      // Check if we can reuse an existing Phi. This occurs when a Phi
      // references another Phi, and the other Phi is scheduled in an
      // earlier stage. We can try to reuse an existing Phi up until the last
      // stage of the current Phi.
      if (LoopDefIsPhi) {
        if (static_cast<int>(PrologStage - np) >= StageScheduled) {
          int LVNumStages = getStagesForPhi(LoopVal);
          int StageDiff = (StageScheduled - LoopValStage);
          LVNumStages -= StageDiff;
          // Make sure the loop value Phi has been processed already.
          if (LVNumStages > (int)np && VRMap[CurStageNum].count(LoopVal)) {
            NewReg = PhiOp2;
            unsigned ReuseStage = CurStageNum;
            if (isLoopCarried(*PhiInst))
              ReuseStage -= LVNumStages;
            // Check if the Phi to reuse has been generated yet. If not, then
            // there is nothing to reuse.
            if (VRMap[ReuseStage - np].count(LoopVal)) {
              NewReg = VRMap[ReuseStage - np][LoopVal];

              rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, np, &*BBI,
                                    Def, NewReg);
              // Update the map with the new Phi name.
              VRMap[CurStageNum - np][Def] = NewReg;
              PhiOp2 = NewReg;
              if (VRMap[LastStageNum - np - 1].count(LoopVal))
                PhiOp2 = VRMap[LastStageNum - np - 1][LoopVal];

              if (IsLast && np == NumPhis - 1)
                replaceRegUsesAfterLoop(Def, NewReg, BB, MRI, LIS);
              continue;
            }
          }
        }
        if (InKernel && StageDiff > 0 &&
            VRMap[CurStageNum - StageDiff - np].count(LoopVal))
          PhiOp2 = VRMap[CurStageNum - StageDiff - np][LoopVal];
      }

      const TargetRegisterClass *RC = MRI.getRegClass(Def);
      NewReg = MRI.createVirtualRegister(RC);

      MachineInstrBuilder NewPhi =
          BuildMI(*NewBB, NewBB->getFirstNonPHI(), DebugLoc(),
                  TII->get(TargetOpcode::PHI), NewReg);
      NewPhi.addReg(PhiOp1).addMBB(BB1);
      NewPhi.addReg(PhiOp2).addMBB(BB2);
      if (np == 0)
        InstrMap[NewPhi] = &*BBI;

      // We define the Phis after creating the new pipelined code, so
      // we need to rename the Phi values in scheduled instructions.

      unsigned PrevReg = 0;
      if (InKernel && VRMap[PrevStage - np].count(LoopVal))
        PrevReg = VRMap[PrevStage - np][LoopVal];
      rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, np, &*BBI, Def,
                            NewReg, PrevReg);
      // If the Phi has been scheduled, use the new name for rewriting.
      if (VRMap[CurStageNum - np].count(Def)) {
        unsigned R = VRMap[CurStageNum - np][Def];
        rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, np, &*BBI, R,
                              NewReg);
      }

      // Check if we need to rename any uses that occurs after the loop. The
      // register to replace depends on whether the Phi is scheduled in the
      // epilog.
      if (IsLast && np == NumPhis - 1)
        replaceRegUsesAfterLoop(Def, NewReg, BB, MRI, LIS);

      // In the kernel, a dependent Phi uses the value from this Phi.
      if (InKernel)
        PhiOp2 = NewReg;

      // Update the map with the new Phi name.
      VRMap[CurStageNum - np][Def] = NewReg;
    }

    while (NumPhis++ < NumStages) {
      rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, NumPhis, &*BBI, Def,
                            NewReg, 0);
    }

    // Check if we need to rename a Phi that has been eliminated due to
    // scheduling.
    if (NumStages == 0 && IsLast && VRMap[CurStageNum].count(LoopVal))
      replaceRegUsesAfterLoop(Def, VRMap[CurStageNum][LoopVal], BB, MRI, LIS);
  }
}

/// Generate Phis for the specified block in the generated pipelined code.
/// These are new Phis needed because the definition is scheduled after the
/// use in the pipelined sequence.
void ModuloScheduleExpander::generatePhis(
    MachineBasicBlock *NewBB, MachineBasicBlock *BB1, MachineBasicBlock *BB2,
    MachineBasicBlock *KernelBB, ValueMapTy *VRMap, InstrMapTy &InstrMap,
    unsigned LastStageNum, unsigned CurStageNum, bool IsLast) {
  // Compute the stage number that contains the initial Phi value, and
  // the Phi from the previous stage.
  unsigned PrologStage = 0;
  unsigned PrevStage = 0;
  unsigned StageDiff = CurStageNum - LastStageNum;
  bool InKernel = (StageDiff == 0);
  if (InKernel) {
    PrologStage = LastStageNum - 1;
    PrevStage = CurStageNum;
  } else {
    PrologStage = LastStageNum - StageDiff;
    PrevStage = LastStageNum + StageDiff - 1;
  }

  for (MachineBasicBlock::iterator BBI = BB->getFirstNonPHI(),
                                   BBE = BB->instr_end();
       BBI != BBE; ++BBI) {
    for (unsigned i = 0, e = BBI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = BBI->getOperand(i);
      if (!MO.isReg() || !MO.isDef() ||
          !Register::isVirtualRegister(MO.getReg()))
        continue;

      int StageScheduled = Schedule.getStage(&*BBI);
      assert(StageScheduled != -1 && "Expecting scheduled instruction.");
      Register Def = MO.getReg();
      unsigned NumPhis = getStagesForReg(Def, CurStageNum);
      // An instruction scheduled in stage 0 and is used after the loop
      // requires a phi in the epilog for the last definition from either
      // the kernel or prolog.
      if (!InKernel && NumPhis == 0 && StageScheduled == 0 &&
          hasUseAfterLoop(Def, BB, MRI))
        NumPhis = 1;
      if (!InKernel && (unsigned)StageScheduled > PrologStage)
        continue;

      unsigned PhiOp2 = VRMap[PrevStage][Def];
      if (MachineInstr *InstOp2 = MRI.getVRegDef(PhiOp2))
        if (InstOp2->isPHI() && InstOp2->getParent() == NewBB)
          PhiOp2 = getLoopPhiReg(*InstOp2, BB2);
      // The number of Phis can't exceed the number of prolog stages. The
      // prolog stage number is zero based.
      if (NumPhis > PrologStage + 1 - StageScheduled)
        NumPhis = PrologStage + 1 - StageScheduled;
      for (unsigned np = 0; np < NumPhis; ++np) {
        unsigned PhiOp1 = VRMap[PrologStage][Def];
        if (np <= PrologStage)
          PhiOp1 = VRMap[PrologStage - np][Def];
        if (MachineInstr *InstOp1 = MRI.getVRegDef(PhiOp1)) {
          if (InstOp1->isPHI() && InstOp1->getParent() == KernelBB)
            PhiOp1 = getInitPhiReg(*InstOp1, KernelBB);
          if (InstOp1->isPHI() && InstOp1->getParent() == NewBB)
            PhiOp1 = getInitPhiReg(*InstOp1, NewBB);
        }
        if (!InKernel)
          PhiOp2 = VRMap[PrevStage - np][Def];

        const TargetRegisterClass *RC = MRI.getRegClass(Def);
        Register NewReg = MRI.createVirtualRegister(RC);

        MachineInstrBuilder NewPhi =
            BuildMI(*NewBB, NewBB->getFirstNonPHI(), DebugLoc(),
                    TII->get(TargetOpcode::PHI), NewReg);
        NewPhi.addReg(PhiOp1).addMBB(BB1);
        NewPhi.addReg(PhiOp2).addMBB(BB2);
        if (np == 0)
          InstrMap[NewPhi] = &*BBI;

        // Rewrite uses and update the map. The actions depend upon whether
        // we generating code for the kernel or epilog blocks.
        if (InKernel) {
          rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, np, &*BBI, PhiOp1,
                                NewReg);
          rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, np, &*BBI, PhiOp2,
                                NewReg);

          PhiOp2 = NewReg;
          VRMap[PrevStage - np - 1][Def] = NewReg;
        } else {
          VRMap[CurStageNum - np][Def] = NewReg;
          if (np == NumPhis - 1)
            rewriteScheduledInstr(NewBB, InstrMap, CurStageNum, np, &*BBI, Def,
                                  NewReg);
        }
        if (IsLast && np == NumPhis - 1)
          replaceRegUsesAfterLoop(Def, NewReg, BB, MRI, LIS);
      }
    }
  }
}

/// Remove instructions that generate values with no uses.
/// Typically, these are induction variable operations that generate values
/// used in the loop itself.  A dead instruction has a definition with
/// no uses, or uses that occur in the original loop only.
void ModuloScheduleExpander::removeDeadInstructions(MachineBasicBlock *KernelBB,
                                                    MBBVectorTy &EpilogBBs) {
  // For each epilog block, check that the value defined by each instruction
  // is used.  If not, delete it.
  for (MBBVectorTy::reverse_iterator MBB = EpilogBBs.rbegin(),
                                     MBE = EpilogBBs.rend();
       MBB != MBE; ++MBB)
    for (MachineBasicBlock::reverse_instr_iterator MI = (*MBB)->instr_rbegin(),
                                                   ME = (*MBB)->instr_rend();
         MI != ME;) {
      // From DeadMachineInstructionElem. Don't delete inline assembly.
      if (MI->isInlineAsm()) {
        ++MI;
        continue;
      }
      bool SawStore = false;
      // Check if it's safe to remove the instruction due to side effects.
      // We can, and want to, remove Phis here.
      if (!MI->isSafeToMove(nullptr, SawStore) && !MI->isPHI()) {
        ++MI;
        continue;
      }
      bool used = true;
      for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
                                      MOE = MI->operands_end();
           MOI != MOE; ++MOI) {
        if (!MOI->isReg() || !MOI->isDef())
          continue;
        Register reg = MOI->getReg();
        // Assume physical registers are used, unless they are marked dead.
        if (Register::isPhysicalRegister(reg)) {
          used = !MOI->isDead();
          if (used)
            break;
          continue;
        }
        unsigned realUses = 0;
        for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(reg),
                                               EI = MRI.use_end();
             UI != EI; ++UI) {
          // Check if there are any uses that occur only in the original
          // loop.  If so, that's not a real use.
          if (UI->getParent()->getParent() != BB) {
            realUses++;
            used = true;
            break;
          }
        }
        if (realUses > 0)
          break;
        used = false;
      }
      if (!used) {
        LIS.RemoveMachineInstrFromMaps(*MI);
        MI++->eraseFromParent();
        continue;
      }
      ++MI;
    }
  // In the kernel block, check if we can remove a Phi that generates a value
  // used in an instruction removed in the epilog block.
  for (MachineBasicBlock::iterator BBI = KernelBB->instr_begin(),
                                   BBE = KernelBB->getFirstNonPHI();
       BBI != BBE;) {
    MachineInstr *MI = &*BBI;
    ++BBI;
    Register reg = MI->getOperand(0).getReg();
    if (MRI.use_begin(reg) == MRI.use_end()) {
      LIS.RemoveMachineInstrFromMaps(*MI);
      MI->eraseFromParent();
    }
  }
}

/// For loop carried definitions, we split the lifetime of a virtual register
/// that has uses past the definition in the next iteration. A copy with a new
/// virtual register is inserted before the definition, which helps with
/// generating a better register assignment.
///
///   v1 = phi(a, v2)     v1 = phi(a, v2)
///   v2 = phi(b, v3)     v2 = phi(b, v3)
///   v3 = ..             v4 = copy v1
///   .. = V1             v3 = ..
///                       .. = v4
void ModuloScheduleExpander::splitLifetimes(MachineBasicBlock *KernelBB,
                                            MBBVectorTy &EpilogBBs) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (auto &PHI : KernelBB->phis()) {
    Register Def = PHI.getOperand(0).getReg();
    // Check for any Phi definition that used as an operand of another Phi
    // in the same block.
    for (MachineRegisterInfo::use_instr_iterator I = MRI.use_instr_begin(Def),
                                                 E = MRI.use_instr_end();
         I != E; ++I) {
      if (I->isPHI() && I->getParent() == KernelBB) {
        // Get the loop carried definition.
        unsigned LCDef = getLoopPhiReg(PHI, KernelBB);
        if (!LCDef)
          continue;
        MachineInstr *MI = MRI.getVRegDef(LCDef);
        if (!MI || MI->getParent() != KernelBB || MI->isPHI())
          continue;
        // Search through the rest of the block looking for uses of the Phi
        // definition. If one occurs, then split the lifetime.
        unsigned SplitReg = 0;
        for (auto &BBJ : make_range(MachineBasicBlock::instr_iterator(MI),
                                    KernelBB->instr_end()))
          if (BBJ.readsRegister(Def)) {
            // We split the lifetime when we find the first use.
            if (SplitReg == 0) {
              SplitReg = MRI.createVirtualRegister(MRI.getRegClass(Def));
              BuildMI(*KernelBB, MI, MI->getDebugLoc(),
                      TII->get(TargetOpcode::COPY), SplitReg)
                  .addReg(Def);
            }
            BBJ.substituteRegister(Def, SplitReg, 0, *TRI);
          }
        if (!SplitReg)
          continue;
        // Search through each of the epilog blocks for any uses to be renamed.
        for (auto &Epilog : EpilogBBs)
          for (auto &I : *Epilog)
            if (I.readsRegister(Def))
              I.substituteRegister(Def, SplitReg, 0, *TRI);
        break;
      }
    }
  }
}

/// Remove the incoming block from the Phis in a basic block.
static void removePhis(MachineBasicBlock *BB, MachineBasicBlock *Incoming) {
  for (MachineInstr &MI : *BB) {
    if (!MI.isPHI())
      break;
    for (unsigned i = 1, e = MI.getNumOperands(); i != e; i += 2)
      if (MI.getOperand(i + 1).getMBB() == Incoming) {
        MI.RemoveOperand(i + 1);
        MI.RemoveOperand(i);
        break;
      }
  }
}

/// Create branches from each prolog basic block to the appropriate epilog
/// block.  These edges are needed if the loop ends before reaching the
/// kernel.
void ModuloScheduleExpander::addBranches(MachineBasicBlock &PreheaderBB,
                                         MBBVectorTy &PrologBBs,
                                         MachineBasicBlock *KernelBB,
                                         MBBVectorTy &EpilogBBs,
                                         ValueMapTy *VRMap) {
  assert(PrologBBs.size() == EpilogBBs.size() && "Prolog/Epilog mismatch");
  MachineInstr *IndVar;
  MachineInstr *Cmp;
  if (TII->analyzeLoop(*Schedule.getLoop(), IndVar, Cmp))
    llvm_unreachable("Must be able to analyze loop!");
  MachineBasicBlock *LastPro = KernelBB;
  MachineBasicBlock *LastEpi = KernelBB;

  // Start from the blocks connected to the kernel and work "out"
  // to the first prolog and the last epilog blocks.
  SmallVector<MachineInstr *, 4> PrevInsts;
  unsigned MaxIter = PrologBBs.size() - 1;
  unsigned LC = UINT_MAX;
  unsigned LCMin = UINT_MAX;
  for (unsigned i = 0, j = MaxIter; i <= MaxIter; ++i, --j) {
    // Add branches to the prolog that go to the corresponding
    // epilog, and the fall-thru prolog/kernel block.
    MachineBasicBlock *Prolog = PrologBBs[j];
    MachineBasicBlock *Epilog = EpilogBBs[i];
    // We've executed one iteration, so decrement the loop count and check for
    // the loop end.
    SmallVector<MachineOperand, 4> Cond;
    // Check if the LOOP0 has already been removed. If so, then there is no need
    // to reduce the trip count.
    if (LC != 0)
      LC = TII->reduceLoopCount(*Prolog, PreheaderBB, IndVar, *Cmp, Cond,
                                PrevInsts, j, MaxIter);

    // Record the value of the first trip count, which is used to determine if
    // branches and blocks can be removed for constant trip counts.
    if (LCMin == UINT_MAX)
      LCMin = LC;

    unsigned numAdded = 0;
    if (Register::isVirtualRegister(LC)) {
      Prolog->addSuccessor(Epilog);
      numAdded = TII->insertBranch(*Prolog, Epilog, LastPro, Cond, DebugLoc());
    } else if (j >= LCMin) {
      Prolog->addSuccessor(Epilog);
      Prolog->removeSuccessor(LastPro);
      LastEpi->removeSuccessor(Epilog);
      numAdded = TII->insertBranch(*Prolog, Epilog, nullptr, Cond, DebugLoc());
      removePhis(Epilog, LastEpi);
      // Remove the blocks that are no longer referenced.
      if (LastPro != LastEpi) {
        LastEpi->clear();
        LastEpi->eraseFromParent();
      }
      LastPro->clear();
      LastPro->eraseFromParent();
    } else {
      numAdded = TII->insertBranch(*Prolog, LastPro, nullptr, Cond, DebugLoc());
      removePhis(Epilog, Prolog);
    }
    LastPro = Prolog;
    LastEpi = Epilog;
    for (MachineBasicBlock::reverse_instr_iterator I = Prolog->instr_rbegin(),
                                                   E = Prolog->instr_rend();
         I != E && numAdded > 0; ++I, --numAdded)
      updateInstruction(&*I, false, j, 0, VRMap);
  }
}

/// Return true if we can compute the amount the instruction changes
/// during each iteration. Set Delta to the amount of the change.
bool ModuloScheduleExpander::computeDelta(MachineInstr &MI, unsigned &Delta) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const MachineOperand *BaseOp;
  int64_t Offset;
  if (!TII->getMemOperandWithOffset(MI, BaseOp, Offset, TRI))
    return false;

  if (!BaseOp->isReg())
    return false;

  Register BaseReg = BaseOp->getReg();

  MachineRegisterInfo &MRI = MF.getRegInfo();
  // Check if there is a Phi. If so, get the definition in the loop.
  MachineInstr *BaseDef = MRI.getVRegDef(BaseReg);
  if (BaseDef && BaseDef->isPHI()) {
    BaseReg = getLoopPhiReg(*BaseDef, MI.getParent());
    BaseDef = MRI.getVRegDef(BaseReg);
  }
  if (!BaseDef)
    return false;

  int D = 0;
  if (!TII->getIncrementValue(*BaseDef, D) && D >= 0)
    return false;

  Delta = D;
  return true;
}

/// Update the memory operand with a new offset when the pipeliner
/// generates a new copy of the instruction that refers to a
/// different memory location.
void ModuloScheduleExpander::updateMemOperands(MachineInstr &NewMI,
                                               MachineInstr &OldMI,
                                               unsigned Num) {
  if (Num == 0)
    return;
  // If the instruction has memory operands, then adjust the offset
  // when the instruction appears in different stages.
  if (NewMI.memoperands_empty())
    return;
  SmallVector<MachineMemOperand *, 2> NewMMOs;
  for (MachineMemOperand *MMO : NewMI.memoperands()) {
    // TODO: Figure out whether isAtomic is really necessary (see D57601).
    if (MMO->isVolatile() || MMO->isAtomic() ||
        (MMO->isInvariant() && MMO->isDereferenceable()) ||
        (!MMO->getValue())) {
      NewMMOs.push_back(MMO);
      continue;
    }
    unsigned Delta;
    if (Num != UINT_MAX && computeDelta(OldMI, Delta)) {
      int64_t AdjOffset = Delta * Num;
      NewMMOs.push_back(
          MF.getMachineMemOperand(MMO, AdjOffset, MMO->getSize()));
    } else {
      NewMMOs.push_back(
          MF.getMachineMemOperand(MMO, 0, MemoryLocation::UnknownSize));
    }
  }
  NewMI.setMemRefs(MF, NewMMOs);
}

/// Clone the instruction for the new pipelined loop and update the
/// memory operands, if needed.
MachineInstr *ModuloScheduleExpander::cloneInstr(MachineInstr *OldMI,
                                                 unsigned CurStageNum,
                                                 unsigned InstStageNum) {
  MachineInstr *NewMI = MF.CloneMachineInstr(OldMI);
  // Check for tied operands in inline asm instructions. This should be handled
  // elsewhere, but I'm not sure of the best solution.
  if (OldMI->isInlineAsm())
    for (unsigned i = 0, e = OldMI->getNumOperands(); i != e; ++i) {
      const auto &MO = OldMI->getOperand(i);
      if (MO.isReg() && MO.isUse())
        break;
      unsigned UseIdx;
      if (OldMI->isRegTiedToUseOperand(i, &UseIdx))
        NewMI->tieOperands(i, UseIdx);
    }
  updateMemOperands(*NewMI, *OldMI, CurStageNum - InstStageNum);
  return NewMI;
}

/// Clone the instruction for the new pipelined loop. If needed, this
/// function updates the instruction using the values saved in the
/// InstrChanges structure.
MachineInstr *ModuloScheduleExpander::cloneAndChangeInstr(
    MachineInstr *OldMI, unsigned CurStageNum, unsigned InstStageNum) {
  MachineInstr *NewMI = MF.CloneMachineInstr(OldMI);
  auto It = InstrChanges.find(OldMI);
  if (It != InstrChanges.end()) {
    std::pair<unsigned, int64_t> RegAndOffset = It->second;
    unsigned BasePos, OffsetPos;
    if (!TII->getBaseAndOffsetPosition(*OldMI, BasePos, OffsetPos))
      return nullptr;
    int64_t NewOffset = OldMI->getOperand(OffsetPos).getImm();
    MachineInstr *LoopDef = findDefInLoop(RegAndOffset.first);
    if (Schedule.getStage(LoopDef) > (signed)InstStageNum)
      NewOffset += RegAndOffset.second * (CurStageNum - InstStageNum);
    NewMI->getOperand(OffsetPos).setImm(NewOffset);
  }
  updateMemOperands(*NewMI, *OldMI, CurStageNum - InstStageNum);
  return NewMI;
}

/// Update the machine instruction with new virtual registers.  This
/// function may change the defintions and/or uses.
void ModuloScheduleExpander::updateInstruction(MachineInstr *NewMI,
                                               bool LastDef,
                                               unsigned CurStageNum,
                                               unsigned InstrStageNum,
                                               ValueMapTy *VRMap) {
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (!MO.isReg() || !Register::isVirtualRegister(MO.getReg()))
      continue;
    Register reg = MO.getReg();
    if (MO.isDef()) {
      // Create a new virtual register for the definition.
      const TargetRegisterClass *RC = MRI.getRegClass(reg);
      Register NewReg = MRI.createVirtualRegister(RC);
      MO.setReg(NewReg);
      VRMap[CurStageNum][reg] = NewReg;
      if (LastDef)
        replaceRegUsesAfterLoop(reg, NewReg, BB, MRI, LIS);
    } else if (MO.isUse()) {
      MachineInstr *Def = MRI.getVRegDef(reg);
      // Compute the stage that contains the last definition for instruction.
      int DefStageNum = Schedule.getStage(Def);
      unsigned StageNum = CurStageNum;
      if (DefStageNum != -1 && (int)InstrStageNum > DefStageNum) {
        // Compute the difference in stages between the defintion and the use.
        unsigned StageDiff = (InstrStageNum - DefStageNum);
        // Make an adjustment to get the last definition.
        StageNum -= StageDiff;
      }
      if (VRMap[StageNum].count(reg))
        MO.setReg(VRMap[StageNum][reg]);
    }
  }
}

/// Return the instruction in the loop that defines the register.
/// If the definition is a Phi, then follow the Phi operand to
/// the instruction in the loop.
MachineInstr *ModuloScheduleExpander::findDefInLoop(unsigned Reg) {
  SmallPtrSet<MachineInstr *, 8> Visited;
  MachineInstr *Def = MRI.getVRegDef(Reg);
  while (Def->isPHI()) {
    if (!Visited.insert(Def).second)
      break;
    for (unsigned i = 1, e = Def->getNumOperands(); i < e; i += 2)
      if (Def->getOperand(i + 1).getMBB() == BB) {
        Def = MRI.getVRegDef(Def->getOperand(i).getReg());
        break;
      }
  }
  return Def;
}

/// Return the new name for the value from the previous stage.
unsigned ModuloScheduleExpander::getPrevMapVal(
    unsigned StageNum, unsigned PhiStage, unsigned LoopVal, unsigned LoopStage,
    ValueMapTy *VRMap, MachineBasicBlock *BB) {
  unsigned PrevVal = 0;
  if (StageNum > PhiStage) {
    MachineInstr *LoopInst = MRI.getVRegDef(LoopVal);
    if (PhiStage == LoopStage && VRMap[StageNum - 1].count(LoopVal))
      // The name is defined in the previous stage.
      PrevVal = VRMap[StageNum - 1][LoopVal];
    else if (VRMap[StageNum].count(LoopVal))
      // The previous name is defined in the current stage when the instruction
      // order is swapped.
      PrevVal = VRMap[StageNum][LoopVal];
    else if (!LoopInst->isPHI() || LoopInst->getParent() != BB)
      // The loop value hasn't yet been scheduled.
      PrevVal = LoopVal;
    else if (StageNum == PhiStage + 1)
      // The loop value is another phi, which has not been scheduled.
      PrevVal = getInitPhiReg(*LoopInst, BB);
    else if (StageNum > PhiStage + 1 && LoopInst->getParent() == BB)
      // The loop value is another phi, which has been scheduled.
      PrevVal =
          getPrevMapVal(StageNum - 1, PhiStage, getLoopPhiReg(*LoopInst, BB),
                        LoopStage, VRMap, BB);
  }
  return PrevVal;
}

/// Rewrite the Phi values in the specified block to use the mappings
/// from the initial operand. Once the Phi is scheduled, we switch
/// to using the loop value instead of the Phi value, so those names
/// do not need to be rewritten.
void ModuloScheduleExpander::rewritePhiValues(MachineBasicBlock *NewBB,
                                              unsigned StageNum,
                                              ValueMapTy *VRMap,
                                              InstrMapTy &InstrMap) {
  for (auto &PHI : BB->phis()) {
    unsigned InitVal = 0;
    unsigned LoopVal = 0;
    getPhiRegs(PHI, BB, InitVal, LoopVal);
    Register PhiDef = PHI.getOperand(0).getReg();

    unsigned PhiStage = (unsigned)Schedule.getStage(MRI.getVRegDef(PhiDef));
    unsigned LoopStage = (unsigned)Schedule.getStage(MRI.getVRegDef(LoopVal));
    unsigned NumPhis = getStagesForPhi(PhiDef);
    if (NumPhis > StageNum)
      NumPhis = StageNum;
    for (unsigned np = 0; np <= NumPhis; ++np) {
      unsigned NewVal =
          getPrevMapVal(StageNum - np, PhiStage, LoopVal, LoopStage, VRMap, BB);
      if (!NewVal)
        NewVal = InitVal;
      rewriteScheduledInstr(NewBB, InstrMap, StageNum - np, np, &PHI, PhiDef,
                            NewVal);
    }
  }
}

/// Rewrite a previously scheduled instruction to use the register value
/// from the new instruction. Make sure the instruction occurs in the
/// basic block, and we don't change the uses in the new instruction.
void ModuloScheduleExpander::rewriteScheduledInstr(
    MachineBasicBlock *BB, InstrMapTy &InstrMap, unsigned CurStageNum,
    unsigned PhiNum, MachineInstr *Phi, unsigned OldReg, unsigned NewReg,
    unsigned PrevReg) {
  bool InProlog = (CurStageNum < (unsigned)Schedule.getNumStages() - 1);
  int StagePhi = Schedule.getStage(Phi) + PhiNum;
  // Rewrite uses that have been scheduled already to use the new
  // Phi register.
  for (MachineRegisterInfo::use_iterator UI = MRI.use_begin(OldReg),
                                         EI = MRI.use_end();
       UI != EI;) {
    MachineOperand &UseOp = *UI;
    MachineInstr *UseMI = UseOp.getParent();
    ++UI;
    if (UseMI->getParent() != BB)
      continue;
    if (UseMI->isPHI()) {
      if (!Phi->isPHI() && UseMI->getOperand(0).getReg() == NewReg)
        continue;
      if (getLoopPhiReg(*UseMI, BB) != OldReg)
        continue;
    }
    InstrMapTy::iterator OrigInstr = InstrMap.find(UseMI);
    assert(OrigInstr != InstrMap.end() && "Instruction not scheduled.");
    MachineInstr *OrigMI = OrigInstr->second;
    int StageSched = Schedule.getStage(OrigMI);
    int CycleSched = Schedule.getCycle(OrigMI);
    unsigned ReplaceReg = 0;
    // This is the stage for the scheduled instruction.
    if (StagePhi == StageSched && Phi->isPHI()) {
      int CyclePhi = Schedule.getCycle(Phi);
      if (PrevReg && InProlog)
        ReplaceReg = PrevReg;
      else if (PrevReg && !isLoopCarried(*Phi) &&
               (CyclePhi <= CycleSched || OrigMI->isPHI()))
        ReplaceReg = PrevReg;
      else
        ReplaceReg = NewReg;
    }
    // The scheduled instruction occurs before the scheduled Phi, and the
    // Phi is not loop carried.
    if (!InProlog && StagePhi + 1 == StageSched && !isLoopCarried(*Phi))
      ReplaceReg = NewReg;
    if (StagePhi > StageSched && Phi->isPHI())
      ReplaceReg = NewReg;
    if (!InProlog && !Phi->isPHI() && StagePhi < StageSched)
      ReplaceReg = NewReg;
    if (ReplaceReg) {
      MRI.constrainRegClass(ReplaceReg, MRI.getRegClass(OldReg));
      UseOp.setReg(ReplaceReg);
    }
  }
}

bool ModuloScheduleExpander::isLoopCarried(MachineInstr &Phi) {
  if (!Phi.isPHI())
    return false;
  unsigned DefCycle = Schedule.getCycle(&Phi);
  int DefStage = Schedule.getStage(&Phi);

  unsigned InitVal = 0;
  unsigned LoopVal = 0;
  getPhiRegs(Phi, Phi.getParent(), InitVal, LoopVal);
  MachineInstr *Use = MRI.getVRegDef(LoopVal);
  if (!Use || Use->isPHI())
    return true;
  unsigned LoopCycle = Schedule.getCycle(Use);
  int LoopStage = Schedule.getStage(Use);
  return (LoopCycle > DefCycle) || (LoopStage <= DefStage);
}
