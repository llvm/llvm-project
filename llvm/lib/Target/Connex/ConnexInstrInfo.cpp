//===-- ConnexInstrInfo.cpp - Connex Instruction Information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Connex implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "ConnexInstrInfo.h"
#include "Connex.h"
#include "ConnexHazardRecognizer.h" // USE_POSTRA_SCHED
#include "ConnexSubtarget.h"
#include "ConnexTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "connex-lower"

#define GET_INSTRINFO_CTOR_DTOR
#include "ConnexGenInstrInfo.inc"

using namespace llvm;

MachineInstr *getPredMachineInstr(MachineInstr *MI, MachineInstr **succMI) {
  MachineBasicBlock *MBB = MI->getParent();
  DebugLoc DL = MBB->findDebugLoc(MI);

  LLVM_DEBUG(dbgs() << "getPredMachineInstr(): MI.getOpcode() = "
                    << MI->getOpcode() << "\n");

  // switch (MI.getOpcode())

  MachineInstr *predMI = NULL;
  *succMI = NULL;

  for (MachineBasicBlock::iterator I = MBB->begin(), IE = MBB->end(); I != IE;
       ++I) {
    MachineInstr *IMI = (MachineInstr *)(&(*I));
    if (IMI == MI) {
      I++;
      *succMI = (MachineInstr *)(&(*I));
      break;
    }
    predMI = (MachineInstr *)(&(*I));
    LLVM_DEBUG(
        dbgs() << "getPredMachineInstr(): (I in MBB of MI) I->getOpcode()  = "
               << I->getOpcode() << "\n");
  }

  LLVM_DEBUG(dbgs() << "getPredMachineInstr(): MI = " << MI << "(" << MI << ")"
                    << "\n");
  if ((*succMI) != NULL && (*succMI) != nullptr) {
    LLVM_DEBUG(dbgs() << "getPredMachineInstr(): succMI = "
                      // We do not put this one because we can have issues with
                      // NULL/invalid MachineInstr (at least in case of
                      //   llc -regalloc=fast) << **succMI
                      << "[TO BE DONE]"
                      << "(" << *succMI << ")"
                      << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "getPredMachineInstr(): succMI = NULL\n");
  }

  if (predMI != NULL) {
    LLVM_DEBUG(dbgs() << "getPredMachineInstr(): *predMI = " << *predMI << "("
                      << predMI << ")"
                      << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "getPredMachineInstr(): predMI = NULL\n");
  }

  return predMI;
}

// Inspired from BPFInstrInfo.cpp (from Oct 2025)
ConnexInstrInfo::ConnexInstrInfo(const ConnexSubtarget &STI)
    : ConnexGenInstrInfo(STI, Connex::ADJCALLSTACKDOWN, Connex::ADJCALLSTACKUP) {}


// Inspired from lib/Target/Mips/MipsInstrInfo.cpp
MachineMemOperand *
ConnexInstrInfo::GetMemOperand(MachineBasicBlock &MBB, int FI,
                               MachineMemOperand::Flags Flag) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstrInfo::GetMemOperand()\n");

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                                 // Flag, MFI.getObjectSize(FI), Align
                                 Flag, MFI.getObjectSize(FI),
                                 Align(MFI.getObjectAlign(FI)));
}

// From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html:
//  virtual void copyPhysReg(...)
//   Emit instructions to copy a pair of physical registers.
void ConnexInstrInfo::copyPhysReg(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, const DebugLoc &DL,
    Register DestReg, Register SrcReg, bool KillSrc,
    bool RenamableDest, bool RenamableSrc) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstrInfo::copyPhysReg(*I = " << *I
                    << ", DestReg = " << DestReg << ", SrcReg = " << SrcReg
                    << ")\n");

  if (Connex::GPRRegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(Connex::MOV_rr), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  }
  /*
  else if (BPF::GPR32RegClass.contains(DestReg, SrcReg)) {
    BuildMI(MBB, I, DL, get(BPF::MOV_rr_32), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
  }
  */
  else if (Connex::VectorHRegClass.contains(DestReg, SrcReg)) {
    // llvm_unreachable("NOT implemented well!");

    /*
    // TODO
    if (SrgReg == ct) {
      BuildMI(MBB, I, DL, get(Connex::VLOAD_H), DestReg)
          .addImm(ct) //, getKillRegState(KillSrc))
          .addReg(SrcReg);
    }
    */

    BuildMI(MBB, I, DL, get(Connex::ORV_H), DestReg)
        .addReg(SrcReg) //, getKillRegState(KillSrc))
        .addReg(SrcReg);
  } else
      // if (Connex::BoolMaskRegClass.contains(DestReg, SrcReg))
      if (Connex::BoolMaskRegClass.contains(DestReg) ||
          Connex::BoolMaskRegClass.contains(SrcReg)) {
    LLVM_DEBUG(dbgs() << "ConnexInstrInfo::copyPhysReg(): DestReg or SrcReg "
                         "are in BoolMask\n");
    /*
    // Important-TODO: what if register Wh31, also called R(31), is already in
    //    use for some other var?
    BuildMI(MBB, I, DL, get(Connex::VLOAD_H), Connex::Wh31)
        .addImm(0);

    BuildMI(MBB, I, DL, get(Connex::ORV_H), DestReg)
        .addReg(SrcReg) //, getKillRegState(KillSrc))
        .addReg(Connex::Wh31, getKillRegState(KillSrc));
    */
  }
  /*
  // PREFERABLY_NOT_2019_03_21
  else
  if ( (Connex::MSA128WRegClass.contains(DestReg) &&
        Connex::VectorHRegClass.contains(SrcReg)) ||
       //
       (Connex::MSA128WRegClass.contains(SrcReg) &&
        Connex::VectorHRegClass.contains(DestReg)) ) {

    if (Connex::MSA128WRegClass.contains(DestReg)) {
      LLVM_DEBUG(dbgs()
        << "ConnexInstrInfo::copyPhysReg(): DestReg is TYPE_VECTOR_I32 and "
           "SrcReg is TYPE_VECTOR_I16\n");
    }
    else
    if (Connex::MSA128WRegClass.contains(DestReg)) {
      LLVM_DEBUG(dbgs()
        << "ConnexInstrInfo::copyPhysReg(): DestReg is TYPE_VECTOR_I16 and "
           "SrcReg is TYPE_VECTOR_I32\n");
    }

      // BuildMI(MBB, I, DL, get(Connex::INLINEASM));
      //    This makes llc give error:
      //      <<llvm/include/llvm/CodeGen/MachineInstr.h:293:
      //const llvm::MachineOperand& llvm::MachineInstr::getOperand(unsigned int)
      //                                                                 const:
      //       Assertion `i < getNumOperands() && "getOperand() out of range!"'
      //        failed.>>
      // This works surprisingly:
      //                   BuildMI(MBB, I, DL, get(Connex::NOP_BITCONVERT_HW));

    #ifdef COPY_REGISTER_IMPLEMENTED_WITH_ORV_H
      //BuildMI(MBB, I, DL, get(Connex::NOP_BOGUS));
      BuildMI(MBB, I, DL, get(Connex::ORV_H), DestReg)
        .addReg(SrcReg) //, getKillRegState(KillSrc))
        .addReg(SrcReg);
    #endif
  }
  */
  else {
    llvm_unreachable("Impossible reg-to-reg copy");
  }
}

// From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html:
//  virtual void storeRegToStackSlot(...)
//    Store the specified register of the given register class to the specified
//       stack frame index.
//  storeRegToStackSlot() and loadRegFromStackSlot() use
//      the FI argument (frame index, the index within the current frame)
//
// This implements spilling of registers (both scalar, and vector).
void ConnexInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          Register SrcReg, bool IsKill, int FI,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI,
                                          Register VReg,
                                          MachineInstr::MIFlag Flags) const {
  DebugLoc DL;

  LLVM_DEBUG(dbgs() << "Entered ConnexInstrInfo::storeRegToStackSlot(): FI = "
                    << FI << "\n");
  // assert(FI >= 2 && "I assumed wrong that frame index >= 2"); // 2019_08_03

  /* MEGA-TODO: the FI is only 1 variable, and we basically have 2 stack frames:
  - 1 for the scalar CPU
  - normally 1 for the separate address-space LS memory Connex vector processor,
     although Connex does NOT allow calls inside vector kernels,
      BUT the CPU does although a good case is not simple.

     Think of a case where this mildly-viciated solution is NOT good for
         programs (remember we output OPINCAA programs and NO CPU assembly code,
         and Connex does NOT allow calls inside vector kernels).

     Also, understand well why FI >= 2 always holds
        - it seems there is some prologue.
  */
  // unsigned ConnexLSOffsetSpillLoad = (CONNEX_MEM_NUM_ROWS + 1) - FI;
  unsigned ConnexLSOffsetSpillLoad =
      (CONNEX_MEM_NUM_ROWS + CONNEX_MEM_NUM_ROWS_EXTRA_FOR_SPILL + 1) - FI;

  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (RC == &Connex::GPRRegClass) {
    BuildMI(MBB, I, DL, get(Connex::STD))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addImm(0);
  } else if (RC == &Connex::VectorHRegClass) {
    LLVM_DEBUG(dbgs() << "  ConnexInstrInfo::storeRegToStackSlot(): Spilling Wh"
                      << SrcReg << " to ConnexLSOffsetSpillLoad = "
                      << ConnexLSOffsetSpillLoad << " (FI = " << FI << "), "
                      << "I == MBB.end() is " << (I == MBB.end())
                      << ", MBB = " << MBB.getFullName()
                      << ", &MBB.front() = " << &(MBB.front()) << "\n"
                      << "MBB = " << MBB
               //<< ", MBB.front() = " << MBB.front()
    );

    /* Very Important: after experimenting (see
       ~/LLVM/Tests/DawnCC/91_SAD_f16/FEATURE_LENGTH_128/A/STDerr_llc_01)
      if we have INLINEASM at the beginning of the MBB, the MBB.front() is
       the 1st instruction AFTER these INLINEASM - this is why we can end up
       adding more NOPs...

      Important-TODO: we should take into consideration that vector.body has
        INLINEASM with host-side for loop here normally.
    */

    // Note: this method is spilling the destination register of the
    //       instruction *(I-1)
    /*
    // I got a strange error in LLVM when printing in certain cases *I
    //          - see e.g. ~/LLVM/Tests/DawnCC/90_SSD_f16/3/STDerr_llc_01_old03
    LLVM_DEBUG(dbgs() << "  ConnexInstrInfo::storeRegToStackSlot(): *I = "
                      << *I);
    */

    MachineBasicBlock::iterator Iprev; // = I;

    MachineInstr *IMI;
    if (I == MBB.end())
      IMI = NULL;
    else
      IMI = (MachineInstr *)(&(*I));

    LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): IMI = " << IMI << "\n");
    LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): IMI == &MBB.front() = "
                      << (IMI == (&MBB.front())) << "\n");

    if ((I != MBB.end()) && (IMI != NULL) && (IMI != (&MBB.front()))) {
      Iprev = I;
      Iprev--;
      MachineInstr *IprevMI = (MachineInstr *)(&(*Iprev));

      LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): *IprevMI = " << *IprevMI
                        << "\n");
      LLVM_DEBUG(
          dbgs() << "  storeRegToStackSlot(): IprevMI->getNumOperands() = "
                 << IprevMI->getNumOperands() << "\n");
      LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): IprevMI->getOpcode() == "
                           "Connex::INLINEASM = "
                        << (IprevMI->getOpcode() == Connex::INLINEASM) << "\n");
      LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): IprevMI->getOpcode() == "
                           "Connex::VLOAD_H_SYM_IMM = "
                        << (IprevMI->getOpcode() == Connex::VLOAD_H_SYM_IMM)
                        << "\n");
      // The case where I screw up is LS[1013] = ...
      //   because the INLINEASM before it is the MBB.front() and is INLINEASM.

      if (IprevMI != NULL &&
          // NOT necessary: (IprevMI != (&MBB.front())) &&
          // (IMI != (&MBB.front())) &&
          (IprevMI->getNumOperands() >
               0 || // MEGA-TODO: understand why I give this
           IprevMI->getOpcode() == Connex::INLINEASM ||
           IprevMI->getOpcode() == Connex::VLOAD_H_SYM_IMM)) {

        LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): Handling special case: "
                             "if (IprevMI != NULL && ...).\n");

        MachineOperand &I0Opnd = IprevMI->getOperand(0);

        // Avoiding separating VLOAD_H_SYM_IMM from its corresponding INLINEASM
        if (IprevMI->getOpcode() == Connex::VLOAD_H_SYM_IMM) {
          // Treating Symbolic immediate operands
          // MEGA-TODO: check
          // assert(0 && "Bogus");
          assert(IprevMI->getNumOperands() > 0); // Just checking
          assert(IMI->getOpcode() == Connex::INLINEASM &&
                 "The INLINEASM with the immediate operand should be next "
                 "for VLOAD_H_SYM_IMM.");

          LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): Treating "
                               "VLOAD_H_SYM_IMM case.\n");
          I++;
          // Iprev++;
        }
      }
    }

    BuildMI(MBB, I, DL, get(Connex::ST_SPILL_H))
        .addReg(SrcReg, getKillRegState(IsKill))
        /*
        // Gives error I guess because it is a vector instruction, not eBPF one:
        //  void llvm::MachineInstr::addOperand(llvm::MachineFunction&,
        // const llvm::MachineOperand&): Assertion `(isImpReg || Op.isRegMask()
        ||
        // MCID->isVariadic() || OpNo < MCID->getNumOperands() || isMetaDataOp)
        &&
        //   "Trying to add an operand to a machine instr that is already
        done!"'
        //   failed.
        .addFrameIndex(FI)
        // Even if Connex does NOT have a stack, we can use LS mem to easily
        //    simulate it.
        */
        .addImm(ConnexLSOffsetSpillLoad);

    LLVM_DEBUG(
        dbgs() << "  storeRegToStackSlot(): Added ST_SPILL_H instruction.\n");
    LLVM_DEBUG(dbgs() << "  storeRegToStackSlot(): MBB = " << MBB << "\n");
  } else if (RC == &Connex::BoolMaskRegClass) {
    /*
    BuildMI(MBB, I, DL, get(Connex::ST_H))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addImm(CONNEX_MEM_NUM_ROWS - 100);
    // TODO: this is just bogus I guess, no need to spill v8i1 register
    */
  } else {
    llvm_unreachable("Connex back end: Can't store register to stack slot");
  }
}


// From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html:
//   virtual void loadRegFromStackSlot(...)
//    Load the specified register of the given register class from the specified
//         stack frame index.
// This implements filling/reloading - i.e., load for spilled registers
//   (both scalar, and vector).
void ConnexInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, Register DestReg,
    int FI, const TargetRegisterClass *RC, const TargetRegisterInfo *TRI,
    Register VReg, MachineInstr::MIFlag Flags) const {
  DebugLoc DL;

  LLVM_DEBUG(dbgs() << "Entered ConnexInstrInfo::loadRegFromStackSlot(): FI = "
                    << FI << "\n");

  // assert(FI >= 2 && "I assumed wrong that frame index >= 2");

  // unsigned ConnexLSOffsetFillLoad = (CONNEX_MEM_NUM_ROWS + 1) - FI;
  unsigned ConnexLSOffsetFillLoad =
      (CONNEX_MEM_NUM_ROWS + CONNEX_MEM_NUM_ROWS_EXTRA_FOR_SPILL + 1) - FI;

  if (I != MBB.end())
    DL = I->getDebugLoc();

  if (RC == &Connex::GPRRegClass) {
    BuildMI(MBB, I, DL, get(Connex::LDD), DestReg).addFrameIndex(FI).addImm(0);
  } else if (RC == &Connex::VectorHRegClass) {
    /*
    // This actually generates a malformed scalar instruction with
    //   vector register
    BuildMI(MBB, I, DL, get(Connex::LDD), DestReg)
        .addFrameIndex(FI)
        .addImm(0);
    */
    /*
    // It is NOT correct since LLVM assumes it uses a stack and the
    // operations are sort of PUSH/POP. Even if Connex does NOT have
    // a stack, we can use LS to easily simulate it.
    BuildMI(MBB, I, DL, get(Connex::LD_H), DestReg)
        .addImm(CONNEX_MEM_NUM_ROWS - 1 - DestReg);
    */

    LLVM_DEBUG(dbgs() << "  ConnexInstrInfo::loadRegFromStackSlot(): Filling Wh"
                      << DestReg << " from ConnexLSOffsetFillLoad = "
                      << ConnexLSOffsetFillLoad << " (FI = " << FI << ")\n");

    /*
    Important: Adding the NOP is NOT required, since the iread Connex
       instruction does NOT require the insertion of a delay slot between
       them and the instruction that uses the register read from the LS memory.
    */
    BuildMI(MBB, I, DL, get(Connex::LD_FILL_H), DestReg)
        .addImm(ConnexLSOffsetFillLoad);
    /* TODO: get num vector registers from ConnexRegisterInfo.td:
         def VectorH: RegisterClass<"Connex", [v128i16], 32, */
  } else {
    llvm_unreachable("Connex back end: Can't load register from stack slot");
  }
}

bool ConnexInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *&TBB,
                                    MachineBasicBlock *&FBB,
                                    SmallVectorImpl<MachineOperand> &Cond,
                                    bool AllowModify) const {
  // Start from the bottom of the block and work up, examining the
  // terminator instructions.
  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;

    // Working from the bottom, when we see a non-terminator
    // instruction, we're done.
    if (!isUnpredicatedTerminator(*I))
      break;

    // A terminator that isn't a branch can't easily be handled
    // by this analysis.
    if (!I->isBranch())
      return true;

    // Handle unconditional branches.
    if (I->getOpcode() == Connex::JMP) {
      if (!AllowModify) {
        TBB = I->getOperand(0).getMBB();
        continue;
      }

      // If the block has any instructions after a J, delete them.
      while (std::next(I) != MBB.end())
        std::next(I)->eraseFromParent();
      Cond.clear();
      FBB = 0;

      // Delete the J if it's equivalent to a fall-through.
      if (MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
        TBB = 0;
        I->eraseFromParent();
        I = MBB.end();
        continue;
      }

      // TBB is used to indicate the unconditinal destination.
      TBB = I->getOperand(0).getMBB();
      continue;
    }
    // Cannot handle conditional branches
    return true;
  }

  return false;
}

unsigned ConnexInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  if (Cond.empty()) {
    // Unconditional branch
    assert(!FBB && "Unconditional branch with multiple successors!");
    BuildMI(&MBB, DL, get(Connex::JMP)).addMBB(TBB);
    return 1;
  }

  llvm_unreachable("Unexpected conditional branch");
}

unsigned ConnexInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                       int *BytesRemoved) const {
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;

  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    if (I->getOpcode() != Connex::JMP)
      break;
    // Remove the branch.
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }

  return Count;
}

/*
TODO: better implement it in ConnexTargetMachine::addPreRegAlloc(), in
  order to avoid any spills the register allocator might create.

Creating in ConnexInstrInfo::expandPostRAPseudo() bundle instructions
  with VLOAD_H_SYM_IMM + INLINEASM.
  This is a decent compromise although I do NOT use pseudo-instructions,
  using this after Register Allocation (PostRA) works because:
    - Important: INLINEASM is considered a pseudo-instruction (NOTE that
            VLOAD_H_SYM_IMM is NOT considered a pseudo-instruction);
    - pre-RA scheduler does NOT break the VLOAD_H_SYM_IMM from its associated
        INLINEASM;
    - register allocator does NOT break either the VLOAD_H_SYM_IMM from its
        associated INLINEASM, more exactly it doesn't insert spills or fills
        between the two instructions as far as I can see. Important: however I
        am NOT sure if this is always going to hold.
As of Feb 2017, class TargetInstrInfo
    (see http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html)
    has a few methods called on MachineInstr, but expandPostRAPseudo() seems
    to be a very good candidate (also it has no method with MachineSDNode).
  Anyhow, we could create and register our own pass working on MachineInstr in
   order to bundle instructions together (or on MachineSDNode, before pre-RA
   scheduler, although I guess it might be DIFFICULT to bundle from
   MachineSDNode to MachineInstr, since we have to perform a simple scheduling).

From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html
  <<This function is called for all pseudo instructions that remain after
     register allocation.
   Many pseudo instructions are created to help register allocation.
   This is the place to convert them into real instructions.
   The target can edit MI in place, or it can insert new instructions and
     erase MI.
   The function should return true if anything was changed.>>
*/
bool ConnexInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  // Making expandPostRAPseudo() do nothing:
  return false;

  LLVM_DEBUG(
      dbgs() << "ConnexInstrInfo::expandPostRAPseudo(): MI.getOpcode() = "
             << MI.getOpcode() << "\n");

  MachineBasicBlock *MBB = MI.getParent();
  DebugLoc DL = MBB->findDebugLoc(MI);

  /*
  // Inspired from lib/Target/PowerPC/PPCCTRLoops.cpp
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
       PIE = MBB->pred_end(); PI != PIE; ++PI)
    Preds.push_back(*PI);
  */
  switch (MI.getOpcode()) {
  default:
    // return expandPostRAPseudo(MI);
    return false;

  case Connex::VLOAD_H_SYM_IMM:
    // This is just a placeholder for register allocation.
    LLVM_DEBUG(
        dbgs()
        << "ConnexInstrInfo::expandPostRAPseudo(): found VLOAD_H_SYM_IMM\n");
    // MI.eraseFromParent();
    break;

  case Connex::INLINEASM:
    // This is just a placeholder for register allocation.
    LLVM_DEBUG(
        dbgs() << "ConnexInstrInfo::expandPostRAPseudo(): found INLINEASM\n");

    /*
    MachineInstr *predMI = NULL;
    MachineInstr *succMI = NULL;
    for (MachineBasicBlock::iterator I = MBB->begin(),
           IE = MBB->end(); I != IE; ++I) {
      MachineInstr *IMI = I;
      if (IMI == &MI) {
          I++;
          succMI = I;
          // predMI contains normally instruction VLOAD_H_SYM_IMM
          break;
      }
      predMI = I;
      LLVM_DEBUG(dbgs() << "expandPostRAPseudo(): (pred) I->getOpcode() = "
               << I->getOpcode() << "\n");
    }
    */
    MachineInstr *succMI;
    MachineInstr *predMI = getPredMachineInstr(&MI, &succMI);

    if (predMI != NULL) {
      LLVM_DEBUG(dbgs() << "expandPostRAPseudo(): *predMI = " << *predMI << "("
                        << predMI << ")"
                        << "\n");
      LLVM_DEBUG(dbgs() << "expandPostRAPseudo(): *succMI = " << *succMI << "("
                        << succMI << ")"
                        << "\n");
      LLVM_DEBUG(dbgs() << "expandPostRAPseudo(): MI = " << MI << "(" << &MI
                        << ")"
                        << "\n");

      if (predMI->getOpcode() == Connex::VLOAD_H_SYM_IMM) {
        // Inspired from lib/Target/AMDGPU/SIInstrInfo.cpp
        //    (or Mips/MipsDelaySlotFiller.cpp)
        /* Create a bundle so these instructions won't be re-ordered by the
         post-RA scheduler. */

        /*
       #ifdef THIS_DOES_NOT_ASMPRINT_BUNDLES
        MIBundleBuilder Bundler(*MBB, MI);

        LLVM_DEBUG(dbgs() << "expandPostRAPseudo(): predMI->getParent() = "
             << predMI->getParent() << "\n");

        // This must NOT be commented. Otherwise, it results in ~strange error
         in ConnexMCInstLower::Lower()
        predMI->eraseFromParent();
        LLVM_DEBUG(dbgs()
                     << "expandPostRAPseudo(): appending predMI to bundle\n");
        Bundler.append(predMI);

        LLVM_DEBUG(dbgs()
                     << "expandPostRAPseudo(): calling finalizeBundle()\n");
        // See llvm.org/docs/doxygen/html/MachineInstrBundle_8cpp_source.html
        llvm::finalizeBundle(*MBB, Bundler.begin());

        MI.eraseFromParent();

        #ifdef NOT_USEFUL
          // Inspired from
          //   llvm.org/docs/doxygen/html/MachineInstrBuilder_8h_source.html
          MI.bundleWithPred();
          // Does NOT compile: llvm::finalizeBundle(MBB, predMI);
        #endif
        */

        /* We now know that MI is the INLINEASM instruction that
         needs to be bundled with the previous instruction, predMI.
        */
        /*
        We do NOT use MIBundleBuilder,
            with eventual MI/predMI/succMI.eraseFromParent().
          Just predMI and succMI iterators.
          Note that succMI is required if we want to bundle
          instructions in the interval
          predMI..MI, where succMI = succ(MI).

        So we normally bundle here: predMI, MI (without succMI).
        */
        /* See llvm.org/docs/doxygen/html/MachineInstrBundle_8cpp_source.html
           and llvm.org/docs/doxygen/html/MachineInstrBundle_8cpp_source.html
        */
        llvm::finalizeBundle(*MBB, (MachineBasicBlock::instr_iterator)predMI,
                             (MachineBasicBlock::instr_iterator)succMI);
        // (MachineBasicBlock::instr_iterator)&MI);

        /*
        // See llvm.org/docs/doxygen/html/classllvm_1_1MIBundleBuilder.html
        // MIBundleBuilder(MachineBasicBlock &BB,
        //                 MachineBasicBlock::iterator B,
        //                 MachineBasicBlock::iterator E)
        // Create a bundle from the sequence of instructions between B and E.
        MIBundleBuilder Bundler(*MBB, predMI, MI);

        // MI.eraseFromParent();
        // Bundler.append(&MI);

        // Bundler.append(&MI);
        //

        // Gives error
        // include/llvm/CodeGen/MachineInstrBundleIterator.h:42:
        // llvm::MachineInstrBundleIterator<Ty>::
        // MachineInstrBundleIterator(Ty*)[with Ty = llvm::MachineInstr]:
        // Assertion `(!MI || !MI->isBundledWithPred()) && "It's not legal to
        // initialize " "MachineInstrBundleIterator "
        // "with a bundled MI"' failed.
        ////MIBundleBuilder Bundler(*MBB, predMI, *succMI);

        // See llvm.org/docs/doxygen/html/MachineInstrBundle_8cpp_source.html
        llvm::finalizeBundle(*MBB, Bundler.begin());

        MI.eraseFromParent();

        // This yields error <<[with Ty = llvm::MachineInstr]:
        // Assertion `(!MI || !MI->isBundledWithPred()) &&
        // "It's not legal to initialize " "MachineInstrBundleIterator "
        // "with a bundled MI"' failed.>>
        // predMI->eraseFromParent();
      */
      }
    }

    break;
  }

  LLVM_DEBUG(dbgs() << "Before exit expandPostRAPseudo():\n");
  // Gives error since MI can be bundled: <<Assertion `!MI.isBundledWithPred()
  //   && "It's not legal to initialize " "MachineInstrBundleIterator with a "
  //   "bundled MI"' failed.>> MachineBasicBlock &MBB = *(MI.getParent());

  // From http://llvm.org/docs/doxygen/html/classllvm_1_1MachineBasicBlock.html
  // for (auto it: *MBB)
  for (MachineBasicBlock::iterator I = MBB->begin(), IE = MBB->end(); I != IE;
       ++I) {
    /*
    LLVM_DEBUG(dbgs()
      << "ConnexInstrInfo::expandPostRAPseudo(): it->getOpcode() = "
      << it->getOpcode() << "\n");
    */
    LLVM_DEBUG(dbgs() << "  I = " << *I << "\n");
    /*
    switch (MI.getOpcode()) {
    }
    */
  }

  /*
  const SIRegisterInfo *TRI
    = static_cast<const SIRegisterInfo *>(ST.getRegisterInfo());
  MachineFunction &MF = MBB->getParent();
  unsigned Reg = MI.getOperand(0).getReg();
  unsigned RegLo = TRI->getSubReg(Reg, AMDGPU::sub0);
  unsigned RegHi = TRI->getSubReg(Reg, AMDGPU::sub1);

  // Create a bundle so these instructions won't be re-ordered by the
  // post-RA scheduler.
  MIBundleBuilder Bundler(*MBB, MI);
  Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_GETPC_B64), Reg));

  // Add 32-bit offset from this instruction to the start of the
  // constant data.
  Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_ADD_U32), RegLo)
                     .addReg(RegLo)
                     .addOperand(MI.getOperand(1)));

  llvm::finalizeBundle(*MBB, Bundler.begin());

  MI.eraseFromParent();
  break;
  */

  return false;
} // End ConnexInstrInfo::expandPostRAPseudo()

// USE_POSTRA_SCHED
// Inspired from llvm/lib/Target/PowerPC/PPCInstrInfo.cpp
// See http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html
ScheduleHazardRecognizer *ConnexInstrInfo::CreateTargetPostRAHazardRecognizer(
    const InstrItineraryData *II, const ScheduleDAG *DAG) const {
  LLVM_DEBUG(
      dbgs()
      << "Entered ConnexInstrInfo::CreateTargetPostRAHazardRecognizer()\n");

  return new ConnexDispatchGroupSBHazardRecognizer(II, DAG);
}

/*
ScheduleHazardRecognizer *
ConnexInstrInfo::CreateTargetPostRAHazardRecognizer(const MachineFunction &MF)
                                                                        const {
  LLVM_DEBUG(dbgs()
                   << "Entered ConnexInstrInfo::"
                      "CreateTargetPostRAHazardRecognizer(MachineFunction)\n");

// TODO: Get inspired from AMDGPU how they added separate
// PostRA HazardRecognizer.
//    See http://llvm.org/doxygen/classllvm_1_1MachineFunction.html
  return new ConnexDispatchGroupSBHazardRecognizer(II, DAG);
}
*/

// Pre-RA mach. instr. scheduler hazard recognizer
//  I guess this method is called if I give llc -enable-misched,
//    which invokes MIScheduler
// (see e.g. https://llvm.org/devmtg/2016-09/slides/Absar-SchedulingInOrder.pdf)
// See http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html
ScheduleHazardRecognizer *ConnexInstrInfo::CreateTargetMIHazardRecognizer(
    const InstrItineraryData *II,
    const ScheduleDAGMI *DAG) const {
  LLVM_DEBUG(
      dbgs() << "Entered ConnexInstrInfo::CreateTargetMIHazardRecognizer()\n");

  llvm_unreachable("ConnexInstrInfo::CreateTargetMIHazardRecognizer() "
                   "not implemented");
  // return new ConnexDispatchGroupSBHazardRecognizerPreRAScheduler(II, DAG);
}

/*
// USE_PRERA_HAZARD_RECOGNIZER

// Pre-RA scheduler - default scheduler (no special param given to llc)
// See http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html
ScheduleHazardRecognizer *ConnexInstrInfo::CreateTargetHazardRecognizer(
                                                const TargetSubtargetInfo *STI,
                                                const ScheduleDAG *DAG) const {
  LLVM_DEBUG(dbgs()
    << "Entered ConnexInstrInfo::CreateTargetHazardRecognizer()\n");

  return new ConnexDispatchGroupSBHazardRecognizerPreRAScheduler(
  // See http://llvm.org/docs/doxygen/html/TargetSubtargetInfo_8h_source.html
                    STI->getInstrItineraryData(),
                    DAG);
}
*/

// Inspired from llvm/lib/Target/PowerPC/PPCInstrInfo.cpp
void ConnexInstrInfo::insertNoop(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstrInfo::insertNoop()\n");

  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(Connex::NOP));
}

// From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetInstrInfo.html:
//              <<Return true if the specified instruction can be predicated.>>
/* From http://llvm.org/docs/doxygen/html/classllvm_1_1MachineInstr.html:
  <<bool isPredicable (QueryType Type=AllInBundle) const
  Return true if this instruction has a predicate operand that
    controls execution.>>
*/
// Inspired from ARMBaseInstrInfo::isPredicable
bool ConnexInstrInfo::isPredicable(MachineInstr &MI) const {
  // if (!MI.isPredicable())
  //  return false;
  LLVM_DEBUG(dbgs() << "ConnexInstrInfo::isPredicable(): MI.getOpcode() = "
                    << MI.getOpcode() << "\n");

  if (MI.getOpcode() == Connex::VLOAD_H) {
    return true;
  }

  return false;
}
