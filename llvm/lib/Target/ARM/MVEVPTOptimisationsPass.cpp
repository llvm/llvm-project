//===-- MVEVPTOptimisationsPass.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass does a few optimisations related to Tail predicated loops
/// and MVE VPT blocks before register allocation is performed. For VPT blocks
/// the goal is to maximize the sizes of the blocks that will be created by the
/// MVE VPT Block Insertion pass (which runs after register allocation). For
/// tail predicated loops we transform the loop into something that will
/// hopefully make the backend ARMLowOverheadLoops pass's job easier.
///
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMSubtarget.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "arm-mve-vpt-opts"

namespace {
class MVEVPTOptimisations : public MachineFunctionPass {
public:
  static char ID;
  const Thumb2InstrInfo *TII;
  MachineRegisterInfo *MRI;

  MVEVPTOptimisations() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &Fn) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "ARM MVE TailPred and VPT Optimisation Pass";
  }

private:
  bool ConvertTailPredLoop(MachineLoop *ML, MachineDominatorTree *DT);
  MachineInstr &ReplaceRegisterUseWithVPNOT(MachineBasicBlock &MBB,
                                            MachineInstr &Instr,
                                            MachineOperand &User,
                                            Register Target);
  bool ReduceOldVCCRValueUses(MachineBasicBlock &MBB);
  bool ReplaceVCMPsByVPNOTs(MachineBasicBlock &MBB);
  bool ConvertVPSEL(MachineBasicBlock &MBB);
};

char MVEVPTOptimisations::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(MVEVPTOptimisations, DEBUG_TYPE,
                      "ARM MVE TailPred and VPT Optimisations pass", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(MVEVPTOptimisations, DEBUG_TYPE,
                    "ARM MVE TailPred and VPT Optimisations pass", false, false)

static MachineInstr *LookThroughCOPY(MachineInstr *MI,
                                     MachineRegisterInfo *MRI) {
  while (MI && MI->getOpcode() == TargetOpcode::COPY &&
         MI->getOperand(1).getReg().isVirtual())
    MI = MRI->getVRegDef(MI->getOperand(1).getReg());
  return MI;
}

// Given a loop ML, this attempts to find the t2LoopEnd, t2LoopDec and
// corresponding PHI that make up a low overhead loop. Only handles 'do' loops
// at the moment, returning a t2DoLoopStart in LoopStart.
static bool findLoopComponents(MachineLoop *ML, MachineRegisterInfo *MRI,
                               MachineInstr *&LoopStart, MachineInstr *&LoopPhi,
                               MachineInstr *&LoopDec, MachineInstr *&LoopEnd) {
  MachineBasicBlock *Header = ML->getHeader();
  MachineBasicBlock *Latch = ML->getLoopLatch();
  if (!Header || !Latch) {
    LLVM_DEBUG(dbgs() << "  no Loop Latch or Header\n");
    return false;
  }

  // Find the loop end from the terminators.
  LoopEnd = nullptr;
  for (auto &T : Latch->terminators()) {
    if (T.getOpcode() == ARM::t2LoopEnd && T.getOperand(1).getMBB() == Header) {
      LoopEnd = &T;
      break;
    }
  }
  if (!LoopEnd) {
    LLVM_DEBUG(dbgs() << "  no LoopEnd\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "  found loop end: " << *LoopEnd);

  // Find the dec from the use of the end. There may be copies between
  // instructions. We expect the loop to loop like:
  //   $vs = t2DoLoopStart ...
  // loop:
  //   $vp = phi [ $vs ], [ $vd ]
  //   ...
  //   $vd = t2LoopDec $vp
  //   ...
  //   t2LoopEnd $vd, loop
  LoopDec =
      LookThroughCOPY(MRI->getVRegDef(LoopEnd->getOperand(0).getReg()), MRI);
  if (!LoopDec || LoopDec->getOpcode() != ARM::t2LoopDec) {
    LLVM_DEBUG(dbgs() << "  didn't find LoopDec where we expected!\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "  found loop dec: " << *LoopDec);

  LoopPhi =
      LookThroughCOPY(MRI->getVRegDef(LoopDec->getOperand(1).getReg()), MRI);
  if (!LoopPhi || LoopPhi->getOpcode() != TargetOpcode::PHI ||
      LoopPhi->getNumOperands() != 5 ||
      (LoopPhi->getOperand(2).getMBB() != Latch &&
       LoopPhi->getOperand(4).getMBB() != Latch)) {
    LLVM_DEBUG(dbgs() << "  didn't find PHI where we expected!\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "  found loop phi: " << *LoopPhi);

  Register StartReg = LoopPhi->getOperand(2).getMBB() == Latch
                          ? LoopPhi->getOperand(3).getReg()
                          : LoopPhi->getOperand(1).getReg();
  LoopStart = LookThroughCOPY(MRI->getVRegDef(StartReg), MRI);
  if (!LoopStart || LoopStart->getOpcode() != ARM::t2DoLoopStart) {
    LLVM_DEBUG(dbgs() << "  didn't find Start where we expected!\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << "  found loop start: " << *LoopStart);

  return true;
}

// Convert t2DoLoopStart to t2DoLoopStartTP if the loop contains VCTP
// instructions. This keeps the VCTP count reg operand on the t2DoLoopStartTP
// instruction, making the backend ARMLowOverheadLoops passes job of finding the
// VCTP operand much simpler.
bool MVEVPTOptimisations::ConvertTailPredLoop(MachineLoop *ML,
                                              MachineDominatorTree *DT) {
  LLVM_DEBUG(dbgs() << "ConvertTailPredLoop on loop "
                    << ML->getHeader()->getName() << "\n");

  // Find some loop components including the LoopEnd/Dec/Start, and any VCTP's
  // in the loop.
  MachineInstr *LoopEnd, *LoopPhi, *LoopStart, *LoopDec;
  if (!findLoopComponents(ML, MRI, LoopStart, LoopPhi, LoopDec, LoopEnd))
    return false;

  SmallVector<MachineInstr *, 4> VCTPs;
  for (MachineBasicBlock *BB : ML->blocks())
    for (MachineInstr &MI : *BB)
      if (isVCTP(&MI))
        VCTPs.push_back(&MI);

  if (VCTPs.empty()) {
    LLVM_DEBUG(dbgs() << "  no VCTPs\n");
    return false;
  }

  // Check all VCTPs are the same.
  MachineInstr *FirstVCTP = *VCTPs.begin();
  for (MachineInstr *VCTP : VCTPs) {
    LLVM_DEBUG(dbgs() << "  with VCTP " << *VCTP);
    if (VCTP->getOpcode() != FirstVCTP->getOpcode() ||
        VCTP->getOperand(0).getReg() != FirstVCTP->getOperand(0).getReg()) {
      LLVM_DEBUG(dbgs() << "  VCTP's are not identical\n");
      return false;
    }
  }

  // Check for the register being used can be setup before the loop. We expect
  // this to be:
  //   $vx = ...
  // loop:
  //   $vp = PHI [ $vx ], [ $vd ]
  //   ..
  //   $vpr = VCTP $vp
  //   ..
  //   $vd = t2SUBri $vp, #n
  //   ..
  Register CountReg = FirstVCTP->getOperand(1).getReg();
  if (!CountReg.isVirtual()) {
    LLVM_DEBUG(dbgs() << "  cannot determine VCTP PHI\n");
    return false;
  }
  MachineInstr *Phi = LookThroughCOPY(MRI->getVRegDef(CountReg), MRI);
  if (!Phi || Phi->getOpcode() != TargetOpcode::PHI ||
      Phi->getNumOperands() != 5 ||
      (Phi->getOperand(2).getMBB() != ML->getLoopLatch() &&
       Phi->getOperand(4).getMBB() != ML->getLoopLatch())) {
    LLVM_DEBUG(dbgs() << "  cannot determine VCTP Count\n");
    return false;
  }
  CountReg = Phi->getOperand(2).getMBB() == ML->getLoopLatch()
                 ? Phi->getOperand(3).getReg()
                 : Phi->getOperand(1).getReg();

  // Replace the t2DoLoopStart with the t2DoLoopStartTP, move it to the end of
  // the preheader and add the new CountReg to it. We attempt to place it late
  // in the preheader, but may need to move that earlier based on uses.
  MachineBasicBlock *MBB = LoopStart->getParent();
  MachineBasicBlock::iterator InsertPt = MBB->getFirstTerminator();
  for (MachineInstr &Use :
       MRI->use_instructions(LoopStart->getOperand(0).getReg()))
    if ((InsertPt != MBB->end() && !DT->dominates(&*InsertPt, &Use)) ||
        !DT->dominates(ML->getHeader(), Use.getParent()))
      InsertPt = &Use;
  if (InsertPt != MBB->end() &&
      !DT->dominates(MRI->getVRegDef(CountReg), &*InsertPt)) {
    LLVM_DEBUG(dbgs() << "  InsertPt does not dominate CountReg!\n");
    return false;
  }

  MachineInstrBuilder MI = BuildMI(*MBB, InsertPt, LoopStart->getDebugLoc(),
                                   TII->get(ARM::t2DoLoopStartTP))
                               .add(LoopStart->getOperand(0))
                               .add(LoopStart->getOperand(1))
                               .addReg(CountReg);
  (void)MI;
  LLVM_DEBUG(dbgs() << "Replacing " << *LoopStart << "  with "
                    << *MI.getInstr());
  MRI->constrainRegClass(CountReg, &ARM::rGPRRegClass);
  LoopStart->eraseFromParent();

  return true;
}

// Returns true if Opcode is any VCMP Opcode.
static bool IsVCMP(unsigned Opcode) { return VCMPOpcodeToVPT(Opcode) != 0; }

// Returns true if a VCMP with this Opcode can have its operands swapped.
// There is 2 kind of VCMP that can't have their operands swapped: Float VCMPs,
// and VCMPr instructions (since the r is always on the right).
static bool CanHaveSwappedOperands(unsigned Opcode) {
  switch (Opcode) {
  default:
    return true;
  case ARM::MVE_VCMPf32:
  case ARM::MVE_VCMPf16:
  case ARM::MVE_VCMPf32r:
  case ARM::MVE_VCMPf16r:
  case ARM::MVE_VCMPi8r:
  case ARM::MVE_VCMPi16r:
  case ARM::MVE_VCMPi32r:
  case ARM::MVE_VCMPu8r:
  case ARM::MVE_VCMPu16r:
  case ARM::MVE_VCMPu32r:
  case ARM::MVE_VCMPs8r:
  case ARM::MVE_VCMPs16r:
  case ARM::MVE_VCMPs32r:
    return false;
  }
}

// Returns the CondCode of a VCMP Instruction.
static ARMCC::CondCodes GetCondCode(MachineInstr &Instr) {
  assert(IsVCMP(Instr.getOpcode()) && "Inst must be a VCMP");
  return ARMCC::CondCodes(Instr.getOperand(3).getImm());
}

// Returns true if Cond is equivalent to a VPNOT instruction on the result of
// Prev. Cond and Prev must be VCMPs.
static bool IsVPNOTEquivalent(MachineInstr &Cond, MachineInstr &Prev) {
  assert(IsVCMP(Cond.getOpcode()) && IsVCMP(Prev.getOpcode()));

  // Opcodes must match.
  if (Cond.getOpcode() != Prev.getOpcode())
    return false;

  MachineOperand &CondOP1 = Cond.getOperand(1), &CondOP2 = Cond.getOperand(2);
  MachineOperand &PrevOP1 = Prev.getOperand(1), &PrevOP2 = Prev.getOperand(2);

  // If the VCMP has the opposite condition with the same operands, we can
  // replace it with a VPNOT
  ARMCC::CondCodes ExpectedCode = GetCondCode(Cond);
  ExpectedCode = ARMCC::getOppositeCondition(ExpectedCode);
  if (ExpectedCode == GetCondCode(Prev))
    if (CondOP1.isIdenticalTo(PrevOP1) && CondOP2.isIdenticalTo(PrevOP2))
      return true;
  // Check again with operands swapped if possible
  if (!CanHaveSwappedOperands(Cond.getOpcode()))
    return false;
  ExpectedCode = ARMCC::getSwappedCondition(ExpectedCode);
  return ExpectedCode == GetCondCode(Prev) && CondOP1.isIdenticalTo(PrevOP2) &&
         CondOP2.isIdenticalTo(PrevOP1);
}

// Returns true if Instr writes to VCCR.
static bool IsWritingToVCCR(MachineInstr &Instr) {
  if (Instr.getNumOperands() == 0)
    return false;
  MachineOperand &Dst = Instr.getOperand(0);
  if (!Dst.isReg())
    return false;
  Register DstReg = Dst.getReg();
  if (!DstReg.isVirtual())
    return false;
  MachineRegisterInfo &RegInfo = Instr.getMF()->getRegInfo();
  const TargetRegisterClass *RegClass = RegInfo.getRegClassOrNull(DstReg);
  return RegClass && (RegClass->getID() == ARM::VCCRRegClassID);
}

// Transforms
//    <Instr that uses %A ('User' Operand)>
// Into
//    %K = VPNOT %Target
//    <Instr that uses %K ('User' Operand)>
// And returns the newly inserted VPNOT.
// This optimization is done in the hopes of preventing spills/reloads of VPR by
// reducing the number of VCCR values with overlapping lifetimes.
MachineInstr &MVEVPTOptimisations::ReplaceRegisterUseWithVPNOT(
    MachineBasicBlock &MBB, MachineInstr &Instr, MachineOperand &User,
    Register Target) {
  Register NewResult = MRI->createVirtualRegister(MRI->getRegClass(Target));

  MachineInstrBuilder MIBuilder =
      BuildMI(MBB, &Instr, Instr.getDebugLoc(), TII->get(ARM::MVE_VPNOT))
          .addDef(NewResult)
          .addReg(Target);
  addUnpredicatedMveVpredNOp(MIBuilder);

  // Make the user use NewResult instead, and clear its kill flag.
  User.setReg(NewResult);
  User.setIsKill(false);

  LLVM_DEBUG(dbgs() << "  Inserting VPNOT (for spill prevention): ";
             MIBuilder.getInstr()->dump());

  return *MIBuilder.getInstr();
}

// Moves a VPNOT before its first user if an instruction that uses Reg is found
// in-between the VPNOT and its user.
// Returns true if there is at least one user of the VPNOT in the block.
static bool MoveVPNOTBeforeFirstUser(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator Iter,
                                     Register Reg) {
  assert(Iter->getOpcode() == ARM::MVE_VPNOT && "Not a VPNOT!");
  assert(getVPTInstrPredicate(*Iter) == ARMVCC::None &&
         "The VPNOT cannot be predicated");

  MachineInstr &VPNOT = *Iter;
  Register VPNOTResult = VPNOT.getOperand(0).getReg();
  Register VPNOTOperand = VPNOT.getOperand(1).getReg();

  // Whether the VPNOT will need to be moved, and whether we found a user of the
  // VPNOT.
  bool MustMove = false, HasUser = false;
  MachineOperand *VPNOTOperandKiller = nullptr;
  for (; Iter != MBB.end(); ++Iter) {
    if (MachineOperand *MO =
            Iter->findRegisterUseOperand(VPNOTOperand, /*isKill*/ true)) {
      // If we find the operand that kills the VPNOTOperand's result, save it.
      VPNOTOperandKiller = MO;
    }

    if (Iter->findRegisterUseOperandIdx(Reg) != -1) {
      MustMove = true;
      continue;
    }

    if (Iter->findRegisterUseOperandIdx(VPNOTResult) == -1)
      continue;

    HasUser = true;
    if (!MustMove)
      break;

    // Move the VPNOT right before Iter
    LLVM_DEBUG(dbgs() << "Moving: "; VPNOT.dump(); dbgs() << "  Before: ";
               Iter->dump());
    MBB.splice(Iter, &MBB, VPNOT.getIterator());
    // If we move the instr, and its operand was killed earlier, remove the kill
    // flag.
    if (VPNOTOperandKiller)
      VPNOTOperandKiller->setIsKill(false);

    break;
  }
  return HasUser;
}

// This optimisation attempts to reduce the number of overlapping lifetimes of
// VCCR values by replacing uses of old VCCR values with VPNOTs. For example,
// this replaces
//    %A:vccr = (something)
//    %B:vccr = VPNOT %A
//    %Foo = (some op that uses %B)
//    %Bar = (some op that uses %A)
// With
//    %A:vccr = (something)
//    %B:vccr = VPNOT %A
//    %Foo = (some op that uses %B)
//    %TMP2:vccr = VPNOT %B
//    %Bar = (some op that uses %A)
bool MVEVPTOptimisations::ReduceOldVCCRValueUses(MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator Iter = MBB.begin(), End = MBB.end();
  SmallVector<MachineInstr *, 4> DeadInstructions;
  bool Modified = false;

  while (Iter != End) {
    Register VCCRValue, OppositeVCCRValue;
    // The first loop looks for 2 unpredicated instructions:
    //    %A:vccr = (instr)     ; A is stored in VCCRValue
    //    %B:vccr = VPNOT %A    ; B is stored in OppositeVCCRValue
    for (; Iter != End; ++Iter) {
      // We're only interested in unpredicated instructions that write to VCCR.
      if (!IsWritingToVCCR(*Iter) ||
          getVPTInstrPredicate(*Iter) != ARMVCC::None)
        continue;
      Register Dst = Iter->getOperand(0).getReg();

      // If we already have a VCCRValue, and this is a VPNOT on VCCRValue, we've
      // found what we were looking for.
      if (VCCRValue && Iter->getOpcode() == ARM::MVE_VPNOT &&
          Iter->findRegisterUseOperandIdx(VCCRValue) != -1) {
        // Move the VPNOT closer to its first user if needed, and ignore if it
        // has no users.
        if (!MoveVPNOTBeforeFirstUser(MBB, Iter, VCCRValue))
          continue;

        OppositeVCCRValue = Dst;
        ++Iter;
        break;
      }

      // Else, just set VCCRValue.
      VCCRValue = Dst;
    }

    // If the first inner loop didn't find anything, stop here.
    if (Iter == End)
      break;

    assert(VCCRValue && OppositeVCCRValue &&
           "VCCRValue and OppositeVCCRValue shouldn't be empty if the loop "
           "stopped before the end of the block!");
    assert(VCCRValue != OppositeVCCRValue &&
           "VCCRValue should not be equal to OppositeVCCRValue!");

    // LastVPNOTResult always contains the same value as OppositeVCCRValue.
    Register LastVPNOTResult = OppositeVCCRValue;

    // This second loop tries to optimize the remaining instructions.
    for (; Iter != End; ++Iter) {
      bool IsInteresting = false;

      if (MachineOperand *MO = Iter->findRegisterUseOperand(VCCRValue)) {
        IsInteresting = true;

        // - If the instruction is a VPNOT, it can be removed, and we can just
        //   replace its uses with LastVPNOTResult.
        // - Else, insert a new VPNOT on LastVPNOTResult to recompute VCCRValue.
        if (Iter->getOpcode() == ARM::MVE_VPNOT) {
          Register Result = Iter->getOperand(0).getReg();

          MRI->replaceRegWith(Result, LastVPNOTResult);
          DeadInstructions.push_back(&*Iter);
          Modified = true;

          LLVM_DEBUG(dbgs()
                     << "Replacing all uses of '" << printReg(Result)
                     << "' with '" << printReg(LastVPNOTResult) << "'\n");
        } else {
          MachineInstr &VPNOT =
              ReplaceRegisterUseWithVPNOT(MBB, *Iter, *MO, LastVPNOTResult);
          Modified = true;

          LastVPNOTResult = VPNOT.getOperand(0).getReg();
          std::swap(VCCRValue, OppositeVCCRValue);

          LLVM_DEBUG(dbgs() << "Replacing use of '" << printReg(VCCRValue)
                            << "' with '" << printReg(LastVPNOTResult)
                            << "' in instr: " << *Iter);
        }
      } else {
        // If the instr uses OppositeVCCRValue, make it use LastVPNOTResult
        // instead as they contain the same value.
        if (MachineOperand *MO =
                Iter->findRegisterUseOperand(OppositeVCCRValue)) {
          IsInteresting = true;

          // This is pointless if LastVPNOTResult == OppositeVCCRValue.
          if (LastVPNOTResult != OppositeVCCRValue) {
            LLVM_DEBUG(dbgs() << "Replacing usage of '"
                              << printReg(OppositeVCCRValue) << "' with '"
                              << printReg(LastVPNOTResult) << " for instr: ";
                       Iter->dump());
            MO->setReg(LastVPNOTResult);
            Modified = true;
          }

          MO->setIsKill(false);
        }

        // If this is an unpredicated VPNOT on
        // LastVPNOTResult/OppositeVCCRValue, we can act like we inserted it.
        if (Iter->getOpcode() == ARM::MVE_VPNOT &&
            getVPTInstrPredicate(*Iter) == ARMVCC::None) {
          Register VPNOTOperand = Iter->getOperand(1).getReg();
          if (VPNOTOperand == LastVPNOTResult ||
              VPNOTOperand == OppositeVCCRValue) {
            IsInteresting = true;

            std::swap(VCCRValue, OppositeVCCRValue);
            LastVPNOTResult = Iter->getOperand(0).getReg();
          }
        }
      }

      // If this instruction was not interesting, and it writes to VCCR, stop.
      if (!IsInteresting && IsWritingToVCCR(*Iter))
        break;
    }
  }

  for (MachineInstr *DeadInstruction : DeadInstructions)
    DeadInstruction->eraseFromParent();

  return Modified;
}

// This optimisation replaces VCMPs with VPNOTs when they are equivalent.
bool MVEVPTOptimisations::ReplaceVCMPsByVPNOTs(MachineBasicBlock &MBB) {
  SmallVector<MachineInstr *, 4> DeadInstructions;

  // The last VCMP that we have seen and that couldn't be replaced.
  // This is reset when an instruction that writes to VCCR/VPR is found, or when
  // a VCMP is replaced with a VPNOT.
  // We'll only replace VCMPs with VPNOTs when this is not null, and when the
  // current VCMP is the opposite of PrevVCMP.
  MachineInstr *PrevVCMP = nullptr;
  // If we find an instruction that kills the result of PrevVCMP, we save the
  // operand here to remove the kill flag in case we need to use PrevVCMP's
  // result.
  MachineOperand *PrevVCMPResultKiller = nullptr;

  for (MachineInstr &Instr : MBB.instrs()) {
    if (PrevVCMP) {
      if (MachineOperand *MO = Instr.findRegisterUseOperand(
              PrevVCMP->getOperand(0).getReg(), /*isKill*/ true)) {
        // If we come accross the instr that kills PrevVCMP's result, record it
        // so we can remove the kill flag later if we need to.
        PrevVCMPResultKiller = MO;
      }
    }

    // Ignore predicated instructions.
    if (getVPTInstrPredicate(Instr) != ARMVCC::None)
      continue;

    // Only look at VCMPs
    if (!IsVCMP(Instr.getOpcode())) {
      // If the instruction writes to VCCR, forget the previous VCMP.
      if (IsWritingToVCCR(Instr))
        PrevVCMP = nullptr;
      continue;
    }

    if (!PrevVCMP || !IsVPNOTEquivalent(Instr, *PrevVCMP)) {
      PrevVCMP = &Instr;
      continue;
    }

    // The register containing the result of the VCMP that we're going to
    // replace.
    Register PrevVCMPResultReg = PrevVCMP->getOperand(0).getReg();

    // Build a VPNOT to replace the VCMP, reusing its operands.
    MachineInstrBuilder MIBuilder =
        BuildMI(MBB, &Instr, Instr.getDebugLoc(), TII->get(ARM::MVE_VPNOT))
            .add(Instr.getOperand(0))
            .addReg(PrevVCMPResultReg);
    addUnpredicatedMveVpredNOp(MIBuilder);
    LLVM_DEBUG(dbgs() << "Inserting VPNOT (to replace VCMP): ";
               MIBuilder.getInstr()->dump(); dbgs() << "  Removed VCMP: ";
               Instr.dump());

    // If we found an instruction that uses, and kills PrevVCMP's result,
    // remove the kill flag.
    if (PrevVCMPResultKiller)
      PrevVCMPResultKiller->setIsKill(false);

    // Finally, mark the old VCMP for removal and reset
    // PrevVCMP/PrevVCMPResultKiller.
    DeadInstructions.push_back(&Instr);
    PrevVCMP = nullptr;
    PrevVCMPResultKiller = nullptr;
  }

  for (MachineInstr *DeadInstruction : DeadInstructions)
    DeadInstruction->eraseFromParent();

  return !DeadInstructions.empty();
}

// Replace VPSEL with a predicated VMOV in blocks with a VCTP. This is a
// somewhat blunt approximation to allow tail predicated with vpsel
// instructions. We turn a vselect into a VPSEL in ISEL, but they have slightly
// different semantics under tail predication. Until that is modelled we just
// convert to a VMOVT (via a predicated VORR) instead.
bool MVEVPTOptimisations::ConvertVPSEL(MachineBasicBlock &MBB) {
  bool HasVCTP = false;
  SmallVector<MachineInstr *, 4> DeadInstructions;

  for (MachineInstr &MI : MBB.instrs()) {
    if (isVCTP(&MI)) {
      HasVCTP = true;
      continue;
    }

    if (!HasVCTP || MI.getOpcode() != ARM::MVE_VPSEL)
      continue;

    MachineInstrBuilder MIBuilder =
        BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(ARM::MVE_VORR))
            .add(MI.getOperand(0))
            .add(MI.getOperand(1))
            .add(MI.getOperand(1))
            .addImm(ARMVCC::Then)
            .add(MI.getOperand(4))
            .add(MI.getOperand(2));
    // Silence unused variable warning in release builds.
    (void)MIBuilder;
    LLVM_DEBUG(dbgs() << "Replacing VPSEL: "; MI.dump();
               dbgs() << "     with VMOVT: "; MIBuilder.getInstr()->dump());
    DeadInstructions.push_back(&MI);
  }

  for (MachineInstr *DeadInstruction : DeadInstructions)
    DeadInstruction->eraseFromParent();

  return !DeadInstructions.empty();
}

bool MVEVPTOptimisations::runOnMachineFunction(MachineFunction &Fn) {
  const ARMSubtarget &STI =
      static_cast<const ARMSubtarget &>(Fn.getSubtarget());

  if (!STI.isThumb2() || !STI.hasMVEIntegerOps())
    return false;

  TII = static_cast<const Thumb2InstrInfo *>(STI.getInstrInfo());
  MRI = &Fn.getRegInfo();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();
  MachineDominatorTree *DT = &getAnalysis<MachineDominatorTree>();

  LLVM_DEBUG(dbgs() << "********** ARM MVE VPT Optimisations **********\n"
                    << "********** Function: " << Fn.getName() << '\n');

  bool Modified = false;
  for (MachineLoop *ML : MLI->getBase().getLoopsInPreorder())
    Modified |= ConvertTailPredLoop(ML, DT);

  for (MachineBasicBlock &MBB : Fn) {
    Modified |= ReplaceVCMPsByVPNOTs(MBB);
    Modified |= ReduceOldVCCRValueUses(MBB);
    Modified |= ConvertVPSEL(MBB);
  }

  LLVM_DEBUG(dbgs() << "**************************************\n");
  return Modified;
}

/// createMVEVPTOptimisationsPass
FunctionPass *llvm::createMVEVPTOptimisationsPass() {
  return new MVEVPTOptimisations();
}
