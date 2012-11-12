//===-- PPCCTRLoops.cpp - Identify and generate CTR loops -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass identifies loops where we can generate the PPC branch instructions
// that decrement and test the count register (CTR) (bdnz and friends).
// This pass is based on the HexagonHardwareLoops pass.
//
// The pattern that defines the induction variable can changed depending on
// prior optimizations.  For example, the IndVarSimplify phase run by 'opt'
// normalizes induction variables, and the Loop Strength Reduction pass
// run by 'llc' may also make changes to the induction variable.
// The pattern detected by this phase is due to running Strength Reduction.
//
// Criteria for CTR loops:
//  - Countable loops (w/ ind. var for a trip count)
//  - Assumes loops are normalized by IndVarSimplify
//  - Try inner-most loops first
//  - No nested CTR loops.
//  - No function calls in loops.
//
//  Note: As with unconverted loops, PPCBranchSelector must be run after this
//  pass in order to convert long-displacement jumps into jump pairs.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ctrloops"
#include "PPC.h"
#include "PPCTargetMachine.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/Constants.h"
#include "llvm/PassSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include <algorithm>

using namespace llvm;

STATISTIC(NumCTRLoops, "Number of loops converted to CTR loops");

namespace {
  class CountValue;
  struct PPCCTRLoops : public MachineFunctionPass {
    MachineLoopInfo       *MLI;
    MachineRegisterInfo   *MRI;
    const TargetInstrInfo *TII;

  public:
    static char ID;   // Pass identification, replacement for typeid

    PPCCTRLoops() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    const char *getPassName() const { return "PPC CTR Loops"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    /// getCanonicalInductionVariable - Check to see if the loop has a canonical
    /// induction variable.
    /// Should be defined in MachineLoop. Based upon version in class Loop.
    void getCanonicalInductionVariable(MachineLoop *L,
                              SmallVector<MachineInstr *, 4> &IVars,
                              SmallVector<MachineInstr *, 4> &IOps) const;

    /// getTripCount - Return a loop-invariant LLVM register indicating the
    /// number of times the loop will be executed.  If the trip-count cannot
    /// be determined, this return null.
    CountValue *getTripCount(MachineLoop *L,
                             SmallVector<MachineInstr *, 2> &OldInsts) const;

    /// isInductionOperation - Return true if the instruction matches the
    /// pattern for an opertion that defines an induction variable.
    bool isInductionOperation(const MachineInstr *MI, unsigned IVReg) const;

    /// isInvalidOperation - Return true if the instruction is not valid within
    /// a CTR loop.
    bool isInvalidLoopOperation(const MachineInstr *MI) const;

    /// containsInavlidInstruction - Return true if the loop contains an
    /// instruction that inhibits using the CTR loop.
    bool containsInvalidInstruction(MachineLoop *L) const;

    /// converToCTRLoop - Given a loop, check if we can convert it to a
    /// CTR loop.  If so, then perform the conversion and return true.
    bool convertToCTRLoop(MachineLoop *L);

    /// isDead - Return true if the instruction is now dead.
    bool isDead(const MachineInstr *MI,
                SmallVector<MachineInstr *, 1> &DeadPhis) const;

    /// removeIfDead - Remove the instruction if it is now dead.
    void removeIfDead(MachineInstr *MI);
  };

  char PPCCTRLoops::ID = 0;


  // CountValue class - Abstraction for a trip count of a loop. A
  // smaller vesrsion of the MachineOperand class without the concerns
  // of changing the operand representation.
  class CountValue {
  public:
    enum CountValueType {
      CV_Register,
      CV_Immediate
    };
  private:
    CountValueType Kind;
    union Values {
      unsigned RegNum;
      int64_t ImmVal;
      Values(unsigned r) : RegNum(r) {}
      Values(int64_t i) : ImmVal(i) {}
    } Contents;
    bool isNegative;

  public:
    CountValue(unsigned r, bool neg) : Kind(CV_Register), Contents(r),
                                       isNegative(neg) {}
    explicit CountValue(int64_t i) : Kind(CV_Immediate), Contents(i),
                                     isNegative(i < 0) {}
    CountValueType getType() const { return Kind; }
    bool isReg() const { return Kind == CV_Register; }
    bool isImm() const { return Kind == CV_Immediate; }
    bool isNeg() const { return isNegative; }

    unsigned getReg() const {
      assert(isReg() && "Wrong CountValue accessor");
      return Contents.RegNum;
    }
    void setReg(unsigned Val) {
      Contents.RegNum = Val;
    }
    int64_t getImm() const {
      assert(isImm() && "Wrong CountValue accessor");
      if (isNegative) {
        return -Contents.ImmVal;
      }
      return Contents.ImmVal;
    }
    void setImm(int64_t Val) {
      Contents.ImmVal = Val;
    }

    void print(raw_ostream &OS, const TargetMachine *TM = 0) const {
      if (isReg()) { OS << PrintReg(getReg()); }
      if (isImm()) { OS << getImm(); }
    }
  };
} // end anonymous namespace


/// isCompareEquals - Returns true if the instruction is a compare equals
/// instruction with an immediate operand.
static bool isCompareEqualsImm(const MachineInstr *MI, bool &SignedCmp) {
  if (MI->getOpcode() == PPC::CMPWI || MI->getOpcode() == PPC::CMPDI) {
    SignedCmp = true;
    return true;
  } else if (MI->getOpcode() == PPC::CMPLWI || MI->getOpcode() == PPC::CMPLDI) {
    SignedCmp = false;
    return true;
  }

  return false;
}


/// createPPCCTRLoops - Factory for creating
/// the CTR loop phase.
FunctionPass *llvm::createPPCCTRLoops() {
  return new PPCCTRLoops();
}


bool PPCCTRLoops::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********* PPC CTR Loops *********\n");

  bool Changed = false;

  // get the loop information
  MLI = &getAnalysis<MachineLoopInfo>();
  // get the register information
  MRI = &MF.getRegInfo();
  // the target specific instructio info.
  TII = MF.getTarget().getInstrInfo();

  for (MachineLoopInfo::iterator I = MLI->begin(), E = MLI->end();
       I != E; ++I) {
    MachineLoop *L = *I;
    if (!L->getParentLoop()) {
      Changed |= convertToCTRLoop(L);
    }
  }

  return Changed;
}

/// getCanonicalInductionVariable - Check to see if the loop has a canonical
/// induction variable. We check for a simple recurrence pattern - an
/// integer recurrence that decrements by one each time through the loop and
/// ends at zero.  If so, return the phi node that corresponds to it.
///
/// Based upon the similar code in LoopInfo except this code is specific to
/// the machine.
/// This method assumes that the IndVarSimplify pass has been run by 'opt'.
///
void
PPCCTRLoops::getCanonicalInductionVariable(MachineLoop *L,
                                  SmallVector<MachineInstr *, 4> &IVars,
                                  SmallVector<MachineInstr *, 4> &IOps) const {
  MachineBasicBlock *TopMBB = L->getTopBlock();
  MachineBasicBlock::pred_iterator PI = TopMBB->pred_begin();
  assert(PI != TopMBB->pred_end() &&
         "Loop must have more than one incoming edge!");
  MachineBasicBlock *Backedge = *PI++;
  if (PI == TopMBB->pred_end()) return;  // dead loop
  MachineBasicBlock *Incoming = *PI++;
  if (PI != TopMBB->pred_end()) return;  // multiple backedges?

  // make sure there is one incoming and one backedge and determine which
  // is which.
  if (L->contains(Incoming)) {
    if (L->contains(Backedge))
      return;
    std::swap(Incoming, Backedge);
  } else if (!L->contains(Backedge))
    return;

  // Loop over all of the PHI nodes, looking for a canonical induction variable:
  //   - The PHI node is "reg1 = PHI reg2, BB1, reg3, BB2".
  //   - The recurrence comes from the backedge.
  //   - the definition is an induction operatio.n
  for (MachineBasicBlock::iterator I = TopMBB->begin(), E = TopMBB->end();
       I != E && I->isPHI(); ++I) {
    MachineInstr *MPhi = &*I;
    unsigned DefReg = MPhi->getOperand(0).getReg();
    for (unsigned i = 1; i != MPhi->getNumOperands(); i += 2) {
      // Check each operand for the value from the backedge.
      MachineBasicBlock *MBB = MPhi->getOperand(i+1).getMBB();
      if (L->contains(MBB)) { // operands comes from the backedge
        // Check if the definition is an induction operation.
        MachineInstr *DI = MRI->getVRegDef(MPhi->getOperand(i).getReg());
        if (isInductionOperation(DI, DefReg)) {
          IOps.push_back(DI);
          IVars.push_back(MPhi);
        }
      }
    }
  }
  return;
}

/// getTripCount - Return a loop-invariant LLVM value indicating the
/// number of times the loop will be executed.  The trip count can
/// be either a register or a constant value.  If the trip-count
/// cannot be determined, this returns null.
///
/// We find the trip count from the phi instruction that defines the
/// induction variable.  We follow the links to the CMP instruction
/// to get the trip count.
///
/// Based upon getTripCount in LoopInfo.
///
CountValue *PPCCTRLoops::getTripCount(MachineLoop *L,
                           SmallVector<MachineInstr *, 2> &OldInsts) const {
  MachineBasicBlock *LastMBB = L->getExitingBlock();
  // Don't generate a CTR loop if the loop has more than one exit.
  if (LastMBB == 0)
    return 0;

  MachineBasicBlock::iterator LastI = LastMBB->getFirstTerminator();
  if (LastI->getOpcode() != PPC::BCC)
    return 0;

  // We need to make sure that this compare is defining the condition
  // register actually used by the terminating branch.

  unsigned PredReg = LastI->getOperand(1).getReg();
  DEBUG(dbgs() << "Examining loop with first terminator: " << *LastI);

  unsigned PredCond = LastI->getOperand(0).getImm();
  if (PredCond != PPC::PRED_EQ && PredCond != PPC::PRED_NE)
    return 0;

  // Check that the loop has a induction variable.
  SmallVector<MachineInstr *, 4> IVars, IOps;
  getCanonicalInductionVariable(L, IVars, IOps);
  for (unsigned i = 0; i < IVars.size(); ++i) {
    MachineInstr *IOp = IOps[i];
    MachineInstr *IV_Inst = IVars[i];

    // Canonical loops will end with a 'cmpwi/cmpdi cr, IV, Imm',
    //  if Imm is 0, get the count from the PHI opnd
    //  if Imm is -M, than M is the count
    //  Otherwise, Imm is the count
    MachineOperand *IV_Opnd;
    const MachineOperand *InitialValue;
    if (!L->contains(IV_Inst->getOperand(2).getMBB())) {
      InitialValue = &IV_Inst->getOperand(1);
      IV_Opnd = &IV_Inst->getOperand(3);
    } else {
      InitialValue = &IV_Inst->getOperand(3);
      IV_Opnd = &IV_Inst->getOperand(1);
    }

    DEBUG(dbgs() << "Considering:\n");
    DEBUG(dbgs() << "  induction operation: " << *IOp);
    DEBUG(dbgs() << "  induction variable: " << *IV_Inst);
    DEBUG(dbgs() << "  initial value: " << *InitialValue << "\n");
  
    // Look for the cmp instruction to determine if we
    // can get a useful trip count.  The trip count can
    // be either a register or an immediate.  The location
    // of the value depends upon the type (reg or imm).
    for (MachineRegisterInfo::reg_iterator
         RI = MRI->reg_begin(IV_Opnd->getReg()), RE = MRI->reg_end();
         RI != RE; ++RI) {
      IV_Opnd = &RI.getOperand();
      bool SignedCmp;
      MachineInstr *MI = IV_Opnd->getParent();
      if (L->contains(MI) && isCompareEqualsImm(MI, SignedCmp) &&
          MI->getOperand(0).getReg() == PredReg) {

        OldInsts.push_back(MI);
        OldInsts.push_back(IOp);
 
        DEBUG(dbgs() << "  compare: " << *MI);
 
        const MachineOperand &MO = MI->getOperand(2);
        assert(MO.isImm() && "IV Cmp Operand should be an immediate");

        int64_t ImmVal;
        if (SignedCmp)
          ImmVal = (short) MO.getImm();
        else
          ImmVal = MO.getImm();
  
        const MachineInstr *IV_DefInstr = MRI->getVRegDef(IV_Opnd->getReg());
        assert(L->contains(IV_DefInstr->getParent()) &&
               "IV definition should occurs in loop");
        int64_t iv_value = (short) IV_DefInstr->getOperand(2).getImm();
  
        assert(InitialValue->isReg() && "Expecting register for init value");
        unsigned InitialValueReg = InitialValue->getReg();
  
        const MachineInstr *DefInstr = MRI->getVRegDef(InitialValueReg);
  
        // Here we need to look for an immediate load (an li or lis/ori pair).
        if (DefInstr && (DefInstr->getOpcode() == PPC::ORI8 ||
                         DefInstr->getOpcode() == PPC::ORI)) {
          int64_t start = (short) DefInstr->getOperand(2).getImm();
          const MachineInstr *DefInstr2 =
            MRI->getVRegDef(DefInstr->getOperand(0).getReg());
          if (DefInstr2 && (DefInstr2->getOpcode() == PPC::LIS8 ||
                            DefInstr2->getOpcode() == PPC::LIS)) {
            DEBUG(dbgs() << "  initial constant: " << *DefInstr);
            DEBUG(dbgs() << "  initial constant: " << *DefInstr2);

            start |= int64_t(short(DefInstr2->getOperand(1).getImm())) << 16;
  
            int64_t count = ImmVal - start;
            if ((count % iv_value) != 0) {
              return 0;
            }
            return new CountValue(count/iv_value);
          }
        } else if (DefInstr && (DefInstr->getOpcode() == PPC::LI8 ||
                                DefInstr->getOpcode() == PPC::LI)) {
          DEBUG(dbgs() << "  initial constant: " << *DefInstr);

          int64_t count = ImmVal - int64_t(short(DefInstr->getOperand(1).getImm()));
          if ((count % iv_value) != 0) {
            return 0;
          }
          return new CountValue(count/iv_value);
        } else if (iv_value == 1 || iv_value == -1) {
          // We can't determine a constant starting value.
          if (ImmVal == 0) {
            return new CountValue(InitialValueReg, iv_value > 0);
          }
          // FIXME: handle non-zero end value.
        }
        // FIXME: handle non-unit increments (we might not want to introduce division
        // but we can handle some 2^n cases with shifts).
  
      }
    }
  }
  return 0;
}

/// isInductionOperation - return true if the operation is matches the
/// pattern that defines an induction variable:
///    addi iv, c
///
bool
PPCCTRLoops::isInductionOperation(const MachineInstr *MI,
                                           unsigned IVReg) const {
  return ((MI->getOpcode() == PPC::ADDI || MI->getOpcode() == PPC::ADDI8) &&
          MI->getOperand(1).isReg() && // could be a frame index instead
          MI->getOperand(1).getReg() == IVReg);
}

/// isInvalidOperation - Return true if the operation is invalid within
/// CTR loop.
bool
PPCCTRLoops::isInvalidLoopOperation(const MachineInstr *MI) const {

  // call is not allowed because the callee may use a CTR loop
  if (MI->getDesc().isCall()) {
    return true;
  }
  // check if the instruction defines a CTR loop register
  // (this will also catch nested CTR loops)
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef() &&
        (MO.getReg() == PPC::CTR || MO.getReg() == PPC::CTR8)) {
      return true;
    }
  }
  return false;
}

/// containsInvalidInstruction - Return true if the loop contains
/// an instruction that inhibits the use of the CTR loop function.
///
bool PPCCTRLoops::containsInvalidInstruction(MachineLoop *L) const {
  const std::vector<MachineBasicBlock*> Blocks = L->getBlocks();
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    MachineBasicBlock *MBB = Blocks[i];
    for (MachineBasicBlock::iterator
           MII = MBB->begin(), E = MBB->end(); MII != E; ++MII) {
      const MachineInstr *MI = &*MII;
      if (isInvalidLoopOperation(MI)) {
        return true;
      }
    }
  }
  return false;
}

/// isDead returns true if the instruction is dead
/// (this was essentially copied from DeadMachineInstructionElim::isDead, but
/// with special cases for inline asm, physical registers and instructions with
/// side effects removed)
bool PPCCTRLoops::isDead(const MachineInstr *MI,
                         SmallVector<MachineInstr *, 1> &DeadPhis) const {
  // Examine each operand.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef()) {
      unsigned Reg = MO.getReg();
      if (!MRI->use_nodbg_empty(Reg)) {
        // This instruction has users, but if the only user is the phi node for the
        // parent block, and the only use of that phi node is this instruction, then
        // this instruction is dead: both it (and the phi node) can be removed.
        MachineRegisterInfo::use_iterator I = MRI->use_begin(Reg);
        if (llvm::next(I) == MRI->use_end() &&
            I.getOperand().getParent()->isPHI()) {
          MachineInstr *OnePhi = I.getOperand().getParent();

          for (unsigned j = 0, f = OnePhi->getNumOperands(); j != f; ++j) {
            const MachineOperand &OPO = OnePhi->getOperand(j);
            if (OPO.isReg() && OPO.isDef()) {
              unsigned OPReg = OPO.getReg();

              MachineRegisterInfo::use_iterator nextJ;
              for (MachineRegisterInfo::use_iterator J = MRI->use_begin(OPReg),
                   E = MRI->use_end(); J!=E; J=nextJ) {
                nextJ = llvm::next(J);
                MachineOperand& Use = J.getOperand();
                MachineInstr *UseMI = Use.getParent();

                if (MI != UseMI) {
                  // The phi node has a user that is not MI, bail...
                  return false;
                }
              }
            }
          }

          DeadPhis.push_back(OnePhi);
        } else {
          // This def has a non-debug use. Don't delete the instruction!
          return false;
        }
      }
    }
  }

  // If there are no defs with uses, the instruction is dead.
  return true;
}

void PPCCTRLoops::removeIfDead(MachineInstr *MI) {
  // This procedure was essentially copied from DeadMachineInstructionElim

  SmallVector<MachineInstr *, 1> DeadPhis;
  if (isDead(MI, DeadPhis)) {
    DEBUG(dbgs() << "CTR looping will remove: " << *MI);

    // It is possible that some DBG_VALUE instructions refer to this
    // instruction.  Examine each def operand for such references;
    // if found, mark the DBG_VALUE as undef (but don't delete it).
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned Reg = MO.getReg();
      MachineRegisterInfo::use_iterator nextI;
      for (MachineRegisterInfo::use_iterator I = MRI->use_begin(Reg),
           E = MRI->use_end(); I!=E; I=nextI) {
        nextI = llvm::next(I);  // I is invalidated by the setReg
        MachineOperand& Use = I.getOperand();
        MachineInstr *UseMI = Use.getParent();
        if (UseMI==MI)
          continue;
        if (Use.isDebug()) // this might also be a instr -> phi -> instr case
                           // which can also be removed.
          UseMI->getOperand(0).setReg(0U);
      }
    }

    MI->eraseFromParent();
    for (unsigned i = 0; i < DeadPhis.size(); ++i) {
      DeadPhis[i]->eraseFromParent();
    }
  }
}

/// converToCTRLoop - check if the loop is a candidate for
/// converting to a CTR loop.  If so, then perform the
/// transformation.
///
/// This function works on innermost loops first.  A loop can
/// be converted if it is a counting loop; either a register
/// value or an immediate.
///
/// The code makes several assumptions about the representation
/// of the loop in llvm.
bool PPCCTRLoops::convertToCTRLoop(MachineLoop *L) {
  bool Changed = false;
  // Process nested loops first.
  for (MachineLoop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
    Changed |= convertToCTRLoop(*I);
  }
  // If a nested loop has been converted, then we can't convert this loop.
  if (Changed) {
    return Changed;
  }

  SmallVector<MachineInstr *, 2> OldInsts;
  // Are we able to determine the trip count for the loop?
  CountValue *TripCount = getTripCount(L, OldInsts);
  if (TripCount == 0) {
    DEBUG(dbgs() << "failed to get trip count!\n");
    return false;
  }
  // Does the loop contain any invalid instructions?
  if (containsInvalidInstruction(L)) {
    return false;
  }
  MachineBasicBlock *Preheader = L->getLoopPreheader();
  // No preheader means there's not place for the loop instr.
  if (Preheader == 0) {
    return false;
  }
  MachineBasicBlock::iterator InsertPos = Preheader->getFirstTerminator();

  DebugLoc dl;
  if (InsertPos != Preheader->end())
    dl = InsertPos->getDebugLoc();

  MachineBasicBlock *LastMBB = L->getExitingBlock();
  // Don't generate CTR loop if the loop has more than one exit.
  if (LastMBB == 0) {
    return false;
  }
  MachineBasicBlock::iterator LastI = LastMBB->getFirstTerminator();

  // Determine the loop start.
  MachineBasicBlock *LoopStart = L->getTopBlock();
  if (L->getLoopLatch() != LastMBB) {
    // When the exit and latch are not the same, use the latch block as the
    // start.
    // The loop start address is used only after the 1st iteration, and the loop
    // latch may contains instrs. that need to be executed after the 1st iter.
    LoopStart = L->getLoopLatch();
    // Make sure the latch is a successor of the exit, otherwise it won't work.
    if (!LastMBB->isSuccessor(LoopStart)) {
      return false;
    }
  }

  // Convert the loop to a CTR loop
  DEBUG(dbgs() << "Change to CTR loop at "; L->dump());

  MachineFunction *MF = LastMBB->getParent();
  const PPCSubtarget &Subtarget = MF->getTarget().getSubtarget<PPCSubtarget>();
  bool isPPC64 = Subtarget.isPPC64();

  const TargetRegisterClass *GPRC = &PPC::GPRCRegClass;
  const TargetRegisterClass *G8RC = &PPC::G8RCRegClass;
  const TargetRegisterClass *RC = isPPC64 ? G8RC : GPRC;

  unsigned CountReg;
  if (TripCount->isReg()) {
    // Create a copy of the loop count register.
    const TargetRegisterClass *SrcRC =
      MF->getRegInfo().getRegClass(TripCount->getReg());
    CountReg = MF->getRegInfo().createVirtualRegister(RC);
    unsigned CopyOp = (isPPC64 && SrcRC == GPRC) ?
                        (unsigned) PPC::EXTSW_32_64 :
                        (unsigned) TargetOpcode::COPY;
    BuildMI(*Preheader, InsertPos, dl,
            TII->get(CopyOp), CountReg).addReg(TripCount->getReg());
    if (TripCount->isNeg()) {
      unsigned CountReg1 = CountReg;
      CountReg = MF->getRegInfo().createVirtualRegister(RC);
      BuildMI(*Preheader, InsertPos, dl,
              TII->get(isPPC64 ? PPC::NEG8 : PPC::NEG),
                       CountReg).addReg(CountReg1);
    }
  } else {
    assert(TripCount->isImm() && "Expecting immedate vaule for trip count");
    // Put the trip count in a register for transfer into the count register.

    int64_t CountImm = TripCount->getImm();
    assert(!TripCount->isNeg() && "Constant trip count must be positive");

    CountReg = MF->getRegInfo().createVirtualRegister(RC);
    if (CountImm > 0xFFFF) {
      BuildMI(*Preheader, InsertPos, dl,
              TII->get(isPPC64 ? PPC::LIS8 : PPC::LIS),
              CountReg).addImm(CountImm >> 16);
      unsigned CountReg1 = CountReg;
      CountReg = MF->getRegInfo().createVirtualRegister(RC);
      BuildMI(*Preheader, InsertPos, dl,
              TII->get(isPPC64 ? PPC::ORI8 : PPC::ORI),
              CountReg).addReg(CountReg1).addImm(CountImm & 0xFFFF);
    } else {
      BuildMI(*Preheader, InsertPos, dl,
              TII->get(isPPC64 ? PPC::LI8 : PPC::LI),
              CountReg).addImm(CountImm);
    }
  }

  // Add the mtctr instruction to the beginning of the loop.
  BuildMI(*Preheader, InsertPos, dl,
          TII->get(isPPC64 ? PPC::MTCTR8 : PPC::MTCTR)).addReg(CountReg,
            TripCount->isImm() ? RegState::Kill : 0);

  // Make sure the loop start always has a reference in the CFG.  We need to
  // create a BlockAddress operand to get this mechanism to work both the
  // MachineBasicBlock and BasicBlock objects need the flag set.
  LoopStart->setHasAddressTaken();
  // This line is needed to set the hasAddressTaken flag on the BasicBlock
  // object
  BlockAddress::get(const_cast<BasicBlock *>(LoopStart->getBasicBlock()));

  // Replace the loop branch with a bdnz instruction.
  dl = LastI->getDebugLoc();
  const std::vector<MachineBasicBlock*> Blocks = L->getBlocks();
  for (unsigned i = 0, e = Blocks.size(); i != e; ++i) {
    MachineBasicBlock *MBB = Blocks[i];
    if (MBB != Preheader)
      MBB->addLiveIn(isPPC64 ? PPC::CTR8 : PPC::CTR);
  }

  // The loop ends with either:
  //  - a conditional branch followed by an unconditional branch, or
  //  - a conditional branch to the loop start.
  assert(LastI->getOpcode() == PPC::BCC &&
         "loop end must start with a BCC instruction");
  // Either the BCC branches to the beginning of the loop, or it
  // branches out of the loop and there is an unconditional branch
  // to the start of the loop.
  MachineBasicBlock *BranchTarget = LastI->getOperand(2).getMBB();
  BuildMI(*LastMBB, LastI, dl,
        TII->get((BranchTarget == LoopStart) ?
                 (isPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                 (isPPC64 ? PPC::BDZ8 : PPC::BDZ))).addMBB(BranchTarget);

  // Conditional branch; just delete it.
  DEBUG(dbgs() << "Removing old branch: " << *LastI);
  LastMBB->erase(LastI);

  delete TripCount;

  // The induction operation (add) and the comparison (cmpwi) may now be
  // unneeded. If these are unneeded, then remove them.
  for (unsigned i = 0; i < OldInsts.size(); ++i)
    removeIfDead(OldInsts[i]);

  ++NumCTRLoops;
  return true;
}

