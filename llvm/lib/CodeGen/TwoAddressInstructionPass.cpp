//===-- TwoAddressInstructionPass.cpp - Two-Address instruction pass ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TwoAddress instruction pass which is used
// by most register allocators. Two-Address instructions are rewritten
// from:
//
//     A = B op C
//
// to:
//
//     A = B
//     A op= C
//
// Note that if a register allocator chooses to use this pass, that it
// has to be capable of handling the non-SSA nature of these rewritten
// virtual registers.
//
// It is also worth noting that the duplicate operand of the two
// address instruction is removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "twoaddrinstr"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumTwoAddressInstrs, "Number of two-address instructions");
STATISTIC(NumCommuted        , "Number of instructions commuted to coalesce");
STATISTIC(NumAggrCommuted    , "Number of instructions aggressively commuted");
STATISTIC(NumConvertedTo3Addr, "Number of instructions promoted to 3-address");
STATISTIC(Num3AddrSunk,        "Number of 3-address instructions sunk");
STATISTIC(NumReMats,           "Number of instructions re-materialized");

namespace {
  class VISIBILITY_HIDDEN TwoAddressInstructionPass
    : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo *MRI;
    LiveVariables *LV;

    bool Sink3AddrInstruction(MachineBasicBlock *MBB, MachineInstr *MI,
                              unsigned Reg,
                              MachineBasicBlock::iterator OldPos);

    bool isProfitableToReMat(unsigned Reg, const TargetRegisterClass *RC,
                             MachineInstr *MI, MachineInstr *DefMI,
                             MachineBasicBlock *MBB, unsigned Loc,
                             DenseMap<MachineInstr*, unsigned> &DistanceMap);

    bool NoUseAfterLastDef(unsigned Reg, MachineBasicBlock *MBB, unsigned Dist,
                           DenseMap<MachineInstr*, unsigned> &DistanceMap,
                           unsigned &LastDef);

    bool isProfitableToCommute(unsigned regB, unsigned regC,
                               MachineInstr *MI, MachineBasicBlock *MBB,
                               unsigned Dist,
                               DenseMap<MachineInstr*, unsigned> &DistanceMap);

    bool CommuteInstruction(MachineBasicBlock::iterator &mi,
                            MachineFunction::iterator &mbbi,
                            unsigned RegC, unsigned Dist,
                            DenseMap<MachineInstr*, unsigned> &DistanceMap);
  public:
    static char ID; // Pass identification, replacement for typeid
    TwoAddressInstructionPass() : MachineFunctionPass(&ID) {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<LiveVariables>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addPreservedID(MachineDominatorsID);
      if (StrongPHIElim)
        AU.addPreservedID(StrongPHIEliminationID);
      else
        AU.addPreservedID(PHIEliminationID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - Pass entry point.
    bool runOnMachineFunction(MachineFunction&);
  };
}

char TwoAddressInstructionPass::ID = 0;
static RegisterPass<TwoAddressInstructionPass>
X("twoaddressinstruction", "Two-Address instruction pass");

const PassInfo *const llvm::TwoAddressInstructionPassID = &X;

/// Sink3AddrInstruction - A two-address instruction has been converted to a
/// three-address instruction to avoid clobbering a register. Try to sink it
/// past the instruction that would kill the above mentioned register to reduce
/// register pressure.
bool TwoAddressInstructionPass::Sink3AddrInstruction(MachineBasicBlock *MBB,
                                           MachineInstr *MI, unsigned SavedReg,
                                           MachineBasicBlock::iterator OldPos) {
  // Check if it's safe to move this instruction.
  bool SeenStore = true; // Be conservative.
  if (!MI->isSafeToMove(TII, SeenStore))
    return false;

  unsigned DefReg = 0;
  SmallSet<unsigned, 4> UseRegs;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned MOReg = MO.getReg();
    if (!MOReg)
      continue;
    if (MO.isUse() && MOReg != SavedReg)
      UseRegs.insert(MO.getReg());
    if (!MO.isDef())
      continue;
    if (MO.isImplicit())
      // Don't try to move it if it implicitly defines a register.
      return false;
    if (DefReg)
      // For now, don't move any instructions that define multiple registers.
      return false;
    DefReg = MO.getReg();
  }

  // Find the instruction that kills SavedReg.
  MachineInstr *KillMI = NULL;
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(SavedReg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    if (!UseMO.isKill())
      continue;
    KillMI = UseMO.getParent();
    break;
  }

  if (!KillMI || KillMI->getParent() != MBB)
    return false;

  // If any of the definitions are used by another instruction between the
  // position and the kill use, then it's not safe to sink it.
  // 
  // FIXME: This can be sped up if there is an easy way to query whether an
  // instruction is before or after another instruction. Then we can use
  // MachineRegisterInfo def / use instead.
  MachineOperand *KillMO = NULL;
  MachineBasicBlock::iterator KillPos = KillMI;
  ++KillPos;

  unsigned NumVisited = 0;
  for (MachineBasicBlock::iterator I = next(OldPos); I != KillPos; ++I) {
    MachineInstr *OtherMI = I;
    if (NumVisited > 30)  // FIXME: Arbitrary limit to reduce compile time cost.
      return false;
    ++NumVisited;
    for (unsigned i = 0, e = OtherMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = OtherMI->getOperand(i);
      if (!MO.isReg())
        continue;
      unsigned MOReg = MO.getReg();
      if (!MOReg)
        continue;
      if (DefReg == MOReg)
        return false;

      if (MO.isKill()) {
        if (OtherMI == KillMI && MOReg == SavedReg)
          // Save the operand that kills the register. We want to unset the kill
          // marker if we can sink MI past it.
          KillMO = &MO;
        else if (UseRegs.count(MOReg))
          // One of the uses is killed before the destination.
          return false;
      }
    }
  }

  // Update kill and LV information.
  KillMO->setIsKill(false);
  KillMO = MI->findRegisterUseOperand(SavedReg, false, TRI);
  KillMO->setIsKill(true);
  
  if (LV)
    LV->replaceKillInstruction(SavedReg, KillMI, MI);

  // Move instruction to its destination.
  MBB->remove(MI);
  MBB->insert(KillPos, MI);

  ++Num3AddrSunk;
  return true;
}

/// isTwoAddrUse - Return true if the specified MI is using the specified
/// register as a two-address operand.
static bool isTwoAddrUse(MachineInstr *UseMI, unsigned Reg) {
  const TargetInstrDesc &TID = UseMI->getDesc();
  for (unsigned i = 0, e = TID.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = UseMI->getOperand(i);
    if (MO.isReg() && MO.getReg() == Reg &&
        (MO.isDef() || TID.getOperandConstraint(i, TOI::TIED_TO) != -1))
      // Earlier use is a two-address one.
      return true;
  }
  return false;
}

/// isProfitableToReMat - Return true if the heuristics determines it is likely
/// to be profitable to re-materialize the definition of Reg rather than copy
/// the register.
bool
TwoAddressInstructionPass::isProfitableToReMat(unsigned Reg,
                                const TargetRegisterClass *RC,
                                MachineInstr *MI, MachineInstr *DefMI,
                                MachineBasicBlock *MBB, unsigned Loc,
                                DenseMap<MachineInstr*, unsigned> &DistanceMap){
  bool OtherUse = false;
  for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(Reg),
         UE = MRI->use_end(); UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = UseMO.getParent();
    MachineBasicBlock *UseMBB = UseMI->getParent();
    if (UseMBB == MBB) {
      DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(UseMI);
      if (DI != DistanceMap.end() && DI->second == Loc)
        continue;  // Current use.
      OtherUse = true;
      // There is at least one other use in the MBB that will clobber the
      // register. 
      if (isTwoAddrUse(UseMI, Reg))
        return true;
    }
  }

  // If other uses in MBB are not two-address uses, then don't remat.
  if (OtherUse)
    return false;

  // No other uses in the same block, remat if it's defined in the same
  // block so it does not unnecessarily extend the live range.
  return MBB == DefMI->getParent();
}

/// NoUseAfterLastDef - Return true if there are no intervening uses between the
/// last instruction in the MBB that defines the specified register and the
/// two-address instruction which is being processed. It also returns the last
/// def location by reference
bool TwoAddressInstructionPass::NoUseAfterLastDef(unsigned Reg,
                                 MachineBasicBlock *MBB, unsigned Dist,
                                 DenseMap<MachineInstr*, unsigned> &DistanceMap,
                                 unsigned &LastDef) {
  LastDef = 0;
  unsigned LastUse = Dist;
  for (MachineRegisterInfo::reg_iterator I = MRI->reg_begin(Reg),
         E = MRI->reg_end(); I != E; ++I) {
    MachineOperand &MO = I.getOperand();
    MachineInstr *MI = MO.getParent();
    if (MI->getParent() != MBB)
      continue;
    DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(MI);
    if (DI == DistanceMap.end())
      continue;
    if (MO.isUse() && DI->second < LastUse)
      LastUse = DI->second;
    if (MO.isDef() && DI->second > LastDef)
      LastDef = DI->second;
  }

  return !(LastUse > LastDef && LastUse < Dist);
}

/// isProfitableToReMat - Return true if it's potentially profitable to commute
/// the two-address instruction that's being processed.
bool
TwoAddressInstructionPass::isProfitableToCommute(unsigned regB, unsigned regC,
                MachineInstr *MI, MachineBasicBlock *MBB,
                unsigned Dist, DenseMap<MachineInstr*, unsigned> &DistanceMap) {
  // Determine if it's profitable to commute this two address instruction. In
  // general, we want no uses between this instruction and the definition of
  // the two-address register.
  // e.g.
  // %reg1028<def> = EXTRACT_SUBREG %reg1027<kill>, 1
  // %reg1029<def> = MOV8rr %reg1028
  // %reg1029<def> = SHR8ri %reg1029, 7, %EFLAGS<imp-def,dead>
  // insert => %reg1030<def> = MOV8rr %reg1028
  // %reg1030<def> = ADD8rr %reg1028<kill>, %reg1029<kill>, %EFLAGS<imp-def,dead>
  // In this case, it might not be possible to coalesce the second MOV8rr
  // instruction if the first one is coalesced. So it would be profitable to
  // commute it:
  // %reg1028<def> = EXTRACT_SUBREG %reg1027<kill>, 1
  // %reg1029<def> = MOV8rr %reg1028
  // %reg1029<def> = SHR8ri %reg1029, 7, %EFLAGS<imp-def,dead>
  // insert => %reg1030<def> = MOV8rr %reg1029
  // %reg1030<def> = ADD8rr %reg1029<kill>, %reg1028<kill>, %EFLAGS<imp-def,dead>  

  if (!MI->killsRegister(regC))
    return false;

  // Ok, we have something like:
  // %reg1030<def> = ADD8rr %reg1028<kill>, %reg1029<kill>, %EFLAGS<imp-def,dead>
  // let's see if it's worth commuting it.

  // If there is a use of regC between its last def (could be livein) and this
  // instruction, then bail.
  unsigned LastDefC = 0;
  if (!NoUseAfterLastDef(regC, MBB, Dist, DistanceMap, LastDefC))
    return false;

  // If there is a use of regB between its last def (could be livein) and this
  // instruction, then go ahead and make this transformation.
  unsigned LastDefB = 0;
  if (!NoUseAfterLastDef(regB, MBB, Dist, DistanceMap, LastDefB))
    return true;

  // Since there are no intervening uses for both registers, then commute
  // if the def of regC is closer. Its live interval is shorter.
  return LastDefB && LastDefC && LastDefC > LastDefB;
}

/// CommuteInstruction - Commute a two-address instruction and update the basic
/// block, distance map, and live variables if needed. Return true if it is
/// successful.
bool
TwoAddressInstructionPass::CommuteInstruction(MachineBasicBlock::iterator &mi,
                                              MachineFunction::iterator &mbbi,
                                              unsigned RegC, unsigned Dist,
                               DenseMap<MachineInstr*, unsigned> &DistanceMap) {
  MachineInstr *MI = mi;
  DOUT << "2addr: COMMUTING  : " << *MI;
  MachineInstr *NewMI = TII->commuteInstruction(MI);

  if (NewMI == 0) {
    DOUT << "2addr: COMMUTING FAILED!\n";
    return false;
  }

  DOUT << "2addr: COMMUTED TO: " << *NewMI;
  // If the instruction changed to commute it, update livevar.
  if (NewMI != MI) {
    if (LV)
      // Update live variables
      LV->replaceKillInstruction(RegC, MI, NewMI);

    mbbi->insert(mi, NewMI);           // Insert the new inst
    mbbi->erase(mi);                   // Nuke the old inst.
    mi = NewMI;
    DistanceMap.insert(std::make_pair(NewMI, Dist));
  }
  return true;
}

/// runOnMachineFunction - Reduce two-address instructions to two operands.
///
bool TwoAddressInstructionPass::runOnMachineFunction(MachineFunction &MF) {
  DOUT << "Machine Function\n";
  const TargetMachine &TM = MF.getTarget();
  MRI = &MF.getRegInfo();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  LV = getAnalysisIfAvailable<LiveVariables>();

  bool MadeChange = false;

  DOUT << "********** REWRITING TWO-ADDR INSTRS **********\n";
  DOUT << "********** Function: " << MF.getFunction()->getName() << '\n';

  // ReMatRegs - Keep track of the registers whose def's are remat'ed.
  BitVector ReMatRegs;
  ReMatRegs.resize(MRI->getLastVirtReg()+1);

  // DistanceMap - Keep track the distance of a MI from the start of the
  // current basic block.
  DenseMap<MachineInstr*, unsigned> DistanceMap;

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    unsigned Dist = 0;
    DistanceMap.clear();
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me; ) {
      MachineBasicBlock::iterator nmi = next(mi);
      const TargetInstrDesc &TID = mi->getDesc();
      bool FirstTied = true;

      DistanceMap.insert(std::make_pair(mi, ++Dist));
      for (unsigned si = 1, e = TID.getNumOperands(); si < e; ++si) {
        int ti = TID.getOperandConstraint(si, TOI::TIED_TO);
        if (ti == -1)
          continue;

        if (FirstTied) {
          ++NumTwoAddressInstrs;
          DOUT << '\t'; DEBUG(mi->print(*cerr.stream(), &TM));
        }

        FirstTied = false;

        assert(mi->getOperand(si).isReg() && mi->getOperand(si).getReg() &&
               mi->getOperand(si).isUse() && "two address instruction invalid");

        // If the two operands are the same we just remove the use
        // and mark the def as def&use, otherwise we have to insert a copy.
        if (mi->getOperand(ti).getReg() != mi->getOperand(si).getReg()) {
          // Rewrite:
          //     a = b op c
          // to:
          //     a = b
          //     a = a op c
          unsigned regA = mi->getOperand(ti).getReg();
          unsigned regB = mi->getOperand(si).getReg();

          assert(TargetRegisterInfo::isVirtualRegister(regA) &&
                 TargetRegisterInfo::isVirtualRegister(regB) &&
                 "cannot update physical register live information");

#ifndef NDEBUG
          // First, verify that we don't have a use of a in the instruction (a =
          // b + a for example) because our transformation will not work. This
          // should never occur because we are in SSA form.
          for (unsigned i = 0; i != mi->getNumOperands(); ++i)
            assert((int)i == ti ||
                   !mi->getOperand(i).isReg() ||
                   mi->getOperand(i).getReg() != regA);
#endif

          // If this instruction is not the killing user of B, see if we can
          // rearrange the code to make it so.  Making it the killing user will
          // allow us to coalesce A and B together, eliminating the copy we are
          // about to insert.
          if (!mi->killsRegister(regB)) {
            // If this instruction is commutative, check to see if C dies.  If
            // so, swap the B and C operands.  This makes the live ranges of A
            // and C joinable.
            // FIXME: This code also works for A := B op C instructions.
            if (TID.isCommutable() && mi->getNumOperands() >= 3) {
              assert(mi->getOperand(3-si).isReg() &&
                     "Not a proper commutative instruction!");
              unsigned regC = mi->getOperand(3-si).getReg();
              if (mi->killsRegister(regC)) {
                if (CommuteInstruction(mi, mbbi, regC, Dist, DistanceMap)) {
                  ++NumCommuted;
                  regB = regC;
                  goto InstructionRearranged;
                }
              }
            }

            // If this instruction is potentially convertible to a true
            // three-address instruction,
            if (TID.isConvertibleTo3Addr()) {
              // FIXME: This assumes there are no more operands which are tied
              // to another register.
#ifndef NDEBUG
              for (unsigned i = si + 1, e = TID.getNumOperands(); i < e; ++i)
                assert(TID.getOperandConstraint(i, TOI::TIED_TO) == -1);
#endif

              MachineInstr *NewMI = TII->convertToThreeAddress(mbbi, mi, LV);
              if (NewMI) {
                DOUT << "2addr: CONVERTING 2-ADDR: " << *mi;
                DOUT << "2addr:         TO 3-ADDR: " << *NewMI;
                bool Sunk = false;

                if (NewMI->findRegisterUseOperand(regB, false, TRI))
                  // FIXME: Temporary workaround. If the new instruction doesn't
                  // uses regB, convertToThreeAddress must have created more
                  // then one instruction.
                  Sunk = Sink3AddrInstruction(mbbi, NewMI, regB, mi);

                mbbi->erase(mi); // Nuke the old inst.

                if (!Sunk) {
                  DistanceMap.insert(std::make_pair(NewMI, Dist));
                  mi = NewMI;
                  nmi = next(mi);
                }

                ++NumConvertedTo3Addr;
                break; // Done with this instruction.
              }
            }
          }

          // If it's profitable to commute the instruction, do so.
          if (TID.isCommutable() && mi->getNumOperands() >= 3) {
            unsigned regC = mi->getOperand(3-si).getReg();
            if (isProfitableToCommute(regB, regC, mi, mbbi, Dist, DistanceMap))
              if (CommuteInstruction(mi, mbbi, regC, Dist, DistanceMap)) {
                ++NumAggrCommuted;
                ++NumCommuted;
                regB = regC;
              }
          }

        InstructionRearranged:
          const TargetRegisterClass* rc = MRI->getRegClass(regA);
          MachineInstr *DefMI = MRI->getVRegDef(regB);
          // If it's safe and profitable, remat the definition instead of
          // copying it.
          if (DefMI &&
              DefMI->getDesc().isAsCheapAsAMove() &&
              DefMI->isSafeToReMat(TII, regB) &&
              isProfitableToReMat(regB, rc, mi, DefMI, mbbi, Dist,DistanceMap)){
            DEBUG(cerr << "2addr: REMATTING : " << *DefMI << "\n");
            TII->reMaterialize(*mbbi, mi, regA, DefMI);
            ReMatRegs.set(regB);
            ++NumReMats;
          } else {
            TII->copyRegToReg(*mbbi, mi, regA, regB, rc, rc);
          }

          MachineBasicBlock::iterator prevMI = prior(mi);
          // Update DistanceMap.
          DistanceMap.insert(std::make_pair(prevMI, Dist));
          DistanceMap[mi] = ++Dist;

          // Update live variables for regB.
          if (LV) {
            LiveVariables::VarInfo& varInfoB = LV->getVarInfo(regB);

            // regB is used in this BB.
            varInfoB.UsedBlocks[mbbi->getNumber()] = true;

            if (LV->removeVirtualRegisterKilled(regB,  mi))
              LV->addVirtualRegisterKilled(regB, prevMI);

            if (LV->removeVirtualRegisterDead(regB, mi))
              LV->addVirtualRegisterDead(regB, prevMI);
          }

          DOUT << "\t\tprepend:\t"; DEBUG(prevMI->print(*cerr.stream(), &TM));
          
          // Replace all occurences of regB with regA.
          for (unsigned i = 0, e = mi->getNumOperands(); i != e; ++i) {
            if (mi->getOperand(i).isReg() &&
                mi->getOperand(i).getReg() == regB)
              mi->getOperand(i).setReg(regA);
          }
        }

        assert(mi->getOperand(ti).isDef() && mi->getOperand(si).isUse());
        mi->getOperand(ti).setReg(mi->getOperand(si).getReg());
        MadeChange = true;

        DOUT << "\t\trewrite to:\t"; DEBUG(mi->print(*cerr.stream(), &TM));
      }

      mi = nmi;
    }
  }

  // Some remat'ed instructions are dead.
  int VReg = ReMatRegs.find_first();
  while (VReg != -1) {
    if (MRI->use_empty(VReg)) {
      MachineInstr *DefMI = MRI->getVRegDef(VReg);
      DefMI->eraseFromParent();
    }
    VReg = ReMatRegs.find_next(VReg);
  }

  return MadeChange;
}
