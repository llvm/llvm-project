//===-- LiveVariables.cpp - Live Variable Analysis for Machine Code -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveVariable analysis pass.  For each machine
// instruction in the function, this pass calculates the set of registers that
// are immediately dead after the instruction (i.e., the instruction calculates
// the value, but it is never used) and the set of registers that are used by
// the instruction, but are never used after the instruction (i.e., they are
// killed).
//
// This class computes live variables using are sparse implementation based on
// the machine code SSA form.  This class computes live variable information for
// each virtual and _register allocatable_ physical register in a function.  It
// uses the dominance properties of SSA form to efficiently compute live
// variables for virtual registers, and assumes that physical registers are only
// live within a single basic block (allowing it to do a single local analysis
// to resolve physical register lifetimes in each basic block).  If a physical
// register is not register allocatable, it is not tracked.  This is useful for
// things like the stack pointer and condition codes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/alloca.h"
#include <algorithm>
using namespace llvm;

char LiveVariables::ID = 0;
static RegisterPass<LiveVariables> X("livevars", "Live Variable Analysis");


void LiveVariables::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredID(UnreachableMachineBlockElimID);
  AU.setPreservesAll();
}

void LiveVariables::VarInfo::dump() const {
  cerr << "  Alive in blocks: ";
  for (int i = AliveBlocks.find_first(); i != -1; i = AliveBlocks.find_next(i))
    cerr << i << ", ";
  cerr << "  Used in blocks: ";
  for (int i = UsedBlocks.find_first(); i != -1; i = UsedBlocks.find_next(i))
    cerr << i << ", ";
  cerr << "\n  Killed by:";
  if (Kills.empty())
    cerr << " No instructions.\n";
  else {
    for (unsigned i = 0, e = Kills.size(); i != e; ++i)
      cerr << "\n    #" << i << ": " << *Kills[i];
    cerr << "\n";
  }
}

/// getVarInfo - Get (possibly creating) a VarInfo object for the given vreg.
LiveVariables::VarInfo &LiveVariables::getVarInfo(unsigned RegIdx) {
  assert(TargetRegisterInfo::isVirtualRegister(RegIdx) &&
         "getVarInfo: not a virtual register!");
  RegIdx -= TargetRegisterInfo::FirstVirtualRegister;
  if (RegIdx >= VirtRegInfo.size()) {
    if (RegIdx >= 2*VirtRegInfo.size())
      VirtRegInfo.resize(RegIdx*2);
    else
      VirtRegInfo.resize(2*VirtRegInfo.size());
  }
  VarInfo &VI = VirtRegInfo[RegIdx];
  VI.AliveBlocks.resize(MF->getNumBlockIDs());
  VI.UsedBlocks.resize(MF->getNumBlockIDs());
  return VI;
}

void LiveVariables::MarkVirtRegAliveInBlock(VarInfo& VRInfo,
                                            MachineBasicBlock *DefBlock,
                                            MachineBasicBlock *MBB,
                                    std::vector<MachineBasicBlock*> &WorkList) {
  unsigned BBNum = MBB->getNumber();
  
  // Check to see if this basic block is one of the killing blocks.  If so,
  // remove it.
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    if (VRInfo.Kills[i]->getParent() == MBB) {
      VRInfo.Kills.erase(VRInfo.Kills.begin()+i);  // Erase entry
      break;
    }
  
  if (MBB == DefBlock) return;  // Terminate recursion

  if (VRInfo.AliveBlocks[BBNum])
    return;  // We already know the block is live

  // Mark the variable known alive in this bb
  VRInfo.AliveBlocks[BBNum] = true;

  for (MachineBasicBlock::const_pred_reverse_iterator PI = MBB->pred_rbegin(),
         E = MBB->pred_rend(); PI != E; ++PI)
    WorkList.push_back(*PI);
}

void LiveVariables::MarkVirtRegAliveInBlock(VarInfo &VRInfo,
                                            MachineBasicBlock *DefBlock,
                                            MachineBasicBlock *MBB) {
  std::vector<MachineBasicBlock*> WorkList;
  MarkVirtRegAliveInBlock(VRInfo, DefBlock, MBB, WorkList);

  while (!WorkList.empty()) {
    MachineBasicBlock *Pred = WorkList.back();
    WorkList.pop_back();
    MarkVirtRegAliveInBlock(VRInfo, DefBlock, Pred, WorkList);
  }
}

void LiveVariables::HandleVirtRegUse(unsigned reg, MachineBasicBlock *MBB,
                                     MachineInstr *MI) {
  assert(MRI->getVRegDef(reg) && "Register use before def!");

  unsigned BBNum = MBB->getNumber();

  VarInfo& VRInfo = getVarInfo(reg);
  VRInfo.UsedBlocks[BBNum] = true;
  VRInfo.NumUses++;

  // Check to see if this basic block is already a kill block.
  if (!VRInfo.Kills.empty() && VRInfo.Kills.back()->getParent() == MBB) {
    // Yes, this register is killed in this basic block already. Increase the
    // live range by updating the kill instruction.
    VRInfo.Kills.back() = MI;
    return;
  }

#ifndef NDEBUG
  for (unsigned i = 0, e = VRInfo.Kills.size(); i != e; ++i)
    assert(VRInfo.Kills[i]->getParent() != MBB && "entry should be at end!");
#endif

  // This situation can occur:
  //
  //     ,------.
  //     |      |
  //     |      v
  //     |   t2 = phi ... t1 ...
  //     |      |
  //     |      v
  //     |   t1 = ...
  //     |  ... = ... t1 ...
  //     |      |
  //     `------'
  //
  // where there is a use in a PHI node that's a predecessor to the defining
  // block. We don't want to mark all predecessors as having the value "alive"
  // in this case.
  if (MBB == MRI->getVRegDef(reg)->getParent()) return;

  // Add a new kill entry for this basic block. If this virtual register is
  // already marked as alive in this basic block, that means it is alive in at
  // least one of the successor blocks, it's not a kill.
  if (!VRInfo.AliveBlocks[BBNum])
    VRInfo.Kills.push_back(MI);

  // Update all dominating blocks to mark them as "known live".
  for (MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(),
         E = MBB->pred_end(); PI != E; ++PI)
    MarkVirtRegAliveInBlock(VRInfo, MRI->getVRegDef(reg)->getParent(), *PI);
}

void LiveVariables::HandleVirtRegDef(unsigned Reg, MachineInstr *MI) {
  VarInfo &VRInfo = getVarInfo(Reg);

  if (VRInfo.AliveBlocks.none())
    // If vr is not alive in any block, then defaults to dead.
    VRInfo.Kills.push_back(MI);
}

/// FindLastPartialDef - Return the last partial def of the specified register.
/// Also returns the sub-register that's defined.
MachineInstr *LiveVariables::FindLastPartialDef(unsigned Reg,
                                                unsigned &PartDefReg) {
  unsigned LastDefReg = 0;
  unsigned LastDefDist = 0;
  MachineInstr *LastDef = NULL;
  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs) {
    MachineInstr *Def = PhysRegDef[SubReg];
    if (!Def)
      continue;
    unsigned Dist = DistanceMap[Def];
    if (Dist > LastDefDist) {
      LastDefReg  = SubReg;
      LastDef     = Def;
      LastDefDist = Dist;
    }
  }
  PartDefReg = LastDefReg;
  return LastDef;
}

/// HandlePhysRegUse - Turn previous partial def's into read/mod/writes. Add
/// implicit defs to a machine instruction if there was an earlier def of its
/// super-register.
void LiveVariables::HandlePhysRegUse(unsigned Reg, MachineInstr *MI) {
  // If there was a previous use or a "full" def all is well.
  if (!PhysRegDef[Reg] && !PhysRegUse[Reg]) {
    // Otherwise, the last sub-register def implicitly defines this register.
    // e.g.
    // AH =
    // AL = ... <imp-def EAX>, <imp-kill AH>
    //    = AH
    // ...
    //    = EAX
    // All of the sub-registers must have been defined before the use of Reg!
    unsigned PartDefReg = 0;
    MachineInstr *LastPartialDef = FindLastPartialDef(Reg, PartDefReg);
    // If LastPartialDef is NULL, it must be using a livein register.
    if (LastPartialDef) {
      LastPartialDef->addOperand(MachineOperand::CreateReg(Reg, true/*IsDef*/,
                                                           true/*IsImp*/));
      PhysRegDef[Reg] = LastPartialDef;
      SmallSet<unsigned, 8> Processed;
      for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
           unsigned SubReg = *SubRegs; ++SubRegs) {
        if (Processed.count(SubReg))
          continue;
        if (SubReg == PartDefReg || TRI->isSubRegister(PartDefReg, SubReg))
          continue;
        // This part of Reg was defined before the last partial def. It's killed
        // here.
        LastPartialDef->addOperand(MachineOperand::CreateReg(SubReg,
                                                             false/*IsDef*/,
                                                             true/*IsImp*/));
        PhysRegDef[SubReg] = LastPartialDef;
        for (const unsigned *SS = TRI->getSubRegisters(SubReg); *SS; ++SS)
          Processed.insert(*SS);
      }
    }
  }

  // There was an earlier def of a super-register. Add implicit def to that MI.
  //
  //   A: EAX = ...
  //   B: ... = AX
  //
  // Add implicit def to A if there isn't a use of AX (or EAX) before B.
  if (!PhysRegUse[Reg]) {
    MachineInstr *Def = PhysRegDef[Reg];
    if (Def && !Def->modifiesRegister(Reg))
      Def->addOperand(MachineOperand::CreateReg(Reg,
                                                true  /*IsDef*/,
                                                true  /*IsImp*/));
  }
  
  // Remember this use.
  PhysRegUse[Reg]  = MI;
  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs)
    PhysRegUse[SubReg] =  MI;
}

/// hasRegisterUseBelow - Return true if the specified register is used after
/// the current instruction and before it's next definition.
bool LiveVariables::hasRegisterUseBelow(unsigned Reg,
                                        MachineBasicBlock::iterator I,
                                        MachineBasicBlock *MBB) {
  if (I == MBB->end())
    return false;

  // First find out if there are any uses / defs below.
  bool hasDistInfo = true;
  unsigned CurDist = DistanceMap[I];
  SmallVector<MachineInstr*, 4> Uses;
  SmallVector<MachineInstr*, 4> Defs;
  for (MachineRegisterInfo::reg_iterator RI = MRI->reg_begin(Reg),
         RE = MRI->reg_end(); RI != RE; ++RI) {
    MachineOperand &UDO = RI.getOperand();
    MachineInstr *UDMI = &*RI;
    if (UDMI->getParent() != MBB)
      continue;
    DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(UDMI);
    bool isBelow = false;
    if (DI == DistanceMap.end()) {
      // Must be below if it hasn't been assigned a distance yet.
      isBelow = true;
      hasDistInfo = false;
    } else if (DI->second > CurDist)
      isBelow = true;
    if (isBelow) {
      if (UDO.isUse())
        Uses.push_back(UDMI);
      if (UDO.isDef())
        Defs.push_back(UDMI);
    }
  }

  if (Uses.empty())
    // No uses below.
    return false;
  else if (!Uses.empty() && Defs.empty())
    // There are uses below but no defs below.
    return true;
  // There are both uses and defs below. We need to know which comes first.
  if (!hasDistInfo) {
    // Complete DistanceMap for this MBB. This information is computed only
    // once per MBB.
    ++I;
    ++CurDist;
    for (MachineBasicBlock::iterator E = MBB->end(); I != E; ++I, ++CurDist)
      DistanceMap.insert(std::make_pair(I, CurDist));
  }

  unsigned EarliestUse = DistanceMap[Uses[0]];
  for (unsigned i = 1, e = Uses.size(); i != e; ++i) {
    unsigned Dist = DistanceMap[Uses[i]];
    if (Dist < EarliestUse)
      EarliestUse = Dist;
  }
  for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
    unsigned Dist = DistanceMap[Defs[i]];
    if (Dist < EarliestUse)
      // The register is defined before its first use below.
      return false;
  }
  return true;
}

bool LiveVariables::HandlePhysRegKill(unsigned Reg, MachineInstr *MI) {
  if (!PhysRegUse[Reg] && !PhysRegDef[Reg])
    return false;

  MachineInstr *LastRefOrPartRef = PhysRegUse[Reg]
    ? PhysRegUse[Reg] : PhysRegDef[Reg];
  unsigned LastRefOrPartRefDist = DistanceMap[LastRefOrPartRef];
  // The whole register is used.
  // AL =
  // AH =
  //
  //    = AX
  //    = AL, AX<imp-use, kill>
  // AX =
  //
  // Or whole register is defined, but not used at all.
  // AX<dead> =
  // ...
  // AX =
  //
  // Or whole register is defined, but only partly used.
  // AX<dead> = AL<imp-def>
  //    = AL<kill>
  // AX = 
  SmallSet<unsigned, 8> PartUses;
  for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
       unsigned SubReg = *SubRegs; ++SubRegs) {
    if (MachineInstr *Use = PhysRegUse[SubReg]) {
      PartUses.insert(SubReg);
      for (const unsigned *SS = TRI->getSubRegisters(SubReg); *SS; ++SS)
        PartUses.insert(*SS);
      unsigned Dist = DistanceMap[Use];
      if (Dist > LastRefOrPartRefDist) {
        LastRefOrPartRefDist = Dist;
        LastRefOrPartRef = Use;
      }
    }
  }

  if (LastRefOrPartRef == PhysRegDef[Reg] && LastRefOrPartRef != MI)
    // If the last reference is the last def, then it's not used at all.
    // That is, unless we are currently processing the last reference itself.
    LastRefOrPartRef->addRegisterDead(Reg, TRI, true);

  /* Partial uses. Mark register def dead and add implicit def of
     sub-registers which are used.
    FIXME: LiveIntervalAnalysis can't handle this yet!
    EAX<dead>  = op  AL<imp-def>
    That is, EAX def is dead but AL def extends pass it.
    Enable this after live interval analysis is fixed to improve codegen!
  else if (!PhysRegUse[Reg]) {
    PhysRegDef[Reg]->addRegisterDead(Reg, TRI, true);
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs) {
      if (PartUses.count(SubReg)) {
        PhysRegDef[Reg]->addOperand(MachineOperand::CreateReg(SubReg,
                                                              true, true));
        LastRefOrPartRef->addRegisterKilled(SubReg, TRI, true);
        for (const unsigned *SS = TRI->getSubRegisters(SubReg); *SS; ++SS)
          PartUses.erase(*SS);
      }
    }
  } */
  else
    LastRefOrPartRef->addRegisterKilled(Reg, TRI, true);
  return true;
}

void LiveVariables::HandlePhysRegDef(unsigned Reg, MachineInstr *MI) {
  // What parts of the register are previously defined?
  SmallSet<unsigned, 32> Live;
  if (PhysRegDef[Reg] || PhysRegUse[Reg]) {
    Live.insert(Reg);
    for (const unsigned *SS = TRI->getSubRegisters(Reg); *SS; ++SS)
      Live.insert(*SS);
  } else {
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs) {
      // If a register isn't itself defined, but all parts that make up of it
      // are defined, then consider it also defined.
      // e.g.
      // AL =
      // AH =
      //    = AX
      if (PhysRegDef[SubReg] || PhysRegUse[SubReg]) {
        Live.insert(SubReg);
        for (const unsigned *SS = TRI->getSubRegisters(SubReg); *SS; ++SS)
          Live.insert(*SS);
      }
    }
  }

  // Start from the largest piece, find the last time any part of the register
  // is referenced.
  if (!HandlePhysRegKill(Reg, MI)) {
    // Only some of the sub-registers are used.
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs) {
      if (!Live.count(SubReg))
        // Skip if this sub-register isn't defined.
        continue;
      if (HandlePhysRegKill(SubReg, MI)) {
        Live.erase(SubReg);
        for (const unsigned *SS = TRI->getSubRegisters(SubReg); *SS; ++SS)
          Live.erase(*SS);
      }
    }
    assert(Live.empty() && "Not all defined registers are killed / dead?");
  }

  if (MI) {
    // Does this extend the live range of a super-register?
    SmallSet<unsigned, 8> Processed;
    for (const unsigned *SuperRegs = TRI->getSuperRegisters(Reg);
         unsigned SuperReg = *SuperRegs; ++SuperRegs) {
      if (Processed.count(SuperReg))
        continue;
      MachineInstr *LastRef = PhysRegUse[SuperReg]
        ? PhysRegUse[SuperReg] : PhysRegDef[SuperReg];
      if (LastRef && LastRef != MI) {
        // The larger register is previously defined. Now a smaller part is
        // being re-defined. Treat it as read/mod/write if there are uses
        // below.
        // EAX =
        // AX  =        EAX<imp-use,kill>, EAX<imp-def>
        // ...
        ///    =  EAX
        if (hasRegisterUseBelow(SuperReg, MI, MI->getParent())) {
          MI->addOperand(MachineOperand::CreateReg(SuperReg, false/*IsDef*/,
                                                   true/*IsImp*/,true/*IsKill*/));
          MI->addOperand(MachineOperand::CreateReg(SuperReg, true/*IsDef*/,
                                                   true/*IsImp*/));
          PhysRegDef[SuperReg]  = MI;
          PhysRegUse[SuperReg]  = NULL;
          Processed.insert(SuperReg);
          for (const unsigned *SS = TRI->getSubRegisters(SuperReg); *SS; ++SS) {
            PhysRegDef[*SS]  = MI;
            PhysRegUse[*SS]  = NULL;
            Processed.insert(*SS);
          }
        } else {
          // Otherwise, the super register is killed.
          if (HandlePhysRegKill(SuperReg, MI)) {
            PhysRegDef[SuperReg]  = NULL;
            PhysRegUse[SuperReg]  = NULL;
            for (const unsigned *SS = TRI->getSubRegisters(SuperReg); *SS; ++SS) {
              PhysRegDef[*SS]  = NULL;
              PhysRegUse[*SS]  = NULL;
              Processed.insert(*SS);
            }
          }
        }
      }
    }

    // Remember this def.
    PhysRegDef[Reg]  = MI;
    PhysRegUse[Reg]  = NULL;
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         unsigned SubReg = *SubRegs; ++SubRegs) {
      PhysRegDef[SubReg]  = MI;
      PhysRegUse[SubReg]  = NULL;
    }
  }
}

bool LiveVariables::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &mf.getRegInfo();
  TRI = MF->getTarget().getRegisterInfo();

  ReservedRegisters = TRI->getReservedRegs(mf);

  unsigned NumRegs = TRI->getNumRegs();
  PhysRegDef  = new MachineInstr*[NumRegs];
  PhysRegUse  = new MachineInstr*[NumRegs];
  PHIVarInfo = new SmallVector<unsigned, 4>[MF->getNumBlockIDs()];
  std::fill(PhysRegDef,  PhysRegDef  + NumRegs, (MachineInstr*)0);
  std::fill(PhysRegUse,  PhysRegUse  + NumRegs, (MachineInstr*)0);

  /// Get some space for a respectable number of registers.
  VirtRegInfo.resize(64);

  analyzePHINodes(mf);

  // Calculate live variable information in depth first order on the CFG of the
  // function.  This guarantees that we will see the definition of a virtual
  // register before its uses due to dominance properties of SSA (except for PHI
  // nodes, which are treated as a special case).
  MachineBasicBlock *Entry = MF->begin();
  SmallPtrSet<MachineBasicBlock*,16> Visited;

  for (df_ext_iterator<MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*,16> >
         DFI = df_ext_begin(Entry, Visited), E = df_ext_end(Entry, Visited);
       DFI != E; ++DFI) {
    MachineBasicBlock *MBB = *DFI;

    // Mark live-in registers as live-in.
    for (MachineBasicBlock::const_livein_iterator II = MBB->livein_begin(),
           EE = MBB->livein_end(); II != EE; ++II) {
      assert(TargetRegisterInfo::isPhysicalRegister(*II) &&
             "Cannot have a live-in virtual register!");
      HandlePhysRegDef(*II, 0);
    }

    // Loop over all of the instructions, processing them.
    DistanceMap.clear();
    unsigned Dist = 0;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      MachineInstr *MI = I;
      DistanceMap.insert(std::make_pair(MI, Dist++));

      // Process all of the operands of the instruction...
      unsigned NumOperandsToProcess = MI->getNumOperands();

      // Unless it is a PHI node.  In this case, ONLY process the DEF, not any
      // of the uses.  They will be handled in other basic blocks.
      if (MI->getOpcode() == TargetInstrInfo::PHI)
        NumOperandsToProcess = 1;

      SmallVector<unsigned, 4> UseRegs;
      SmallVector<unsigned, 4> DefRegs;
      for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || MO.getReg() == 0)
          continue;
        unsigned MOReg = MO.getReg();
        if (MO.isUse())
          UseRegs.push_back(MOReg);
        if (MO.isDef())
          DefRegs.push_back(MOReg);
      }

      // Process all uses.
      for (unsigned i = 0, e = UseRegs.size(); i != e; ++i) {
        unsigned MOReg = UseRegs[i];
        if (TargetRegisterInfo::isVirtualRegister(MOReg))
          HandleVirtRegUse(MOReg, MBB, MI);
        else if (!ReservedRegisters[MOReg])
          HandlePhysRegUse(MOReg, MI);
      }

      // Process all defs.
      for (unsigned i = 0, e = DefRegs.size(); i != e; ++i) {
        unsigned MOReg = DefRegs[i];
        if (TargetRegisterInfo::isVirtualRegister(MOReg))
          HandleVirtRegDef(MOReg, MI);
        else if (!ReservedRegisters[MOReg])
          HandlePhysRegDef(MOReg, MI);
      }
    }

    // Handle any virtual assignments from PHI nodes which might be at the
    // bottom of this basic block.  We check all of our successor blocks to see
    // if they have PHI nodes, and if so, we simulate an assignment at the end
    // of the current block.
    if (!PHIVarInfo[MBB->getNumber()].empty()) {
      SmallVector<unsigned, 4>& VarInfoVec = PHIVarInfo[MBB->getNumber()];

      for (SmallVector<unsigned, 4>::iterator I = VarInfoVec.begin(),
             E = VarInfoVec.end(); I != E; ++I)
        // Mark it alive only in the block we are representing.
        MarkVirtRegAliveInBlock(getVarInfo(*I),MRI->getVRegDef(*I)->getParent(),
                                MBB);
    }

    // Finally, if the last instruction in the block is a return, make sure to
    // mark it as using all of the live-out values in the function.
    if (!MBB->empty() && MBB->back().getDesc().isReturn()) {
      MachineInstr *Ret = &MBB->back();

      for (MachineRegisterInfo::liveout_iterator
           I = MF->getRegInfo().liveout_begin(),
           E = MF->getRegInfo().liveout_end(); I != E; ++I) {
        assert(TargetRegisterInfo::isPhysicalRegister(*I) &&
               "Cannot have a live-out virtual register!");
        HandlePhysRegUse(*I, Ret);

        // Add live-out registers as implicit uses.
        if (!Ret->readsRegister(*I))
          Ret->addOperand(MachineOperand::CreateReg(*I, false, true));
      }
    }

    // Loop over PhysRegDef / PhysRegUse, killing any registers that are
    // available at the end of the basic block.
    for (unsigned i = 0; i != NumRegs; ++i)
      if (PhysRegDef[i] || PhysRegUse[i])
        HandlePhysRegDef(i, 0);

    std::fill(PhysRegDef,  PhysRegDef  + NumRegs, (MachineInstr*)0);
    std::fill(PhysRegUse,  PhysRegUse  + NumRegs, (MachineInstr*)0);
  }

  // Convert and transfer the dead / killed information we have gathered into
  // VirtRegInfo onto MI's.
  for (unsigned i = 0, e1 = VirtRegInfo.size(); i != e1; ++i)
    for (unsigned j = 0, e2 = VirtRegInfo[i].Kills.size(); j != e2; ++j)
      if (VirtRegInfo[i].Kills[j] ==
          MRI->getVRegDef(i + TargetRegisterInfo::FirstVirtualRegister))
        VirtRegInfo[i]
          .Kills[j]->addRegisterDead(i +
                                     TargetRegisterInfo::FirstVirtualRegister,
                                     TRI);
      else
        VirtRegInfo[i]
          .Kills[j]->addRegisterKilled(i +
                                       TargetRegisterInfo::FirstVirtualRegister,
                                       TRI);

  // Check to make sure there are no unreachable blocks in the MC CFG for the
  // function.  If so, it is due to a bug in the instruction selector or some
  // other part of the code generator if this happens.
#ifndef NDEBUG
  for(MachineFunction::iterator i = MF->begin(), e = MF->end(); i != e; ++i)
    assert(Visited.count(&*i) != 0 && "unreachable basic block found");
#endif

  delete[] PhysRegDef;
  delete[] PhysRegUse;
  delete[] PHIVarInfo;

  return false;
}

/// replaceKillInstruction - Update register kill info by replacing a kill
/// instruction with a new one.
void LiveVariables::replaceKillInstruction(unsigned Reg, MachineInstr *OldMI,
                                           MachineInstr *NewMI) {
  VarInfo &VI = getVarInfo(Reg);
  std::replace(VI.Kills.begin(), VI.Kills.end(), OldMI, NewMI);
}

/// removeVirtualRegistersKilled - Remove all killed info for the specified
/// instruction.
void LiveVariables::removeVirtualRegistersKilled(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isKill()) {
      MO.setIsKill(false);
      unsigned Reg = MO.getReg();
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        bool removed = getVarInfo(Reg).removeKill(MI);
        assert(removed && "kill not in register's VarInfo?");
        removed = true;
      }
    }
  }
}

/// analyzePHINodes - Gather information about the PHI nodes in here. In
/// particular, we want to map the variable information of a virtual register
/// which is used in a PHI node. We map that to the BB the vreg is coming from.
///
void LiveVariables::analyzePHINodes(const MachineFunction& Fn) {
  for (MachineFunction::const_iterator I = Fn.begin(), E = Fn.end();
       I != E; ++I)
    for (MachineBasicBlock::const_iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->getOpcode() == TargetInstrInfo::PHI; ++BBI)
      for (unsigned i = 1, e = BBI->getNumOperands(); i != e; i += 2)
        PHIVarInfo[BBI->getOperand(i + 1).getMBB()->getNumber()]
          .push_back(BBI->getOperand(i).getReg());
}
