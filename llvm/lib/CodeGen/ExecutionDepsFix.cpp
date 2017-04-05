//===- ExecutionDepsFix.cpp - Fix execution dependecy issues ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExecutionDepsFix.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "execution-deps-fix"

/// Translate TRI register number to a list of indices into our smaller tables
/// of interesting registers.
iterator_range<SmallVectorImpl<int>::const_iterator>
ExecutionDepsFix::regIndices(unsigned Reg) const {
  assert(Reg < AliasMap.size() && "Invalid register");
  const auto &Entry = AliasMap[Reg];
  return make_range(Entry.begin(), Entry.end());
}

DomainValue *ExecutionDepsFix::alloc(int domain) {
  DomainValue *dv = Avail.empty() ?
                      new(Allocator.Allocate()) DomainValue :
                      Avail.pop_back_val();
  if (domain >= 0)
    dv->addDomain(domain);
  assert(dv->Refs == 0 && "Reference count wasn't cleared");
  assert(!dv->Next && "Chained DomainValue shouldn't have been recycled");
  return dv;
}

/// Release a reference to DV.  When the last reference is released,
/// collapse if needed.
void ExecutionDepsFix::release(DomainValue *DV) {
  while (DV) {
    assert(DV->Refs && "Bad DomainValue");
    if (--DV->Refs)
      return;

    // There are no more DV references. Collapse any contained instructions.
    if (DV->AvailableDomains && !DV->isCollapsed())
      collapse(DV, DV->getFirstDomain());

    DomainValue *Next = DV->Next;
    DV->clear();
    Avail.push_back(DV);
    // Also release the next DomainValue in the chain.
    DV = Next;
  }
}

/// Follow the chain of dead DomainValues until a live DomainValue is reached.
/// Update the referenced pointer when necessary.
DomainValue *ExecutionDepsFix::resolve(DomainValue *&DVRef) {
  DomainValue *DV = DVRef;
  if (!DV || !DV->Next)
    return DV;

  // DV has a chain. Find the end.
  do DV = DV->Next;
  while (DV->Next);

  // Update DVRef to point to DV.
  retain(DV);
  release(DVRef);
  DVRef = DV;
  return DV;
}

/// Set LiveRegs[rx] = dv, updating reference counts.
void ExecutionDepsFix::setLiveReg(int rx, DomainValue *dv) {
  assert(unsigned(rx) < NumRegs && "Invalid index");
  assert(LiveRegs && "Must enter basic block first.");

  if (LiveRegs[rx].Value == dv)
    return;
  if (LiveRegs[rx].Value)
    release(LiveRegs[rx].Value);
  LiveRegs[rx].Value = retain(dv);
}

// Kill register rx, recycle or collapse any DomainValue.
void ExecutionDepsFix::kill(int rx) {
  assert(unsigned(rx) < NumRegs && "Invalid index");
  assert(LiveRegs && "Must enter basic block first.");
  if (!LiveRegs[rx].Value)
    return;

  release(LiveRegs[rx].Value);
  LiveRegs[rx].Value = nullptr;
}

/// Force register rx into domain.
void ExecutionDepsFix::force(int rx, unsigned domain) {
  assert(unsigned(rx) < NumRegs && "Invalid index");
  assert(LiveRegs && "Must enter basic block first.");
  if (DomainValue *dv = LiveRegs[rx].Value) {
    if (dv->isCollapsed())
      dv->addDomain(domain);
    else if (dv->hasDomain(domain))
      collapse(dv, domain);
    else {
      // This is an incompatible open DomainValue. Collapse it to whatever and
      // force the new value into domain. This costs a domain crossing.
      collapse(dv, dv->getFirstDomain());
      assert(LiveRegs[rx].Value && "Not live after collapse?");
      LiveRegs[rx].Value->addDomain(domain);
    }
  } else {
    // Set up basic collapsed DomainValue.
    setLiveReg(rx, alloc(domain));
  }
}

/// Collapse open DomainValue into given domain. If there are multiple
/// registers using dv, they each get a unique collapsed DomainValue.
void ExecutionDepsFix::collapse(DomainValue *dv, unsigned domain) {
  assert(dv->hasDomain(domain) && "Cannot collapse");

  // Collapse all the instructions.
  while (!dv->Instrs.empty())
    TII->setExecutionDomain(*dv->Instrs.pop_back_val(), domain);
  dv->setSingleDomain(domain);

  // If there are multiple users, give them new, unique DomainValues.
  if (LiveRegs && dv->Refs > 1)
    for (unsigned rx = 0; rx != NumRegs; ++rx)
      if (LiveRegs[rx].Value == dv)
        setLiveReg(rx, alloc(domain));
}

/// All instructions and registers in B are moved to A, and B is released.
bool ExecutionDepsFix::merge(DomainValue *A, DomainValue *B) {
  assert(!A->isCollapsed() && "Cannot merge into collapsed");
  assert(!B->isCollapsed() && "Cannot merge from collapsed");
  if (A == B)
    return true;
  // Restrict to the domains that A and B have in common.
  unsigned common = A->getCommonDomains(B->AvailableDomains);
  if (!common)
    return false;
  A->AvailableDomains = common;
  A->Instrs.append(B->Instrs.begin(), B->Instrs.end());

  // Clear the old DomainValue so we won't try to swizzle instructions twice.
  B->clear();
  // All uses of B are referred to A.
  B->Next = retain(A);

  for (unsigned rx = 0; rx != NumRegs; ++rx) {
    assert(LiveRegs && "no space allocated for live registers");
    if (LiveRegs[rx].Value == B)
      setLiveReg(rx, A);
  }
  return true;
}

/// Set up LiveRegs by merging predecessor live-out values.
void ExecutionDepsFix::enterBasicBlock(MachineBasicBlock *MBB) {
  // Reset instruction counter in each basic block.
  CurInstr = 0;

  // Set up UndefReads to track undefined register reads.
  UndefReads.clear();
  LiveRegSet.clear();

  // Set up LiveRegs to represent registers entering MBB.
  if (!LiveRegs)
    LiveRegs = new LiveReg[NumRegs];

  // Default values are 'nothing happened a long time ago'.
  for (unsigned rx = 0; rx != NumRegs; ++rx) {
    LiveRegs[rx].Value = nullptr;
    LiveRegs[rx].Def = -(1 << 20);
  }

  // This is the entry block.
  if (MBB->pred_empty()) {
    for (const auto &LI : MBB->liveins()) {
      for (int rx : regIndices(LI.PhysReg)) {
        // Treat function live-ins as if they were defined just before the first
        // instruction.  Usually, function arguments are set up immediately
        // before the call.
        LiveRegs[rx].Def = -1;
      }
    }
    DEBUG(dbgs() << "BB#" << MBB->getNumber() << ": entry\n");
    return;
  }

  // Try to coalesce live-out registers from predecessors.
  for (MachineBasicBlock::const_pred_iterator pi = MBB->pred_begin(),
       pe = MBB->pred_end(); pi != pe; ++pi) {
    auto fi = MBBInfos.find(*pi);
    assert(fi != MBBInfos.end() &&
           "Should have pre-allocated MBBInfos for all MBBs");
    LiveReg *Incoming = fi->second.OutRegs;
    // Incoming is null if this is a backedge from a BB
    // we haven't processed yet
    if (Incoming == nullptr) {
      continue;
    }

    for (unsigned rx = 0; rx != NumRegs; ++rx) {
      // Use the most recent predecessor def for each register.
      LiveRegs[rx].Def = std::max(LiveRegs[rx].Def, Incoming[rx].Def);

      DomainValue *pdv = resolve(Incoming[rx].Value);
      if (!pdv)
        continue;
      if (!LiveRegs[rx].Value) {
        setLiveReg(rx, pdv);
        continue;
      }

      // We have a live DomainValue from more than one predecessor.
      if (LiveRegs[rx].Value->isCollapsed()) {
        // We are already collapsed, but predecessor is not. Force it.
        unsigned Domain = LiveRegs[rx].Value->getFirstDomain();
        if (!pdv->isCollapsed() && pdv->hasDomain(Domain))
          collapse(pdv, Domain);
        continue;
      }

      // Currently open, merge in predecessor.
      if (!pdv->isCollapsed())
        merge(LiveRegs[rx].Value, pdv);
      else
        force(rx, pdv->getFirstDomain());
    }
  }
  DEBUG(
      dbgs() << "BB#" << MBB->getNumber()
             << (!isBlockDone(MBB) ? ": incomplete\n" : ": all preds known\n"));
}

void ExecutionDepsFix::leaveBasicBlock(MachineBasicBlock *MBB) {
  assert(LiveRegs && "Must enter basic block first.");
  LiveReg *OldOutRegs = MBBInfos[MBB].OutRegs;
  // Save register clearances at end of MBB - used by enterBasicBlock().
  MBBInfos[MBB].OutRegs = LiveRegs;

  // While processing the basic block, we kept `Def` relative to the start
  // of the basic block for convenience. However, future use of this information
  // only cares about the clearance from the end of the block, so adjust
  // everything to be relative to the end of the basic block.
  for (unsigned i = 0, e = NumRegs; i != e; ++i)
    LiveRegs[i].Def -= CurInstr;
  if (OldOutRegs) {
    // This must be the second pass.
    // Release all the DomainValues instead of keeping them.
    for (unsigned i = 0, e = NumRegs; i != e; ++i)
      release(OldOutRegs[i].Value);
    delete[] OldOutRegs;
  }
  LiveRegs = nullptr;
}

bool ExecutionDepsFix::visitInstr(MachineInstr *MI) {
  // Update instructions with explicit execution domains.
  std::pair<uint16_t, uint16_t> DomP = TII->getExecutionDomain(*MI);
  if (DomP.first) {
    if (DomP.second)
      visitSoftInstr(MI, DomP.second);
    else
      visitHardInstr(MI, DomP.first);
  }

  return !DomP.first;
}

/// \brief Helps avoid false dependencies on undef registers by updating the
/// machine instructions' undef operand to use a register that the instruction
/// is truly dependent on, or use a register with clearance higher than Pref.
/// Returns true if it was able to find a true dependency, thus not requiring
/// a dependency breaking instruction regardless of clearance.
bool ExecutionDepsFix::pickBestRegisterForUndef(MachineInstr *MI,
                                                unsigned OpIdx, unsigned Pref) {
  MachineOperand &MO = MI->getOperand(OpIdx);
  assert(MO.isUndef() && "Expected undef machine operand");

  unsigned OriginalReg = MO.getReg();

  // Update only undef operands that are mapped to one register.
  if (AliasMap[OriginalReg].size() != 1)
    return false;

  // Get the undef operand's register class
  const TargetRegisterClass *OpRC =
      TII->getRegClass(MI->getDesc(), OpIdx, TRI, *MF);

  // If the instruction has a true dependency, we can hide the false depdency
  // behind it.
  for (MachineOperand &CurrMO : MI->operands()) {
    if (!CurrMO.isReg() || CurrMO.isDef() || CurrMO.isUndef() ||
        !OpRC->contains(CurrMO.getReg()))
      continue;
    // We found a true dependency - replace the undef register with the true
    // dependency.
    MO.setReg(CurrMO.getReg());
    return true;
  }

  // Go over all registers in the register class and find the register with
  // max clearance or clearance higher than Pref.
  unsigned MaxClearance = 0;
  unsigned MaxClearanceReg = OriginalReg;
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(OpRC);
  for (auto Reg : Order) {
    assert(AliasMap[Reg].size() == 1 &&
           "Reg is expected to be mapped to a single index");
    int RCrx = *regIndices(Reg).begin();
    unsigned Clearance = CurInstr - LiveRegs[RCrx].Def;
    if (Clearance <= MaxClearance)
      continue;
    MaxClearance = Clearance;
    MaxClearanceReg = Reg;

    if (MaxClearance > Pref)
      break;
  }

  // Update the operand if we found a register with better clearance.
  if (MaxClearanceReg != OriginalReg)
    MO.setReg(MaxClearanceReg);

  return false;
}

/// \brief Return true to if it makes sense to break dependence on a partial def
/// or undef use.
bool ExecutionDepsFix::shouldBreakDependence(MachineInstr *MI, unsigned OpIdx,
                                             unsigned Pref) {
  unsigned reg = MI->getOperand(OpIdx).getReg();
  for (int rx : regIndices(reg)) {
    unsigned Clearance = CurInstr - LiveRegs[rx].Def;
    DEBUG(dbgs() << "Clearance: " << Clearance << ", want " << Pref);

    if (Pref > Clearance) {
      DEBUG(dbgs() << ": Break dependency.\n");
      continue;
    }
    DEBUG(dbgs() << ": OK .\n");
    return false;
  }
  return true;
}

// Update def-ages for registers defined by MI.
// If Kill is set, also kill off DomainValues clobbered by the defs.
//
// Also break dependencies on partial defs and undef uses.
void ExecutionDepsFix::processDefs(MachineInstr *MI, bool breakDependency,
                                   bool Kill) {
  assert(!MI->isDebugValue() && "Won't process debug values");

  // Break dependence on undef uses. Do this before updating LiveRegs below.
  unsigned OpNum;
  if (breakDependency) {
    unsigned Pref = TII->getUndefRegClearance(*MI, OpNum, TRI);
    if (Pref) {
      bool HadTrueDependency = pickBestRegisterForUndef(MI, OpNum, Pref);
      // We don't need to bother trying to break a dependency if this
      // instruction has a true dependency on that register through another
      // operand - we'll have to wait for it to be available regardless.
      if (!HadTrueDependency && shouldBreakDependence(MI, OpNum, Pref))
        UndefReads.push_back(std::make_pair(MI, OpNum));
    }
  }
  const MCInstrDesc &MCID = MI->getDesc();
  for (unsigned i = 0,
         e = MI->isVariadic() ? MI->getNumOperands() : MCID.getNumDefs();
         i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    if (MO.isUse())
      continue;
    for (int rx : regIndices(MO.getReg())) {
      // This instruction explicitly defines rx.
      DEBUG(dbgs() << TRI->getName(RC->getRegister(rx)) << ":\t" << CurInstr
                   << '\t' << *MI);

      if (breakDependency) {
        // Check clearance before partial register updates.
        // Call breakDependence before setting LiveRegs[rx].Def.
        unsigned Pref = TII->getPartialRegUpdateClearance(*MI, i, TRI);
        if (Pref && shouldBreakDependence(MI, i, Pref))
          TII->breakPartialRegDependency(*MI, i, TRI);
      }

      // How many instructions since rx was last written?
      LiveRegs[rx].Def = CurInstr;

      // Kill off domains redefined by generic instructions.
      if (Kill)
        kill(rx);
    }
  }
  ++CurInstr;
}

/// \break Break false dependencies on undefined register reads.
///
/// Walk the block backward computing precise liveness. This is expensive, so we
/// only do it on demand. Note that the occurrence of undefined register reads
/// that should be broken is very rare, but when they occur we may have many in
/// a single block.
void ExecutionDepsFix::processUndefReads(MachineBasicBlock *MBB) {
  if (UndefReads.empty())
    return;

  // Collect this block's live out register units.
  LiveRegSet.init(*TRI);
  // We do not need to care about pristine registers as they are just preserved
  // but not actually used in the function.
  LiveRegSet.addLiveOutsNoPristines(*MBB);

  MachineInstr *UndefMI = UndefReads.back().first;
  unsigned OpIdx = UndefReads.back().second;

  for (MachineInstr &I : make_range(MBB->rbegin(), MBB->rend())) {
    // Update liveness, including the current instruction's defs.
    LiveRegSet.stepBackward(I);

    if (UndefMI == &I) {
      if (!LiveRegSet.contains(UndefMI->getOperand(OpIdx).getReg()))
        TII->breakPartialRegDependency(*UndefMI, OpIdx, TRI);

      UndefReads.pop_back();
      if (UndefReads.empty())
        return;

      UndefMI = UndefReads.back().first;
      OpIdx = UndefReads.back().second;
    }
  }
}

// A hard instruction only works in one domain. All input registers will be
// forced into that domain.
void ExecutionDepsFix::visitHardInstr(MachineInstr *mi, unsigned domain) {
  // Collapse all uses.
  for (unsigned i = mi->getDesc().getNumDefs(),
                e = mi->getDesc().getNumOperands(); i != e; ++i) {
    MachineOperand &mo = mi->getOperand(i);
    if (!mo.isReg()) continue;
    for (int rx : regIndices(mo.getReg())) {
      force(rx, domain);
    }
  }

  // Kill all defs and force them.
  for (unsigned i = 0, e = mi->getDesc().getNumDefs(); i != e; ++i) {
    MachineOperand &mo = mi->getOperand(i);
    if (!mo.isReg()) continue;
    for (int rx : regIndices(mo.getReg())) {
      kill(rx);
      force(rx, domain);
    }
  }
}

// A soft instruction can be changed to work in other domains given by mask.
void ExecutionDepsFix::visitSoftInstr(MachineInstr *mi, unsigned mask) {
  // Bitmask of available domains for this instruction after taking collapsed
  // operands into account.
  unsigned available = mask;

  // Scan the explicit use operands for incoming domains.
  SmallVector<int, 4> used;
  if (LiveRegs)
    for (unsigned i = mi->getDesc().getNumDefs(),
                  e = mi->getDesc().getNumOperands(); i != e; ++i) {
      MachineOperand &mo = mi->getOperand(i);
      if (!mo.isReg()) continue;
      for (int rx : regIndices(mo.getReg())) {
        DomainValue *dv = LiveRegs[rx].Value;
        if (dv == nullptr)
          continue;
        // Bitmask of domains that dv and available have in common.
        unsigned common = dv->getCommonDomains(available);
        // Is it possible to use this collapsed register for free?
        if (dv->isCollapsed()) {
          // Restrict available domains to the ones in common with the operand.
          // If there are no common domains, we must pay the cross-domain
          // penalty for this operand.
          if (common) available = common;
        } else if (common)
          // Open DomainValue is compatible, save it for merging.
          used.push_back(rx);
        else
          // Open DomainValue is not compatible with instruction. It is useless
          // now.
          kill(rx);
      }
    }

  // If the collapsed operands force a single domain, propagate the collapse.
  if (isPowerOf2_32(available)) {
    unsigned domain = countTrailingZeros(available);
    TII->setExecutionDomain(*mi, domain);
    visitHardInstr(mi, domain);
    return;
  }

  // Kill off any remaining uses that don't match available, and build a list of
  // incoming DomainValues that we want to merge.
  SmallVector<const LiveReg *, 4> Regs;
  for (int rx : used) {
    assert(LiveRegs && "no space allocated for live registers");
    const LiveReg &LR = LiveRegs[rx];
    // This useless DomainValue could have been missed above.
    if (!LR.Value->getCommonDomains(available)) {
      kill(rx);
      continue;
    }
    // Sorted insertion.
    auto I = std::upper_bound(Regs.begin(), Regs.end(), &LR,
                              [](const LiveReg *LHS, const LiveReg *RHS) {
                                return LHS->Def < RHS->Def;
                              });
    Regs.insert(I, &LR);
  }

  // doms are now sorted in order of appearance. Try to merge them all, giving
  // priority to the latest ones.
  DomainValue *dv = nullptr;
  while (!Regs.empty()) {
    if (!dv) {
      dv = Regs.pop_back_val()->Value;
      // Force the first dv to match the current instruction.
      dv->AvailableDomains = dv->getCommonDomains(available);
      assert(dv->AvailableDomains && "Domain should have been filtered");
      continue;
    }

    DomainValue *Latest = Regs.pop_back_val()->Value;
    // Skip already merged values.
    if (Latest == dv || Latest->Next)
      continue;
    if (merge(dv, Latest))
      continue;

    // If latest didn't merge, it is useless now. Kill all registers using it.
    for (int i : used) {
      assert(LiveRegs && "no space allocated for live registers");
      if (LiveRegs[i].Value == Latest)
        kill(i);
    }
  }

  // dv is the DomainValue we are going to use for this instruction.
  if (!dv) {
    dv = alloc();
    dv->AvailableDomains = available;
  }
  dv->Instrs.push_back(mi);

  // Finally set all defs and non-collapsed uses to dv. We must iterate through
  // all the operators, including imp-def ones.
  for (MachineInstr::mop_iterator ii = mi->operands_begin(),
                                  ee = mi->operands_end();
                                  ii != ee; ++ii) {
    MachineOperand &mo = *ii;
    if (!mo.isReg()) continue;
    for (int rx : regIndices(mo.getReg())) {
      if (!LiveRegs[rx].Value || (mo.isDef() && LiveRegs[rx].Value != dv)) {
        kill(rx);
        setLiveReg(rx, dv);
      }
    }
  }
}

void ExecutionDepsFix::processBasicBlock(MachineBasicBlock *MBB,
                                         bool PrimaryPass) {
  enterBasicBlock(MBB);
  // If this block is not done, it makes little sense to make any decisions
  // based on clearance information. We need to make a second pass anyway,
  // and by then we'll have better information, so we can avoid doing the work
  // to try and break dependencies now.
  bool breakDependency = isBlockDone(MBB);
  for (MachineInstr &MI : *MBB) {
    if (!MI.isDebugValue()) {
      bool Kill = false;
      if (PrimaryPass)
        Kill = visitInstr(&MI);
      processDefs(&MI, breakDependency, Kill);
    }
  }
  if (breakDependency)
    processUndefReads(MBB);
  leaveBasicBlock(MBB);
}

bool ExecutionDepsFix::isBlockDone(MachineBasicBlock *MBB) {
  return MBBInfos[MBB].PrimaryCompleted &&
         MBBInfos[MBB].IncomingCompleted == MBBInfos[MBB].PrimaryIncoming &&
         MBBInfos[MBB].IncomingProcessed == MBB->pred_size();
}

bool ExecutionDepsFix::runOnMachineFunction(MachineFunction &mf) {
  if (skipFunction(*mf.getFunction()))
    return false;
  MF = &mf;
  TII = MF->getSubtarget().getInstrInfo();
  TRI = MF->getSubtarget().getRegisterInfo();
  RegClassInfo.runOnMachineFunction(mf);
  LiveRegs = nullptr;
  assert(NumRegs == RC->getNumRegs() && "Bad regclass");

  DEBUG(dbgs() << "********** FIX EXECUTION DEPENDENCIES: "
               << TRI->getRegClassName(RC) << " **********\n");

  // If no relevant registers are used in the function, we can skip it
  // completely.
  bool anyregs = false;
  const MachineRegisterInfo &MRI = mf.getRegInfo();
  for (unsigned Reg : *RC) {
    if (MRI.isPhysRegUsed(Reg)) {
      anyregs = true;
      break;
    }
  }
  if (!anyregs) return false;

  // Initialize the AliasMap on the first use.
  if (AliasMap.empty()) {
    // Given a PhysReg, AliasMap[PhysReg] returns a list of indices into RC and
    // therefore the LiveRegs array.
    AliasMap.resize(TRI->getNumRegs());
    for (unsigned i = 0, e = RC->getNumRegs(); i != e; ++i)
      for (MCRegAliasIterator AI(RC->getRegister(i), TRI, true);
           AI.isValid(); ++AI)
        AliasMap[*AI].push_back(i);
  }

  // Initialize the MMBInfos
  for (auto &MBB : mf) {
    MBBInfo InitialInfo;
    MBBInfos.insert(std::make_pair(&MBB, InitialInfo));
  }

  /*
   *  We want to visit every instruction in every basic block in order to update
   *  it's execution domain or break any false dependencies. However, for the
   *  dependency breaking, we need to know clearances from all predecessors
   *  (including any backedges). One way to do so would be to do two complete
   *  passes over all basic blocks/instructions, the first for recording
   *  clearances, the second to break the dependencies. However, for functions
   *  without backedges, or functions with a lot of straight-line code, and
   *  a small loop, that would be a lot of unnecessary work (since only the
   *  BBs that are part of the loop require two passes). As an example,
   *  consider the following loop.
   *
   *
   *     PH -> A -> B (xmm<Undef> -> xmm<Def>) -> C -> D -> EXIT
   *           ^                                  |
   *           +----------------------------------+
   *
   *  The iteration order is as follows:
   *  Naive: PH A B C D A' B' C' D'
   *  Optimized: PH A B C A' B' C' D
   *
   *  Note that we avoid processing D twice, because we can entirely process
   *  the predecessors before getting to D. We call a block that is ready
   *  for its second round of processing `done` (isBlockDone). Once we finish
   *  processing some block, we update the counters in MBBInfos and re-process
   *  any successors that are now done.
   */

  MachineBasicBlock *Entry = &*MF->begin();
  ReversePostOrderTraversal<MachineBasicBlock*> RPOT(Entry);
  SmallVector<MachineBasicBlock *, 4> Workqueue;
  for (ReversePostOrderTraversal<MachineBasicBlock*>::rpo_iterator
         MBBI = RPOT.begin(), MBBE = RPOT.end(); MBBI != MBBE; ++MBBI) {
    MachineBasicBlock *MBB = *MBBI;
    // N.B: IncomingProcessed and IncomingCompleted were already updated while
    // processing this block's predecessors.
    MBBInfos[MBB].PrimaryCompleted = true;
    MBBInfos[MBB].PrimaryIncoming = MBBInfos[MBB].IncomingProcessed;
    bool Primary = true;
    Workqueue.push_back(MBB);
    while (!Workqueue.empty()) {
      MachineBasicBlock *ActiveMBB = &*Workqueue.back();
      Workqueue.pop_back();
      processBasicBlock(ActiveMBB, Primary);
      bool Done = isBlockDone(ActiveMBB);
      for (auto *Succ : ActiveMBB->successors()) {
        if (!isBlockDone(Succ)) {
          if (Primary) {
            MBBInfos[Succ].IncomingProcessed++;
          }
          if (Done) {
            MBBInfos[Succ].IncomingCompleted++;
          }
          if (isBlockDone(Succ)) {
            Workqueue.push_back(Succ);
          }
        }
      }
      Primary = false;
    }
  }

  // We need to go through again and finalize any blocks that are not done yet.
  // This is possible if blocks have dead predecessors, so we didn't visit them
  // above.
  for (ReversePostOrderTraversal<MachineBasicBlock *>::rpo_iterator
           MBBI = RPOT.begin(),
           MBBE = RPOT.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock *MBB = *MBBI;
    if (!isBlockDone(MBB)) {
      processBasicBlock(MBB, false);
      // Don't update successors here. We'll get to them anyway through this
      // loop.
    }
  }

  // Clear the LiveOuts vectors and collapse any remaining DomainValues.
  for (ReversePostOrderTraversal<MachineBasicBlock*>::rpo_iterator
         MBBI = RPOT.begin(), MBBE = RPOT.end(); MBBI != MBBE; ++MBBI) {
    auto FI = MBBInfos.find(*MBBI);
    if (FI == MBBInfos.end() || !FI->second.OutRegs)
      continue;
    for (unsigned i = 0, e = NumRegs; i != e; ++i)
      if (FI->second.OutRegs[i].Value)
        release(FI->second.OutRegs[i].Value);
    delete[] FI->second.OutRegs;
  }
  MBBInfos.clear();
  UndefReads.clear();
  Avail.clear();
  Allocator.DestroyAll();

  return false;
}
