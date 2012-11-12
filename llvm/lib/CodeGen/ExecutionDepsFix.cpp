//===- ExecutionDepsFix.cpp - Fix execution dependecy issues ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the execution dependency fix pass.
//
// Some X86 SSE instructions like mov, and, or, xor are available in different
// variants for different operand types. These variant instructions are
// equivalent, but on Nehalem and newer cpus there is extra latency
// transferring data between integer and floating point domains.  ARM cores
// have similar issues when they are configured with both VFP and NEON
// pipelines.
//
// This pass changes the variant instructions to minimize domain crossings.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "execution-fix"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// A DomainValue is a bit like LiveIntervals' ValNo, but it also keeps track
/// of execution domains.
///
/// An open DomainValue represents a set of instructions that can still switch
/// execution domain. Multiple registers may refer to the same open
/// DomainValue - they will eventually be collapsed to the same execution
/// domain.
///
/// A collapsed DomainValue represents a single register that has been forced
/// into one of more execution domains. There is a separate collapsed
/// DomainValue for each register, but it may contain multiple execution
/// domains. A register value is initially created in a single execution
/// domain, but if we were forced to pay the penalty of a domain crossing, we
/// keep track of the fact that the register is now available in multiple
/// domains.
namespace {
struct DomainValue {
  // Basic reference counting.
  unsigned Refs;

  // Bitmask of available domains. For an open DomainValue, it is the still
  // possible domains for collapsing. For a collapsed DomainValue it is the
  // domains where the register is available for free.
  unsigned AvailableDomains;

  // Pointer to the next DomainValue in a chain.  When two DomainValues are
  // merged, Victim.Next is set to point to Victor, so old DomainValue
  // references can be updated by following the chain.
  DomainValue *Next;

  // Twiddleable instructions using or defining these registers.
  SmallVector<MachineInstr*, 8> Instrs;

  // A collapsed DomainValue has no instructions to twiddle - it simply keeps
  // track of the domains where the registers are already available.
  bool isCollapsed() const { return Instrs.empty(); }

  // Is domain available?
  bool hasDomain(unsigned domain) const {
    return AvailableDomains & (1u << domain);
  }

  // Mark domain as available.
  void addDomain(unsigned domain) {
    AvailableDomains |= 1u << domain;
  }

  // Restrict to a single domain available.
  void setSingleDomain(unsigned domain) {
    AvailableDomains = 1u << domain;
  }

  // Return bitmask of domains that are available and in mask.
  unsigned getCommonDomains(unsigned mask) const {
    return AvailableDomains & mask;
  }

  // First domain available.
  unsigned getFirstDomain() const {
    return CountTrailingZeros_32(AvailableDomains);
  }

  DomainValue() : Refs(0) { clear(); }

  // Clear this DomainValue and point to next which has all its data.
  void clear() {
    AvailableDomains = 0;
    Next = 0;
    Instrs.clear();
  }
};
}

namespace {
/// LiveReg - Information about a live register.
struct LiveReg {
  /// Value currently in this register, or NULL when no value is being tracked.
  /// This counts as a DomainValue reference.
  DomainValue *Value;

  /// Instruction that defined this register, relative to the beginning of the
  /// current basic block.  When a LiveReg is used to represent a live-out
  /// register, this value is relative to the end of the basic block, so it
  /// will be a negative number.
  int Def;
};
} // anonynous namespace

namespace {
class ExeDepsFix : public MachineFunctionPass {
  static char ID;
  SpecificBumpPtrAllocator<DomainValue> Allocator;
  SmallVector<DomainValue*,16> Avail;

  const TargetRegisterClass *const RC;
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  std::vector<int> AliasMap;
  const unsigned NumRegs;
  LiveReg *LiveRegs;
  typedef DenseMap<MachineBasicBlock*, LiveReg*> LiveOutMap;
  LiveOutMap LiveOuts;

  /// Current instruction number.
  /// The first instruction in each basic block is 0.
  int CurInstr;

  /// True when the current block has a predecessor that hasn't been visited
  /// yet.
  bool SeenUnknownBackEdge;

public:
  ExeDepsFix(const TargetRegisterClass *rc)
    : MachineFunctionPass(ID), RC(rc), NumRegs(RC->getNumRegs()) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "Execution dependency fix";
  }

private:
  // Register mapping.
  int regIndex(unsigned Reg);

  // DomainValue allocation.
  DomainValue *alloc(int domain = -1);
  DomainValue *retain(DomainValue *DV) {
    if (DV) ++DV->Refs;
    return DV;
  }
  void release(DomainValue*);
  DomainValue *resolve(DomainValue*&);

  // LiveRegs manipulations.
  void setLiveReg(int rx, DomainValue *DV);
  void kill(int rx);
  void force(int rx, unsigned domain);
  void collapse(DomainValue *dv, unsigned domain);
  bool merge(DomainValue *A, DomainValue *B);

  void enterBasicBlock(MachineBasicBlock*);
  void leaveBasicBlock(MachineBasicBlock*);
  void visitInstr(MachineInstr*);
  void processDefs(MachineInstr*, bool Kill);
  void visitSoftInstr(MachineInstr*, unsigned mask);
  void visitHardInstr(MachineInstr*, unsigned domain);
};
}

char ExeDepsFix::ID = 0;

/// Translate TRI register number to an index into our smaller tables of
/// interesting registers. Return -1 for boring registers.
int ExeDepsFix::regIndex(unsigned Reg) {
  assert(Reg < AliasMap.size() && "Invalid register");
  return AliasMap[Reg];
}

DomainValue *ExeDepsFix::alloc(int domain) {
  DomainValue *dv = Avail.empty() ?
                      new(Allocator.Allocate()) DomainValue :
                      Avail.pop_back_val();
  if (domain >= 0)
    dv->addDomain(domain);
  assert(dv->Refs == 0 && "Reference count wasn't cleared");
  assert(!dv->Next && "Chained DomainValue shouldn't have been recycled");
  return dv;
}

/// release - Release a reference to DV.  When the last reference is released,
/// collapse if needed.
void ExeDepsFix::release(DomainValue *DV) {
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

/// resolve - Follow the chain of dead DomainValues until a live DomainValue is
/// reached.  Update the referenced pointer when necessary.
DomainValue *ExeDepsFix::resolve(DomainValue *&DVRef) {
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
void ExeDepsFix::setLiveReg(int rx, DomainValue *dv) {
  assert(unsigned(rx) < NumRegs && "Invalid index");
  assert(LiveRegs && "Must enter basic block first.");

  if (LiveRegs[rx].Value == dv)
    return;
  if (LiveRegs[rx].Value)
    release(LiveRegs[rx].Value);
  LiveRegs[rx].Value = retain(dv);
}

// Kill register rx, recycle or collapse any DomainValue.
void ExeDepsFix::kill(int rx) {
  assert(unsigned(rx) < NumRegs && "Invalid index");
  assert(LiveRegs && "Must enter basic block first.");
  if (!LiveRegs[rx].Value)
    return;

  release(LiveRegs[rx].Value);
  LiveRegs[rx].Value = 0;
}

/// Force register rx into domain.
void ExeDepsFix::force(int rx, unsigned domain) {
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
void ExeDepsFix::collapse(DomainValue *dv, unsigned domain) {
  assert(dv->hasDomain(domain) && "Cannot collapse");

  // Collapse all the instructions.
  while (!dv->Instrs.empty())
    TII->setExecutionDomain(dv->Instrs.pop_back_val(), domain);
  dv->setSingleDomain(domain);

  // If there are multiple users, give them new, unique DomainValues.
  if (LiveRegs && dv->Refs > 1)
    for (unsigned rx = 0; rx != NumRegs; ++rx)
      if (LiveRegs[rx].Value == dv)
        setLiveReg(rx, alloc(domain));
}

/// Merge - All instructions and registers in B are moved to A, and B is
/// released.
bool ExeDepsFix::merge(DomainValue *A, DomainValue *B) {
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

  for (unsigned rx = 0; rx != NumRegs; ++rx)
    if (LiveRegs[rx].Value == B)
      setLiveReg(rx, A);
  return true;
}

// enterBasicBlock - Set up LiveRegs by merging predecessor live-out values.
void ExeDepsFix::enterBasicBlock(MachineBasicBlock *MBB) {
  // Detect back-edges from predecessors we haven't processed yet.
  SeenUnknownBackEdge = false;

  // Reset instruction counter in each basic block.
  CurInstr = 0;

  // Set up LiveRegs to represent registers entering MBB.
  if (!LiveRegs)
    LiveRegs = new LiveReg[NumRegs];

  // Default values are 'nothing happened a long time ago'.
  for (unsigned rx = 0; rx != NumRegs; ++rx) {
    LiveRegs[rx].Value = 0;
    LiveRegs[rx].Def = -(1 << 20);
  }

  // This is the entry block.
  if (MBB->pred_empty()) {
    for (MachineBasicBlock::livein_iterator i = MBB->livein_begin(),
         e = MBB->livein_end(); i != e; ++i) {
      int rx = regIndex(*i);
      if (rx < 0)
        continue;
      // Treat function live-ins as if they were defined just before the first
      // instruction.  Usually, function arguments are set up immediately
      // before the call.
      LiveRegs[rx].Def = -1;
    }
    DEBUG(dbgs() << "BB#" << MBB->getNumber() << ": entry\n");
    return;
  }

  // Try to coalesce live-out registers from predecessors.
  for (MachineBasicBlock::const_pred_iterator pi = MBB->pred_begin(),
       pe = MBB->pred_end(); pi != pe; ++pi) {
    LiveOutMap::const_iterator fi = LiveOuts.find(*pi);
    if (fi == LiveOuts.end()) {
      SeenUnknownBackEdge = true;
      continue;
    }
    assert(fi->second && "Can't have NULL entries");

    for (unsigned rx = 0; rx != NumRegs; ++rx) {
      // Use the most recent predecessor def for each register.
      LiveRegs[rx].Def = std::max(LiveRegs[rx].Def, fi->second[rx].Def);

      DomainValue *pdv = resolve(fi->second[rx].Value);
      if (!pdv)
        continue;
      if (!LiveRegs[rx].Value) {
        setLiveReg(rx, pdv);
        continue;
      }

      // We have a live DomainValue from more than one predecessor.
      if (LiveRegs[rx].Value->isCollapsed()) {
        // We are already collapsed, but predecessor is not. Force him.
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
  DEBUG(dbgs() << "BB#" << MBB->getNumber()
        << (SeenUnknownBackEdge ? ": incomplete\n" : ": all preds known\n"));
}

void ExeDepsFix::leaveBasicBlock(MachineBasicBlock *MBB) {
  assert(LiveRegs && "Must enter basic block first.");
  // Save live registers at end of MBB - used by enterBasicBlock().
  // Also use LiveOuts as a visited set to detect back-edges.
  bool First = LiveOuts.insert(std::make_pair(MBB, LiveRegs)).second;

  if (First) {
    // LiveRegs was inserted in LiveOuts.  Adjust all defs to be relative to
    // the end of this block instead of the beginning.
    for (unsigned i = 0, e = NumRegs; i != e; ++i)
      LiveRegs[i].Def -= CurInstr;
  } else {
    // Insertion failed, this must be the second pass.
    // Release all the DomainValues instead of keeping them.
    for (unsigned i = 0, e = NumRegs; i != e; ++i)
      release(LiveRegs[i].Value);
    delete[] LiveRegs;
  }
  LiveRegs = 0;
}

void ExeDepsFix::visitInstr(MachineInstr *MI) {
  if (MI->isDebugValue())
    return;

  // Update instructions with explicit execution domains.
  std::pair<uint16_t, uint16_t> DomP = TII->getExecutionDomain(MI);
  if (DomP.first) {
    if (DomP.second)
      visitSoftInstr(MI, DomP.second);
    else
      visitHardInstr(MI, DomP.first);
  }

  // Process defs to track register ages, and kill values clobbered by generic
  // instructions.
  processDefs(MI, !DomP.first);
}

// Update def-ages for registers defined by MI.
// If Kill is set, also kill off DomainValues clobbered by the defs.
void ExeDepsFix::processDefs(MachineInstr *MI, bool Kill) {
  assert(!MI->isDebugValue() && "Won't process debug values");
  const MCInstrDesc &MCID = MI->getDesc();
  for (unsigned i = 0,
         e = MI->isVariadic() ? MI->getNumOperands() : MCID.getNumDefs();
         i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    if (MO.isImplicit())
      break;
    if (MO.isUse())
      continue;
    int rx = regIndex(MO.getReg());
    if (rx < 0)
      continue;

    // This instruction explicitly defines rx.
    DEBUG(dbgs() << TRI->getName(RC->getRegister(rx)) << ":\t" << CurInstr
                 << '\t' << *MI);

    // How many instructions since rx was last written?
    unsigned Clearance = CurInstr - LiveRegs[rx].Def;
    LiveRegs[rx].Def = CurInstr;

    // Kill off domains redefined by generic instructions.
    if (Kill)
      kill(rx);

    // Verify clearance before partial register updates.
    unsigned Pref = TII->getPartialRegUpdateClearance(MI, i, TRI);
    if (!Pref)
      continue;
    DEBUG(dbgs() << "Clearance: " << Clearance << ", want " << Pref);
    if (Pref > Clearance) {
      DEBUG(dbgs() << ": Break dependency.\n");
      TII->breakPartialRegDependency(MI, i, TRI);
      continue;
    }

    // The current clearance seems OK, but we may be ignoring a def from a
    // back-edge.
    if (!SeenUnknownBackEdge || Pref <= unsigned(CurInstr)) {
      DEBUG(dbgs() << ": OK.\n");
      continue;
    }

    // A def from an unprocessed back-edge may make us break this dependency.
    DEBUG(dbgs() << ": Wait for back-edge to resolve.\n");
  }

  ++CurInstr;
}

// A hard instruction only works in one domain. All input registers will be
// forced into that domain.
void ExeDepsFix::visitHardInstr(MachineInstr *mi, unsigned domain) {
  // Collapse all uses.
  for (unsigned i = mi->getDesc().getNumDefs(),
                e = mi->getDesc().getNumOperands(); i != e; ++i) {
    MachineOperand &mo = mi->getOperand(i);
    if (!mo.isReg()) continue;
    int rx = regIndex(mo.getReg());
    if (rx < 0) continue;
    force(rx, domain);
  }

  // Kill all defs and force them.
  for (unsigned i = 0, e = mi->getDesc().getNumDefs(); i != e; ++i) {
    MachineOperand &mo = mi->getOperand(i);
    if (!mo.isReg()) continue;
    int rx = regIndex(mo.getReg());
    if (rx < 0) continue;
    kill(rx);
    force(rx, domain);
  }
}

// A soft instruction can be changed to work in other domains given by mask.
void ExeDepsFix::visitSoftInstr(MachineInstr *mi, unsigned mask) {
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
      int rx = regIndex(mo.getReg());
      if (rx < 0) continue;
      if (DomainValue *dv = LiveRegs[rx].Value) {
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
    unsigned domain = CountTrailingZeros_32(available);
    TII->setExecutionDomain(mi, domain);
    visitHardInstr(mi, domain);
    return;
  }

  // Kill off any remaining uses that don't match available, and build a list of
  // incoming DomainValues that we want to merge.
  SmallVector<LiveReg, 4> Regs;
  for (SmallVector<int, 4>::iterator i=used.begin(), e=used.end(); i!=e; ++i) {
    int rx = *i;
    const LiveReg &LR = LiveRegs[rx];
    // This useless DomainValue could have been missed above.
    if (!LR.Value->getCommonDomains(available)) {
      kill(rx);
      continue;
    }
    // Sorted insertion.
    bool Inserted = false;
    for (SmallVector<LiveReg, 4>::iterator i = Regs.begin(), e = Regs.end();
           i != e && !Inserted; ++i) {
      if (LR.Def < i->Def) {
        Inserted = true;
        Regs.insert(i, LR);
      }
    }
    if (!Inserted)
      Regs.push_back(LR);
  }

  // doms are now sorted in order of appearance. Try to merge them all, giving
  // priority to the latest ones.
  DomainValue *dv = 0;
  while (!Regs.empty()) {
    if (!dv) {
      dv = Regs.pop_back_val().Value;
      // Force the first dv to match the current instruction.
      dv->AvailableDomains = dv->getCommonDomains(available);
      assert(dv->AvailableDomains && "Domain should have been filtered");
      continue;
    }

    DomainValue *Latest = Regs.pop_back_val().Value;
    // Skip already merged values.
    if (Latest == dv || Latest->Next)
      continue;
    if (merge(dv, Latest))
      continue;

    // If latest didn't merge, it is useless now. Kill all registers using it.
    for (SmallVector<int,4>::iterator i=used.begin(), e=used.end(); i != e; ++i)
      if (LiveRegs[*i].Value == Latest)
        kill(*i);
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
    int rx = regIndex(mo.getReg());
    if (rx < 0) continue;
    if (!LiveRegs[rx].Value || (mo.isDef() && LiveRegs[rx].Value != dv)) {
      kill(rx);
      setLiveReg(rx, dv);
    }
  }
}

bool ExeDepsFix::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  TII = MF->getTarget().getInstrInfo();
  TRI = MF->getTarget().getRegisterInfo();
  LiveRegs = 0;
  assert(NumRegs == RC->getNumRegs() && "Bad regclass");

  DEBUG(dbgs() << "********** FIX EXECUTION DEPENDENCIES: "
               << RC->getName() << " **********\n");

  // If no relevant registers are used in the function, we can skip it
  // completely.
  bool anyregs = false;
  for (TargetRegisterClass::const_iterator I = RC->begin(), E = RC->end();
       I != E; ++I)
    if (MF->getRegInfo().isPhysRegUsed(*I)) {
      anyregs = true;
      break;
    }
  if (!anyregs) return false;

  // Initialize the AliasMap on the first use.
  if (AliasMap.empty()) {
    // Given a PhysReg, AliasMap[PhysReg] is either the relevant index into RC,
    // or -1.
    AliasMap.resize(TRI->getNumRegs(), -1);
    for (unsigned i = 0, e = RC->getNumRegs(); i != e; ++i)
      for (MCRegAliasIterator AI(RC->getRegister(i), TRI, true);
           AI.isValid(); ++AI)
        AliasMap[*AI] = i;
  }

  MachineBasicBlock *Entry = MF->begin();
  ReversePostOrderTraversal<MachineBasicBlock*> RPOT(Entry);
  SmallVector<MachineBasicBlock*, 16> Loops;
  for (ReversePostOrderTraversal<MachineBasicBlock*>::rpo_iterator
         MBBI = RPOT.begin(), MBBE = RPOT.end(); MBBI != MBBE; ++MBBI) {
    MachineBasicBlock *MBB = *MBBI;
    enterBasicBlock(MBB);
    if (SeenUnknownBackEdge)
      Loops.push_back(MBB);
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
        ++I)
      visitInstr(I);
    leaveBasicBlock(MBB);
  }

  // Visit all the loop blocks again in order to merge DomainValues from
  // back-edges.
  for (unsigned i = 0, e = Loops.size(); i != e; ++i) {
    MachineBasicBlock *MBB = Loops[i];
    enterBasicBlock(MBB);
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
        ++I)
      if (!I->isDebugValue())
        processDefs(I, false);
    leaveBasicBlock(MBB);
  }

  // Clear the LiveOuts vectors and collapse any remaining DomainValues.
  for (ReversePostOrderTraversal<MachineBasicBlock*>::rpo_iterator
         MBBI = RPOT.begin(), MBBE = RPOT.end(); MBBI != MBBE; ++MBBI) {
    LiveOutMap::const_iterator FI = LiveOuts.find(*MBBI);
    if (FI == LiveOuts.end() || !FI->second)
      continue;
    for (unsigned i = 0, e = NumRegs; i != e; ++i)
      if (FI->second[i].Value)
        release(FI->second[i].Value);
    delete[] FI->second;
  }
  LiveOuts.clear();
  Avail.clear();
  Allocator.DestroyAll();

  return false;
}

FunctionPass *
llvm::createExecutionDependencyFixPass(const TargetRegisterClass *RC) {
  return new ExeDepsFix(RC);
}
