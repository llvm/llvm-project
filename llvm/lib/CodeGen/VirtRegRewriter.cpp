//===-- llvm/CodeGen/Rewriter.cpp -  Rewriter -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "virtregrewriter"
#include "VirtRegRewriter.h"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumDSE     , "Number of dead stores elided");
STATISTIC(NumDSS     , "Number of dead spill slots removed");
STATISTIC(NumCommutes, "Number of instructions commuted");
STATISTIC(NumDRM     , "Number of re-materializable defs elided");
STATISTIC(NumStores  , "Number of stores added");
STATISTIC(NumPSpills , "Number of physical register spills");
STATISTIC(NumOmitted , "Number of reloads omited");
STATISTIC(NumAvoided , "Number of reloads deemed unnecessary");
STATISTIC(NumCopified, "Number of available reloads turned into copies");
STATISTIC(NumReMats  , "Number of re-materialization");
STATISTIC(NumLoads   , "Number of loads added");
STATISTIC(NumReused  , "Number of values reused");
STATISTIC(NumDCE     , "Number of copies elided");
STATISTIC(NumSUnfold , "Number of stores unfolded");
STATISTIC(NumModRefUnfold, "Number of modref unfolded");

namespace {
  enum RewriterName { local, trivial };
}

static cl::opt<RewriterName>
RewriterOpt("rewriter",
            cl::desc("Rewriter to use (default=local)"),
            cl::Prefix,
            cl::values(clEnumVal(local,   "local rewriter"),
                       clEnumVal(trivial, "trivial rewriter"),
                       clEnumValEnd),
            cl::init(local));

static cl::opt<bool>
ScheduleSpills("schedule-spills",
               cl::desc("Schedule spill code"),
               cl::init(false));

VirtRegRewriter::~VirtRegRewriter() {}

/// substitutePhysReg - Replace virtual register in MachineOperand with a
/// physical register. Do the right thing with the sub-register index.
/// Note that operands may be added, so the MO reference is no longer valid.
static void substitutePhysReg(MachineOperand &MO, unsigned Reg,
                              const TargetRegisterInfo &TRI) {
  if (MO.getSubReg()) {
    MO.substPhysReg(Reg, TRI);

    // Any kill flags apply to the full virtual register, so they also apply to
    // the full physical register.
    // We assume that partial defs have already been decorated with a super-reg
    // <imp-def> operand by LiveIntervals.
    MachineInstr &MI = *MO.getParent();
    if (MO.isUse() && !MO.isUndef() &&
        (MO.isKill() || MI.isRegTiedToDefOperand(&MO-&MI.getOperand(0))))
      MI.addRegisterKilled(Reg, &TRI, /*AddIfNotFound=*/ true);
  } else {
    MO.setReg(Reg);
  }
}

namespace {

/// This class is intended for use with the new spilling framework only. It
/// rewrites vreg def/uses to use the assigned preg, but does not insert any
/// spill code.
struct TrivialRewriter : public VirtRegRewriter {

  bool runOnMachineFunction(MachineFunction &MF, VirtRegMap &VRM,
                            LiveIntervals* LIs) {
    DEBUG(dbgs() << "********** REWRITE MACHINE CODE **********\n");
    DEBUG(dbgs() << "********** Function: "
          << MF.getFunction()->getName() << '\n');
    DEBUG(dbgs() << "**** Machine Instrs"
          << "(NOTE! Does not include spills and reloads!) ****\n");
    DEBUG(MF.dump());

    MachineRegisterInfo *mri = &MF.getRegInfo();
    const TargetRegisterInfo *tri = MF.getTarget().getRegisterInfo();

    bool changed = false;

    for (LiveIntervals::iterator liItr = LIs->begin(), liEnd = LIs->end();
         liItr != liEnd; ++liItr) {

      const LiveInterval *li = liItr->second;
      unsigned reg = li->reg;

      if (TargetRegisterInfo::isPhysicalRegister(reg)) {
        if (!li->empty())
          mri->setPhysRegUsed(reg);
      }
      else {
        if (!VRM.hasPhys(reg))
          continue;
        unsigned pReg = VRM.getPhys(reg);
        mri->setPhysRegUsed(pReg);
        // Copy the register use-list before traversing it.
        SmallVector<std::pair<MachineInstr*, unsigned>, 32> reglist;
        for (MachineRegisterInfo::reg_iterator I = mri->reg_begin(reg),
               E = mri->reg_end(); I != E; ++I)
          reglist.push_back(std::make_pair(&*I, I.getOperandNo()));
        for (unsigned N=0; N != reglist.size(); ++N)
          substitutePhysReg(reglist[N].first->getOperand(reglist[N].second),
                            pReg, *tri);
        changed |= !reglist.empty();
      }
    }

    DEBUG(dbgs() << "**** Post Machine Instrs ****\n");
    DEBUG(MF.dump());

    return changed;
  }

};

}

// ************************************************************************ //

namespace {

/// AvailableSpills - As the local rewriter is scanning and rewriting an MBB
/// from top down, keep track of which spill slots or remat are available in
/// each register.
///
/// Note that not all physregs are created equal here.  In particular, some
/// physregs are reloads that we are allowed to clobber or ignore at any time.
/// Other physregs are values that the register allocated program is using
/// that we cannot CHANGE, but we can read if we like.  We keep track of this
/// on a per-stack-slot / remat id basis as the low bit in the value of the
/// SpillSlotsAvailable entries.  The predicate 'canClobberPhysReg()' checks
/// this bit and addAvailable sets it if.
class AvailableSpills {
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  // SpillSlotsOrReMatsAvailable - This map keeps track of all of the spilled
  // or remat'ed virtual register values that are still available, due to
  // being loaded or stored to, but not invalidated yet.
  std::map<int, unsigned> SpillSlotsOrReMatsAvailable;

  // PhysRegsAvailable - This is the inverse of SpillSlotsOrReMatsAvailable,
  // indicating which stack slot values are currently held by a physreg.  This
  // is used to invalidate entries in SpillSlotsOrReMatsAvailable when a
  // physreg is modified.
  std::multimap<unsigned, int> PhysRegsAvailable;

  void disallowClobberPhysRegOnly(unsigned PhysReg);

  void ClobberPhysRegOnly(unsigned PhysReg);
public:
  AvailableSpills(const TargetRegisterInfo *tri, const TargetInstrInfo *tii)
    : TRI(tri), TII(tii) {
  }

  /// clear - Reset the state.
  void clear() {
    SpillSlotsOrReMatsAvailable.clear();
    PhysRegsAvailable.clear();
  }

  const TargetRegisterInfo *getRegInfo() const { return TRI; }

  /// getSpillSlotOrReMatPhysReg - If the specified stack slot or remat is
  /// available in a physical register, return that PhysReg, otherwise
  /// return 0.
  unsigned getSpillSlotOrReMatPhysReg(int Slot) const {
    std::map<int, unsigned>::const_iterator I =
      SpillSlotsOrReMatsAvailable.find(Slot);
    if (I != SpillSlotsOrReMatsAvailable.end()) {
      return I->second >> 1;  // Remove the CanClobber bit.
    }
    return 0;
  }

  /// addAvailable - Mark that the specified stack slot / remat is available
  /// in the specified physreg.  If CanClobber is true, the physreg can be
  /// modified at any time without changing the semantics of the program.
  void addAvailable(int SlotOrReMat, unsigned Reg, bool CanClobber = true) {
    // If this stack slot is thought to be available in some other physreg,
    // remove its record.
    ModifyStackSlotOrReMat(SlotOrReMat);

    PhysRegsAvailable.insert(std::make_pair(Reg, SlotOrReMat));
    SpillSlotsOrReMatsAvailable[SlotOrReMat]= (Reg << 1) |
                                              (unsigned)CanClobber;

    if (SlotOrReMat > VirtRegMap::MAX_STACK_SLOT)
      DEBUG(dbgs() << "Remembering RM#"
                   << SlotOrReMat-VirtRegMap::MAX_STACK_SLOT-1);
    else
      DEBUG(dbgs() << "Remembering SS#" << SlotOrReMat);
    DEBUG(dbgs() << " in physreg " << TRI->getName(Reg) << "\n");
  }

  /// canClobberPhysRegForSS - Return true if the spiller is allowed to change
  /// the value of the specified stackslot register if it desires. The
  /// specified stack slot must be available in a physreg for this query to
  /// make sense.
  bool canClobberPhysRegForSS(int SlotOrReMat) const {
    assert(SpillSlotsOrReMatsAvailable.count(SlotOrReMat) &&
           "Value not available!");
    return SpillSlotsOrReMatsAvailable.find(SlotOrReMat)->second & 1;
  }

  /// canClobberPhysReg - Return true if the spiller is allowed to clobber the
  /// physical register where values for some stack slot(s) might be
  /// available.
  bool canClobberPhysReg(unsigned PhysReg) const {
    std::multimap<unsigned, int>::const_iterator I =
      PhysRegsAvailable.lower_bound(PhysReg);
    while (I != PhysRegsAvailable.end() && I->first == PhysReg) {
      int SlotOrReMat = I->second;
      I++;
      if (!canClobberPhysRegForSS(SlotOrReMat))
        return false;
    }
    return true;
  }

  /// disallowClobberPhysReg - Unset the CanClobber bit of the specified
  /// stackslot register. The register is still available but is no longer
  /// allowed to be modifed.
  void disallowClobberPhysReg(unsigned PhysReg);

  /// ClobberPhysReg - This is called when the specified physreg changes
  /// value.  We use this to invalidate any info about stuff that lives in
  /// it and any of its aliases.
  void ClobberPhysReg(unsigned PhysReg);

  /// ModifyStackSlotOrReMat - This method is called when the value in a stack
  /// slot changes.  This removes information about which register the
  /// previous value for this slot lives in (as the previous value is dead
  /// now).
  void ModifyStackSlotOrReMat(int SlotOrReMat);

  /// AddAvailableRegsToLiveIn - Availability information is being kept coming
  /// into the specified MBB. Add available physical registers as potential
  /// live-in's. If they are reused in the MBB, they will be added to the
  /// live-in set to make register scavenger and post-allocation scheduler.
  void AddAvailableRegsToLiveIn(MachineBasicBlock &MBB, BitVector &RegKills,
                                std::vector<MachineOperand*> &KillOps);
};

}

// ************************************************************************ //

// Given a location where a reload of a spilled register or a remat of
// a constant is to be inserted, attempt to find a safe location to
// insert the load at an earlier point in the basic-block, to hide
// latency of the load and to avoid address-generation interlock
// issues.
static MachineBasicBlock::iterator
ComputeReloadLoc(MachineBasicBlock::iterator const InsertLoc,
                 MachineBasicBlock::iterator const Begin,
                 unsigned PhysReg,
                 const TargetRegisterInfo *TRI,
                 bool DoReMat,
                 int SSorRMId,
                 const TargetInstrInfo *TII,
                 const MachineFunction &MF)
{
  if (!ScheduleSpills)
    return InsertLoc;

  // Spill backscheduling is of primary interest to addresses, so
  // don't do anything if the register isn't in the register class
  // used for pointers.

  const TargetLowering *TL = MF.getTarget().getTargetLowering();

  if (!TL->isTypeLegal(TL->getPointerTy()))
    // Believe it or not, this is true on PIC16.
    return InsertLoc;

  const TargetRegisterClass *ptrRegClass =
    TL->getRegClassFor(TL->getPointerTy());
  if (!ptrRegClass->contains(PhysReg))
    return InsertLoc;

  // Scan upwards through the preceding instructions. If an instruction doesn't
  // reference the stack slot or the register we're loading, we can
  // backschedule the reload up past it.
  MachineBasicBlock::iterator NewInsertLoc = InsertLoc;
  while (NewInsertLoc != Begin) {
    MachineBasicBlock::iterator Prev = prior(NewInsertLoc);
    for (unsigned i = 0; i < Prev->getNumOperands(); ++i) {
      MachineOperand &Op = Prev->getOperand(i);
      if (!DoReMat && Op.isFI() && Op.getIndex() == SSorRMId)
        goto stop;
    }
    if (Prev->findRegisterUseOperandIdx(PhysReg) != -1 ||
        Prev->findRegisterDefOperand(PhysReg))
      goto stop;
    for (const unsigned *Alias = TRI->getAliasSet(PhysReg); *Alias; ++Alias)
      if (Prev->findRegisterUseOperandIdx(*Alias) != -1 ||
          Prev->findRegisterDefOperand(*Alias))
        goto stop;
    NewInsertLoc = Prev;
  }
stop:;

  // If we made it to the beginning of the block, turn around and move back
  // down just past any existing reloads. They're likely to be reloads/remats
  // for instructions earlier than what our current reload/remat is for, so
  // they should be scheduled earlier.
  if (NewInsertLoc == Begin) {
    int FrameIdx;
    while (InsertLoc != NewInsertLoc &&
           (TII->isLoadFromStackSlot(NewInsertLoc, FrameIdx) ||
            TII->isTriviallyReMaterializable(NewInsertLoc)))
      ++NewInsertLoc;
  }

  return NewInsertLoc;
}

namespace {

// ReusedOp - For each reused operand, we keep track of a bit of information,
// in case we need to rollback upon processing a new operand.  See comments
// below.
struct ReusedOp {
  // The MachineInstr operand that reused an available value.
  unsigned Operand;

  // StackSlotOrReMat - The spill slot or remat id of the value being reused.
  unsigned StackSlotOrReMat;

  // PhysRegReused - The physical register the value was available in.
  unsigned PhysRegReused;

  // AssignedPhysReg - The physreg that was assigned for use by the reload.
  unsigned AssignedPhysReg;

  // VirtReg - The virtual register itself.
  unsigned VirtReg;

  ReusedOp(unsigned o, unsigned ss, unsigned prr, unsigned apr,
           unsigned vreg)
    : Operand(o), StackSlotOrReMat(ss), PhysRegReused(prr),
      AssignedPhysReg(apr), VirtReg(vreg) {}
};

/// ReuseInfo - This maintains a collection of ReuseOp's for each operand that
/// is reused instead of reloaded.
class ReuseInfo {
  MachineInstr &MI;
  std::vector<ReusedOp> Reuses;
  BitVector PhysRegsClobbered;
public:
  ReuseInfo(MachineInstr &mi, const TargetRegisterInfo *tri) : MI(mi) {
    PhysRegsClobbered.resize(tri->getNumRegs());
  }

  bool hasReuses() const {
    return !Reuses.empty();
  }

  /// addReuse - If we choose to reuse a virtual register that is already
  /// available instead of reloading it, remember that we did so.
  void addReuse(unsigned OpNo, unsigned StackSlotOrReMat,
                unsigned PhysRegReused, unsigned AssignedPhysReg,
                unsigned VirtReg) {
    // If the reload is to the assigned register anyway, no undo will be
    // required.
    if (PhysRegReused == AssignedPhysReg) return;

    // Otherwise, remember this.
    Reuses.push_back(ReusedOp(OpNo, StackSlotOrReMat, PhysRegReused,
                              AssignedPhysReg, VirtReg));
  }

  void markClobbered(unsigned PhysReg) {
    PhysRegsClobbered.set(PhysReg);
  }

  bool isClobbered(unsigned PhysReg) const {
    return PhysRegsClobbered.test(PhysReg);
  }

  /// GetRegForReload - We are about to emit a reload into PhysReg.  If there
  /// is some other operand that is using the specified register, either pick
  /// a new register to use, or evict the previous reload and use this reg.
  unsigned GetRegForReload(const TargetRegisterClass *RC, unsigned PhysReg,
                           MachineFunction &MF, MachineInstr *MI,
                           AvailableSpills &Spills,
                           std::vector<MachineInstr*> &MaybeDeadStores,
                           SmallSet<unsigned, 8> &Rejected,
                           BitVector &RegKills,
                           std::vector<MachineOperand*> &KillOps,
                           VirtRegMap &VRM);

  /// GetRegForReload - Helper for the above GetRegForReload(). Add a
  /// 'Rejected' set to remember which registers have been considered and
  /// rejected for the reload. This avoids infinite looping in case like
  /// this:
  /// t1 := op t2, t3
  /// t2 <- assigned r0 for use by the reload but ended up reuse r1
  /// t3 <- assigned r1 for use by the reload but ended up reuse r0
  /// t1 <- desires r1
  ///       sees r1 is taken by t2, tries t2's reload register r0
  ///       sees r0 is taken by t3, tries t3's reload register r1
  ///       sees r1 is taken by t2, tries t2's reload register r0 ...
  unsigned GetRegForReload(unsigned VirtReg, unsigned PhysReg, MachineInstr *MI,
                           AvailableSpills &Spills,
                           std::vector<MachineInstr*> &MaybeDeadStores,
                           BitVector &RegKills,
                           std::vector<MachineOperand*> &KillOps,
                           VirtRegMap &VRM) {
    SmallSet<unsigned, 8> Rejected;
    MachineFunction &MF = *MI->getParent()->getParent();
    const TargetRegisterClass* RC = MF.getRegInfo().getRegClass(VirtReg);
    return GetRegForReload(RC, PhysReg, MF, MI, Spills, MaybeDeadStores,
                           Rejected, RegKills, KillOps, VRM);
  }
};

}

// ****************** //
// Utility Functions  //
// ****************** //

/// findSinglePredSuccessor - Return via reference a vector of machine basic
/// blocks each of which is a successor of the specified BB and has no other
/// predecessor.
static void findSinglePredSuccessor(MachineBasicBlock *MBB,
                                   SmallVectorImpl<MachineBasicBlock *> &Succs){
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock *SuccMBB = *SI;
    if (SuccMBB->pred_size() == 1)
      Succs.push_back(SuccMBB);
  }
}

/// InvalidateKill - Invalidate register kill information for a specific
/// register. This also unsets the kills marker on the last kill operand.
static void InvalidateKill(unsigned Reg,
                           const TargetRegisterInfo* TRI,
                           BitVector &RegKills,
                           std::vector<MachineOperand*> &KillOps) {
  if (RegKills[Reg]) {
    KillOps[Reg]->setIsKill(false);
    // KillOps[Reg] might be a def of a super-register.
    unsigned KReg = KillOps[Reg]->getReg();
    KillOps[KReg] = NULL;
    RegKills.reset(KReg);
    for (const unsigned *SR = TRI->getSubRegisters(KReg); *SR; ++SR) {
      if (RegKills[*SR]) {
        KillOps[*SR]->setIsKill(false);
        KillOps[*SR] = NULL;
        RegKills.reset(*SR);
      }
    }
  }
}

/// InvalidateKills - MI is going to be deleted. If any of its operands are
/// marked kill, then invalidate the information.
static void InvalidateKills(MachineInstr &MI,
                            const TargetRegisterInfo* TRI,
                            BitVector &RegKills,
                            std::vector<MachineOperand*> &KillOps,
                            SmallVector<unsigned, 2> *KillRegs = NULL) {
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isUse() || !MO.isKill() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (KillRegs)
      KillRegs->push_back(Reg);
    assert(Reg < KillOps.size());
    if (KillOps[Reg] == &MO) {
      KillOps[Reg] = NULL;
      RegKills.reset(Reg);
      for (const unsigned *SR = TRI->getSubRegisters(Reg); *SR; ++SR) {
        if (RegKills[*SR]) {
          KillOps[*SR] = NULL;
          RegKills.reset(*SR);
        }
      }
    }
  }
}

/// InvalidateRegDef - If the def operand of the specified def MI is now dead
/// (since its spill instruction is removed), mark it isDead. Also checks if
/// the def MI has other definition operands that are not dead. Returns it by
/// reference.
static bool InvalidateRegDef(MachineBasicBlock::iterator I,
                             MachineInstr &NewDef, unsigned Reg,
                             bool &HasLiveDef,
                             const TargetRegisterInfo *TRI) {
  // Due to remat, it's possible this reg isn't being reused. That is,
  // the def of this reg (by prev MI) is now dead.
  MachineInstr *DefMI = I;
  MachineOperand *DefOp = NULL;
  for (unsigned i = 0, e = DefMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = DefMI->getOperand(i);
    if (!MO.isReg() || !MO.isDef() || !MO.isKill() || MO.isUndef())
      continue;
    if (MO.getReg() == Reg)
      DefOp = &MO;
    else if (!MO.isDead())
      HasLiveDef = true;
  }
  if (!DefOp)
    return false;

  bool FoundUse = false, Done = false;
  MachineBasicBlock::iterator E = &NewDef;
  ++I; ++E;
  for (; !Done && I != E; ++I) {
    MachineInstr *NMI = I;
    for (unsigned j = 0, ee = NMI->getNumOperands(); j != ee; ++j) {
      MachineOperand &MO = NMI->getOperand(j);
      if (!MO.isReg() || MO.getReg() == 0 ||
          (MO.getReg() != Reg && !TRI->isSubRegister(Reg, MO.getReg())))
        continue;
      if (MO.isUse())
        FoundUse = true;
      Done = true; // Stop after scanning all the operands of this MI.
    }
  }
  if (!FoundUse) {
    // Def is dead!
    DefOp->setIsDead();
    return true;
  }
  return false;
}

/// UpdateKills - Track and update kill info. If a MI reads a register that is
/// marked kill, then it must be due to register reuse. Transfer the kill info
/// over.
static void UpdateKills(MachineInstr &MI, const TargetRegisterInfo* TRI,
                        BitVector &RegKills,
                        std::vector<MachineOperand*> &KillOps) {
  // These do not affect kill info at all.
  if (MI.isDebugValue())
    return;
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.isUse() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;

    if (RegKills[Reg] && KillOps[Reg]->getParent() != &MI) {
      // That can't be right. Register is killed but not re-defined and it's
      // being reused. Let's fix that.
      KillOps[Reg]->setIsKill(false);
      // KillOps[Reg] might be a def of a super-register.
      unsigned KReg = KillOps[Reg]->getReg();
      KillOps[KReg] = NULL;
      RegKills.reset(KReg);

      // Must be a def of a super-register. Its other sub-regsters are no
      // longer killed as well.
      for (const unsigned *SR = TRI->getSubRegisters(KReg); *SR; ++SR) {
        KillOps[*SR] = NULL;
        RegKills.reset(*SR);
      }
    } else {
      // Check for subreg kills as well.
      // d4 =
      // store d4, fi#0
      // ...
      //    = s8<kill>
      // ...
      //    = d4  <avoiding reload>
      for (const unsigned *SR = TRI->getSubRegisters(Reg); *SR; ++SR) {
        unsigned SReg = *SR;
        if (RegKills[SReg] && KillOps[SReg]->getParent() != &MI) {
          KillOps[SReg]->setIsKill(false);
          unsigned KReg = KillOps[SReg]->getReg();
          KillOps[KReg] = NULL;
          RegKills.reset(KReg);

          for (const unsigned *SSR = TRI->getSubRegisters(KReg); *SSR; ++SSR) {
            KillOps[*SSR] = NULL;
            RegKills.reset(*SSR);
          }
        }
      }
    }

    if (MO.isKill()) {
      RegKills.set(Reg);
      KillOps[Reg] = &MO;
      for (const unsigned *SR = TRI->getSubRegisters(Reg); *SR; ++SR) {
        RegKills.set(*SR);
        KillOps[*SR] = &MO;
      }
    }
  }

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.getReg() || !MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    RegKills.reset(Reg);
    KillOps[Reg] = NULL;
    // It also defines (or partially define) aliases.
    for (const unsigned *SR = TRI->getSubRegisters(Reg); *SR; ++SR) {
      RegKills.reset(*SR);
      KillOps[*SR] = NULL;
    }
    for (const unsigned *SR = TRI->getSuperRegisters(Reg); *SR; ++SR) {
      RegKills.reset(*SR);
      KillOps[*SR] = NULL;
    }
  }
}

/// ReMaterialize - Re-materialize definition for Reg targetting DestReg.
///
static void ReMaterialize(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &MII,
                          unsigned DestReg, unsigned Reg,
                          const TargetInstrInfo *TII,
                          const TargetRegisterInfo *TRI,
                          VirtRegMap &VRM) {
  MachineInstr *ReMatDefMI = VRM.getReMaterializedMI(Reg);
#ifndef NDEBUG
  const TargetInstrDesc &TID = ReMatDefMI->getDesc();
  assert(TID.getNumDefs() == 1 &&
         "Don't know how to remat instructions that define > 1 values!");
#endif
  TII->reMaterialize(MBB, MII, DestReg, 0, ReMatDefMI, *TRI);
  MachineInstr *NewMI = prior(MII);
  for (unsigned i = 0, e = NewMI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = NewMI->getOperand(i);
    if (!MO.isReg() || MO.getReg() == 0)
      continue;
    unsigned VirtReg = MO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(VirtReg))
      continue;
    assert(MO.isUse());
    unsigned Phys = VRM.getPhys(VirtReg);
    assert(Phys && "Virtual register is not assigned a register?");
    substitutePhysReg(MO, Phys, *TRI);
  }
  ++NumReMats;
}

/// findSuperReg - Find the SubReg's super-register of given register class
/// where its SubIdx sub-register is SubReg.
static unsigned findSuperReg(const TargetRegisterClass *RC, unsigned SubReg,
                             unsigned SubIdx, const TargetRegisterInfo *TRI) {
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I) {
    unsigned Reg = *I;
    if (TRI->getSubReg(Reg, SubIdx) == SubReg)
      return Reg;
  }
  return 0;
}

// ******************************** //
// Available Spills Implementation  //
// ******************************** //

/// disallowClobberPhysRegOnly - Unset the CanClobber bit of the specified
/// stackslot register. The register is still available but is no longer
/// allowed to be modifed.
void AvailableSpills::disallowClobberPhysRegOnly(unsigned PhysReg) {
  std::multimap<unsigned, int>::iterator I =
    PhysRegsAvailable.lower_bound(PhysReg);
  while (I != PhysRegsAvailable.end() && I->first == PhysReg) {
    int SlotOrReMat = I->second;
    I++;
    assert((SpillSlotsOrReMatsAvailable[SlotOrReMat] >> 1) == PhysReg &&
           "Bidirectional map mismatch!");
    SpillSlotsOrReMatsAvailable[SlotOrReMat] &= ~1;
    DEBUG(dbgs() << "PhysReg " << TRI->getName(PhysReg)
         << " copied, it is available for use but can no longer be modified\n");
  }
}

/// disallowClobberPhysReg - Unset the CanClobber bit of the specified
/// stackslot register and its aliases. The register and its aliases may
/// still available but is no longer allowed to be modifed.
void AvailableSpills::disallowClobberPhysReg(unsigned PhysReg) {
  for (const unsigned *AS = TRI->getAliasSet(PhysReg); *AS; ++AS)
    disallowClobberPhysRegOnly(*AS);
  disallowClobberPhysRegOnly(PhysReg);
}

/// ClobberPhysRegOnly - This is called when the specified physreg changes
/// value.  We use this to invalidate any info about stuff we thing lives in it.
void AvailableSpills::ClobberPhysRegOnly(unsigned PhysReg) {
  std::multimap<unsigned, int>::iterator I =
    PhysRegsAvailable.lower_bound(PhysReg);
  while (I != PhysRegsAvailable.end() && I->first == PhysReg) {
    int SlotOrReMat = I->second;
    PhysRegsAvailable.erase(I++);
    assert((SpillSlotsOrReMatsAvailable[SlotOrReMat] >> 1) == PhysReg &&
           "Bidirectional map mismatch!");
    SpillSlotsOrReMatsAvailable.erase(SlotOrReMat);
    DEBUG(dbgs() << "PhysReg " << TRI->getName(PhysReg)
          << " clobbered, invalidating ");
    if (SlotOrReMat > VirtRegMap::MAX_STACK_SLOT)
      DEBUG(dbgs() << "RM#" << SlotOrReMat-VirtRegMap::MAX_STACK_SLOT-1 <<"\n");
    else
      DEBUG(dbgs() << "SS#" << SlotOrReMat << "\n");
  }
}

/// ClobberPhysReg - This is called when the specified physreg changes
/// value.  We use this to invalidate any info about stuff we thing lives in
/// it and any of its aliases.
void AvailableSpills::ClobberPhysReg(unsigned PhysReg) {
  for (const unsigned *AS = TRI->getAliasSet(PhysReg); *AS; ++AS)
    ClobberPhysRegOnly(*AS);
  ClobberPhysRegOnly(PhysReg);
}

/// AddAvailableRegsToLiveIn - Availability information is being kept coming
/// into the specified MBB. Add available physical registers as potential
/// live-in's. If they are reused in the MBB, they will be added to the
/// live-in set to make register scavenger and post-allocation scheduler.
void AvailableSpills::AddAvailableRegsToLiveIn(MachineBasicBlock &MBB,
                                        BitVector &RegKills,
                                        std::vector<MachineOperand*> &KillOps) {
  std::set<unsigned> NotAvailable;
  for (std::multimap<unsigned, int>::iterator
         I = PhysRegsAvailable.begin(), E = PhysRegsAvailable.end();
       I != E; ++I) {
    unsigned Reg = I->first;
    const TargetRegisterClass* RC = TRI->getMinimalPhysRegClass(Reg);
    // FIXME: A temporary workaround. We can't reuse available value if it's
    // not safe to move the def of the virtual register's class. e.g.
    // X86::RFP* register classes. Do not add it as a live-in.
    if (!TII->isSafeToMoveRegClassDefs(RC))
      // This is no longer available.
      NotAvailable.insert(Reg);
    else {
      MBB.addLiveIn(Reg);
      InvalidateKill(Reg, TRI, RegKills, KillOps);
    }

    // Skip over the same register.
    std::multimap<unsigned, int>::iterator NI = llvm::next(I);
    while (NI != E && NI->first == Reg) {
      ++I;
      ++NI;
    }
  }

  for (std::set<unsigned>::iterator I = NotAvailable.begin(),
         E = NotAvailable.end(); I != E; ++I) {
    ClobberPhysReg(*I);
    for (const unsigned *SubRegs = TRI->getSubRegisters(*I);
       *SubRegs; ++SubRegs)
      ClobberPhysReg(*SubRegs);
  }
}

/// ModifyStackSlotOrReMat - This method is called when the value in a stack
/// slot changes.  This removes information about which register the previous
/// value for this slot lives in (as the previous value is dead now).
void AvailableSpills::ModifyStackSlotOrReMat(int SlotOrReMat) {
  std::map<int, unsigned>::iterator It =
    SpillSlotsOrReMatsAvailable.find(SlotOrReMat);
  if (It == SpillSlotsOrReMatsAvailable.end()) return;
  unsigned Reg = It->second >> 1;
  SpillSlotsOrReMatsAvailable.erase(It);

  // This register may hold the value of multiple stack slots, only remove this
  // stack slot from the set of values the register contains.
  std::multimap<unsigned, int>::iterator I = PhysRegsAvailable.lower_bound(Reg);
  for (; ; ++I) {
    assert(I != PhysRegsAvailable.end() && I->first == Reg &&
           "Map inverse broken!");
    if (I->second == SlotOrReMat) break;
  }
  PhysRegsAvailable.erase(I);
}

// ************************** //
// Reuse Info Implementation  //
// ************************** //

/// GetRegForReload - We are about to emit a reload into PhysReg.  If there
/// is some other operand that is using the specified register, either pick
/// a new register to use, or evict the previous reload and use this reg.
unsigned ReuseInfo::GetRegForReload(const TargetRegisterClass *RC,
                         unsigned PhysReg,
                         MachineFunction &MF,
                         MachineInstr *MI, AvailableSpills &Spills,
                         std::vector<MachineInstr*> &MaybeDeadStores,
                         SmallSet<unsigned, 8> &Rejected,
                         BitVector &RegKills,
                         std::vector<MachineOperand*> &KillOps,
                         VirtRegMap &VRM) {
  const TargetInstrInfo* TII = MF.getTarget().getInstrInfo();
  const TargetRegisterInfo *TRI = Spills.getRegInfo();

  if (Reuses.empty()) return PhysReg;  // This is most often empty.

  for (unsigned ro = 0, e = Reuses.size(); ro != e; ++ro) {
    ReusedOp &Op = Reuses[ro];
    // If we find some other reuse that was supposed to use this register
    // exactly for its reload, we can change this reload to use ITS reload
    // register. That is, unless its reload register has already been
    // considered and subsequently rejected because it has also been reused
    // by another operand.
    if (Op.PhysRegReused == PhysReg &&
        Rejected.count(Op.AssignedPhysReg) == 0 &&
        RC->contains(Op.AssignedPhysReg)) {
      // Yup, use the reload register that we didn't use before.
      unsigned NewReg = Op.AssignedPhysReg;
      Rejected.insert(PhysReg);
      return GetRegForReload(RC, NewReg, MF, MI, Spills, MaybeDeadStores,
                             Rejected, RegKills, KillOps, VRM);
    } else {
      // Otherwise, we might also have a problem if a previously reused
      // value aliases the new register. If so, codegen the previous reload
      // and use this one.
      unsigned PRRU = Op.PhysRegReused;
      if (TRI->regsOverlap(PRRU, PhysReg)) {
        // Okay, we found out that an alias of a reused register
        // was used.  This isn't good because it means we have
        // to undo a previous reuse.
        MachineBasicBlock *MBB = MI->getParent();
        const TargetRegisterClass *AliasRC =
          MBB->getParent()->getRegInfo().getRegClass(Op.VirtReg);

        // Copy Op out of the vector and remove it, we're going to insert an
        // explicit load for it.
        ReusedOp NewOp = Op;
        Reuses.erase(Reuses.begin()+ro);

        // MI may be using only a sub-register of PhysRegUsed.
        unsigned RealPhysRegUsed = MI->getOperand(NewOp.Operand).getReg();
        unsigned SubIdx = 0;
        assert(TargetRegisterInfo::isPhysicalRegister(RealPhysRegUsed) &&
               "A reuse cannot be a virtual register");
        if (PRRU != RealPhysRegUsed) {
          // What was the sub-register index?
          SubIdx = TRI->getSubRegIndex(PRRU, RealPhysRegUsed);
          assert(SubIdx &&
                 "Operand physreg is not a sub-register of PhysRegUsed");
        }

        // Ok, we're going to try to reload the assigned physreg into the
        // slot that we were supposed to in the first place.  However, that
        // register could hold a reuse.  Check to see if it conflicts or
        // would prefer us to use a different register.
        unsigned NewPhysReg = GetRegForReload(RC, NewOp.AssignedPhysReg,
                                              MF, MI, Spills, MaybeDeadStores,
                                              Rejected, RegKills, KillOps, VRM);

        bool DoReMat = NewOp.StackSlotOrReMat > VirtRegMap::MAX_STACK_SLOT;
        int SSorRMId = DoReMat
          ? VRM.getReMatId(NewOp.VirtReg) : (int) NewOp.StackSlotOrReMat;

        // Back-schedule reloads and remats.
        MachineBasicBlock::iterator InsertLoc =
          ComputeReloadLoc(MI, MBB->begin(), PhysReg, TRI,
                           DoReMat, SSorRMId, TII, MF);

        if (DoReMat) {
          ReMaterialize(*MBB, InsertLoc, NewPhysReg, NewOp.VirtReg, TII,
                        TRI, VRM);
        } else {
          TII->loadRegFromStackSlot(*MBB, InsertLoc, NewPhysReg,
                                    NewOp.StackSlotOrReMat, AliasRC, TRI);
          MachineInstr *LoadMI = prior(InsertLoc);
          VRM.addSpillSlotUse(NewOp.StackSlotOrReMat, LoadMI);
          // Any stores to this stack slot are not dead anymore.
          MaybeDeadStores[NewOp.StackSlotOrReMat] = NULL;
          ++NumLoads;
        }
        Spills.ClobberPhysReg(NewPhysReg);
        Spills.ClobberPhysReg(NewOp.PhysRegReused);

        unsigned RReg = SubIdx ? TRI->getSubReg(NewPhysReg, SubIdx) :NewPhysReg;
        MI->getOperand(NewOp.Operand).setReg(RReg);
        MI->getOperand(NewOp.Operand).setSubReg(0);

        Spills.addAvailable(NewOp.StackSlotOrReMat, NewPhysReg);
        UpdateKills(*prior(InsertLoc), TRI, RegKills, KillOps);
        DEBUG(dbgs() << '\t' << *prior(InsertLoc));

        DEBUG(dbgs() << "Reuse undone!\n");
        --NumReused;

        // Finally, PhysReg is now available, go ahead and use it.
        return PhysReg;
      }
    }
  }
  return PhysReg;
}

// ************************************************************************ //

/// FoldsStackSlotModRef - Return true if the specified MI folds the specified
/// stack slot mod/ref. It also checks if it's possible to unfold the
/// instruction by having it define a specified physical register instead.
static bool FoldsStackSlotModRef(MachineInstr &MI, int SS, unsigned PhysReg,
                                 const TargetInstrInfo *TII,
                                 const TargetRegisterInfo *TRI,
                                 VirtRegMap &VRM) {
  if (VRM.hasEmergencySpills(&MI) || VRM.isSpillPt(&MI))
    return false;

  bool Found = false;
  VirtRegMap::MI2VirtMapTy::const_iterator I, End;
  for (tie(I, End) = VRM.getFoldedVirts(&MI); I != End; ++I) {
    unsigned VirtReg = I->second.first;
    VirtRegMap::ModRef MR = I->second.second;
    if (MR & VirtRegMap::isModRef)
      if (VRM.getStackSlot(VirtReg) == SS) {
        Found= TII->getOpcodeAfterMemoryUnfold(MI.getOpcode(), true, true) != 0;
        break;
      }
  }
  if (!Found)
    return false;

  // Does the instruction uses a register that overlaps the scratch register?
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || MO.getReg() == 0)
      continue;
    unsigned Reg = MO.getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg)) {
      if (!VRM.hasPhys(Reg))
        continue;
      Reg = VRM.getPhys(Reg);
    }
    if (TRI->regsOverlap(PhysReg, Reg))
      return false;
  }
  return true;
}

/// FindFreeRegister - Find a free register of a given register class by looking
/// at (at most) the last two machine instructions.
static unsigned FindFreeRegister(MachineBasicBlock::iterator MII,
                                 MachineBasicBlock &MBB,
                                 const TargetRegisterClass *RC,
                                 const TargetRegisterInfo *TRI,
                                 BitVector &AllocatableRegs) {
  BitVector Defs(TRI->getNumRegs());
  BitVector Uses(TRI->getNumRegs());
  SmallVector<unsigned, 4> LocalUses;
  SmallVector<unsigned, 4> Kills;

  // Take a look at 2 instructions at most.
  unsigned Count = 0;
  while (Count < 2) {
    if (MII == MBB.begin())
      break;
    MachineInstr *PrevMI = prior(MII);
    MII = PrevMI;

    if (PrevMI->isDebugValue())
      continue; // Skip over dbg_value instructions.
    ++Count;

    for (unsigned i = 0, e = PrevMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = PrevMI->getOperand(i);
      if (!MO.isReg() || MO.getReg() == 0)
        continue;
      unsigned Reg = MO.getReg();
      if (MO.isDef()) {
        Defs.set(Reg);
        for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS)
          Defs.set(*AS);
      } else  {
        LocalUses.push_back(Reg);
        if (MO.isKill() && AllocatableRegs[Reg])
          Kills.push_back(Reg);
      }
    }

    for (unsigned i = 0, e = Kills.size(); i != e; ++i) {
      unsigned Kill = Kills[i];
      if (!Defs[Kill] && !Uses[Kill] &&
          RC->contains(Kill))
        return Kill;
    }
    for (unsigned i = 0, e = LocalUses.size(); i != e; ++i) {
      unsigned Reg = LocalUses[i];
      Uses.set(Reg);
      for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS)
        Uses.set(*AS);
    }
  }

  return 0;
}

static
void AssignPhysToVirtReg(MachineInstr *MI, unsigned VirtReg, unsigned PhysReg,
                         const TargetRegisterInfo &TRI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == VirtReg)
      substitutePhysReg(MO, PhysReg, TRI);
  }
}

namespace {

struct RefSorter {
  bool operator()(const std::pair<MachineInstr*, int> &A,
                  const std::pair<MachineInstr*, int> &B) {
    return A.second < B.second;
  }
};

// ***************************** //
// Local Spiller Implementation  //
// ***************************** //

class LocalRewriter : public VirtRegRewriter {
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  VirtRegMap *VRM;
  BitVector AllocatableRegs;
  DenseMap<MachineInstr*, unsigned> DistanceMap;
  DenseMap<int, SmallVector<MachineInstr*,4> > Slot2DbgValues;

  MachineBasicBlock *MBB;       // Basic block currently being processed.

public:

  bool runOnMachineFunction(MachineFunction &MF, VirtRegMap &VRM,
                            LiveIntervals* LIs);

private:

  bool OptimizeByUnfold2(unsigned VirtReg, int SS,
                         MachineBasicBlock::iterator &MII,
                         std::vector<MachineInstr*> &MaybeDeadStores,
                         AvailableSpills &Spills,
                         BitVector &RegKills,
                         std::vector<MachineOperand*> &KillOps);

  bool OptimizeByUnfold(MachineBasicBlock::iterator &MII,
                        std::vector<MachineInstr*> &MaybeDeadStores,
                        AvailableSpills &Spills,
                        BitVector &RegKills,
                        std::vector<MachineOperand*> &KillOps);

  bool CommuteToFoldReload(MachineBasicBlock::iterator &MII,
                           unsigned VirtReg, unsigned SrcReg, int SS,
                           AvailableSpills &Spills,
                           BitVector &RegKills,
                           std::vector<MachineOperand*> &KillOps,
                           const TargetRegisterInfo *TRI);

  void SpillRegToStackSlot(MachineBasicBlock::iterator &MII,
                           int Idx, unsigned PhysReg, int StackSlot,
                           const TargetRegisterClass *RC,
                           bool isAvailable, MachineInstr *&LastStore,
                           AvailableSpills &Spills,
                           SmallSet<MachineInstr*, 4> &ReMatDefs,
                           BitVector &RegKills,
                           std::vector<MachineOperand*> &KillOps);

  void TransferDeadness(unsigned Reg, BitVector &RegKills,
                        std::vector<MachineOperand*> &KillOps);

  bool InsertEmergencySpills(MachineInstr *MI);

  bool InsertRestores(MachineInstr *MI,
                      AvailableSpills &Spills,
                      BitVector &RegKills,
                      std::vector<MachineOperand*> &KillOps);

  bool InsertSpills(MachineInstr *MI);

  void RewriteMBB(LiveIntervals *LIs,
                  AvailableSpills &Spills, BitVector &RegKills,
                  std::vector<MachineOperand*> &KillOps);
};
}

bool LocalRewriter::runOnMachineFunction(MachineFunction &MF, VirtRegMap &vrm,
                                         LiveIntervals* LIs) {
  MRI = &MF.getRegInfo();
  TRI = MF.getTarget().getRegisterInfo();
  TII = MF.getTarget().getInstrInfo();
  VRM = &vrm;
  AllocatableRegs = TRI->getAllocatableSet(MF);
  DEBUG(dbgs() << "\n**** Local spiller rewriting function '"
        << MF.getFunction()->getName() << "':\n");
  DEBUG(dbgs() << "**** Machine Instrs (NOTE! Does not include spills and"
        " reloads!) ****\n");
  DEBUG(MF.dump());

  // Spills - Keep track of which spilled values are available in physregs
  // so that we can choose to reuse the physregs instead of emitting
  // reloads. This is usually refreshed per basic block.
  AvailableSpills Spills(TRI, TII);

  // Keep track of kill information.
  BitVector RegKills(TRI->getNumRegs());
  std::vector<MachineOperand*> KillOps;
  KillOps.resize(TRI->getNumRegs(), NULL);

  // SingleEntrySuccs - Successor blocks which have a single predecessor.
  SmallVector<MachineBasicBlock*, 4> SinglePredSuccs;
  SmallPtrSet<MachineBasicBlock*,16> EarlyVisited;

  // Traverse the basic blocks depth first.
  MachineBasicBlock *Entry = MF.begin();
  SmallPtrSet<MachineBasicBlock*,16> Visited;
  for (df_ext_iterator<MachineBasicBlock*,
         SmallPtrSet<MachineBasicBlock*,16> >
         DFI = df_ext_begin(Entry, Visited), E = df_ext_end(Entry, Visited);
       DFI != E; ++DFI) {
    MBB = *DFI;
    if (!EarlyVisited.count(MBB))
      RewriteMBB(LIs, Spills, RegKills, KillOps);

    // If this MBB is the only predecessor of a successor. Keep the
    // availability information and visit it next.
    do {
      // Keep visiting single predecessor successor as long as possible.
      SinglePredSuccs.clear();
      findSinglePredSuccessor(MBB, SinglePredSuccs);
      if (SinglePredSuccs.empty())
        MBB = 0;
      else {
        // FIXME: More than one successors, each of which has MBB has
        // the only predecessor.
        MBB = SinglePredSuccs[0];
        if (!Visited.count(MBB) && EarlyVisited.insert(MBB)) {
          Spills.AddAvailableRegsToLiveIn(*MBB, RegKills, KillOps);
          RewriteMBB(LIs, Spills, RegKills, KillOps);
        }
      }
    } while (MBB);

    // Clear the availability info.
    Spills.clear();
  }

  DEBUG(dbgs() << "**** Post Machine Instrs ****\n");
  DEBUG(MF.dump());

  // Mark unused spill slots.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  int SS = VRM->getLowSpillSlot();
  if (SS != VirtRegMap::NO_STACK_SLOT) {
    for (int e = VRM->getHighSpillSlot(); SS <= e; ++SS) {
      SmallVector<MachineInstr*, 4> &DbgValues = Slot2DbgValues[SS];
      if (!VRM->isSpillSlotUsed(SS)) {
        MFI->RemoveStackObject(SS);
        for (unsigned j = 0, ee = DbgValues.size(); j != ee; ++j) {
          MachineInstr *DVMI = DbgValues[j];
          MachineBasicBlock *DVMBB = DVMI->getParent();
          DEBUG(dbgs() << "Removing debug info referencing FI#" << SS << '\n');
          VRM->RemoveMachineInstrFromMaps(DVMI);
          DVMBB->erase(DVMI);
        }
        ++NumDSS;
      }
      DbgValues.clear();
    }
  }
  Slot2DbgValues.clear();

  return true;
}

/// OptimizeByUnfold2 - Unfold a series of load / store folding instructions if
/// a scratch register is available.
///     xorq  %r12<kill>, %r13
///     addq  %rax, -184(%rbp)
///     addq  %r13, -184(%rbp)
/// ==>
///     xorq  %r12<kill>, %r13
///     movq  -184(%rbp), %r12
///     addq  %rax, %r12
///     addq  %r13, %r12
///     movq  %r12, -184(%rbp)
bool LocalRewriter::
OptimizeByUnfold2(unsigned VirtReg, int SS,
                  MachineBasicBlock::iterator &MII,
                  std::vector<MachineInstr*> &MaybeDeadStores,
                  AvailableSpills &Spills,
                  BitVector &RegKills,
                  std::vector<MachineOperand*> &KillOps) {

  MachineBasicBlock::iterator NextMII = llvm::next(MII);
  // Skip over dbg_value instructions.
  while (NextMII != MBB->end() && NextMII->isDebugValue())
    NextMII = llvm::next(NextMII);
  if (NextMII == MBB->end())
    return false;

  if (TII->getOpcodeAfterMemoryUnfold(MII->getOpcode(), true, true) == 0)
    return false;

  // Now let's see if the last couple of instructions happens to have freed up
  // a register.
  const TargetRegisterClass* RC = MRI->getRegClass(VirtReg);
  unsigned PhysReg = FindFreeRegister(MII, *MBB, RC, TRI, AllocatableRegs);
  if (!PhysReg)
    return false;

  MachineFunction &MF = *MBB->getParent();
  TRI = MF.getTarget().getRegisterInfo();
  MachineInstr &MI = *MII;
  if (!FoldsStackSlotModRef(MI, SS, PhysReg, TII, TRI, *VRM))
    return false;

  // If the next instruction also folds the same SS modref and can be unfoled,
  // then it's worthwhile to issue a load from SS into the free register and
  // then unfold these instructions.
  if (!FoldsStackSlotModRef(*NextMII, SS, PhysReg, TII, TRI, *VRM))
    return false;

  // Back-schedule reloads and remats.
  ComputeReloadLoc(MII, MBB->begin(), PhysReg, TRI, false, SS, TII, MF);

  // Load from SS to the spare physical register.
  TII->loadRegFromStackSlot(*MBB, MII, PhysReg, SS, RC, TRI);
  // This invalidates Phys.
  Spills.ClobberPhysReg(PhysReg);
  // Remember it's available.
  Spills.addAvailable(SS, PhysReg);
  MaybeDeadStores[SS] = NULL;

  // Unfold current MI.
  SmallVector<MachineInstr*, 4> NewMIs;
  if (!TII->unfoldMemoryOperand(MF, &MI, VirtReg, false, false, NewMIs))
    llvm_unreachable("Unable unfold the load / store folding instruction!");
  assert(NewMIs.size() == 1);
  AssignPhysToVirtReg(NewMIs[0], VirtReg, PhysReg, *TRI);
  VRM->transferRestorePts(&MI, NewMIs[0]);
  MII = MBB->insert(MII, NewMIs[0]);
  InvalidateKills(MI, TRI, RegKills, KillOps);
  VRM->RemoveMachineInstrFromMaps(&MI);
  MBB->erase(&MI);
  ++NumModRefUnfold;

  // Unfold next instructions that fold the same SS.
  do {
    MachineInstr &NextMI = *NextMII;
    NextMII = llvm::next(NextMII);
    NewMIs.clear();
    if (!TII->unfoldMemoryOperand(MF, &NextMI, VirtReg, false, false, NewMIs))
      llvm_unreachable("Unable unfold the load / store folding instruction!");
    assert(NewMIs.size() == 1);
    AssignPhysToVirtReg(NewMIs[0], VirtReg, PhysReg, *TRI);
    VRM->transferRestorePts(&NextMI, NewMIs[0]);
    MBB->insert(NextMII, NewMIs[0]);
    InvalidateKills(NextMI, TRI, RegKills, KillOps);
    VRM->RemoveMachineInstrFromMaps(&NextMI);
    MBB->erase(&NextMI);
    ++NumModRefUnfold;
    // Skip over dbg_value instructions.
    while (NextMII != MBB->end() && NextMII->isDebugValue())
      NextMII = llvm::next(NextMII);
    if (NextMII == MBB->end())
      break;
  } while (FoldsStackSlotModRef(*NextMII, SS, PhysReg, TII, TRI, *VRM));

  // Store the value back into SS.
  TII->storeRegToStackSlot(*MBB, NextMII, PhysReg, true, SS, RC, TRI);
  MachineInstr *StoreMI = prior(NextMII);
  VRM->addSpillSlotUse(SS, StoreMI);
  VRM->virtFolded(VirtReg, StoreMI, VirtRegMap::isMod);

  return true;
}

/// OptimizeByUnfold - Turn a store folding instruction into a load folding
/// instruction. e.g.
///     xorl  %edi, %eax
///     movl  %eax, -32(%ebp)
///     movl  -36(%ebp), %eax
///     orl   %eax, -32(%ebp)
/// ==>
///     xorl  %edi, %eax
///     orl   -36(%ebp), %eax
///     mov   %eax, -32(%ebp)
/// This enables unfolding optimization for a subsequent instruction which will
/// also eliminate the newly introduced store instruction.
bool LocalRewriter::
OptimizeByUnfold(MachineBasicBlock::iterator &MII,
                 std::vector<MachineInstr*> &MaybeDeadStores,
                 AvailableSpills &Spills,
                 BitVector &RegKills,
                 std::vector<MachineOperand*> &KillOps) {
  MachineFunction &MF = *MBB->getParent();
  MachineInstr &MI = *MII;
  unsigned UnfoldedOpc = 0;
  unsigned UnfoldPR = 0;
  unsigned UnfoldVR = 0;
  int FoldedSS = VirtRegMap::NO_STACK_SLOT;
  VirtRegMap::MI2VirtMapTy::const_iterator I, End;
  for (tie(I, End) = VRM->getFoldedVirts(&MI); I != End; ) {
    // Only transform a MI that folds a single register.
    if (UnfoldedOpc)
      return false;
    UnfoldVR = I->second.first;
    VirtRegMap::ModRef MR = I->second.second;
    // MI2VirtMap be can updated which invalidate the iterator.
    // Increment the iterator first.
    ++I;
    if (VRM->isAssignedReg(UnfoldVR))
      continue;
    // If this reference is not a use, any previous store is now dead.
    // Otherwise, the store to this stack slot is not dead anymore.
    FoldedSS = VRM->getStackSlot(UnfoldVR);
    MachineInstr* DeadStore = MaybeDeadStores[FoldedSS];
    if (DeadStore && (MR & VirtRegMap::isModRef)) {
      unsigned PhysReg = Spills.getSpillSlotOrReMatPhysReg(FoldedSS);
      if (!PhysReg || !DeadStore->readsRegister(PhysReg))
        continue;
      UnfoldPR = PhysReg;
      UnfoldedOpc = TII->getOpcodeAfterMemoryUnfold(MI.getOpcode(),
                                                    false, true);
    }
  }

  if (!UnfoldedOpc) {
    if (!UnfoldVR)
      return false;

    // Look for other unfolding opportunities.
    return OptimizeByUnfold2(UnfoldVR, FoldedSS, MII, MaybeDeadStores, Spills,
                             RegKills, KillOps);
  }

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || MO.getReg() == 0 || !MO.isUse())
      continue;
    unsigned VirtReg = MO.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(VirtReg) || MO.getSubReg())
      continue;
    if (VRM->isAssignedReg(VirtReg)) {
      unsigned PhysReg = VRM->getPhys(VirtReg);
      if (PhysReg && TRI->regsOverlap(PhysReg, UnfoldPR))
        return false;
    } else if (VRM->isReMaterialized(VirtReg))
      continue;
    int SS = VRM->getStackSlot(VirtReg);
    unsigned PhysReg = Spills.getSpillSlotOrReMatPhysReg(SS);
    if (PhysReg) {
      if (TRI->regsOverlap(PhysReg, UnfoldPR))
        return false;
      continue;
    }
    if (VRM->hasPhys(VirtReg)) {
      PhysReg = VRM->getPhys(VirtReg);
      if (!TRI->regsOverlap(PhysReg, UnfoldPR))
        continue;
    }

    // Ok, we'll need to reload the value into a register which makes
    // it impossible to perform the store unfolding optimization later.
    // Let's see if it is possible to fold the load if the store is
    // unfolded. This allows us to perform the store unfolding
    // optimization.
    SmallVector<MachineInstr*, 4> NewMIs;
    if (TII->unfoldMemoryOperand(MF, &MI, UnfoldVR, false, false, NewMIs)) {
      assert(NewMIs.size() == 1);
      MachineInstr *NewMI = NewMIs.back();
      MBB->insert(MII, NewMI);
      NewMIs.clear();
      int Idx = NewMI->findRegisterUseOperandIdx(VirtReg, false);
      assert(Idx != -1);
      SmallVector<unsigned, 1> Ops;
      Ops.push_back(Idx);
      MachineInstr *FoldedMI = TII->foldMemoryOperand(NewMI, Ops, SS);
      NewMI->eraseFromParent();
      if (FoldedMI) {
        VRM->addSpillSlotUse(SS, FoldedMI);
        if (!VRM->hasPhys(UnfoldVR))
          VRM->assignVirt2Phys(UnfoldVR, UnfoldPR);
        VRM->virtFolded(VirtReg, FoldedMI, VirtRegMap::isRef);
        MII = FoldedMI;
        InvalidateKills(MI, TRI, RegKills, KillOps);
        VRM->RemoveMachineInstrFromMaps(&MI);
        MBB->erase(&MI);
        return true;
      }
    }
  }

  return false;
}

/// CommuteChangesDestination - We are looking for r0 = op r1, r2 and
/// where SrcReg is r1 and it is tied to r0. Return true if after
/// commuting this instruction it will be r0 = op r2, r1.
static bool CommuteChangesDestination(MachineInstr *DefMI,
                                      const TargetInstrDesc &TID,
                                      unsigned SrcReg,
                                      const TargetInstrInfo *TII,
                                      unsigned &DstIdx) {
  if (TID.getNumDefs() != 1 && TID.getNumOperands() != 3)
    return false;
  if (!DefMI->getOperand(1).isReg() ||
      DefMI->getOperand(1).getReg() != SrcReg)
    return false;
  unsigned DefIdx;
  if (!DefMI->isRegTiedToDefOperand(1, &DefIdx) || DefIdx != 0)
    return false;
  unsigned SrcIdx1, SrcIdx2;
  if (!TII->findCommutedOpIndices(DefMI, SrcIdx1, SrcIdx2))
    return false;
  if (SrcIdx1 == 1 && SrcIdx2 == 2) {
    DstIdx = 2;
    return true;
  }
  return false;
}

/// CommuteToFoldReload -
/// Look for
/// r1 = load fi#1
/// r1 = op r1, r2<kill>
/// store r1, fi#1
///
/// If op is commutable and r2 is killed, then we can xform these to
/// r2 = op r2, fi#1
/// store r2, fi#1
bool LocalRewriter::
CommuteToFoldReload(MachineBasicBlock::iterator &MII,
                    unsigned VirtReg, unsigned SrcReg, int SS,
                    AvailableSpills &Spills,
                    BitVector &RegKills,
                    std::vector<MachineOperand*> &KillOps,
                    const TargetRegisterInfo *TRI) {
  if (MII == MBB->begin() || !MII->killsRegister(SrcReg))
    return false;

  MachineInstr &MI = *MII;
  MachineBasicBlock::iterator DefMII = prior(MII);
  MachineInstr *DefMI = DefMII;
  const TargetInstrDesc &TID = DefMI->getDesc();
  unsigned NewDstIdx;
  if (DefMII != MBB->begin() &&
      TID.isCommutable() &&
      CommuteChangesDestination(DefMI, TID, SrcReg, TII, NewDstIdx)) {
    MachineOperand &NewDstMO = DefMI->getOperand(NewDstIdx);
    unsigned NewReg = NewDstMO.getReg();
    if (!NewDstMO.isKill() || TRI->regsOverlap(NewReg, SrcReg))
      return false;
    MachineInstr *ReloadMI = prior(DefMII);
    int FrameIdx;
    unsigned DestReg = TII->isLoadFromStackSlot(ReloadMI, FrameIdx);
    if (DestReg != SrcReg || FrameIdx != SS)
      return false;
    int UseIdx = DefMI->findRegisterUseOperandIdx(DestReg, false);
    if (UseIdx == -1)
      return false;
    unsigned DefIdx;
    if (!MI.isRegTiedToDefOperand(UseIdx, &DefIdx))
      return false;
    assert(DefMI->getOperand(DefIdx).isReg() &&
           DefMI->getOperand(DefIdx).getReg() == SrcReg);

    // Now commute def instruction.
    MachineInstr *CommutedMI = TII->commuteInstruction(DefMI, true);
    if (!CommutedMI)
      return false;
    MBB->insert(MII, CommutedMI);
    SmallVector<unsigned, 1> Ops;
    Ops.push_back(NewDstIdx);
    MachineInstr *FoldedMI = TII->foldMemoryOperand(CommutedMI, Ops, SS);
    // Not needed since foldMemoryOperand returns new MI.
    CommutedMI->eraseFromParent();
    if (!FoldedMI)
      return false;

    VRM->addSpillSlotUse(SS, FoldedMI);
    VRM->virtFolded(VirtReg, FoldedMI, VirtRegMap::isRef);
    // Insert new def MI and spill MI.
    const TargetRegisterClass* RC = MRI->getRegClass(VirtReg);
    TII->storeRegToStackSlot(*MBB, &MI, NewReg, true, SS, RC, TRI);
    MII = prior(MII);
    MachineInstr *StoreMI = MII;
    VRM->addSpillSlotUse(SS, StoreMI);
    VRM->virtFolded(VirtReg, StoreMI, VirtRegMap::isMod);
    MII = FoldedMI;  // Update MII to backtrack.

    // Delete all 3 old instructions.
    InvalidateKills(*ReloadMI, TRI, RegKills, KillOps);
    VRM->RemoveMachineInstrFromMaps(ReloadMI);
    MBB->erase(ReloadMI);
    InvalidateKills(*DefMI, TRI, RegKills, KillOps);
    VRM->RemoveMachineInstrFromMaps(DefMI);
    MBB->erase(DefMI);
    InvalidateKills(MI, TRI, RegKills, KillOps);
    VRM->RemoveMachineInstrFromMaps(&MI);
    MBB->erase(&MI);

    // If NewReg was previously holding value of some SS, it's now clobbered.
    // This has to be done now because it's a physical register. When this
    // instruction is re-visited, it's ignored.
    Spills.ClobberPhysReg(NewReg);

    ++NumCommutes;
    return true;
  }

  return false;
}

/// SpillRegToStackSlot - Spill a register to a specified stack slot. Check if
/// the last store to the same slot is now dead. If so, remove the last store.
void LocalRewriter::
SpillRegToStackSlot(MachineBasicBlock::iterator &MII,
                    int Idx, unsigned PhysReg, int StackSlot,
                    const TargetRegisterClass *RC,
                    bool isAvailable, MachineInstr *&LastStore,
                    AvailableSpills &Spills,
                    SmallSet<MachineInstr*, 4> &ReMatDefs,
                    BitVector &RegKills,
                    std::vector<MachineOperand*> &KillOps) {

  MachineBasicBlock::iterator oldNextMII = llvm::next(MII);
  TII->storeRegToStackSlot(*MBB, llvm::next(MII), PhysReg, true, StackSlot, RC,
                           TRI);
  MachineInstr *StoreMI = prior(oldNextMII);
  VRM->addSpillSlotUse(StackSlot, StoreMI);
  DEBUG(dbgs() << "Store:\t" << *StoreMI);

  // If there is a dead store to this stack slot, nuke it now.
  if (LastStore) {
    DEBUG(dbgs() << "Removed dead store:\t" << *LastStore);
    ++NumDSE;
    SmallVector<unsigned, 2> KillRegs;
    InvalidateKills(*LastStore, TRI, RegKills, KillOps, &KillRegs);
    MachineBasicBlock::iterator PrevMII = LastStore;
    bool CheckDef = PrevMII != MBB->begin();
    if (CheckDef)
      --PrevMII;
    VRM->RemoveMachineInstrFromMaps(LastStore);
    MBB->erase(LastStore);
    if (CheckDef) {
      // Look at defs of killed registers on the store. Mark the defs
      // as dead since the store has been deleted and they aren't
      // being reused.
      for (unsigned j = 0, ee = KillRegs.size(); j != ee; ++j) {
        bool HasOtherDef = false;
        if (InvalidateRegDef(PrevMII, *MII, KillRegs[j], HasOtherDef, TRI)) {
          MachineInstr *DeadDef = PrevMII;
          if (ReMatDefs.count(DeadDef) && !HasOtherDef) {
            // FIXME: This assumes a remat def does not have side effects.
            VRM->RemoveMachineInstrFromMaps(DeadDef);
            MBB->erase(DeadDef);
            ++NumDRM;
          }
        }
      }
    }
  }

  // Allow for multi-instruction spill sequences, as on PPC Altivec.  Presume
  // the last of multiple instructions is the actual store.
  LastStore = prior(oldNextMII);

  // If the stack slot value was previously available in some other
  // register, change it now.  Otherwise, make the register available,
  // in PhysReg.
  Spills.ModifyStackSlotOrReMat(StackSlot);
  Spills.ClobberPhysReg(PhysReg);
  Spills.addAvailable(StackSlot, PhysReg, isAvailable);
  ++NumStores;
}

/// isSafeToDelete - Return true if this instruction doesn't produce any side
/// effect and all of its defs are dead.
static bool isSafeToDelete(MachineInstr &MI) {
  const TargetInstrDesc &TID = MI.getDesc();
  if (TID.mayLoad() || TID.mayStore() || TID.isCall() || TID.isTerminator() ||
      TID.isCall() || TID.isBarrier() || TID.isReturn() ||
      TID.hasUnmodeledSideEffects())
    return false;
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MO.isDef() && !MO.isDead())
      return false;
    if (MO.isUse() && MO.isKill())
      // FIXME: We can't remove kill markers or else the scavenger will assert.
      // An alternative is to add a ADD pseudo instruction to replace kill
      // markers.
      return false;
  }
  return true;
}

/// TransferDeadness - A identity copy definition is dead and it's being
/// removed. Find the last def or use and mark it as dead / kill.
void LocalRewriter::
TransferDeadness(unsigned Reg, BitVector &RegKills,
                 std::vector<MachineOperand*> &KillOps) {
  SmallPtrSet<MachineInstr*, 4> Seens;
  SmallVector<std::pair<MachineInstr*, int>,8> Refs;
  for (MachineRegisterInfo::reg_iterator RI = MRI->reg_begin(Reg),
         RE = MRI->reg_end(); RI != RE; ++RI) {
    MachineInstr *UDMI = &*RI;
    if (UDMI->isDebugValue() || UDMI->getParent() != MBB)
      continue;
    DenseMap<MachineInstr*, unsigned>::iterator DI = DistanceMap.find(UDMI);
    if (DI == DistanceMap.end())
      continue;
    if (Seens.insert(UDMI))
      Refs.push_back(std::make_pair(UDMI, DI->second));
  }

  if (Refs.empty())
    return;
  std::sort(Refs.begin(), Refs.end(), RefSorter());

  while (!Refs.empty()) {
    MachineInstr *LastUDMI = Refs.back().first;
    Refs.pop_back();

    MachineOperand *LastUD = NULL;
    for (unsigned i = 0, e = LastUDMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = LastUDMI->getOperand(i);
      if (!MO.isReg() || MO.getReg() != Reg)
        continue;
      if (!LastUD || (LastUD->isUse() && MO.isDef()))
        LastUD = &MO;
      if (LastUDMI->isRegTiedToDefOperand(i))
        break;
    }
    if (LastUD->isDef()) {
      // If the instruction has no side effect, delete it and propagate
      // backward further. Otherwise, mark is dead and we are done.
      if (!isSafeToDelete(*LastUDMI)) {
        LastUD->setIsDead();
        break;
      }
      VRM->RemoveMachineInstrFromMaps(LastUDMI);
      MBB->erase(LastUDMI);
    } else {
      LastUD->setIsKill();
      RegKills.set(Reg);
      KillOps[Reg] = LastUD;
      break;
    }
  }
}

/// InsertEmergencySpills - Insert emergency spills before MI if requested by
/// VRM. Return true if spills were inserted.
bool LocalRewriter::InsertEmergencySpills(MachineInstr *MI) {
  if (!VRM->hasEmergencySpills(MI))
    return false;
  MachineBasicBlock::iterator MII = MI;
  SmallSet<int, 4> UsedSS;
  std::vector<unsigned> &EmSpills = VRM->getEmergencySpills(MI);
  for (unsigned i = 0, e = EmSpills.size(); i != e; ++i) {
    unsigned PhysReg = EmSpills[i];
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(PhysReg);
    assert(RC && "Unable to determine register class!");
    int SS = VRM->getEmergencySpillSlot(RC);
    if (UsedSS.count(SS))
      llvm_unreachable("Need to spill more than one physical registers!");
    UsedSS.insert(SS);
    TII->storeRegToStackSlot(*MBB, MII, PhysReg, true, SS, RC, TRI);
    MachineInstr *StoreMI = prior(MII);
    VRM->addSpillSlotUse(SS, StoreMI);

    // Back-schedule reloads and remats.
    MachineBasicBlock::iterator InsertLoc =
      ComputeReloadLoc(llvm::next(MII), MBB->begin(), PhysReg, TRI, false, SS,
                       TII, *MBB->getParent());

    TII->loadRegFromStackSlot(*MBB, InsertLoc, PhysReg, SS, RC, TRI);

    MachineInstr *LoadMI = prior(InsertLoc);
    VRM->addSpillSlotUse(SS, LoadMI);
    ++NumPSpills;
    DistanceMap.insert(std::make_pair(LoadMI, DistanceMap.size()));
  }
  return true;
}

/// InsertRestores - Restore registers before MI is requested by VRM. Return
/// true is any instructions were inserted.
bool LocalRewriter::InsertRestores(MachineInstr *MI,
                                   AvailableSpills &Spills,
                                   BitVector &RegKills,
                                   std::vector<MachineOperand*> &KillOps) {
  if (!VRM->isRestorePt(MI))
    return false;
  MachineBasicBlock::iterator MII = MI;
  std::vector<unsigned> &RestoreRegs = VRM->getRestorePtRestores(MI);
  for (unsigned i = 0, e = RestoreRegs.size(); i != e; ++i) {
    unsigned VirtReg = RestoreRegs[e-i-1];  // Reverse order.
    if (!VRM->getPreSplitReg(VirtReg))
      continue; // Split interval spilled again.
    unsigned Phys = VRM->getPhys(VirtReg);
    MRI->setPhysRegUsed(Phys);

    // Check if the value being restored if available. If so, it must be
    // from a predecessor BB that fallthrough into this BB. We do not
    // expect:
    // BB1:
    // r1 = load fi#1
    // ...
    //    = r1<kill>
    // ... # r1 not clobbered
    // ...
    //    = load fi#1
    bool DoReMat = VRM->isReMaterialized(VirtReg);
    int SSorRMId = DoReMat
      ? VRM->getReMatId(VirtReg) : VRM->getStackSlot(VirtReg);
    unsigned InReg = Spills.getSpillSlotOrReMatPhysReg(SSorRMId);
    if (InReg == Phys) {
      // If the value is already available in the expected register, save
      // a reload / remat.
      if (SSorRMId)
        DEBUG(dbgs() << "Reusing RM#"
                     << SSorRMId-VirtRegMap::MAX_STACK_SLOT-1);
      else
        DEBUG(dbgs() << "Reusing SS#" << SSorRMId);
      DEBUG(dbgs() << " from physreg "
                   << TRI->getName(InReg) << " for vreg"
                   << VirtReg <<" instead of reloading into physreg "
                   << TRI->getName(Phys) << '\n');
      ++NumOmitted;
      continue;
    } else if (InReg && InReg != Phys) {
      if (SSorRMId)
        DEBUG(dbgs() << "Reusing RM#"
                     << SSorRMId-VirtRegMap::MAX_STACK_SLOT-1);
      else
        DEBUG(dbgs() << "Reusing SS#" << SSorRMId);
      DEBUG(dbgs() << " from physreg "
                   << TRI->getName(InReg) << " for vreg"
                   << VirtReg <<" by copying it into physreg "
                   << TRI->getName(Phys) << '\n');

      // If the reloaded / remat value is available in another register,
      // copy it to the desired register.

      // Back-schedule reloads and remats.
      MachineBasicBlock::iterator InsertLoc =
        ComputeReloadLoc(MII, MBB->begin(), Phys, TRI, DoReMat, SSorRMId, TII,
                         *MBB->getParent());
      MachineInstr *CopyMI = BuildMI(*MBB, InsertLoc, MI->getDebugLoc(),
                                     TII->get(TargetOpcode::COPY), Phys)
                               .addReg(InReg, RegState::Kill);

      // This invalidates Phys.
      Spills.ClobberPhysReg(Phys);
      // Remember it's available.
      Spills.addAvailable(SSorRMId, Phys);

      CopyMI->setAsmPrinterFlag(MachineInstr::ReloadReuse);
      UpdateKills(*CopyMI, TRI, RegKills, KillOps);

      DEBUG(dbgs() << '\t' << *CopyMI);
      ++NumCopified;
      continue;
    }

    // Back-schedule reloads and remats.
    MachineBasicBlock::iterator InsertLoc =
      ComputeReloadLoc(MII, MBB->begin(), Phys, TRI, DoReMat, SSorRMId, TII,
                       *MBB->getParent());

    if (VRM->isReMaterialized(VirtReg)) {
      ReMaterialize(*MBB, InsertLoc, Phys, VirtReg, TII, TRI, *VRM);
    } else {
      const TargetRegisterClass* RC = MRI->getRegClass(VirtReg);
      TII->loadRegFromStackSlot(*MBB, InsertLoc, Phys, SSorRMId, RC, TRI);
      MachineInstr *LoadMI = prior(InsertLoc);
      VRM->addSpillSlotUse(SSorRMId, LoadMI);
      ++NumLoads;
      DistanceMap.insert(std::make_pair(LoadMI, DistanceMap.size()));
    }

    // This invalidates Phys.
    Spills.ClobberPhysReg(Phys);
    // Remember it's available.
    Spills.addAvailable(SSorRMId, Phys);

    UpdateKills(*prior(InsertLoc), TRI, RegKills, KillOps);
    DEBUG(dbgs() << '\t' << *prior(MII));
  }
  return true;
}

/// InsertEmergencySpills - Insert spills after MI if requested by VRM. Return
/// true if spills were inserted.
bool LocalRewriter::InsertSpills(MachineInstr *MI) {
  if (!VRM->isSpillPt(MI))
    return false;
  MachineBasicBlock::iterator MII = MI;
  std::vector<std::pair<unsigned,bool> > &SpillRegs =
    VRM->getSpillPtSpills(MI);
  for (unsigned i = 0, e = SpillRegs.size(); i != e; ++i) {
    unsigned VirtReg = SpillRegs[i].first;
    bool isKill = SpillRegs[i].second;
    if (!VRM->getPreSplitReg(VirtReg))
      continue; // Split interval spilled again.
    const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
    unsigned Phys = VRM->getPhys(VirtReg);
    int StackSlot = VRM->getStackSlot(VirtReg);
    MachineBasicBlock::iterator oldNextMII = llvm::next(MII);
    TII->storeRegToStackSlot(*MBB, llvm::next(MII), Phys, isKill, StackSlot,
                             RC, TRI);
    MachineInstr *StoreMI = prior(oldNextMII);
    VRM->addSpillSlotUse(StackSlot, StoreMI);
    DEBUG(dbgs() << "Store:\t" << *StoreMI);
    VRM->virtFolded(VirtReg, StoreMI, VirtRegMap::isMod);
  }
  return true;
}


/// rewriteMBB - Keep track of which spills are available even after the
/// register allocator is done with them.  If possible, avoid reloading vregs.
void
LocalRewriter::RewriteMBB(LiveIntervals *LIs,
                          AvailableSpills &Spills, BitVector &RegKills,
                          std::vector<MachineOperand*> &KillOps) {

  DEBUG(dbgs() << "\n**** Local spiller rewriting MBB '"
               << MBB->getName() << "':\n");

  MachineFunction &MF = *MBB->getParent();

  // MaybeDeadStores - When we need to write a value back into a stack slot,
  // keep track of the inserted store.  If the stack slot value is never read
  // (because the value was used from some available register, for example), and
  // subsequently stored to, the original store is dead.  This map keeps track
  // of inserted stores that are not used.  If we see a subsequent store to the
  // same stack slot, the original store is deleted.
  std::vector<MachineInstr*> MaybeDeadStores;
  MaybeDeadStores.resize(MF.getFrameInfo()->getObjectIndexEnd(), NULL);

  // ReMatDefs - These are rematerializable def MIs which are not deleted.
  SmallSet<MachineInstr*, 4> ReMatDefs;

  // Clear kill info.
  SmallSet<unsigned, 2> KilledMIRegs;

  // Keep track of the registers we have already spilled in case there are
  // multiple defs of the same register in MI.
  SmallSet<unsigned, 8> SpilledMIRegs;

  RegKills.reset();
  KillOps.clear();
  KillOps.resize(TRI->getNumRegs(), NULL);

  DistanceMap.clear();
  for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
       MII != E; ) {
    MachineBasicBlock::iterator NextMII = llvm::next(MII);

    if (OptimizeByUnfold(MII, MaybeDeadStores, Spills, RegKills, KillOps))
      NextMII = llvm::next(MII);

    if (InsertEmergencySpills(MII))
      NextMII = llvm::next(MII);

    InsertRestores(MII, Spills, RegKills, KillOps);

    if (InsertSpills(MII))
      NextMII = llvm::next(MII);

    bool Erased = false;
    bool BackTracked = false;
    MachineInstr &MI = *MII;

    // Remember DbgValue's which reference stack slots.
    if (MI.isDebugValue() && MI.getOperand(0).isFI())
      Slot2DbgValues[MI.getOperand(0).getIndex()].push_back(&MI);

    /// ReusedOperands - Keep track of operand reuse in case we need to undo
    /// reuse.
    ReuseInfo ReusedOperands(MI, TRI);
    SmallVector<unsigned, 4> VirtUseOps;
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI.getOperand(i);
      if (!MO.isReg() || MO.getReg() == 0)
        continue;   // Ignore non-register operands.

      unsigned VirtReg = MO.getReg();
      if (TargetRegisterInfo::isPhysicalRegister(VirtReg)) {
        // Ignore physregs for spilling, but remember that it is used by this
        // function.
        MRI->setPhysRegUsed(VirtReg);
        continue;
      }

      // We want to process implicit virtual register uses first.
      if (MO.isImplicit())
        // If the virtual register is implicitly defined, emit a implicit_def
        // before so scavenger knows it's "defined".
        // FIXME: This is a horrible hack done the by register allocator to
        // remat a definition with virtual register operand.
        VirtUseOps.insert(VirtUseOps.begin(), i);
      else
        VirtUseOps.push_back(i);
    }

    // Process all of the spilled uses and all non spilled reg references.
    SmallVector<int, 2> PotentialDeadStoreSlots;
    KilledMIRegs.clear();
    for (unsigned j = 0, e = VirtUseOps.size(); j != e; ++j) {
      unsigned i = VirtUseOps[j];
      unsigned VirtReg = MI.getOperand(i).getReg();
      assert(TargetRegisterInfo::isVirtualRegister(VirtReg) &&
             "Not a virtual register?");

      unsigned SubIdx = MI.getOperand(i).getSubReg();
      if (VRM->isAssignedReg(VirtReg)) {
        // This virtual register was assigned a physreg!
        unsigned Phys = VRM->getPhys(VirtReg);
        MRI->setPhysRegUsed(Phys);
        if (MI.getOperand(i).isDef())
          ReusedOperands.markClobbered(Phys);
        substitutePhysReg(MI.getOperand(i), Phys, *TRI);
        if (VRM->isImplicitlyDefined(VirtReg))
          // FIXME: Is this needed?
          BuildMI(*MBB, &MI, MI.getDebugLoc(),
                  TII->get(TargetOpcode::IMPLICIT_DEF), Phys);
        continue;
      }

      // This virtual register is now known to be a spilled value.
      if (!MI.getOperand(i).isUse())
        continue;  // Handle defs in the loop below (handle use&def here though)

      bool AvoidReload = MI.getOperand(i).isUndef();
      // Check if it is defined by an implicit def. It should not be spilled.
      // Note, this is for correctness reason. e.g.
      // 8   %reg1024<def> = IMPLICIT_DEF
      // 12  %reg1024<def> = INSERT_SUBREG %reg1024<kill>, %reg1025, 2
      // The live range [12, 14) are not part of the r1024 live interval since
      // it's defined by an implicit def. It will not conflicts with live
      // interval of r1025. Now suppose both registers are spilled, you can
      // easily see a situation where both registers are reloaded before
      // the INSERT_SUBREG and both target registers that would overlap.
      bool DoReMat = VRM->isReMaterialized(VirtReg);
      int SSorRMId = DoReMat
        ? VRM->getReMatId(VirtReg) : VRM->getStackSlot(VirtReg);
      int ReuseSlot = SSorRMId;

      // Check to see if this stack slot is available.
      unsigned PhysReg = Spills.getSpillSlotOrReMatPhysReg(SSorRMId);

      // If this is a sub-register use, make sure the reuse register is in the
      // right register class. For example, for x86 not all of the 32-bit
      // registers have accessible sub-registers.
      // Similarly so for EXTRACT_SUBREG. Consider this:
      // EDI = op
      // MOV32_mr fi#1, EDI
      // ...
      //       = EXTRACT_SUBREG fi#1
      // fi#1 is available in EDI, but it cannot be reused because it's not in
      // the right register file.
      if (PhysReg && !AvoidReload && SubIdx) {
        const TargetRegisterClass* RC = MRI->getRegClass(VirtReg);
        if (!RC->contains(PhysReg))
          PhysReg = 0;
      }

      if (PhysReg && !AvoidReload) {
        // This spilled operand might be part of a two-address operand.  If this
        // is the case, then changing it will necessarily require changing the
        // def part of the instruction as well.  However, in some cases, we
        // aren't allowed to modify the reused register.  If none of these cases
        // apply, reuse it.
        bool CanReuse = true;
        bool isTied = MI.isRegTiedToDefOperand(i);
        if (isTied) {
          // Okay, we have a two address operand.  We can reuse this physreg as
          // long as we are allowed to clobber the value and there isn't an
          // earlier def that has already clobbered the physreg.
          CanReuse = !ReusedOperands.isClobbered(PhysReg) &&
            Spills.canClobberPhysReg(PhysReg);
        }
        // If this is an asm, and a PhysReg alias is used elsewhere as an
        // earlyclobber operand, we can't also use it as an input.
        if (MI.isInlineAsm()) {
          for (unsigned k = 0, e = MI.getNumOperands(); k != e; ++k) {
            MachineOperand &MOk = MI.getOperand(k);
            if (MOk.isReg() && MOk.isEarlyClobber() &&
                TRI->regsOverlap(MOk.getReg(), PhysReg)) {
              CanReuse = false;
              DEBUG(dbgs() << "Not reusing physreg " << TRI->getName(PhysReg)
                           << " for vreg" << VirtReg << ": " << MOk << '\n');
              break;
            }
          }
        }

        if (CanReuse) {
          // If this stack slot value is already available, reuse it!
          if (ReuseSlot > VirtRegMap::MAX_STACK_SLOT)
            DEBUG(dbgs() << "Reusing RM#"
                  << ReuseSlot-VirtRegMap::MAX_STACK_SLOT-1);
          else
            DEBUG(dbgs() << "Reusing SS#" << ReuseSlot);
          DEBUG(dbgs() << " from physreg "
                << TRI->getName(PhysReg) << " for vreg"
                << VirtReg <<" instead of reloading into physreg "
                << TRI->getName(VRM->getPhys(VirtReg)) << '\n');
          unsigned RReg = SubIdx ? TRI->getSubReg(PhysReg, SubIdx) : PhysReg;
          MI.getOperand(i).setReg(RReg);
          MI.getOperand(i).setSubReg(0);

          // The only technical detail we have is that we don't know that
          // PhysReg won't be clobbered by a reloaded stack slot that occurs
          // later in the instruction.  In particular, consider 'op V1, V2'.
          // If V1 is available in physreg R0, we would choose to reuse it
          // here, instead of reloading it into the register the allocator
          // indicated (say R1).  However, V2 might have to be reloaded
          // later, and it might indicate that it needs to live in R0.  When
          // this occurs, we need to have information available that
          // indicates it is safe to use R1 for the reload instead of R0.
          //
          // To further complicate matters, we might conflict with an alias,
          // or R0 and R1 might not be compatible with each other.  In this
          // case, we actually insert a reload for V1 in R1, ensuring that
          // we can get at R0 or its alias.
          ReusedOperands.addReuse(i, ReuseSlot, PhysReg,
                                  VRM->getPhys(VirtReg), VirtReg);
          if (isTied)
            // Only mark it clobbered if this is a use&def operand.
            ReusedOperands.markClobbered(PhysReg);
          ++NumReused;

          if (MI.getOperand(i).isKill() &&
              ReuseSlot <= VirtRegMap::MAX_STACK_SLOT) {

            // The store of this spilled value is potentially dead, but we
            // won't know for certain until we've confirmed that the re-use
            // above is valid, which means waiting until the other operands
            // are processed. For now we just track the spill slot, we'll
            // remove it after the other operands are processed if valid.

            PotentialDeadStoreSlots.push_back(ReuseSlot);
          }

          // Mark is isKill if it's there no other uses of the same virtual
          // register and it's not a two-address operand. IsKill will be
          // unset if reg is reused.
          if (!isTied && KilledMIRegs.count(VirtReg) == 0) {
            MI.getOperand(i).setIsKill();
            KilledMIRegs.insert(VirtReg);
          }

          continue;
        }  // CanReuse

        // Otherwise we have a situation where we have a two-address instruction
        // whose mod/ref operand needs to be reloaded.  This reload is already
        // available in some register "PhysReg", but if we used PhysReg as the
        // operand to our 2-addr instruction, the instruction would modify
        // PhysReg.  This isn't cool if something later uses PhysReg and expects
        // to get its initial value.
        //
        // To avoid this problem, and to avoid doing a load right after a store,
        // we emit a copy from PhysReg into the designated register for this
        // operand.
        //
        // This case also applies to an earlyclobber'd PhysReg.
        unsigned DesignatedReg = VRM->getPhys(VirtReg);
        assert(DesignatedReg && "Must map virtreg to physreg!");

        // Note that, if we reused a register for a previous operand, the
        // register we want to reload into might not actually be
        // available.  If this occurs, use the register indicated by the
        // reuser.
        if (ReusedOperands.hasReuses())
          DesignatedReg = ReusedOperands.
            GetRegForReload(VirtReg, DesignatedReg, &MI, Spills,
                            MaybeDeadStores, RegKills, KillOps, *VRM);

        // If the mapped designated register is actually the physreg we have
        // incoming, we don't need to inserted a dead copy.
        if (DesignatedReg == PhysReg) {
          // If this stack slot value is already available, reuse it!
          if (ReuseSlot > VirtRegMap::MAX_STACK_SLOT)
            DEBUG(dbgs() << "Reusing RM#"
                  << ReuseSlot-VirtRegMap::MAX_STACK_SLOT-1);
          else
            DEBUG(dbgs() << "Reusing SS#" << ReuseSlot);
          DEBUG(dbgs() << " from physreg " << TRI->getName(PhysReg)
                << " for vreg" << VirtReg
                << " instead of reloading into same physreg.\n");
          unsigned RReg = SubIdx ? TRI->getSubReg(PhysReg, SubIdx) : PhysReg;
          MI.getOperand(i).setReg(RReg);
          MI.getOperand(i).setSubReg(0);
          ReusedOperands.markClobbered(RReg);
          ++NumReused;
          continue;
        }

        MRI->setPhysRegUsed(DesignatedReg);
        ReusedOperands.markClobbered(DesignatedReg);

        // Back-schedule reloads and remats.
        MachineBasicBlock::iterator InsertLoc =
          ComputeReloadLoc(&MI, MBB->begin(), PhysReg, TRI, DoReMat,
                           SSorRMId, TII, MF);
        MachineInstr *CopyMI = BuildMI(*MBB, InsertLoc, MI.getDebugLoc(),
                                       TII->get(TargetOpcode::COPY),
                                       DesignatedReg).addReg(PhysReg);
        CopyMI->setAsmPrinterFlag(MachineInstr::ReloadReuse);
        UpdateKills(*CopyMI, TRI, RegKills, KillOps);

        // This invalidates DesignatedReg.
        Spills.ClobberPhysReg(DesignatedReg);

        Spills.addAvailable(ReuseSlot, DesignatedReg);
        unsigned RReg =
          SubIdx ? TRI->getSubReg(DesignatedReg, SubIdx) : DesignatedReg;
        MI.getOperand(i).setReg(RReg);
        MI.getOperand(i).setSubReg(0);
        DEBUG(dbgs() << '\t' << *prior(MII));
        ++NumReused;
        continue;
      } // if (PhysReg)

        // Otherwise, reload it and remember that we have it.
      PhysReg = VRM->getPhys(VirtReg);
      assert(PhysReg && "Must map virtreg to physreg!");

      // Note that, if we reused a register for a previous operand, the
      // register we want to reload into might not actually be
      // available.  If this occurs, use the register indicated by the
      // reuser.
      if (ReusedOperands.hasReuses())
        PhysReg = ReusedOperands.GetRegForReload(VirtReg, PhysReg, &MI,
                    Spills, MaybeDeadStores, RegKills, KillOps, *VRM);

      MRI->setPhysRegUsed(PhysReg);
      ReusedOperands.markClobbered(PhysReg);
      if (AvoidReload)
        ++NumAvoided;
      else {
        // Back-schedule reloads and remats.
        MachineBasicBlock::iterator InsertLoc =
          ComputeReloadLoc(MII, MBB->begin(), PhysReg, TRI, DoReMat,
                           SSorRMId, TII, MF);

        if (DoReMat) {
          ReMaterialize(*MBB, InsertLoc, PhysReg, VirtReg, TII, TRI, *VRM);
        } else {
          const TargetRegisterClass* RC = MRI->getRegClass(VirtReg);
          TII->loadRegFromStackSlot(*MBB, InsertLoc, PhysReg, SSorRMId, RC,TRI);
          MachineInstr *LoadMI = prior(InsertLoc);
          VRM->addSpillSlotUse(SSorRMId, LoadMI);
          ++NumLoads;
          DistanceMap.insert(std::make_pair(LoadMI, DistanceMap.size()));
        }
        // This invalidates PhysReg.
        Spills.ClobberPhysReg(PhysReg);

        // Any stores to this stack slot are not dead anymore.
        if (!DoReMat)
          MaybeDeadStores[SSorRMId] = NULL;
        Spills.addAvailable(SSorRMId, PhysReg);
        // Assumes this is the last use. IsKill will be unset if reg is reused
        // unless it's a two-address operand.
        if (!MI.isRegTiedToDefOperand(i) &&
            KilledMIRegs.count(VirtReg) == 0) {
          MI.getOperand(i).setIsKill();
          KilledMIRegs.insert(VirtReg);
        }

        UpdateKills(*prior(InsertLoc), TRI, RegKills, KillOps);
        DEBUG(dbgs() << '\t' << *prior(InsertLoc));
      }
      unsigned RReg = SubIdx ? TRI->getSubReg(PhysReg, SubIdx) : PhysReg;
      MI.getOperand(i).setReg(RReg);
      MI.getOperand(i).setSubReg(0);
    }

    // Ok - now we can remove stores that have been confirmed dead.
    for (unsigned j = 0, e = PotentialDeadStoreSlots.size(); j != e; ++j) {
      // This was the last use and the spilled value is still available
      // for reuse. That means the spill was unnecessary!
      int PDSSlot = PotentialDeadStoreSlots[j];
      MachineInstr* DeadStore = MaybeDeadStores[PDSSlot];
      if (DeadStore) {
        DEBUG(dbgs() << "Removed dead store:\t" << *DeadStore);
        InvalidateKills(*DeadStore, TRI, RegKills, KillOps);
        VRM->RemoveMachineInstrFromMaps(DeadStore);
        MBB->erase(DeadStore);
        MaybeDeadStores[PDSSlot] = NULL;
        ++NumDSE;
      }
    }


    DEBUG(dbgs() << '\t' << MI);


    // If we have folded references to memory operands, make sure we clear all
    // physical registers that may contain the value of the spilled virtual
    // register

    // Copy the folded virts to a small vector, we may change MI2VirtMap.
    SmallVector<std::pair<unsigned, VirtRegMap::ModRef>, 4> FoldedVirts;
    // C++0x FTW!
    for (std::pair<VirtRegMap::MI2VirtMapTy::const_iterator,
                   VirtRegMap::MI2VirtMapTy::const_iterator> FVRange =
           VRM->getFoldedVirts(&MI);
         FVRange.first != FVRange.second; ++FVRange.first)
      FoldedVirts.push_back(FVRange.first->second);

    SmallSet<int, 2> FoldedSS;
    for (unsigned FVI = 0, FVE = FoldedVirts.size(); FVI != FVE; ++FVI) {
      unsigned VirtReg = FoldedVirts[FVI].first;
      VirtRegMap::ModRef MR = FoldedVirts[FVI].second;
      DEBUG(dbgs() << "Folded vreg: " << VirtReg << "  MR: " << MR);

      int SS = VRM->getStackSlot(VirtReg);
      if (SS == VirtRegMap::NO_STACK_SLOT)
        continue;
      FoldedSS.insert(SS);
      DEBUG(dbgs() << " - StackSlot: " << SS << "\n");

      // If this folded instruction is just a use, check to see if it's a
      // straight load from the virt reg slot.
      if ((MR & VirtRegMap::isRef) && !(MR & VirtRegMap::isMod)) {
        int FrameIdx;
        unsigned DestReg = TII->isLoadFromStackSlot(&MI, FrameIdx);
        if (DestReg && FrameIdx == SS) {
          // If this spill slot is available, turn it into a copy (or nothing)
          // instead of leaving it as a load!
          if (unsigned InReg = Spills.getSpillSlotOrReMatPhysReg(SS)) {
            DEBUG(dbgs() << "Promoted Load To Copy: " << MI);
            if (DestReg != InReg) {
              MachineOperand *DefMO = MI.findRegisterDefOperand(DestReg);
              MachineInstr *CopyMI = BuildMI(*MBB, &MI, MI.getDebugLoc(),
                                             TII->get(TargetOpcode::COPY))
                .addReg(DestReg, RegState::Define, DefMO->getSubReg())
                .addReg(InReg, RegState::Kill);
              // Revisit the copy so we make sure to notice the effects of the
              // operation on the destreg (either needing to RA it if it's
              // virtual or needing to clobber any values if it's physical).
              NextMII = CopyMI;
              NextMII->setAsmPrinterFlag(MachineInstr::ReloadReuse);
              BackTracked = true;
            } else {
              DEBUG(dbgs() << "Removing now-noop copy: " << MI);
              // Unset last kill since it's being reused.
              InvalidateKill(InReg, TRI, RegKills, KillOps);
              Spills.disallowClobberPhysReg(InReg);
            }

            InvalidateKills(MI, TRI, RegKills, KillOps);
            VRM->RemoveMachineInstrFromMaps(&MI);
            MBB->erase(&MI);
            Erased = true;
            goto ProcessNextInst;
          }
        } else {
          unsigned PhysReg = Spills.getSpillSlotOrReMatPhysReg(SS);
          SmallVector<MachineInstr*, 4> NewMIs;
          if (PhysReg &&
              TII->unfoldMemoryOperand(MF, &MI, PhysReg, false, false, NewMIs)){
            MBB->insert(MII, NewMIs[0]);
            InvalidateKills(MI, TRI, RegKills, KillOps);
            VRM->RemoveMachineInstrFromMaps(&MI);
            MBB->erase(&MI);
            Erased = true;
            --NextMII;  // backtrack to the unfolded instruction.
            BackTracked = true;
            goto ProcessNextInst;
          }
        }
      }

      // If this reference is not a use, any previous store is now dead.
      // Otherwise, the store to this stack slot is not dead anymore.
      MachineInstr* DeadStore = MaybeDeadStores[SS];
      if (DeadStore) {
        bool isDead = !(MR & VirtRegMap::isRef);
        MachineInstr *NewStore = NULL;
        if (MR & VirtRegMap::isModRef) {
          unsigned PhysReg = Spills.getSpillSlotOrReMatPhysReg(SS);
          SmallVector<MachineInstr*, 4> NewMIs;
          // We can reuse this physreg as long as we are allowed to clobber
          // the value and there isn't an earlier def that has already clobbered
          // the physreg.
          if (PhysReg &&
              !ReusedOperands.isClobbered(PhysReg) &&
              Spills.canClobberPhysReg(PhysReg) &&
              !TII->isStoreToStackSlot(&MI, SS)) { // Not profitable!
            MachineOperand *KillOpnd =
              DeadStore->findRegisterUseOperand(PhysReg, true);
            // Note, if the store is storing a sub-register, it's possible the
            // super-register is needed below.
            if (KillOpnd && !KillOpnd->getSubReg() &&
                TII->unfoldMemoryOperand(MF, &MI, PhysReg, false, true,NewMIs)){
              MBB->insert(MII, NewMIs[0]);
              NewStore = NewMIs[1];
              MBB->insert(MII, NewStore);
              VRM->addSpillSlotUse(SS, NewStore);
              InvalidateKills(MI, TRI, RegKills, KillOps);
              VRM->RemoveMachineInstrFromMaps(&MI);
              MBB->erase(&MI);
              Erased = true;
              --NextMII;
              --NextMII;  // backtrack to the unfolded instruction.
              BackTracked = true;
              isDead = true;
              ++NumSUnfold;
            }
          }
        }

        if (isDead) {  // Previous store is dead.
          // If we get here, the store is dead, nuke it now.
          DEBUG(dbgs() << "Removed dead store:\t" << *DeadStore);
          InvalidateKills(*DeadStore, TRI, RegKills, KillOps);
          VRM->RemoveMachineInstrFromMaps(DeadStore);
          MBB->erase(DeadStore);
          if (!NewStore)
            ++NumDSE;
        }

        MaybeDeadStores[SS] = NULL;
        if (NewStore) {
          // Treat this store as a spill merged into a copy. That makes the
          // stack slot value available.
          VRM->virtFolded(VirtReg, NewStore, VirtRegMap::isMod);
          goto ProcessNextInst;
        }
      }

      // If the spill slot value is available, and this is a new definition of
      // the value, the value is not available anymore.
      if (MR & VirtRegMap::isMod) {
        // Notice that the value in this stack slot has been modified.
        Spills.ModifyStackSlotOrReMat(SS);

        // If this is *just* a mod of the value, check to see if this is just a
        // store to the spill slot (i.e. the spill got merged into the copy). If
        // so, realize that the vreg is available now, and add the store to the
        // MaybeDeadStore info.
        int StackSlot;
        if (!(MR & VirtRegMap::isRef)) {
          if (unsigned SrcReg = TII->isStoreToStackSlot(&MI, StackSlot)) {
            assert(TargetRegisterInfo::isPhysicalRegister(SrcReg) &&
                   "Src hasn't been allocated yet?");

            if (CommuteToFoldReload(MII, VirtReg, SrcReg, StackSlot,
                                    Spills, RegKills, KillOps, TRI)) {
              NextMII = llvm::next(MII);
              BackTracked = true;
              goto ProcessNextInst;
            }

            // Okay, this is certainly a store of SrcReg to [StackSlot].  Mark
            // this as a potentially dead store in case there is a subsequent
            // store into the stack slot without a read from it.
            MaybeDeadStores[StackSlot] = &MI;

            // If the stack slot value was previously available in some other
            // register, change it now.  Otherwise, make the register
            // available in PhysReg.
            Spills.addAvailable(StackSlot, SrcReg, MI.killsRegister(SrcReg));
          }
        }
      }
    }

    // Process all of the spilled defs.
    SpilledMIRegs.clear();
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI.getOperand(i);
      if (!(MO.isReg() && MO.getReg() && MO.isDef()))
        continue;

      unsigned VirtReg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(VirtReg)) {
        // Check to see if this is a noop copy.  If so, eliminate the
        // instruction before considering the dest reg to be changed.
        // Also check if it's copying from an "undef", if so, we can't
        // eliminate this or else the undef marker is lost and it will
        // confuses the scavenger. This is extremely rare.
        if (MI.isIdentityCopy() && !MI.getOperand(1).isUndef() &&
            MI.getNumOperands() == 2) {
          ++NumDCE;
          DEBUG(dbgs() << "Removing now-noop copy: " << MI);
          SmallVector<unsigned, 2> KillRegs;
          InvalidateKills(MI, TRI, RegKills, KillOps, &KillRegs);
          if (MO.isDead() && !KillRegs.empty()) {
            // Source register or an implicit super/sub-register use is killed.
            assert(TRI->regsOverlap(KillRegs[0], MI.getOperand(0).getReg()));
            // Last def is now dead.
            TransferDeadness(MI.getOperand(1).getReg(), RegKills, KillOps);
          }
          VRM->RemoveMachineInstrFromMaps(&MI);
          MBB->erase(&MI);
          Erased = true;
          Spills.disallowClobberPhysReg(VirtReg);
          goto ProcessNextInst;
        }

        // If it's not a no-op copy, it clobbers the value in the destreg.
        Spills.ClobberPhysReg(VirtReg);
        ReusedOperands.markClobbered(VirtReg);

        // Check to see if this instruction is a load from a stack slot into
        // a register.  If so, this provides the stack slot value in the reg.
        int FrameIdx;
        if (unsigned DestReg = TII->isLoadFromStackSlot(&MI, FrameIdx)) {
          assert(DestReg == VirtReg && "Unknown load situation!");

          // If it is a folded reference, then it's not safe to clobber.
          bool Folded = FoldedSS.count(FrameIdx);
          // Otherwise, if it wasn't available, remember that it is now!
          Spills.addAvailable(FrameIdx, DestReg, !Folded);
          goto ProcessNextInst;
        }

        continue;
      }

      unsigned SubIdx = MO.getSubReg();
      bool DoReMat = VRM->isReMaterialized(VirtReg);
      if (DoReMat)
        ReMatDefs.insert(&MI);

      // The only vregs left are stack slot definitions.
      int StackSlot = VRM->getStackSlot(VirtReg);
      const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);

      // If this def is part of a two-address operand, make sure to execute
      // the store from the correct physical register.
      unsigned PhysReg;
      unsigned TiedOp;
      if (MI.isRegTiedToUseOperand(i, &TiedOp)) {
        PhysReg = MI.getOperand(TiedOp).getReg();
        if (SubIdx) {
          unsigned SuperReg = findSuperReg(RC, PhysReg, SubIdx, TRI);
          assert(SuperReg && TRI->getSubReg(SuperReg, SubIdx) == PhysReg &&
                 "Can't find corresponding super-register!");
          PhysReg = SuperReg;
        }
      } else {
        PhysReg = VRM->getPhys(VirtReg);
        if (ReusedOperands.isClobbered(PhysReg)) {
          // Another def has taken the assigned physreg. It must have been a
          // use&def which got it due to reuse. Undo the reuse!
          PhysReg = ReusedOperands.GetRegForReload(VirtReg, PhysReg, &MI,
                      Spills, MaybeDeadStores, RegKills, KillOps, *VRM);
        }
      }

      assert(PhysReg && "VR not assigned a physical register?");
      MRI->setPhysRegUsed(PhysReg);
      unsigned RReg = SubIdx ? TRI->getSubReg(PhysReg, SubIdx) : PhysReg;
      ReusedOperands.markClobbered(RReg);
      MI.getOperand(i).setReg(RReg);
      MI.getOperand(i).setSubReg(0);

      if (!MO.isDead() && SpilledMIRegs.insert(VirtReg)) {
        MachineInstr *&LastStore = MaybeDeadStores[StackSlot];
        SpillRegToStackSlot(MII, -1, PhysReg, StackSlot, RC, true,
          LastStore, Spills, ReMatDefs, RegKills, KillOps);
        NextMII = llvm::next(MII);

        // Check to see if this is a noop copy.  If so, eliminate the
        // instruction before considering the dest reg to be changed.
        if (MI.isIdentityCopy()) {
          ++NumDCE;
          DEBUG(dbgs() << "Removing now-noop copy: " << MI);
          InvalidateKills(MI, TRI, RegKills, KillOps);
          VRM->RemoveMachineInstrFromMaps(&MI);
          MBB->erase(&MI);
          Erased = true;
          UpdateKills(*LastStore, TRI, RegKills, KillOps);
          goto ProcessNextInst;
        }
      }
    }
    ProcessNextInst:
    // Delete dead instructions without side effects.
    if (!Erased && !BackTracked && isSafeToDelete(MI)) {
      InvalidateKills(MI, TRI, RegKills, KillOps);
      VRM->RemoveMachineInstrFromMaps(&MI);
      MBB->erase(&MI);
      Erased = true;
    }
    if (!Erased)
      DistanceMap.insert(std::make_pair(&MI, DistanceMap.size()));
    if (!Erased && !BackTracked) {
      for (MachineBasicBlock::iterator II = &MI; II != NextMII; ++II)
        UpdateKills(*II, TRI, RegKills, KillOps);
    }
    MII = NextMII;
  }

}

llvm::VirtRegRewriter* llvm::createVirtRegRewriter() {
  switch (RewriterOpt) {
  default: llvm_unreachable("Unreachable!");
  case local:
    return new LocalRewriter();
  case trivial:
    return new TrivialRewriter();
  }
}
