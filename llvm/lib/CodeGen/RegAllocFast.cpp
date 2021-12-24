//===- RegAllocFast.cpp - A fast register allocator for debug code --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This register allocator allocates registers to a basic block at a
/// time, attempting to keep values in registers and reusing registers as
/// appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <tuple>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");
STATISTIC(NumCoalesced, "Number of copies coalesced");

// FIXME: Remove this switch when all testcases are fixed!
static cl::opt<bool> IgnoreMissingDefs("rafast-ignore-missing-defs",
                                       cl::Hidden);

static RegisterRegAlloc
  fastRegAlloc("fast", "fast register allocator", createFastRegisterAllocator);

namespace {

  class RegAllocFast : public MachineFunctionPass {
  public:
    static char ID;

    RegAllocFast(const RegClassFilterFunc F = allocateAllRegClasses,
                 bool ClearVirtRegs_ = true) :
      MachineFunctionPass(ID),
      ShouldAllocateClass(F),
      StackSlotForVirtReg(-1),
      ClearVirtRegs(ClearVirtRegs_) {
    }

  private:
    MachineFrameInfo *MFI;
    MachineRegisterInfo *MRI;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    RegisterClassInfo RegClassInfo;
    const RegClassFilterFunc ShouldAllocateClass;

    /// Basic block currently being allocated.
    MachineBasicBlock *MBB;

    /// Maps virtual regs to the frame index where these values are spilled.
    IndexedMap<int, VirtReg2IndexFunctor> StackSlotForVirtReg;

    bool ClearVirtRegs;

    /// Everything we know about a live virtual register.
    struct LiveReg {
      MachineInstr *LastUse = nullptr; ///< Last instr to use reg.
      Register VirtReg;                ///< Virtual register number.
      MCPhysReg PhysReg = 0;           ///< Currently held here.
      bool LiveOut = false;            ///< Register is possibly live out.
      bool Reloaded = false;           ///< Register was reloaded.
      bool Error = false;              ///< Could not allocate.

      explicit LiveReg(Register VirtReg) : VirtReg(VirtReg) {}

      unsigned getSparseSetIndex() const {
        return Register::virtReg2Index(VirtReg);
      }
    };

    using LiveRegMap = SparseSet<LiveReg>;
    /// This map contains entries for each virtual register that is currently
    /// available in a physical register.
    LiveRegMap LiveVirtRegs;

    /// Stores assigned virtual registers present in the bundle MI.
    DenseMap<Register, MCPhysReg> BundleVirtRegsMap;

    DenseMap<unsigned, SmallVector<MachineOperand *, 2>> LiveDbgValueMap;
    /// List of DBG_VALUE that we encountered without the vreg being assigned
    /// because they were placed after the last use of the vreg.
    DenseMap<unsigned, SmallVector<MachineInstr *, 1>> DanglingDbgValues;

    /// Has a bit set for every virtual register for which it was determined
    /// that it is alive across blocks.
    BitVector MayLiveAcrossBlocks;

    /// State of a register unit.
    enum RegUnitState {
      /// A free register is not currently in use and can be allocated
      /// immediately without checking aliases.
      regFree,

      /// A pre-assigned register has been assigned before register allocation
      /// (e.g., setting up a call parameter).
      regPreAssigned,

      /// Used temporarily in reloadAtBegin() to mark register units that are
      /// live-in to the basic block.
      regLiveIn,

      /// A register state may also be a virtual register number, indication
      /// that the physical register is currently allocated to a virtual
      /// register. In that case, LiveVirtRegs contains the inverse mapping.
    };

    /// Maps each physical register to a RegUnitState enum or virtual register.
    std::vector<unsigned> RegUnitStates;

    SmallVector<MachineInstr *, 32> Coalesced;

    using RegUnitSet = SparseSet<uint16_t, identity<uint16_t>>;
    /// Set of register units that are used in the current instruction, and so
    /// cannot be allocated.
    RegUnitSet UsedInInstr;
    RegUnitSet PhysRegUses;
    SmallVector<uint16_t, 8> DefOperandIndexes;
    // Register masks attached to the current instruction.
    SmallVector<const uint32_t *> RegMasks;

    void setPhysRegState(MCPhysReg PhysReg, unsigned NewState);
    bool isPhysRegFree(MCPhysReg PhysReg) const;

    /// Mark a physreg as used in this instruction.
    void markRegUsedInInstr(MCPhysReg PhysReg) {
      for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units)
        UsedInInstr.insert(*Units);
    }

    // Check if physreg is clobbered by instruction's regmask(s).
    bool isClobberedByRegMasks(MCPhysReg PhysReg) const {
      return llvm::any_of(RegMasks, [PhysReg](const uint32_t *Mask) {
        return MachineOperand::clobbersPhysReg(Mask, PhysReg);
      });
    }

    /// Check if a physreg or any of its aliases are used in this instruction.
    bool isRegUsedInInstr(MCPhysReg PhysReg, bool LookAtPhysRegUses) const {
      if (LookAtPhysRegUses && isClobberedByRegMasks(PhysReg))
        return true;
      for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units) {
        if (UsedInInstr.count(*Units))
          return true;
        if (LookAtPhysRegUses && PhysRegUses.count(*Units))
          return true;
      }
      return false;
    }

    /// Mark physical register as being used in a register use operand.
    /// This is only used by the special livethrough handling code.
    void markPhysRegUsedInInstr(MCPhysReg PhysReg) {
      for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units)
        PhysRegUses.insert(*Units);
    }

    /// Remove mark of physical register being used in the instruction.
    void unmarkRegUsedInInstr(MCPhysReg PhysReg) {
      for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units)
        UsedInInstr.erase(*Units);
    }

    enum : unsigned {
      spillClean = 50,
      spillDirty = 100,
      spillPrefBonus = 20,
      spillImpossible = ~0u
    };

  public:
    StringRef getPassName() const override { return "Fast Register Allocator"; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoPHIs);
    }

    MachineFunctionProperties getSetProperties() const override {
      if (ClearVirtRegs) {
        return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
      }

      return MachineFunctionProperties();
    }

    MachineFunctionProperties getClearedProperties() const override {
      return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
    }

  private:
    bool runOnMachineFunction(MachineFunction &MF) override;

    void allocateBasicBlock(MachineBasicBlock &MBB);

    void addRegClassDefCounts(std::vector<unsigned> &RegClassDefCounts,
                              Register Reg) const;

    void allocateInstruction(MachineInstr &MI);
    void handleDebugValue(MachineInstr &MI);
    void handleBundle(MachineInstr &MI);

    bool usePhysReg(MachineInstr &MI, MCPhysReg PhysReg);
    bool definePhysReg(MachineInstr &MI, MCPhysReg PhysReg);
    bool displacePhysReg(MachineInstr &MI, MCPhysReg PhysReg);
    void freePhysReg(MCPhysReg PhysReg);

    unsigned calcSpillCost(MCPhysReg PhysReg) const;

    LiveRegMap::iterator findLiveVirtReg(Register VirtReg) {
      return LiveVirtRegs.find(Register::virtReg2Index(VirtReg));
    }

    LiveRegMap::const_iterator findLiveVirtReg(Register VirtReg) const {
      return LiveVirtRegs.find(Register::virtReg2Index(VirtReg));
    }

    void assignVirtToPhysReg(MachineInstr &MI, LiveReg &, MCPhysReg PhysReg);
    void allocVirtReg(MachineInstr &MI, LiveReg &LR, Register Hint,
                      bool LookAtPhysRegUses = false);
    void allocVirtRegUndef(MachineOperand &MO);
    void assignDanglingDebugValues(MachineInstr &Def, Register VirtReg,
                                   MCPhysReg Reg);
    void defineLiveThroughVirtReg(MachineInstr &MI, unsigned OpNum,
                                  Register VirtReg);
    void defineVirtReg(MachineInstr &MI, unsigned OpNum, Register VirtReg,
                       bool LookAtPhysRegUses = false);
    void useVirtReg(MachineInstr &MI, unsigned OpNum, Register VirtReg);

    MachineBasicBlock::iterator
    getMBBBeginInsertionPoint(MachineBasicBlock &MBB,
                              SmallSet<Register, 2> &PrologLiveIns) const;

    void reloadAtBegin(MachineBasicBlock &MBB);
    void setPhysReg(MachineInstr &MI, MachineOperand &MO, MCPhysReg PhysReg);

    Register traceCopies(Register VirtReg) const;
    Register traceCopyChain(Register Reg) const;

    int getStackSpaceFor(Register VirtReg);
    void spill(MachineBasicBlock::iterator Before, Register VirtReg,
               MCPhysReg AssignedReg, bool Kill, bool LiveOut);
    void reload(MachineBasicBlock::iterator Before, Register VirtReg,
                MCPhysReg PhysReg);

    bool mayLiveOut(Register VirtReg);
    bool mayLiveIn(Register VirtReg);

    void dumpState() const;
  };

} // end anonymous namespace

char RegAllocFast::ID = 0;

INITIALIZE_PASS(RegAllocFast, "regallocfast", "Fast Register Allocator", false,
                false)

void RegAllocFast::setPhysRegState(MCPhysReg PhysReg, unsigned NewState) {
  for (MCRegUnitIterator UI(PhysReg, TRI); UI.isValid(); ++UI)
    RegUnitStates[*UI] = NewState;
}

bool RegAllocFast::isPhysRegFree(MCPhysReg PhysReg) const {
  for (MCRegUnitIterator UI(PhysReg, TRI); UI.isValid(); ++UI) {
    if (RegUnitStates[*UI] != regFree)
      return false;
  }
  return true;
}

/// This allocates space for the specified virtual register to be held on the
/// stack.
int RegAllocFast::getStackSpaceFor(Register VirtReg) {
  // Find the location Reg would belong...
  int SS = StackSlotForVirtReg[VirtReg];
  // Already has space allocated?
  if (SS != -1)
    return SS;

  // Allocate a new stack object for this spill location...
  const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
  unsigned Size = TRI->getSpillSize(RC);
  Align Alignment = TRI->getSpillAlign(RC);
  int FrameIdx = MFI->CreateSpillStackObject(Size, Alignment);

  // Assign the slot.
  StackSlotForVirtReg[VirtReg] = FrameIdx;
  return FrameIdx;
}

static bool dominates(MachineBasicBlock &MBB,
                      MachineBasicBlock::const_iterator A,
                      MachineBasicBlock::const_iterator B) {
  auto MBBEnd = MBB.end();
  if (B == MBBEnd)
    return true;

  MachineBasicBlock::const_iterator I = MBB.begin();
  for (; &*I != A && &*I != B; ++I)
    ;

  return &*I == A;
}

/// Returns false if \p VirtReg is known to not live out of the current block.
bool RegAllocFast::mayLiveOut(Register VirtReg) {
  if (MayLiveAcrossBlocks.test(Register::virtReg2Index(VirtReg))) {
    // Cannot be live-out if there are no successors.
    return !MBB->succ_empty();
  }

  const MachineInstr *SelfLoopDef = nullptr;

  // If this block loops back to itself, it is necessary to check whether the
  // use comes after the def.
  if (MBB->isSuccessor(MBB)) {
    SelfLoopDef = MRI->getUniqueVRegDef(VirtReg);
    if (!SelfLoopDef) {
      MayLiveAcrossBlocks.set(Register::virtReg2Index(VirtReg));
      return true;
    }
  }

  // See if the first \p Limit uses of the register are all in the current
  // block.
  static const unsigned Limit = 8;
  unsigned C = 0;
  for (const MachineInstr &UseInst : MRI->use_nodbg_instructions(VirtReg)) {
    if (UseInst.getParent() != MBB || ++C >= Limit) {
      MayLiveAcrossBlocks.set(Register::virtReg2Index(VirtReg));
      // Cannot be live-out if there are no successors.
      return !MBB->succ_empty();
    }

    if (SelfLoopDef) {
      // Try to handle some simple cases to avoid spilling and reloading every
      // value inside a self looping block.
      if (SelfLoopDef == &UseInst ||
          !dominates(*MBB, SelfLoopDef->getIterator(), UseInst.getIterator())) {
        MayLiveAcrossBlocks.set(Register::virtReg2Index(VirtReg));
        return true;
      }
    }
  }

  return false;
}

/// Returns false if \p VirtReg is known to not be live into the current block.
bool RegAllocFast::mayLiveIn(Register VirtReg) {
  if (MayLiveAcrossBlocks.test(Register::virtReg2Index(VirtReg)))
    return !MBB->pred_empty();

  // See if the first \p Limit def of the register are all in the current block.
  static const unsigned Limit = 8;
  unsigned C = 0;
  for (const MachineInstr &DefInst : MRI->def_instructions(VirtReg)) {
    if (DefInst.getParent() != MBB || ++C >= Limit) {
      MayLiveAcrossBlocks.set(Register::virtReg2Index(VirtReg));
      return !MBB->pred_empty();
    }
  }

  return false;
}

/// Insert spill instruction for \p AssignedReg before \p Before. Update
/// DBG_VALUEs with \p VirtReg operands with the stack slot.
void RegAllocFast::spill(MachineBasicBlock::iterator Before, Register VirtReg,
                         MCPhysReg AssignedReg, bool Kill, bool LiveOut) {
  LLVM_DEBUG(dbgs() << "Spilling " << printReg(VirtReg, TRI)
                    << " in " << printReg(AssignedReg, TRI));
  int FI = getStackSpaceFor(VirtReg);
  LLVM_DEBUG(dbgs() << " to stack slot #" << FI << '\n');

  const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
  TII->storeRegToStackSlot(*MBB, Before, AssignedReg, Kill, FI, &RC, TRI);
  ++NumStores;

  MachineBasicBlock::iterator FirstTerm = MBB->getFirstTerminator();

  // When we spill a virtual register, we will have spill instructions behind
  // every definition of it, meaning we can switch all the DBG_VALUEs over
  // to just reference the stack slot.
  SmallVectorImpl<MachineOperand *> &LRIDbgOperands = LiveDbgValueMap[VirtReg];
  SmallMapVector<MachineInstr *, SmallVector<const MachineOperand *>, 2>
      SpilledOperandsMap;
  for (MachineOperand *MO : LRIDbgOperands)
    SpilledOperandsMap[MO->getParent()].push_back(MO);
  for (auto MISpilledOperands : SpilledOperandsMap) {
    MachineInstr &DBG = *MISpilledOperands.first;
    MachineInstr *NewDV = buildDbgValueForSpill(
        *MBB, Before, *MISpilledOperands.first, FI, MISpilledOperands.second);
    assert(NewDV->getParent() == MBB && "dangling parent pointer");
    (void)NewDV;
    LLVM_DEBUG(dbgs() << "Inserting debug info due to spill:\n" << *NewDV);

    if (LiveOut) {
      // We need to insert a DBG_VALUE at the end of the block if the spill slot
      // is live out, but there is another use of the value after the
      // spill. This will allow LiveDebugValues to see the correct live out
      // value to propagate to the successors.
      MachineInstr *ClonedDV = MBB->getParent()->CloneMachineInstr(NewDV);
      MBB->insert(FirstTerm, ClonedDV);
      LLVM_DEBUG(dbgs() << "Cloning debug info due to live out spill\n");
    }

    // Rewrite unassigned dbg_values to use the stack slot.
    // TODO We can potentially do this for list debug values as well if we know
    // how the dbg_values are getting unassigned.
    if (DBG.isNonListDebugValue()) {
      MachineOperand &MO = DBG.getDebugOperand(0);
      if (MO.isReg() && MO.getReg() == 0) {
        updateDbgValueForSpill(DBG, FI, 0);
      }
    }
  }
  // Now this register is spilled there is should not be any DBG_VALUE
  // pointing to this register because they are all pointing to spilled value
  // now.
  LRIDbgOperands.clear();
}

/// Insert reload instruction for \p PhysReg before \p Before.
void RegAllocFast::reload(MachineBasicBlock::iterator Before, Register VirtReg,
                          MCPhysReg PhysReg) {
  LLVM_DEBUG(dbgs() << "Reloading " << printReg(VirtReg, TRI) << " into "
                    << printReg(PhysReg, TRI) << '\n');
  int FI = getStackSpaceFor(VirtReg);
  const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
  TII->loadRegFromStackSlot(*MBB, Before, PhysReg, FI, &RC, TRI);
  ++NumLoads;
}

/// Get basic block begin insertion point.
/// This is not just MBB.begin() because surprisingly we have EH_LABEL
/// instructions marking the begin of a basic block. This means we must insert
/// new instructions after such labels...
MachineBasicBlock::iterator
RegAllocFast::getMBBBeginInsertionPoint(
  MachineBasicBlock &MBB, SmallSet<Register, 2> &PrologLiveIns) const {
  MachineBasicBlock::iterator I = MBB.begin();
  while (I != MBB.end()) {
    if (I->isLabel()) {
      ++I;
      continue;
    }

    // Most reloads should be inserted after prolog instructions.
    if (!TII->isBasicBlockPrologue(*I))
      break;

    // However if a prolog instruction reads a register that needs to be
    // reloaded, the reload should be inserted before the prolog.
    for (MachineOperand &MO : I->operands()) {
      if (MO.isReg())
        PrologLiveIns.insert(MO.getReg());
    }

    ++I;
  }

  return I;
}

/// Reload all currently assigned virtual registers.
void RegAllocFast::reloadAtBegin(MachineBasicBlock &MBB) {
  if (LiveVirtRegs.empty())
    return;

  for (MachineBasicBlock::RegisterMaskPair P : MBB.liveins()) {
    MCPhysReg Reg = P.PhysReg;
    // Set state to live-in. This possibly overrides mappings to virtual
    // registers but we don't care anymore at this point.
    setPhysRegState(Reg, regLiveIn);
  }


  SmallSet<Register, 2> PrologLiveIns;

  // The LiveRegMap is keyed by an unsigned (the virtreg number), so the order
  // of spilling here is deterministic, if arbitrary.
  MachineBasicBlock::iterator InsertBefore
    = getMBBBeginInsertionPoint(MBB, PrologLiveIns);
  for (const LiveReg &LR : LiveVirtRegs) {
    MCPhysReg PhysReg = LR.PhysReg;
    if (PhysReg == 0)
      continue;

    MCRegister FirstUnit = *MCRegUnitIterator(PhysReg, TRI);
    if (RegUnitStates[FirstUnit] == regLiveIn)
      continue;

    assert((&MBB != &MBB.getParent()->front() || IgnoreMissingDefs) &&
           "no reload in start block. Missing vreg def?");

    if (PrologLiveIns.count(PhysReg)) {
      // FIXME: Theoretically this should use an insert point skipping labels
      // but I'm not sure how labels should interact with prolog instruction
      // that need reloads.
      reload(MBB.begin(), LR.VirtReg, PhysReg);
    } else
      reload(InsertBefore, LR.VirtReg, PhysReg);
  }
  LiveVirtRegs.clear();
}

/// Handle the direct use of a physical register.  Check that the register is
/// not used by a virtreg. Kill the physreg, marking it free. This may add
/// implicit kills to MO->getParent() and invalidate MO.
bool RegAllocFast::usePhysReg(MachineInstr &MI, MCPhysReg Reg) {
  assert(Register::isPhysicalRegister(Reg) && "expected physreg");
  bool displacedAny = displacePhysReg(MI, Reg);
  setPhysRegState(Reg, regPreAssigned);
  markRegUsedInInstr(Reg);
  return displacedAny;
}

bool RegAllocFast::definePhysReg(MachineInstr &MI, MCPhysReg Reg) {
  bool displacedAny = displacePhysReg(MI, Reg);
  setPhysRegState(Reg, regPreAssigned);
  return displacedAny;
}

/// Mark PhysReg as reserved or free after spilling any virtregs. This is very
/// similar to defineVirtReg except the physreg is reserved instead of
/// allocated.
bool RegAllocFast::displacePhysReg(MachineInstr &MI, MCPhysReg PhysReg) {
  bool displacedAny = false;

  for (MCRegUnitIterator UI(PhysReg, TRI); UI.isValid(); ++UI) {
    unsigned Unit = *UI;
    switch (unsigned VirtReg = RegUnitStates[Unit]) {
    default: {
      LiveRegMap::iterator LRI = findLiveVirtReg(VirtReg);
      assert(LRI != LiveVirtRegs.end() && "datastructures in sync");
      MachineBasicBlock::iterator ReloadBefore =
          std::next((MachineBasicBlock::iterator)MI.getIterator());
      reload(ReloadBefore, VirtReg, LRI->PhysReg);

      setPhysRegState(LRI->PhysReg, regFree);
      LRI->PhysReg = 0;
      LRI->Reloaded = true;
      displacedAny = true;
      break;
    }
    case regPreAssigned:
      RegUnitStates[Unit] = regFree;
      displacedAny = true;
      break;
    case regFree:
      break;
    }
  }
  return displacedAny;
}

void RegAllocFast::freePhysReg(MCPhysReg PhysReg) {
  LLVM_DEBUG(dbgs() << "Freeing " << printReg(PhysReg, TRI) << ':');

  MCRegister FirstUnit = *MCRegUnitIterator(PhysReg, TRI);
  switch (unsigned VirtReg = RegUnitStates[FirstUnit]) {
  case regFree:
    LLVM_DEBUG(dbgs() << '\n');
    return;
  case regPreAssigned:
    LLVM_DEBUG(dbgs() << '\n');
    setPhysRegState(PhysReg, regFree);
    return;
  default: {
      LiveRegMap::iterator LRI = findLiveVirtReg(VirtReg);
      assert(LRI != LiveVirtRegs.end());
      LLVM_DEBUG(dbgs() << ' ' << printReg(LRI->VirtReg, TRI) << '\n');
      setPhysRegState(LRI->PhysReg, regFree);
      LRI->PhysReg = 0;
    }
    return;
  }
}

/// Return the cost of spilling clearing out PhysReg and aliases so it is free
/// for allocation. Returns 0 when PhysReg is free or disabled with all aliases
/// disabled - it can be allocated directly.
/// \returns spillImpossible when PhysReg or an alias can't be spilled.
unsigned RegAllocFast::calcSpillCost(MCPhysReg PhysReg) const {
  for (MCRegUnitIterator UI(PhysReg, TRI); UI.isValid(); ++UI) {
    switch (unsigned VirtReg = RegUnitStates[*UI]) {
    case regFree:
      break;
    case regPreAssigned:
      LLVM_DEBUG(dbgs() << "Cannot spill pre-assigned "
                        << printReg(PhysReg, TRI) << '\n');
      return spillImpossible;
    default: {
      bool SureSpill = StackSlotForVirtReg[VirtReg] != -1 ||
                       findLiveVirtReg(VirtReg)->LiveOut;
      return SureSpill ? spillClean : spillDirty;
    }
    }
  }
  return 0;
}

void RegAllocFast::assignDanglingDebugValues(MachineInstr &Definition,
                                             Register VirtReg, MCPhysReg Reg) {
  auto UDBGValIter = DanglingDbgValues.find(VirtReg);
  if (UDBGValIter == DanglingDbgValues.end())
    return;

  SmallVectorImpl<MachineInstr*> &Dangling = UDBGValIter->second;
  for (MachineInstr *DbgValue : Dangling) {
    assert(DbgValue->isDebugValue());
    if (!DbgValue->hasDebugOperandForReg(VirtReg))
      continue;

    // Test whether the physreg survives from the definition to the DBG_VALUE.
    MCPhysReg SetToReg = Reg;
    unsigned Limit = 20;
    for (MachineBasicBlock::iterator I = std::next(Definition.getIterator()),
         E = DbgValue->getIterator(); I != E; ++I) {
      if (I->modifiesRegister(Reg, TRI) || --Limit == 0) {
        LLVM_DEBUG(dbgs() << "Register did not survive for " << *DbgValue
                   << '\n');
        SetToReg = 0;
        break;
      }
    }
    for (MachineOperand &MO : DbgValue->getDebugOperandsForReg(VirtReg)) {
      MO.setReg(SetToReg);
      if (SetToReg != 0)
        MO.setIsRenamable();
    }
  }
  Dangling.clear();
}

/// This method updates local state so that we know that PhysReg is the
/// proper container for VirtReg now.  The physical register must not be used
/// for anything else when this is called.
void RegAllocFast::assignVirtToPhysReg(MachineInstr &AtMI, LiveReg &LR,
                                       MCPhysReg PhysReg) {
  Register VirtReg = LR.VirtReg;
  LLVM_DEBUG(dbgs() << "Assigning " << printReg(VirtReg, TRI) << " to "
                    << printReg(PhysReg, TRI) << '\n');
  assert(LR.PhysReg == 0 && "Already assigned a physreg");
  assert(PhysReg != 0 && "Trying to assign no register");
  LR.PhysReg = PhysReg;
  setPhysRegState(PhysReg, VirtReg);

  assignDanglingDebugValues(AtMI, VirtReg, PhysReg);
}

static bool isCoalescable(const MachineInstr &MI) {
  return MI.isFullCopy();
}

Register RegAllocFast::traceCopyChain(Register Reg) const {
  static const unsigned ChainLengthLimit = 3;
  unsigned C = 0;
  do {
    if (Reg.isPhysical())
      return Reg;
    assert(Reg.isVirtual());

    MachineInstr *VRegDef = MRI->getUniqueVRegDef(Reg);
    if (!VRegDef || !isCoalescable(*VRegDef))
      return 0;
    Reg = VRegDef->getOperand(1).getReg();
  } while (++C <= ChainLengthLimit);
  return 0;
}

/// Check if any of \p VirtReg's definitions is a copy. If it is follow the
/// chain of copies to check whether we reach a physical register we can
/// coalesce with.
Register RegAllocFast::traceCopies(Register VirtReg) const {
  static const unsigned DefLimit = 3;
  unsigned C = 0;
  for (const MachineInstr &MI : MRI->def_instructions(VirtReg)) {
    if (isCoalescable(MI)) {
      Register Reg = MI.getOperand(1).getReg();
      Reg = traceCopyChain(Reg);
      if (Reg.isValid())
        return Reg;
    }

    if (++C >= DefLimit)
      break;
  }
  return Register();
}

/// Allocates a physical register for VirtReg.
void RegAllocFast::allocVirtReg(MachineInstr &MI, LiveReg &LR,
                                Register Hint0, bool LookAtPhysRegUses) {
  const Register VirtReg = LR.VirtReg;
  assert(LR.PhysReg == 0);

  const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
  LLVM_DEBUG(dbgs() << "Search register for " << printReg(VirtReg)
                    << " in class " << TRI->getRegClassName(&RC)
                    << " with hint " << printReg(Hint0, TRI) << '\n');

  // Take hint when possible.
  if (Hint0.isPhysical() && MRI->isAllocatable(Hint0) && RC.contains(Hint0) &&
      !isRegUsedInInstr(Hint0, LookAtPhysRegUses)) {
    // Take hint if the register is currently free.
    if (isPhysRegFree(Hint0)) {
      LLVM_DEBUG(dbgs() << "\tPreferred Register 1: " << printReg(Hint0, TRI)
                        << '\n');
      assignVirtToPhysReg(MI, LR, Hint0);
      return;
    } else {
      LLVM_DEBUG(dbgs() << "\tPreferred Register 0: " << printReg(Hint0, TRI)
                        << " occupied\n");
    }
  } else {
    Hint0 = Register();
  }


  // Try other hint.
  Register Hint1 = traceCopies(VirtReg);
  if (Hint1.isPhysical() && MRI->isAllocatable(Hint1) && RC.contains(Hint1) &&
      !isRegUsedInInstr(Hint1, LookAtPhysRegUses)) {
    // Take hint if the register is currently free.
    if (isPhysRegFree(Hint1)) {
      LLVM_DEBUG(dbgs() << "\tPreferred Register 0: " << printReg(Hint1, TRI)
                 << '\n');
      assignVirtToPhysReg(MI, LR, Hint1);
      return;
    } else {
      LLVM_DEBUG(dbgs() << "\tPreferred Register 1: " << printReg(Hint1, TRI)
                 << " occupied\n");
    }
  } else {
    Hint1 = Register();
  }

  MCPhysReg BestReg = 0;
  unsigned BestCost = spillImpossible;
  ArrayRef<MCPhysReg> AllocationOrder = RegClassInfo.getOrder(&RC);
  for (MCPhysReg PhysReg : AllocationOrder) {
    LLVM_DEBUG(dbgs() << "\tRegister: " << printReg(PhysReg, TRI) << ' ');
    if (isRegUsedInInstr(PhysReg, LookAtPhysRegUses)) {
      LLVM_DEBUG(dbgs() << "already used in instr.\n");
      continue;
    }

    unsigned Cost = calcSpillCost(PhysReg);
    LLVM_DEBUG(dbgs() << "Cost: " << Cost << " BestCost: " << BestCost << '\n');
    // Immediate take a register with cost 0.
    if (Cost == 0) {
      assignVirtToPhysReg(MI, LR, PhysReg);
      return;
    }

    if (PhysReg == Hint0 || PhysReg == Hint1)
      Cost -= spillPrefBonus;

    if (Cost < BestCost) {
      BestReg = PhysReg;
      BestCost = Cost;
    }
  }

  if (!BestReg) {
    // Nothing we can do: Report an error and keep going with an invalid
    // allocation.
    if (MI.isInlineAsm())
      MI.emitError("inline assembly requires more registers than available");
    else
      MI.emitError("ran out of registers during register allocation");

    LR.Error = true;
    LR.PhysReg = 0;
    return;
  }

  displacePhysReg(MI, BestReg);
  assignVirtToPhysReg(MI, LR, BestReg);
}

void RegAllocFast::allocVirtRegUndef(MachineOperand &MO) {
  assert(MO.isUndef() && "expected undef use");
  Register VirtReg = MO.getReg();
  assert(Register::isVirtualRegister(VirtReg) && "Expected virtreg");

  LiveRegMap::const_iterator LRI = findLiveVirtReg(VirtReg);
  MCPhysReg PhysReg;
  if (LRI != LiveVirtRegs.end() && LRI->PhysReg) {
    PhysReg = LRI->PhysReg;
  } else {
    const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
    ArrayRef<MCPhysReg> AllocationOrder = RegClassInfo.getOrder(&RC);
    assert(!AllocationOrder.empty() && "Allocation order must not be empty");
    PhysReg = AllocationOrder[0];
  }

  unsigned SubRegIdx = MO.getSubReg();
  if (SubRegIdx != 0) {
    PhysReg = TRI->getSubReg(PhysReg, SubRegIdx);
    MO.setSubReg(0);
  }
  MO.setReg(PhysReg);
  MO.setIsRenamable(true);
}

/// Variation of defineVirtReg() with special handling for livethrough regs
/// (tied or earlyclobber) that may interfere with preassigned uses.
void RegAllocFast::defineLiveThroughVirtReg(MachineInstr &MI, unsigned OpNum,
                                            Register VirtReg) {
  LiveRegMap::iterator LRI = findLiveVirtReg(VirtReg);
  if (LRI != LiveVirtRegs.end()) {
    MCPhysReg PrevReg = LRI->PhysReg;
    if (PrevReg != 0 && isRegUsedInInstr(PrevReg, true)) {
      LLVM_DEBUG(dbgs() << "Need new assignment for " << printReg(PrevReg, TRI)
                        << " (tied/earlyclobber resolution)\n");
      freePhysReg(PrevReg);
      LRI->PhysReg = 0;
      allocVirtReg(MI, *LRI, 0, true);
      MachineBasicBlock::iterator InsertBefore =
        std::next((MachineBasicBlock::iterator)MI.getIterator());
      LLVM_DEBUG(dbgs() << "Copy " << printReg(LRI->PhysReg, TRI) << " to "
                        << printReg(PrevReg, TRI) << '\n');
      BuildMI(*MBB, InsertBefore, MI.getDebugLoc(),
              TII->get(TargetOpcode::COPY), PrevReg)
        .addReg(LRI->PhysReg, llvm::RegState::Kill);
    }
    MachineOperand &MO = MI.getOperand(OpNum);
    if (MO.getSubReg() && !MO.isUndef()) {
      LRI->LastUse = &MI;
    }
  }
  return defineVirtReg(MI, OpNum, VirtReg, true);
}

/// Allocates a register for VirtReg definition. Typically the register is
/// already assigned from a use of the virtreg, however we still need to
/// perform an allocation if:
/// - It is a dead definition without any uses.
/// - The value is live out and all uses are in different basic blocks.
void RegAllocFast::defineVirtReg(MachineInstr &MI, unsigned OpNum,
                                 Register VirtReg, bool LookAtPhysRegUses) {
  assert(VirtReg.isVirtual() && "Not a virtual register");
  MachineOperand &MO = MI.getOperand(OpNum);
  LiveRegMap::iterator LRI;
  bool New;
  std::tie(LRI, New) = LiveVirtRegs.insert(LiveReg(VirtReg));
  if (New) {
    if (!MO.isDead()) {
      if (mayLiveOut(VirtReg)) {
        LRI->LiveOut = true;
      } else {
        // It is a dead def without the dead flag; add the flag now.
        MO.setIsDead(true);
      }
    }
  }
  if (LRI->PhysReg == 0)
    allocVirtReg(MI, *LRI, 0, LookAtPhysRegUses);
  else {
    assert(!isRegUsedInInstr(LRI->PhysReg, LookAtPhysRegUses) &&
           "TODO: preassign mismatch");
    LLVM_DEBUG(dbgs() << "In def of " << printReg(VirtReg, TRI)
                      << " use existing assignment to "
                      << printReg(LRI->PhysReg, TRI) << '\n');
  }

  MCPhysReg PhysReg = LRI->PhysReg;
  assert(PhysReg != 0 && "Register not assigned");
  if (LRI->Reloaded || LRI->LiveOut) {
    if (!MI.isImplicitDef()) {
      MachineBasicBlock::iterator SpillBefore =
          std::next((MachineBasicBlock::iterator)MI.getIterator());
      LLVM_DEBUG(dbgs() << "Spill Reason: LO: " << LRI->LiveOut << " RL: "
                        << LRI->Reloaded << '\n');
      bool Kill = LRI->LastUse == nullptr;
      spill(SpillBefore, VirtReg, PhysReg, Kill, LRI->LiveOut);
      LRI->LastUse = nullptr;
    }
    LRI->LiveOut = false;
    LRI->Reloaded = false;
  }
  if (MI.getOpcode() == TargetOpcode::BUNDLE) {
    BundleVirtRegsMap[VirtReg] = PhysReg;
  }
  markRegUsedInInstr(PhysReg);
  setPhysReg(MI, MO, PhysReg);
}

/// Allocates a register for a VirtReg use.
void RegAllocFast::useVirtReg(MachineInstr &MI, unsigned OpNum,
                              Register VirtReg) {
  assert(VirtReg.isVirtual() && "Not a virtual register");
  MachineOperand &MO = MI.getOperand(OpNum);
  LiveRegMap::iterator LRI;
  bool New;
  std::tie(LRI, New) = LiveVirtRegs.insert(LiveReg(VirtReg));
  if (New) {
    MachineOperand &MO = MI.getOperand(OpNum);
    if (!MO.isKill()) {
      if (mayLiveOut(VirtReg)) {
        LRI->LiveOut = true;
      } else {
        // It is a last (killing) use without the kill flag; add the flag now.
        MO.setIsKill(true);
      }
    }
  } else {
    assert((!MO.isKill() || LRI->LastUse == &MI) && "Invalid kill flag");
  }

  // If necessary allocate a register.
  if (LRI->PhysReg == 0) {
    assert(!MO.isTied() && "tied op should be allocated");
    Register Hint;
    if (MI.isCopy() && MI.getOperand(1).getSubReg() == 0) {
      Hint = MI.getOperand(0).getReg();
      assert(Hint.isPhysical() &&
             "Copy destination should already be assigned");
    }
    allocVirtReg(MI, *LRI, Hint, false);
    if (LRI->Error) {
      const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
      ArrayRef<MCPhysReg> AllocationOrder = RegClassInfo.getOrder(&RC);
      setPhysReg(MI, MO, *AllocationOrder.begin());
      return;
    }
  }

  LRI->LastUse = &MI;

  if (MI.getOpcode() == TargetOpcode::BUNDLE) {
    BundleVirtRegsMap[VirtReg] = LRI->PhysReg;
  }
  markRegUsedInInstr(LRI->PhysReg);
  setPhysReg(MI, MO, LRI->PhysReg);
}

/// Changes operand OpNum in MI the refer the PhysReg, considering subregs. This
/// may invalidate any operand pointers.  Return true if the operand kills its
/// register.
void RegAllocFast::setPhysReg(MachineInstr &MI, MachineOperand &MO,
                              MCPhysReg PhysReg) {
  if (!MO.getSubReg()) {
    MO.setReg(PhysReg);
    MO.setIsRenamable(true);
    return;
  }

  // Handle subregister index.
  MO.setReg(PhysReg ? TRI->getSubReg(PhysReg, MO.getSubReg()) : MCRegister());
  MO.setIsRenamable(true);
  // Note: We leave the subreg number around a little longer in case of defs.
  // This is so that the register freeing logic in allocateInstruction can still
  // recognize this as subregister defs. The code there will clear the number.
  if (!MO.isDef())
    MO.setSubReg(0);

  // A kill flag implies killing the full register. Add corresponding super
  // register kill.
  if (MO.isKill()) {
    MI.addRegisterKilled(PhysReg, TRI, true);
    return;
  }

  // A <def,read-undef> of a sub-register requires an implicit def of the full
  // register.
  if (MO.isDef() && MO.isUndef()) {
    if (MO.isDead())
      MI.addRegisterDead(PhysReg, TRI, true);
    else
      MI.addRegisterDefined(PhysReg, TRI);
  }
}

#ifndef NDEBUG

void RegAllocFast::dumpState() const {
  for (unsigned Unit = 1, UnitE = TRI->getNumRegUnits(); Unit != UnitE;
       ++Unit) {
    switch (unsigned VirtReg = RegUnitStates[Unit]) {
    case regFree:
      break;
    case regPreAssigned:
      dbgs() << " " << printRegUnit(Unit, TRI) << "[P]";
      break;
    case regLiveIn:
      llvm_unreachable("Should not have regLiveIn in map");
    default: {
      dbgs() << ' ' << printRegUnit(Unit, TRI) << '=' << printReg(VirtReg);
      LiveRegMap::const_iterator I = findLiveVirtReg(VirtReg);
      assert(I != LiveVirtRegs.end() && "have LiveVirtRegs entry");
      if (I->LiveOut || I->Reloaded) {
        dbgs() << '[';
        if (I->LiveOut) dbgs() << 'O';
        if (I->Reloaded) dbgs() << 'R';
        dbgs() << ']';
      }
      assert(TRI->hasRegUnit(I->PhysReg, Unit) && "inverse mapping present");
      break;
    }
    }
  }
  dbgs() << '\n';
  // Check that LiveVirtRegs is the inverse.
  for (const LiveReg &LR : LiveVirtRegs) {
    Register VirtReg = LR.VirtReg;
    assert(VirtReg.isVirtual() && "Bad map key");
    MCPhysReg PhysReg = LR.PhysReg;
    if (PhysReg != 0) {
      assert(Register::isPhysicalRegister(PhysReg) &&
             "mapped to physreg");
      for (MCRegUnitIterator UI(PhysReg, TRI); UI.isValid(); ++UI) {
        assert(RegUnitStates[*UI] == VirtReg && "inverse map valid");
      }
    }
  }
}
#endif

/// Count number of defs consumed from each register class by \p Reg
void RegAllocFast::addRegClassDefCounts(std::vector<unsigned> &RegClassDefCounts,
                                        Register Reg) const {
  assert(RegClassDefCounts.size() == TRI->getNumRegClasses());

  if (Reg.isVirtual()) {
    const TargetRegisterClass *OpRC = MRI->getRegClass(Reg);
    for (unsigned RCIdx = 0, RCIdxEnd = TRI->getNumRegClasses();
         RCIdx != RCIdxEnd; ++RCIdx) {
      const TargetRegisterClass *IdxRC = TRI->getRegClass(RCIdx);
      // FIXME: Consider aliasing sub/super registers.
      if (OpRC->hasSubClassEq(IdxRC))
        ++RegClassDefCounts[RCIdx];
    }

    return;
  }

  for (unsigned RCIdx = 0, RCIdxEnd = TRI->getNumRegClasses();
       RCIdx != RCIdxEnd; ++RCIdx) {
    const TargetRegisterClass *IdxRC = TRI->getRegClass(RCIdx);
    for (MCRegAliasIterator Alias(Reg, TRI, true); Alias.isValid(); ++Alias) {
      if (IdxRC->contains(*Alias)) {
        ++RegClassDefCounts[RCIdx];
        break;
      }
    }
  }
}

void RegAllocFast::allocateInstruction(MachineInstr &MI) {
  // The basic algorithm here is:
  // 1. Mark registers of def operands as free
  // 2. Allocate registers to use operands and place reload instructions for
  //    registers displaced by the allocation.
  //
  // However we need to handle some corner cases:
  // - pre-assigned defs and uses need to be handled before the other def/use
  //   operands are processed to avoid the allocation heuristics clashing with
  //   the pre-assignment.
  // - The "free def operands" step has to come last instead of first for tied
  //   operands and early-clobbers.

  UsedInInstr.clear();
  RegMasks.clear();
  BundleVirtRegsMap.clear();

  // Scan for special cases; Apply pre-assigned register defs to state.
  bool HasPhysRegUse = false;
  bool HasRegMask = false;
  bool HasVRegDef = false;
  bool HasDef = false;
  bool HasEarlyClobber = false;
  bool NeedToAssignLiveThroughs = false;
  for (MachineOperand &MO : MI.operands()) {
    if (MO.isReg()) {
      Register Reg = MO.getReg();
      if (Reg.isVirtual()) {
        if (MO.isDef()) {
          HasDef = true;
          HasVRegDef = true;
          if (MO.isEarlyClobber()) {
            HasEarlyClobber = true;
            NeedToAssignLiveThroughs = true;
          }
          if (MO.isTied() || (MO.getSubReg() != 0 && !MO.isUndef()))
            NeedToAssignLiveThroughs = true;
        }
      } else if (Reg.isPhysical()) {
        if (!MRI->isReserved(Reg)) {
          if (MO.isDef()) {
            HasDef = true;
            bool displacedAny = definePhysReg(MI, Reg);
            if (MO.isEarlyClobber())
              HasEarlyClobber = true;
            if (!displacedAny)
              MO.setIsDead(true);
          }
          if (MO.readsReg())
            HasPhysRegUse = true;
        }
      }
    } else if (MO.isRegMask()) {
      HasRegMask = true;
      RegMasks.push_back(MO.getRegMask());
    }
  }

  // Allocate virtreg defs.
  if (HasDef) {
    if (HasVRegDef) {
      // Special handling for early clobbers, tied operands or subregister defs:
      // Compared to "normal" defs these:
      // - Must not use a register that is pre-assigned for a use operand.
      // - In order to solve tricky inline assembly constraints we change the
      //   heuristic to figure out a good operand order before doing
      //   assignments.
      if (NeedToAssignLiveThroughs) {
        DefOperandIndexes.clear();
        PhysRegUses.clear();

        // Track number of defs which may consume a register from the class.
        std::vector<unsigned> RegClassDefCounts(TRI->getNumRegClasses(), 0);
        assert(RegClassDefCounts[0] == 0);

        LLVM_DEBUG(dbgs() << "Need to assign livethroughs\n");
        for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (!MO.isReg())
            continue;
          Register Reg = MO.getReg();
          if (MO.readsReg()) {
            if (Reg.isPhysical()) {
              LLVM_DEBUG(dbgs() << "mark extra used: " << printReg(Reg, TRI)
                                << '\n');
              markPhysRegUsedInInstr(Reg);
            }
          }

          if (MO.isDef()) {
            if (Reg.isVirtual())
              DefOperandIndexes.push_back(I);

            addRegClassDefCounts(RegClassDefCounts, Reg);
          }
        }

        llvm::sort(DefOperandIndexes, [&](uint16_t I0, uint16_t I1) {
          const MachineOperand &MO0 = MI.getOperand(I0);
          const MachineOperand &MO1 = MI.getOperand(I1);
          Register Reg0 = MO0.getReg();
          Register Reg1 = MO1.getReg();
          const TargetRegisterClass &RC0 = *MRI->getRegClass(Reg0);
          const TargetRegisterClass &RC1 = *MRI->getRegClass(Reg1);

          // Identify regclass that are easy to use up completely just in this
          // instruction.
          unsigned ClassSize0 = RegClassInfo.getOrder(&RC0).size();
          unsigned ClassSize1 = RegClassInfo.getOrder(&RC1).size();

          bool SmallClass0 = ClassSize0 < RegClassDefCounts[RC0.getID()];
          bool SmallClass1 = ClassSize1 < RegClassDefCounts[RC1.getID()];
          if (SmallClass0 > SmallClass1)
            return true;
          if (SmallClass0 < SmallClass1)
            return false;

          // Allocate early clobbers and livethrough operands first.
          bool Livethrough0 = MO0.isEarlyClobber() || MO0.isTied() ||
                              (MO0.getSubReg() == 0 && !MO0.isUndef());
          bool Livethrough1 = MO1.isEarlyClobber() || MO1.isTied() ||
                              (MO1.getSubReg() == 0 && !MO1.isUndef());
          if (Livethrough0 > Livethrough1)
            return true;
          if (Livethrough0 < Livethrough1)
            return false;

          // Tie-break rule: operand index.
          return I0 < I1;
        });

        for (uint16_t OpIdx : DefOperandIndexes) {
          MachineOperand &MO = MI.getOperand(OpIdx);
          LLVM_DEBUG(dbgs() << "Allocating " << MO << '\n');
          unsigned Reg = MO.getReg();
          if (MO.isEarlyClobber() || MO.isTied() ||
              (MO.getSubReg() && !MO.isUndef())) {
            defineLiveThroughVirtReg(MI, OpIdx, Reg);
          } else {
            defineVirtReg(MI, OpIdx, Reg);
          }
        }
      } else {
        // Assign virtual register defs.
        for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
          MachineOperand &MO = MI.getOperand(I);
          if (!MO.isReg() || !MO.isDef())
            continue;
          Register Reg = MO.getReg();
          if (Reg.isVirtual())
            defineVirtReg(MI, I, Reg);
        }
      }
    }

    // Free registers occupied by defs.
    // Iterate operands in reverse order, so we see the implicit super register
    // defs first (we added them earlier in case of <def,read-undef>).
    for (MachineOperand &MO : llvm::reverse(MI.operands())) {
      if (!MO.isReg() || !MO.isDef())
        continue;

      // subreg defs don't free the full register. We left the subreg number
      // around as a marker in setPhysReg() to recognize this case here.
      if (MO.getSubReg() != 0) {
        MO.setSubReg(0);
        continue;
      }

      assert((!MO.isTied() || !isClobberedByRegMasks(MO.getReg())) &&
             "tied def assigned to clobbered register");

      // Do not free tied operands and early clobbers.
      if (MO.isTied() || MO.isEarlyClobber())
        continue;
      Register Reg = MO.getReg();
      if (!Reg)
        continue;
      assert(Reg.isPhysical());
      if (MRI->isReserved(Reg))
        continue;
      freePhysReg(Reg);
      unmarkRegUsedInInstr(Reg);
    }
  }

  // Displace clobbered registers.
  if (HasRegMask) {
    assert(!RegMasks.empty() && "expected RegMask");
    // MRI bookkeeping.
    for (const auto *RM : RegMasks)
      MRI->addPhysRegsUsedFromRegMask(RM);

    // Displace clobbered registers.
    for (const LiveReg &LR : LiveVirtRegs) {
      MCPhysReg PhysReg = LR.PhysReg;
      if (PhysReg != 0 && isClobberedByRegMasks(PhysReg))
        displacePhysReg(MI, PhysReg);
    }
  }

  // Apply pre-assigned register uses to state.
  if (HasPhysRegUse) {
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.readsReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isPhysical())
        continue;
      if (MRI->isReserved(Reg))
        continue;
      bool displacedAny = usePhysReg(MI, Reg);
      if (!displacedAny && !MRI->isReserved(Reg))
        MO.setIsKill(true);
    }
  }

  // Allocate virtreg uses and insert reloads as necessary.
  bool HasUndefUse = false;
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse())
      continue;
    Register Reg = MO.getReg();
    if (!Reg.isVirtual())
      continue;

    if (MO.isUndef()) {
      HasUndefUse = true;
      continue;
    }


    // Populate MayLiveAcrossBlocks in case the use block is allocated before
    // the def block (removing the vreg uses).
    mayLiveIn(Reg);


    assert(!MO.isInternalRead() && "Bundles not supported");
    assert(MO.readsReg() && "reading use");
    useVirtReg(MI, I, Reg);
  }

  // Allocate undef operands. This is a separate step because in a situation
  // like  ` = OP undef %X, %X`    both operands need the same register assign
  // so we should perform the normal assignment first.
  if (HasUndefUse) {
    for (MachineOperand &MO : MI.uses()) {
      if (!MO.isReg() || !MO.isUse())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isVirtual())
        continue;

      assert(MO.isUndef() && "Should only have undef virtreg uses left");
      allocVirtRegUndef(MO);
    }
  }

  // Free early clobbers.
  if (HasEarlyClobber) {
    for (MachineOperand &MO : llvm::reverse(MI.operands())) {
      if (!MO.isReg() || !MO.isDef() || !MO.isEarlyClobber())
        continue;
      // subreg defs don't free the full register. We left the subreg number
      // around as a marker in setPhysReg() to recognize this case here.
      if (MO.getSubReg() != 0) {
        MO.setSubReg(0);
        continue;
      }

      Register Reg = MO.getReg();
      if (!Reg)
        continue;
      assert(Reg.isPhysical() && "should have register assigned");

      // We sometimes get odd situations like:
      //    early-clobber %x0 = INSTRUCTION %x0
      // which is semantically questionable as the early-clobber should
      // apply before the use. But in practice we consider the use to
      // happen before the early clobber now. Don't free the early clobber
      // register in this case.
      if (MI.readsRegister(Reg, TRI))
        continue;

      freePhysReg(Reg);
    }
  }

  LLVM_DEBUG(dbgs() << "<< " << MI);
  if (MI.isCopy() && MI.getOperand(0).getReg() == MI.getOperand(1).getReg() &&
      MI.getNumOperands() == 2) {
    LLVM_DEBUG(dbgs() << "Mark identity copy for removal\n");
    Coalesced.push_back(&MI);
  }
}

void RegAllocFast::handleDebugValue(MachineInstr &MI) {
  // Ignore DBG_VALUEs that aren't based on virtual registers. These are
  // mostly constants and frame indices.
  for (Register Reg : MI.getUsedDebugRegs()) {
    if (!Register::isVirtualRegister(Reg))
      continue;

    // Already spilled to a stackslot?
    int SS = StackSlotForVirtReg[Reg];
    if (SS != -1) {
      // Modify DBG_VALUE now that the value is in a spill slot.
      updateDbgValueForSpill(MI, SS, Reg);
      LLVM_DEBUG(dbgs() << "Rewrite DBG_VALUE for spilled memory: " << MI);
      continue;
    }

    // See if this virtual register has already been allocated to a physical
    // register or spilled to a stack slot.
    LiveRegMap::iterator LRI = findLiveVirtReg(Reg);
    SmallVector<MachineOperand *> DbgOps;
    for (MachineOperand &Op : MI.getDebugOperandsForReg(Reg))
      DbgOps.push_back(&Op);

    if (LRI != LiveVirtRegs.end() && LRI->PhysReg) {
      // Update every use of Reg within MI.
      for (auto &RegMO : DbgOps)
        setPhysReg(MI, *RegMO, LRI->PhysReg);
    } else {
      DanglingDbgValues[Reg].push_back(&MI);
    }

    // If Reg hasn't been spilled, put this DBG_VALUE in LiveDbgValueMap so
    // that future spills of Reg will have DBG_VALUEs.
    LiveDbgValueMap[Reg].append(DbgOps.begin(), DbgOps.end());
  }
}

void RegAllocFast::handleBundle(MachineInstr &MI) {
  MachineBasicBlock::instr_iterator BundledMI = MI.getIterator();
  ++BundledMI;
  while (BundledMI->isBundledWithPred()) {
    for (MachineOperand &MO : BundledMI->operands()) {
      if (!MO.isReg())
        continue;

      Register Reg = MO.getReg();
      if (!Reg.isVirtual())
        continue;

      DenseMap<Register, MCPhysReg>::iterator DI;
      DI = BundleVirtRegsMap.find(Reg);
      assert(DI != BundleVirtRegsMap.end() && "Unassigned virtual register");

      setPhysReg(MI, MO, DI->second);
    }

    ++BundledMI;
  }
}

void RegAllocFast::allocateBasicBlock(MachineBasicBlock &MBB) {
  this->MBB = &MBB;
  LLVM_DEBUG(dbgs() << "\nAllocating " << MBB);

  RegUnitStates.assign(TRI->getNumRegUnits(), regFree);
  assert(LiveVirtRegs.empty() && "Mapping not cleared from last block?");

  for (auto &LiveReg : MBB.liveouts())
    setPhysRegState(LiveReg.PhysReg, regPreAssigned);

  Coalesced.clear();

  // Traverse block in reverse order allocating instructions one by one.
  for (MachineInstr &MI : reverse(MBB)) {
    LLVM_DEBUG(
      dbgs() << "\n>> " << MI << "Regs:";
      dumpState()
    );

    // Special handling for debug values. Note that they are not allowed to
    // affect codegen of the other instructions in any way.
    if (MI.isDebugValue()) {
      handleDebugValue(MI);
      continue;
    }

    allocateInstruction(MI);

    // Once BUNDLE header is assigned registers, same assignments need to be
    // done for bundled MIs.
    if (MI.getOpcode() == TargetOpcode::BUNDLE) {
      handleBundle(MI);
    }
  }

  LLVM_DEBUG(
    dbgs() << "Begin Regs:";
    dumpState()
  );

  // Spill all physical registers holding virtual registers now.
  LLVM_DEBUG(dbgs() << "Loading live registers at begin of block.\n");
  reloadAtBegin(MBB);

  // Erase all the coalesced copies. We are delaying it until now because
  // LiveVirtRegs might refer to the instrs.
  for (MachineInstr *MI : Coalesced)
    MBB.erase(MI);
  NumCoalesced += Coalesced.size();

  for (auto &UDBGPair : DanglingDbgValues) {
    for (MachineInstr *DbgValue : UDBGPair.second) {
      assert(DbgValue->isDebugValue() && "expected DBG_VALUE");
      // Nothing to do if the vreg was spilled in the meantime.
      if (!DbgValue->hasDebugOperandForReg(UDBGPair.first))
        continue;
      LLVM_DEBUG(dbgs() << "Register did not survive for " << *DbgValue
                 << '\n');
      DbgValue->setDebugValueUndef();
    }
  }
  DanglingDbgValues.clear();

  LLVM_DEBUG(MBB.dump());
}

bool RegAllocFast::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** FAST REGISTER ALLOCATION **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  MRI = &MF.getRegInfo();
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  TRI = STI.getRegisterInfo();
  TII = STI.getInstrInfo();
  MFI = &MF.getFrameInfo();
  MRI->freezeReservedRegs(MF);
  RegClassInfo.runOnMachineFunction(MF);
  unsigned NumRegUnits = TRI->getNumRegUnits();
  UsedInInstr.clear();
  UsedInInstr.setUniverse(NumRegUnits);
  PhysRegUses.clear();
  PhysRegUses.setUniverse(NumRegUnits);

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  unsigned NumVirtRegs = MRI->getNumVirtRegs();
  StackSlotForVirtReg.resize(NumVirtRegs);
  LiveVirtRegs.setUniverse(NumVirtRegs);
  MayLiveAcrossBlocks.clear();
  MayLiveAcrossBlocks.resize(NumVirtRegs);

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineBasicBlock &MBB : MF)
    allocateBasicBlock(MBB);

  if (ClearVirtRegs) {
    // All machine operands and other references to virtual registers have been
    // replaced. Remove the virtual registers.
    MRI->clearVirtRegs();
  }

  StackSlotForVirtReg.clear();
  LiveDbgValueMap.clear();
  return true;
}

FunctionPass *llvm::createFastRegisterAllocator() {
  return new RegAllocFast();
}

FunctionPass *llvm::createFastRegisterAllocator(
  std::function<bool(const TargetRegisterInfo &TRI,
                     const TargetRegisterClass &RC)> Ftor, bool ClearVirtRegs) {
  return new RegAllocFast(Ftor, ClearVirtRegs);
}
