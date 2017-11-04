//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This pass implements a data flow analysis that propagates debug location
/// information by inserting additional DBG_VALUE instructions into the machine
/// instruction stream. The pass internally builds debug location liveness
/// ranges to determine the points where additional DBG_VALUEs need to be
/// inserted.
///
/// This is a separate pass from DbgValueHistoryCalculator to facilitate
/// testing and improve modularity.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");

// \brief If @MI is a DBG_VALUE with debug value described by a defined
// register, returns the number of this register. In the other case, returns 0.
static unsigned isDbgValueDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue() && "expected a DBG_VALUE");
  assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
  // If location of variable is described using a register (directly
  // or indirectly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : 0;
}

namespace {

class LiveDebugValues : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  LexicalScopes LS;

  /// Keeps track of lexical scopes associated with a user value's source
  /// location.
  class UserValueScopes {
    DebugLoc DL;
    LexicalScopes &LS;
    SmallPtrSet<const MachineBasicBlock *, 4> LBlocks;

  public:
    UserValueScopes(DebugLoc D, LexicalScopes &L) : DL(std::move(D)), LS(L) {}

    /// Return true if current scope dominates at least one machine
    /// instruction in a given machine basic block.
    bool dominates(MachineBasicBlock *MBB) {
      if (LBlocks.empty())
        LS.getMachineBasicBlocks(DL, LBlocks);
      return LBlocks.count(MBB) != 0 || LS.dominates(DL, MBB);
    }
  };

  /// Based on std::pair so it can be used as an index into a DenseMap.
  using DebugVariableBase =
      std::pair<const DILocalVariable *, const DILocation *>;
  /// A potentially inlined instance of a variable.
  struct DebugVariable : public DebugVariableBase {
    DebugVariable(const DILocalVariable *Var, const DILocation *InlinedAt)
        : DebugVariableBase(Var, InlinedAt) {}

    const DILocalVariable *getVar() const { return this->first; }
    const DILocation *getInlinedAt() const { return this->second; }

    bool operator<(const DebugVariable &DV) const {
      if (getVar() == DV.getVar())
        return getInlinedAt() < DV.getInlinedAt();
      return getVar() < DV.getVar();
    }
  };

  /// A pair of debug variable and value location.
  struct VarLoc {
    const DebugVariable Var;
    const MachineInstr &MI; ///< Only used for cloning a new DBG_VALUE.
    mutable UserValueScopes UVS;
    enum { InvalidKind = 0, RegisterKind } Kind = InvalidKind;

    /// The value location. Stored separately to avoid repeatedly
    /// extracting it from MI.
    union {
      uint64_t RegNo;
      uint64_t Hash;
    } Loc;

    VarLoc(const MachineInstr &MI, LexicalScopes &LS)
        : Var(MI.getDebugVariable(), MI.getDebugLoc()->getInlinedAt()), MI(MI),
          UVS(MI.getDebugLoc(), LS) {
      static_assert((sizeof(Loc) == sizeof(uint64_t)),
                    "hash does not cover all members of Loc");
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      if (int RegNo = isDbgValueDescribedByReg(MI)) {
        Kind = RegisterKind;
        Loc.RegNo = RegNo;
      }
    }

    /// If this variable is described by a register, return it,
    /// otherwise return 0.
    unsigned isDescribedByReg() const {
      if (Kind == RegisterKind)
        return Loc.RegNo;
      return 0;
    }

    /// Determine whether the lexical scope of this value's debug location
    /// dominates MBB.
    bool dominates(MachineBasicBlock &MBB) const { return UVS.dominates(&MBB); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    LLVM_DUMP_METHOD void dump() const { MI.dump(); }
#endif

    bool operator==(const VarLoc &Other) const {
      return Var == Other.Var && Loc.Hash == Other.Loc.Hash;
    }

    /// This operator guarantees that VarLocs are sorted by Variable first.
    bool operator<(const VarLoc &Other) const {
      if (Var == Other.Var)
        return Loc.Hash < Other.Loc.Hash;
      return Var < Other.Var;
    }
  };

  using VarLocMap = UniqueVector<VarLoc>;
  using VarLocSet = SparseBitVector<>;
  using VarLocInMBB = SmallDenseMap<const MachineBasicBlock *, VarLocSet>;
  struct SpillDebugPair {
    MachineInstr *SpillInst;
    MachineInstr *DebugInst;
  };
  using SpillMap = SmallVector<SpillDebugPair, 4>;

  /// This holds the working set of currently open ranges. For fast
  /// access, this is done both as a set of VarLocIDs, and a map of
  /// DebugVariable to recent VarLocID. Note that a DBG_VALUE ends all
  /// previous open ranges for the same variable.
  class OpenRangesSet {
    VarLocSet VarLocs;
    SmallDenseMap<DebugVariableBase, unsigned, 8> Vars;

  public:
    const VarLocSet &getVarLocs() const { return VarLocs; }

    /// Terminate all open ranges for Var by removing it from the set.
    void erase(DebugVariable Var) {
      auto It = Vars.find(Var);
      if (It != Vars.end()) {
        unsigned ID = It->second;
        VarLocs.reset(ID);
        Vars.erase(It);
      }
    }

    /// Terminate all open ranges listed in \c KillSet by removing
    /// them from the set.
    void erase(const VarLocSet &KillSet, const VarLocMap &VarLocIDs) {
      VarLocs.intersectWithComplement(KillSet);
      for (unsigned ID : KillSet)
        Vars.erase(VarLocIDs[ID].Var);
    }

    /// Insert a new range into the set.
    void insert(unsigned VarLocID, DebugVariableBase Var) {
      VarLocs.set(VarLocID);
      Vars.insert({Var, VarLocID});
    }

    /// Empty the set.
    void clear() {
      VarLocs.clear();
      Vars.clear();
    }

    /// Return whether the set is empty or not.
    bool empty() const {
      assert(Vars.empty() == VarLocs.empty() && "open ranges are inconsistent");
      return VarLocs.empty();
    }
  };

  bool isSpillInstruction(const MachineInstr &MI, MachineFunction *MF,
                          unsigned &Reg);
  int extractSpillBaseRegAndOffset(const MachineInstr &MI, unsigned &Reg);

  void transferDebugValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                          VarLocMap &VarLocIDs);
  void transferSpillInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                         VarLocMap &VarLocIDs, SpillMap &Spills);
  void transferRegisterDef(MachineInstr &MI, OpenRangesSet &OpenRanges,
                           const VarLocMap &VarLocIDs);
  bool transferTerminatorInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocInMBB &OutLocs, const VarLocMap &VarLocIDs);
  bool transfer(MachineInstr &MI, OpenRangesSet &OpenRanges,
                VarLocInMBB &OutLocs, VarLocMap &VarLocIDs, SpillMap &Spills,
                bool transferSpills);

  bool join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
            const VarLocMap &VarLocIDs,
            SmallPtrSet<const MachineBasicBlock *, 16> &Visited);

  bool ExtendRanges(MachineFunction &MF);

public:
  static char ID;

  /// Default construct and initialize the pass.
  LiveDebugValues();

  /// Tell the pass manager which passes we depend on and what
  /// information we preserve.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  /// Print to ostream with a message.
  void printVarLocInMBB(const MachineFunction &MF, const VarLocInMBB &V,
                        const VarLocMap &VarLocIDs, const char *msg,
                        raw_ostream &Out) const;

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char LiveDebugValues::ID = 0;

char &llvm::LiveDebugValuesID = LiveDebugValues::ID;

INITIALIZE_PASS(LiveDebugValues, DEBUG_TYPE, "Live DEBUG_VALUE analysis",
                false, false)

/// Default construct and initialize the pass.
LiveDebugValues::LiveDebugValues() : MachineFunctionPass(ID) {
  initializeLiveDebugValuesPass(*PassRegistry::getPassRegistry());
}

/// Tell the pass manager which passes we depend on and what information we
/// preserve.
void LiveDebugValues::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
void LiveDebugValues::printVarLocInMBB(const MachineFunction &MF,
                                       const VarLocInMBB &V,
                                       const VarLocMap &VarLocIDs,
                                       const char *msg,
                                       raw_ostream &Out) const {
  Out << '\n' << msg << '\n';
  for (const MachineBasicBlock &BB : MF) {
    const auto &L = V.lookup(&BB);
    Out << "MBB: " << BB.getName() << ":\n";
    for (unsigned VLL : L) {
      const VarLoc &VL = VarLocIDs[VLL];
      Out << " Var: " << VL.Var.getVar()->getName();
      Out << " MI: ";
      VL.dump();
    }
  }
  Out << "\n";
}
#endif

/// Given a spill instruction, extract the register and offset used to
/// address the spill location in a target independent way.
int LiveDebugValues::extractSpillBaseRegAndOffset(const MachineInstr &MI,
                                                  unsigned &Reg) {
  assert(MI.hasOneMemOperand() && 
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  return TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
void LiveDebugValues::transferDebugValue(const MachineInstr &MI,
                                         OpenRangesSet &OpenRanges,
                                         VarLocMap &VarLocIDs) {
  if (!MI.isDebugValue())
    return;
  const DILocalVariable *Var = MI.getDebugVariable();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  // End all previous ranges of Var.
  DebugVariable V(Var, InlinedAt);
  OpenRanges.erase(V);

  // Add the VarLoc to OpenRanges from this DBG_VALUE.
  // TODO: Currently handles DBG_VALUE which has only reg as location.
  if (isDbgValueDescribedByReg(MI)) {
    VarLoc VL(MI, LS);
    unsigned ID = VarLocIDs.insert(VL);
    OpenRanges.insert(ID, VL.Var);
  }
}

/// A definition of a register may mark the end of a range.
void LiveDebugValues::transferRegisterDef(MachineInstr &MI,
                                          OpenRangesSet &OpenRanges,
                                          const VarLocMap &VarLocIDs) {
  MachineFunction *MF = MI.getMF();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  SparseBitVector<> KillSet;
  for (const MachineOperand &MO : MI.operands()) {
    // Determine whether the operand is a register def.  Assume that call
    // instructions never clobber SP, because some backends (e.g., AArch64)
    // never list SP in the regmask.
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        TRI->isPhysicalRegister(MO.getReg()) &&
        !(MI.isCall() && MO.getReg() == SP)) {
      // Remove ranges of all aliased registers.
      for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
        for (unsigned ID : OpenRanges.getVarLocs())
          if (VarLocIDs[ID].isDescribedByReg() == *RAI)
            KillSet.set(ID);
    } else if (MO.isRegMask()) {
      // Remove ranges of all clobbered registers. Register masks don't usually
      // list SP as preserved.  While the debug info may be off for an
      // instruction or two around callee-cleanup calls, transferring the
      // DEBUG_VALUE across the call is still a better user experience.
      for (unsigned ID : OpenRanges.getVarLocs()) {
        unsigned Reg = VarLocIDs[ID].isDescribedByReg();
        if (Reg && Reg != SP && MO.clobbersPhysReg(Reg))
          KillSet.set(ID);
      }
    }
  }
  OpenRanges.erase(KillSet, VarLocIDs);
}

/// Decide if @MI is a spill instruction and return true if it is. We use 2
/// criteria to make this decision:
/// - Is this instruction a store to a spill slot?
/// - Is there a register operand that is both used and killed?
/// TODO: Store optimization can fold spills into other stores (including
/// other spills). We do not handle this yet (more than one memory operand).
bool LiveDebugValues::isSpillInstruction(const MachineInstr &MI,
                                         MachineFunction *MF, unsigned &Reg) {
  const MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  int FI;
  const MachineMemOperand *MMO;

  // TODO: Handle multiple stores folded into one. 
  if (!MI.hasOneMemOperand())
    return false;

  // To identify a spill instruction, use the same criteria as in AsmPrinter.
  if (!((TII->isStoreToStackSlotPostFE(MI, FI) ||
         TII->hasStoreToStackSlot(MI, MMO, FI)) &&
        FrameInfo.isSpillSlotObjectIndex(FI)))
    return false;

  // In a spill instruction generated by the InlineSpiller the spilled register
  // has its kill flag set. Return false if we don't find such a register.
  Reg = 0;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isUse() && MO.isKill()) {
      Reg = MO.getReg();
      break;
    }
  }
  return Reg != 0;
}

/// A spilled register may indicate that we have to end the current range of
/// a variable and create a new one for the spill location.
/// We don't want to insert any instructions in transfer(), so we just create
/// the DBG_VALUE witout inserting it and keep track of it in @Spills.
/// It will be inserted into the BB when we're done iterating over the
/// instructions.
void LiveDebugValues::transferSpillInst(MachineInstr &MI,
                                        OpenRangesSet &OpenRanges,
                                        VarLocMap &VarLocIDs,
                                        SpillMap &Spills) {
  unsigned Reg;
  MachineFunction *MF = MI.getMF();
  if (!isSpillInstruction(MI, MF, Reg))
    return;

  // Check if the register is the location of a debug value.
  for (unsigned ID : OpenRanges.getVarLocs()) {
    if (VarLocIDs[ID].isDescribedByReg() == Reg) {
      DEBUG(dbgs() << "Spilling Register " << PrintReg(Reg, TRI) << '('
                   << VarLocIDs[ID].Var.getVar()->getName() << ")\n");

      // Create a DBG_VALUE instruction to describe the Var in its spilled
      // location, but don't insert it yet to avoid invalidating the
      // iterator in our caller.
      unsigned SpillBase;
      int SpillOffset = extractSpillBaseRegAndOffset(MI, SpillBase);
      const MachineInstr *DMI = &VarLocIDs[ID].MI;
      auto *SpillExpr = DIExpression::prepend(
          DMI->getDebugExpression(), DIExpression::NoDeref, SpillOffset);
      MachineInstr *SpDMI =
          BuildMI(*MF, DMI->getDebugLoc(), DMI->getDesc(), true, SpillBase,
                  DMI->getDebugVariable(), SpillExpr);
      DEBUG(dbgs() << "Creating DBG_VALUE inst for spill: ";
            SpDMI->print(dbgs(), false, TII));

      // The newly created DBG_VALUE instruction SpDMI must be inserted after
      // MI. Keep track of the pairing.
      SpillDebugPair MIP = {&MI, SpDMI};
      Spills.push_back(MIP);

      // End all previous ranges of Var.
      OpenRanges.erase(VarLocIDs[ID].Var);

      // Add the VarLoc to OpenRanges.
      VarLoc VL(*SpDMI, LS);
      unsigned SpillLocID = VarLocIDs.insert(VL);
      OpenRanges.insert(SpillLocID, VL.Var);
      return;
    }
  }
}

/// Terminate all open ranges at the end of the current basic block.
bool LiveDebugValues::transferTerminatorInst(MachineInstr &MI,
                                             OpenRangesSet &OpenRanges,
                                             VarLocInMBB &OutLocs,
                                             const VarLocMap &VarLocIDs) {
  bool Changed = false;
  const MachineBasicBlock *CurMBB = MI.getParent();
  if (!(MI.isTerminator() || (&MI == &CurMBB->instr_back())))
    return false;

  if (OpenRanges.empty())
    return false;

  DEBUG(for (unsigned ID : OpenRanges.getVarLocs()) {
          // Copy OpenRanges to OutLocs, if not already present.
          dbgs() << "Add to OutLocs: "; VarLocIDs[ID].dump();
        });
  VarLocSet &VLS = OutLocs[CurMBB];
  Changed = VLS |= OpenRanges.getVarLocs();
  OpenRanges.clear();
  return Changed;
}

/// This routine creates OpenRanges and OutLocs.
bool LiveDebugValues::transfer(MachineInstr &MI, OpenRangesSet &OpenRanges,
                               VarLocInMBB &OutLocs, VarLocMap &VarLocIDs,
                               SpillMap &Spills, bool transferSpills) {
  bool Changed = false;
  transferDebugValue(MI, OpenRanges, VarLocIDs);
  transferRegisterDef(MI, OpenRanges, VarLocIDs);
  if (transferSpills)
    transferSpillInst(MI, OpenRanges, VarLocIDs, Spills);
  Changed = transferTerminatorInst(MI, OpenRanges, OutLocs, VarLocIDs);
  return Changed;
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool LiveDebugValues::join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs,
                           VarLocInMBB &InLocs, const VarLocMap &VarLocIDs,
                           SmallPtrSet<const MachineBasicBlock *, 16> &Visited) {
  DEBUG(dbgs() << "join MBB: " << MBB.getName() << "\n");
  bool Changed = false;

  VarLocSet InLocsT; // Temporary incoming locations.

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  for (auto p : MBB.predecessors()) {
    // Ignore unvisited predecessor blocks.  As we are processing
    // the blocks in reverse post-order any unvisited block can
    // be considered to not remove any incoming values.
    if (!Visited.count(p))
      continue;
    auto OL = OutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == OutLocs.end())
      return false;

    // Just copy over the Out locs to incoming locs for the first visited
    // predecessor, and for all other predecessors join the Out locs.
    if (!NumVisited)
      InLocsT = OL->second;
    else
      InLocsT &= OL->second;
    NumVisited++;
  }

  // Filter out DBG_VALUES that are out of scope.
  VarLocSet KillSet;
  for (auto ID : InLocsT)
    if (!VarLocIDs[ID].dominates(MBB))
      KillSet.set(ID);
  InLocsT.intersectWithComplement(KillSet);

  // As we are processing blocks in reverse post-order we
  // should have processed at least one predecessor, unless it
  // is the entry block which has no predecessor.
  assert((NumVisited || MBB.pred_empty()) &&
         "Should have processed at least one predecessor");
  if (InLocsT.empty())
    return false;

  VarLocSet &ILS = InLocs[&MBB];

  // Insert DBG_VALUE instructions, if not already inserted.
  VarLocSet Diff = InLocsT;
  Diff.intersectWithComplement(ILS);
  for (auto ID : Diff) {
    // This VarLoc is not found in InLocs i.e. it is not yet inserted. So, a
    // new range is started for the var from the mbb's beginning by inserting
    // a new DBG_VALUE. transfer() will end this range however appropriate.
    const VarLoc &DiffIt = VarLocIDs[ID];
    const MachineInstr *DMI = &DiffIt.MI;
    MachineInstr *MI =
        BuildMI(MBB, MBB.instr_begin(), DMI->getDebugLoc(), DMI->getDesc(),
                DMI->isIndirectDebugValue(), DMI->getOperand(0).getReg(),
                DMI->getDebugVariable(), DMI->getDebugExpression());
    if (DMI->isIndirectDebugValue())
      MI->getOperand(1).setImm(DMI->getOperand(1).getImm());
    DEBUG(dbgs() << "Inserted: "; MI->dump(););
    ILS.set(ID);
    ++NumInserted;
    Changed = true;
  }
  return Changed;
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  DEBUG(dbgs() << "\nDebug Range Extension\n");

  bool Changed = false;
  bool OLChanged = false;
  bool MBBJoined = false;

  VarLocMap VarLocIDs;      // Map VarLoc<>unique ID for use in bitvectors.
  OpenRangesSet OpenRanges; // Ranges that are open until end of bb.
  VarLocInMBB OutLocs;      // Ranges that exist beyond bb.
  VarLocInMBB InLocs;       // Ranges that are incoming after joining.
  SpillMap Spills;          // DBG_VALUEs associated with spills.

  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;

  // Initialize every mbb with OutLocs.
  // We are not looking at any spill instructions during the initial pass
  // over the BBs. The LiveDebugVariables pass has already created DBG_VALUE
  // instructions for spills of registers that are known to be user variables
  // within the BB in which the spill occurs.
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      transfer(MI, OpenRanges, OutLocs, VarLocIDs, Spills,
               /*transferSpills=*/false);

  DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs, "OutLocs after initialization",
                         dbgs()));

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    Worklist.push(RPONumber);
    ++RPONumber;
  }
  // This is a standard "union of predecessor outs" dataflow problem.
  // To solve it, we perform join() and transfer() using the two worklist method
  // until the ranges converge.
  // Ranges have converged when both worklists are empty.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      Worklist.pop();
      MBBJoined = join(*MBB, OutLocs, InLocs, VarLocIDs, Visited);
      Visited.insert(MBB);
      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;
        // Now that we have started to extend ranges across BBs we need to
        // examine spill instructions to see whether they spill registers that
        // correspond to user variables.
        for (auto &MI : *MBB)
          OLChanged |= transfer(MI, OpenRanges, OutLocs, VarLocIDs, Spills,
                                /*transferSpills=*/true);

        // Add any DBG_VALUE instructions necessitated by spills.
        for (auto &SP : Spills)
          MBB->insertAfter(MachineBasicBlock::iterator(*SP.SpillInst),
                           SP.DebugInst);
        Spills.clear();

        DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                               "OutLocs after propagating", dbgs()));
        DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs,
                               "InLocs after propagating", dbgs()));

        if (OLChanged) {
          OLChanged = false;
          for (auto s : MBB->successors())
            if (OnPending.insert(s).second) {
              Pending.push(BBToOrder[s]);
            }
        }
      }
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs, "Final OutLocs", dbgs()));
  DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs, "Final InLocs", dbgs()));
  return Changed;
}

bool LiveDebugValues::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction()->getSubprogram())
    // LiveDebugValues will already have removed all DBG_VALUEs.
    return false;

  // Skip functions from NoDebug compilation units.
  if (MF.getFunction()->getSubprogram()->getUnit()->getEmissionKind() ==
      DICompileUnit::NoDebug)
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  LS.initialize(MF);

  bool Changed = ExtendRanges(MF);
  return Changed;
}
