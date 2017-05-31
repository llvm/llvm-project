//===------ LiveDebugValues.cpp - Tracking Debug Value MIs ----------------===//
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

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <list>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "live-debug-values"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");

namespace {

// \brief If @MI is a DBG_VALUE with debug value described by a defined
// register, returns the number of this register. In the other case, returns 0.
static unsigned isDbgValueDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue() && "expected a DBG_VALUE");
  assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
  // If location of variable is described using a register (directly
  // or indirectly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : 0;
}

class LiveDebugValues : public MachineFunctionPass {

private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  /// Based on std::pair so it can be used as an index into a DenseMap.
  typedef std::pair<const DILocalVariable *, const DILocation *>
      DebugVariableBase;
  /// A potentially inlined instance of a variable.
  struct DebugVariable : public DebugVariableBase {
    DebugVariable(const DILocalVariable *Var, const DILocation *InlinedAt)
        : DebugVariableBase(Var, InlinedAt) {}

    const DILocalVariable *getVar() const { return this->first; };
    const DILocation *getInlinedAt() const { return this->second; };

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

    enum { InvalidKind = 0, RegisterKind } Kind;

    /// The value location. Stored separately to avoid repeatedly
    /// extracting it from MI.
    union {
      struct {
        uint32_t RegNo;
        uint32_t Offset;
      } RegisterLoc;
      uint64_t Hash;
    } Loc;

    VarLoc(const MachineInstr &MI)
        : Var(MI.getDebugVariable(), MI.getDebugLoc()->getInlinedAt()), MI(MI),
          Kind(InvalidKind) {
      static_assert((sizeof(Loc) == sizeof(uint64_t)),
                    "hash does not cover all members of Loc");
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      if (int RegNo = isDbgValueDescribedByReg(MI)) {
        Kind = RegisterKind;
        Loc.RegisterLoc.RegNo = RegNo;
        uint64_t Offset =
            MI.isIndirectDebugValue() ? MI.getOperand(1).getImm() : 0;
        // We don't support offsets larger than 4GiB here. They are
        // slated to be replaced with DIExpressions anyway.
        if (Offset >= (1ULL << 32))
          Kind = InvalidKind;
        else
          Loc.RegisterLoc.Offset = Offset;
      }
    }

    /// If this variable is described by a register, return it,
    /// otherwise return 0.
    unsigned isDescribedByReg() const {
      if (Kind == RegisterKind)
        return Loc.RegisterLoc.RegNo;
      return 0;
    }

    void dump() const { MI.dump(); }

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

  typedef UniqueVector<VarLoc> VarLocMap;
  typedef SparseBitVector<> VarLocSet;
  typedef SmallDenseMap<const MachineBasicBlock *, VarLocSet> VarLocInMBB;

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

  void transferDebugValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                          VarLocMap &VarLocIDs);
  void transferRegisterDef(MachineInstr &MI, OpenRangesSet &OpenRanges,
                           const VarLocMap &VarLocIDs);
  bool transferTerminatorInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocInMBB &OutLocs, const VarLocMap &VarLocIDs);
  bool transfer(MachineInstr &MI, OpenRangesSet &OpenRanges,
                VarLocInMBB &OutLocs, VarLocMap &VarLocIDs);

  bool join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
            const VarLocMap &VarLocIDs);

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
        MachineFunctionProperties::Property::AllVRegsAllocated);
  }

  /// Print to ostream with a message.
  void printVarLocInMBB(const MachineFunction &MF, const VarLocInMBB &V,
                        const VarLocMap &VarLocIDs, const char *msg,
                        raw_ostream &Out) const;

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char LiveDebugValues::ID = 0;
char &llvm::LiveDebugValuesID = LiveDebugValues::ID;
INITIALIZE_PASS(LiveDebugValues, "livedebugvalues", "Live DEBUG_VALUE analysis",
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

void LiveDebugValues::printVarLocInMBB(const MachineFunction &MF,
                                       const VarLocInMBB &V,
                                       const VarLocMap &VarLocIDs,
                                       const char *msg,
                                       raw_ostream &Out) const {
  for (const MachineBasicBlock &BB : MF) {
    const auto &L = V.lookup(&BB);
    Out << "MBB: " << BB.getName() << ":\n";
    for (unsigned VLL : L) {
      const VarLoc &VL = VarLocIDs[VLL];
      Out << " Var: " << VL.Var.getVar()->getName();
      Out << " MI: ";
      VL.dump();
      Out << "\n";
    }
  }
  Out << "\n";
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
    VarLoc VL(MI);
    unsigned ID = VarLocIDs.insert(VL);
    OpenRanges.insert(ID, VL.Var);
  }
}

/// A definition of a register may mark the end of a range.
void LiveDebugValues::transferRegisterDef(MachineInstr &MI,
                                          OpenRangesSet &OpenRanges,
                                          const VarLocMap &VarLocIDs) {
  MachineFunction *MF = MI.getParent()->getParent();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  SparseBitVector<> KillSet;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        TRI->isPhysicalRegister(MO.getReg())) {
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
                               VarLocInMBB &OutLocs, VarLocMap &VarLocIDs) {
  bool Changed = false;
  transferDebugValue(MI, OpenRanges, VarLocIDs);
  transferRegisterDef(MI, OpenRanges, VarLocIDs);
  Changed = transferTerminatorInst(MI, OpenRanges, OutLocs, VarLocIDs);
  return Changed;
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool LiveDebugValues::join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs,
                           VarLocInMBB &InLocs, const VarLocMap &VarLocIDs) {
  DEBUG(dbgs() << "join MBB: " << MBB.getName() << "\n");
  bool Changed = false;

  VarLocSet InLocsT; // Temporary incoming locations.

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  for (auto p : MBB.predecessors()) {
    auto OL = OutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == OutLocs.end())
      return false;

    // Just copy over the Out locs to incoming locs for the first predecessor.
    if (p == *MBB.pred_begin()) {
      InLocsT = OL->second;
      continue;
    }
    // Join with this predecessor.
    InLocsT &= OL->second;
  }

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
                DMI->isIndirectDebugValue(), DMI->getOperand(0).getReg(), 0,
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

  VarLocMap VarLocIDs;   // Map VarLoc<>unique ID for use in bitvectors.
  OpenRangesSet OpenRanges; // Ranges that are open until end of bb.
  VarLocInMBB OutLocs;   // Ranges that exist beyond bb.
  VarLocInMBB InLocs;    // Ranges that are incoming after joining.

  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;

  // Initialize every mbb with OutLocs.
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      transfer(MI, OpenRanges, OutLocs, VarLocIDs);

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
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      Worklist.pop();
      MBBJoined = join(*MBB, OutLocs, InLocs, VarLocIDs);

      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;
        for (auto &MI : *MBB)
          OLChanged |= transfer(MI, OpenRanges, OutLocs, VarLocIDs);

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
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();

  bool Changed = false;

  Changed |= ExtendRanges(MF);

  return Changed;
}
