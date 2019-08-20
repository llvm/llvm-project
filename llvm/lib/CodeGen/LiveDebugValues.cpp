//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/ADT/SmallSet.h"
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
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DIBuilder.h"
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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");

// If @MI is a DBG_VALUE with debug value described by a defined
// register, returns the number of this register. In the other case, returns 0.
static Register isDbgValueDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue() && "expected a DBG_VALUE");
  assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
  // If location of variable is described using a register (directly
  // or indirectly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : Register();
}

namespace {

class LiveDebugValues : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;

  enum struct TransferKind { TransferCopy, TransferSpill, TransferRestore };

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

  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  /// Storage for identifying a potentially inlined instance of a variable,
  /// or a fragment thereof.
  class DebugVariable {
    const DILocalVariable *Variable;
    OptFragmentInfo Fragment;
    const DILocation *InlinedAt;

    /// Fragment that will overlap all other fragments. Used as default when
    /// caller demands a fragment.
    static const FragmentInfo DefaultFragment;

  public:
    DebugVariable(const DILocalVariable *Var, OptFragmentInfo &&FragmentInfo,
                  const DILocation *InlinedAt)
        : Variable(Var), Fragment(FragmentInfo), InlinedAt(InlinedAt) {}

    DebugVariable(const DILocalVariable *Var, OptFragmentInfo &FragmentInfo,
                  const DILocation *InlinedAt)
        : Variable(Var), Fragment(FragmentInfo), InlinedAt(InlinedAt) {}

    DebugVariable(const DILocalVariable *Var, const DIExpression *DIExpr,
                  const DILocation *InlinedAt)
        : DebugVariable(Var, DIExpr->getFragmentInfo(), InlinedAt) {}

    DebugVariable(const MachineInstr &MI)
        : DebugVariable(MI.getDebugVariable(),
                        MI.getDebugExpression()->getFragmentInfo(),
                        MI.getDebugLoc()->getInlinedAt()) {}

    const DILocalVariable *getVar() const { return Variable; }
    const OptFragmentInfo &getFragment() const { return Fragment; }
    const DILocation *getInlinedAt() const { return InlinedAt; }

    const FragmentInfo getFragmentDefault() const {
      return Fragment.getValueOr(DefaultFragment);
    }

    static bool isFragmentDefault(FragmentInfo &F) {
      return F == DefaultFragment;
    }

    bool operator==(const DebugVariable &Other) const {
      return std::tie(Variable, Fragment, InlinedAt) ==
             std::tie(Other.Variable, Other.Fragment, Other.InlinedAt);
    }

    bool operator<(const DebugVariable &Other) const {
      return std::tie(Variable, Fragment, InlinedAt) <
             std::tie(Other.Variable, Other.Fragment, Other.InlinedAt);
    }
  };

  friend struct llvm::DenseMapInfo<DebugVariable>;

  /// A pair of debug variable and value location.
  struct VarLoc {
    // The location at which a spilled variable resides. It consists of a
    // register and an offset.
    struct SpillLoc {
      unsigned SpillBase;
      int SpillOffset;
      bool operator==(const SpillLoc &Other) const {
        return SpillBase == Other.SpillBase && SpillOffset == Other.SpillOffset;
      }
    };

    const DebugVariable Var;
    const MachineInstr &MI; ///< Only used for cloning a new DBG_VALUE.
    mutable UserValueScopes UVS;
    enum VarLocKind {
      InvalidKind = 0,
      RegisterKind,
      SpillLocKind,
      ImmediateKind,
      EntryValueKind
    } Kind = InvalidKind;

    /// The value location. Stored separately to avoid repeatedly
    /// extracting it from MI.
    union {
      uint64_t RegNo;
      SpillLoc SpillLocation;
      uint64_t Hash;
      int64_t Immediate;
      const ConstantFP *FPImm;
      const ConstantInt *CImm;
    } Loc;

    VarLoc(const MachineInstr &MI, LexicalScopes &LS,
          VarLocKind K = InvalidKind)
        : Var(MI), MI(MI), UVS(MI.getDebugLoc(), LS){
      static_assert((sizeof(Loc) == sizeof(uint64_t)),
                    "hash does not cover all members of Loc");
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      if (int RegNo = isDbgValueDescribedByReg(MI)) {
        Kind = MI.isDebugEntryValue() ? EntryValueKind : RegisterKind;
        Loc.RegNo = RegNo;
      } else if (MI.getOperand(0).isImm()) {
        Kind = ImmediateKind;
        Loc.Immediate = MI.getOperand(0).getImm();
      } else if (MI.getOperand(0).isFPImm()) {
        Kind = ImmediateKind;
        Loc.FPImm = MI.getOperand(0).getFPImm();
      } else if (MI.getOperand(0).isCImm()) {
        Kind = ImmediateKind;
        Loc.CImm = MI.getOperand(0).getCImm();
      }
      assert((Kind != ImmediateKind || !MI.isDebugEntryValue()) &&
             "entry values must be register locations");
    }

    /// The constructor for spill locations.
    VarLoc(const MachineInstr &MI, unsigned SpillBase, int SpillOffset,
           LexicalScopes &LS)
        : Var(MI), MI(MI), UVS(MI.getDebugLoc(), LS) {
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      Kind = SpillLocKind;
      Loc.SpillLocation = {SpillBase, SpillOffset};
    }

    // Is the Loc field a constant or constant object?
    bool isConstant() const { return Kind == ImmediateKind; }

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
      return Kind == Other.Kind && Var == Other.Var &&
             Loc.Hash == Other.Loc.Hash;
    }

    /// This operator guarantees that VarLocs are sorted by Variable first.
    bool operator<(const VarLoc &Other) const {
      return std::tie(Var, Kind, Loc.Hash) <
             std::tie(Other.Var, Other.Kind, Other.Loc.Hash);
    }
  };

  using DebugParamMap = SmallDenseMap<const DILocalVariable *, MachineInstr *>;
  using VarLocMap = UniqueVector<VarLoc>;
  using VarLocSet = SparseBitVector<>;
  using VarLocInMBB = SmallDenseMap<const MachineBasicBlock *, VarLocSet>;
  struct TransferDebugPair {
    MachineInstr *TransferInst;
    MachineInstr *DebugInst;
  };
  using TransferMap = SmallVector<TransferDebugPair, 4>;

  // Types for recording sets of variable fragments that overlap. For a given
  // local variable, we record all other fragments of that variable that could
  // overlap it, to reduce search time.
  using FragmentOfVar =
      std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
  using OverlapMap =
      DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// This holds the working set of currently open ranges. For fast
  /// access, this is done both as a set of VarLocIDs, and a map of
  /// DebugVariable to recent VarLocID. Note that a DBG_VALUE ends all
  /// previous open ranges for the same variable.
  class OpenRangesSet {
    VarLocSet VarLocs;
    SmallDenseMap<DebugVariable, unsigned, 8> Vars;
    OverlapMap &OverlappingFragments;

  public:
    OpenRangesSet(OverlapMap &_OLapMap) : OverlappingFragments(_OLapMap) {}

    const VarLocSet &getVarLocs() const { return VarLocs; }

    /// Terminate all open ranges for Var by removing it from the set.
    void erase(DebugVariable Var);

    /// Terminate all open ranges listed in \c KillSet by removing
    /// them from the set.
    void erase(const VarLocSet &KillSet, const VarLocMap &VarLocIDs) {
      VarLocs.intersectWithComplement(KillSet);
      for (unsigned ID : KillSet)
        Vars.erase(VarLocIDs[ID].Var);
    }

    /// Insert a new range into the set.
    void insert(unsigned VarLocID, DebugVariable Var) {
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
  /// If a given instruction is identified as a spill, return the spill location
  /// and set \p Reg to the spilled register.
  Optional<VarLoc::SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                                  MachineFunction *MF,
                                                  unsigned &Reg);
  /// Given a spill instruction, extract the register and offset used to
  /// address the spill location in a target independent way.
  VarLoc::SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);
  void insertTransferDebugPair(MachineInstr &MI, OpenRangesSet &OpenRanges,
                               TransferMap &Transfers, VarLocMap &VarLocIDs,
                               unsigned OldVarID, TransferKind Kind,
                               unsigned NewReg = 0);

  void transferDebugValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                          VarLocMap &VarLocIDs);
  void transferSpillOrRestoreInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                                  VarLocMap &VarLocIDs, TransferMap &Transfers);
  void emitEntryValues(MachineInstr &MI, OpenRangesSet &OpenRanges,
                       VarLocMap &VarLocIDs, TransferMap &Transfers,
                       DebugParamMap &DebugEntryVals,
                       SparseBitVector<> &KillSet);
  void transferRegisterCopy(MachineInstr &MI, OpenRangesSet &OpenRanges,
                            VarLocMap &VarLocIDs, TransferMap &Transfers);
  void transferRegisterDef(MachineInstr &MI, OpenRangesSet &OpenRanges,
                           VarLocMap &VarLocIDs, TransferMap &Transfers,
                           DebugParamMap &DebugEntryVals);
  bool transferTerminatorInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocInMBB &OutLocs, const VarLocMap &VarLocIDs);

  bool process(MachineInstr &MI, OpenRangesSet &OpenRanges,
               VarLocInMBB &OutLocs, VarLocMap &VarLocIDs,
               TransferMap &Transfers, DebugParamMap &DebugEntryVals,
               bool transferChanges, OverlapMap &OverlapFragments,
               VarToFragments &SeenFragments);

  void accumulateFragmentMap(MachineInstr &MI, VarToFragments &SeenFragments,
                             OverlapMap &OLapMap);

  bool join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
            const VarLocMap &VarLocIDs,
            SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
            SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks);

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

namespace llvm {

template <> struct DenseMapInfo<LiveDebugValues::DebugVariable> {
  using DV = LiveDebugValues::DebugVariable;
  using OptFragmentInfo = LiveDebugValues::OptFragmentInfo;
  using FragmentInfo = LiveDebugValues::FragmentInfo;

  // Empty key: no key should be generated that has no DILocalVariable.
  static inline DV getEmptyKey() {
    return DV(nullptr, OptFragmentInfo(), nullptr);
  }

  // Difference in tombstone is that the Optional is meaningful
  static inline DV getTombstoneKey() {
    return DV(nullptr, OptFragmentInfo({0, 0}), nullptr);
  }

  static unsigned getHashValue(const DV &D) {
    unsigned HV = 0;
    const OptFragmentInfo &Fragment = D.getFragment();
    if (Fragment)
      HV = DenseMapInfo<FragmentInfo>::getHashValue(*Fragment);

    return hash_combine(D.getVar(), HV, D.getInlinedAt());
  }

  static bool isEqual(const DV &A, const DV &B) { return A == B; }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

const DIExpression::FragmentInfo
    LiveDebugValues::DebugVariable::DefaultFragment = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::min()};

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

/// Erase a variable from the set of open ranges, and additionally erase any
/// fragments that may overlap it.
void LiveDebugValues::OpenRangesSet::erase(DebugVariable Var) {
  // Erasure helper.
  auto DoErase = [this](DebugVariable VarToErase) {
    auto It = Vars.find(VarToErase);
    if (It != Vars.end()) {
      unsigned ID = It->second;
      VarLocs.reset(ID);
      Vars.erase(It);
    }
  };

  // Erase the variable/fragment that ends here.
  DoErase(Var);

  // Extract the fragment. Interpret an empty fragment as one that covers all
  // possible bits.
  FragmentInfo ThisFragment = Var.getFragmentDefault();

  // There may be fragments that overlap the designated fragment. Look them up
  // in the pre-computed overlap map, and erase them too.
  auto MapIt = OverlappingFragments.find({Var.getVar(), ThisFragment});
  if (MapIt != OverlappingFragments.end()) {
    for (auto Fragment : MapIt->second) {
      LiveDebugValues::OptFragmentInfo FragmentHolder;
      if (!DebugVariable::isFragmentDefault(Fragment))
        FragmentHolder = LiveDebugValues::OptFragmentInfo(Fragment);
      DoErase({Var.getVar(), FragmentHolder, Var.getInlinedAt()});
    }
  }
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
    const VarLocSet &L = V.lookup(&BB);
    if (L.empty())
      continue;
    Out << "MBB: " << BB.getNumber() << ":\n";
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

LiveDebugValues::VarLoc::SpillLoc
LiveDebugValues::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
  assert(MI.hasOneMemOperand() &&
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  unsigned Reg;
  int Offset = TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
  return {Reg, Offset};
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
void LiveDebugValues::transferDebugValue(const MachineInstr &MI,
                                         OpenRangesSet &OpenRanges,
                                         VarLocMap &VarLocIDs) {
  if (!MI.isDebugValue())
    return;
  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  // End all previous ranges of Var.
  DebugVariable V(Var, Expr, InlinedAt);
  OpenRanges.erase(V);

  // Add the VarLoc to OpenRanges from this DBG_VALUE.
  unsigned ID;
  if (isDbgValueDescribedByReg(MI) || MI.getOperand(0).isImm() ||
      MI.getOperand(0).isFPImm() || MI.getOperand(0).isCImm()) {
    // Use normal VarLoc constructor for registers and immediates.
    VarLoc VL(MI, LS);
    ID = VarLocIDs.insert(VL);
    OpenRanges.insert(ID, VL.Var);
  } else if (MI.hasOneMemOperand()) {
    // It's a stack spill -- fetch spill base and offset.
    VarLoc::SpillLoc SpillLocation = extractSpillBaseRegAndOffset(MI);
    VarLoc VL(MI, SpillLocation.SpillBase, SpillLocation.SpillOffset, LS);
    ID = VarLocIDs.insert(VL);
    OpenRanges.insert(ID, VL.Var);
  } else {
    // This must be an undefined location. We should leave OpenRanges closed.
    assert(MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == 0 &&
           "Unexpected non-undef DBG_VALUE encountered");
  }
}

void LiveDebugValues::emitEntryValues(MachineInstr &MI,
                                      OpenRangesSet &OpenRanges,
                                      VarLocMap &VarLocIDs,
                                      TransferMap &Transfers,
                                      DebugParamMap &DebugEntryVals,
                                      SparseBitVector<> &KillSet) {
  MachineFunction *MF = MI.getParent()->getParent();
  for (unsigned ID : KillSet) {
    if (!VarLocIDs[ID].Var.getVar()->isParameter())
      continue;

    const MachineInstr *CurrDebugInstr = &VarLocIDs[ID].MI;

    // If parameter's DBG_VALUE is not in the map that means we can't
    // generate parameter's entry value.
    if (!DebugEntryVals.count(CurrDebugInstr->getDebugVariable()))
      continue;

    auto ParamDebugInstr = DebugEntryVals[CurrDebugInstr->getDebugVariable()];
    DIExpression *NewExpr = DIExpression::prepend(
        ParamDebugInstr->getDebugExpression(), DIExpression::EntryValue);
    MachineInstr *EntryValDbgMI =
        BuildMI(*MF, ParamDebugInstr->getDebugLoc(), ParamDebugInstr->getDesc(),
                ParamDebugInstr->isIndirectDebugValue(),
                ParamDebugInstr->getOperand(0).getReg(),
                ParamDebugInstr->getDebugVariable(), NewExpr);

    if (ParamDebugInstr->isIndirectDebugValue())
      EntryValDbgMI->getOperand(1).setImm(
          ParamDebugInstr->getOperand(1).getImm());

    Transfers.push_back({&MI, EntryValDbgMI});
    VarLoc VL(*EntryValDbgMI, LS);
    unsigned EntryValLocID = VarLocIDs.insert(VL);
    OpenRanges.insert(EntryValLocID, VL.Var);
  }
}

/// Create new TransferDebugPair and insert it in \p Transfers. The VarLoc
/// with \p OldVarID should be deleted form \p OpenRanges and replaced with
/// new VarLoc. If \p NewReg is different than default zero value then the
/// new location will be register location created by the copy like instruction,
/// otherwise it is variable's location on the stack.
void LiveDebugValues::insertTransferDebugPair(
    MachineInstr &MI, OpenRangesSet &OpenRanges, TransferMap &Transfers,
    VarLocMap &VarLocIDs, unsigned OldVarID, TransferKind Kind,
    unsigned NewReg) {
  const MachineInstr *DebugInstr = &VarLocIDs[OldVarID].MI;
  MachineFunction *MF = MI.getParent()->getParent();
  MachineInstr *NewDebugInstr;

  auto ProcessVarLoc = [&MI, &OpenRanges, &Transfers, &DebugInstr,
                        &VarLocIDs](VarLoc &VL, MachineInstr *NewDebugInstr) {
    unsigned LocId = VarLocIDs.insert(VL);

    // Close this variable's previous location range.
    DebugVariable V(*DebugInstr);
    OpenRanges.erase(V);

    OpenRanges.insert(LocId, VL.Var);
    // The newly created DBG_VALUE instruction NewDebugInstr must be inserted
    // after MI. Keep track of the pairing.
    TransferDebugPair MIP = {&MI, NewDebugInstr};
    Transfers.push_back(MIP);
  };

  // End all previous ranges of Var.
  OpenRanges.erase(VarLocIDs[OldVarID].Var);
  switch (Kind) {
  case TransferKind::TransferCopy: {
    assert(NewReg &&
           "No register supplied when handling a copy of a debug value");
    // Create a DBG_VALUE instruction to describe the Var in its new
    // register location.
    NewDebugInstr = BuildMI(
        *MF, DebugInstr->getDebugLoc(), DebugInstr->getDesc(),
        DebugInstr->isIndirectDebugValue(), NewReg,
        DebugInstr->getDebugVariable(), DebugInstr->getDebugExpression());
    if (DebugInstr->isIndirectDebugValue())
      NewDebugInstr->getOperand(1).setImm(DebugInstr->getOperand(1).getImm());
    VarLoc VL(*NewDebugInstr, LS);
    ProcessVarLoc(VL, NewDebugInstr);
    LLVM_DEBUG(dbgs() << "Creating DBG_VALUE inst for register copy: ";
               NewDebugInstr->print(dbgs(), /*IsStandalone*/false,
                                    /*SkipOpers*/false, /*SkipDebugLoc*/false,
                                    /*AddNewLine*/true, TII));
    return;
  }
  case TransferKind::TransferSpill: {
    // Create a DBG_VALUE instruction to describe the Var in its spilled
    // location.
    VarLoc::SpillLoc SpillLocation = extractSpillBaseRegAndOffset(MI);
    auto *SpillExpr = DIExpression::prepend(DebugInstr->getDebugExpression(),
                                            DIExpression::ApplyOffset,
                                            SpillLocation.SpillOffset);
    NewDebugInstr = BuildMI(
        *MF, DebugInstr->getDebugLoc(), DebugInstr->getDesc(), true,
        SpillLocation.SpillBase, DebugInstr->getDebugVariable(), SpillExpr);
    VarLoc VL(*NewDebugInstr, SpillLocation.SpillBase,
              SpillLocation.SpillOffset, LS);
    ProcessVarLoc(VL, NewDebugInstr);
    LLVM_DEBUG(dbgs() << "Creating DBG_VALUE inst for spill: ";
               NewDebugInstr->print(dbgs(), /*IsStandalone*/false,
                                    /*SkipOpers*/false, /*SkipDebugLoc*/false,
                                    /*AddNewLine*/true, TII));
    return;
  }
  case TransferKind::TransferRestore: {
    assert(NewReg &&
           "No register supplied when handling a restore of a debug value");
    MachineFunction *MF = MI.getMF();
    DIBuilder DIB(*const_cast<Function &>(MF->getFunction()).getParent());

    const DIExpression *NewExpr;
    if (auto Fragment = DebugInstr->getDebugExpression()->getFragmentInfo())
      NewExpr = *DIExpression::createFragmentExpression(DIB.createExpression(),
        Fragment->OffsetInBits, Fragment->SizeInBits);
    else
      NewExpr = DIB.createExpression();

    NewDebugInstr =
        BuildMI(*MF, DebugInstr->getDebugLoc(), DebugInstr->getDesc(), false,
                NewReg, DebugInstr->getDebugVariable(), NewExpr);
    VarLoc VL(*NewDebugInstr, LS);
    ProcessVarLoc(VL, NewDebugInstr);
    LLVM_DEBUG(dbgs() << "Creating DBG_VALUE inst for register restore: ";
               NewDebugInstr->print(dbgs(), /*IsStandalone*/false,
                                    /*SkipOpers*/false, /*SkipDebugLoc*/false,
                                    /*AddNewLine*/true, TII));
    return;
  }
  }
  llvm_unreachable("Invalid transfer kind");
}

/// A definition of a register may mark the end of a range.
void LiveDebugValues::transferRegisterDef(
    MachineInstr &MI, OpenRangesSet &OpenRanges, VarLocMap &VarLocIDs,
    TransferMap &Transfers, DebugParamMap &DebugEntryVals) {
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

  if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>()) {
    auto &TM = TPC->getTM<TargetMachine>();
    if (TM.Options.EnableDebugEntryValues)
      emitEntryValues(MI, OpenRanges, VarLocIDs, Transfers, DebugEntryVals,
                      KillSet);
  }
}

/// Decide if @MI is a spill instruction and return true if it is. We use 2
/// criteria to make this decision:
/// - Is this instruction a store to a spill slot?
/// - Is there a register operand that is both used and killed?
/// TODO: Store optimization can fold spills into other stores (including
/// other spills). We do not handle this yet (more than one memory operand).
bool LiveDebugValues::isSpillInstruction(const MachineInstr &MI,
                                         MachineFunction *MF, unsigned &Reg) {
  SmallVector<const MachineMemOperand*, 1> Accesses;

  // TODO: Handle multiple stores folded into one.
  if (!MI.hasOneMemOperand())
    return false;

  if (!MI.getSpillSize(TII) && !MI.getFoldedSpillSize(TII))
    return false; // This is not a spill instruction, since no valid size was
                  // returned from either function.

  auto isKilledReg = [&](const MachineOperand MO, unsigned &Reg) {
    if (!MO.isReg() || !MO.isUse()) {
      Reg = 0;
      return false;
    }
    Reg = MO.getReg();
    return MO.isKill();
  };

  for (const MachineOperand &MO : MI.operands()) {
    // In a spill instruction generated by the InlineSpiller the spilled
    // register has its kill flag set.
    if (isKilledReg(MO, Reg))
      return true;
    if (Reg != 0) {
      // Check whether next instruction kills the spilled register.
      // FIXME: Current solution does not cover search for killed register in
      // bundles and instructions further down the chain.
      auto NextI = std::next(MI.getIterator());
      // Skip next instruction that points to basic block end iterator.
      if (MI.getParent()->end() == NextI)
        continue;
      unsigned RegNext;
      for (const MachineOperand &MONext : NextI->operands()) {
        // Return true if we came across the register from the
        // previous spill instruction that is killed in NextI.
        if (isKilledReg(MONext, RegNext) && RegNext == Reg)
          return true;
      }
    }
  }
  // Return false if we didn't find spilled register.
  return false;
}

Optional<LiveDebugValues::VarLoc::SpillLoc>
LiveDebugValues::isRestoreInstruction(const MachineInstr &MI,
                                      MachineFunction *MF, unsigned &Reg) {
  if (!MI.hasOneMemOperand())
    return None;

  // FIXME: Handle folded restore instructions with more than one memory
  // operand.
  if (MI.getRestoreSize(TII)) {
    Reg = MI.getOperand(0).getReg();
    return extractSpillBaseRegAndOffset(MI);
  }
  return None;
}

/// A spilled register may indicate that we have to end the current range of
/// a variable and create a new one for the spill location.
/// A restored register may indicate the reverse situation.
/// We don't want to insert any instructions in process(), so we just create
/// the DBG_VALUE without inserting it and keep track of it in \p Transfers.
/// It will be inserted into the BB when we're done iterating over the
/// instructions.
void LiveDebugValues::transferSpillOrRestoreInst(MachineInstr &MI,
                                                 OpenRangesSet &OpenRanges,
                                                 VarLocMap &VarLocIDs,
                                                 TransferMap &Transfers) {
  MachineFunction *MF = MI.getMF();
  TransferKind TKind;
  unsigned Reg;
  Optional<VarLoc::SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  if (isSpillInstruction(MI, MF, Reg)) {
    TKind = TransferKind::TransferSpill;
    LLVM_DEBUG(dbgs() << "Recognized as spill: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return;
    TKind = TransferKind::TransferRestore;
    LLVM_DEBUG(dbgs() << "Recognized as restore: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  }
  // Check if the register or spill location is the location of a debug value.
  // FIXME: Don't create a spill transfer if there is a complex expression,
  // because we currently cannot recover the original expression on restore.
  for (unsigned ID : OpenRanges.getVarLocs()) {
    const MachineInstr *DebugInstr = &VarLocIDs[ID].MI;

    if (TKind == TransferKind::TransferSpill &&
        VarLocIDs[ID].isDescribedByReg() == Reg &&
        !DebugInstr->getDebugExpression()->isComplex()) {
      LLVM_DEBUG(dbgs() << "Spilling Register " << printReg(Reg, TRI) << '('
                        << VarLocIDs[ID].Var.getVar()->getName() << ")\n");
    } else if (TKind == TransferKind::TransferRestore &&
               VarLocIDs[ID].Loc.SpillLocation == *Loc) {
      LLVM_DEBUG(dbgs() << "Restoring Register " << printReg(Reg, TRI) << '('
                        << VarLocIDs[ID].Var.getVar()->getName() << ")\n");
    } else
      continue;
    insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, ID, TKind,
                            Reg);
    return;
  }
}

/// If \p MI is a register copy instruction, that copies a previously tracked
/// value from one register to another register that is callee saved, we
/// create new DBG_VALUE instruction  described with copy destination register.
void LiveDebugValues::transferRegisterCopy(MachineInstr &MI,
                                           OpenRangesSet &OpenRanges,
                                           VarLocMap &VarLocIDs,
                                           TransferMap &Transfers) {
  const MachineOperand *SrcRegOp, *DestRegOp;

  if (!TII->isCopyInstr(MI, SrcRegOp, DestRegOp) || !SrcRegOp->isKill() ||
      !DestRegOp->isDef())
    return;

  auto isCalleSavedReg = [&](unsigned Reg) {
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  };

  unsigned SrcReg = SrcRegOp->getReg();
  unsigned DestReg = DestRegOp->getReg();

  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register location, which is callee
  // saved, is going to stay unclobbered longer, even if it is killed.
  if (!isCalleSavedReg(DestReg))
    return;

  for (unsigned ID : OpenRanges.getVarLocs()) {
    if (VarLocIDs[ID].isDescribedByReg() == SrcReg) {
      insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, ID,
                              TransferKind::TransferCopy, DestReg);
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
  if (!(MI.isTerminator() || (&MI == &CurMBB->back())))
    return false;

  if (OpenRanges.empty())
    return false;

  LLVM_DEBUG(for (unsigned ID
                  : OpenRanges.getVarLocs()) {
    // Copy OpenRanges to OutLocs, if not already present.
    dbgs() << "Add to OutLocs in MBB #" << CurMBB->getNumber() << ":  ";
    VarLocIDs[ID].dump();
  });
  VarLocSet &VLS = OutLocs[CurMBB];
  Changed = VLS |= OpenRanges.getVarLocs();
  // New OutLocs set may be different due to spill, restore or register
  // copy instruction processing.
  if (Changed)
    VLS = OpenRanges.getVarLocs();
  OpenRanges.clear();
  return Changed;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
/// \param SeenFragments Map from DILocalVariable to all fragments of that
///           Variable which are known to exist.
/// \param OverlappingFragments The overlap map being constructed, from one
///           Var/Fragment pair to a vector of fragments known to overlap.
void LiveDebugValues::accumulateFragmentMap(MachineInstr &MI,
                                            VarToFragments &SeenFragments,
                                            OverlapMap &OverlappingFragments) {
  DebugVariable MIVar(MI);
  FragmentInfo ThisFragment = MIVar.getFragmentDefault();

  // If this is the first sighting of this variable, then we are guaranteed
  // there are currently no overlapping fragments either. Initialize the set
  // of seen fragments, record no overlaps for the current one, and return.
  auto SeenIt = SeenFragments.find(MIVar.getVar());
  if (SeenIt == SeenFragments.end()) {
    SmallSet<FragmentInfo, 4> OneFragment;
    OneFragment.insert(ThisFragment);
    SeenFragments.insert({MIVar.getVar(), OneFragment});

    OverlappingFragments.insert({{MIVar.getVar(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlappingFragments.insert({{MIVar.getVar(), ThisFragment}, {}});
  if (!IsInOLapMap.second)
    return;

  auto &ThisFragmentsOverlaps = IsInOLapMap.first->second;
  auto &AllSeenFragments = SeenIt->second;

  // Otherwise, examine all other seen fragments for this variable, with "this"
  // fragment being a previously unseen fragment. Record any pair of
  // overlapping fragments.
  for (auto &ASeenFragment : AllSeenFragments) {
    // Does this previously seen fragment overlap?
    if (DIExpression::fragmentsOverlap(ThisFragment, ASeenFragment)) {
      // Yes: Mark the current fragment as being overlapped.
      ThisFragmentsOverlaps.push_back(ASeenFragment);
      // Mark the previously seen fragment as being overlapped by the current
      // one.
      auto ASeenFragmentsOverlaps =
          OverlappingFragments.find({MIVar.getVar(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlappingFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

/// This routine creates OpenRanges and OutLocs.
bool LiveDebugValues::process(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocInMBB &OutLocs, VarLocMap &VarLocIDs,
                              TransferMap &Transfers, DebugParamMap &DebugEntryVals,
                              bool transferChanges,
                              OverlapMap &OverlapFragments,
                              VarToFragments &SeenFragments) {
  bool Changed = false;
  transferDebugValue(MI, OpenRanges, VarLocIDs);
  transferRegisterDef(MI, OpenRanges, VarLocIDs, Transfers,
                      DebugEntryVals);
  if (transferChanges) {
    transferRegisterCopy(MI, OpenRanges, VarLocIDs, Transfers);
    transferSpillOrRestoreInst(MI, OpenRanges, VarLocIDs, Transfers);
  } else {
    // Build up a map of overlapping fragments on the first run through.
    if (MI.isDebugValue())
      accumulateFragmentMap(MI, SeenFragments, OverlapFragments);
  }
  Changed = transferTerminatorInst(MI, OpenRanges, OutLocs, VarLocIDs);
  return Changed;
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool LiveDebugValues::join(
    MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
    const VarLocMap &VarLocIDs,
    SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  VarLocSet InLocsT; // Temporary incoming locations.

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  for (auto p : MBB.predecessors()) {
    // Ignore unvisited predecessor blocks.  As we are processing
    // the blocks in reverse post-order any unvisited block can
    // be considered to not remove any incoming values.
    if (!Visited.count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }
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

    LLVM_DEBUG({
      if (!InLocsT.empty()) {
        for (auto ID : InLocsT)
          dbgs() << "  gathered candidate incoming var: "
                 << VarLocIDs[ID].Var.getVar()->getName() << "\n";
      }
    });

    NumVisited++;
  }

  // Filter out DBG_VALUES that are out of scope.
  VarLocSet KillSet;
  bool IsArtificial = ArtificialBlocks.count(&MBB);
  if (!IsArtificial) {
    for (auto ID : InLocsT) {
      if (!VarLocIDs[ID].dominates(MBB)) {
        KillSet.set(ID);
        LLVM_DEBUG({
          auto Name = VarLocIDs[ID].Var.getVar()->getName();
          dbgs() << "  killing " << Name << ", it doesn't dominate MBB\n";
        });
      }
    }
  }
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
    // a new DBG_VALUE. process() will end this range however appropriate.
    const VarLoc &DiffIt = VarLocIDs[ID];
    const MachineInstr *DebugInstr = &DiffIt.MI;
    MachineInstr *MI = nullptr;
    if (DiffIt.isConstant()) {
      MachineOperand MO(DebugInstr->getOperand(0));
      MI = BuildMI(MBB, MBB.instr_begin(), DebugInstr->getDebugLoc(),
                   DebugInstr->getDesc(), false, MO,
                   DebugInstr->getDebugVariable(),
                   DebugInstr->getDebugExpression());
    } else {
      MI = BuildMI(MBB, MBB.instr_begin(), DebugInstr->getDebugLoc(),
                   DebugInstr->getDesc(), DebugInstr->isIndirectDebugValue(),
                   DebugInstr->getOperand(0).getReg(),
                   DebugInstr->getDebugVariable(),
                   DebugInstr->getDebugExpression());
      if (DebugInstr->isIndirectDebugValue())
        MI->getOperand(1).setImm(DebugInstr->getOperand(1).getImm());
    }
    LLVM_DEBUG(dbgs() << "Inserted: "; MI->dump(););
    ILS.set(ID);
    ++NumInserted;
    Changed = true;
  }
  return Changed;
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  bool Changed = false;
  bool OLChanged = false;
  bool MBBJoined = false;

  VarLocMap VarLocIDs;         // Map VarLoc<>unique ID for use in bitvectors.
  OverlapMap OverlapFragments; // Map of overlapping variable fragments
  OpenRangesSet OpenRanges(OverlapFragments);
                              // Ranges that are open until end of bb.
  VarLocInMBB OutLocs;        // Ranges that exist beyond bb.
  VarLocInMBB InLocs;         // Ranges that are incoming after joining.
  TransferMap Transfers;      // DBG_VALUEs associated with spills.

  VarToFragments SeenFragments;

  // Blocks which are artificial, i.e. blocks which exclusively contain
  // instructions without locations, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;

  enum : bool { dontTransferChanges = false, transferChanges = true };

  // Besides parameter's modification, check whether a DBG_VALUE is inlined
  // in order to deduce whether the variable that it tracks comes from
  // a different function. If that is the case we can't track its entry value.
  auto IsUnmodifiedFuncParam = [&](const MachineInstr &MI) {
    auto *DIVar = MI.getDebugVariable();
    return DIVar->isParameter() && DIVar->isNotModified() &&
           !MI.getDebugLoc()->getInlinedAt();
  };

  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  unsigned FP = TRI->getFrameRegister(MF);
  auto IsRegOtherThanSPAndFP = [&](const MachineOperand &Op) -> bool {
    return Op.isReg() && Op.getReg() != SP && Op.getReg() != FP;
  };

  // Working set of currently collected debug variables mapped to DBG_VALUEs
  // representing candidates for production of debug entry values.
  DebugParamMap DebugEntryVals;

  MachineBasicBlock &First_MBB = *(MF.begin());
  // Only in the case of entry MBB collect DBG_VALUEs representing
  // function parameters in order to generate debug entry values for them.
  // Currently, we generate debug entry values only for parameters that are
  // unmodified throughout the function and located in a register.
  // TODO: Add support for parameters that are described as fragments.
  // TODO: Add support for modified arguments that can be expressed
  // by using its entry value.
  // TODO: Add support for local variables that are expressed in terms of
  // parameters entry values.
  for (auto &MI : First_MBB)
    if (MI.isDebugValue() && IsUnmodifiedFuncParam(MI) &&
        !MI.isIndirectDebugValue() && IsRegOtherThanSPAndFP(MI.getOperand(0)) &&
        !DebugEntryVals.count(MI.getDebugVariable()) &&
        !MI.getDebugExpression()->isFragment())
      DebugEntryVals[MI.getDebugVariable()] = &MI;

  // Initialize every mbb with OutLocs.
  // We are not looking at any spill instructions during the initial pass
  // over the BBs. The LiveDebugVariables pass has already created DBG_VALUE
  // instructions for spills of registers that are known to be user variables
  // within the BB in which the spill occurs.
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      process(MI, OpenRanges, OutLocs, VarLocIDs, Transfers, DebugEntryVals,
              dontTransferChanges, OverlapFragments, SeenFragments);
    }
    // Add any entry DBG_VALUE instructions necessitated by parameter
    // clobbering.
    for (auto &TR : Transfers) {
      MBB.insertAfter(MachineBasicBlock::iterator(*TR.TransferInst),
                     TR.DebugInst);
    }
    Transfers.clear();
  }

  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                              "OutLocs after initialization", dbgs()));

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    Worklist.push(RPONumber);
    ++RPONumber;
  }
  // This is a standard "union of predecessor outs" dataflow problem.
  // To solve it, we perform join() and process() using the two worklist method
  // until the ranges converge.
  // Ranges have converged when both worklists are empty.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    LLVM_DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      Worklist.pop();
      MBBJoined =
          join(*MBB, OutLocs, InLocs, VarLocIDs, Visited, ArtificialBlocks);
      Visited.insert(MBB);
      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;
        // Now that we have started to extend ranges across BBs we need to
        // examine spill instructions to see whether they spill registers that
        // correspond to user variables.
        for (auto &MI : *MBB)
          OLChanged |=
              process(MI, OpenRanges, OutLocs, VarLocIDs, Transfers,
                      DebugEntryVals, transferChanges, OverlapFragments,
                      SeenFragments);

        // Add any DBG_VALUE instructions necessitated by spills.
        for (auto &TR : Transfers)
          MBB->insertAfter(MachineBasicBlock::iterator(*TR.TransferInst),
                           TR.DebugInst);
        Transfers.clear();

        LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                                    "OutLocs after propagating", dbgs()));
        LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs,
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

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs, "Final OutLocs", dbgs()));
  LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs, "Final InLocs", dbgs()));
  return Changed;
}

bool LiveDebugValues::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().getSubprogram())
    // LiveDebugValues will already have removed all DBG_VALUEs.
    return false;

  // Skip functions from NoDebug compilation units.
  if (MF.getFunction().getSubprogram()->getUnit()->getEmissionKind() ==
      DICompileUnit::NoDebug)
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  TFI->determineCalleeSaves(MF, CalleeSavedRegs,
                            make_unique<RegScavenger>().get());
  LS.initialize(MF);

  bool Changed = ExtendRanges(MF);
  return Changed;
}
