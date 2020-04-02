//===- LiveDebugVariables.cpp - Tracking debug info variables -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveDebugVariables analysis.
//
// Remove all DBG_VALUE instructions referencing virtual registers and replace
// them with a data structure tracking where live user variables are kept - in a
// virtual register or in a stack slot.
//
// Allow the data structure to be updated during register allocation when values
// are moved between registers and stack slots. Finally emit new DBG_VALUE
// instructions after register allocation is complete.
//
//===----------------------------------------------------------------------===//

#include "LiveDebugVariables.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "livedebugvars"

static cl::opt<bool>
EnableLDV("live-debug-variables", cl::init(true),
          cl::desc("Enable the live debug variables pass"), cl::Hidden);

STATISTIC(NumInsertedDebugValues, "Number of DBG_VALUEs inserted");
STATISTIC(NumInsertedDebugLabels, "Number of DBG_LABELs inserted");

char LiveDebugVariables::ID = 0;

INITIALIZE_PASS_BEGIN(LiveDebugVariables, DEBUG_TYPE,
                "Debug Variable Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(LiveDebugVariables, DEBUG_TYPE,
                "Debug Variable Analysis", false, false)

void LiveDebugVariables::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTree>();
  AU.addRequiredTransitive<LiveIntervals>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

LiveDebugVariables::LiveDebugVariables() : MachineFunctionPass(ID) {
  initializeLiveDebugVariablesPass(*PassRegistry::getPassRegistry());
}

enum : unsigned { UndefLocNo = ~0U };

/// Describes a debug variable value by location number and expression along
/// with some flags about the original usage of the location.
class DbgVariableValue {
public:
  DbgVariableValue(unsigned LocNo, bool WasIndirect,
                   const DIExpression &Expression)
      : LocNo(LocNo), WasIndirect(WasIndirect), Expression(&Expression) {
    assert(getLocNo() == LocNo && "location truncation");
  }

  DbgVariableValue() : LocNo(0), WasIndirect(0) {}

  const DIExpression *getExpression() const { return Expression; }
  unsigned getLocNo() const {
    // Fix up the undef location number, which gets truncated.
    return LocNo == INT_MAX ? UndefLocNo : LocNo;
  }
  bool getWasIndirect() const { return WasIndirect; }
  bool isUndef() const { return getLocNo() == UndefLocNo; }

  DbgVariableValue changeLocNo(unsigned NewLocNo) const {
    return DbgVariableValue(NewLocNo, WasIndirect, *Expression);
  }

  friend inline bool operator==(const DbgVariableValue &LHS,
                                const DbgVariableValue &RHS) {
    return LHS.LocNo == RHS.LocNo && LHS.WasIndirect == RHS.WasIndirect &&
           LHS.Expression == RHS.Expression;
  }

  friend inline bool operator!=(const DbgVariableValue &LHS,
                                const DbgVariableValue &RHS) {
    return !(LHS == RHS);
  }

private:
  unsigned LocNo : 31;
  unsigned WasIndirect : 1;
  const DIExpression *Expression = nullptr;
};

/// Map of where a user value is live to that value.
using LocMap = IntervalMap<SlotIndex, DbgVariableValue, 4>;

/// Map of stack slot offsets for spilled locations.
/// Non-spilled locations are not added to the map.
using SpillOffsetMap = DenseMap<unsigned, unsigned>;

namespace {

class LDVImpl;

/// A user value is a part of a debug info user variable.
///
/// A DBG_VALUE instruction notes that (a sub-register of) a virtual register
/// holds part of a user variable. The part is identified by a byte offset.
///
/// UserValues are grouped into equivalence classes for easier searching. Two
/// user values are related if they are held by the same virtual register. The
/// equivalence class is the transitive closure of that relation.
class UserValue {
  const DILocalVariable *Variable; ///< The debug info variable we are part of.
  /// The part of the variable we describe.
  const Optional<DIExpression::FragmentInfo> Fragment;
  DebugLoc dl;            ///< The debug location for the variable. This is
                          ///< used by dwarf writer to find lexical scope.
  UserValue *leader;      ///< Equivalence class leader.
  UserValue *next = nullptr; ///< Next value in equivalence class, or null.

  /// Numbered locations referenced by locmap.
  SmallVector<MachineOperand, 4> locations;

  /// Map of slot indices where this value is live.
  LocMap locInts;

  /// Set of interval start indexes that have been trimmed to the
  /// lexical scope.
  SmallSet<SlotIndex, 2> trimmedDefs;

  /// Insert a DBG_VALUE into MBB at Idx for DbgValue.
  void insertDebugValue(MachineBasicBlock *MBB, SlotIndex StartIdx,
                        SlotIndex StopIdx, DbgVariableValue DbgValue,
                        bool Spilled, unsigned SpillOffset, LiveIntervals &LIS,
                        const TargetInstrInfo &TII,
                        const TargetRegisterInfo &TRI);

  /// Replace OldLocNo ranges with NewRegs ranges where NewRegs
  /// is live. Returns true if any changes were made.
  bool splitLocation(unsigned OldLocNo, ArrayRef<unsigned> NewRegs,
                     LiveIntervals &LIS);

public:
  /// Create a new UserValue.
  UserValue(const DILocalVariable *var,
            Optional<DIExpression::FragmentInfo> Fragment, DebugLoc L,
            LocMap::Allocator &alloc)
      : Variable(var), Fragment(Fragment), dl(std::move(L)), leader(this),
        locInts(alloc) {}

  /// Get the leader of this value's equivalence class.
  UserValue *getLeader() {
    UserValue *l = leader;
    while (l != l->leader)
      l = l->leader;
    return leader = l;
  }

  /// Return the next UserValue in the equivalence class.
  UserValue *getNext() const { return next; }

  /// Merge equivalence classes.
  static UserValue *merge(UserValue *L1, UserValue *L2) {
    L2 = L2->getLeader();
    if (!L1)
      return L2;
    L1 = L1->getLeader();
    if (L1 == L2)
      return L1;
    // Splice L2 before L1's members.
    UserValue *End = L2;
    while (End->next) {
      End->leader = L1;
      End = End->next;
    }
    End->leader = L1;
    End->next = L1->next;
    L1->next = L2;
    return L1;
  }

  /// Return the location number that matches Loc.
  ///
  /// For undef values we always return location number UndefLocNo without
  /// inserting anything in locations. Since locations is a vector and the
  /// location number is the position in the vector and UndefLocNo is ~0,
  /// we would need a very big vector to put the value at the right position.
  unsigned getLocationNo(const MachineOperand &LocMO) {
    if (LocMO.isReg()) {
      if (LocMO.getReg() == 0)
        return UndefLocNo;
      // For register locations we dont care about use/def and other flags.
      for (unsigned i = 0, e = locations.size(); i != e; ++i)
        if (locations[i].isReg() &&
            locations[i].getReg() == LocMO.getReg() &&
            locations[i].getSubReg() == LocMO.getSubReg())
          return i;
    } else
      for (unsigned i = 0, e = locations.size(); i != e; ++i)
        if (LocMO.isIdenticalTo(locations[i]))
          return i;
    locations.push_back(LocMO);
    // We are storing a MachineOperand outside a MachineInstr.
    locations.back().clearParent();
    // Don't store def operands.
    if (locations.back().isReg()) {
      if (locations.back().isDef())
        locations.back().setIsDead(false);
      locations.back().setIsUse();
    }
    return locations.size() - 1;
  }

  /// Remove (recycle) a location number. If \p LocNo still is used by the
  /// locInts nothing is done.
  void removeLocationIfUnused(unsigned LocNo) {
    // Bail out if LocNo still is used.
    for (LocMap::const_iterator I = locInts.begin(); I.valid(); ++I) {
      DbgVariableValue DbgValue = I.value();
      if (DbgValue.getLocNo() == LocNo)
        return;
    }
    // Remove the entry in the locations vector, and adjust all references to
    // location numbers above the removed entry.
    locations.erase(locations.begin() + LocNo);
    for (LocMap::iterator I = locInts.begin(); I.valid(); ++I) {
      DbgVariableValue DbgValue = I.value();
      if (!DbgValue.isUndef() && DbgValue.getLocNo() > LocNo)
        I.setValueUnchecked(DbgValue.changeLocNo(DbgValue.getLocNo() - 1));
    }
  }

  /// Ensure that all virtual register locations are mapped.
  void mapVirtRegs(LDVImpl *LDV);

  /// Add a definition point to this user value.
  void addDef(SlotIndex Idx, const MachineOperand &LocMO, bool IsIndirect,
              const DIExpression &Expr) {
    DbgVariableValue DbgValue(getLocationNo(LocMO), IsIndirect, Expr);
    // Add a singular (Idx,Idx) -> value mapping.
    LocMap::iterator I = locInts.find(Idx);
    if (!I.valid() || I.start() != Idx)
      I.insert(Idx, Idx.getNextSlot(), DbgValue);
    else
      // A later DBG_VALUE at the same SlotIndex overrides the old location.
      I.setValue(DbgValue);
  }

  /// Extend the current definition as far as possible down.
  ///
  /// Stop when meeting an existing def or when leaving the live
  /// range of VNI. End points where VNI is no longer live are added to Kills.
  ///
  /// We only propagate DBG_VALUES locally here. LiveDebugValues performs a
  /// data-flow analysis to propagate them beyond basic block boundaries.
  ///
  /// \param Idx Starting point for the definition.
  /// \param DbgValue value to propagate.
  /// \param LR Restrict liveness to where LR has the value VNI. May be null.
  /// \param VNI When LR is not null, this is the value to restrict to.
  /// \param [out] Kills Append end points of VNI's live range to Kills.
  /// \param LIS Live intervals analysis.
  void extendDef(SlotIndex Idx, DbgVariableValue DbgValue, LiveRange *LR,
                 const VNInfo *VNI, SmallVectorImpl<SlotIndex> *Kills,
                 LiveIntervals &LIS);

  /// The value in LI may be copies to other registers. Determine if
  /// any of the copies are available at the kill points, and add defs if
  /// possible.
  ///
  /// \param LI Scan for copies of the value in LI->reg.
  /// \param DbgValue Location number of LI->reg, and DIExpression.
  /// \param Kills Points where the range of DbgValue could be extended.
  /// \param [in,out] NewDefs Append (Idx, DbgValue) of inserted defs here.
  void addDefsFromCopies(
      LiveInterval *LI, DbgVariableValue DbgValue,
      const SmallVectorImpl<SlotIndex> &Kills,
      SmallVectorImpl<std::pair<SlotIndex, DbgVariableValue>> &NewDefs,
      MachineRegisterInfo &MRI, LiveIntervals &LIS);

  /// Compute the live intervals of all locations after collecting all their
  /// def points.
  void computeIntervals(MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
                        LiveIntervals &LIS, LexicalScopes &LS);

  /// Replace OldReg ranges with NewRegs ranges where NewRegs is
  /// live. Returns true if any changes were made.
  bool splitRegister(unsigned OldReg, ArrayRef<unsigned> NewRegs,
                     LiveIntervals &LIS);

  /// Rewrite virtual register locations according to the provided virtual
  /// register map. Record the stack slot offsets for the locations that
  /// were spilled.
  void rewriteLocations(VirtRegMap &VRM, const MachineFunction &MF,
                        const TargetInstrInfo &TII,
                        const TargetRegisterInfo &TRI,
                        SpillOffsetMap &SpillOffsets);

  /// Recreate DBG_VALUE instruction from data structures.
  void emitDebugValues(VirtRegMap *VRM, LiveIntervals &LIS,
                       const TargetInstrInfo &TII,
                       const TargetRegisterInfo &TRI,
                       const SpillOffsetMap &SpillOffsets);

  /// Return DebugLoc of this UserValue.
  DebugLoc getDebugLoc() { return dl;}

  void print(raw_ostream &, const TargetRegisterInfo *);
};

/// A user label is a part of a debug info user label.
class UserLabel {
  const DILabel *Label; ///< The debug info label we are part of.
  DebugLoc dl;          ///< The debug location for the label. This is
                        ///< used by dwarf writer to find lexical scope.
  SlotIndex loc;        ///< Slot used by the debug label.

  /// Insert a DBG_LABEL into MBB at Idx.
  void insertDebugLabel(MachineBasicBlock *MBB, SlotIndex Idx,
                        LiveIntervals &LIS, const TargetInstrInfo &TII);

public:
  /// Create a new UserLabel.
  UserLabel(const DILabel *label, DebugLoc L, SlotIndex Idx)
      : Label(label), dl(std::move(L)), loc(Idx) {}

  /// Does this UserLabel match the parameters?
  bool matches(const DILabel *L, const DILocation *IA,
             const SlotIndex Index) const {
    return Label == L && dl->getInlinedAt() == IA && loc == Index;
  }

  /// Recreate DBG_LABEL instruction from data structures.
  void emitDebugLabel(LiveIntervals &LIS, const TargetInstrInfo &TII);

  /// Return DebugLoc of this UserLabel.
  DebugLoc getDebugLoc() { return dl; }

  void print(raw_ostream &, const TargetRegisterInfo *);
};

/// Implementation of the LiveDebugVariables pass.
class LDVImpl {
  LiveDebugVariables &pass;
  LocMap::Allocator allocator;
  MachineFunction *MF = nullptr;
  LiveIntervals *LIS;
  const TargetRegisterInfo *TRI;

  /// Whether emitDebugValues is called.
  bool EmitDone = false;

  /// Whether the machine function is modified during the pass.
  bool ModifiedMF = false;

  /// All allocated UserValue instances.
  SmallVector<std::unique_ptr<UserValue>, 8> userValues;

  /// All allocated UserLabel instances.
  SmallVector<std::unique_ptr<UserLabel>, 2> userLabels;

  /// Map virtual register to eq class leader.
  using VRMap = DenseMap<unsigned, UserValue *>;
  VRMap virtRegToEqClass;

  /// Map to find existing UserValue instances.
  using UVMap = DenseMap<DebugVariable, UserValue *>;
  UVMap userVarMap;

  /// Find or create a UserValue.
  UserValue *getUserValue(const DILocalVariable *Var,
                          Optional<DIExpression::FragmentInfo> Fragment,
                          const DebugLoc &DL);

  /// Find the EC leader for VirtReg or null.
  UserValue *lookupVirtReg(unsigned VirtReg);

  /// Add DBG_VALUE instruction to our maps.
  ///
  /// \param MI DBG_VALUE instruction
  /// \param Idx Last valid SLotIndex before instruction.
  ///
  /// \returns True if the DBG_VALUE instruction should be deleted.
  bool handleDebugValue(MachineInstr &MI, SlotIndex Idx);

  /// Add DBG_LABEL instruction to UserLabel.
  ///
  /// \param MI DBG_LABEL instruction
  /// \param Idx Last valid SlotIndex before instruction.
  ///
  /// \returns True if the DBG_LABEL instruction should be deleted.
  bool handleDebugLabel(MachineInstr &MI, SlotIndex Idx);

  /// Collect and erase all DBG_VALUE instructions, adding a UserValue def
  /// for each instruction.
  ///
  /// \param mf MachineFunction to be scanned.
  ///
  /// \returns True if any debug values were found.
  bool collectDebugValues(MachineFunction &mf);

  /// Compute the live intervals of all user values after collecting all
  /// their def points.
  void computeIntervals();

public:
  LDVImpl(LiveDebugVariables *ps) : pass(*ps) {}

  bool runOnMachineFunction(MachineFunction &mf);

  /// Release all memory.
  void clear() {
    MF = nullptr;
    userValues.clear();
    userLabels.clear();
    virtRegToEqClass.clear();
    userVarMap.clear();
    // Make sure we call emitDebugValues if the machine function was modified.
    assert((!ModifiedMF || EmitDone) &&
           "Dbg values are not emitted in LDV");
    EmitDone = false;
    ModifiedMF = false;
  }

  /// Map virtual register to an equivalence class.
  void mapVirtReg(unsigned VirtReg, UserValue *EC);

  /// Replace all references to OldReg with NewRegs.
  void splitRegister(unsigned OldReg, ArrayRef<unsigned> NewRegs);

  /// Recreate DBG_VALUE instruction from data structures.
  void emitDebugValues(VirtRegMap *VRM);

  void print(raw_ostream&);
};

} // end anonymous namespace

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static void printDebugLoc(const DebugLoc &DL, raw_ostream &CommentOS,
                          const LLVMContext &Ctx) {
  if (!DL)
    return;

  auto *Scope = cast<DIScope>(DL.getScope());
  // Omit the directory, because it's likely to be long and uninteresting.
  CommentOS << Scope->getFilename();
  CommentOS << ':' << DL.getLine();
  if (DL.getCol() != 0)
    CommentOS << ':' << DL.getCol();

  DebugLoc InlinedAtDL = DL.getInlinedAt();
  if (!InlinedAtDL)
    return;

  CommentOS << " @[ ";
  printDebugLoc(InlinedAtDL, CommentOS, Ctx);
  CommentOS << " ]";
}

static void printExtendedName(raw_ostream &OS, const DINode *Node,
                              const DILocation *DL) {
  const LLVMContext &Ctx = Node->getContext();
  StringRef Res;
  unsigned Line = 0;
  if (const auto *V = dyn_cast<const DILocalVariable>(Node)) {
    Res = V->getName();
    Line = V->getLine();
  } else if (const auto *L = dyn_cast<const DILabel>(Node)) {
    Res = L->getName();
    Line = L->getLine();
  }

  if (!Res.empty())
    OS << Res << "," << Line;
  auto *InlinedAt = DL ? DL->getInlinedAt() : nullptr;
  if (InlinedAt) {
    if (DebugLoc InlinedAtDL = InlinedAt) {
      OS << " @[";
      printDebugLoc(InlinedAtDL, OS, Ctx);
      OS << "]";
    }
  }
}

void UserValue::print(raw_ostream &OS, const TargetRegisterInfo *TRI) {
  OS << "!\"";
  printExtendedName(OS, Variable, dl);

  OS << "\"\t";
  for (LocMap::const_iterator I = locInts.begin(); I.valid(); ++I) {
    OS << " [" << I.start() << ';' << I.stop() << "):";
    if (I.value().isUndef())
      OS << "undef";
    else {
      OS << I.value().getLocNo();
      if (I.value().getWasIndirect())
        OS << " ind";
    }
  }
  for (unsigned i = 0, e = locations.size(); i != e; ++i) {
    OS << " Loc" << i << '=';
    locations[i].print(OS, TRI);
  }
  OS << '\n';
}

void UserLabel::print(raw_ostream &OS, const TargetRegisterInfo *TRI) {
  OS << "!\"";
  printExtendedName(OS, Label, dl);

  OS << "\"\t";
  OS << loc;
  OS << '\n';
}

void LDVImpl::print(raw_ostream &OS) {
  OS << "********** DEBUG VARIABLES **********\n";
  for (auto &userValue : userValues)
    userValue->print(OS, TRI);
  OS << "********** DEBUG LABELS **********\n";
  for (auto &userLabel : userLabels)
    userLabel->print(OS, TRI);
}
#endif

void UserValue::mapVirtRegs(LDVImpl *LDV) {
  for (unsigned i = 0, e = locations.size(); i != e; ++i)
    if (locations[i].isReg() &&
        Register::isVirtualRegister(locations[i].getReg()))
      LDV->mapVirtReg(locations[i].getReg(), this);
}

UserValue *LDVImpl::getUserValue(const DILocalVariable *Var,
                                 Optional<DIExpression::FragmentInfo> Fragment,
                                 const DebugLoc &DL) {
  // FIXME: Handle partially overlapping fragments. See
  // https://reviews.llvm.org/D70121#1849741.
  DebugVariable ID(Var, Fragment, DL->getInlinedAt());
  UserValue *&UV = userVarMap[ID];
  if (!UV) {
    userValues.push_back(
        std::make_unique<UserValue>(Var, Fragment, DL, allocator));
    UV = userValues.back().get();
  }
  return UV;
}

void LDVImpl::mapVirtReg(unsigned VirtReg, UserValue *EC) {
  assert(Register::isVirtualRegister(VirtReg) && "Only map VirtRegs");
  UserValue *&Leader = virtRegToEqClass[VirtReg];
  Leader = UserValue::merge(Leader, EC);
}

UserValue *LDVImpl::lookupVirtReg(unsigned VirtReg) {
  if (UserValue *UV = virtRegToEqClass.lookup(VirtReg))
    return UV->getLeader();
  return nullptr;
}

bool LDVImpl::handleDebugValue(MachineInstr &MI, SlotIndex Idx) {
  // DBG_VALUE loc, offset, variable
  if (MI.getNumOperands() != 4 ||
      !(MI.getOperand(1).isReg() || MI.getOperand(1).isImm()) ||
      !MI.getOperand(2).isMetadata()) {
    LLVM_DEBUG(dbgs() << "Can't handle " << MI);
    return false;
  }

  // Detect invalid DBG_VALUE instructions, with a debug-use of a virtual
  // register that hasn't been defined yet. If we do not remove those here, then
  // the re-insertion of the DBG_VALUE instruction after register allocation
  // will be incorrect.
  // TODO: If earlier passes are corrected to generate sane debug information
  // (and if the machine verifier is improved to catch this), then these checks
  // could be removed or replaced by asserts.
  bool Discard = false;
  if (MI.getOperand(0).isReg() &&
      Register::isVirtualRegister(MI.getOperand(0).getReg())) {
    const Register Reg = MI.getOperand(0).getReg();
    if (!LIS->hasInterval(Reg)) {
      // The DBG_VALUE is described by a virtual register that does not have a
      // live interval. Discard the DBG_VALUE.
      Discard = true;
      LLVM_DEBUG(dbgs() << "Discarding debug info (no LIS interval): " << Idx
                        << " " << MI);
    } else {
      // The DBG_VALUE is only valid if either Reg is live out from Idx, or Reg
      // is defined dead at Idx (where Idx is the slot index for the instruction
      // preceding the DBG_VALUE).
      const LiveInterval &LI = LIS->getInterval(Reg);
      LiveQueryResult LRQ = LI.Query(Idx);
      if (!LRQ.valueOutOrDead()) {
        // We have found a DBG_VALUE with the value in a virtual register that
        // is not live. Discard the DBG_VALUE.
        Discard = true;
        LLVM_DEBUG(dbgs() << "Discarding debug info (reg not live): " << Idx
                          << " " << MI);
      }
    }
  }

  // Get or create the UserValue for (variable,offset) here.
  bool IsIndirect = MI.getOperand(1).isImm();
  if (IsIndirect)
    assert(MI.getOperand(1).getImm() == 0 && "DBG_VALUE with nonzero offset");
  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  UserValue *UV = getUserValue(Var, Expr->getFragmentInfo(), MI.getDebugLoc());
  if (!Discard)
    UV->addDef(Idx, MI.getOperand(0), IsIndirect, *Expr);
  else {
    MachineOperand MO = MachineOperand::CreateReg(0U, false);
    MO.setIsDebug();
    UV->addDef(Idx, MO, false, *Expr);
  }
  return true;
}

bool LDVImpl::handleDebugLabel(MachineInstr &MI, SlotIndex Idx) {
  // DBG_LABEL label
  if (MI.getNumOperands() != 1 || !MI.getOperand(0).isMetadata()) {
    LLVM_DEBUG(dbgs() << "Can't handle " << MI);
    return false;
  }

  // Get or create the UserLabel for label here.
  const DILabel *Label = MI.getDebugLabel();
  const DebugLoc &DL = MI.getDebugLoc();
  bool Found = false;
  for (auto const &L : userLabels) {
    if (L->matches(Label, DL->getInlinedAt(), Idx)) {
      Found = true;
      break;
    }
  }
  if (!Found)
    userLabels.push_back(std::make_unique<UserLabel>(Label, DL, Idx));

  return true;
}

bool LDVImpl::collectDebugValues(MachineFunction &mf) {
  bool Changed = false;
  for (MachineFunction::iterator MFI = mf.begin(), MFE = mf.end(); MFI != MFE;
       ++MFI) {
    MachineBasicBlock *MBB = &*MFI;
    for (MachineBasicBlock::iterator MBBI = MBB->begin(), MBBE = MBB->end();
         MBBI != MBBE;) {
      // Use the first debug instruction in the sequence to get a SlotIndex
      // for following consecutive debug instructions.
      if (!MBBI->isDebugInstr()) {
        ++MBBI;
        continue;
      }
      // Debug instructions has no slot index. Use the previous
      // non-debug instruction's SlotIndex as its SlotIndex.
      SlotIndex Idx =
          MBBI == MBB->begin()
              ? LIS->getMBBStartIdx(MBB)
              : LIS->getInstructionIndex(*std::prev(MBBI)).getRegSlot();
      // Handle consecutive debug instructions with the same slot index.
      do {
        // Only handle DBG_VALUE in handleDebugValue(). Skip all other
        // kinds of debug instructions.
        if ((MBBI->isDebugValue() && handleDebugValue(*MBBI, Idx)) ||
            (MBBI->isDebugLabel() && handleDebugLabel(*MBBI, Idx))) {
          MBBI = MBB->erase(MBBI);
          Changed = true;
        } else
          ++MBBI;
      } while (MBBI != MBBE && MBBI->isDebugInstr());
    }
  }
  return Changed;
}

void UserValue::extendDef(SlotIndex Idx, DbgVariableValue DbgValue, LiveRange *LR,
                          const VNInfo *VNI, SmallVectorImpl<SlotIndex> *Kills,
                          LiveIntervals &LIS) {
  SlotIndex Start = Idx;
  MachineBasicBlock *MBB = LIS.getMBBFromIndex(Start);
  SlotIndex Stop = LIS.getMBBEndIdx(MBB);
  LocMap::iterator I = locInts.find(Start);

  // Limit to VNI's live range.
  bool ToEnd = true;
  if (LR && VNI) {
    LiveInterval::Segment *Segment = LR->getSegmentContaining(Start);
    if (!Segment || Segment->valno != VNI) {
      if (Kills)
        Kills->push_back(Start);
      return;
    }
    if (Segment->end < Stop) {
      Stop = Segment->end;
      ToEnd = false;
    }
  }

  // There could already be a short def at Start.
  if (I.valid() && I.start() <= Start) {
    // Stop when meeting a different location or an already extended interval.
    Start = Start.getNextSlot();
    if (I.value() != DbgValue || I.stop() != Start)
      return;
    // This is a one-slot placeholder. Just skip it.
    ++I;
  }

  // Limited by the next def.
  if (I.valid() && I.start() < Stop)
    Stop = I.start();
  // Limited by VNI's live range.
  else if (!ToEnd && Kills)
    Kills->push_back(Stop);

  if (Start < Stop)
    I.insert(Start, Stop, DbgValue);
}

void UserValue::addDefsFromCopies(
    LiveInterval *LI, DbgVariableValue DbgValue,
    const SmallVectorImpl<SlotIndex> &Kills,
    SmallVectorImpl<std::pair<SlotIndex, DbgVariableValue>> &NewDefs,
    MachineRegisterInfo &MRI, LiveIntervals &LIS) {
  if (Kills.empty())
    return;
  // Don't track copies from physregs, there are too many uses.
  if (!Register::isVirtualRegister(LI->reg))
    return;

  // Collect all the (vreg, valno) pairs that are copies of LI.
  SmallVector<std::pair<LiveInterval*, const VNInfo*>, 8> CopyValues;
  for (MachineOperand &MO : MRI.use_nodbg_operands(LI->reg)) {
    MachineInstr *MI = MO.getParent();
    // Copies of the full value.
    if (MO.getSubReg() || !MI->isCopy())
      continue;
    Register DstReg = MI->getOperand(0).getReg();

    // Don't follow copies to physregs. These are usually setting up call
    // arguments, and the argument registers are always call clobbered. We are
    // better off in the source register which could be a callee-saved register,
    // or it could be spilled.
    if (!Register::isVirtualRegister(DstReg))
      continue;

    // Is the value extended to reach this copy? If not, another def may be
    // blocking it, or we are looking at a wrong value of LI.
    SlotIndex Idx = LIS.getInstructionIndex(*MI);
    LocMap::iterator I = locInts.find(Idx.getRegSlot(true));
    if (!I.valid() || I.value() != DbgValue)
      continue;

    if (!LIS.hasInterval(DstReg))
      continue;
    LiveInterval *DstLI = &LIS.getInterval(DstReg);
    const VNInfo *DstVNI = DstLI->getVNInfoAt(Idx.getRegSlot());
    assert(DstVNI && DstVNI->def == Idx.getRegSlot() && "Bad copy value");
    CopyValues.push_back(std::make_pair(DstLI, DstVNI));
  }

  if (CopyValues.empty())
    return;

  LLVM_DEBUG(dbgs() << "Got " << CopyValues.size() << " copies of " << *LI
                    << '\n');

  // Try to add defs of the copied values for each kill point.
  for (unsigned i = 0, e = Kills.size(); i != e; ++i) {
    SlotIndex Idx = Kills[i];
    for (unsigned j = 0, e = CopyValues.size(); j != e; ++j) {
      LiveInterval *DstLI = CopyValues[j].first;
      const VNInfo *DstVNI = CopyValues[j].second;
      if (DstLI->getVNInfoAt(Idx) != DstVNI)
        continue;
      // Check that there isn't already a def at Idx
      LocMap::iterator I = locInts.find(Idx);
      if (I.valid() && I.start() <= Idx)
        continue;
      LLVM_DEBUG(dbgs() << "Kill at " << Idx << " covered by valno #"
                        << DstVNI->id << " in " << *DstLI << '\n');
      MachineInstr *CopyMI = LIS.getInstructionFromIndex(DstVNI->def);
      assert(CopyMI && CopyMI->isCopy() && "Bad copy value");
      unsigned LocNo = getLocationNo(CopyMI->getOperand(0));
      DbgVariableValue NewValue = DbgValue.changeLocNo(LocNo);
      I.insert(Idx, Idx.getNextSlot(), NewValue);
      NewDefs.push_back(std::make_pair(Idx, NewValue));
      break;
    }
  }
}

void UserValue::computeIntervals(MachineRegisterInfo &MRI,
                                 const TargetRegisterInfo &TRI,
                                 LiveIntervals &LIS, LexicalScopes &LS) {
  SmallVector<std::pair<SlotIndex, DbgVariableValue>, 16> Defs;

  // Collect all defs to be extended (Skipping undefs).
  for (LocMap::const_iterator I = locInts.begin(); I.valid(); ++I)
    if (!I.value().isUndef())
      Defs.push_back(std::make_pair(I.start(), I.value()));

  // Extend all defs, and possibly add new ones along the way.
  for (unsigned i = 0; i != Defs.size(); ++i) {
    SlotIndex Idx = Defs[i].first;
    DbgVariableValue DbgValue = Defs[i].second;
    const MachineOperand &LocMO = locations[DbgValue.getLocNo()];

    if (!LocMO.isReg()) {
      extendDef(Idx, DbgValue, nullptr, nullptr, nullptr, LIS);
      continue;
    }

    // Register locations are constrained to where the register value is live.
    if (Register::isVirtualRegister(LocMO.getReg())) {
      LiveInterval *LI = nullptr;
      const VNInfo *VNI = nullptr;
      if (LIS.hasInterval(LocMO.getReg())) {
        LI = &LIS.getInterval(LocMO.getReg());
        VNI = LI->getVNInfoAt(Idx);
      }
      SmallVector<SlotIndex, 16> Kills;
      extendDef(Idx, DbgValue, LI, VNI, &Kills, LIS);
      // FIXME: Handle sub-registers in addDefsFromCopies. The problem is that
      // if the original location for example is %vreg0:sub_hi, and we find a
      // full register copy in addDefsFromCopies (at the moment it only handles
      // full register copies), then we must add the sub1 sub-register index to
      // the new location. However, that is only possible if the new virtual
      // register is of the same regclass (or if there is an equivalent
      // sub-register in that regclass). For now, simply skip handling copies if
      // a sub-register is involved.
      if (LI && !LocMO.getSubReg())
        addDefsFromCopies(LI, DbgValue, Kills, Defs, MRI, LIS);
      continue;
    }

    // For physregs, we only mark the start slot idx. DwarfDebug will see it
    // as if the DBG_VALUE is valid up until the end of the basic block, or
    // the next def of the physical register. So we do not need to extend the
    // range. It might actually happen that the DBG_VALUE is the last use of
    // the physical register (e.g. if this is an unused input argument to a
    // function).
  }

  // The computed intervals may extend beyond the range of the debug
  // location's lexical scope. In this case, splitting of an interval
  // can result in an interval outside of the scope being created,
  // causing extra unnecessary DBG_VALUEs to be emitted. To prevent
  // this, trim the intervals to the lexical scope.

  LexicalScope *Scope = LS.findLexicalScope(dl);
  if (!Scope)
    return;

  SlotIndex PrevEnd;
  LocMap::iterator I = locInts.begin();

  // Iterate over the lexical scope ranges. Each time round the loop
  // we check the intervals for overlap with the end of the previous
  // range and the start of the next. The first range is handled as
  // a special case where there is no PrevEnd.
  for (const InsnRange &Range : Scope->getRanges()) {
    SlotIndex RStart = LIS.getInstructionIndex(*Range.first);
    SlotIndex REnd = LIS.getInstructionIndex(*Range.second);

    // Variable locations at the first instruction of a block should be
    // based on the block's SlotIndex, not the first instruction's index.
    if (Range.first == Range.first->getParent()->begin())
      RStart = LIS.getSlotIndexes()->getIndexBefore(*Range.first);

    // At the start of each iteration I has been advanced so that
    // I.stop() >= PrevEnd. Check for overlap.
    if (PrevEnd && I.start() < PrevEnd) {
      SlotIndex IStop = I.stop();
      DbgVariableValue DbgValue = I.value();

      // Stop overlaps previous end - trim the end of the interval to
      // the scope range.
      I.setStopUnchecked(PrevEnd);
      ++I;

      // If the interval also overlaps the start of the "next" (i.e.
      // current) range create a new interval for the remainder (which
      // may be further trimmed).
      if (RStart < IStop)
        I.insert(RStart, IStop, DbgValue);
    }

    // Advance I so that I.stop() >= RStart, and check for overlap.
    I.advanceTo(RStart);
    if (!I.valid())
      return;

    if (I.start() < RStart) {
      // Interval start overlaps range - trim to the scope range.
      I.setStartUnchecked(RStart);
      // Remember that this interval was trimmed.
      trimmedDefs.insert(RStart);
    }

    // The end of a lexical scope range is the last instruction in the
    // range. To convert to an interval we need the index of the
    // instruction after it.
    REnd = REnd.getNextIndex();

    // Advance I to first interval outside current range.
    I.advanceTo(REnd);
    if (!I.valid())
      return;

    PrevEnd = REnd;
  }

  // Check for overlap with end of final range.
  if (PrevEnd && I.start() < PrevEnd)
    I.setStopUnchecked(PrevEnd);
}

void LDVImpl::computeIntervals() {
  LexicalScopes LS;
  LS.initialize(*MF);

  for (unsigned i = 0, e = userValues.size(); i != e; ++i) {
    userValues[i]->computeIntervals(MF->getRegInfo(), *TRI, *LIS, LS);
    userValues[i]->mapVirtRegs(this);
  }
}

bool LDVImpl::runOnMachineFunction(MachineFunction &mf) {
  clear();
  MF = &mf;
  LIS = &pass.getAnalysis<LiveIntervals>();
  TRI = mf.getSubtarget().getRegisterInfo();
  LLVM_DEBUG(dbgs() << "********** COMPUTING LIVE DEBUG VARIABLES: "
                    << mf.getName() << " **********\n");

  bool Changed = collectDebugValues(mf);
  computeIntervals();
  LLVM_DEBUG(print(dbgs()));
  ModifiedMF = Changed;
  return Changed;
}

static void removeDebugValues(MachineFunction &mf) {
  for (MachineBasicBlock &MBB : mf) {
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
      if (!MBBI->isDebugValue()) {
        ++MBBI;
        continue;
      }
      MBBI = MBB.erase(MBBI);
    }
  }
}

bool LiveDebugVariables::runOnMachineFunction(MachineFunction &mf) {
  if (!EnableLDV)
    return false;
  if (!mf.getFunction().getSubprogram()) {
    removeDebugValues(mf);
    return false;
  }
  if (!pImpl)
    pImpl = new LDVImpl(this);
  return static_cast<LDVImpl*>(pImpl)->runOnMachineFunction(mf);
}

void LiveDebugVariables::releaseMemory() {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->clear();
}

LiveDebugVariables::~LiveDebugVariables() {
  if (pImpl)
    delete static_cast<LDVImpl*>(pImpl);
}

//===----------------------------------------------------------------------===//
//                           Live Range Splitting
//===----------------------------------------------------------------------===//

bool
UserValue::splitLocation(unsigned OldLocNo, ArrayRef<unsigned> NewRegs,
                         LiveIntervals& LIS) {
  LLVM_DEBUG({
    dbgs() << "Splitting Loc" << OldLocNo << '\t';
    print(dbgs(), nullptr);
  });
  bool DidChange = false;
  LocMap::iterator LocMapI;
  LocMapI.setMap(locInts);
  for (unsigned i = 0; i != NewRegs.size(); ++i) {
    LiveInterval *LI = &LIS.getInterval(NewRegs[i]);
    if (LI->empty())
      continue;

    // Don't allocate the new LocNo until it is needed.
    unsigned NewLocNo = UndefLocNo;

    // Iterate over the overlaps between locInts and LI.
    LocMapI.find(LI->beginIndex());
    if (!LocMapI.valid())
      continue;
    LiveInterval::iterator LII = LI->advanceTo(LI->begin(), LocMapI.start());
    LiveInterval::iterator LIE = LI->end();
    while (LocMapI.valid() && LII != LIE) {
      // At this point, we know that LocMapI.stop() > LII->start.
      LII = LI->advanceTo(LII, LocMapI.start());
      if (LII == LIE)
        break;

      // Now LII->end > LocMapI.start(). Do we have an overlap?
      if (LocMapI.value().getLocNo() == OldLocNo &&
          LII->start < LocMapI.stop()) {
        // Overlapping correct location. Allocate NewLocNo now.
        if (NewLocNo == UndefLocNo) {
          MachineOperand MO = MachineOperand::CreateReg(LI->reg, false);
          MO.setSubReg(locations[OldLocNo].getSubReg());
          NewLocNo = getLocationNo(MO);
          DidChange = true;
        }

        SlotIndex LStart = LocMapI.start();
        SlotIndex LStop = LocMapI.stop();
        DbgVariableValue OldDbgValue = LocMapI.value();

        // Trim LocMapI down to the LII overlap.
        if (LStart < LII->start)
          LocMapI.setStartUnchecked(LII->start);
        if (LStop > LII->end)
          LocMapI.setStopUnchecked(LII->end);

        // Change the value in the overlap. This may trigger coalescing.
        LocMapI.setValue(OldDbgValue.changeLocNo(NewLocNo));

        // Re-insert any removed OldDbgValue ranges.
        if (LStart < LocMapI.start()) {
          LocMapI.insert(LStart, LocMapI.start(), OldDbgValue);
          ++LocMapI;
          assert(LocMapI.valid() && "Unexpected coalescing");
        }
        if (LStop > LocMapI.stop()) {
          ++LocMapI;
          LocMapI.insert(LII->end, LStop, OldDbgValue);
          --LocMapI;
        }
      }

      // Advance to the next overlap.
      if (LII->end < LocMapI.stop()) {
        if (++LII == LIE)
          break;
        LocMapI.advanceTo(LII->start);
      } else {
        ++LocMapI;
        if (!LocMapI.valid())
          break;
        LII = LI->advanceTo(LII, LocMapI.start());
      }
    }
  }

  // Finally, remove OldLocNo unless it is still used by some interval in the
  // locInts map. One case when OldLocNo still is in use is when the register
  // has been spilled. In such situations the spilled register is kept as a
  // location until rewriteLocations is called (VirtRegMap is mapping the old
  // register to the spill slot). So for a while we can have locations that map
  // to virtual registers that have been removed from both the MachineFunction
  // and from LiveIntervals.
  //
  // We may also just be using the location for a value with a different
  // expression.
  removeLocationIfUnused(OldLocNo);

  LLVM_DEBUG({
    dbgs() << "Split result: \t";
    print(dbgs(), nullptr);
  });
  return DidChange;
}

bool
UserValue::splitRegister(unsigned OldReg, ArrayRef<unsigned> NewRegs,
                         LiveIntervals &LIS) {
  bool DidChange = false;
  // Split locations referring to OldReg. Iterate backwards so splitLocation can
  // safely erase unused locations.
  for (unsigned i = locations.size(); i ; --i) {
    unsigned LocNo = i-1;
    const MachineOperand *Loc = &locations[LocNo];
    if (!Loc->isReg() || Loc->getReg() != OldReg)
      continue;
    DidChange |= splitLocation(LocNo, NewRegs, LIS);
  }
  return DidChange;
}

void LDVImpl::splitRegister(unsigned OldReg, ArrayRef<unsigned> NewRegs) {
  bool DidChange = false;
  for (UserValue *UV = lookupVirtReg(OldReg); UV; UV = UV->getNext())
    DidChange |= UV->splitRegister(OldReg, NewRegs, *LIS);

  if (!DidChange)
    return;

  // Map all of the new virtual registers.
  UserValue *UV = lookupVirtReg(OldReg);
  for (unsigned i = 0; i != NewRegs.size(); ++i)
    mapVirtReg(NewRegs[i], UV);
}

void LiveDebugVariables::
splitRegister(unsigned OldReg, ArrayRef<unsigned> NewRegs, LiveIntervals &LIS) {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->splitRegister(OldReg, NewRegs);
}

void UserValue::rewriteLocations(VirtRegMap &VRM, const MachineFunction &MF,
                                 const TargetInstrInfo &TII,
                                 const TargetRegisterInfo &TRI,
                                 SpillOffsetMap &SpillOffsets) {
  // Build a set of new locations with new numbers so we can coalesce our
  // IntervalMap if two vreg intervals collapse to the same physical location.
  // Use MapVector instead of SetVector because MapVector::insert returns the
  // position of the previously or newly inserted element. The boolean value
  // tracks if the location was produced by a spill.
  // FIXME: This will be problematic if we ever support direct and indirect
  // frame index locations, i.e. expressing both variables in memory and
  // 'int x, *px = &x'. The "spilled" bit must become part of the location.
  MapVector<MachineOperand, std::pair<bool, unsigned>> NewLocations;
  SmallVector<unsigned, 4> LocNoMap(locations.size());
  for (unsigned I = 0, E = locations.size(); I != E; ++I) {
    bool Spilled = false;
    unsigned SpillOffset = 0;
    MachineOperand Loc = locations[I];
    // Only virtual registers are rewritten.
    if (Loc.isReg() && Loc.getReg() &&
        Register::isVirtualRegister(Loc.getReg())) {
      Register VirtReg = Loc.getReg();
      if (VRM.isAssignedReg(VirtReg) &&
          Register::isPhysicalRegister(VRM.getPhys(VirtReg))) {
        // This can create a %noreg operand in rare cases when the sub-register
        // index is no longer available. That means the user value is in a
        // non-existent sub-register, and %noreg is exactly what we want.
        Loc.substPhysReg(VRM.getPhys(VirtReg), TRI);
      } else if (VRM.getStackSlot(VirtReg) != VirtRegMap::NO_STACK_SLOT) {
        // Retrieve the stack slot offset.
        unsigned SpillSize;
        const MachineRegisterInfo &MRI = MF.getRegInfo();
        const TargetRegisterClass *TRC = MRI.getRegClass(VirtReg);
        bool Success = TII.getStackSlotRange(TRC, Loc.getSubReg(), SpillSize,
                                             SpillOffset, MF);

        // FIXME: Invalidate the location if the offset couldn't be calculated.
        (void)Success;

        Loc = MachineOperand::CreateFI(VRM.getStackSlot(VirtReg));
        Spilled = true;
      } else {
        Loc.setReg(0);
        Loc.setSubReg(0);
      }
    }

    // Insert this location if it doesn't already exist and record a mapping
    // from the old number to the new number.
    auto InsertResult = NewLocations.insert({Loc, {Spilled, SpillOffset}});
    unsigned NewLocNo = std::distance(NewLocations.begin(), InsertResult.first);
    LocNoMap[I] = NewLocNo;
  }

  // Rewrite the locations and record the stack slot offsets for spills.
  locations.clear();
  SpillOffsets.clear();
  for (auto &Pair : NewLocations) {
    bool Spilled;
    unsigned SpillOffset;
    std::tie(Spilled, SpillOffset) = Pair.second;
    locations.push_back(Pair.first);
    if (Spilled) {
      unsigned NewLocNo = std::distance(&*NewLocations.begin(), &Pair);
      SpillOffsets[NewLocNo] = SpillOffset;
    }
  }

  // Update the interval map, but only coalesce left, since intervals to the
  // right use the old location numbers. This should merge two contiguous
  // DBG_VALUE intervals with different vregs that were allocated to the same
  // physical register.
  for (LocMap::iterator I = locInts.begin(); I.valid(); ++I) {
    DbgVariableValue DbgValue = I.value();
    // Undef values don't exist in locations (and thus not in LocNoMap either)
    // so skip over them. See getLocationNo().
    if (DbgValue.isUndef())
      continue;
    unsigned NewLocNo = LocNoMap[DbgValue.getLocNo()];
    I.setValueUnchecked(DbgValue.changeLocNo(NewLocNo));
    I.setStart(I.start());
  }
}

/// Find an iterator for inserting a DBG_VALUE instruction.
static MachineBasicBlock::iterator
findInsertLocation(MachineBasicBlock *MBB, SlotIndex Idx,
                   LiveIntervals &LIS) {
  SlotIndex Start = LIS.getMBBStartIdx(MBB);
  Idx = Idx.getBaseIndex();

  // Try to find an insert location by going backwards from Idx.
  MachineInstr *MI;
  while (!(MI = LIS.getInstructionFromIndex(Idx))) {
    // We've reached the beginning of MBB.
    if (Idx == Start) {
      MachineBasicBlock::iterator I = MBB->SkipPHIsLabelsAndDebug(MBB->begin());
      return I;
    }
    Idx = Idx.getPrevIndex();
  }

  // Don't insert anything after the first terminator, though.
  return MI->isTerminator() ? MBB->getFirstTerminator() :
                              std::next(MachineBasicBlock::iterator(MI));
}

/// Find an iterator for inserting the next DBG_VALUE instruction
/// (or end if no more insert locations found).
static MachineBasicBlock::iterator
findNextInsertLocation(MachineBasicBlock *MBB,
                       MachineBasicBlock::iterator I,
                       SlotIndex StopIdx, MachineOperand &LocMO,
                       LiveIntervals &LIS,
                       const TargetRegisterInfo &TRI) {
  if (!LocMO.isReg())
    return MBB->instr_end();
  Register Reg = LocMO.getReg();

  // Find the next instruction in the MBB that define the register Reg.
  while (I != MBB->end() && !I->isTerminator()) {
    if (!LIS.isNotInMIMap(*I) &&
        SlotIndex::isEarlierEqualInstr(StopIdx, LIS.getInstructionIndex(*I)))
      break;
    if (I->definesRegister(Reg, &TRI))
      // The insert location is directly after the instruction/bundle.
      return std::next(I);
    ++I;
  }
  return MBB->end();
}

void UserValue::insertDebugValue(MachineBasicBlock *MBB, SlotIndex StartIdx,
                                 SlotIndex StopIdx, DbgVariableValue DbgValue,
                                 bool Spilled, unsigned SpillOffset,
                                 LiveIntervals &LIS, const TargetInstrInfo &TII,
                                 const TargetRegisterInfo &TRI) {
  SlotIndex MBBEndIdx = LIS.getMBBEndIdx(&*MBB);
  // Only search within the current MBB.
  StopIdx = (MBBEndIdx < StopIdx) ? MBBEndIdx : StopIdx;
  MachineBasicBlock::iterator I = findInsertLocation(MBB, StartIdx, LIS);
  // Undef values don't exist in locations so create new "noreg" register MOs
  // for them. See getLocationNo().
  MachineOperand MO =
      !DbgValue.isUndef()
          ? locations[DbgValue.getLocNo()]
          : MachineOperand::CreateReg(
                /* Reg */ 0, /* isDef */ false, /* isImp */ false,
                /* isKill */ false, /* isDead */ false,
                /* isUndef */ false, /* isEarlyClobber */ false,
                /* SubReg */ 0, /* isDebug */ true);

  ++NumInsertedDebugValues;

  assert(cast<DILocalVariable>(Variable)
             ->isValidLocationForIntrinsic(getDebugLoc()) &&
         "Expected inlined-at fields to agree");

  // If the location was spilled, the new DBG_VALUE will be indirect. If the
  // original DBG_VALUE was indirect, we need to add DW_OP_deref to indicate
  // that the original virtual register was a pointer. Also, add the stack slot
  // offset for the spilled register to the expression.
  const DIExpression *Expr = DbgValue.getExpression();
  uint8_t DIExprFlags = DIExpression::ApplyOffset;
  bool IsIndirect = DbgValue.getWasIndirect();
  if (Spilled) {
    if (IsIndirect)
      DIExprFlags |= DIExpression::DerefAfter;
    Expr =
        DIExpression::prepend(Expr, DIExprFlags, SpillOffset);
    IsIndirect = true;
  }

  assert((!Spilled || MO.isFI()) && "a spilled location must be a frame index");

  do {
    BuildMI(*MBB, I, getDebugLoc(), TII.get(TargetOpcode::DBG_VALUE),
            IsIndirect, MO, Variable, Expr);

    // Continue and insert DBG_VALUES after every redefinition of register
    // associated with the debug value within the range
    I = findNextInsertLocation(MBB, I, StopIdx, MO, LIS, TRI);
  } while (I != MBB->end());
}

void UserLabel::insertDebugLabel(MachineBasicBlock *MBB, SlotIndex Idx,
                                 LiveIntervals &LIS,
                                 const TargetInstrInfo &TII) {
  MachineBasicBlock::iterator I = findInsertLocation(MBB, Idx, LIS);
  ++NumInsertedDebugLabels;
  BuildMI(*MBB, I, getDebugLoc(), TII.get(TargetOpcode::DBG_LABEL))
      .addMetadata(Label);
}

void UserValue::emitDebugValues(VirtRegMap *VRM, LiveIntervals &LIS,
                                const TargetInstrInfo &TII,
                                const TargetRegisterInfo &TRI,
                                const SpillOffsetMap &SpillOffsets) {
  MachineFunction::iterator MFEnd = VRM->getMachineFunction().end();

  for (LocMap::const_iterator I = locInts.begin(); I.valid();) {
    SlotIndex Start = I.start();
    SlotIndex Stop = I.stop();
    DbgVariableValue DbgValue = I.value();
    auto SpillIt = !DbgValue.isUndef() ? SpillOffsets.find(DbgValue.getLocNo())
                                       : SpillOffsets.end();
    bool Spilled = SpillIt != SpillOffsets.end();
    unsigned SpillOffset = Spilled ? SpillIt->second : 0;

    // If the interval start was trimmed to the lexical scope insert the
    // DBG_VALUE at the previous index (otherwise it appears after the
    // first instruction in the range).
    if (trimmedDefs.count(Start))
      Start = Start.getPrevIndex();

    LLVM_DEBUG(dbgs() << "\t[" << Start << ';' << Stop
                      << "):" << DbgValue.getLocNo());
    MachineFunction::iterator MBB = LIS.getMBBFromIndex(Start)->getIterator();
    SlotIndex MBBEnd = LIS.getMBBEndIdx(&*MBB);

    LLVM_DEBUG(dbgs() << ' ' << printMBBReference(*MBB) << '-' << MBBEnd);
    insertDebugValue(&*MBB, Start, Stop, DbgValue, Spilled, SpillOffset, LIS,
                     TII, TRI);
    // This interval may span multiple basic blocks.
    // Insert a DBG_VALUE into each one.
    while (Stop > MBBEnd) {
      // Move to the next block.
      Start = MBBEnd;
      if (++MBB == MFEnd)
        break;
      MBBEnd = LIS.getMBBEndIdx(&*MBB);
      LLVM_DEBUG(dbgs() << ' ' << printMBBReference(*MBB) << '-' << MBBEnd);
      insertDebugValue(&*MBB, Start, Stop, DbgValue, Spilled, SpillOffset, LIS,
                       TII, TRI);
    }
    LLVM_DEBUG(dbgs() << '\n');
    if (MBB == MFEnd)
      break;

    ++I;
  }
}

void UserLabel::emitDebugLabel(LiveIntervals &LIS, const TargetInstrInfo &TII) {
  LLVM_DEBUG(dbgs() << "\t" << loc);
  MachineFunction::iterator MBB = LIS.getMBBFromIndex(loc)->getIterator();

  LLVM_DEBUG(dbgs() << ' ' << printMBBReference(*MBB));
  insertDebugLabel(&*MBB, loc, LIS, TII);

  LLVM_DEBUG(dbgs() << '\n');
}

void LDVImpl::emitDebugValues(VirtRegMap *VRM) {
  LLVM_DEBUG(dbgs() << "********** EMITTING LIVE DEBUG VARIABLES **********\n");
  if (!MF)
    return;
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  SpillOffsetMap SpillOffsets;
  for (auto &userValue : userValues) {
    LLVM_DEBUG(userValue->print(dbgs(), TRI));
    userValue->rewriteLocations(*VRM, *MF, *TII, *TRI, SpillOffsets);
    userValue->emitDebugValues(VRM, *LIS, *TII, *TRI, SpillOffsets);
  }
  LLVM_DEBUG(dbgs() << "********** EMITTING LIVE DEBUG LABELS **********\n");
  for (auto &userLabel : userLabels) {
    LLVM_DEBUG(userLabel->print(dbgs(), TRI));
    userLabel->emitDebugLabel(*LIS, *TII);
  }
  EmitDone = true;
}

void LiveDebugVariables::emitDebugValues(VirtRegMap *VRM) {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->emitDebugValues(VRM);
}

bool LiveDebugVariables::doInitialization(Module &M) {
  return Pass::doInitialization(M);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void LiveDebugVariables::dump() const {
  if (pImpl)
    static_cast<LDVImpl*>(pImpl)->print(dbgs());
}
#endif
