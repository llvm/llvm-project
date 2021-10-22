//===- InstrRefBasedImpl.cpp - Tracking Debug Value MIs -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file InstrRefBasedImpl.cpp
///
/// This is a separate implementation of LiveDebugValues, see
/// LiveDebugValues.cpp and VarLocBasedImpl.cpp for more information.
///
/// This pass propagates variable locations between basic blocks, resolving
/// control flow conflicts between them. The problem is SSA construction, where
/// each debug instruction assigns the *value* that a variable has, and every
/// instruction where the variable is in scope uses that variable. The resulting
/// map of instruction-to-value is then translated into a register (or spill)
/// location for each variable over each instruction.
///
/// The primary difference from normal SSA construction is that we cannot
/// _create_ PHI values that contain variable values. CodeGen has already
/// completed, and we can't alter it just to make debug-info complete. Thus:
/// we can identify function positions where we would like a PHI value for a
/// variable, but must search the MachineFunction to see whether such a PHI is
/// available. If no such PHI exists, the variable location must be dropped.
///
/// To achieve this, we perform two kinds of analysis. First, we identify
/// every value defined by every instruction (ignoring those that only move
/// another value), then re-compute an SSA-form representation of the
/// MachineFunction, using value propagation to eliminate any un-necessary
/// PHI values. This gives us a map of every value computed in the function,
/// and its location within the register file / stack.
///
/// Secondly, for each variable we perform the same analysis, where each debug
/// instruction is considered a def, and every instruction where the variable
/// is in lexical scope as a use. Value propagation is used again to eliminate
/// any un-necessary PHIs. This gives us a map of each variable to the value
/// it should have in a block.
///
/// Once both are complete, we have two maps for each block:
///  * Variables to the values they should have,
///  * Values to the register / spill slot they are located in.
/// After which we can marry-up variable values with a location, and emit
/// DBG_VALUE instructions specifying those locations. Variable locations may
/// be dropped in this process due to the desired variable value not being
/// resident in any machine location, or because there is no PHI value in any
/// location that accurately represents the desired value.  The building of
/// location lists for each block is left to DbgEntityHistoryCalculator.
///
/// This pass is kept efficient because the size of the first SSA problem
/// is proportional to the working-set size of the function, which the compiler
/// tries to keep small. (It's also proportional to the number of blocks).
/// Additionally, we repeatedly perform the second SSA problem analysis with
/// only the variables and blocks in a single lexical scope, exploiting their
/// locality.
///
/// ### Terminology
///
/// A machine location is a register or spill slot, a value is something that's
/// defined by an instruction or PHI node, while a variable value is the value
/// assigned to a variable. A variable location is a machine location, that must
/// contain the appropriate variable value. A value that is a PHI node is
/// occasionally called an mphi.
///
/// The first SSA problem is the "machine value location" problem,
/// because we're determining which machine locations contain which values.
/// The "locations" are constant: what's unknown is what value they contain.
///
/// The second SSA problem (the one for variables) is the "variable value
/// problem", because it's determining what values a variable has, rather than
/// what location those values are placed in.
///
/// TODO:
///   Overlapping fragments
///   Entry values
///   Add back DEBUG statements for debugging this
///   Collect statistics
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
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
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/SSAUpdaterImpl.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits.h>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "InstrRefBasedImpl.h"
#include "LiveDebugValues.h"

using namespace llvm;
using namespace LiveDebugValues;

// SSAUpdaterImple sets DEBUG_TYPE, change it.
#undef DEBUG_TYPE
#define DEBUG_TYPE "livedebugvalues"

// Act more like the VarLoc implementation, by propagating some locations too
// far and ignoring some transfers.
static cl::opt<bool> EmulateOldLDV("emulate-old-livedebugvalues", cl::Hidden,
                                   cl::desc("Act like old LiveDebugValues did"),
                                   cl::init(false));

/// Thin wrapper around an integer -- designed to give more type safety to
/// spill location numbers.
class SpillLocationNo {
public:
  explicit SpillLocationNo(unsigned SpillNo) : SpillNo(SpillNo) {}
  unsigned SpillNo;
  unsigned id() const { return SpillNo; }
};

/// Tracker for converting machine value locations and variable values into
/// variable locations (the output of LiveDebugValues), recorded as DBG_VALUEs
/// specifying block live-in locations and transfers within blocks.
///
/// Operating on a per-block basis, this class takes a (pre-loaded) MLocTracker
/// and must be initialized with the set of variable values that are live-in to
/// the block. The caller then repeatedly calls process(). TransferTracker picks
/// out variable locations for the live-in variable values (if there _is_ a
/// location) and creates the corresponding DBG_VALUEs. Then, as the block is
/// stepped through, transfers of values between machine locations are
/// identified and if profitable, a DBG_VALUE created.
///
/// This is where debug use-before-defs would be resolved: a variable with an
/// unavailable value could materialize in the middle of a block, when the
/// value becomes available. Or, we could detect clobbers and re-specify the
/// variable in a backup location. (XXX these are unimplemented).
class TransferTracker {
public:
  const TargetInstrInfo *TII;
  const TargetLowering *TLI;
  /// This machine location tracker is assumed to always contain the up-to-date
  /// value mapping for all machine locations. TransferTracker only reads
  /// information from it. (XXX make it const?)
  MLocTracker *MTracker;
  MachineFunction &MF;
  bool ShouldEmitDebugEntryValues;

  /// Record of all changes in variable locations at a block position. Awkwardly
  /// we allow inserting either before or after the point: MBB != nullptr
  /// indicates it's before, otherwise after.
  struct Transfer {
    MachineBasicBlock::instr_iterator Pos; /// Position to insert DBG_VALUes
    MachineBasicBlock *MBB; /// non-null if we should insert after.
    SmallVector<MachineInstr *, 4> Insts; /// Vector of DBG_VALUEs to insert.
  };

  struct LocAndProperties {
    LocIdx Loc;
    DbgValueProperties Properties;
  };

  /// Collection of transfers (DBG_VALUEs) to be inserted.
  SmallVector<Transfer, 32> Transfers;

  /// Local cache of what-value-is-in-what-LocIdx. Used to identify differences
  /// between TransferTrackers view of variable locations and MLocTrackers. For
  /// example, MLocTracker observes all clobbers, but TransferTracker lazily
  /// does not.
  std::vector<ValueIDNum> VarLocs;

  /// Map from LocIdxes to which DebugVariables are based that location.
  /// Mantained while stepping through the block. Not accurate if
  /// VarLocs[Idx] != MTracker->LocIdxToIDNum[Idx].
  std::map<LocIdx, SmallSet<DebugVariable, 4>> ActiveMLocs;

  /// Map from DebugVariable to it's current location and qualifying meta
  /// information. To be used in conjunction with ActiveMLocs to construct
  /// enough information for the DBG_VALUEs for a particular LocIdx.
  DenseMap<DebugVariable, LocAndProperties> ActiveVLocs;

  /// Temporary cache of DBG_VALUEs to be entered into the Transfers collection.
  SmallVector<MachineInstr *, 4> PendingDbgValues;

  /// Record of a use-before-def: created when a value that's live-in to the
  /// current block isn't available in any machine location, but it will be
  /// defined in this block.
  struct UseBeforeDef {
    /// Value of this variable, def'd in block.
    ValueIDNum ID;
    /// Identity of this variable.
    DebugVariable Var;
    /// Additional variable properties.
    DbgValueProperties Properties;
  };

  /// Map from instruction index (within the block) to the set of UseBeforeDefs
  /// that become defined at that instruction.
  DenseMap<unsigned, SmallVector<UseBeforeDef, 1>> UseBeforeDefs;

  /// The set of variables that are in UseBeforeDefs and can become a location
  /// once the relevant value is defined. An element being erased from this
  /// collection prevents the use-before-def materializing.
  DenseSet<DebugVariable> UseBeforeDefVariables;

  const TargetRegisterInfo &TRI;
  const BitVector &CalleeSavedRegs;

  TransferTracker(const TargetInstrInfo *TII, MLocTracker *MTracker,
                  MachineFunction &MF, const TargetRegisterInfo &TRI,
                  const BitVector &CalleeSavedRegs, const TargetPassConfig &TPC)
      : TII(TII), MTracker(MTracker), MF(MF), TRI(TRI),
        CalleeSavedRegs(CalleeSavedRegs) {
    TLI = MF.getSubtarget().getTargetLowering();
    auto &TM = TPC.getTM<TargetMachine>();
    ShouldEmitDebugEntryValues = TM.Options.ShouldEmitDebugEntryValues();
  }

  /// Load object with live-in variable values. \p mlocs contains the live-in
  /// values in each machine location, while \p vlocs the live-in variable
  /// values. This method picks variable locations for the live-in variables,
  /// creates DBG_VALUEs and puts them in #Transfers, then prepares the other
  /// object fields to track variable locations as we step through the block.
  /// FIXME: could just examine mloctracker instead of passing in \p mlocs?
  void loadInlocs(MachineBasicBlock &MBB, ValueIDNum *MLocs,
                  SmallVectorImpl<std::pair<DebugVariable, DbgValue>> &VLocs,
                  unsigned NumLocs) {
    ActiveMLocs.clear();
    ActiveVLocs.clear();
    VarLocs.clear();
    VarLocs.reserve(NumLocs);
    UseBeforeDefs.clear();
    UseBeforeDefVariables.clear();

    auto isCalleeSaved = [&](LocIdx L) {
      unsigned Reg = MTracker->LocIdxToLocID[L];
      if (Reg >= MTracker->NumRegs)
        return false;
      for (MCRegAliasIterator RAI(Reg, &TRI, true); RAI.isValid(); ++RAI)
        if (CalleeSavedRegs.test(*RAI))
          return true;
      return false;
    };

    // Map of the preferred location for each value.
    std::map<ValueIDNum, LocIdx> ValueToLoc;

    // Produce a map of value numbers to the current machine locs they live
    // in. When emulating VarLocBasedImpl, there should only be one
    // location; when not, we get to pick.
    for (auto Location : MTracker->locations()) {
      LocIdx Idx = Location.Idx;
      ValueIDNum &VNum = MLocs[Idx.asU64()];
      VarLocs.push_back(VNum);
      auto it = ValueToLoc.find(VNum);
      // In order of preference, pick:
      //  * Callee saved registers,
      //  * Other registers,
      //  * Spill slots.
      if (it == ValueToLoc.end() || MTracker->isSpill(it->second) ||
          (!isCalleeSaved(it->second) && isCalleeSaved(Idx.asU64()))) {
        // Insert, or overwrite if insertion failed.
        auto PrefLocRes = ValueToLoc.insert(std::make_pair(VNum, Idx));
        if (!PrefLocRes.second)
          PrefLocRes.first->second = Idx;
      }
    }

    // Now map variables to their picked LocIdxes.
    for (auto Var : VLocs) {
      if (Var.second.Kind == DbgValue::Const) {
        PendingDbgValues.push_back(
            emitMOLoc(*Var.second.MO, Var.first, Var.second.Properties));
        continue;
      }

      // If the value has no location, we can't make a variable location.
      const ValueIDNum &Num = Var.second.ID;
      auto ValuesPreferredLoc = ValueToLoc.find(Num);
      if (ValuesPreferredLoc == ValueToLoc.end()) {
        // If it's a def that occurs in this block, register it as a
        // use-before-def to be resolved as we step through the block.
        if (Num.getBlock() == (unsigned)MBB.getNumber() && !Num.isPHI())
          addUseBeforeDef(Var.first, Var.second.Properties, Num);
        else
          recoverAsEntryValue(Var.first, Var.second.Properties, Num);
        continue;
      }

      LocIdx M = ValuesPreferredLoc->second;
      auto NewValue = LocAndProperties{M, Var.second.Properties};
      auto Result = ActiveVLocs.insert(std::make_pair(Var.first, NewValue));
      if (!Result.second)
        Result.first->second = NewValue;
      ActiveMLocs[M].insert(Var.first);
      PendingDbgValues.push_back(
          MTracker->emitLoc(M, Var.first, Var.second.Properties));
    }
    flushDbgValues(MBB.begin(), &MBB);
  }

  /// Record that \p Var has value \p ID, a value that becomes available
  /// later in the function.
  void addUseBeforeDef(const DebugVariable &Var,
                       const DbgValueProperties &Properties, ValueIDNum ID) {
    UseBeforeDef UBD = {ID, Var, Properties};
    UseBeforeDefs[ID.getInst()].push_back(UBD);
    UseBeforeDefVariables.insert(Var);
  }

  /// After the instruction at index \p Inst and position \p pos has been
  /// processed, check whether it defines a variable value in a use-before-def.
  /// If so, and the variable value hasn't changed since the start of the
  /// block, create a DBG_VALUE.
  void checkInstForNewValues(unsigned Inst, MachineBasicBlock::iterator pos) {
    auto MIt = UseBeforeDefs.find(Inst);
    if (MIt == UseBeforeDefs.end())
      return;

    for (auto &Use : MIt->second) {
      LocIdx L = Use.ID.getLoc();

      // If something goes very wrong, we might end up labelling a COPY
      // instruction or similar with an instruction number, where it doesn't
      // actually define a new value, instead it moves a value. In case this
      // happens, discard.
      if (MTracker->readMLoc(L) != Use.ID)
        continue;

      // If a different debug instruction defined the variable value / location
      // since the start of the block, don't materialize this use-before-def.
      if (!UseBeforeDefVariables.count(Use.Var))
        continue;

      PendingDbgValues.push_back(MTracker->emitLoc(L, Use.Var, Use.Properties));
    }
    flushDbgValues(pos, nullptr);
  }

  /// Helper to move created DBG_VALUEs into Transfers collection.
  void flushDbgValues(MachineBasicBlock::iterator Pos, MachineBasicBlock *MBB) {
    if (PendingDbgValues.size() == 0)
      return;

    // Pick out the instruction start position.
    MachineBasicBlock::instr_iterator BundleStart;
    if (MBB && Pos == MBB->begin())
      BundleStart = MBB->instr_begin();
    else
      BundleStart = getBundleStart(Pos->getIterator());

    Transfers.push_back({BundleStart, MBB, PendingDbgValues});
    PendingDbgValues.clear();
  }

  bool isEntryValueVariable(const DebugVariable &Var,
                            const DIExpression *Expr) const {
    if (!Var.getVariable()->isParameter())
      return false;

    if (Var.getInlinedAt())
      return false;

    if (Expr->getNumElements() > 0)
      return false;

    return true;
  }

  bool isEntryValueValue(const ValueIDNum &Val) const {
    // Must be in entry block (block number zero), and be a PHI / live-in value.
    if (Val.getBlock() || !Val.isPHI())
      return false;

    // Entry values must enter in a register.
    if (MTracker->isSpill(Val.getLoc()))
      return false;

    Register SP = TLI->getStackPointerRegisterToSaveRestore();
    Register FP = TRI.getFrameRegister(MF);
    Register Reg = MTracker->LocIdxToLocID[Val.getLoc()];
    return Reg != SP && Reg != FP;
  }

  bool recoverAsEntryValue(const DebugVariable &Var, DbgValueProperties &Prop,
                           const ValueIDNum &Num) {
    // Is this variable location a candidate to be an entry value. First,
    // should we be trying this at all?
    if (!ShouldEmitDebugEntryValues)
      return false;

    // Is the variable appropriate for entry values (i.e., is a parameter).
    if (!isEntryValueVariable(Var, Prop.DIExpr))
      return false;

    // Is the value assigned to this variable still the entry value?
    if (!isEntryValueValue(Num))
      return false;

    // Emit a variable location using an entry value expression.
    DIExpression *NewExpr =
        DIExpression::prepend(Prop.DIExpr, DIExpression::EntryValue);
    Register Reg = MTracker->LocIdxToLocID[Num.getLoc()];
    MachineOperand MO = MachineOperand::CreateReg(Reg, false);

    PendingDbgValues.push_back(emitMOLoc(MO, Var, {NewExpr, Prop.Indirect}));
    return true;
  }

  /// Change a variable value after encountering a DBG_VALUE inside a block.
  void redefVar(const MachineInstr &MI) {
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    DbgValueProperties Properties(MI);

    const MachineOperand &MO = MI.getOperand(0);

    // Ignore non-register locations, we don't transfer those.
    if (!MO.isReg() || MO.getReg() == 0) {
      auto It = ActiveVLocs.find(Var);
      if (It != ActiveVLocs.end()) {
        ActiveMLocs[It->second.Loc].erase(Var);
        ActiveVLocs.erase(It);
     }
      // Any use-before-defs no longer apply.
      UseBeforeDefVariables.erase(Var);
      return;
    }

    Register Reg = MO.getReg();
    LocIdx NewLoc = MTracker->getRegMLoc(Reg);
    redefVar(MI, Properties, NewLoc);
  }

  /// Handle a change in variable location within a block. Terminate the
  /// variables current location, and record the value it now refers to, so
  /// that we can detect location transfers later on.
  void redefVar(const MachineInstr &MI, const DbgValueProperties &Properties,
                Optional<LocIdx> OptNewLoc) {
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    // Any use-before-defs no longer apply.
    UseBeforeDefVariables.erase(Var);

    // Erase any previous location,
    auto It = ActiveVLocs.find(Var);
    if (It != ActiveVLocs.end())
      ActiveMLocs[It->second.Loc].erase(Var);

    // If there _is_ no new location, all we had to do was erase.
    if (!OptNewLoc)
      return;
    LocIdx NewLoc = *OptNewLoc;

    // Check whether our local copy of values-by-location in #VarLocs is out of
    // date. Wipe old tracking data for the location if it's been clobbered in
    // the meantime.
    if (MTracker->readMLoc(NewLoc) != VarLocs[NewLoc.asU64()]) {
      for (auto &P : ActiveMLocs[NewLoc]) {
        ActiveVLocs.erase(P);
      }
      ActiveMLocs[NewLoc.asU64()].clear();
      VarLocs[NewLoc.asU64()] = MTracker->readMLoc(NewLoc);
    }

    ActiveMLocs[NewLoc].insert(Var);
    if (It == ActiveVLocs.end()) {
      ActiveVLocs.insert(
          std::make_pair(Var, LocAndProperties{NewLoc, Properties}));
    } else {
      It->second.Loc = NewLoc;
      It->second.Properties = Properties;
    }
  }

  /// Account for a location \p mloc being clobbered. Examine the variable
  /// locations that will be terminated: and try to recover them by using
  /// another location. Optionally, given \p MakeUndef, emit a DBG_VALUE to
  /// explicitly terminate a location if it can't be recovered.
  void clobberMloc(LocIdx MLoc, MachineBasicBlock::iterator Pos,
                   bool MakeUndef = true) {
    auto ActiveMLocIt = ActiveMLocs.find(MLoc);
    if (ActiveMLocIt == ActiveMLocs.end())
      return;

    // What was the old variable value?
    ValueIDNum OldValue = VarLocs[MLoc.asU64()];
    VarLocs[MLoc.asU64()] = ValueIDNum::EmptyValue;

    // Examine the remaining variable locations: if we can find the same value
    // again, we can recover the location.
    Optional<LocIdx> NewLoc = None;
    for (auto Loc : MTracker->locations())
      if (Loc.Value == OldValue)
        NewLoc = Loc.Idx;

    // If there is no location, and we weren't asked to make the variable
    // explicitly undef, then stop here.
    if (!NewLoc && !MakeUndef) {
      // Try and recover a few more locations with entry values.
      for (auto &Var : ActiveMLocIt->second) {
        auto &Prop = ActiveVLocs.find(Var)->second.Properties;
        recoverAsEntryValue(Var, Prop, OldValue);
      }
      flushDbgValues(Pos, nullptr);
      return;
    }

    // Examine all the variables based on this location.
    DenseSet<DebugVariable> NewMLocs;
    for (auto &Var : ActiveMLocIt->second) {
      auto ActiveVLocIt = ActiveVLocs.find(Var);
      // Re-state the variable location: if there's no replacement then NewLoc
      // is None and a $noreg DBG_VALUE will be created. Otherwise, a DBG_VALUE
      // identifying the alternative location will be emitted.
      const DIExpression *Expr = ActiveVLocIt->second.Properties.DIExpr;
      DbgValueProperties Properties(Expr, false);
      PendingDbgValues.push_back(MTracker->emitLoc(NewLoc, Var, Properties));

      // Update machine locations <=> variable locations maps. Defer updating
      // ActiveMLocs to avoid invalidaing the ActiveMLocIt iterator.
      if (!NewLoc) {
        ActiveVLocs.erase(ActiveVLocIt);
      } else {
        ActiveVLocIt->second.Loc = *NewLoc;
        NewMLocs.insert(Var);
      }
    }

    // Commit any deferred ActiveMLoc changes.
    if (!NewMLocs.empty())
      for (auto &Var : NewMLocs)
        ActiveMLocs[*NewLoc].insert(Var);

    // We lazily track what locations have which values; if we've found a new
    // location for the clobbered value, remember it.
    if (NewLoc)
      VarLocs[NewLoc->asU64()] = OldValue;

    flushDbgValues(Pos, nullptr);

    ActiveMLocIt->second.clear();
  }

  /// Transfer variables based on \p Src to be based on \p Dst. This handles
  /// both register copies as well as spills and restores. Creates DBG_VALUEs
  /// describing the movement.
  void transferMlocs(LocIdx Src, LocIdx Dst, MachineBasicBlock::iterator Pos) {
    // Does Src still contain the value num we expect? If not, it's been
    // clobbered in the meantime, and our variable locations are stale.
    if (VarLocs[Src.asU64()] != MTracker->readMLoc(Src))
      return;

    // assert(ActiveMLocs[Dst].size() == 0);
    //^^^ Legitimate scenario on account of un-clobbered slot being assigned to?
    ActiveMLocs[Dst] = ActiveMLocs[Src];
    VarLocs[Dst.asU64()] = VarLocs[Src.asU64()];

    // For each variable based on Src; create a location at Dst.
    for (auto &Var : ActiveMLocs[Src]) {
      auto ActiveVLocIt = ActiveVLocs.find(Var);
      assert(ActiveVLocIt != ActiveVLocs.end());
      ActiveVLocIt->second.Loc = Dst;

      MachineInstr *MI =
          MTracker->emitLoc(Dst, Var, ActiveVLocIt->second.Properties);
      PendingDbgValues.push_back(MI);
    }
    ActiveMLocs[Src].clear();
    flushDbgValues(Pos, nullptr);

    // XXX XXX XXX "pretend to be old LDV" means dropping all tracking data
    // about the old location.
    if (EmulateOldLDV)
      VarLocs[Src.asU64()] = ValueIDNum::EmptyValue;
  }

  MachineInstrBuilder emitMOLoc(const MachineOperand &MO,
                                const DebugVariable &Var,
                                const DbgValueProperties &Properties) {
    DebugLoc DL = DILocation::get(Var.getVariable()->getContext(), 0, 0,
                                  Var.getVariable()->getScope(),
                                  const_cast<DILocation *>(Var.getInlinedAt()));
    auto MIB = BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE));
    MIB.add(MO);
    if (Properties.Indirect)
      MIB.addImm(0);
    else
      MIB.addReg(0);
    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(Properties.DIExpr);
    return MIB;
  }
};

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

ValueIDNum ValueIDNum::EmptyValue = {UINT_MAX, UINT_MAX, UINT_MAX};

#ifndef NDEBUG
void DbgValue::dump(const MLocTracker *MTrack) const {
  if (Kind == Const) {
    MO->dump();
  } else if (Kind == NoVal) {
    dbgs() << "NoVal(" << BlockNo << ")";
  } else if (Kind == VPHI) {
    dbgs() << "VPHI(" << BlockNo << "," << MTrack->IDAsString(ID) << ")";
  } else {
    assert(Kind == Def);
    dbgs() << MTrack->IDAsString(ID);
  }
  if (Properties.Indirect)
    dbgs() << " indir";
  if (Properties.DIExpr)
    dbgs() << " " << *Properties.DIExpr;
}
#endif

MLocTracker::MLocTracker(MachineFunction &MF, const TargetInstrInfo &TII,
                         const TargetRegisterInfo &TRI,
                         const TargetLowering &TLI)
    : MF(MF), TII(TII), TRI(TRI), TLI(TLI),
      LocIdxToIDNum(ValueIDNum::EmptyValue), LocIdxToLocID(0) {
  NumRegs = TRI.getNumRegs();
  reset();
  LocIDToLocIdx.resize(NumRegs, LocIdx::MakeIllegalLoc());
  assert(NumRegs < (1u << NUM_LOC_BITS)); // Detect bit packing failure

  // Always track SP. This avoids the implicit clobbering caused by regmasks
  // from affectings its values. (LiveDebugValues disbelieves calls and
  // regmasks that claim to clobber SP).
  Register SP = TLI.getStackPointerRegisterToSaveRestore();
  if (SP) {
    unsigned ID = getLocID(SP, false);
    (void)lookupOrTrackRegister(ID);

    for (MCRegAliasIterator RAI(SP, &TRI, true); RAI.isValid(); ++RAI)
      SPAliases.insert(*RAI);
  }
}

LocIdx MLocTracker::trackRegister(unsigned ID) {
  assert(ID != 0);
  LocIdx NewIdx = LocIdx(LocIdxToIDNum.size());
  LocIdxToIDNum.grow(NewIdx);
  LocIdxToLocID.grow(NewIdx);

  // Default: it's an mphi.
  ValueIDNum ValNum = {CurBB, 0, NewIdx};
  // Was this reg ever touched by a regmask?
  for (const auto &MaskPair : reverse(Masks)) {
    if (MaskPair.first->clobbersPhysReg(ID)) {
      // There was an earlier def we skipped.
      ValNum = {CurBB, MaskPair.second, NewIdx};
      break;
    }
  }

  LocIdxToIDNum[NewIdx] = ValNum;
  LocIdxToLocID[NewIdx] = ID;
  return NewIdx;
}

void MLocTracker::writeRegMask(const MachineOperand *MO, unsigned CurBB,
                               unsigned InstID) {
  // Def any register we track have that isn't preserved. The regmask
  // terminates the liveness of a register, meaning its value can't be
  // relied upon -- we represent this by giving it a new value.
  for (auto Location : locations()) {
    unsigned ID = LocIdxToLocID[Location.Idx];
    // Don't clobber SP, even if the mask says it's clobbered.
    if (ID < NumRegs && !SPAliases.count(ID) && MO->clobbersPhysReg(ID))
      defReg(ID, CurBB, InstID);
  }
  Masks.push_back(std::make_pair(MO, InstID));
}

LocIdx MLocTracker::getOrTrackSpillLoc(SpillLoc L) {
  unsigned SpillID = SpillLocs.idFor(L);
  if (SpillID == 0) {
    SpillID = SpillLocs.insert(L);
    unsigned L = getLocID(SpillID, true);
    LocIdx Idx = LocIdx(LocIdxToIDNum.size()); // New idx
    LocIdxToIDNum.grow(Idx);
    LocIdxToLocID.grow(Idx);
    LocIDToLocIdx.push_back(Idx);
    LocIdxToLocID[Idx] = L;
    return Idx;
  } else {
    unsigned L = getLocID(SpillID, true);
    LocIdx Idx = LocIDToLocIdx[L];
    return Idx;
  }
}

std::string MLocTracker::LocIdxToName(LocIdx Idx) const {
  unsigned ID = LocIdxToLocID[Idx];
  if (ID >= NumRegs)
    return Twine("slot ").concat(Twine(ID - NumRegs)).str();
  else
    return TRI.getRegAsmName(ID).str();
}

std::string MLocTracker::IDAsString(const ValueIDNum &Num) const {
  std::string DefName = LocIdxToName(Num.getLoc());
  return Num.asString(DefName);
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void MLocTracker::dump() {
  for (auto Location : locations()) {
    std::string MLocName = LocIdxToName(Location.Value.getLoc());
    std::string DefName = Location.Value.asString(MLocName);
    dbgs() << LocIdxToName(Location.Idx) << " --> " << DefName << "\n";
  }
}

LLVM_DUMP_METHOD void MLocTracker::dump_mloc_map() {
  for (auto Location : locations()) {
    std::string foo = LocIdxToName(Location.Idx);
    dbgs() << "Idx " << Location.Idx.asU64() << " " << foo << "\n";
  }
}
#endif

MachineInstrBuilder MLocTracker::emitLoc(Optional<LocIdx> MLoc,
                                         const DebugVariable &Var,
                                         const DbgValueProperties &Properties) {
  DebugLoc DL = DILocation::get(Var.getVariable()->getContext(), 0, 0,
                                Var.getVariable()->getScope(),
                                const_cast<DILocation *>(Var.getInlinedAt()));
  auto MIB = BuildMI(MF, DL, TII.get(TargetOpcode::DBG_VALUE));

  const DIExpression *Expr = Properties.DIExpr;
  if (!MLoc) {
    // No location -> DBG_VALUE $noreg
    MIB.addReg(0);
    MIB.addReg(0);
  } else if (LocIdxToLocID[*MLoc] >= NumRegs) {
    unsigned LocID = LocIdxToLocID[*MLoc];
    const SpillLoc &Spill = SpillLocs[LocID - NumRegs + 1];

    auto *TRI = MF.getSubtarget().getRegisterInfo();
    Expr = TRI->prependOffsetExpression(Expr, DIExpression::ApplyOffset,
                                        Spill.SpillOffset);
    unsigned Base = Spill.SpillBase;
    MIB.addReg(Base);
    MIB.addImm(0);
  } else {
    unsigned LocID = LocIdxToLocID[*MLoc];
    MIB.addReg(LocID);
    if (Properties.Indirect)
      MIB.addImm(0);
    else
      MIB.addReg(0);
  }

  MIB.addMetadata(Var.getVariable());
  MIB.addMetadata(Expr);
  return MIB;
}

/// Default construct and initialize the pass.
InstrRefBasedLDV::InstrRefBasedLDV() {}

bool InstrRefBasedLDV::isCalleeSaved(LocIdx L) const {
  unsigned Reg = MTracker->LocIdxToLocID[L];
  for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
    if (CalleeSavedRegs.test(*RAI))
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
// Something to restore in the future.
// void InstrRefBasedLDV::printVarLocInMBB(..)
#endif

SpillLoc
InstrRefBasedLDV::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
  assert(MI.hasOneMemOperand() &&
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  Register Reg;
  StackOffset Offset = TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
  return {Reg, Offset};
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
bool InstrRefBasedLDV::transferDebugValue(const MachineInstr &MI) {
  if (!MI.isDebugValue())
    return false;

  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  DebugVariable V(Var, Expr, InlinedAt);
  DbgValueProperties Properties(MI);

  // If there are no instructions in this lexical scope, do no location tracking
  // at all, this variable shouldn't get a legitimate location range.
  auto *Scope = LS.findLexicalScope(MI.getDebugLoc().get());
  if (Scope == nullptr)
    return true; // handled it; by doing nothing

  // For now, ignore DBG_VALUE_LISTs when extending ranges. Allow it to
  // contribute to locations in this block, but don't propagate further.
  // Interpret it like a DBG_VALUE $noreg.
  if (MI.isDebugValueList()) {
    if (VTracker)
      VTracker->defVar(MI, Properties, None);
    if (TTracker)
      TTracker->redefVar(MI, Properties, None);
    return true;
  }

  const MachineOperand &MO = MI.getOperand(0);

  // MLocTracker needs to know that this register is read, even if it's only
  // read by a debug inst.
  if (MO.isReg() && MO.getReg() != 0)
    (void)MTracker->readReg(MO.getReg());

  // If we're preparing for the second analysis (variables), the machine value
  // locations are already solved, and we report this DBG_VALUE and the value
  // it refers to to VLocTracker.
  if (VTracker) {
    if (MO.isReg()) {
      // Feed defVar the new variable location, or if this is a
      // DBG_VALUE $noreg, feed defVar None.
      if (MO.getReg())
        VTracker->defVar(MI, Properties, MTracker->readReg(MO.getReg()));
      else
        VTracker->defVar(MI, Properties, None);
    } else if (MI.getOperand(0).isImm() || MI.getOperand(0).isFPImm() ||
               MI.getOperand(0).isCImm()) {
      VTracker->defVar(MI, MI.getOperand(0));
    }
  }

  // If performing final tracking of transfers, report this variable definition
  // to the TransferTracker too.
  if (TTracker)
    TTracker->redefVar(MI);
  return true;
}

bool InstrRefBasedLDV::transferDebugInstrRef(MachineInstr &MI,
                                             ValueIDNum **MLiveOuts,
                                             ValueIDNum **MLiveIns) {
  if (!MI.isDebugRef())
    return false;

  // Only handle this instruction when we are building the variable value
  // transfer function.
  if (!VTracker)
    return false;

  unsigned InstNo = MI.getOperand(0).getImm();
  unsigned OpNo = MI.getOperand(1).getImm();

  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  DebugVariable V(Var, Expr, InlinedAt);

  auto *Scope = LS.findLexicalScope(MI.getDebugLoc().get());
  if (Scope == nullptr)
    return true; // Handled by doing nothing. This variable is never in scope.

  const MachineFunction &MF = *MI.getParent()->getParent();

  // Various optimizations may have happened to the value during codegen,
  // recorded in the value substitution table. Apply any substitutions to
  // the instruction / operand number in this DBG_INSTR_REF, and collect
  // any subregister extractions performed during optimization.

  // Create dummy substitution with Src set, for lookup.
  auto SoughtSub =
      MachineFunction::DebugSubstitution({InstNo, OpNo}, {0, 0}, 0);

  SmallVector<unsigned, 4> SeenSubregs;
  auto LowerBoundIt = llvm::lower_bound(MF.DebugValueSubstitutions, SoughtSub);
  while (LowerBoundIt != MF.DebugValueSubstitutions.end() &&
         LowerBoundIt->Src == SoughtSub.Src) {
    std::tie(InstNo, OpNo) = LowerBoundIt->Dest;
    SoughtSub.Src = LowerBoundIt->Dest;
    if (unsigned Subreg = LowerBoundIt->Subreg)
      SeenSubregs.push_back(Subreg);
    LowerBoundIt = llvm::lower_bound(MF.DebugValueSubstitutions, SoughtSub);
  }

  // Default machine value number is <None> -- if no instruction defines
  // the corresponding value, it must have been optimized out.
  Optional<ValueIDNum> NewID = None;

  // Try to lookup the instruction number, and find the machine value number
  // that it defines. It could be an instruction, or a PHI.
  auto InstrIt = DebugInstrNumToInstr.find(InstNo);
  auto PHIIt = std::lower_bound(DebugPHINumToValue.begin(),
                                DebugPHINumToValue.end(), InstNo);
  if (InstrIt != DebugInstrNumToInstr.end()) {
    const MachineInstr &TargetInstr = *InstrIt->second.first;
    uint64_t BlockNo = TargetInstr.getParent()->getNumber();

    // Pick out the designated operand.
    assert(OpNo < TargetInstr.getNumOperands());
    const MachineOperand &MO = TargetInstr.getOperand(OpNo);

    // Today, this can only be a register.
    assert(MO.isReg() && MO.isDef());

    unsigned LocID = MTracker->getLocID(MO.getReg(), false);
    LocIdx L = MTracker->LocIDToLocIdx[LocID];
    NewID = ValueIDNum(BlockNo, InstrIt->second.second, L);
  } else if (PHIIt != DebugPHINumToValue.end() && PHIIt->InstrNum == InstNo) {
    // It's actually a PHI value. Which value it is might not be obvious, use
    // the resolver helper to find out.
    NewID = resolveDbgPHIs(*MI.getParent()->getParent(), MLiveOuts, MLiveIns,
                           MI, InstNo);
  }

  // Apply any subregister extractions, in reverse. We might have seen code
  // like this:
  //    CALL64 @foo, implicit-def $rax
  //    %0:gr64 = COPY $rax
  //    %1:gr32 = COPY %0.sub_32bit
  //    %2:gr16 = COPY %1.sub_16bit
  //    %3:gr8  = COPY %2.sub_8bit
  // In which case each copy would have been recorded as a substitution with
  // a subregister qualifier. Apply those qualifiers now.
  if (NewID && !SeenSubregs.empty()) {
    unsigned Offset = 0;
    unsigned Size = 0;

    // Look at each subregister that we passed through, and progressively
    // narrow in, accumulating any offsets that occur. Substitutions should
    // only ever be the same or narrower width than what they read from;
    // iterate in reverse order so that we go from wide to small.
    for (unsigned Subreg : reverse(SeenSubregs)) {
      unsigned ThisSize = TRI->getSubRegIdxSize(Subreg);
      unsigned ThisOffset = TRI->getSubRegIdxOffset(Subreg);
      Offset += ThisOffset;
      Size = (Size == 0) ? ThisSize : std::min(Size, ThisSize);
    }

    // If that worked, look for an appropriate subregister with the register
    // where the define happens. Don't look at values that were defined during
    // a stack write: we can't currently express register locations within
    // spills.
    LocIdx L = NewID->getLoc();
    if (NewID && !MTracker->isSpill(L)) {
      // Find the register class for the register where this def happened.
      // FIXME: no index for this?
      Register Reg = MTracker->LocIdxToLocID[L];
      const TargetRegisterClass *TRC = nullptr;
      for (auto *TRCI : TRI->regclasses())
        if (TRCI->contains(Reg))
          TRC = TRCI;
      assert(TRC && "Couldn't find target register class?");

      // If the register we have isn't the right size or in the right place,
      // Try to find a subregister inside it.
      unsigned MainRegSize = TRI->getRegSizeInBits(*TRC);
      if (Size != MainRegSize || Offset) {
        // Enumerate all subregisters, searching.
        Register NewReg = 0;
        for (MCSubRegIterator SRI(Reg, TRI, false); SRI.isValid(); ++SRI) {
          unsigned Subreg = TRI->getSubRegIndex(Reg, *SRI);
          unsigned SubregSize = TRI->getSubRegIdxSize(Subreg);
          unsigned SubregOffset = TRI->getSubRegIdxOffset(Subreg);
          if (SubregSize == Size && SubregOffset == Offset) {
            NewReg = *SRI;
            break;
          }
        }

        // If we didn't find anything: there's no way to express our value.
        if (!NewReg) {
          NewID = None;
        } else {
          // Re-state the value as being defined within the subregister
          // that we found.
          LocIdx NewLoc = MTracker->lookupOrTrackRegister(NewReg);
          NewID = ValueIDNum(NewID->getBlock(), NewID->getInst(), NewLoc);
        }
      }
    } else {
      // If we can't handle subregisters, unset the new value.
      NewID = None;
    }
  }

  // We, we have a value number or None. Tell the variable value tracker about
  // it. The rest of this LiveDebugValues implementation acts exactly the same
  // for DBG_INSTR_REFs as DBG_VALUEs (just, the former can refer to values that
  // aren't immediately available).
  DbgValueProperties Properties(Expr, false);
  VTracker->defVar(MI, Properties, NewID);

  // If we're on the final pass through the function, decompose this INSTR_REF
  // into a plain DBG_VALUE.
  if (!TTracker)
    return true;

  // Pick a location for the machine value number, if such a location exists.
  // (This information could be stored in TransferTracker to make it faster).
  Optional<LocIdx> FoundLoc = None;
  for (auto Location : MTracker->locations()) {
    LocIdx CurL = Location.Idx;
    ValueIDNum ID = MTracker->readMLoc(CurL);
    if (NewID && ID == NewID) {
      // If this is the first location with that value, pick it. Otherwise,
      // consider whether it's a "longer term" location.
      if (!FoundLoc) {
        FoundLoc = CurL;
        continue;
      }

      if (MTracker->isSpill(CurL))
        FoundLoc = CurL; // Spills are a longer term location.
      else if (!MTracker->isSpill(*FoundLoc) &&
               !MTracker->isSpill(CurL) &&
               !isCalleeSaved(*FoundLoc) &&
               isCalleeSaved(CurL))
        FoundLoc = CurL; // Callee saved regs are longer term than normal.
    }
  }

  // Tell transfer tracker that the variable value has changed.
  TTracker->redefVar(MI, Properties, FoundLoc);

  // If there was a value with no location; but the value is defined in a
  // later instruction in this block, this is a block-local use-before-def.
  if (!FoundLoc && NewID && NewID->getBlock() == CurBB &&
      NewID->getInst() > CurInst)
    TTracker->addUseBeforeDef(V, {MI.getDebugExpression(), false}, *NewID);

  // Produce a DBG_VALUE representing what this DBG_INSTR_REF meant.
  // This DBG_VALUE is potentially a $noreg / undefined location, if
  // FoundLoc is None.
  // (XXX -- could morph the DBG_INSTR_REF in the future).
  MachineInstr *DbgMI = MTracker->emitLoc(FoundLoc, V, Properties);
  TTracker->PendingDbgValues.push_back(DbgMI);
  TTracker->flushDbgValues(MI.getIterator(), nullptr);
  return true;
}

bool InstrRefBasedLDV::transferDebugPHI(MachineInstr &MI) {
  if (!MI.isDebugPHI())
    return false;

  // Analyse these only when solving the machine value location problem.
  if (VTracker || TTracker)
    return true;

  // First operand is the value location, either a stack slot or register.
  // Second is the debug instruction number of the original PHI.
  const MachineOperand &MO = MI.getOperand(0);
  unsigned InstrNum = MI.getOperand(1).getImm();

  if (MO.isReg()) {
    // The value is whatever's currently in the register. Read and record it,
    // to be analysed later.
    Register Reg = MO.getReg();
    ValueIDNum Num = MTracker->readReg(Reg);
    auto PHIRec = DebugPHIRecord(
        {InstrNum, MI.getParent(), Num, MTracker->lookupOrTrackRegister(Reg)});
    DebugPHINumToValue.push_back(PHIRec);

    // Subsequent register operations, or variable locations, might occur for
    // any of the subregisters of this DBG_PHIs operand. Ensure that all
    // registers aliasing this register are tracked.
    for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
      MTracker->lookupOrTrackRegister(*RAI);
  } else {
    // The value is whatever's in this stack slot.
    assert(MO.isFI());
    unsigned FI = MO.getIndex();

    // If the stack slot is dead, then this was optimized away.
    // FIXME: stack slot colouring should account for slots that get merged.
    if (MFI->isDeadObjectIndex(FI))
      return true;

    // Identify this spill slot.
    Register Base;
    StackOffset Offs = TFI->getFrameIndexReference(*MI.getMF(), FI, Base);
    SpillLoc SL = {Base, Offs};
    Optional<ValueIDNum> Num = MTracker->readSpill(SL);

    if (!Num)
      // Nothing ever writes to this slot. Curious, but nothing we can do.
      return true;

    // Record this DBG_PHI for later analysis.
    auto DbgPHI = DebugPHIRecord(
        {InstrNum, MI.getParent(), *Num, *MTracker->getSpillMLoc(SL)});
    DebugPHINumToValue.push_back(DbgPHI);
  }

  return true;
}

void InstrRefBasedLDV::transferRegisterDef(MachineInstr &MI) {
  // Meta Instructions do not affect the debug liveness of any register they
  // define.
  if (MI.isImplicitDef()) {
    // Except when there's an implicit def, and the location it's defining has
    // no value number. The whole point of an implicit def is to announce that
    // the register is live, without be specific about it's value. So define
    // a value if there isn't one already.
    ValueIDNum Num = MTracker->readReg(MI.getOperand(0).getReg());
    // Has a legitimate value -> ignore the implicit def.
    if (Num.getLoc() != 0)
      return;
    // Otherwise, def it here.
  } else if (MI.isMetaInstruction())
    return;

  // Find the regs killed by MI, and find regmasks of preserved regs.
  // Max out the number of statically allocated elements in `DeadRegs`, as this
  // prevents fallback to std::set::count() operations.
  SmallSet<uint32_t, 32> DeadRegs;
  SmallVector<const uint32_t *, 4> RegMasks;
  SmallVector<const MachineOperand *, 4> RegMaskPtrs;
  for (const MachineOperand &MO : MI.operands()) {
    // Determine whether the operand is a register def.
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        Register::isPhysicalRegister(MO.getReg()) &&
        !(MI.isCall() && MTracker->SPAliases.count(MO.getReg()))) {
      // Remove ranges of all aliased registers.
      for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
        // FIXME: Can we break out of this loop early if no insertion occurs?
        DeadRegs.insert(*RAI);
    } else if (MO.isRegMask()) {
      RegMasks.push_back(MO.getRegMask());
      RegMaskPtrs.push_back(&MO);
    }
  }

  // Tell MLocTracker about all definitions, of regmasks and otherwise.
  for (uint32_t DeadReg : DeadRegs)
    MTracker->defReg(DeadReg, CurBB, CurInst);

  for (auto *MO : RegMaskPtrs)
    MTracker->writeRegMask(MO, CurBB, CurInst);

  if (!TTracker)
    return;

  // When committing variable values to locations: tell transfer tracker that
  // we've clobbered things. It may be able to recover the variable from a
  // different location.

  // Inform TTracker about any direct clobbers.
  for (uint32_t DeadReg : DeadRegs) {
    LocIdx Loc = MTracker->lookupOrTrackRegister(DeadReg);
    TTracker->clobberMloc(Loc, MI.getIterator(), false);
  }

  // Look for any clobbers performed by a register mask. Only test locations
  // that are actually being tracked.
  for (auto L : MTracker->locations()) {
    // Stack locations can't be clobbered by regmasks.
    if (MTracker->isSpill(L.Idx))
      continue;

    Register Reg = MTracker->LocIdxToLocID[L.Idx];
    for (auto *MO : RegMaskPtrs)
      if (MO->clobbersPhysReg(Reg))
        TTracker->clobberMloc(L.Idx, MI.getIterator(), false);
  }
}

void InstrRefBasedLDV::performCopy(Register SrcRegNum, Register DstRegNum) {
  // In all circumstances, re-def all aliases. It's definitely a new value now.
  for (MCRegAliasIterator RAI(DstRegNum, TRI, true); RAI.isValid(); ++RAI)
    MTracker->defReg(*RAI, CurBB, CurInst);

  ValueIDNum SrcValue = MTracker->readReg(SrcRegNum);
  MTracker->setReg(DstRegNum, SrcValue);

  // Copy subregisters from one location to another.
  for (MCSubRegIndexIterator SRI(SrcRegNum, TRI); SRI.isValid(); ++SRI) {
    unsigned SrcSubReg = SRI.getSubReg();
    unsigned SubRegIdx = SRI.getSubRegIndex();
    unsigned DstSubReg = TRI->getSubReg(DstRegNum, SubRegIdx);
    if (!DstSubReg)
      continue;

    // Do copy. There are two matching subregisters, the source value should
    // have been def'd when the super-reg was, the latter might not be tracked
    // yet.
    // This will force SrcSubReg to be tracked, if it isn't yet. Will read
    // mphi values if it wasn't tracked.
    LocIdx SrcL = MTracker->lookupOrTrackRegister(SrcSubReg);
    LocIdx DstL = MTracker->lookupOrTrackRegister(DstSubReg);
    (void)SrcL;
    (void)DstL;
    ValueIDNum CpyValue = MTracker->readReg(SrcSubReg);

    MTracker->setReg(DstSubReg, CpyValue);
  }
}

bool InstrRefBasedLDV::isSpillInstruction(const MachineInstr &MI,
                                          MachineFunction *MF) {
  // TODO: Handle multiple stores folded into one.
  if (!MI.hasOneMemOperand())
    return false;

  if (!MI.getSpillSize(TII) && !MI.getFoldedSpillSize(TII))
    return false; // This is not a spill instruction, since no valid size was
                  // returned from either function.

  return true;
}

bool InstrRefBasedLDV::isLocationSpill(const MachineInstr &MI,
                                       MachineFunction *MF, unsigned &Reg) {
  if (!isSpillInstruction(MI, MF))
    return false;

  int FI;
  Reg = TII->isStoreToStackSlotPostFE(MI, FI);
  return Reg != 0;
}

Optional<SpillLoc>
InstrRefBasedLDV::isRestoreInstruction(const MachineInstr &MI,
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

bool InstrRefBasedLDV::transferSpillOrRestoreInst(MachineInstr &MI) {
  // XXX -- it's too difficult to implement VarLocBasedImpl's  stack location
  // limitations under the new model. Therefore, when comparing them, compare
  // versions that don't attempt spills or restores at all.
  if (EmulateOldLDV)
    return false;

  MachineFunction *MF = MI.getMF();
  unsigned Reg;
  Optional<SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, terminate that variable location. The value in memory
  // will have changed. DbgEntityHistoryCalculator doesn't try to detect this.
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);

    if (TTracker) {
      Optional<LocIdx> MLoc = MTracker->getSpillMLoc(*Loc);
      if (MLoc) {
        // Un-set this location before clobbering, so that we don't salvage
        // the variable location back to the same place.
        MTracker->setMLoc(*MLoc, ValueIDNum::EmptyValue);
        TTracker->clobberMloc(*MLoc, MI.getIterator());
      }
    }
  }

  // Try to recognise spill and restore instructions that may transfer a value.
  if (isLocationSpill(MI, MF, Reg)) {
    Loc = extractSpillBaseRegAndOffset(MI);
    auto ValueID = MTracker->readReg(Reg);

    // If the location is empty, produce a phi, signify it's the live-in value.
    if (ValueID.getLoc() == 0)
      ValueID = {CurBB, 0, MTracker->getRegMLoc(Reg)};

    MTracker->setSpill(*Loc, ValueID);
    auto OptSpillLocIdx = MTracker->getSpillMLoc(*Loc);
    assert(OptSpillLocIdx && "Spill slot set but has no LocIdx?");
    LocIdx SpillLocIdx = *OptSpillLocIdx;

    // Tell TransferTracker about this spill, produce DBG_VALUEs for it.
    if (TTracker)
      TTracker->transferMlocs(MTracker->getRegMLoc(Reg), SpillLocIdx,
                              MI.getIterator());
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return false;

    // Is there a value to be restored?
    auto OptValueID = MTracker->readSpill(*Loc);
    if (OptValueID) {
      ValueIDNum ValueID = *OptValueID;
      LocIdx SpillLocIdx = *MTracker->getSpillMLoc(*Loc);
      // XXX -- can we recover sub-registers of this value? Until we can, first
      // overwrite all defs of the register being restored to.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);

      // Now override the reg we're restoring to.
      MTracker->setReg(Reg, ValueID);

      // Report this restore to the transfer tracker too.
      if (TTracker)
        TTracker->transferMlocs(SpillLocIdx, MTracker->getRegMLoc(Reg),
                                MI.getIterator());
    } else {
      // There isn't anything in the location; not clear if this is a code path
      // that still runs. Def this register anyway just in case.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);

      // Force the spill slot to be tracked.
      LocIdx L = MTracker->getOrTrackSpillLoc(*Loc);

      // Set the restored value to be a machine phi number, signifying that it's
      // whatever the spills live-in value is in this block. Definitely has
      // a LocIdx due to the setSpill above.
      ValueIDNum ValueID = {CurBB, 0, L};
      MTracker->setReg(Reg, ValueID);
      MTracker->setSpill(*Loc, ValueID);
    }
  }
  return true;
}

bool InstrRefBasedLDV::transferRegisterCopy(MachineInstr &MI) {
  auto DestSrc = TII->isCopyInstr(MI);
  if (!DestSrc)
    return false;

  const MachineOperand *DestRegOp = DestSrc->Destination;
  const MachineOperand *SrcRegOp = DestSrc->Source;

  auto isCalleeSavedReg = [&](unsigned Reg) {
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  };

  Register SrcReg = SrcRegOp->getReg();
  Register DestReg = DestRegOp->getReg();

  // Ignore identity copies. Yep, these make it as far as LiveDebugValues.
  if (SrcReg == DestReg)
    return true;

  // For emulating VarLocBasedImpl:
  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register, which is callee saved, is
  // going to stay unclobbered longer, even if it is killed.
  //
  // For InstrRefBasedImpl, we can track multiple locations per value, so
  // ignore this condition.
  if (EmulateOldLDV && !isCalleeSavedReg(DestReg))
    return false;

  // InstrRefBasedImpl only followed killing copies.
  if (EmulateOldLDV && !SrcRegOp->isKill())
    return false;

  // Copy MTracker info, including subregs if available.
  InstrRefBasedLDV::performCopy(SrcReg, DestReg);

  // Only produce a transfer of DBG_VALUE within a block where old LDV
  // would have. We might make use of the additional value tracking in some
  // other way, later.
  if (TTracker && isCalleeSavedReg(DestReg) && SrcRegOp->isKill())
    TTracker->transferMlocs(MTracker->getRegMLoc(SrcReg),
                            MTracker->getRegMLoc(DestReg), MI.getIterator());

  // VarLocBasedImpl would quit tracking the old location after copying.
  if (EmulateOldLDV && SrcReg != DestReg)
    MTracker->defReg(SrcReg, CurBB, CurInst);

  // Finally, the copy might have clobbered variables based on the destination
  // register. Tell TTracker about it, in case a backup location exists.
  if (TTracker) {
    for (MCRegAliasIterator RAI(DestReg, TRI, true); RAI.isValid(); ++RAI) {
      LocIdx ClobberedLoc = MTracker->getRegMLoc(*RAI);
      TTracker->clobberMloc(ClobberedLoc, MI.getIterator(), false);
    }
  }

  return true;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
void InstrRefBasedLDV::accumulateFragmentMap(MachineInstr &MI) {
  DebugVariable MIVar(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
  FragmentInfo ThisFragment = MIVar.getFragmentOrDefault();

  // If this is the first sighting of this variable, then we are guaranteed
  // there are currently no overlapping fragments either. Initialize the set
  // of seen fragments, record no overlaps for the current one, and return.
  auto SeenIt = SeenFragments.find(MIVar.getVariable());
  if (SeenIt == SeenFragments.end()) {
    SmallSet<FragmentInfo, 4> OneFragment;
    OneFragment.insert(ThisFragment);
    SeenFragments.insert({MIVar.getVariable(), OneFragment});

    OverlapFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlapFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
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
          OverlapFragments.find({MIVar.getVariable(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlapFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

void InstrRefBasedLDV::process(MachineInstr &MI, ValueIDNum **MLiveOuts,
                               ValueIDNum **MLiveIns) {
  // Try to interpret an MI as a debug or transfer instruction. Only if it's
  // none of these should we interpret it's register defs as new value
  // definitions.
  if (transferDebugValue(MI))
    return;
  if (transferDebugInstrRef(MI, MLiveOuts, MLiveIns))
    return;
  if (transferDebugPHI(MI))
    return;
  if (transferRegisterCopy(MI))
    return;
  if (transferSpillOrRestoreInst(MI))
    return;
  transferRegisterDef(MI);
}

void InstrRefBasedLDV::produceMLocTransferFunction(
    MachineFunction &MF, SmallVectorImpl<MLocTransferMap> &MLocTransfer,
    unsigned MaxNumBlocks) {
  // Because we try to optimize around register mask operands by ignoring regs
  // that aren't currently tracked, we set up something ugly for later: RegMask
  // operands that are seen earlier than the first use of a register, still need
  // to clobber that register in the transfer function. But this information
  // isn't actively recorded. Instead, we track each RegMask used in each block,
  // and accumulated the clobbered but untracked registers in each block into
  // the following bitvector. Later, if new values are tracked, we can add
  // appropriate clobbers.
  SmallVector<BitVector, 32> BlockMasks;
  BlockMasks.resize(MaxNumBlocks);

  // Reserve one bit per register for the masks described above.
  unsigned BVWords = MachineOperand::getRegMaskSize(TRI->getNumRegs());
  for (auto &BV : BlockMasks)
    BV.resize(TRI->getNumRegs(), true);

  // Step through all instructions and inhale the transfer function.
  for (auto &MBB : MF) {
    // Object fields that are read by trackers to know where we are in the
    // function.
    CurBB = MBB.getNumber();
    CurInst = 1;

    // Set all machine locations to a PHI value. For transfer function
    // production only, this signifies the live-in value to the block.
    MTracker->reset();
    MTracker->setMPhis(CurBB);

    // Step through each instruction in this block.
    for (auto &MI : MBB) {
      process(MI);
      // Also accumulate fragment map.
      if (MI.isDebugValue())
        accumulateFragmentMap(MI);

      // Create a map from the instruction number (if present) to the
      // MachineInstr and its position.
      if (uint64_t InstrNo = MI.peekDebugInstrNum()) {
        auto InstrAndPos = std::make_pair(&MI, CurInst);
        auto InsertResult =
            DebugInstrNumToInstr.insert(std::make_pair(InstrNo, InstrAndPos));

        // There should never be duplicate instruction numbers.
        assert(InsertResult.second);
        (void)InsertResult;
      }

      ++CurInst;
    }

    // Produce the transfer function, a map of machine location to new value. If
    // any machine location has the live-in phi value from the start of the
    // block, it's live-through and doesn't need recording in the transfer
    // function.
    for (auto Location : MTracker->locations()) {
      LocIdx Idx = Location.Idx;
      ValueIDNum &P = Location.Value;
      if (P.isPHI() && P.getLoc() == Idx.asU64())
        continue;

      // Insert-or-update.
      auto &TransferMap = MLocTransfer[CurBB];
      auto Result = TransferMap.insert(std::make_pair(Idx.asU64(), P));
      if (!Result.second)
        Result.first->second = P;
    }

    // Accumulate any bitmask operands into the clobberred reg mask for this
    // block.
    for (auto &P : MTracker->Masks) {
      BlockMasks[CurBB].clearBitsNotInMask(P.first->getRegMask(), BVWords);
    }
  }

  // Compute a bitvector of all the registers that are tracked in this block.
  BitVector UsedRegs(TRI->getNumRegs());
  for (auto Location : MTracker->locations()) {
    unsigned ID = MTracker->LocIdxToLocID[Location.Idx];
    // Ignore stack slots, and aliases of the stack pointer.
    if (ID >= TRI->getNumRegs() || MTracker->SPAliases.count(ID))
      continue;
    UsedRegs.set(ID);
  }

  // Check that any regmask-clobber of a register that gets tracked, is not
  // live-through in the transfer function. It needs to be clobbered at the
  // very least.
  for (unsigned int I = 0; I < MaxNumBlocks; ++I) {
    BitVector &BV = BlockMasks[I];
    BV.flip();
    BV &= UsedRegs;
    // This produces all the bits that we clobber, but also use. Check that
    // they're all clobbered or at least set in the designated transfer
    // elem.
    for (unsigned Bit : BV.set_bits()) {
      unsigned ID = MTracker->getLocID(Bit, false);
      LocIdx Idx = MTracker->LocIDToLocIdx[ID];
      auto &TransferMap = MLocTransfer[I];

      // Install a value representing the fact that this location is effectively
      // written to in this block. As there's no reserved value, instead use
      // a value number that is never generated. Pick the value number for the
      // first instruction in the block, def'ing this location, which we know
      // this block never used anyway.
      ValueIDNum NotGeneratedNum = ValueIDNum(I, 1, Idx);
      auto Result =
        TransferMap.insert(std::make_pair(Idx.asU64(), NotGeneratedNum));
      if (!Result.second) {
        ValueIDNum &ValueID = Result.first->second;
        if (ValueID.getBlock() == I && ValueID.isPHI())
          // It was left as live-through. Set it to clobbered.
          ValueID = NotGeneratedNum;
      }
    }
  }
}

bool InstrRefBasedLDV::mlocJoin(
    MachineBasicBlock &MBB, SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    ValueIDNum **OutLocs, ValueIDNum *InLocs) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Handle value-propagation when control flow merges on entry to a block. For
  // any location without a PHI already placed, the location has the same value
  // as its predecessors. If a PHI is placed, test to see whether it's now a
  // redundant PHI that we can eliminate.

  SmallVector<const MachineBasicBlock *, 8> BlockOrders;
  for (auto Pred : MBB.predecessors())
    BlockOrders.push_back(Pred);

  // Visit predecessors in RPOT order.
  auto Cmp = [&](const MachineBasicBlock *A, const MachineBasicBlock *B) {
    return BBToOrder.find(A)->second < BBToOrder.find(B)->second;
  };
  llvm::sort(BlockOrders, Cmp);

  // Skip entry block.
  if (BlockOrders.size() == 0)
    return false;

  // Step through all machine locations, look at each predecessor and test
  // whether we can eliminate redundant PHIs.
  for (auto Location : MTracker->locations()) {
    LocIdx Idx = Location.Idx;

    // Pick out the first predecessors live-out value for this location. It's
    // guaranteed to not be a backedge, as we order by RPO.
    ValueIDNum FirstVal = OutLocs[BlockOrders[0]->getNumber()][Idx.asU64()];

    // If we've already eliminated a PHI here, do no further checking, just
    // propagate the first live-in value into this block.
    if (InLocs[Idx.asU64()] != ValueIDNum(MBB.getNumber(), 0, Idx)) {
      if (InLocs[Idx.asU64()] != FirstVal) {
        InLocs[Idx.asU64()] = FirstVal;
        Changed |= true;
      }
      continue;
    }

    // We're now examining a PHI to see whether it's un-necessary. Loop around
    // the other live-in values and test whether they're all the same.
    bool Disagree = false;
    for (unsigned int I = 1; I < BlockOrders.size(); ++I) {
      const MachineBasicBlock *PredMBB = BlockOrders[I];
      const ValueIDNum &PredLiveOut =
          OutLocs[PredMBB->getNumber()][Idx.asU64()];

      // Incoming values agree, continue trying to eliminate this PHI.
      if (FirstVal == PredLiveOut)
        continue;

      // We can also accept a PHI value that feeds back into itself.
      if (PredLiveOut == ValueIDNum(MBB.getNumber(), 0, Idx))
        continue;

      // Live-out of a predecessor disagrees with the first predecessor.
      Disagree = true;
    }

    // No disagreement? No PHI. Otherwise, leave the PHI in live-ins.
    if (!Disagree) {
      InLocs[Idx.asU64()] = FirstVal;
      Changed |= true;
    }
  }

  // TODO: Reimplement NumInserted and NumRemoved.
  return Changed;
}

void InstrRefBasedLDV::placeMLocPHIs(MachineFunction &MF,
                              SmallPtrSetImpl<MachineBasicBlock *> &AllBlocks,
                              ValueIDNum **MInLocs,
                              SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
  // To avoid repeatedly running the PHI placement algorithm, leverage the
  // fact that a def of register MUST also def its register units. Find the
  // units for registers, place PHIs for them, and then replicate them for 
  // aliasing registers. Some inputs that are never def'd (DBG_PHIs of
  // arguments) don't lead to register units being tracked, just place PHIs for
  // those registers directly. Do the same for stack slots.
  SmallSet<Register, 32> RegUnitsToPHIUp;
  SmallSet<LocIdx, 32> LocsToPHI;
  for (auto Location : MTracker->locations()) {
    LocIdx L = Location.Idx;
    if (MTracker->isSpill(L)) {
      LocsToPHI.insert(L);
      continue;
    }

    Register R = MTracker->LocIdxToLocID[L];
    SmallSet<Register, 8> FoundRegUnits;
    bool AnyIllegal = false;
    for (MCRegUnitIterator RUI(R.asMCReg(), TRI); RUI.isValid(); ++RUI) {
      for (MCRegUnitRootIterator URoot(*RUI, TRI); URoot.isValid(); ++URoot){
        if (!MTracker->isRegisterTracked(*URoot)) {
          // Not all roots were loaded into the tracking map: this register
          // isn't actually def'd anywhere, we only read from it. Generate PHIs
          // for this reg, but don't iterate units.
          AnyIllegal = true;
        } else {
          FoundRegUnits.insert(*URoot);
        }
      }
    }

    if (AnyIllegal) {
      LocsToPHI.insert(L);
      continue;
    }

    RegUnitsToPHIUp.insert(FoundRegUnits.begin(), FoundRegUnits.end());
  }

  // Lambda to fetch PHIs for a given location, and write into the PHIBlocks
  // collection.
  SmallVector<MachineBasicBlock *, 32> PHIBlocks;
  auto CollectPHIsForLoc = [&](LocIdx L) {
    // Collect the set of defs.
    SmallPtrSet<MachineBasicBlock *, 32> DefBlocks;
    for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
      MachineBasicBlock *MBB = OrderToBB[I];
      const auto &TransferFunc = MLocTransfer[MBB->getNumber()];
      if (TransferFunc.find(L) != TransferFunc.end())
        DefBlocks.insert(MBB);
    }

    // The entry block defs the location too: it's the live-in / argument value.
    // Only insert if there are other defs though; everything is trivially live
    // through otherwise.
    if (!DefBlocks.empty())
      DefBlocks.insert(&*MF.begin());

    // Ask the SSA construction algorithm where we should put PHIs. Clear
    // anything that might have been hanging around from earlier.
    PHIBlocks.clear();
    BlockPHIPlacement(AllBlocks, DefBlocks, PHIBlocks);
  };

  // For spill slots, and locations with no reg units, just place PHIs.
  for (LocIdx L : LocsToPHI) {
    CollectPHIsForLoc(L);
    // Install those PHI values into the live-in value array.
    for (const MachineBasicBlock *MBB : PHIBlocks)
      MInLocs[MBB->getNumber()][L.asU64()] = ValueIDNum(MBB->getNumber(), 0, L);
  }

  // For reg units, place PHIs, and then place them for any aliasing registers.
  for (Register R : RegUnitsToPHIUp) {
    LocIdx L = MTracker->lookupOrTrackRegister(R);
    CollectPHIsForLoc(L);

    // Install those PHI values into the live-in value array.
    for (const MachineBasicBlock *MBB : PHIBlocks)
      MInLocs[MBB->getNumber()][L.asU64()] = ValueIDNum(MBB->getNumber(), 0, L);

    // Now find aliases and install PHIs for those.
    for (MCRegAliasIterator RAI(R, TRI, true); RAI.isValid(); ++RAI) {
      // Super-registers that are "above" the largest register read/written by
      // the function will alias, but will not be tracked.
      if (!MTracker->isRegisterTracked(*RAI))
        continue;

      LocIdx AliasLoc = MTracker->lookupOrTrackRegister(*RAI);
      for (const MachineBasicBlock *MBB : PHIBlocks)
        MInLocs[MBB->getNumber()][AliasLoc.asU64()] =
            ValueIDNum(MBB->getNumber(), 0, AliasLoc);
    }
  }
}

void InstrRefBasedLDV::buildMLocValueMap(
    MachineFunction &MF, ValueIDNum **MInLocs, ValueIDNum **MOutLocs,
    SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  // We track what is on the current and pending worklist to avoid inserting
  // the same thing twice. We could avoid this with a custom priority queue,
  // but this is probably not worth it.
  SmallPtrSet<MachineBasicBlock *, 16> OnPending, OnWorklist;

  // Initialize worklist with every block to be visited. Also produce list of
  // all blocks.
  SmallPtrSet<MachineBasicBlock *, 32> AllBlocks;
  for (unsigned int I = 0; I < BBToOrder.size(); ++I) {
    Worklist.push(I);
    OnWorklist.insert(OrderToBB[I]);
    AllBlocks.insert(OrderToBB[I]);
  }

  // Initialize entry block to PHIs. These represent arguments.
  for (auto Location : MTracker->locations())
    MInLocs[0][Location.Idx.asU64()] = ValueIDNum(0, 0, Location.Idx);

  MTracker->reset();

  // Start by placing PHIs, using the usual SSA constructor algorithm. Consider
  // any machine-location that isn't live-through a block to be def'd in that
  // block.
  placeMLocPHIs(MF, AllBlocks, MInLocs, MLocTransfer);

  // Propagate values to eliminate redundant PHIs. At the same time, this
  // produces the table of Block x Location => Value for the entry to each
  // block.
  // The kind of PHIs we can eliminate are, for example, where one path in a
  // conditional spills and restores a register, and the register still has
  // the same value once control flow joins, unbeknowns to the PHI placement
  // code. Propagating values allows us to identify such un-necessary PHIs and
  // remove them.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // Vector for storing the evaluated block transfer function.
    SmallVector<std::pair<LocIdx, ValueIDNum>, 32> ToRemap;

    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      CurBB = MBB->getNumber();
      Worklist.pop();

      // Join the values in all predecessor blocks.
      bool InLocsChanged;
      InLocsChanged = mlocJoin(*MBB, Visited, MOutLocs, MInLocs[CurBB]);
      InLocsChanged |= Visited.insert(MBB).second;

      // Don't examine transfer function if we've visited this loc at least
      // once, and inlocs haven't changed.
      if (!InLocsChanged)
        continue;

      // Load the current set of live-ins into MLocTracker.
      MTracker->loadFromArray(MInLocs[CurBB], CurBB);

      // Each element of the transfer function can be a new def, or a read of
      // a live-in value. Evaluate each element, and store to "ToRemap".
      ToRemap.clear();
      for (auto &P : MLocTransfer[CurBB]) {
        if (P.second.getBlock() == CurBB && P.second.isPHI()) {
          // This is a movement of whatever was live in. Read it.
          ValueIDNum NewID = MTracker->readMLoc(P.second.getLoc());
          ToRemap.push_back(std::make_pair(P.first, NewID));
        } else {
          // It's a def. Just set it.
          assert(P.second.getBlock() == CurBB);
          ToRemap.push_back(std::make_pair(P.first, P.second));
        }
      }

      // Commit the transfer function changes into mloc tracker, which
      // transforms the contents of the MLocTracker into the live-outs.
      for (auto &P : ToRemap)
        MTracker->setMLoc(P.first, P.second);

      // Now copy out-locs from mloc tracker into out-loc vector, checking
      // whether changes have occurred. These changes can have come from both
      // the transfer function, and mlocJoin.
      bool OLChanged = false;
      for (auto Location : MTracker->locations()) {
        OLChanged |= MOutLocs[CurBB][Location.Idx.asU64()] != Location.Value;
        MOutLocs[CurBB][Location.Idx.asU64()] = Location.Value;
      }

      MTracker->reset();

      // No need to examine successors again if out-locs didn't change.
      if (!OLChanged)
        continue;

      // All successors should be visited: put any back-edges on the pending
      // list for the next pass-through, and any other successors to be
      // visited this pass, if they're not going to be already.
      for (auto s : MBB->successors()) {
        // Does branching to this successor represent a back-edge?
        if (BBToOrder[s] > BBToOrder[MBB]) {
          // No: visit it during this dataflow iteration.
          if (OnWorklist.insert(s).second)
            Worklist.push(BBToOrder[s]);
        } else {
          // Yes: visit it on the next iteration.
          if (OnPending.insert(s).second)
            Pending.push(BBToOrder[s]);
        }
      }
    }

    Worklist.swap(Pending);
    std::swap(OnPending, OnWorklist);
    OnPending.clear();
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  // Once all the live-ins don't change on mlocJoin(), we've eliminated all
  // redundant PHIs.
}

// Boilerplate for feeding MachineBasicBlocks into IDF calculator. Provide
// template specialisations for graph traits and a successor enumerator.
namespace llvm {
template <> struct GraphTraits<MachineBasicBlock> {
  using NodeRef = MachineBasicBlock *;
  using ChildIteratorType = MachineBasicBlock::succ_iterator;

  static NodeRef getEntryNode(MachineBasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return N->succ_begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->succ_end(); }
};

template <> struct GraphTraits<const MachineBasicBlock> {
  using NodeRef = const MachineBasicBlock *;
  using ChildIteratorType = MachineBasicBlock::const_succ_iterator;

  static NodeRef getEntryNode(const MachineBasicBlock *BB) { return BB; }
  static ChildIteratorType child_begin(NodeRef N) { return N->succ_begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->succ_end(); }
};

using MachineDomTreeBase = DomTreeBase<MachineBasicBlock>::NodeType;
using MachineDomTreeChildGetter =
    typename IDFCalculatorDetail::ChildrenGetterTy<MachineDomTreeBase, false>;

namespace IDFCalculatorDetail {
template <>
typename MachineDomTreeChildGetter::ChildrenTy
MachineDomTreeChildGetter::get(const NodeRef &N) {
  return {N->succ_begin(), N->succ_end()};
}
} // namespace IDFCalculatorDetail
} // namespace llvm

void InstrRefBasedLDV::BlockPHIPlacement(
    const SmallPtrSetImpl<MachineBasicBlock *> &AllBlocks,
    const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
    SmallVectorImpl<MachineBasicBlock *> &PHIBlocks) {
  // Apply IDF calculator to the designated set of location defs, storing
  // required PHIs into PHIBlocks. Uses the dominator tree stored in the
  // InstrRefBasedLDV object.
  IDFCalculatorDetail::ChildrenGetterTy<MachineDomTreeBase, false> foo;
  IDFCalculatorBase<MachineDomTreeBase, false> IDF(DomTree->getBase(), foo);

  IDF.setLiveInBlocks(AllBlocks);
  IDF.setDefiningBlocks(DefBlocks);
  IDF.calculate(PHIBlocks);
}

Optional<ValueIDNum> InstrRefBasedLDV::pickVPHILoc(
    const MachineBasicBlock &MBB, const DebugVariable &Var,
    const LiveIdxT &LiveOuts, ValueIDNum **MOutLocs,
    const SmallVectorImpl<const MachineBasicBlock *> &BlockOrders) {
  // Collect a set of locations from predecessor where its live-out value can
  // be found.
  SmallVector<SmallVector<LocIdx, 4>, 8> Locs;
  SmallVector<const DbgValueProperties *, 4> Properties;
  unsigned NumLocs = MTracker->getNumLocs();

  // No predecessors means no PHIs.
  if (BlockOrders.empty())
    return None;

  for (auto p : BlockOrders) {
    unsigned ThisBBNum = p->getNumber();
    auto OutValIt = LiveOuts.find(p);
    if (OutValIt == LiveOuts.end())
      // If we have a predecessor not in scope, we'll never find a PHI position.
      return None;
    const DbgValue &OutVal = *OutValIt->second;

    if (OutVal.Kind == DbgValue::Const || OutVal.Kind == DbgValue::NoVal)
      // Consts and no-values cannot have locations we can join on.
      return None;

    Properties.push_back(&OutVal.Properties);

    // Create new empty vector of locations.
    Locs.resize(Locs.size() + 1);

    // If the live-in value is a def, find the locations where that value is
    // present. Do the same for VPHIs where we know the VPHI value.
    if (OutVal.Kind == DbgValue::Def ||
        (OutVal.Kind == DbgValue::VPHI && OutVal.BlockNo != MBB.getNumber() &&
         OutVal.ID != ValueIDNum::EmptyValue)) {
      ValueIDNum ValToLookFor = OutVal.ID;
      // Search the live-outs of the predecessor for the specified value.
      for (unsigned int I = 0; I < NumLocs; ++I) {
        if (MOutLocs[ThisBBNum][I] == ValToLookFor)
          Locs.back().push_back(LocIdx(I));
      }
    } else {
      assert(OutVal.Kind == DbgValue::VPHI);
      // For VPHIs where we don't know the location, we definitely can't find
      // a join loc.
      if (OutVal.BlockNo != MBB.getNumber())
        return None;

      // Otherwise: this is a VPHI on a backedge feeding back into itself, i.e.
      // a value that's live-through the whole loop. (It has to be a backedge,
      // because a block can't dominate itself). We can accept as a PHI location
      // any location where the other predecessors agree, _and_ the machine
      // locations feed back into themselves. Therefore, add all self-looping
      // machine-value PHI locations.
      for (unsigned int I = 0; I < NumLocs; ++I) {
        ValueIDNum MPHI(MBB.getNumber(), 0, LocIdx(I));
        if (MOutLocs[ThisBBNum][I] == MPHI)
          Locs.back().push_back(LocIdx(I));
      }
    }
  }

  // We should have found locations for all predecessors, or returned.
  assert(Locs.size() == BlockOrders.size());

  // Check that all properties are the same. We can't pick a location if they're
  // not.
  const DbgValueProperties *Properties0 = Properties[0];
  for (auto *Prop : Properties)
    if (*Prop != *Properties0)
      return None;

  // Starting with the first set of locations, take the intersection with
  // subsequent sets.
  SmallVector<LocIdx, 4> CandidateLocs = Locs[0];
  for (unsigned int I = 1; I < Locs.size(); ++I) {
    auto &LocVec = Locs[I];
    SmallVector<LocIdx, 4> NewCandidates;
    std::set_intersection(CandidateLocs.begin(), CandidateLocs.end(),
                          LocVec.begin(), LocVec.end(), std::inserter(NewCandidates, NewCandidates.begin()));
    CandidateLocs = NewCandidates;
  }
  if (CandidateLocs.empty())
    return None;

  // We now have a set of LocIdxes that contain the right output value in
  // each of the predecessors. Pick the lowest; if there's a register loc,
  // that'll be it.
  LocIdx L = *CandidateLocs.begin();

  // Return a PHI-value-number for the found location.
  ValueIDNum PHIVal = {(unsigned)MBB.getNumber(), 0, L};
  return PHIVal;
}

bool InstrRefBasedLDV::vlocJoin(
    MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs,
    SmallPtrSet<const MachineBasicBlock *, 8> &InScopeBlocks,
    SmallPtrSet<const MachineBasicBlock *, 8> &BlocksToExplore,
    DbgValue &LiveIn) {
  // To emulate VarLocBasedImpl, process this block if it's not in scope but
  // _does_ assign a variable value. No live-ins for this scope are transferred
  // in though, so we can return immediately.
  if (InScopeBlocks.count(&MBB) == 0 && !ArtificialBlocks.count(&MBB))
    return false;

  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Order predecessors by RPOT order, for exploring them in that order.
  SmallVector<MachineBasicBlock *, 8> BlockOrders(MBB.predecessors());

  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  llvm::sort(BlockOrders, Cmp);

  unsigned CurBlockRPONum = BBToOrder[&MBB];

  // Collect all the incoming DbgValues for this variable, from predecessor
  // live-out values.
  SmallVector<InValueT, 8> Values;
  bool Bail = false;
  int BackEdgesStart = 0;
  for (auto p : BlockOrders) {
    // If the predecessor isn't in scope / to be explored, we'll never be
    // able to join any locations.
    if (!BlocksToExplore.contains(p)) {
      Bail = true;
      break;
    }

    // All Live-outs will have been initialized.
    DbgValue &OutLoc = *VLOCOutLocs.find(p)->second;

    // Keep track of where back-edges begin in the Values vector. Relies on
    // BlockOrders being sorted by RPO.
    unsigned ThisBBRPONum = BBToOrder[p];
    if (ThisBBRPONum < CurBlockRPONum)
      ++BackEdgesStart;

    Values.push_back(std::make_pair(p, &OutLoc));
  }

  // If there were no values, or one of the predecessors couldn't have a
  // value, then give up immediately. It's not safe to produce a live-in
  // value. Leave as whatever it was before.
  if (Bail || Values.size() == 0)
    return false;

  // All (non-entry) blocks have at least one non-backedge predecessor.
  // Pick the variable value from the first of these, to compare against
  // all others.
  const DbgValue &FirstVal = *Values[0].second;

  // If the old live-in value is not a PHI then either a) no PHI is needed
  // here, or b) we eliminated the PHI that was here. If so, we can just
  // propagate in the first parent's incoming value.
  if (LiveIn.Kind != DbgValue::VPHI || LiveIn.BlockNo != MBB.getNumber()) {
    Changed = LiveIn != FirstVal;
    if (Changed)
      LiveIn = FirstVal;
    return Changed;
  }

  // Scan for variable values that can never be resolved: if they have
  // different DIExpressions, different indirectness, or are mixed constants /
  // non-constants.
  for (auto &V : Values) {
    if (V.second->Properties != FirstVal.Properties)
      return false;
    if (V.second->Kind == DbgValue::NoVal)
      return false;
    if (V.second->Kind == DbgValue::Const && FirstVal.Kind != DbgValue::Const)
      return false;
  }

  // Try to eliminate this PHI. Do the incoming values all agree?
  bool Disagree = false;
  for (auto &V : Values) {
    if (*V.second == FirstVal)
      continue; // No disagreement.

    // Eliminate if a backedge feeds a VPHI back into itself.
    if (V.second->Kind == DbgValue::VPHI &&
        V.second->BlockNo == MBB.getNumber() &&
        // Is this a backedge?
        std::distance(Values.begin(), &V) >= BackEdgesStart)
      continue;

    Disagree = true;
  }

  // No disagreement -> live-through value.
  if (!Disagree) {
    Changed = LiveIn != FirstVal;
    if (Changed)
      LiveIn = FirstVal;
    return Changed;
  } else {
    // Otherwise use a VPHI.
    DbgValue VPHI(MBB.getNumber(), FirstVal.Properties, DbgValue::VPHI);
    Changed = LiveIn != VPHI;
    if (Changed)
      LiveIn = VPHI;
    return Changed;
  }
}

void InstrRefBasedLDV::buildVLocValueMap(const DILocation *DILoc,
    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks, LiveInsT &Output,
    ValueIDNum **MOutLocs, ValueIDNum **MInLocs,
    SmallVectorImpl<VLocTracker> &AllTheVLocs) {
  // This method is much like buildMLocValueMap: but focuses on a single
  // LexicalScope at a time. Pick out a set of blocks and variables that are
  // to have their value assignments solved, then run our dataflow algorithm
  // until a fixedpoint is reached.
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;
  SmallPtrSet<MachineBasicBlock *, 16> OnWorklist, OnPending;

  // The set of blocks we'll be examining.
  SmallPtrSet<const MachineBasicBlock *, 8> BlocksToExplore;

  // The order in which to examine them (RPO).
  SmallVector<MachineBasicBlock *, 8> BlockOrders;

  // RPO ordering function.
  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  LS.getMachineBasicBlocks(DILoc, BlocksToExplore);

  // A separate container to distinguish "blocks we're exploring" versus
  // "blocks that are potentially in scope. See comment at start of vlocJoin.
  SmallPtrSet<const MachineBasicBlock *, 8> InScopeBlocks = BlocksToExplore;

  // Old LiveDebugValues tracks variable locations that come out of blocks
  // not in scope, where DBG_VALUEs occur. This is something we could
  // legitimately ignore, but lets allow it for now.
  if (EmulateOldLDV)
    BlocksToExplore.insert(AssignBlocks.begin(), AssignBlocks.end());

  // We also need to propagate variable values through any artificial blocks
  // that immediately follow blocks in scope.
  DenseSet<const MachineBasicBlock *> ToAdd;

  // Helper lambda: For a given block in scope, perform a depth first search
  // of all the artificial successors, adding them to the ToAdd collection.
  auto AccumulateArtificialBlocks =
      [this, &ToAdd, &BlocksToExplore,
       &InScopeBlocks](const MachineBasicBlock *MBB) {
        // Depth-first-search state: each node is a block and which successor
        // we're currently exploring.
        SmallVector<std::pair<const MachineBasicBlock *,
                              MachineBasicBlock::const_succ_iterator>,
                    8>
            DFS;

        // Find any artificial successors not already tracked.
        for (auto *succ : MBB->successors()) {
          if (BlocksToExplore.count(succ) || InScopeBlocks.count(succ))
            continue;
          if (!ArtificialBlocks.count(succ))
            continue;
          DFS.push_back(std::make_pair(succ, succ->succ_begin()));
          ToAdd.insert(succ);
        }

        // Search all those blocks, depth first.
        while (!DFS.empty()) {
          const MachineBasicBlock *CurBB = DFS.back().first;
          MachineBasicBlock::const_succ_iterator &CurSucc = DFS.back().second;
          // Walk back if we've explored this blocks successors to the end.
          if (CurSucc == CurBB->succ_end()) {
            DFS.pop_back();
            continue;
          }

          // If the current successor is artificial and unexplored, descend into
          // it.
          if (!ToAdd.count(*CurSucc) && ArtificialBlocks.count(*CurSucc)) {
            DFS.push_back(std::make_pair(*CurSucc, (*CurSucc)->succ_begin()));
            ToAdd.insert(*CurSucc);
            continue;
          }

          ++CurSucc;
        }
      };

  // Search in-scope blocks and those containing a DBG_VALUE from this scope
  // for artificial successors.
  for (auto *MBB : BlocksToExplore)
    AccumulateArtificialBlocks(MBB);
  for (auto *MBB : InScopeBlocks)
    AccumulateArtificialBlocks(MBB);

  BlocksToExplore.insert(ToAdd.begin(), ToAdd.end());
  InScopeBlocks.insert(ToAdd.begin(), ToAdd.end());

  // Single block scope: not interesting! No propagation at all. Note that
  // this could probably go above ArtificialBlocks without damage, but
  // that then produces output differences from original-live-debug-values,
  // which propagates from a single block into many artificial ones.
  if (BlocksToExplore.size() == 1)
    return;

  // Convert a const set to a non-const set. LexicalScopes
  // getMachineBasicBlocks returns const MBB pointers, IDF wants mutable ones.
  // (Neither of them mutate anything).
  SmallPtrSet<MachineBasicBlock *, 8> MutBlocksToExplore;
  for (const auto *MBB : BlocksToExplore)
    MutBlocksToExplore.insert(const_cast<MachineBasicBlock *>(MBB));

  // Picks out relevants blocks RPO order and sort them.
  for (auto *MBB : BlocksToExplore)
    BlockOrders.push_back(const_cast<MachineBasicBlock *>(MBB));

  llvm::sort(BlockOrders, Cmp);
  unsigned NumBlocks = BlockOrders.size();

  // Allocate some vectors for storing the live ins and live outs. Large.
  SmallVector<DbgValue, 32> LiveIns, LiveOuts;
  LiveIns.reserve(NumBlocks);
  LiveOuts.reserve(NumBlocks);

  // Initialize all values to start as NoVals. This signifies "it's live
  // through, but we don't know what it is".
  DbgValueProperties EmptyProperties(EmptyExpr, false);
  for (unsigned int I = 0; I < NumBlocks; ++I) {
    DbgValue EmptyDbgValue(I, EmptyProperties, DbgValue::NoVal);
    LiveIns.push_back(EmptyDbgValue);
    LiveOuts.push_back(EmptyDbgValue);
  }

  // Produce by-MBB indexes of live-in/live-outs, to ease lookup within
  // vlocJoin.
  LiveIdxT LiveOutIdx, LiveInIdx;
  LiveOutIdx.reserve(NumBlocks);
  LiveInIdx.reserve(NumBlocks);
  for (unsigned I = 0; I < NumBlocks; ++I) {
    LiveOutIdx[BlockOrders[I]] = &LiveOuts[I];
    LiveInIdx[BlockOrders[I]] = &LiveIns[I];
  }

  // Loop over each variable and place PHIs for it, then propagate values
  // between blocks. This keeps the locality of working on one lexical scope at
  // at time, but avoids re-processing variable values because some other
  // variable has been assigned.
  for (auto &Var : VarsWeCareAbout) {
    // Re-initialize live-ins and live-outs, to clear the remains of previous
    // variables live-ins / live-outs.
    for (unsigned int I = 0; I < NumBlocks; ++I) {
      DbgValue EmptyDbgValue(I, EmptyProperties, DbgValue::NoVal);
      LiveIns[I] = EmptyDbgValue;
      LiveOuts[I] = EmptyDbgValue;
    }

    // Place PHIs for variable values, using the LLVM IDF calculator.
    // Collect the set of blocks where variables are def'd.
    SmallPtrSet<MachineBasicBlock *, 32> DefBlocks;
    for (const MachineBasicBlock *ExpMBB : BlocksToExplore) {
      auto &TransferFunc = AllTheVLocs[ExpMBB->getNumber()].Vars;
      if (TransferFunc.find(Var) != TransferFunc.end())
        DefBlocks.insert(const_cast<MachineBasicBlock *>(ExpMBB));
    }

    SmallVector<MachineBasicBlock *, 32> PHIBlocks;

    // Request the set of PHIs we should insert for this variable.
    BlockPHIPlacement(MutBlocksToExplore, DefBlocks, PHIBlocks);

    // Insert PHIs into the per-block live-in tables for this variable.
    for (MachineBasicBlock *PHIMBB : PHIBlocks) {
      unsigned BlockNo = PHIMBB->getNumber();
      DbgValue *LiveIn = LiveInIdx[PHIMBB];
      *LiveIn = DbgValue(BlockNo, EmptyProperties, DbgValue::VPHI);
    }

    for (auto *MBB : BlockOrders) {
      Worklist.push(BBToOrder[MBB]);
      OnWorklist.insert(MBB);
    }

    // Iterate over all the blocks we selected, propagating the variables value.
    // This loop does two things:
    //  * Eliminates un-necessary VPHIs in vlocJoin,
    //  * Evaluates the blocks transfer function (i.e. variable assignments) and
    //    stores the result to the blocks live-outs.
    // Always evaluate the transfer function on the first iteration, and when
    // the live-ins change thereafter.
    bool FirstTrip = true;
    while (!Worklist.empty() || !Pending.empty()) {
      while (!Worklist.empty()) {
        auto *MBB = OrderToBB[Worklist.top()];
        CurBB = MBB->getNumber();
        Worklist.pop();

        auto LiveInsIt = LiveInIdx.find(MBB);
        assert(LiveInsIt != LiveInIdx.end());
        DbgValue *LiveIn = LiveInsIt->second;

        // Join values from predecessors. Updates LiveInIdx, and writes output
        // into JoinedInLocs.
        bool InLocsChanged =
            vlocJoin(*MBB, LiveOutIdx, InScopeBlocks, BlocksToExplore, *LiveIn);

        SmallVector<const MachineBasicBlock *, 8> Preds;
        for (const auto *Pred : MBB->predecessors())
          Preds.push_back(Pred);

        // If this block's live-in value is a VPHI, try to pick a machine-value
        // for it. This makes the machine-value available and propagated
        // through all blocks by the time value propagation finishes. We can't
        // do this any earlier as it needs to read the block live-outs.
        if (LiveIn->Kind == DbgValue::VPHI && LiveIn->BlockNo == (int)CurBB) {
          // There's a small possibility that on a preceeding path, a VPHI is
          // eliminated and transitions from VPHI-with-location to
          // live-through-value. As a result, the selected location of any VPHI
          // might change, so we need to re-compute it on each iteration.
          Optional<ValueIDNum> ValueNum =
              pickVPHILoc(*MBB, Var, LiveOutIdx, MOutLocs, Preds);

          if (ValueNum) {
            InLocsChanged |= LiveIn->ID != *ValueNum;
            LiveIn->ID = *ValueNum;
          }
        }

        if (!InLocsChanged && !FirstTrip)
          continue;

        DbgValue *LiveOut = LiveOutIdx[MBB];
        bool OLChanged = false;

        // Do transfer function.
        auto &VTracker = AllTheVLocs[MBB->getNumber()];
        auto TransferIt = VTracker.Vars.find(Var);
        if (TransferIt != VTracker.Vars.end()) {
          // Erase on empty transfer (DBG_VALUE $noreg).
          if (TransferIt->second.Kind == DbgValue::Undef) {
            DbgValue NewVal(MBB->getNumber(), EmptyProperties, DbgValue::NoVal);
            if (*LiveOut != NewVal) {
              *LiveOut = NewVal;
              OLChanged = true;
            }
          } else {
            // Insert new variable value; or overwrite.
            if (*LiveOut != TransferIt->second) {
              *LiveOut = TransferIt->second;
              OLChanged = true;
            }
          }
        } else {
          // Just copy live-ins to live-outs, for anything not transferred.
          if (*LiveOut != *LiveIn) {
            *LiveOut = *LiveIn;
            OLChanged = true;
          }
        }

        // If no live-out value changed, there's no need to explore further.
        if (!OLChanged)
          continue;

        // We should visit all successors. Ensure we'll visit any non-backedge
        // successors during this dataflow iteration; book backedge successors
        // to be visited next time around.
        for (auto s : MBB->successors()) {
          // Ignore out of scope / not-to-be-explored successors.
          if (LiveInIdx.find(s) == LiveInIdx.end())
            continue;

          if (BBToOrder[s] > BBToOrder[MBB]) {
            if (OnWorklist.insert(s).second)
              Worklist.push(BBToOrder[s]);
          } else if (OnPending.insert(s).second && (FirstTrip || OLChanged)) {
            Pending.push(BBToOrder[s]);
          }
        }
      }
      Worklist.swap(Pending);
      std::swap(OnWorklist, OnPending);
      OnPending.clear();
      assert(Pending.empty());
      FirstTrip = false;
    }

    // Save live-ins to output vector. Ignore any that are still marked as being
    // VPHIs with no location -- those are variables that we know the value of,
    // but are not actually available in the register file.
    for (auto *MBB : BlockOrders) {
      DbgValue *BlockLiveIn = LiveInIdx[MBB];
      if (BlockLiveIn->Kind == DbgValue::NoVal)
        continue;
      if (BlockLiveIn->Kind == DbgValue::VPHI &&
          BlockLiveIn->ID == ValueIDNum::EmptyValue)
        continue;
      if (BlockLiveIn->Kind == DbgValue::VPHI)
        BlockLiveIn->Kind = DbgValue::Def;
      Output[MBB->getNumber()].push_back(std::make_pair(Var, *BlockLiveIn));
    }
  } // Per-variable loop.

  BlockOrders.clear();
  BlocksToExplore.clear();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void InstrRefBasedLDV::dump_mloc_transfer(
    const MLocTransferMap &mloc_transfer) const {
  for (auto &P : mloc_transfer) {
    std::string foo = MTracker->LocIdxToName(P.first);
    std::string bar = MTracker->IDAsString(P.second);
    dbgs() << "Loc " << foo << " --> " << bar << "\n";
  }
}
#endif

void InstrRefBasedLDV::emitLocations(
    MachineFunction &MF, LiveInsT SavedLiveIns, ValueIDNum **MOutLocs,
    ValueIDNum **MInLocs, DenseMap<DebugVariable, unsigned> &AllVarsNumbering,
    const TargetPassConfig &TPC) {
  TTracker = new TransferTracker(TII, MTracker, MF, *TRI, CalleeSavedRegs, TPC);
  unsigned NumLocs = MTracker->getNumLocs();

  // For each block, load in the machine value locations and variable value
  // live-ins, then step through each instruction in the block. New DBG_VALUEs
  // to be inserted will be created along the way.
  for (MachineBasicBlock &MBB : MF) {
    unsigned bbnum = MBB.getNumber();
    MTracker->reset();
    MTracker->loadFromArray(MInLocs[bbnum], bbnum);
    TTracker->loadInlocs(MBB, MInLocs[bbnum], SavedLiveIns[MBB.getNumber()],
                         NumLocs);

    CurBB = bbnum;
    CurInst = 1;
    for (auto &MI : MBB) {
      process(MI, MOutLocs, MInLocs);
      TTracker->checkInstForNewValues(CurInst, MI.getIterator());
      ++CurInst;
    }
  }

  // We have to insert DBG_VALUEs in a consistent order, otherwise they appeaer
  // in DWARF in different orders. Use the order that they appear when walking
  // through each block / each instruction, stored in AllVarsNumbering.
  auto OrderDbgValues = [&](const MachineInstr *A,
                            const MachineInstr *B) -> bool {
    DebugVariable VarA(A->getDebugVariable(), A->getDebugExpression(),
                       A->getDebugLoc()->getInlinedAt());
    DebugVariable VarB(B->getDebugVariable(), B->getDebugExpression(),
                       B->getDebugLoc()->getInlinedAt());
    return AllVarsNumbering.find(VarA)->second <
           AllVarsNumbering.find(VarB)->second;
  };

  // Go through all the transfers recorded in the TransferTracker -- this is
  // both the live-ins to a block, and any movements of values that happen
  // in the middle.
  for (auto &P : TTracker->Transfers) {
    // Sort them according to appearance order.
    llvm::sort(P.Insts, OrderDbgValues);
    // Insert either before or after the designated point...
    if (P.MBB) {
      MachineBasicBlock &MBB = *P.MBB;
      for (auto *MI : P.Insts) {
        MBB.insert(P.Pos, MI);
      }
    } else {
      // Terminators, like tail calls, can clobber things. Don't try and place
      // transfers after them.
      if (P.Pos->isTerminator())
        continue;

      MachineBasicBlock &MBB = *P.Pos->getParent();
      for (auto *MI : P.Insts) {
        MBB.insertAfterBundle(P.Pos, MI);
      }
    }
  }
}

void InstrRefBasedLDV::initialSetup(MachineFunction &MF) {
  // Build some useful data structures.

  LLVMContext &Context = MF.getFunction().getContext();
  EmptyExpr = DIExpression::get(Context, {});

  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  // Collect a set of all the artificial blocks.
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  // Compute mappings of block <=> RPO order.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (MachineBasicBlock *MBB : RPOT) {
    OrderToBB[RPONumber] = MBB;
    BBToOrder[MBB] = RPONumber;
    BBNumToRPO[MBB->getNumber()] = RPONumber;
    ++RPONumber;
  }

  // Order value substitutions by their "source" operand pair, for quick lookup.
  llvm::sort(MF.DebugValueSubstitutions);

#ifdef EXPENSIVE_CHECKS
  // As an expensive check, test whether there are any duplicate substitution
  // sources in the collection.
  if (MF.DebugValueSubstitutions.size() > 2) {
    for (auto It = MF.DebugValueSubstitutions.begin();
         It != std::prev(MF.DebugValueSubstitutions.end()); ++It) {
      assert(It->Src != std::next(It)->Src && "Duplicate variable location "
                                              "substitution seen");
    }
  }
#endif
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool InstrRefBasedLDV::ExtendRanges(MachineFunction &MF,
                                    MachineDominatorTree *DomTree,
                                    TargetPassConfig *TPC,
                                    unsigned InputBBLimit,
                                    unsigned InputDbgValLimit) {
  // No subprogram means this function contains no debuginfo.
  if (!MF.getFunction().getSubprogram())
    return false;

  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");
  this->TPC = TPC;

  this->DomTree = DomTree;
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  TFI->getCalleeSaves(MF, CalleeSavedRegs);
  MFI = &MF.getFrameInfo();
  LS.initialize(MF);

  MTracker =
      new MLocTracker(MF, *TII, *TRI, *MF.getSubtarget().getTargetLowering());
  VTracker = nullptr;
  TTracker = nullptr;

  SmallVector<MLocTransferMap, 32> MLocTransfer;
  SmallVector<VLocTracker, 8> vlocs;
  LiveInsT SavedLiveIns;

  int MaxNumBlocks = -1;
  for (auto &MBB : MF)
    MaxNumBlocks = std::max(MBB.getNumber(), MaxNumBlocks);
  assert(MaxNumBlocks >= 0);
  ++MaxNumBlocks;

  MLocTransfer.resize(MaxNumBlocks);
  vlocs.resize(MaxNumBlocks);
  SavedLiveIns.resize(MaxNumBlocks);

  initialSetup(MF);

  produceMLocTransferFunction(MF, MLocTransfer, MaxNumBlocks);

  // Allocate and initialize two array-of-arrays for the live-in and live-out
  // machine values. The outer dimension is the block number; while the inner
  // dimension is a LocIdx from MLocTracker.
  ValueIDNum **MOutLocs = new ValueIDNum *[MaxNumBlocks];
  ValueIDNum **MInLocs = new ValueIDNum *[MaxNumBlocks];
  unsigned NumLocs = MTracker->getNumLocs();
  for (int i = 0; i < MaxNumBlocks; ++i) {
    // These all auto-initialize to ValueIDNum::EmptyValue
    MOutLocs[i] = new ValueIDNum[NumLocs];
    MInLocs[i] = new ValueIDNum[NumLocs];
  }

  // Solve the machine value dataflow problem using the MLocTransfer function,
  // storing the computed live-ins / live-outs into the array-of-arrays. We use
  // both live-ins and live-outs for decision making in the variable value
  // dataflow problem.
  buildMLocValueMap(MF, MInLocs, MOutLocs, MLocTransfer);

  // Patch up debug phi numbers, turning unknown block-live-in values into
  // either live-through machine values, or PHIs.
  for (auto &DBG_PHI : DebugPHINumToValue) {
    // Identify unresolved block-live-ins.
    ValueIDNum &Num = DBG_PHI.ValueRead;
    if (!Num.isPHI())
      continue;

    unsigned BlockNo = Num.getBlock();
    LocIdx LocNo = Num.getLoc();
    Num = MInLocs[BlockNo][LocNo.asU64()];
  }
  // Later, we'll be looking up ranges of instruction numbers.
  llvm::sort(DebugPHINumToValue);

  // Walk back through each block / instruction, collecting DBG_VALUE
  // instructions and recording what machine value their operands refer to.
  for (auto &OrderPair : OrderToBB) {
    MachineBasicBlock &MBB = *OrderPair.second;
    CurBB = MBB.getNumber();
    VTracker = &vlocs[CurBB];
    VTracker->MBB = &MBB;
    MTracker->loadFromArray(MInLocs[CurBB], CurBB);
    CurInst = 1;
    for (auto &MI : MBB) {
      process(MI, MOutLocs, MInLocs);
      ++CurInst;
    }
    MTracker->reset();
  }

  // Number all variables in the order that they appear, to be used as a stable
  // insertion order later.
  DenseMap<DebugVariable, unsigned> AllVarsNumbering;

  // Map from one LexicalScope to all the variables in that scope.
  DenseMap<const LexicalScope *, SmallSet<DebugVariable, 4>> ScopeToVars;

  // Map from One lexical scope to all blocks in that scope.
  DenseMap<const LexicalScope *, SmallPtrSet<MachineBasicBlock *, 4>>
      ScopeToBlocks;

  // Store a DILocation that describes a scope.
  DenseMap<const LexicalScope *, const DILocation *> ScopeToDILocation;

  // To mirror old LiveDebugValues, enumerate variables in RPOT order. Otherwise
  // the order is unimportant, it just has to be stable.
  unsigned VarAssignCount = 0;
  for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
    auto *MBB = OrderToBB[I];
    auto *VTracker = &vlocs[MBB->getNumber()];
    // Collect each variable with a DBG_VALUE in this block.
    for (auto &idx : VTracker->Vars) {
      const auto &Var = idx.first;
      const DILocation *ScopeLoc = VTracker->Scopes[Var];
      assert(ScopeLoc != nullptr);
      auto *Scope = LS.findLexicalScope(ScopeLoc);

      // No insts in scope -> shouldn't have been recorded.
      assert(Scope != nullptr);

      AllVarsNumbering.insert(std::make_pair(Var, AllVarsNumbering.size()));
      ScopeToVars[Scope].insert(Var);
      ScopeToBlocks[Scope].insert(VTracker->MBB);
      ScopeToDILocation[Scope] = ScopeLoc;
      ++VarAssignCount;
    }
  }

  bool Changed = false;

  // If we have an extremely large number of variable assignments and blocks,
  // bail out at this point. We've burnt some time doing analysis already,
  // however we should cut our losses.
  if ((unsigned)MaxNumBlocks > InputBBLimit &&
      VarAssignCount > InputDbgValLimit) {
    LLVM_DEBUG(dbgs() << "Disabling InstrRefBasedLDV: " << MF.getName()
                      << " has " << MaxNumBlocks << " basic blocks and "
                      << VarAssignCount
                      << " variable assignments, exceeding limits.\n");
  } else {
    // Compute the extended ranges, iterating over scopes. There might be
    // something to be said for ordering them by size/locality, but that's for
    // the future. For each scope, solve the variable value problem, producing
    // a map of variables to values in SavedLiveIns.
    for (auto &P : ScopeToVars) {
      buildVLocValueMap(ScopeToDILocation[P.first], P.second,
                   ScopeToBlocks[P.first], SavedLiveIns, MOutLocs, MInLocs,
                   vlocs);
    }

    // Using the computed value locations and variable values for each block,
    // create the DBG_VALUE instructions representing the extended variable
    // locations.
    emitLocations(MF, SavedLiveIns, MOutLocs, MInLocs, AllVarsNumbering, *TPC);

    // Did we actually make any changes? If we created any DBG_VALUEs, then yes.
    Changed = TTracker->Transfers.size() != 0;
  }

  // Common clean-up of memory.
  for (int Idx = 0; Idx < MaxNumBlocks; ++Idx) {
    delete[] MOutLocs[Idx];
    delete[] MInLocs[Idx];
  }
  delete[] MOutLocs;
  delete[] MInLocs;

  delete MTracker;
  delete TTracker;
  MTracker = nullptr;
  VTracker = nullptr;
  TTracker = nullptr;

  ArtificialBlocks.clear();
  OrderToBB.clear();
  BBToOrder.clear();
  BBNumToRPO.clear();
  DebugInstrNumToInstr.clear();
  DebugPHINumToValue.clear();

  return Changed;
}

LDVImpl *llvm::makeInstrRefBasedLiveDebugValues() {
  return new InstrRefBasedLDV();
}

namespace {
class LDVSSABlock;
class LDVSSAUpdater;

// Pick a type to identify incoming block values as we construct SSA. We
// can't use anything more robust than an integer unfortunately, as SSAUpdater
// expects to zero-initialize the type.
typedef uint64_t BlockValueNum;

/// Represents an SSA PHI node for the SSA updater class. Contains the block
/// this PHI is in, the value number it would have, and the expected incoming
/// values from parent blocks.
class LDVSSAPhi {
public:
  SmallVector<std::pair<LDVSSABlock *, BlockValueNum>, 4> IncomingValues;
  LDVSSABlock *ParentBlock;
  BlockValueNum PHIValNum;
  LDVSSAPhi(BlockValueNum PHIValNum, LDVSSABlock *ParentBlock)
      : ParentBlock(ParentBlock), PHIValNum(PHIValNum) {}

  LDVSSABlock *getParent() { return ParentBlock; }
};

/// Thin wrapper around a block predecessor iterator. Only difference from a
/// normal block iterator is that it dereferences to an LDVSSABlock.
class LDVSSABlockIterator {
public:
  MachineBasicBlock::pred_iterator PredIt;
  LDVSSAUpdater &Updater;

  LDVSSABlockIterator(MachineBasicBlock::pred_iterator PredIt,
                      LDVSSAUpdater &Updater)
      : PredIt(PredIt), Updater(Updater) {}

  bool operator!=(const LDVSSABlockIterator &OtherIt) const {
    return OtherIt.PredIt != PredIt;
  }

  LDVSSABlockIterator &operator++() {
    ++PredIt;
    return *this;
  }

  LDVSSABlock *operator*();
};

/// Thin wrapper around a block for SSA Updater interface. Necessary because
/// we need to track the PHI value(s) that we may have observed as necessary
/// in this block.
class LDVSSABlock {
public:
  MachineBasicBlock &BB;
  LDVSSAUpdater &Updater;
  using PHIListT = SmallVector<LDVSSAPhi, 1>;
  /// List of PHIs in this block. There should only ever be one.
  PHIListT PHIList;

  LDVSSABlock(MachineBasicBlock &BB, LDVSSAUpdater &Updater)
      : BB(BB), Updater(Updater) {}

  LDVSSABlockIterator succ_begin() {
    return LDVSSABlockIterator(BB.succ_begin(), Updater);
  }

  LDVSSABlockIterator succ_end() {
    return LDVSSABlockIterator(BB.succ_end(), Updater);
  }

  /// SSAUpdater has requested a PHI: create that within this block record.
  LDVSSAPhi *newPHI(BlockValueNum Value) {
    PHIList.emplace_back(Value, this);
    return &PHIList.back();
  }

  /// SSAUpdater wishes to know what PHIs already exist in this block.
  PHIListT &phis() { return PHIList; }
};

/// Utility class for the SSAUpdater interface: tracks blocks, PHIs and values
/// while SSAUpdater is exploring the CFG. It's passed as a handle / baton to
// SSAUpdaterTraits<LDVSSAUpdater>.
class LDVSSAUpdater {
public:
  /// Map of value numbers to PHI records.
  DenseMap<BlockValueNum, LDVSSAPhi *> PHIs;
  /// Map of which blocks generate Undef values -- blocks that are not
  /// dominated by any Def.
  DenseMap<MachineBasicBlock *, BlockValueNum> UndefMap;
  /// Map of machine blocks to our own records of them.
  DenseMap<MachineBasicBlock *, LDVSSABlock *> BlockMap;
  /// Machine location where any PHI must occur.
  LocIdx Loc;
  /// Table of live-in machine value numbers for blocks / locations.
  ValueIDNum **MLiveIns;

  LDVSSAUpdater(LocIdx L, ValueIDNum **MLiveIns) : Loc(L), MLiveIns(MLiveIns) {}

  void reset() {
    for (auto &Block : BlockMap)
      delete Block.second;

    PHIs.clear();
    UndefMap.clear();
    BlockMap.clear();
  }

  ~LDVSSAUpdater() { reset(); }

  /// For a given MBB, create a wrapper block for it. Stores it in the
  /// LDVSSAUpdater block map.
  LDVSSABlock *getSSALDVBlock(MachineBasicBlock *BB) {
    auto it = BlockMap.find(BB);
    if (it == BlockMap.end()) {
      BlockMap[BB] = new LDVSSABlock(*BB, *this);
      it = BlockMap.find(BB);
    }
    return it->second;
  }

  /// Find the live-in value number for the given block. Looks up the value at
  /// the PHI location on entry.
  BlockValueNum getValue(LDVSSABlock *LDVBB) {
    return MLiveIns[LDVBB->BB.getNumber()][Loc.asU64()].asU64();
  }
};

LDVSSABlock *LDVSSABlockIterator::operator*() {
  return Updater.getSSALDVBlock(*PredIt);
}

#ifndef NDEBUG

raw_ostream &operator<<(raw_ostream &out, const LDVSSAPhi &PHI) {
  out << "SSALDVPHI " << PHI.PHIValNum;
  return out;
}

#endif

} // namespace

namespace llvm {

/// Template specialization to give SSAUpdater access to CFG and value
/// information. SSAUpdater calls methods in these traits, passing in the
/// LDVSSAUpdater object, to learn about blocks and the values they define.
/// It also provides methods to create PHI nodes and track them.
template <> class SSAUpdaterTraits<LDVSSAUpdater> {
public:
  using BlkT = LDVSSABlock;
  using ValT = BlockValueNum;
  using PhiT = LDVSSAPhi;
  using BlkSucc_iterator = LDVSSABlockIterator;

  // Methods to access block successors -- dereferencing to our wrapper class.
  static BlkSucc_iterator BlkSucc_begin(BlkT *BB) { return BB->succ_begin(); }
  static BlkSucc_iterator BlkSucc_end(BlkT *BB) { return BB->succ_end(); }

  /// Iterator for PHI operands.
  class PHI_iterator {
  private:
    LDVSSAPhi *PHI;
    unsigned Idx;

  public:
    explicit PHI_iterator(LDVSSAPhi *P) // begin iterator
        : PHI(P), Idx(0) {}
    PHI_iterator(LDVSSAPhi *P, bool) // end iterator
        : PHI(P), Idx(PHI->IncomingValues.size()) {}

    PHI_iterator &operator++() {
      Idx++;
      return *this;
    }
    bool operator==(const PHI_iterator &X) const { return Idx == X.Idx; }
    bool operator!=(const PHI_iterator &X) const { return !operator==(X); }

    BlockValueNum getIncomingValue() { return PHI->IncomingValues[Idx].second; }

    LDVSSABlock *getIncomingBlock() { return PHI->IncomingValues[Idx].first; }
  };

  static inline PHI_iterator PHI_begin(PhiT *PHI) { return PHI_iterator(PHI); }

  static inline PHI_iterator PHI_end(PhiT *PHI) {
    return PHI_iterator(PHI, true);
  }

  /// FindPredecessorBlocks - Put the predecessors of BB into the Preds
  /// vector.
  static void FindPredecessorBlocks(LDVSSABlock *BB,
                                    SmallVectorImpl<LDVSSABlock *> *Preds) {
    for (MachineBasicBlock::pred_iterator PI = BB->BB.pred_begin(),
                                          E = BB->BB.pred_end();
         PI != E; ++PI)
      Preds->push_back(BB->Updater.getSSALDVBlock(*PI));
  }

  /// GetUndefVal - Normally creates an IMPLICIT_DEF instruction with a new
  /// register. For LiveDebugValues, represents a block identified as not having
  /// any DBG_PHI predecessors.
  static BlockValueNum GetUndefVal(LDVSSABlock *BB, LDVSSAUpdater *Updater) {
    // Create a value number for this block -- it needs to be unique and in the
    // "undef" collection, so that we know it's not real. Use a number
    // representing a PHI into this block.
    BlockValueNum Num = ValueIDNum(BB->BB.getNumber(), 0, Updater->Loc).asU64();
    Updater->UndefMap[&BB->BB] = Num;
    return Num;
  }

  /// CreateEmptyPHI - Create a (representation of a) PHI in the given block.
  /// SSAUpdater will populate it with information about incoming values. The
  /// value number of this PHI is whatever the  machine value number problem
  /// solution determined it to be. This includes non-phi values if SSAUpdater
  /// tries to create a PHI where the incoming values are identical.
  static BlockValueNum CreateEmptyPHI(LDVSSABlock *BB, unsigned NumPreds,
                                   LDVSSAUpdater *Updater) {
    BlockValueNum PHIValNum = Updater->getValue(BB);
    LDVSSAPhi *PHI = BB->newPHI(PHIValNum);
    Updater->PHIs[PHIValNum] = PHI;
    return PHIValNum;
  }

  /// AddPHIOperand - Add the specified value as an operand of the PHI for
  /// the specified predecessor block.
  static void AddPHIOperand(LDVSSAPhi *PHI, BlockValueNum Val, LDVSSABlock *Pred) {
    PHI->IncomingValues.push_back(std::make_pair(Pred, Val));
  }

  /// ValueIsPHI - Check if the instruction that defines the specified value
  /// is a PHI instruction.
  static LDVSSAPhi *ValueIsPHI(BlockValueNum Val, LDVSSAUpdater *Updater) {
    auto PHIIt = Updater->PHIs.find(Val);
    if (PHIIt == Updater->PHIs.end())
      return nullptr;
    return PHIIt->second;
  }

  /// ValueIsNewPHI - Like ValueIsPHI but also check if the PHI has no source
  /// operands, i.e., it was just added.
  static LDVSSAPhi *ValueIsNewPHI(BlockValueNum Val, LDVSSAUpdater *Updater) {
    LDVSSAPhi *PHI = ValueIsPHI(Val, Updater);
    if (PHI && PHI->IncomingValues.size() == 0)
      return PHI;
    return nullptr;
  }

  /// GetPHIValue - For the specified PHI instruction, return the value
  /// that it defines.
  static BlockValueNum GetPHIValue(LDVSSAPhi *PHI) { return PHI->PHIValNum; }
};

} // end namespace llvm

Optional<ValueIDNum> InstrRefBasedLDV::resolveDbgPHIs(MachineFunction &MF,
                                                      ValueIDNum **MLiveOuts,
                                                      ValueIDNum **MLiveIns,
                                                      MachineInstr &Here,
                                                      uint64_t InstrNum) {
  // Pick out records of DBG_PHI instructions that have been observed. If there
  // are none, then we cannot compute a value number.
  auto RangePair = std::equal_range(DebugPHINumToValue.begin(),
                                    DebugPHINumToValue.end(), InstrNum);
  auto LowerIt = RangePair.first;
  auto UpperIt = RangePair.second;

  // No DBG_PHI means there can be no location.
  if (LowerIt == UpperIt)
    return None;

  // If there's only one DBG_PHI, then that is our value number.
  if (std::distance(LowerIt, UpperIt) == 1)
    return LowerIt->ValueRead;

  auto DBGPHIRange = make_range(LowerIt, UpperIt);

  // Pick out the location (physreg, slot) where any PHIs must occur. It's
  // technically possible for us to merge values in different registers in each
  // block, but highly unlikely that LLVM will generate such code after register
  // allocation.
  LocIdx Loc = LowerIt->ReadLoc;

  // We have several DBG_PHIs, and a use position (the Here inst). All each
  // DBG_PHI does is identify a value at a program position. We can treat each
  // DBG_PHI like it's a Def of a value, and the use position is a Use of a
  // value, just like SSA. We use the bulk-standard LLVM SSA updater class to
  // determine which Def is used at the Use, and any PHIs that happen along
  // the way.
  // Adapted LLVM SSA Updater:
  LDVSSAUpdater Updater(Loc, MLiveIns);
  // Map of which Def or PHI is the current value in each block.
  DenseMap<LDVSSABlock *, BlockValueNum> AvailableValues;
  // Set of PHIs that we have created along the way.
  SmallVector<LDVSSAPhi *, 8> CreatedPHIs;

  // Each existing DBG_PHI is a Def'd value under this model. Record these Defs
  // for the SSAUpdater.
  for (const auto &DBG_PHI : DBGPHIRange) {
    LDVSSABlock *Block = Updater.getSSALDVBlock(DBG_PHI.MBB);
    const ValueIDNum &Num = DBG_PHI.ValueRead;
    AvailableValues.insert(std::make_pair(Block, Num.asU64()));
  }

  LDVSSABlock *HereBlock = Updater.getSSALDVBlock(Here.getParent());
  const auto &AvailIt = AvailableValues.find(HereBlock);
  if (AvailIt != AvailableValues.end()) {
    // Actually, we already know what the value is -- the Use is in the same
    // block as the Def.
    return ValueIDNum::fromU64(AvailIt->second);
  }

  // Otherwise, we must use the SSA Updater. It will identify the value number
  // that we are to use, and the PHIs that must happen along the way.
  SSAUpdaterImpl<LDVSSAUpdater> Impl(&Updater, &AvailableValues, &CreatedPHIs);
  BlockValueNum ResultInt = Impl.GetValue(Updater.getSSALDVBlock(Here.getParent()));
  ValueIDNum Result = ValueIDNum::fromU64(ResultInt);

  // We have the number for a PHI, or possibly live-through value, to be used
  // at this Use. There are a number of things we have to check about it though:
  //  * Does any PHI use an 'Undef' (like an IMPLICIT_DEF) value? If so, this
  //    Use was not completely dominated by DBG_PHIs and we should abort.
  //  * Are the Defs or PHIs clobbered in a block? SSAUpdater isn't aware that
  //    we've left SSA form. Validate that the inputs to each PHI are the
  //    expected values.
  //  * Is a PHI we've created actually a merging of values, or are all the
  //    predecessor values the same, leading to a non-PHI machine value number?
  //    (SSAUpdater doesn't know that either). Remap validated PHIs into the
  //    the ValidatedValues collection below to sort this out.
  DenseMap<LDVSSABlock *, ValueIDNum> ValidatedValues;

  // Define all the input DBG_PHI values in ValidatedValues.
  for (const auto &DBG_PHI : DBGPHIRange) {
    LDVSSABlock *Block = Updater.getSSALDVBlock(DBG_PHI.MBB);
    const ValueIDNum &Num = DBG_PHI.ValueRead;
    ValidatedValues.insert(std::make_pair(Block, Num));
  }

  // Sort PHIs to validate into RPO-order.
  SmallVector<LDVSSAPhi *, 8> SortedPHIs;
  for (auto &PHI : CreatedPHIs)
    SortedPHIs.push_back(PHI);

  std::sort(
      SortedPHIs.begin(), SortedPHIs.end(), [&](LDVSSAPhi *A, LDVSSAPhi *B) {
        return BBToOrder[&A->getParent()->BB] < BBToOrder[&B->getParent()->BB];
      });

  for (auto &PHI : SortedPHIs) {
    ValueIDNum ThisBlockValueNum =
        MLiveIns[PHI->ParentBlock->BB.getNumber()][Loc.asU64()];

    // Are all these things actually defined?
    for (auto &PHIIt : PHI->IncomingValues) {
      // Any undef input means DBG_PHIs didn't dominate the use point.
      if (Updater.UndefMap.find(&PHIIt.first->BB) != Updater.UndefMap.end())
        return None;

      ValueIDNum ValueToCheck;
      ValueIDNum *BlockLiveOuts = MLiveOuts[PHIIt.first->BB.getNumber()];

      auto VVal = ValidatedValues.find(PHIIt.first);
      if (VVal == ValidatedValues.end()) {
        // We cross a loop, and this is a backedge. LLVMs tail duplication
        // happens so late that DBG_PHI instructions should not be able to
        // migrate into loops -- meaning we can only be live-through this
        // loop.
        ValueToCheck = ThisBlockValueNum;
      } else {
        // Does the block have as a live-out, in the location we're examining,
        // the value that we expect? If not, it's been moved or clobbered.
        ValueToCheck = VVal->second;
      }

      if (BlockLiveOuts[Loc.asU64()] != ValueToCheck)
        return None;
    }

    // Record this value as validated.
    ValidatedValues.insert({PHI->ParentBlock, ThisBlockValueNum});
  }

  // All the PHIs are valid: we can return what the SSAUpdater said our value
  // number was.
  return Result;
}
