//===- llvm/CodeGen/AsmPrinter/DbgEntityHistoryCalculator.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DbgEntityHistoryCalculator.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <map>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "dwarfdebug"

namespace {
using EntryIndex = DbgValueHistoryMap::EntryIndex;
}

// If @MI is a DBG_VALUE with debug value described by a
// defined register, returns the number of this register.
// In the other case, returns 0.
static unsigned isDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue());
  assert(MI.getNumOperands() == 4);
  // If location of variable is described using a register (directly or
  // indirectly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : 0;
}

bool DbgValueHistoryMap::startDbgValue(InlinedEntity Var,
                                       const MachineInstr &MI,
                                       EntryIndex &NewIndex) {
  // Instruction range should start with a DBG_VALUE instruction for the
  // variable.
  assert(MI.isDebugValue() && "not a DBG_VALUE");
  auto &Entries = VarEntries[Var];
  if (!Entries.empty() && Entries.back().isDbgValue() &&
      !Entries.back().isClosed() &&
      Entries.back().getInstr()->isIdenticalTo(MI)) {
    LLVM_DEBUG(dbgs() << "Coalescing identical DBG_VALUE entries:\n"
                      << "\t" << Entries.back().getInstr() << "\t" << MI
                      << "\n");
    return false;
  }
  Entries.emplace_back(&MI, Entry::DbgValue);
  NewIndex = Entries.size() - 1;
  return true;
}

EntryIndex DbgValueHistoryMap::startClobber(InlinedEntity Var,
                                            const MachineInstr &MI) {
  auto &Entries = VarEntries[Var];
  // If an instruction clobbers multiple registers that the variable is
  // described by, then we may have already created a clobbering instruction.
  if (Entries.back().isClobber() && Entries.back().getInstr() == &MI)
    return Entries.size() - 1;
  Entries.emplace_back(&MI, Entry::Clobber);
  return Entries.size() - 1;
}

void DbgValueHistoryMap::Entry::endEntry(EntryIndex Index) {
  // For now, instruction ranges are not allowed to cross basic block
  // boundaries.
  assert(isDbgValue() && "Setting end index for non-debug value");
  assert(!isClosed() && "End index has already been set");
  EndIndex = Index;
}

void DbgLabelInstrMap::addInstr(InlinedEntity Label, const MachineInstr &MI) {
  assert(MI.isDebugLabel() && "not a DBG_LABEL");
  LabelInstr[Label] = &MI;
}

namespace {

// Maps physreg numbers to the variables they describe.
using InlinedEntity = DbgValueHistoryMap::InlinedEntity;
using RegDescribedVarsMap = std::map<unsigned, SmallVector<InlinedEntity, 1>>;

// Keeps track of the debug value entries that are currently live for each
// inlined entity. As the history map entries are stored in a SmallVector, they
// may be moved at insertion of new entries, so store indices rather than
// pointers.
using DbgValueEntriesMap = std::map<InlinedEntity, SmallSet<EntryIndex, 1>>;

} // end anonymous namespace

// Claim that @Var is not described by @RegNo anymore.
static void dropRegDescribedVar(RegDescribedVarsMap &RegVars, unsigned RegNo,
                                InlinedEntity Var) {
  const auto &I = RegVars.find(RegNo);
  assert(RegNo != 0U && I != RegVars.end());
  auto &VarSet = I->second;
  const auto &VarPos = llvm::find(VarSet, Var);
  assert(VarPos != VarSet.end());
  VarSet.erase(VarPos);
  // Don't keep empty sets in a map to keep it as small as possible.
  if (VarSet.empty())
    RegVars.erase(I);
}

// Claim that @Var is now described by @RegNo.
static void addRegDescribedVar(RegDescribedVarsMap &RegVars, unsigned RegNo,
                               InlinedEntity Var) {
  assert(RegNo != 0U);
  auto &VarSet = RegVars[RegNo];
  assert(!is_contained(VarSet, Var));
  VarSet.push_back(Var);
}

/// Create a clobbering entry and end all open debug value entries
/// for \p Var that are described by \p RegNo using that entry.
static void clobberRegEntries(InlinedEntity Var, unsigned RegNo,
                              const MachineInstr &ClobberingInstr,
                              DbgValueEntriesMap &LiveEntries,
                              DbgValueHistoryMap &HistMap) {
  EntryIndex ClobberIndex = HistMap.startClobber(Var, ClobberingInstr);

  // Close all entries whose values are described by the register.
  SmallVector<EntryIndex, 4> IndicesToErase;
  for (auto Index : LiveEntries[Var]) {
    auto &Entry = HistMap.getEntry(Var, Index);
    assert(Entry.isDbgValue() && "Not a DBG_VALUE in LiveEntries");
    if (isDescribedByReg(*Entry.getInstr()) == RegNo) {
      IndicesToErase.push_back(Index);
      Entry.endEntry(ClobberIndex);
    }
  }

  // Drop all entries that have ended.
  for (auto Index : IndicesToErase)
    LiveEntries[Var].erase(Index);
}

/// Add a new debug value for \p Var. Closes all overlapping debug values.
static void handleNewDebugValue(InlinedEntity Var, const MachineInstr &DV,
                                RegDescribedVarsMap &RegVars,
                                DbgValueEntriesMap &LiveEntries,
                                DbgValueHistoryMap &HistMap) {
  EntryIndex NewIndex;
  if (HistMap.startDbgValue(Var, DV, NewIndex)) {
    SmallDenseMap<unsigned, bool, 4> TrackedRegs;

    // If we have created a new debug value entry, close all preceding
    // live entries that overlap.
    SmallVector<EntryIndex, 4> IndicesToErase;
    const DIExpression *DIExpr = DV.getDebugExpression();
    for (auto Index : LiveEntries[Var]) {
      auto &Entry = HistMap.getEntry(Var, Index);
      assert(Entry.isDbgValue() && "Not a DBG_VALUE in LiveEntries");
      const MachineInstr &DV = *Entry.getInstr();
      bool Overlaps = DIExpr->fragmentsOverlap(DV.getDebugExpression());
      if (Overlaps) {
        IndicesToErase.push_back(Index);
        Entry.endEntry(NewIndex);
      }
      if (unsigned Reg = isDescribedByReg(DV))
        TrackedRegs[Reg] |= !Overlaps;
    }

    // If the new debug value is described by a register, add tracking of
    // that register if it is not already tracked.
    if (unsigned NewReg = isDescribedByReg(DV)) {
      if (!TrackedRegs.count(NewReg))
        addRegDescribedVar(RegVars, NewReg, Var);
      LiveEntries[Var].insert(NewIndex);
      TrackedRegs[NewReg] = true;
    }

    // Drop tracking of registers that are no longer used.
    for (auto I : TrackedRegs)
      if (!I.second)
        dropRegDescribedVar(RegVars, I.first, Var);

    // Drop all entries that have ended, and mark the new entry as live.
    for (auto Index : IndicesToErase)
      LiveEntries[Var].erase(Index);
    LiveEntries[Var].insert(NewIndex);
  }
}

// Terminate the location range for variables described by register at
// @I by inserting @ClobberingInstr to their history.
static void clobberRegisterUses(RegDescribedVarsMap &RegVars,
                                RegDescribedVarsMap::iterator I,
                                DbgValueHistoryMap &HistMap,
                                DbgValueEntriesMap &LiveEntries,
                                const MachineInstr &ClobberingInstr) {
  // Iterate over all variables described by this register and add this
  // instruction to their history, clobbering it.
  for (const auto &Var : I->second)
    clobberRegEntries(Var, I->first, ClobberingInstr, LiveEntries, HistMap);
  RegVars.erase(I);
}

// Terminate the location range for variables described by register
// @RegNo by inserting @ClobberingInstr to their history.
static void clobberRegisterUses(RegDescribedVarsMap &RegVars, unsigned RegNo,
                                DbgValueHistoryMap &HistMap,
                                DbgValueEntriesMap &LiveEntries,
                                const MachineInstr &ClobberingInstr) {
  const auto &I = RegVars.find(RegNo);
  if (I == RegVars.end())
    return;
  clobberRegisterUses(RegVars, I, HistMap, LiveEntries, ClobberingInstr);
}

// Returns the first instruction in @MBB which corresponds to
// the function epilogue, or nullptr if @MBB doesn't contain an epilogue.
static const MachineInstr *getFirstEpilogueInst(const MachineBasicBlock &MBB) {
  auto LastMI = MBB.getLastNonDebugInstr();
  if (LastMI == MBB.end() || !LastMI->isReturn())
    return nullptr;
  // Assume that epilogue starts with instruction having the same debug location
  // as the return instruction.
  DebugLoc LastLoc = LastMI->getDebugLoc();
  auto Res = LastMI;
  for (MachineBasicBlock::const_reverse_iterator I = LastMI.getReverse(),
                                                 E = MBB.rend();
       I != E; ++I) {
    if (I->getDebugLoc() != LastLoc)
      return &*Res;
    Res = &*I;
  }
  // If all instructions have the same debug location, assume whole MBB is
  // an epilogue.
  return &*MBB.begin();
}

// Collect registers that are modified in the function body (their
// contents is changed outside of the prologue and epilogue).
static void collectChangingRegs(const MachineFunction *MF,
                                const TargetRegisterInfo *TRI,
                                BitVector &Regs) {
  for (const auto &MBB : *MF) {
    auto FirstEpilogueInst = getFirstEpilogueInst(MBB);

    for (const auto &MI : MBB) {
      // Avoid looking at prologue or epilogue instructions.
      if (&MI == FirstEpilogueInst)
        break;
      if (MI.getFlag(MachineInstr::FrameSetup))
        continue;

      // Look for register defs and register masks. Register masks are
      // typically on calls and they clobber everything not in the mask.
      for (const MachineOperand &MO : MI.operands()) {
        // Skip virtual registers since they are handled by the parent.
        if (MO.isReg() && MO.isDef() && MO.getReg() &&
            !TRI->isVirtualRegister(MO.getReg())) {
          for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid();
               ++AI)
            Regs.set(*AI);
        } else if (MO.isRegMask()) {
          Regs.setBitsNotInMask(MO.getRegMask());
        }
      }
    }
  }
}

void llvm::calculateDbgEntityHistory(const MachineFunction *MF,
                                     const TargetRegisterInfo *TRI,
                                     DbgValueHistoryMap &DbgValues,
                                     DbgLabelInstrMap &DbgLabels) {
  BitVector ChangingRegs(TRI->getNumRegs());
  collectChangingRegs(MF, TRI, ChangingRegs);

  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  RegDescribedVarsMap RegVars;
  DbgValueEntriesMap LiveEntries;
  for (const auto &MBB : *MF) {
    for (const auto &MI : MBB) {
      if (!MI.isDebugInstr()) {
        // Not a DBG_VALUE instruction. It may clobber registers which describe
        // some variables.
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.isDef() && MO.getReg()) {
            // Ignore call instructions that claim to clobber SP. The AArch64
            // backend does this for aggregate function arguments.
            if (MI.isCall() && MO.getReg() == SP)
              continue;
            // If this is a virtual register, only clobber it since it doesn't
            // have aliases.
            if (TRI->isVirtualRegister(MO.getReg()))
              clobberRegisterUses(RegVars, MO.getReg(), DbgValues, LiveEntries,
                                  MI);
            // If this is a register def operand, it may end a debug value
            // range.
            else {
              for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid();
                   ++AI)
                if (ChangingRegs.test(*AI))
                  clobberRegisterUses(RegVars, *AI, DbgValues, LiveEntries, MI);
            }
          } else if (MO.isRegMask()) {
            // If this is a register mask operand, clobber all debug values in
            // non-CSRs.
            for (unsigned I : ChangingRegs.set_bits()) {
              // Don't consider SP to be clobbered by register masks.
              if (unsigned(I) != SP && TRI->isPhysicalRegister(I) &&
                  MO.clobbersPhysReg(I)) {
                clobberRegisterUses(RegVars, I, DbgValues, LiveEntries, MI);
              }
            }
          }
        }
        continue;
      }

      if (MI.isDebugValue()) {
        assert(MI.getNumOperands() > 1 && "Invalid DBG_VALUE instruction!");
        // Use the base variable (without any DW_OP_piece expressions)
        // as index into History. The full variables including the
        // piece expressions are attached to the MI.
        const DILocalVariable *RawVar = MI.getDebugVariable();
        assert(RawVar->isValidLocationForIntrinsic(MI.getDebugLoc()) &&
               "Expected inlined-at fields to agree");
        InlinedEntity Var(RawVar, MI.getDebugLoc()->getInlinedAt());

        handleNewDebugValue(Var, MI, RegVars, LiveEntries, DbgValues);
      } else if (MI.isDebugLabel()) {
        assert(MI.getNumOperands() == 1 && "Invalid DBG_LABEL instruction!");
        const DILabel *RawLabel = MI.getDebugLabel();
        assert(RawLabel->isValidLocationForIntrinsic(MI.getDebugLoc()) &&
            "Expected inlined-at fields to agree");
        // When collecting debug information for labels, there is no MCSymbol
        // generated for it. So, we keep MachineInstr in DbgLabels in order
        // to query MCSymbol afterward.
        InlinedEntity L(RawLabel, MI.getDebugLoc()->getInlinedAt());
        DbgLabels.addInstr(L, MI);
      }
    }

    // Make sure locations for all variables are valid only until the end of
    // the basic block (unless it's the last basic block, in which case let
    // their liveness run off to the end of the function).
    if (!MBB.empty() && &MBB != &MF->back()) {
      // Iterate over all variables that have open debug values.
      SmallSet<unsigned, 8> RegsToClobber;
      for (auto &Pair : LiveEntries) {
        // Iterate over history entries for all open fragments.
        SmallVector<EntryIndex, 8> IdxesToRemove;
        for (EntryIndex Idx : Pair.second) {
          DbgValueHistoryMap::Entry &Ent = DbgValues.getEntry(Pair.first, Idx);
          assert(Ent.isDbgValue() && !Ent.isClosed());
          const MachineInstr *DbgValue = Ent.getInstr();

          // If this is a register or indirect DBG_VALUE, apply some futher
          // tests to see if we should clobber it. Perform the clobbering
          // later though, to keep LiveEntries iteration stable.
          if (DbgValue->getOperand(0).isReg()) {
            unsigned RegNo = DbgValue->getOperand(0).getReg();
            if (TRI->isVirtualRegister(RegNo) || ChangingRegs.test(RegNo))
              RegsToClobber.insert(RegNo);
          } else {
            // This is a constant, terminate it at end of the block. Store
            // eliminated EntryIdx and delete later, for iteration stability.
            EntryIndex ClobIdx = DbgValues.startClobber(Pair.first, MBB.back());
            DbgValues.getEntry(Pair.first, Idx).endEntry(ClobIdx);
            IdxesToRemove.push_back(Idx);
          }
        }

        for (EntryIndex Idx : IdxesToRemove)
          Pair.second.erase(Idx);
      }

      // Implement clobbering of registers at the end of BB.
      for (unsigned Reg : RegsToClobber)
        clobberRegisterUses(RegVars, Reg, DbgValues, LiveEntries, MBB.back());
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void DbgValueHistoryMap::dump() const {
  dbgs() << "DbgValueHistoryMap:\n";
  for (const auto &VarRangePair : *this) {
    const InlinedEntity &Var = VarRangePair.first;
    const Entries &Entries = VarRangePair.second;

    const DILocalVariable *LocalVar = cast<DILocalVariable>(Var.first);
    const DILocation *Location = Var.second;

    dbgs() << " - " << LocalVar->getName() << " at ";

    if (Location)
      dbgs() << Location->getFilename() << ":" << Location->getLine() << ":"
             << Location->getColumn();
    else
      dbgs() << "<unknown location>";

    dbgs() << " --\n";

    for (const auto &E : enumerate(Entries)) {
      const auto &Entry = E.value();
      dbgs() << "  Entry[" << E.index() << "]: ";
      if (Entry.isDbgValue())
        dbgs() << "Debug value\n";
      else
        dbgs() << "Clobber\n";
      dbgs() << "   Instr: " << *Entry.getInstr();
      if (Entry.isDbgValue()) {
        if (Entry.getEndIndex() == NoEntry)
          dbgs() << "   - Valid until end of function\n";
        else
          dbgs() << "   - Closed by Entry[" << Entry.getEndIndex() << "]\n";
      }
      dbgs() << "\n";
    }
  }
}
#endif
