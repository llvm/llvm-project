//===- llvm/CodeGen/DbgEntityHistoryCalculator.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DBGVALUEHISTORYCALCULATOR_H
#define LLVM_CODEGEN_DBGVALUEHISTORYCALCULATOR_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <utility>

namespace llvm {

class DILocalVariable;
class MachineFunction;
class MachineInstr;
class TargetRegisterInfo;

/// For each user variable, keep a list of instruction ranges where this
/// variable is accessible. The variables are listed in order of appearance.
class DbgValueHistoryMap {
public:
  /// Index in the entry vector.
  typedef size_t EntryIndex;

  /// Special value to indicate that an entry is valid until the end of the
  /// function.
  static const EntryIndex NoEntry = std::numeric_limits<EntryIndex>::max();

  /// Specifies a change in a variable's debug value history.
  ///
  /// There exist two types of entries:
  ///
  /// * Debug value entry:
  ///
  ///   A new debug value becomes live. If the entry's \p EndIndex is \p NoEntry,
  ///   the value is valid until the end of the function. For other values, the
  ///   index points to the entry in the entry vector that ends this debug
  ///   value. The ending entry can either be an overlapping debug value, or
  ///   an instruction that clobbers the value.
  ///
  /// * Clobbering entry:
  ///
  ///   This entry's instruction clobbers one or more preceding
  ///   register-described debug values that have their end index
  ///   set to this entry's position in the entry vector.
  class Entry {
  public:
    enum EntryKind { DbgValue, Clobber };

    Entry(const MachineInstr *Instr, EntryKind Kind)
        : Instr(Instr, Kind), EndIndex(NoEntry) {}

    const MachineInstr *getInstr() const { return Instr.getPointer(); }
    EntryIndex getEndIndex() const { return EndIndex; }
    EntryKind getEntryKind() const { return Instr.getInt(); }

    bool isClobber() const { return getEntryKind() == Clobber; }
    bool isDbgValue() const { return getEntryKind() == DbgValue; }
    bool isClosed() const { return EndIndex != NoEntry; }

    void endEntry(EntryIndex EndIndex);

  private:
    PointerIntPair<const MachineInstr *, 1, EntryKind> Instr;
    EntryIndex EndIndex;
  };
  using Entries = SmallVector<Entry, 4>;
  using InlinedEntity = std::pair<const DINode *, const DILocation *>;
  using EntriesMap = MapVector<InlinedEntity, Entries>;

private:
  EntriesMap VarEntries;

public:
  bool startDbgValue(InlinedEntity Var, const MachineInstr &MI,
                     EntryIndex &NewIndex);
  EntryIndex startClobber(InlinedEntity Var, const MachineInstr &MI);

  Entry &getEntry(InlinedEntity Var, EntryIndex Index) {
    auto &Entries = VarEntries[Var];
    return Entries[Index];
  }

  bool empty() const { return VarEntries.empty(); }
  void clear() { VarEntries.clear(); }
  EntriesMap::const_iterator begin() const { return VarEntries.begin(); }
  EntriesMap::const_iterator end() const { return VarEntries.end(); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// For each inlined instance of a source-level label, keep the corresponding
/// DBG_LABEL instruction. The DBG_LABEL instruction could be used to generate
/// a temporary (assembler) label before it.
class DbgLabelInstrMap {
public:
  using InlinedEntity = std::pair<const DINode *, const DILocation *>;
  using InstrMap = MapVector<InlinedEntity, const MachineInstr *>;

private:
  InstrMap LabelInstr;

public:
  void  addInstr(InlinedEntity Label, const MachineInstr &MI);

  bool empty() const { return LabelInstr.empty(); }
  void clear() { LabelInstr.clear(); }
  InstrMap::const_iterator begin() const { return LabelInstr.begin(); }
  InstrMap::const_iterator end() const { return LabelInstr.end(); }
};

void calculateDbgEntityHistory(const MachineFunction *MF,
                               const TargetRegisterInfo *TRI,
                               DbgValueHistoryMap &DbgValues,
                               DbgLabelInstrMap &DbgLabels);

} // end namespace llvm

#endif // LLVM_CODEGEN_DBGVALUEHISTORYCALCULATOR_H
