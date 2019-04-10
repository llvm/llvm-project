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
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <utility>

namespace llvm {

class DILocalVariable;
class MachineFunction;
class MachineInstr;
class TargetRegisterInfo;

// For each user variable, keep a list of instruction ranges where this variable
// is accessible. The variables are listed in order of appearance.
class DbgValueHistoryMap {
public:
  /// Specifies an instruction range where a DBG_VALUE is valid.
  ///
  /// \p Begin is a DBG_VALUE instruction, specifying the location of a
  /// variable, which is assumed to be valid until the end of the range. If \p
  /// End is not specified, the location is valid until the first overlapping
  /// DBG_VALUE if any such DBG_VALUE exists, otherwise it is valid until the
  /// end of the function.
  class Entry {
    const MachineInstr *Begin;
    const MachineInstr *End;

  public:
    Entry(const MachineInstr *Begin) : Begin(Begin), End(nullptr) {}

    const MachineInstr *getBegin() const { return Begin; }
    const MachineInstr *getEnd() const { return End; }

    bool isClosed() const { return End; }

    void endEntry(const MachineInstr &End);
  };
  using Entries = SmallVector<Entry, 4>;
  using InlinedEntity = std::pair<const DINode *, const DILocation *>;
  using EntriesMap = MapVector<InlinedEntity, Entries>;

private:
  EntriesMap VarEntries;

public:
  void startEntry(InlinedEntity Var, const MachineInstr &MI);
  void endEntry(InlinedEntity Var, const MachineInstr &MI);

  // Returns register currently describing @Var. If @Var is currently
  // unaccessible or is not described by a register, returns 0.
  unsigned getRegisterForVar(InlinedEntity Var) const;

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
