//===- llvm/CodeGen/AsmPrinter/DbgEntityHistoryCalculator.h -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H

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
  // Each instruction range starts with a DBG_VALUE instruction, specifying the
  // location of a variable, which is assumed to be valid until the end of the
  // range. If end is not specified, location is valid until the start
  // instruction of the next instruction range, or until the end of the
  // function.
public:
  using InstrRange = std::pair<const MachineInstr *, const MachineInstr *>;
  using InstrRanges = SmallVector<InstrRange, 4>;
  using InlinedVariable =
      std::pair<const DILocalVariable *, const DILocation *>;
  using InstrRangesMap = MapVector<InlinedVariable, InstrRanges>;

private:
  InstrRangesMap VarInstrRanges;

public:
  void startInstrRange(InlinedVariable Var, const MachineInstr &MI);
  void endInstrRange(InlinedVariable Var, const MachineInstr &MI);

  // Returns register currently describing @Var. If @Var is currently
  // unaccessible or is not described by a register, returns 0.
  unsigned getRegisterForVar(InlinedVariable Var) const;

  bool empty() const { return VarInstrRanges.empty(); }
  void clear() { VarInstrRanges.clear(); }
  InstrRangesMap::const_iterator begin() const { return VarInstrRanges.begin(); }
  InstrRangesMap::const_iterator end() const { return VarInstrRanges.end(); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// For each inlined instance of a source-level label, keep the corresponding
/// DBG_LABEL instruction. The DBG_LABEL instruction could be used to generate
/// a temporary (assembler) label before it.
class DbgLabelInstrMap {
public:
  using InlinedLabel = std::pair<const DILabel *, const DILocation *>;
  using InstrMap = MapVector<InlinedLabel, const MachineInstr *>;

private:
  InstrMap LabelInstr;

public:
  void  addInstr(InlinedLabel Label, const MachineInstr &MI);

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

#endif // LLVM_LIB_CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H
