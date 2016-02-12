//===-- llvm/lib/CodeGen/AsmPrinter/DebugHandlerBase.h --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common functionality for different debug information format backends.
// LLVM currently supports DWARF and CodeView.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGHANDLERBASE_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGHANDLERBASE_H

#include "AsmPrinterHandler.h"
#include "DbgValueHistoryCalculator.h"
#include "llvm/CodeGen/LexicalScopes.h"

namespace llvm {

class AsmPrinter;
class MachineModuleInfo;

/// Base class for debug information backends. Common functionality related to
/// tracking which variables and scopes are alive at a given PC live here.
class DebugHandlerBase : public AsmPrinterHandler {
protected:
  DebugHandlerBase(AsmPrinter *A);

  /// Target of debug info emission.
  AsmPrinter *Asm;

  /// Collected machine module information.
  MachineModuleInfo *MMI;

  /// Previous instruction's location information. This is used to
  /// determine label location to indicate scope boundries in dwarf
  /// debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel = nullptr;

  /// This location indicates end of function prologue and beginning of
  /// function body.
  DebugLoc PrologEndLoc;

  /// If nonnull, stores the current machine instruction we're processing.
  const MachineInstr *CurMI = nullptr;

  LexicalScopes LScopes;

  /// History of DBG_VALUE and clobber instructions for each user
  /// variable.  Variables are listed in order of appearance.
  DbgValueHistoryMap DbgValues;

  /// Maps instruction with label emitted before instruction.
  /// FIXME: Make this private from DwarfDebug, we have the necessary accessors
  /// for it.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsBeforeInsn;

  /// Maps instruction with label emitted after instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsAfterInsn;

  /// Indentify instructions that are marking the beginning of or
  /// ending of a scope.
  void identifyScopeMarkers();

  /// Ensure that a label will be emitted before MI.
  void requestLabelBeforeInsn(const MachineInstr *MI) {
    LabelsBeforeInsn.insert(std::make_pair(MI, nullptr));
  }

  /// Ensure that a label will be emitted after MI.
  void requestLabelAfterInsn(const MachineInstr *MI) {
    LabelsAfterInsn.insert(std::make_pair(MI, nullptr));
  }

  // AsmPrinterHandler overrides.
public:
  void beginInstruction(const MachineInstr *MI) override;
  void endInstruction() override;

  void beginFunction(const MachineFunction *MF) override;
  void endFunction(const MachineFunction *MF) override;

  /// Return Label preceding the instruction.
  MCSymbol *getLabelBeforeInsn(const MachineInstr *MI);

  /// Return Label immediately following the instruction.
  MCSymbol *getLabelAfterInsn(const MachineInstr *MI);

  /// Determine the relative position of the pieces described by P1 and P2.
  /// Returns  -1 if P1 is entirely before P2, 0 if P1 and P2 overlap,
  /// 1 if P1 is entirely after P2.
  static int pieceCmp(const DIExpression *P1, const DIExpression *P2);

  /// Determine whether two variable pieces overlap.
  static bool piecesOverlap(const DIExpression *P1, const DIExpression *P2);
};

}

#endif
