//===-- DwarfException.h - Dwarf Exception Framework -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf exception info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ASMPRINTER_DWARFEXCEPTION_H
#define LLVM_CODEGEN_ASMPRINTER_DWARFEXCEPTION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include <vector>

namespace llvm {

template <typename T> class SmallVectorImpl;
struct LandingPadInfo;
class MachineModuleInfo;
class MachineMove;
class MachineInstr;
class MachineFunction;
class MCAsmInfo;
class MCExpr;
class MCSymbol;
class Function;
class AsmPrinter;

//===----------------------------------------------------------------------===//
/// DwarfException - Emits Dwarf exception handling directives.
///
class DwarfException {
protected:
  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  /// SharedTypeIds - How many leading type ids two landing pads have in common.
  static unsigned SharedTypeIds(const LandingPadInfo *L,
                                const LandingPadInfo *R);

  /// PadLT - Order landing pads lexicographically by type id.
  static bool PadLT(const LandingPadInfo *L, const LandingPadInfo *R);

  /// PadRange - Structure holding a try-range and the associated landing pad.
  struct PadRange {
    // The index of the landing pad.
    unsigned PadIndex;
    // The index of the begin and end labels in the landing pad's label lists.
    unsigned RangeIndex;
  };

  typedef DenseMap<MCSymbol *, PadRange> RangeMapType;

  /// ActionEntry - Structure describing an entry in the actions table.
  struct ActionEntry {
    int ValueForTypeID; // The value to write - may not be equal to the type id.
    int NextAction;
    unsigned Previous;
  };

  /// CallSiteEntry - Structure describing an entry in the call-site table.
  struct CallSiteEntry {
    // The 'try-range' is BeginLabel .. EndLabel.
    MCSymbol *BeginLabel; // zero indicates the start of the function.
    MCSymbol *EndLabel;   // zero indicates the end of the function.

    // The landing pad starts at PadLabel.
    MCSymbol *PadLabel;   // zero indicates that there is no landing pad.
    unsigned Action;
  };

  /// ComputeActionsTable - Compute the actions table and gather the first
  /// action index for each landing pad site.
  unsigned ComputeActionsTable(const SmallVectorImpl<const LandingPadInfo*>&LPs,
                               SmallVectorImpl<ActionEntry> &Actions,
                               SmallVectorImpl<unsigned> &FirstActions);

  /// CallToNoUnwindFunction - Return `true' if this is a call to a function
  /// marked `nounwind'. Return `false' otherwise.
  bool CallToNoUnwindFunction(const MachineInstr *MI);

  /// ComputeCallSiteTable - Compute the call-site table.  The entry for an
  /// invoke has a try-range containing the call, a non-zero landing pad and an
  /// appropriate action.  The entry for an ordinary call has a try-range
  /// containing the call and zero for the landing pad and the action.  Calls
  /// marked 'nounwind' have no entry and must not be contained in the try-range
  /// of any entry - they form gaps in the table.  Entries must be ordered by
  /// try-range address.
  void ComputeCallSiteTable(SmallVectorImpl<CallSiteEntry> &CallSites,
                            const RangeMapType &PadMap,
                            const SmallVectorImpl<const LandingPadInfo *> &LPs,
                            const SmallVectorImpl<unsigned> &FirstActions);

  /// EmitExceptionTable - Emit landing pads and actions.
  ///
  /// The general organization of the table is complex, but the basic concepts
  /// are easy.  First there is a header which describes the location and
  /// organization of the three components that follow.
  ///  1. The landing pad site information describes the range of code covered
  ///     by the try.  In our case it's an accumulation of the ranges covered
  ///     by the invokes in the try.  There is also a reference to the landing
  ///     pad that handles the exception once processed.  Finally an index into
  ///     the actions table.
  ///  2. The action table, in our case, is composed of pairs of type ids
  ///     and next action offset.  Starting with the action index from the
  ///     landing pad site, each type Id is checked for a match to the current
  ///     exception.  If it matches then the exception and type id are passed
  ///     on to the landing pad.  Otherwise the next action is looked up.  This
  ///     chain is terminated with a next action of zero.  If no type id is
  ///     found the frame is unwound and handling continues.
  ///  3. Type id table contains references to all the C++ typeinfo for all
  ///     catches in the function.  This tables is reversed indexed base 1.
  void EmitExceptionTable();

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfException(AsmPrinter *A);
  virtual ~DwarfException();

  /// EndModule - Emit all exception information that should come after the
  /// content.
  virtual void EndModule();

  /// BeginFunction - Gather pre-function exception information.  Assumes being
  /// emitted immediately after the function entry point.
  virtual void BeginFunction(const MachineFunction *MF);

  /// EndFunction - Gather and emit post-function exception information.
  virtual void EndFunction();
};

class DwarfCFIException : public DwarfException {
  /// shouldEmitPersonality - Per-function flag to indicate if .cfi_personality
  /// should be emitted.
  bool shouldEmitPersonality;

  /// shouldEmitLSDA - Per-function flag to indicate if .cfi_lsda
  /// should be emitted.
  bool shouldEmitLSDA;

  /// shouldEmitMoves - Per-function flag to indicate if frame moves info
  /// should be emitted.
  bool shouldEmitMoves;

  AsmPrinter::CFIMoveType moveTypeModule;

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfCFIException(AsmPrinter *A);
  virtual ~DwarfCFIException();

  /// EndModule - Emit all exception information that should come after the
  /// content.
  virtual void EndModule();

  /// BeginFunction - Gather pre-function exception information.  Assumes being
  /// emitted immediately after the function entry point.
  virtual void BeginFunction(const MachineFunction *MF);

  /// EndFunction - Gather and emit post-function exception information.
  virtual void EndFunction();
};

class ARMException : public DwarfException {
public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  ARMException(AsmPrinter *A);
  virtual ~ARMException();

  /// EndModule - Emit all exception information that should come after the
  /// content.
  virtual void EndModule();

  /// BeginFunction - Gather pre-function exception information.  Assumes being
  /// emitted immediately after the function entry point.
  virtual void BeginFunction(const MachineFunction *MF);

  /// EndFunction - Gather and emit post-function exception information.
  virtual void EndFunction();
};

class Win64Exception : public DwarfException {
  /// shouldEmitPersonality - Per-function flag to indicate if personality
  /// info should be emitted.
  bool shouldEmitPersonality;

  /// shouldEmitLSDA - Per-function flag to indicate if the LSDA
  /// should be emitted.
  bool shouldEmitLSDA;

  /// shouldEmitMoves - Per-function flag to indicate if frame moves info
  /// should be emitted.
  bool shouldEmitMoves;

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  Win64Exception(AsmPrinter *A);
  virtual ~Win64Exception();

  /// EndModule - Emit all exception information that should come after the
  /// content.
  virtual void EndModule();

  /// BeginFunction - Gather pre-function exception information.  Assumes being
  /// emitted immediately after the function entry point.
  virtual void BeginFunction(const MachineFunction *MF);

  /// EndFunction - Gather and emit post-function exception information.
  virtual void EndFunction();
};

} // End of namespace llvm

#endif
