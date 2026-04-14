//===-- NVPTXDwarfDebug.h - NVPTX DwarfDebug Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTXDwarfDebug class, the NVPTX-specific subclass
// of DwarfDebug. It customizes DWARF emission for PTX: address space
// attributes, compile-unit range suppression, base address handling, and
// enhanced line information with inlined_at directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXDWARFDEBUG_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXDWARFDEBUG_H

#include "../../CodeGen/AsmPrinter/DwarfCompileUnit.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

/// NVPTX-specific DwarfDebug implementation.
///
/// Customizes DWARF emission for PTX targets: DWARF v2 defaults, address
/// space attributes (DW_AT_address_class) for cuda-gdb, compile-unit range
/// suppression, range-list base address handling, and enhanced line
/// information with inlined_at directives.
class NVPTXDwarfDebug : public DwarfDebug {
private:
  /// Set of inlined_at locations that have already been emitted.
  /// Used to avoid redundant emission of parent chain .loc directives.
  DenseSet<const DILocation *> EmittedInlinedAtLocs;

public:
  NVPTXDwarfDebug(AsmPrinter *A);

  /// Get or create an MCSymbol in .debug_str for a function's linkage name.
  /// Used to reference the function name in .loc directives with inlined_at.
  MCSymbol *getOrCreateFuncNameSymbol(StringRef LinkageName);

  /// Returns true if the enhanced lineinfo mode (with inlined_at) is active
  /// for the given MachineFunction.
  bool isEnhancedLineinfo(const MachineFunction &MF) const;

  bool shouldResetBaseAddress(const MCSection &Section) const override;
  const DIExpression *adjustExpressionForTarget(
      const DIExpression *Expr,
      std::optional<unsigned> &TargetAddrSpace) const override;
  void addTargetVariableAttributes(
      DwarfCompileUnit &CU, DIE &Die, std::optional<unsigned> TargetAddrSpace,
      VariableLocationKind VarLocKind,
      const GlobalVariable *GV = nullptr) const override;

protected:
  void initializeTargetDebugInfo(const MachineFunction &MF) override;
  void recordTargetSourceLine(const DebugLoc &DL, unsigned Flags) override;
  bool shouldAttachCompileUnitRanges() const override;
  bool shouldEmitDwarfPubSections() const override { return false; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_NVPTXDWARFDEBUG_H
