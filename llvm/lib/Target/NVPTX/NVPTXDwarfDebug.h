//===-- NVPTXDwarfDebug.h - NVPTX DwarfDebug Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper classes and functions for NVPTX-specific debug
// information processing, particularly for inlined function call sites and
// enhanced line information with inlined_at directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXDWARFDEBUG_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXDWARFDEBUG_H

#include "../../CodeGen/AsmPrinter/DwarfCompileUnit.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

/// NVPTXDwarfDebug - NVPTX-specific DwarfDebug implementation.
/// Inherits from DwarfDebug to provide enhanced line information with
/// inlined_at support.
class NVPTXDwarfDebug : public DwarfDebug {
private:
  /// Set of inlined_at locations that have already been emitted.
  /// Used to avoid redundant emission of parent chain .loc directives.
  DenseSet<const DILocation *> EmittedInlinedAtLocs;

public:
  /// Constructor - Pass through to DwarfDebug constructor.
  NVPTXDwarfDebug(AsmPrinter *A);

protected:
  /// Override to collect inlined_at locations.
  void initializeTargetDebugInfo(const MachineFunction &MF) override;
  /// Override to record source line information with inlined_at support.
  void recordTargetSourceLine(const DebugLoc &DL, unsigned Flags) override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_NVPTX_NVPTXDWARFDEBUG_H
