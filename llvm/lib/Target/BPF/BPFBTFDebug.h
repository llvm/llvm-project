//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// BPF-specific BTF debug info emission. Extends the target-independent
/// BTFDebug with CO-RE relocations, .maps section handling, and BPF
/// instruction processing.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFBTFDEBUG_H
#define LLVM_LIB_TARGET_BPF_BPFBTFDEBUG_H

#include "llvm/CodeGen/BTFDebug.h"

namespace llvm {

class MCInst;

/// BPF-specific BTF debug handler.
///
/// Extends the target-independent BTFDebug with:
/// - CO-RE (Compile-Once Run-Everywhere) field relocations
/// - .maps section handling for BPF map definitions
/// - BPF instruction lowering for patchable instructions
class BPFBTFDebug : public BTFDebug {
  bool MapDefNotCollected;
  std::map<const GlobalVariable *, std::pair<int64_t, uint32_t>> PatchImms;

  /// Visit a .maps type definition.
  void visitMapDefType(const DIType *Ty, uint32_t &TypeId);

  /// Process global variable references from BPF instructions (CO-RE).
  void processGlobalValue(const MachineOperand &MO);

  /// Generate a field relocation record for CO-RE.
  void generatePatchImmReloc(const MCSymbol *ORSym, uint32_t RootId,
                             const GlobalVariable *GVar, bool IsAma);

  /// Process .maps globals separately.
  void processMapDefGlobals();

  /// Process all globals including .maps handling.
  void processGlobals() override;

protected:
  void beginFunctionImpl(const MachineFunction *MF) override;

public:
  BPFBTFDebug(AsmPrinter *AP);

  /// Emit proper patchable instructions (CO-RE lowering).
  bool InstLower(const MachineInstr *MI, MCInst &OutMI);

  void processBeginInstruction(const MachineInstr *MI) override;
  void endModule() override;
};

} // end namespace llvm

#endif
