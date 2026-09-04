//===-- llvm/CodeGen/MachineModuleInfo.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULESLOTTRACKER_H
#define LLVM_CODEGEN_MACHINEMODULESLOTTRACKER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Function;
class MachineModuleInfo;
class MachineFunction;
class Module;

using MFGetterFnT = function_ref<MachineFunction *(const Function &)>;

class LLVM_ABI MachineModuleSlotTracker : public ModuleSlotTracker {
  const MachineFunction *TheMF;
  MachineMDNodeListType MachineMDNodes;
  SmallPtrSet<const DILocation *, 4> InlineDebugLocations;

  void collectMachineFunctionMetadata(
      SmallVectorImpl<const MDNode *> &Metadata, const MachineFunction &MF,
      SmallVectorImpl<const MDNode *> *DebugLocations = nullptr) const;

public:
  MachineModuleSlotTracker(MFGetterFnT Fn, const MachineFunction *MF);
  ~MachineModuleSlotTracker() override;

  /// Renumber module and machine metadata for canonical MIR output.
  void renumberMetadataForAssembly();
  void collectMachineMDNodes(MachineMDNodeListType &L) const;
  bool shouldPrintDebugLocationInline(const DILocation *DL) const override;
};

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINEMODULESLOTTRACKER_H
