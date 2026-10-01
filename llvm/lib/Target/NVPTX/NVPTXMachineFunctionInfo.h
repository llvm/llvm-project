//===-- NVPTXMachineFunctionInfo.h - NVPTX-specific Function Info  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class is attached to a MachineFunction instance and tracks target-
// dependent information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXMACHINEFUNCTIONINFO_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MCContext.h"

namespace llvm {
class CallBase;
class MCSymbol;

class NVPTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  /// The parameter symbols whose image handles were replaced with image
  /// references.
  SmallPtrSet<const MCSymbol *, 8> ImageHandleSymbols;

  using CallProtoTy = std::pair<const CallBase *, MCSymbol *>;
  /// Stores the call instructions that need an indirect-call prototype emitted.
  std::vector<CallProtoTy> CallPrototypes;

public:
  NVPTXMachineFunctionInfo(const Function &F, const TargetSubtargetInfo *STI) {}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override {
    return DestMF.cloneInfo<NVPTXMachineFunctionInfo>(*this);
  }

  /// Record that \p Symbol's handle was replaced with an image reference.
  void addImageHandleSymbol(const MCSymbol *Symbol) {
    ImageHandleSymbols.insert(Symbol);
  }

  /// Check whether \p Symbol's handle was replaced with an image reference.
  bool checkImageHandleSymbol(const MCSymbol *Symbol) const {
    return ImageHandleSymbols.contains(Symbol);
  }

  MCSymbol *addCallPrototype(const CallBase *CB, MachineFunction &MF) {
    MCSymbol *Symbol = MF.getContext().createTempSymbol("prototype_");
    CallPrototypes.push_back({CB, Symbol});
    return Symbol;
  }

  ArrayRef<CallProtoTy> getCallPrototypes() const { return CallPrototypes; }
};
}

#endif
