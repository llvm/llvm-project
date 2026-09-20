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
#include <map>

namespace llvm {
class CallBase;
class MCSymbol;

class NVPTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  /// The parameter symbols whose image handles were replaced with image
  /// references.
  SmallPtrSet<const MCSymbol *, 8> ImageHandleSymbols;

  /// Stores a mapping from a unique call-site id to the call instruction that
  /// needs an indirect-call prototype emitted.
  std::map<unsigned, const CallBase *> CallPrototypes;

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

  void addCallPrototype(unsigned Id, const CallBase *CB) {
    CallPrototypes.try_emplace(Id, CB);
  }

  const std::map<unsigned, const CallBase *> &getCallPrototypes() const {
    return CallPrototypes;
  }
};
}

#endif
