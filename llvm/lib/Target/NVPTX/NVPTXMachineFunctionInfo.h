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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

// Cache policy data for a single memory operation.
// Stored per-MMO to avoid pointer collisions when multiple memops share
// the same pointer value but have different cache policies.
struct NVPTXMMOCachePolicyData {
  uint64_t Policy;           // The 64-bit cache policy value for L2::cache_hint
  unsigned CacheControlHint; // Cache control hints (L1/L2 eviction, prefetch)
};

class NVPTXMachineFunctionInfo : public MachineFunctionInfo {
private:
  /// Stores a mapping from index to symbol name for image handles that are
  /// replaced with image references
  SmallVector<std::string, 8> ImageHandleList;

  /// Per-MMO cache policy data for !mem.cache_hint metadata.
  /// Keyed by MachineMemOperand* for direct lookup during instruction
  /// selection.
  DenseMap<MachineMemOperand *, NVPTXMMOCachePolicyData> CachePolicyMap;

public:
  NVPTXMachineFunctionInfo(const Function &F, const TargetSubtargetInfo *STI) {}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override {
    return DestMF.cloneInfo<NVPTXMachineFunctionInfo>(*this);
  }

  /// Returns the index for the symbol \p Symbol. If the symbol was previously,
  /// added, the same index is returned. Otherwise, the symbol is added and the
  /// new index is returned.
  unsigned getImageHandleSymbolIndex(StringRef Symbol) {
    // Is the symbol already present?
    for (unsigned i = 0, e = ImageHandleList.size(); i != e; ++i)
      if (ImageHandleList[i] == Symbol)
        return i;
    // Nope, insert it
    ImageHandleList.push_back(Symbol.str());
    return ImageHandleList.size()-1;
  }

  /// Check if the symbol has a mapping. Having a mapping means the handle is
  /// replaced with a reference
  bool checkImageHandleSymbol(StringRef Symbol) const {
    return llvm::is_contained(ImageHandleList, Symbol);
  }

  /// Store cache policy data for a MachineMemOperand.
  void setCachePolicyData(MachineMemOperand *MMO, uint64_t Policy,
                          unsigned CacheControlHint) {
    CachePolicyMap[MMO] = {Policy, CacheControlHint};
  }

  /// Get cache policy data for a MachineMemOperand.
  /// Returns nullptr if no data exists for this MMO.
  const NVPTXMMOCachePolicyData *
  getCachePolicyData(MachineMemOperand *MMO) const {
    auto It = CachePolicyMap.find(MMO);
    if (It == CachePolicyMap.end())
      return nullptr;
    return &It->second;
  }
};
}

#endif
