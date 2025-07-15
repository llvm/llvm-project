//===- DXILPostOptimizationValidation.h - Opt DXIL Validations -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Pass for validating IR after optimizations are applied and before
// lowering to DXIL.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H
#define LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H

#include "DXILRootSignature.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class RootSignatureBindingValidation {
private:
  llvm::SmallVector<dxil::ResourceInfo::ResourceBinding, 16> Bindings;
  struct TypeRange {
    uint32_t Start;
    uint32_t End;
  };
  std::unordered_map<uint32_t, TypeRange> Ranges;

public:
  void addBinding(const uint32_t &Type,
                  const dxil::ResourceInfo::ResourceBinding &Binding) {
    auto It = Ranges.find(Type);

    if (It == Ranges.end()) {
      uint32_t InsertPos = Bindings.size();
      Bindings.push_back(Binding);
      Ranges[Type] = {InsertPos, InsertPos + 1};
    } else {
      uint32_t InsertPos = It->second.End;
      Bindings.insert(Bindings.begin() + InsertPos, Binding);

      It->second.End++;

      for (auto &[type, range] : Ranges) {
        if (range.Start > InsertPos) {
          range.Start++;
          range.End++;
        }
      }
    }
  }

  llvm::ArrayRef<dxil::ResourceInfo::ResourceBinding>
  getBindingsOfType(const dxbc::DescriptorRangeType &Type) const {
    auto It = Ranges.find(llvm::to_underlying(Type));
    if (It == Ranges.end()) {
      return {};
    }
    return llvm::ArrayRef<dxil::ResourceInfo::ResourceBinding>(
        Bindings.data() + It->second.Start, It->second.End - It->second.Start);
  }
};

class DXILPostOptimizationValidation
    : public PassInfoMixin<DXILPostOptimizationValidation> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H
