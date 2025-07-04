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
#include "llvm/ADT/IntervalMap.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

static uint64_t combineUint32ToUint64(uint32_t High, uint32_t Low) {
  return (static_cast<uint64_t>(High) << 32) | Low;
}

class RootSignatureBindingValidation {
  using MapT =
      llvm::IntervalMap<uint64_t, dxil::ResourceInfo::ResourceBinding,
                        sizeof(llvm::dxil::ResourceInfo::ResourceBinding),
                        llvm::IntervalMapInfo<uint64_t>>;

private:
  MapT::Allocator Allocator;
  MapT CRegBindingsMap;
  MapT TRegBindingsMap;
  MapT URegBindingsMap;

  void addRange(const dxbc::RTS0::v2::RootDescriptor &Desc, uint32_t Type) {
    assert((Type == llvm::to_underlying(dxbc::RootParameterType::CBV) ||
            Type == llvm::to_underlying(dxbc::RootParameterType::SRV) ||
            Type == llvm::to_underlying(dxbc::RootParameterType::UAV)) &&
           "Invalid Type");

    llvm::dxil::ResourceInfo::ResourceBinding Binding;
    Binding.LowerBound = Desc.ShaderRegister;
    Binding.Space = Desc.RegisterSpace;
    Binding.Size = 1;

    uint64_t LowRange =
        combineUint32ToUint64(Binding.Space, Binding.LowerBound);
    uint64_t HighRange = combineUint32ToUint64(
        Binding.Space, Binding.LowerBound + Binding.Size - 1);

    switch (Type) {

    case llvm::to_underlying(dxbc::RootParameterType::CBV):
      CRegBindingsMap.insert(LowRange, HighRange, Binding);
      return;
    case llvm::to_underlying(dxbc::RootParameterType::SRV):
      TRegBindingsMap.insert(LowRange, HighRange, Binding);
      return;
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
      URegBindingsMap.insert(LowRange, HighRange, Binding);
      return;
    }
    llvm_unreachable("Invalid Type in add Range Method");
  }

  void addRange(const dxbc::RTS0::v2::DescriptorRange &Range) {

    llvm::dxil::ResourceInfo::ResourceBinding Binding;
    Binding.LowerBound = Range.BaseShaderRegister;
    Binding.Space = Range.RegisterSpace;
    Binding.Size = Range.NumDescriptors;

    uint64_t LowRange =
        combineUint32ToUint64(Binding.Space, Binding.LowerBound);
    uint64_t HighRange = combineUint32ToUint64(
        Binding.Space, Binding.LowerBound + Binding.Size - 1);

    switch (Range.RangeType) {
    case llvm::to_underlying(dxbc::DescriptorRangeType::CBV):
      CRegBindingsMap.insert(LowRange, HighRange, Binding);
      return;
    case llvm::to_underlying(dxbc::DescriptorRangeType::SRV):
      TRegBindingsMap.insert(LowRange, HighRange, Binding);
      return;
    case llvm::to_underlying(dxbc::DescriptorRangeType::UAV):
      URegBindingsMap.insert(LowRange, HighRange, Binding);
      return;
    }
    llvm_unreachable("Invalid Type in add Range Method");
  }

public:
  RootSignatureBindingValidation()
      : Allocator(), CRegBindingsMap(Allocator), TRegBindingsMap(Allocator),
        URegBindingsMap(Allocator) {}

  void addRsBindingInfo(mcdxbc::RootSignatureDesc &RSD,
                        dxbc::ShaderVisibility Visibility);

  bool checkCregBinding(dxil::ResourceInfo::ResourceBinding Binding);

  bool checkTRegBinding(dxil::ResourceInfo::ResourceBinding Binding);

  bool checkURegBinding(dxil::ResourceInfo::ResourceBinding Binding);
};

class DXILPostOptimizationValidation
    : public PassInfoMixin<DXILPostOptimizationValidation> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILPOSTOPTIMIZATIONVALIDATION_H
