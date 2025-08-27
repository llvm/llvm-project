//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DXCONTAINERROOTSIGNATURE_H
#define LLVM_MC_DXCONTAINERROOTSIGNATURE_H

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <limits>

namespace llvm {

class raw_ostream;
namespace mcdxbc {

struct RootParameterInfo {
  dxbc::RootParameterType Type;
  dxbc::ShaderVisibility Visibility;
  size_t Location;

  RootParameterInfo(dxbc::RootParameterType Type,
                    dxbc::ShaderVisibility Visibility, size_t Location)
      : Type(Type), Visibility(Visibility), Location(Location) {}
};

struct DescriptorTable {
  SmallVector<dxbc::RTS0::v2::DescriptorRange> Ranges;
  SmallVector<dxbc::RTS0::v2::DescriptorRange>::const_iterator begin() const {
    return Ranges.begin();
  }
  SmallVector<dxbc::RTS0::v2::DescriptorRange>::const_iterator end() const {
    return Ranges.end();
  }
};

struct RootParametersContainer {
  SmallVector<RootParameterInfo> ParametersInfo;

  SmallVector<dxbc::RTS0::v1::RootConstants> Constants;
  SmallVector<dxbc::RTS0::v2::RootDescriptor> Descriptors;
  SmallVector<DescriptorTable> Tables;

  void addInfo(dxbc::RootParameterType Type, dxbc::ShaderVisibility Visibility,
               size_t Location) {
    ParametersInfo.emplace_back(Type, Visibility, Location);
  }

  void addParameter(dxbc::RootParameterType Type,
                    dxbc::ShaderVisibility Visibility,
                    dxbc::RTS0::v1::RootConstants Constant) {
    addInfo(Type, Visibility, Constants.size());
    Constants.push_back(Constant);
  }

  void addParameter(dxbc::RootParameterType Type,
                    dxbc::ShaderVisibility Visibility,
                    dxbc::RTS0::v2::RootDescriptor Descriptor) {
    addInfo(Type, Visibility, Descriptors.size());
    Descriptors.push_back(Descriptor);
  }

  void addParameter(dxbc::RootParameterType Type,
                    dxbc::ShaderVisibility Visibility, DescriptorTable Table) {
    addInfo(Type, Visibility, Tables.size());
    Tables.push_back(Table);
  }

  const RootParameterInfo &getInfo(uint32_t Location) const {
    const RootParameterInfo &Info = ParametersInfo[Location];
    return Info;
  }

  const dxbc::RTS0::v1::RootConstants &getConstant(size_t Index) const {
    return Constants[Index];
  }

  const dxbc::RTS0::v2::RootDescriptor &getRootDescriptor(size_t Index) const {
    return Descriptors[Index];
  }

  const DescriptorTable &getDescriptorTable(size_t Index) const {
    return Tables[Index];
  }

  size_t size() const { return ParametersInfo.size(); }

  SmallVector<RootParameterInfo>::const_iterator begin() const {
    return ParametersInfo.begin();
  }
  SmallVector<RootParameterInfo>::const_iterator end() const {
    return ParametersInfo.end();
  }
};
struct RootSignatureDesc {

  uint32_t Version = 2U;
  uint32_t Flags = 0U;
  uint32_t RootParameterOffset = 0U;
  uint32_t StaticSamplersOffset = 0u;
  uint32_t NumStaticSamplers = 0u;
  mcdxbc::RootParametersContainer ParametersContainer;
  SmallVector<dxbc::RTS0::v1::StaticSampler> StaticSamplers;

  LLVM_ABI void write(raw_ostream &OS) const;

  LLVM_ABI size_t getSize() const;
  LLVM_ABI uint32_t computeRootParametersOffset() const;
  LLVM_ABI uint32_t computeStaticSamplersOffset() const;
};
} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_MC_DXCONTAINERROOTSIGNATURE_H
