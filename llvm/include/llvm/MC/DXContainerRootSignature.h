//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace llvm {

class raw_ostream;
namespace mcdxbc {

struct RootParameterInfo {
  dxbc::RTS0::v0::RootParameterHeader Header;
  size_t Location;

  RootParameterInfo() = default;

  RootParameterInfo(dxbc::RTS0::v0::RootParameterHeader H, size_t L)
      : Header(H), Location(L) {}
};
using DescriptorRanges = std::variant<dxbc::RTS0::v0::DescriptorRange,
                                      dxbc::RTS0::v1::DescriptorRange>;
struct DescriptorTable {
  SmallVector<DescriptorRanges> Ranges;

  SmallVector<DescriptorRanges>::const_iterator begin() const {
    return Ranges.begin();
  }
  SmallVector<DescriptorRanges>::const_iterator end() const {
    return Ranges.end();
  }
};

using RootDescriptor = std::variant<dxbc::RTS0::v0::RootDescriptor,
                                    dxbc::RTS0::v1::RootDescriptor>;

using ParametersView = std::variant<const dxbc::RTS0::v0::RootConstants *,
                                    const dxbc::RTS0::v0::RootDescriptor *,
                                    const dxbc::RTS0::v1::RootDescriptor *,
                                    const DescriptorTable *>;
struct RootParametersContainer {
  SmallVector<RootParameterInfo> ParametersInfo;
  SmallVector<dxbc::RTS0::v0::RootConstants> Constants;
  SmallVector<RootDescriptor> Descriptors;
  SmallVector<DescriptorTable> Tables;

  void addInfo(dxbc::RTS0::v0::RootParameterHeader H, size_t L) {
    ParametersInfo.push_back(RootParameterInfo(H, L));
  }

  void addParameter(dxbc::RTS0::v0::RootParameterHeader H,
                    dxbc::RTS0::v0::RootConstants C) {
    addInfo(H, Constants.size());
    Constants.push_back(C);
  }

  void addParameter(dxbc::RTS0::v0::RootParameterHeader H,
                    dxbc::RTS0::v0::RootDescriptor D) {
    addInfo(H, Descriptors.size());
    Descriptors.push_back(D);
  }

  void addParameter(dxbc::RTS0::v0::RootParameterHeader H,
                    dxbc::RTS0::v1::RootDescriptor D) {
    addInfo(H, Descriptors.size());
    Descriptors.push_back(D);
  }

  void addParameter(dxbc::RTS0::v0::RootParameterHeader H, DescriptorTable D) {
    addInfo(H, Tables.size());
    Tables.push_back(D);
  }

  std::optional<ParametersView> getParameter(const RootParameterInfo *H) const {
    switch (H->Header.ParameterType) {
    case llvm::to_underlying(dxbc::RTS0::RootParameterType::Constants32Bit):
      return &Constants[H->Location];
    case llvm::to_underlying(dxbc::RTS0::RootParameterType::CBV):
    case llvm::to_underlying(dxbc::RTS0::RootParameterType::SRV):
    case llvm::to_underlying(dxbc::RTS0::RootParameterType::UAV): {
      const RootDescriptor &VersionedParam = Descriptors[H->Location];
      if (std::holds_alternative<dxbc::RTS0::v0::RootDescriptor>(
              VersionedParam)) {
        return &std::get<dxbc::RTS0::v0::RootDescriptor>(VersionedParam);
      }
      return &std::get<dxbc::RTS0::v1::RootDescriptor>(VersionedParam);
    }
    case llvm::to_underlying(dxbc::RTS0::RootParameterType::DescriptorTable):
      return &Tables[H->Location];
    }

    return std::nullopt;
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

  void write(raw_ostream &OS) const;

  size_t getSize() const;
};
} // namespace mcdxbc
} // namespace llvm
