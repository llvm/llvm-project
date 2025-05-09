//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/DXContainer.h"
#include <cstdint>
#include <limits>
#include <variant>

namespace llvm {

class raw_ostream;
namespace mcdxbc {

struct RootParameterInfo {
  dxbc::RootParameterHeader Header;
  size_t Location;

  RootParameterInfo() = default;

  RootParameterInfo(dxbc::RootParameterHeader H, size_t L)
      : Header(H), Location(L) {}
};

using RootDescriptor = std::variant<dxbc::RTS0::v1::RootDescriptor,
                                    dxbc::RTS0::v2::RootDescriptor>;
using ParametersView = std::variant<const dxbc::RootConstants *,
                                    const dxbc::RTS0::v1::RootDescriptor *,
                                    const dxbc::RTS0::v2::RootDescriptor *>;
struct RootParametersContainer {
  SmallVector<RootParameterInfo> ParametersInfo;

  SmallVector<dxbc::RootConstants> Constants;
  SmallVector<RootDescriptor> Descriptors;

  void addInfo(dxbc::RootParameterHeader H, size_t L) {
    ParametersInfo.push_back(RootParameterInfo(H, L));
  }

  void addParameter(dxbc::RootParameterHeader H, dxbc::RootConstants C) {
    addInfo(H, Constants.size());
    Constants.push_back(C);
  }

  void addParameter(dxbc::RootParameterHeader H,
                    dxbc::RTS0::v1::RootDescriptor D) {
    addInfo(H, Descriptors.size());
    Descriptors.push_back(D);
  }

  void addParameter(dxbc::RootParameterHeader H,
                    dxbc::RTS0::v2::RootDescriptor D) {
    addInfo(H, Descriptors.size());
    Descriptors.push_back(D);
  }

  std::optional<ParametersView> getParameter(const RootParameterInfo *H) const {
    switch (H->Header.ParameterType) {
    case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit):
      return &Constants[H->Location];
    case llvm::to_underlying(dxbc::RootParameterType::CBV):
    case llvm::to_underlying(dxbc::RootParameterType::SRV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
      const RootDescriptor &VersionedParam = Descriptors[H->Location];
      if (std::holds_alternative<dxbc::RTS0::v1::RootDescriptor>(
              VersionedParam)) {
        return &std::get<dxbc::RTS0::v1::RootDescriptor>(VersionedParam);
      }
      return &std::get<dxbc::RTS0::v2::RootDescriptor>(VersionedParam);
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
