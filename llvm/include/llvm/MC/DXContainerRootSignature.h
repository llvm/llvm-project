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

namespace llvm {

class raw_ostream;
namespace mcdxbc {

struct RootParameterInfo {
  dxbc::RootParameterHeader Header;
  size_t Location;

  RootParameterInfo() = default;

  RootParameterInfo(dxbc::RootParameterHeader Header, size_t Location)
      : Header(Header), Location(Location) {}
};

struct RootParametersContainer {
  SmallVector<RootParameterInfo> ParametersInfo;

  SmallVector<dxbc::RootConstants> Constants;
  SmallVector<dxbc::RTS0::v2::RootDescriptor> Descriptors;

  void addInfo(dxbc::RootParameterHeader Header, size_t Location) {
    ParametersInfo.push_back(RootParameterInfo(Header, Location));
  }

  void addParameter(dxbc::RootParameterHeader Header,
                    dxbc::RootConstants Constant) {
    addInfo(Header, Constants.size());
    Constants.push_back(Constant);
  }

  void addInvalidParameter(dxbc::RootParameterHeader Header) {
    addInfo(Header, -1);
  }

  void addParameter(dxbc::RootParameterHeader Header,
                    dxbc::RTS0::v2::RootDescriptor Descriptor) {
    addInfo(Header, Descriptors.size());
    Descriptors.push_back(Descriptor);
  }

  const std::pair<uint32_t, uint32_t>
  getTypeAndLocForParameter(uint32_t Location) const {
    const RootParameterInfo &Info = ParametersInfo[Location];
    return {Info.Header.ParameterType, Info.Location};
  }

  const dxbc::RootParameterHeader &getHeader(size_t Location) const {
    const RootParameterInfo &Info = ParametersInfo[Location];
    return Info.Header;
  }

  const dxbc::RootConstants &getConstant(size_t Index) const {
    return Constants[Index];
  }

  const dxbc::RTS0::v2::RootDescriptor &getRootDescriptor(size_t Index) const {
    return Descriptors[Index];
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
