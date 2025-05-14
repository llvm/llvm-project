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

struct RootParameter {
  dxbc::RootParameterHeader Header;
  union {
    dxbc::RootConstants Constants;
    dxbc::RTS0::v2::RootDescriptor Descriptor;
  };
};
struct RootSignatureDesc {

  uint32_t Version = 2U;
  uint32_t Flags = 0U;
  uint32_t RootParameterOffset = 0U;
  uint32_t StaticSamplersOffset = 0u;
  uint32_t NumStaticSamplers = 0u;
  SmallVector<mcdxbc::RootParameter> Parameters;

  void write(raw_ostream &OS) const;

  size_t getSize() const;
};
} // namespace mcdxbc
} // namespace llvm
