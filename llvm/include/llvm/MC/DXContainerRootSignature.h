//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <limits>

namespace llvm {

class raw_ostream;

namespace mcdxbc {
struct RootSignatureDesc {
  uint32_t Version = 2;
  uint32_t NumParameters = 0;
  uint32_t RootParametersOffset = 0;
  uint32_t NumStaticSamplers = 0;
  uint32_t StaticSamplersOffset = 0;
  uint32_t Flags = 0;

  void write(raw_ostream &OS) const;
};
} // namespace mcdxbc
} // namespace llvm
