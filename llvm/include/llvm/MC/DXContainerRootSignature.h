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



struct RootSignatureHeader {
  uint32_t Version = 2;
  uint32_t NumParameters = 0;
  uint32_t RootParametersOffset = 0;
  uint32_t NumStaticSamplers = 0;
  uint32_t StaticSamplersOffset = 0;
  uint32_t Flags = 0;

  void write(raw_ostream &OS);
};

struct RootConstants {
  uint32_t ShaderRegister;
  uint32_t RegisterSpace;
  uint32_t Num32BitValues;

  void write(raw_ostream &OS);
};

struct RootParameter {
  dxbc::RootParameterType ParameterType;
  union {
    RootConstants Constants;
  };
  dxbc::ShaderVisibilityFlag ShaderVisibility;

  void write(raw_ostream &OS);
}; 
} // namespace mcdxbc
} // namespace llvm
