//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <limits>

namespace llvm {

class raw_ostream;
namespace mcdxbc {

struct RootParameter {
  dxbc::RootParameterHeader Header;
  union {
    dxbc::RootConstants Constants;
  };
};
struct RootSignatureDesc {

  dxbc::RootSignatureHeader Header;
  SmallVector<mcdxbc::RootParameter> Parameters;
  RootSignatureDesc()
      : Header(dxbc::RootSignatureHeader{
            2, 0, sizeof(dxbc::RootSignatureHeader), 0, 0, 0}) {}

  void write(raw_ostream &OS) const;

  size_t getSize() const {
    size_t size = sizeof(dxbc::RootSignatureHeader);

    for (const auto &P : Parameters) {
      switch (P.Header.ParameterType) {

      case dxbc::RootParameterType::Constants32Bit:
        size += sizeof(dxbc::RootConstants);
        break;
      }
    }
    return size;
  }
};
} // namespace mcdxbc
} // namespace llvm
