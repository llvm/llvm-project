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
struct RootSignatureDesc {

  dxbc::RootSignatureHeader Header;
  SmallVector<dxbc::RootParameter> Parameters;
  RootSignatureDesc() : Header(dxbc::RootSignatureHeader{2, 0}) {}

  void write(raw_ostream &OS) const;
};
} // namespace mcdxbc
} // namespace llvm
