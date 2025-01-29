//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace llvm::mcdxbc;

void RootSignatureHeader::write(raw_ostream &OS) {

  uint32_t SizeInfo = sizeof(RootSignatureHeader);
  uint32_t ParamsSize = Parameters.size();
  support::endian::write(OS, SizeInfo, llvm::endianness::little);
  support::endian::write(OS, Flags, llvm::endianness::little);
  support::endian::write(OS, ParamsSize, llvm::endianness::little);

  if (Parameters.size() > 0) {
    uint32_t BindingSize = sizeof(dxbc::RootParameter);

    support::endian::write(OS, BindingSize, llvm::endianness::little);

    for (const auto &Param : Parameters)
      OS.write(reinterpret_cast<const char *>(&Param), BindingSize);
  }
}

void RootSignatureHeader::pushPart(dxbc::RootParameter Param) {
  Parameters.push_back(Param);
}
