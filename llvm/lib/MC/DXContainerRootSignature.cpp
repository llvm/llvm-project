//===- llvm/MC/DXContainerRootSignature.cpp - RootSignature -*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace llvm::mcdxbc;

void RootSignatureHeader::write(raw_ostream &OS) {

  support::endian::write(OS, Version, llvm::endianness::little);
  support::endian::write(OS, NumParameters, llvm::endianness::little);
  support::endian::write(OS, RootParametersOffset, llvm::endianness::little);
  support::endian::write(OS, NumStaticSamplers, llvm::endianness::little);
  support::endian::write(OS, StaticSamplersOffset, llvm::endianness::little);
  support::endian::write(OS, Flags, llvm::endianness::little);
}
