//===- llvm/MC/DXContainerRootSignature.cpp - DXContainer RootSignature -*- C++
//-------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/SwapByteOrder.h"
#include <iterator>

using namespace llvm;
using namespace llvm::mcdxbc;

void RootSignatureHeader::write(raw_ostream &OS, uint32_t Version) {

  uint32_t SizeInfo = sizeof(this);
  // support::endian::write(OS, SizeInfo, llvm::endianness::little);

  if (sys::IsBigEndianHost) {
    sys::swapByteOrder(Version);
    sys::swapByteOrder(Flags);
  }

  OS.write(reinterpret_cast<const char *>(this), SizeInfo);
}
