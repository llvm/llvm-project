//===- llvm/MC/DXContainerPSVInfo.cpp - DXContainer PSVInfo -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerPSVInfo.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::mcdxbc;
using namespace llvm::dxbc::PSV;

void PSVRuntimeInfo::write(raw_ostream &OS, uint32_t Version) const {
  uint32_t InfoSize;
  switch (Version) {
  case 0:
    InfoSize = sizeof(dxbc::PSV::v0::RuntimeInfo);
    break;
  case 1:
    InfoSize = sizeof(dxbc::PSV::v1::RuntimeInfo);
    break;
  case 2:
  default:
    InfoSize = sizeof(dxbc::PSV::v2::RuntimeInfo);
  }
  uint32_t InfoSizeSwapped = InfoSize;
  if (sys::IsBigEndianHost)
    sys::swapByteOrder(InfoSizeSwapped);
  // Write the size of the info.
  OS.write(reinterpret_cast<const char *>(&InfoSizeSwapped), sizeof(uint32_t));
  // Write the info itself.
  OS.write(reinterpret_cast<const char *>(&BaseData), InfoSize);

  uint32_t ResourceCount = static_cast<uint32_t>(Resources.size());
  if (sys::IsBigEndianHost)
    sys::swapByteOrder(ResourceCount);
  OS.write(reinterpret_cast<const char *>(&ResourceCount), sizeof(uint32_t));

  size_t BindingSize = (Version < 2) ? sizeof(v0::ResourceBindInfo)
                                     : sizeof(v2::ResourceBindInfo);
  for (const auto &Res : Resources)
    OS.write(reinterpret_cast<const char *>(&Res), BindingSize);
}
