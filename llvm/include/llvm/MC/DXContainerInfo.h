//===----- llvm/MC/DXContainerInfo.h - DXContainer Info ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_DXCONTAINERINFO_H
#define LLVM_MC_DXCONTAINERINFO_H

#include "llvm/Object/DXContainer.h"

namespace llvm {

class raw_ostream;

namespace mcdxbc {

struct DebugName {
  object::DXContainer::ILDNData BaseData;

  DebugName() { BaseData.first.Flags = 0; }

  void setFileName(StringRef FileName);
  void write(raw_ostream &OS) const;
};

} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_MC_DXCONTAINERINFO_H
