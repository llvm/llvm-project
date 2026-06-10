//===- AMDGPUHWEvents.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUHWEvents.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace AMDGPU {
void HWEventSet::print(raw_ostream &OS) const {
  ListSeparator LS(", ");
  for (HWEvent Event : hw_events()) {
    if (contains(Event))
      OS << LS << toString(Event);
  }
}

void HWEventSet::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
} // namespace AMDGPU
} // namespace llvm
