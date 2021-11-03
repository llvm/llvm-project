//===- HexagonArch.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONARCH_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONARCH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "HexagonDepArch.h"
#include <algorithm>

namespace llvm {
namespace Hexagon {

template <class ArchCont, typename Val>
llvm::Optional<ArchEnum> GetCpu(ArchCont const &ArchList, Val CPUString) {
  llvm::Optional<ArchEnum> Res;
  auto Entry = ArchList.find(CPUString);
  if (Entry != ArchList.end())
    Res = Entry->second;
  return Res;
}
} // namespace Hexagon
} // namespace llvm
#endif  // LLVM_LIB_TARGET_HEXAGON_HEXAGONARCH_H
