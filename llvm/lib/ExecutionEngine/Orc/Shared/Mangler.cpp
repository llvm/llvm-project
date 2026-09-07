//===------------- Mangler.cpp -- Linker name mangling for ORC ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/Mangler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"

#include <cassert>

#define DEBUG_TYPE "orc"

namespace llvm::orc {

Mangler::ManglingMode Mangler::fromDataLayoutStr(StringRef DLStr) {
  for (StringRef Spec : split(DLStr, '-')) {
    if (!Spec.starts_with("m:"))
      continue;
    auto ModeStr = Spec.drop_front(2);
    assert(ModeStr.size() == 1 &&
           "invalid data layout string from Triple::computeDataLayout");
    switch (ModeStr[0]) {
    case 'e':
      return ManglingMode::ELF;
    case 'l':
      return ManglingMode::GOFF;
    case 'o':
      return ManglingMode::MachO;
    case 'm':
      return ManglingMode::Mips;
    case 'w':
      return ManglingMode::WinCOFF;
    case 'x':
      return ManglingMode::WinCOFFX86;
    case 'a':
      return ManglingMode::XCOFF;
    default:
      llvm_unreachable("Invalid mangling mode from Triple::computeDataLayout");
    }
  }
  return ManglingMode::None;
}

Mangler::ManglingMode Mangler::fromTriple(const Triple &TT, StringRef ABIName) {
  return fromDataLayoutStr(TT.computeDataLayout(ABIName));
}

} // namespace llvm::orc
