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

Mangler::Mode Mangler::fromDataLayoutStr(StringRef DLStr) {
  for (StringRef Spec : split(DLStr, '-')) {
    if (!Spec.starts_with("m:"))
      continue;
    auto ModeStr = Spec.drop_front(2);
    assert(ModeStr.size() == 1 &&
           "invalid data layout string from Triple::computeDataLayout");
    switch (ModeStr[0]) {
    case 'e':
      return Mode::ELF;
    case 'l':
      return Mode::GOFF;
    case 'o':
      return Mode::MachO;
    case 'm':
      return Mode::Mips;
    case 'w':
      return Mode::WinCOFF;
    case 'x':
      return Mode::WinCOFFX86;
    case 'a':
      return Mode::XCOFF;
    default:
      llvm_unreachable("Invalid mangling mode from Triple::computeDataLayout");
    }
  }
  return Mode::None;
}

Mangler::Mode Mangler::fromTriple(const Triple &TT, StringRef ABIName) {
  return fromDataLayoutStr(TT.computeDataLayout(ABIName));
}

} // namespace llvm::orc
