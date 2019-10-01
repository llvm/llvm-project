//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"

namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const LLT Ty) {
  std::string Repr;
  raw_string_ostream SS{Repr};
  Ty.print(SS);
  OS << SS.str();
  return OS;
}

std::ostream &
operator<<(std::ostream &OS, const MachineFunction &MF) {
  std::string Repr;
  raw_string_ostream SS{Repr};
  MF.print(SS);
  OS << SS.str();
  return OS;
}

}
