//===-- DIERef.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DIERef.h"
#include "llvm/Support/Format.h"

void llvm::format_provider<DIERef>::format(const DIERef &ref, raw_ostream &OS,
                                           StringRef Style) {
  if (ref.dwo_num())
    OS << format_hex_no_prefix(*ref.dwo_num(), 8) << "/";
  OS << (ref.section() == DIERef::DebugInfo ? "INFO" : "TYPE");
  OS << "/" << format_hex_no_prefix(ref.die_offset(), 8);
}
