//===------------ SYCLUtils.cpp - SYCL utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SYCL utility functions.
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Utils/SYCLUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

void writeSYCLStringTable(const SYCLStringTable &Table, raw_ostream &OS) {
  assert(!Table.empty() && "table should contain at least column titles");
  assert(!Table[0].empty() && "table should be non-empty");
  OS << '[' << join(Table[0].begin(), Table[0].end(), "|") << "]\n";
  for (size_t I = 1, E = Table.size(); I != E; ++I) {
    assert(Table[I].size() == Table[0].size() && "row's size should be equal");
    OS << join(Table[I].begin(), Table[I].end(), "|") << '\n';
  }
}

} // namespace llvm
