//===------- SymbolStringPool.cpp - SymbolStringPool implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::orc {

raw_ostream &operator<<(raw_ostream &OS, const SymbolStringPtrBase &Sym) {
  return OS << Sym.S->first();
}

} // namespace llvm::orc
