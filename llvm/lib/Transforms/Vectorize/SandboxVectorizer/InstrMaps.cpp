//===- InstructionMaps.cpp - Maps scalars to vectors and reverse ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"

namespace llvm::sandboxir {

#ifndef NDEBUG
void Action::print(raw_ostream &OS) const {
  OS << Idx << ". " << *LegalityRes << " Depth:" << Depth << "\n";
  OS.indent(2) << "Bndl:\n";
  for (Value *V : Bndl)
    OS.indent(4) << *V << "\n";
  OS.indent(2) << "UserBndl:\n";
  for (Value *V : UserBndl)
    OS.indent(4) << *V << "\n";
}

void Action::dump() const { print(dbgs()); }

void InstrMaps::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
