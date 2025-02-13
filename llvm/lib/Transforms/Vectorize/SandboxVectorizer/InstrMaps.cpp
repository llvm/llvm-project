//===- InstructionMaps.cpp - Maps scalars to vectors and reverse ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Support/Debug.h"

namespace llvm::sandboxir {

#ifndef NDEBUG
void InstrMaps::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
