//===- PackReuse.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pack de-duplication pass.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PACKREUSE_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PACKREUSE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/SandboxIR/Region.h"

namespace llvm::sandboxir {

/// This pass aims at de-duplicating packs, i.e., try to reuse already existing
/// pack patterns instead of keeping both.
/// This is useful because even though the duplicates will most probably be
/// optimized away by future passes, their added cost can make vectorization
/// more conservative than it should be.
class PackReuse final : public RegionPass {
  bool Change = false;

public:
  PackReuse() : RegionPass("pack-reuse") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_PACKREUSE_H
