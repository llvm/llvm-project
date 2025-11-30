//===- CGDataPatchItem.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for patching codegen data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CGDATA_CGDATAPATCHITEM_H
#define LLVM_CGDATA_CGDATAPATCHITEM_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

/// A struct to define how the data stream should be patched.
struct CGDataPatchItem {
  // Where to patch.
  uint64_t Pos;
  // Source data.
  std::vector<uint64_t> D;

  CGDataPatchItem(uint64_t Pos, const uint64_t *D, int N)
      : Pos(Pos), D(D, D + N) {}
};

} // namespace llvm

#endif // LLVM_CGDATA_CGDATAPATCHITEM_H
