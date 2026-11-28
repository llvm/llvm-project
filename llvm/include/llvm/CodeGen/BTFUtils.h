//===- BTFUtils.h - Helpers for BTF emission -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BTFUTILS_H
#define LLVM_CODEGEN_BTFUTILS_H

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// Return the bit offset used to order an element of a BTF structure record.
inline uint64_t getBTFRecordElementOffset(const DINode *Element) {
  switch (Element->getTag()) {
  case dwarf::DW_TAG_member:
    return cast<DIDerivedType>(Element)->getOffsetInBits();
  case dwarf::DW_TAG_variant_part:
    return cast<DICompositeType>(Element)->getOffsetInBits();
  default:
    llvm_unreachable("Unexpected DI tag of a struct element");
  }
}

} // namespace llvm

#endif
