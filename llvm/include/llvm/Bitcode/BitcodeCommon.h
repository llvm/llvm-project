//===- BitcodeCommon.h - Common code for encode/decode   --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines common code to be used by BitcodeWriter and
// BitcodeReader.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_BITCODECOMMON_H
#define LLVM_BITCODE_BITCODECOMMON_H

#include "llvm/ADT/Bitfields.h"

namespace llvm {

struct AllocaPackedValues {
  using Align = Bitfield::Element<unsigned, 0, 5>;
  using UsedWithInAlloca = Bitfield::Element<bool, Align::NextBit, 1>;
  using ExplicitType = Bitfield::Element<bool, UsedWithInAlloca::NextBit, 1>;
  using SwiftError = Bitfield::Element<bool, ExplicitType::NextBit, 1>;
};

} // namespace llvm

#endif // LLVM_BITCODE_BITCODECOMMON_H
