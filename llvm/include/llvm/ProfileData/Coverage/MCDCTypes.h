//===- MCDCTypes.h - Types related to MC/DC Coverage ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types related to MC/DC Coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_COVERAGE_MCDCTYPES_H
#define LLVM_PROFILEDATA_COVERAGE_MCDCTYPES_H

namespace llvm::coverage::mcdc {

/// The ID for MCDCBranch.
using ConditionID = unsigned int;

/// MC/DC-specifig parameters
struct Parameters {
  /// Byte Index of Bitmap Coverage Object for a Decision Region.
  unsigned BitmapIdx = 0;

  /// Number of Conditions used for a Decision Region.
  unsigned NumConditions = 0;

  /// IDs used to represent a branch region and other branch regions
  /// evaluated based on True and False branches.
  ConditionID ID = 0, TrueID = 0, FalseID = 0;
};

} // namespace llvm::coverage::mcdc

#endif // LLVM_PROFILEDATA_COVERAGE_MCDCTYPES_H
