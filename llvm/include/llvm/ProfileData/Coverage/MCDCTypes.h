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

#include <array>
#include <variant>

namespace llvm::coverage::mcdc {

/// The ID for MCDCBranch.
using ConditionID = unsigned int;
using ConditionIDs = std::array<ConditionID, 2>;

struct DecisionParameters {
  /// Byte Index of Bitmap Coverage Object for a Decision Region.
  unsigned BitmapIdx;

  /// Number of Conditions used for a Decision Region.
  unsigned NumConditions;

  DecisionParameters() = delete;
  DecisionParameters(unsigned BitmapIdx, unsigned NumConditions)
      : BitmapIdx(BitmapIdx), NumConditions(NumConditions) {}
};

struct BranchParameters {
  /// IDs used to represent a branch region and other branch regions
  /// evaluated based on True and False branches.
  ConditionID ID;
  ConditionIDs Conds;

  BranchParameters() = delete;
  BranchParameters(ConditionID ID, const ConditionIDs &Conds)
      : ID(ID), Conds(Conds) {}
};

/// The type of MC/DC-specific parameters.
using Parameters =
    std::variant<std::monostate, DecisionParameters, BranchParameters>;

} // namespace llvm::coverage::mcdc

#endif // LLVM_PROFILEDATA_COVERAGE_MCDCTYPES_H
