//===-- LVSort.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the sort algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSORT_H
#define LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSORT_H

#include "llvm/Support/Compiler.h"

namespace llvm {
namespace logicalview {

class LVObject;

// Object Sorting Mode.
enum class LVSortMode {
  None = 0, // No given sort.
  Kind,     // Sort by kind.
  Line,     // Sort by line.
  Name,     // Sort by name.
  Offset    // Sort by offset.
};

// Type of function to be called when sorting an object.
using LVSortValue = int;
using LVSortFunction = LVSortValue (*)(const LVObject *LHS,
                                       const LVObject *RHS);

// Get the comparator function, based on the command line options.
LLVM_ABI LVSortFunction getSortFunction();

// Comparator functions that can be used for sorting.
LLVM_ABI LVSortValue compareKind(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue compareLine(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue compareName(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue compareOffset(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue compareRange(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue sortByKind(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue sortByLine(const LVObject *LHS, const LVObject *RHS);
LLVM_ABI LVSortValue sortByName(const LVObject *LHS, const LVObject *RHS);

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSORT_H
