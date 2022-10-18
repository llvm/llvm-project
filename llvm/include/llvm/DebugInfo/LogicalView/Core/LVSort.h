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

namespace llvm {
namespace logicalview {

// Object Sorting Mode.
enum class LVSortMode {
  None = 0, // No given sort.
  Kind,     // Sort by kind.
  Line,     // Sort by line.
  Name,     // Sort by name.
  Offset    // Sort by offset.
};

} // end namespace logicalview
} // end namespace llvm

#endif // LLVM_DEBUGINFO_LOGICALVIEW_CORE_LVSORT_H
