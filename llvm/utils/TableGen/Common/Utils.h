//===- Utils.h - Common Utilities -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_UTILS_H
#define LLVM_UTILS_TABLEGEN_COMMON_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Record;

/// Sort an array of Records on the "Name" field, and check for records with
/// duplicate "Name" field. If duplicates are found, report a fatal error.
void sortAndReportDuplicates(MutableArrayRef<Record *> Records,
                             StringRef ObjectName);

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_UTILS_H
