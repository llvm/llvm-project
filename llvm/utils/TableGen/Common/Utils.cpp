//===- Utils.cpp - Common Utilities -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>

using namespace llvm;

namespace {
/// Sorting predicate to sort record pointers by their Name field, and break
/// ties using record ID (which corresponds to creation/parse order).
struct LessRecordFieldNameAndID {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return std::tuple(Rec1->getValueAsString("Name"), Rec1->getID()) <
           std::tuple(Rec2->getValueAsString("Name"), Rec2->getID());
  }
};
} // End anonymous namespace

/// Sort an array of Records on the "Name" field, and check for records with
/// duplicate "Name" field. If duplicates are found, report a fatal error.
void llvm::sortAndReportDuplicates(MutableArrayRef<Record *> Records,
                                   StringRef ObjectName) {
  llvm::sort(Records, LessRecordFieldNameAndID());

  auto I = std::adjacent_find(Records.begin(), Records.end(),
                              [](const Record *Rec1, const Record *Rec2) {
                                return Rec1->getValueAsString("Name") ==
                                       Rec2->getValueAsString("Name");
                              });
  if (I == Records.end())
    return;

  // Found a duplicate name.
  const Record *First = *I;
  const Record *Second = *(I + 1);
  StringRef Name = First->getValueAsString("Name");
  PrintError(Second, ObjectName + " `" + Name + "` is already defined.");
  PrintFatalNote(First, "Previous definition here.");
}
