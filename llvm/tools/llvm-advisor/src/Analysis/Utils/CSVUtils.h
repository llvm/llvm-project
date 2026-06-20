//===--- CSVUtils.h - LLVM Advisor ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm::advisor {

/// Split one CSV row respecting double-quoted fields.
inline SmallVector<StringRef, 16> splitCSVRow(StringRef Row) {
  SmallVector<StringRef, 16> Fields;
  const char *P = Row.data();
  const char *End = P + Row.size();
  while (P < End) {
    if (*P == '"') {
      const char *Start = ++P;
      while (P < End && *P != '"')
        ++P;
      Fields.push_back(StringRef(Start, P - Start));
      if (P < End)
        ++P;
    } else {
      const char *Start = P;
      while (P < End && *P != ',')
        ++P;
      Fields.push_back(StringRef(Start, P - Start).trim());
    }
    if (P < End && *P == ',')
      ++P;
  }
  return Fields;
}

/// Find the column index for a header name (case-insensitive).
inline int findCol(const SmallVector<StringRef, 16> &Headers, StringRef Name) {
  for (int I = 0, E = static_cast<int>(Headers.size()); I < E; ++I)
    if (Headers[I].equals_insensitive(Name))
      return I;
  return -1;
}

} // namespace llvm::advisor
