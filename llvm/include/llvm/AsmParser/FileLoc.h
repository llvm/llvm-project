//===-- FileLoc.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASMPARSER_FILELOC_H
#define LLVM_ASMPARSER_FILELOC_H

#include <cassert>
#include <utility>

namespace llvm {

/// Struct holding Line:Column location
struct FileLoc {
  /// 0-based line number
  unsigned Line;
  /// 0-based column number
  unsigned Col;

  bool operator<=(const FileLoc &RHS) const {
    return Line < RHS.Line || (Line == RHS.Line && Col <= RHS.Col);
  }

  bool operator<(const FileLoc &RHS) const {
    return Line < RHS.Line || (Line == RHS.Line && Col < RHS.Col);
  }

  FileLoc(unsigned L, unsigned C) : Line(L), Col(C) {}
  FileLoc(std::pair<unsigned, unsigned> LC) : Line(LC.first), Col(LC.second) {}
};

/// Struct holding a semiopen range [Start; End)
struct FileLocRange {
  FileLoc Start;
  FileLoc End;

  FileLocRange() : Start(0, 0), End(0, 0) {}

  FileLocRange(FileLoc S, FileLoc E) : Start(S), End(E) {
    assert(Start <= End);
  }

  bool contains(FileLoc L) const { return Start <= L && L < End; }

  bool contains(FileLocRange LR) const {
    return Start <= LR.Start && LR.End <= End;
  }
};

} // namespace llvm

#endif
