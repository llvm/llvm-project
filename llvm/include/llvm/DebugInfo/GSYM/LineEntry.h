//===- LineEntry.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_LINEENTRY_H
#define LLVM_DEBUGINFO_GSYM_LINEENTRY_H

#include "llvm/DebugInfo/GSYM/Range.h"

namespace llvm {
namespace gsym {

/// Line entries are used to encode the line tables in FunctionInfo objects.
/// They are stored as a sorted vector of these objects and store the
/// address, file and line of the line table row for a given address. The
/// size of a line table entry is calculated by looking at the next entry
/// in the FunctionInfo's vector of entries.
struct LineEntry {
  uint64_t Addr; ///< Start address of this line entry.
  uint32_t File; ///< 1 based index of file in FileTable
  uint32_t Line; ///< Source line number.
  LineEntry(uint64_t A = 0, uint32_t F = 0, uint32_t L = 0)
      : Addr(A), File(F), Line(L) {}
  bool isValid() { return File != 0; }
};

inline raw_ostream &operator<<(raw_ostream &OS, const LineEntry &LE) {
  return OS << "addr=" << HEX64(LE.Addr) << ", file=" << format("%3u", LE.File)
      << ", line=" << format("%3u", LE.Line);
}

inline bool operator==(const LineEntry &LHS, const LineEntry &RHS) {
  return LHS.Addr == RHS.Addr && LHS.File == RHS.File && LHS.Line == RHS.Line;
}
inline bool operator!=(const LineEntry &LHS, const LineEntry &RHS) {
  return !(LHS == RHS);
}
inline bool operator<(const LineEntry &LHS, const LineEntry &RHS) {
  return LHS.Addr < RHS.Addr;
}
} // namespace gsym
} // namespace llvm
#endif // #ifndef LLVM_DEBUGINFO_GSYM_LINEENTRY_H
