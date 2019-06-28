//===- FunctionInfo.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_FUNCTIONINFO_H
#define LLVM_DEBUGINFO_GSYM_FUNCTIONINFO_H

#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/LineEntry.h"
#include "llvm/DebugInfo/GSYM/Range.h"
#include "llvm/DebugInfo/GSYM/StringTable.h"
#include <tuple>
#include <vector>

namespace llvm {
class raw_ostream;
namespace gsym {

/// Function information in GSYM files encodes information for one
/// contiguous address range. The name of the function is encoded as
/// a string table offset and allows multiple functions with the same
/// name to share the name string in the string table. Line tables are
/// stored in a sorted vector of gsym::LineEntry objects and are split
/// into line tables for each function. If a function has a discontiguous
/// range, it will be split into two gsym::FunctionInfo objects. If the
/// function has inline functions, the information will be encoded in
/// the "Inline" member, see gsym::InlineInfo for more information.
struct FunctionInfo {
  AddressRange Range;
  uint32_t Name; ///< String table offset in the string table.
  std::vector<gsym::LineEntry> Lines;
  InlineInfo Inline;

  FunctionInfo(uint64_t Addr = 0, uint64_t Size = 0, uint32_t N = 0)
      : Range(Addr, Addr + Size), Name(N) {}

  bool hasRichInfo() const {
    /// Returns whether we have something else than range and name. When
    /// converting information from a symbol table and from debug info, we
    /// might end up with multiple FunctionInfo objects for the same range
    /// and we need to be able to tell which one is the better object to use.
    return !Lines.empty() || Inline.isValid();
  }

  bool isValid() const {
    /// Address and size can be zero and there can be no line entries for a
    /// symbol so the only indication this entry is valid is if the name is
    /// not zero. This can happen when extracting information from symbol
    /// tables that do not encode symbol sizes. In that case only the
    /// address and name will be filled in.
    return Name != 0;
  }

  uint64_t startAddress() const { return Range.Start; }
  uint64_t endAddress() const { return Range.End; }
  uint64_t size() const { return Range.size(); }
  void setStartAddress(uint64_t Addr) { Range.Start = Addr; }
  void setEndAddress(uint64_t Addr) { Range.End = Addr; }
  void setSize(uint64_t Size) { Range.End = Range.Start + Size; }

  void clear() {
    Range = {0, 0};
    Name = 0;
    Lines.clear();
    Inline.clear();
  }
};

inline bool operator==(const FunctionInfo &LHS, const FunctionInfo &RHS) {
  return LHS.Range == RHS.Range && LHS.Name == RHS.Name &&
         LHS.Lines == RHS.Lines && LHS.Inline == RHS.Inline;
}
inline bool operator!=(const FunctionInfo &LHS, const FunctionInfo &RHS) {
  return !(LHS == RHS);
}
/// This sorting will order things consistently by address range first, but then
/// followed by inlining being valid and line tables. We might end up with a
/// FunctionInfo from debug info that will have the same range as one from the
/// symbol table, but we want to quickly be able to sort and use the best version
/// when creating the final GSYM file.
inline bool operator<(const FunctionInfo &LHS, const FunctionInfo &RHS) {
  // First sort by address range
  if (LHS.Range != RHS.Range)
    return LHS.Range < RHS.Range;

  // Then sort by inline
  if (LHS.Inline.isValid() != RHS.Inline.isValid())
    return RHS.Inline.isValid();

  // If the number of lines is the same, then compare line table entries
  if (LHS.Lines.size() == RHS.Lines.size())
    return LHS.Lines < RHS.Lines;
  // Then sort by number of line table entries (more is better)
  return LHS.Lines.size() < RHS.Lines.size();
}

raw_ostream &operator<<(raw_ostream &OS, const FunctionInfo &R);

} // namespace gsym
} // namespace llvm

#endif // #ifndef LLVM_DEBUGINFO_GSYM_FUNCTIONINFO_H
