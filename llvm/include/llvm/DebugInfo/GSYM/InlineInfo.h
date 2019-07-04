//===- InlineInfo.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_INLINEINFO_H
#define LLVM_DEBUGINFO_GSYM_INLINEINFO_H

#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/GSYM/Range.h"
#include <stdint.h>
#include <vector>


namespace llvm {
class raw_ostream;

namespace gsym {

/// Inline information stores the name of the inline function along with
/// an array of address ranges. It also stores the call file and call line
/// that called this inline function. This allows us to unwind inline call
/// stacks back to the inline or concrete function that called this
/// function. Inlined functions contained in this function are stored in the
/// "Children" variable. All address ranges must be sorted and all address
/// ranges of all children must be contained in the ranges of this function.
/// Any clients that encode information will need to ensure the ranges are
/// all contined correctly or lookups could fail. Add ranges in these objects
/// must be contained in the top level FunctionInfo address ranges as well.
struct InlineInfo {

  uint32_t Name; ///< String table offset in the string table.
  uint32_t CallFile; ///< 1 based file index in the file table.
  uint32_t CallLine; ///< Source line number.
  AddressRanges Ranges;
  std::vector<InlineInfo> Children;
  InlineInfo() : Name(0), CallFile(0), CallLine(0) {}
  void clear() {
    Name = 0;
    CallFile = 0;
    CallLine = 0;
    Ranges.clear();
    Children.clear();
  }
  bool isValid() const { return !Ranges.empty(); }

  using InlineArray = std::vector<const InlineInfo *>;

  /// Lookup an address in the InlineInfo object
  ///
  /// This function is used to symbolicate an inline call stack and can
  /// turn one address in the program into one or more inline call stacks
  /// and have the stack trace show the original call site from
  /// non-inlined code.
  ///
  /// \param Addr the address to lookup
  ///
  /// \returns optional vector of InlineInfo objects that describe the
  /// inline call stack for a given address, false otherwise.
  llvm::Optional<InlineArray> getInlineStack(uint64_t Addr) const;
};

inline bool operator==(const InlineInfo &LHS, const InlineInfo &RHS) {
  return LHS.Name == RHS.Name && LHS.CallFile == RHS.CallFile &&
         LHS.CallLine == RHS.CallLine && LHS.Ranges == RHS.Ranges &&
         LHS.Children == RHS.Children;
}

raw_ostream &operator<<(raw_ostream &OS, const InlineInfo &FI);

} // namespace gsym
} // namespace llvm

#endif // #ifndef LLVM_DEBUGINFO_GSYM_INLINEINFO_H
