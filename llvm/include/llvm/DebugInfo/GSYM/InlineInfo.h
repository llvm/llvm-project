//===- InlineInfo.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_INLINEINFO_H
#define LLVM_DEBUGINFO_GSYM_INLINEINFO_H

#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/GSYM/Range.h"
#include "llvm/Support/Error.h"
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
///
/// ENCODING
///
/// When saved to disk, the inline info encodes all ranges to be relative to
/// a parent address range. This will be the FunctionInfo's start address if
/// the InlineInfo is directly contained in a FunctionInfo, or a the start
/// address of the containing parent InlineInfo's first "Ranges" member. This
/// allows address ranges to be efficiently encoded using ULEB128 encodings as
/// we encode the offset and size of each range instead of full addresses. This
/// also makes any encoded addresses easy to relocate as we just need to
/// relocate the FunctionInfo's start address.
///
/// - The AddressRanges member "Ranges" is encoded using an approriate base
///   address as described above.
/// - UINT8 boolean value that specifies if the InlineInfo object has children.
/// - UINT32 string table offset that points to the name of the inline
///   function.
/// - ULEB128 integer that specifies the file of the call site that called
///   this function.
/// - ULEB128 integer that specifies the source line of the call site that
///   called this function.
/// - if this object has children, enocode each child InlineInfo using the
///   the first address range's start address as the base address.
///
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

  /// Decode an InlineInfo object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from. This object must
  /// have the data for the InlineInfo object starting at offset zero. The data
  /// can contain more data than needed.
  ///
  /// \param BaseAddr The base address to use when decoding all address ranges.
  /// This will be the FunctionInfo's start address if this object is directly
  /// contained in a FunctionInfo object, or the start address of the first
  /// address range in an InlineInfo object of this object is a child of
  /// another InlineInfo object.
  /// \returns An InlineInfo or an error describing the issue that was
  /// encountered during decoding.
  static llvm::Expected<InlineInfo> decode(DataExtractor &Data,
                                           uint64_t BaseAddr);

  /// Encode this InlineInfo object into FileWriter stream.
  ///
  /// \param O The binary stream to write the data to at the current file
  /// position.
  ///
  /// \param BaseAddr The base address to use when encoding all address ranges.
  /// This will be the FunctionInfo's start address if this object is directly
  /// contained in a FunctionInfo object, or the start address of the first
  /// address range in an InlineInfo object of this object is a child of
  /// another InlineInfo object.
  ///
  /// \returns An error object that indicates success or failure or the
  /// encoding process.
  llvm::Error encode(FileWriter &O, uint64_t BaseAddr) const;
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
