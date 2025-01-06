//===- MergedFunctionsInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_MERGEDFUNCTIONSINFO_H
#define LLVM_DEBUGINFO_GSYM_MERGEDFUNCTIONSINFO_H

#include "llvm/DebugInfo/GSYM/ExtractRanges.h"
#include "llvm/Support/Error.h"
#include <stdint.h>
#include <vector>

namespace llvm {
class raw_ostream;

namespace gsym {

class GsymReader;
struct FunctionInfo;
struct MergedFunctionsInfo {
  std::vector<FunctionInfo> MergedFunctions;

  void clear();

  /// Query if a MergedFunctionsInfo object is valid.
  ///
  /// \returns A boolean indicating if this FunctionInfo is valid.
  bool isValid() { return !MergedFunctions.empty(); }

  /// Decode an MergedFunctionsInfo object from a binary data stream.
  ///
  /// \param Data The binary stream to read the data from. This object must have
  /// the data for the MergedFunctionsInfo object starting at offset zero. The
  /// data can contain more data than needed.
  ///
  /// \param BaseAddr The base address to use when encoding all address ranges.
  ///
  /// \returns An MergedFunctionsInfo or an error describing the issue that was
  /// encountered during decoding.
  static llvm::Expected<MergedFunctionsInfo> decode(DataExtractor &Data,
                                                    uint64_t BaseAddr);

  /// Encode this MergedFunctionsInfo object into FileWriter stream.
  ///
  /// \param O The binary stream to write the data to at the current file
  /// position.
  /// \returns An error object that indicates success or failure for the
  /// encoding process.
  llvm::Error encode(FileWriter &O) const;
};

bool operator==(const MergedFunctionsInfo &LHS, const MergedFunctionsInfo &RHS);

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_MERGEDFUNCTIONSINFO_H
