//===- GsymDataExtractor.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMDATAEXTRACTOR_H
#define LLVM_DEBUGINFO_GSYM_GSYMDATAEXTRACTOR_H

#include "llvm/Support/DataExtractor.h"

namespace llvm {
namespace gsym {
class GsymReader;

/// A DataExtractor subclass with GSYM-specific functionality.
///
/// Extends DataExtractor with the ability to read string table offsets (strp)
/// of variable byte size, as determined by the GSYM header's StrpSize field.
class GsymDataExtractor : public DataExtractor {
  uint8_t StrpSize = 4;

public:
  /// Construct from an existing DataExtractor with an optional GsymReader.
  GsymDataExtractor(const DataExtractor &DE, const GsymReader *GR = nullptr);

  /// Construct a sub-extractor from a parent, inheriting StrpSize.
  GsymDataExtractor(StringRef Data, const GsymDataExtractor &Parent)
      : DataExtractor(Data, Parent.isLittleEndian(), Parent.getAddressSize()),
        StrpSize(Parent.StrpSize) {}

  /// Get the string table offset byte size.
  uint8_t getStrpSize() const { return StrpSize; }

  /// Read a string table offset (strp) of StrpSize bytes from the given
  /// offset, and update the offset past the read data.
  ///
  /// \param OffsetPtr A pointer to the offset to read from. Updated on
  /// success.
  /// \returns The string table offset value.
  uint64_t getStrp(uint64_t *OffsetPtr) const {
    return getUnsigned(OffsetPtr, StrpSize);
  }
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMDATAEXTRACTOR_H
