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

/// A DataExtractor subclass that adds GSYM-specific string offset support.
///
/// GSYM files use variable-width string offsets (1-8 bytes). This subclass
/// adds getStringOffsetSize() and getStringOffset() methods to support reading
/// string offsets of the configured size.
class GsymDataExtractor : public DataExtractor {
  uint8_t StringOffsetSize;

public:
  /// Construct from raw bytes.
  GsymDataExtractor(StringRef Data, bool IsLittleEndian,
                    uint8_t StringOffsetSize = 8)
      : DataExtractor(Data, IsLittleEndian),
        StringOffsetSize(StringOffsetSize) {}

  /// Construct a sub-range extractor from a parent, copying its endianness
  /// and string offset size.
  GsymDataExtractor(const GsymDataExtractor &Parent, uint64_t Offset,
                    uint64_t Length)
      : DataExtractor(Parent.getData().substr(Offset, Length),
                      Parent.isLittleEndian()),
        StringOffsetSize(Parent.getStringOffsetSize()) {}

  /// Get the string offset size in bytes.
  uint8_t getStringOffsetSize() const { return StringOffsetSize; }

  /// Extract a string offset of StringOffsetSize bytes from \a *offset_ptr.
  uint64_t getStringOffset(uint64_t *offset_ptr) const {
    return getUnsigned(offset_ptr, StringOffsetSize);
  }

  /// Extract a string offset of StringOffsetSize bytes from the location given
  /// by the cursor.
  uint64_t getStringOffset(Cursor &C) const {
    return getUnsigned(C, StringOffsetSize);
  }
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMDATAEXTRACTOR_H
