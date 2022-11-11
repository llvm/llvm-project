//===-- SwiftStringIndex.h --------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftStringIndex_h
#define liblldb_SwiftStringIndex_h

#include "lldb/lldb-forward.h"

namespace lldb_private {
namespace formatters {
namespace swift {

// From SwiftIndex.swift
//  ┌──────────┬────────────────╥────────────────┬───────╥───────┐
//  │ b63:b16  │      b15:b14   ║     b13:b8     │ b7:b4 ║ b3:b0 │
//  ├──────────┼────────────────╫────────────────┼───────╫───────┤
//  │ position │ transc. offset ║ grapheme cache │ rsvd  ║ flags │
//  └──────────┴────────────────╨────────────────┴───────╨───────┘
//                              └────── resilient ───────┘
class StringIndex {
  uint64_t _rawBits;

  enum Flags : uint8_t {
    IsScalarAligned = 1 << 0,
    IsCharacterAligned = 1 << 1,
    CanBeUTF8 = 1 << 2,
    CanBeUTF16 = 1 << 3,
  };

public:
  StringIndex(uint64_t rawBits) : _rawBits(rawBits) {}

  uint64_t encodedOffset() { return _rawBits >> 16; }

  const char *encodingName() {
    auto flags = _flags();
    bool canBeUTF8 = flags & Flags::CanBeUTF8;
    bool canBeUTF16 = flags & Flags::CanBeUTF16;
    if (canBeUTF8 && canBeUTF16)
      return "any";
    else if (canBeUTF8)
      return "utf8";
    else if (canBeUTF16)
      return "utf16";
    else
      return "unknown";
  }

  uint8_t transcodedOffset() { return (_rawBits >> 14) & 0b11; }

  bool matchesEncoding(StringIndex other) {
    // Either both are valid utf8 indexes, or valid utf16 indexes.
    return (_utfFlags() & other._utfFlags()) != 0;
  }

private:
  uint8_t _flags() const { return _rawBits & 0b1111; }

  uint8_t _utfFlags() const {
    return _flags() & (Flags::CanBeUTF8 | Flags::CanBeUTF16);
  }
};

}; // namespace swift
}; // namespace formatters
}; // namespace lldb_private

#endif
