//===-- llvm/Remarks/Remark.h - The remark type -----------------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an interface for parsing remarks in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_REMARK_PARSER_H
#define LLVM_REMARKS_REMARK_PARSER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Remarks/Remark.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
namespace remarks {

struct ParserImpl;

/// Parser used to parse a raw buffer to remarks::Remark objects.
struct Parser {
  /// The hidden implementation of the parser.
  std::unique_ptr<ParserImpl> Impl;

  /// Create a parser parsing \p Buffer to Remark objects.
  /// This constructor should be only used for parsing YAML remarks.
  Parser(StringRef Buffer);

  /// Create a parser parsing \p Buffer to Remark objects, using \p StrTabBuf as
  /// string table.
  /// This constructor should be only used for parsing YAML remarks.
  Parser(StringRef Buffer, StringRef StrTabBuf);

  // Needed because ParserImpl is an incomplete type.
  ~Parser();

  /// Returns an empty Optional if it reached the end.
  /// Returns a valid remark otherwise.
  Expected<const Remark *> getNext() const;
};

/// In-memory representation of the string table parsed from a buffer (e.g. the
/// remarks section).
struct ParsedStringTable {
  /// The buffer mapped from the section contents.
  StringRef Buffer;
  /// Collection of offsets in the buffer for each string entry.
  SmallVector<size_t, 8> Offsets;

  Expected<StringRef> operator[](size_t Index);
  ParsedStringTable(StringRef Buffer);
};

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_PARSER_H */
