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
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
namespace remarks {

struct ParserImpl;
struct ParsedStringTable;

class EndOfFileError : public ErrorInfo<EndOfFileError> {
public:
  static char ID;

  EndOfFileError() {}

  void log(raw_ostream &OS) const override { OS << "End of file reached."; }
  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }
};

/// Parser used to parse a raw buffer to remarks::Remark objects.
struct Parser {
  /// The format of the parser.
  Format ParserFormat;

  Parser(Format ParserFormat) : ParserFormat(ParserFormat) {}

  /// If no error occurs, this returns a valid Remark object.
  /// If an error of type EndOfFileError occurs, it is safe to recover from it
  /// by stopping the parsing.
  /// If any other error occurs, it should be propagated to the user.
  /// The pointer should never be null.
  virtual Expected<std::unique_ptr<Remark>> next() = 0;

  virtual ~Parser() = default;
};

/// In-memory representation of the string table parsed from a buffer (e.g. the
/// remarks section).
struct ParsedStringTable {
  /// The buffer mapped from the section contents.
  StringRef Buffer;
  /// Collection of offsets in the buffer for each string entry.
  SmallVector<size_t, 8> Offsets;

  Expected<StringRef> operator[](size_t Index) const;
  ParsedStringTable(StringRef Buffer);
};

Expected<std::unique_ptr<Parser>>
createRemarkParser(Format ParserFormat, StringRef Buf,
                   Optional<const ParsedStringTable *> StrTab = None);

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_PARSER_H */
