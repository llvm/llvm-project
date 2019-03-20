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

  // Needed because ParserImpl is an incomplete type.
  ~Parser();

  /// Returns an empty Optional if it reached the end.
  /// Returns a valid remark otherwise.
  Expected<const Remark *> getNext() const;
};

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_PARSER_H */
