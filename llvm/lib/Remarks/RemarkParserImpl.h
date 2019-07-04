//===-- RemarkParserImpl.h - Implementation details -------------*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides implementation details for the remark parser.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_REMARK_PARSER_IMPL_H
#define LLVM_REMARKS_REMARK_PARSER_IMPL_H

#include "llvm/Remarks/RemarkParser.h"

namespace llvm {
namespace remarks {
/// This is used as a base for any parser implementation.
struct ParserImpl {
  explicit ParserImpl(ParserFormat Format) : Format(Format) {}
  // Virtual destructor prevents mismatched deletes
  virtual ~ParserImpl() {}

  // The parser format. This is used as a tag to safely cast between
  // implementations.
  ParserFormat Format;
};
} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_REMARKS_REMARK_PARSER_IMPL_H */
