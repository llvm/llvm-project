//===- llvm/TableGen/Parser.h - tblgen parser entry point -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an entry point into the tablegen parser for use by tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_PARSER_H
#define LLVM_TABLEGEN_PARSER_H

#include "llvm/ADT/STLExtras.h"
#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
class RecordKeeper;

/// Peform the tablegen action using the given set of parsed records. Returns
/// true on error, false otherwise.
using TableGenParserFn = function_ref<bool(RecordKeeper &)>;

/// Parse the given input buffer containing a tablegen file, invoking the
/// provided parser function with the set of parsed records. All tablegen state
/// is reset after the provided parser function is invoked, i.e., the provided
/// parser function should not maintain references to any tablegen constructs
/// after executing. Returns true on failure, false otherwise.
bool TableGenParseFile(std::unique_ptr<MemoryBuffer> Buffer,
                       std::vector<std::string> IncludeDirs,
                       TableGenParserFn ParserFn);

} // end namespace llvm

#endif // LLVM_TABLEGEN_PARSER_H
