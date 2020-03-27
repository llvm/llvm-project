//===-- XCOFFDump.cpp - XCOFF-specific dumper -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XCOFF-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/Object/XCOFFObjectFile.h"

using namespace llvm::object;

llvm::Error llvm::getXCOFFRelocationValueString(const XCOFFObjectFile *Obj,
                                                const RelocationRef &Rel,
                                                SmallVectorImpl<char> &Result) {
  symbol_iterator SymI = Rel.getSymbol();
  if (SymI == Obj->symbol_end())
    return make_error<GenericBinaryError>(
        "invalid symbol reference in relocation entry",
        object_error::parse_failed);

  Expected<StringRef> SymNameOrErr = SymI->getName();
  if (!SymNameOrErr)
    return SymNameOrErr.takeError();
  StringRef SymName = *SymNameOrErr;
  Result.append(SymName.begin(), SymName.end());
  return Error::success();
}
