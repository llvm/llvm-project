//===- Parser.cpp - Top-Level TableGen Parser implementation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Parser.h"
#include "TGParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

bool llvm::TableGenParseFile(std::unique_ptr<MemoryBuffer> Buffer,
                             std::vector<std::string> IncludeDirs,
                             TableGenParserFn ParserFn) {
  RecordKeeper Records;
  Records.saveInputFilename(Buffer->getBufferIdentifier().str());

  SrcMgr = SourceMgr();
  SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  SrcMgr.setIncludeDirs(IncludeDirs);
  TGParser Parser(SrcMgr, /*Macros=*/None, Records);
  if (Parser.ParseFile())
    return true;

  // Invoke the provided handler function.
  if (ParserFn(Records))
    return true;

  // After parsing, reset the tablegen data.
  SrcMgr = SourceMgr();
  return false;
}
