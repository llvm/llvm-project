//===- Parser.cpp - MLIR Unified Parser Interface -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

LogicalResult mlir::parseSourceFile(const llvm::SourceMgr &sourceMgr,
                                    Block *block, const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  if (sourceFileLoc) {
    *sourceFileLoc = FileLineColLoc::get(config.getContext(),
                                         sourceBuf->getBufferIdentifier(),
                                         /*line=*/0, /*column=*/0);
  }
  return parseAsmSourceFile(sourceMgr, block, config);
}

LogicalResult mlir::parseSourceFile(llvm::StringRef filename, Block *block,
                                    const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  llvm::SourceMgr sourceMgr;
  return parseSourceFile(filename, sourceMgr, block, config, sourceFileLoc);
}

LogicalResult mlir::parseSourceFile(llvm::StringRef filename,
                                    llvm::SourceMgr &sourceMgr, Block *block,
                                    const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  if (sourceMgr.getNumBuffers() != 0) {
    // TODO: Extend to support multiple buffers.
    return emitError(mlir::UnknownLoc::get(config.getContext()),
                     "only main buffer parsed at the moment");
  }
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    return emitError(mlir::UnknownLoc::get(config.getContext()),
                     "could not open input file " + filename);

  // Load the MLIR source file.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}

LogicalResult mlir::parseSourceString(llvm::StringRef sourceStr, Block *block,
                                      const ParserConfig &config,
                                      LocationAttr *sourceFileLoc) {
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(sourceStr);
  if (!memBuffer)
    return failure();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}
