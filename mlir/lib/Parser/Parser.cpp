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
#include "mlir/Bytecode/BytecodeReader.h"
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
  if (isBytecode(*sourceBuf))
    return readBytecodeFile(*sourceBuf, block, config);
  return parseAsmSourceFile(sourceMgr, block, config);
}
LogicalResult
mlir::parseSourceFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                      Block *block, const ParserConfig &config,
                      LocationAttr *sourceFileLoc) {
  const auto *sourceBuf =
      sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());
  if (sourceFileLoc) {
    *sourceFileLoc = FileLineColLoc::get(config.getContext(),
                                         sourceBuf->getBufferIdentifier(),
                                         /*line=*/0, /*column=*/0);
  }
  if (isBytecode(*sourceBuf))
    return readBytecodeFile(sourceMgr, block, config);
  return parseAsmSourceFile(*sourceMgr, block, config);
}

LogicalResult mlir::parseSourceFile(llvm::StringRef filename, Block *block,
                                    const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  return parseSourceFile(filename, sourceMgr, block, config, sourceFileLoc);
}

static LogicalResult loadSourceFileBuffer(llvm::StringRef filename,
                                          llvm::SourceMgr &sourceMgr,
                                          MLIRContext *ctx) {
  if (sourceMgr.getNumBuffers() != 0) {
    // TODO: Extend to support multiple buffers.
    return emitError(mlir::UnknownLoc::get(ctx),
                     "only main buffer parsed at the moment");
  }
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (fileOrErr.getError())
    return emitError(mlir::UnknownLoc::get(ctx),
                     "could not open input file " + filename);

  // Load the MLIR source file.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  return success();
}

LogicalResult mlir::parseSourceFile(llvm::StringRef filename,
                                    llvm::SourceMgr &sourceMgr, Block *block,
                                    const ParserConfig &config,
                                    LocationAttr *sourceFileLoc) {
  if (failed(loadSourceFileBuffer(filename, sourceMgr, config.getContext())))
    return failure();
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}
LogicalResult mlir::parseSourceFile(
    llvm::StringRef filename, const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
    Block *block, const ParserConfig &config, LocationAttr *sourceFileLoc) {
  if (failed(loadSourceFileBuffer(filename, *sourceMgr, config.getContext())))
    return failure();
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}

LogicalResult mlir::parseSourceString(llvm::StringRef sourceStr, Block *block,
                                      const ParserConfig &config,
                                      StringRef sourceName,
                                      LocationAttr *sourceFileLoc) {
  auto memBuffer =
      llvm::MemoryBuffer::getMemBuffer(sourceStr, sourceName,
                                       /*RequiresNullTerminator=*/false);
  if (!memBuffer)
    return failure();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  return parseSourceFile(sourceMgr, block, config, sourceFileLoc);
}
