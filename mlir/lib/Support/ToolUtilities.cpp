//===- ToolUtilities.cpp - MLIR Tool Utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for implementing MLIR tools.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/ToolUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

LogicalResult
mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> originalBuffer,
                            ChunkBufferHandler processChunkBuffer,
                            raw_ostream &os) {
  const char splitMarkerConst[] = "// -----";
  StringRef splitMarker(splitMarkerConst);
  const int splitMarkerLen = splitMarker.size();

  auto *origMemBuffer = originalBuffer.get();
  SmallVector<StringRef, 8> rawSourceBuffers;
  const int checkLen = 2;
  // Split dropping the last checkLen chars to enable flagging near misses.
  origMemBuffer->getBuffer().split(rawSourceBuffers,
                                   splitMarker.drop_back(checkLen));
  if (rawSourceBuffers.empty())
    return success();

  // Add the original buffer to the source manager.
  llvm::SourceMgr fileSourceMgr;
  fileSourceMgr.AddNewSourceBuffer(std::move(originalBuffer), SMLoc());

  // Flag near misses by iterating over all the sub-buffers found when splitting
  // with the prefix of the splitMarker. Use a sliding window where we only add
  // a buffer as a sourceBuffer if terminated by a full match of the
  // splitMarker, else flag a warning (if near miss) and extend the size of the
  // buffer under consideration.
  SmallVector<StringRef, 8> sourceBuffers;
  StringRef prev;
  for (auto buffer : rawSourceBuffers) {
    if (prev.empty()) {
      prev = buffer;
      continue;
    }

    // Check that suffix is as expected and doesn't have any dash post.
    bool expectedSuffix = buffer.startswith(splitMarker.take_back(checkLen)) &&
                          buffer.size() > checkLen && buffer[checkLen] != '0';
    if (expectedSuffix) {
      sourceBuffers.push_back(prev);
      prev = buffer.drop_front(checkLen);
    } else {
      // TODO: Consider making this a failure.
      auto splitLoc = SMLoc::getFromPointer(buffer.data());
      fileSourceMgr.PrintMessage(llvm::errs(), splitLoc,
                                 llvm::SourceMgr::DK_Warning,
                                 "near miss with file split marker");
      prev = StringRef(prev.data(),
                       prev.size() + splitMarkerLen - checkLen + buffer.size());
    }
  }
  if (!prev.empty())
    sourceBuffers.push_back(prev);

  // Process each chunk in turn.
  bool hadFailure = false;
  for (auto &subBuffer : sourceBuffers) {
    auto splitLoc = SMLoc::getFromPointer(subBuffer.data());
    unsigned splitLine = fileSourceMgr.getLineAndColumn(splitLoc).first;
    auto subMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(
        subBuffer, Twine("within split at ") +
                       origMemBuffer->getBufferIdentifier() + ":" +
                       Twine(splitLine) + " offset ");
    if (failed(processChunkBuffer(std::move(subMemBuffer), os)))
      hadFailure = true;
  }

  // If any fails, then return a failure of the tool.
  return failure(hadFailure);
}
