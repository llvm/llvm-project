//===- DebugFrameDataSubsection.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/DebugFrameDataSubsection.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"

using namespace llvm;
using namespace llvm::codeview;

Error DebugFrameDataSubsectionRef::initialize(BinaryStreamReader Reader) {
  if (auto EC = Reader.readObject(RelocPtr))
    return EC;
  if (Reader.bytesRemaining() % sizeof(FrameData) != 0)
    return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                     "Invalid frame data record format!");

  uint32_t Count = Reader.bytesRemaining() / sizeof(FrameData);
  if (auto EC = Reader.readArray(Frames, Count))
    return EC;
  return Error::success();
}

uint32_t DebugFrameDataSubsection::calculateSerializedSize() const {
  return 4 + sizeof(FrameData) * Frames.size();
}

Error DebugFrameDataSubsection::commit(BinaryStreamWriter &Writer) const {
  if (auto EC = Writer.writeInteger<uint32_t>(0))
    return EC;

  if (auto EC = Writer.writeArray(makeArrayRef(Frames)))
    return EC;
  return Error::success();
}

void DebugFrameDataSubsection::addFrameData(const FrameData &Frame) {
  Frames.push_back(Frame);
}
