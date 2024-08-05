//===-- AbortWithPayloadFrameRecognizer.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_MACOSX_ABORTWITHPAYLOADFRAMERECOGNIZER_H
#define LLDB_MACOSX_ABORTWITHPAYLOADFRAMERECOGNIZER_H

#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/FileSpec.h"

#include <tuple>

namespace lldb_private {

void RegisterAbortWithPayloadFrameRecognizer(Process *process);

class AbortWithPayloadRecognizedStackFrame : public RecognizedStackFrame {
public:
  AbortWithPayloadRecognizedStackFrame(lldb::StackFrameSP &frame_sp,
                                       lldb::ValueObjectListSP &args_sp);
};

class AbortWithPayloadFrameRecognizer : public StackFrameRecognizer {
public:
  std::string GetName() override {
    return "abort_with_payload StackFrame Recognizer";
  }
  lldb::RecognizedStackFrameSP
  RecognizeFrame(lldb::StackFrameSP frame_sp) override;
};
} // namespace lldb_private

#endif // LLDB_MACOSX_ABORTWITHPAYLOADFRAMERECOGNIZER_H
