//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSVCRTCFrameRecognizer.h"

#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

void RegisterMSVCRTCFrameRecognizer(ProcessWindows &process) {
  process.GetTarget().GetFrameRecognizerManager().AddRecognizer(
      std::make_shared<MSVCRTCFrameRecognizer>(), ConstString(""),
      {ConstString("failwithmessage")}, Mangled::ePreferDemangled,
      /*first_instruction_only=*/false);
}

lldb::RecognizedStackFrameSP
MSVCRTCFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame_sp) {
  // failwithmessage calls __debugbreak() which lands at frame 0.
  if (frame_sp->GetFrameIndex() != 0)
    return RecognizedStackFrameSP();
  // Only fire on EXCEPTION_BREAKPOINT (0x80000003), not on other exceptions
  // that might incidentally have failwithmessage somewhere in the call stack.
  auto *pw =
      static_cast<ProcessWindows *>(frame_sp->GetThread()->GetProcess().get());
  auto exc_code = pw->GetActiveExceptionCode();
  if (!exc_code || *exc_code != EXCEPTION_BREAKPOINT)
    return RecognizedStackFrameSP();

  const char *fn_name = frame_sp->GetFunctionName();
  if (!fn_name)
    return RecognizedStackFrameSP();
  if (!llvm::StringRef(fn_name).contains("failwithmessage"))
    return RecognizedStackFrameSP();

  VariableListSP vars = frame_sp->GetInScopeVariableList(false);
  if (!vars)
    return RecognizedStackFrameSP();

  for (size_t i = 0; i < vars->GetSize(); ++i) {
    VariableSP var = vars->GetVariableAtIndex(i);
    if (!var || var->GetName() != ConstString("msg"))
      continue;

    ValueObjectSP val =
        frame_sp->GetValueObjectForFrameVariable(var, eNoDynamicValues);
    if (!val)
      break;

    uint64_t ptr = val->GetValueAsUnsigned(0);
    if (!ptr)
      break;

    std::string msg;
    Status err;
    frame_sp->GetThread()->GetProcess()->ReadCStringFromMemory(ptr, msg, err);
    if (err.Success() && !msg.empty())
      return lldb::RecognizedStackFrameSP(
          new MSVCRTCRecognizedFrame("Run-time check failure: " + msg));
    break;
  }

  return RecognizedStackFrameSP();
}

} // namespace lldb_private
