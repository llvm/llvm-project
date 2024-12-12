/* TO_UPSTREAM(BoundsSafety) ON */
#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "lldb/Target/BoundsSafetyTrapFrameRecognizer.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

BoundsSafetyTrapRecognizedStackFrame::BoundsSafetyTrapRecognizedStackFrame(
    StackFrameSP most_relevant_frame_sp, StringRef stop_desc)
    : m_most_relevant_frame(most_relevant_frame_sp) {
  m_stop_desc = std::string(stop_desc);
}

lldb::RecognizedStackFrameSP
BoundsSafetyTrapFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame_sp) {
  if (frame_sp->GetFrameIndex())
    return {};

  ThreadSP thread_sp = frame_sp->GetThread();
  ProcessSP process_sp = thread_sp->GetProcess();

  StackFrameSP most_relevant_frame_sp = thread_sp->GetStackFrameAtIndex(1);

  if (!most_relevant_frame_sp) {
    Log *log = GetLog(LLDBLog::Unwind);
    LLDB_LOG(
        log,
        "Bounds check failed Recognizer: Hit unwinding bound (1 frame)!");
    return {};
  }

  SymbolContext sc = frame_sp->GetSymbolContext(eSymbolContextEverything);

  if (!sc.block)
    return {};

  // The runtime error is set as the function name in the inlined function info
  // of frame #0 by the compiler
  const InlineFunctionInfo *inline_info = nullptr;
  Block *inline_block = sc.block->GetContainingInlinedBlock();

  if (!inline_block)
    return {};

  inline_info = sc.block->GetInlinedFunctionInfo();

  if (!inline_info)
    return {};

  StringRef runtime_error = inline_info->GetName().AsCString();

  if (runtime_error.empty())
    return {};

  return lldb::RecognizedStackFrameSP(new BoundsSafetyTrapRecognizedStackFrame(
      most_relevant_frame_sp, runtime_error));
}

lldb::StackFrameSP BoundsSafetyTrapRecognizedStackFrame::GetMostRelevantFrame() {
  return m_most_relevant_frame;
}

namespace lldb_private {

void RegisterBoundsSafetyTrapFrameRecognizer(Process &process) {
  RegularExpressionSP module_regex_sp = nullptr;
  // TODO(dliew): rdar://90866345
  // `Pointer Check runtime failure` is the legacy trap prefix.
  RegularExpressionSP symbol_regex_sp(
      new RegularExpression("(Bounds check failed)|(Pointer Check runtime failure)"));

  StackFrameRecognizerSP srf_recognizer_sp =
      std::make_shared<BoundsSafetyTrapFrameRecognizer>();

  process.GetTarget().GetFrameRecognizerManager().AddRecognizer(
      srf_recognizer_sp, module_regex_sp, symbol_regex_sp,
      Mangled::NamePreference::ePreferDemangled, false);
}

} // namespace lldb_private
/* TO_UPSTREAM(BoundsSafety) ON */
