#include "SwiftRuntimeFailureRecognizer.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "swift/Strings.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

SwiftRuntimeFailureRecognizedStackFrame::
    SwiftRuntimeFailureRecognizedStackFrame(StackFrameSP most_relevant_frame_sp,
                                            StringRef stop_desc)
    : m_most_relevant_frame(most_relevant_frame_sp) {
  m_stop_desc = std::string(stop_desc);
}

lldb::StackFrameSP
SwiftRuntimeFailureRecognizedStackFrame::GetMostRelevantFrame() {
  return m_most_relevant_frame;
}

lldb::RecognizedStackFrameSP SwiftRuntimeFailureFrameRecognizer::RecognizeFrame(
    lldb::StackFrameSP frame_sp) {
  if (frame_sp->GetFrameIndex())
    return {};

  ThreadSP thread_sp = frame_sp->GetThread();
  if (!thread_sp)
    return {};
  ProcessSP process_sp = thread_sp->GetProcess();

  StackFrameSP most_relevant_frame_sp = thread_sp->GetStackFrameAtIndex(1);

  if (!most_relevant_frame_sp) {
    Log *log = GetLog(LLDBLog::Unwind);
    LLDB_LOG(
        log,
        "Swift Runtime Failure Recognizer: Hit unwinding bound (1 frame)!");
    return {};
  }

  SymbolContext sc = frame_sp->GetSymbolContext(eSymbolContextEverything);

  if (!sc.block)
    return {};

  // The runtime error is set as the function name in the inlined function info
  // of frame #0 by the compiler (https://github.com/apple/swift/pull/29506)
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

  return lldb::RecognizedStackFrameSP(
      new SwiftRuntimeFailureRecognizedStackFrame(most_relevant_frame_sp,
                                                  runtime_error));
}

lldb::RecognizedStackFrameSP
SwiftRuntimeInstrumentedFrameRecognizer::RecognizeFrame(
    lldb::StackFrameSP frame_sp) {
  if (frame_sp->GetFrameIndex())
    return {};

  ThreadSP thread_sp = frame_sp->GetThread();
  if (!thread_sp)
    return {};

  StackFrameSP most_relevant_frame_sp;
  // Unwind until we leave the standard library.
  unsigned max_depth = 16;
  for (unsigned i = 1; i < max_depth; ++i) {
    most_relevant_frame_sp = thread_sp->GetStackFrameAtIndex(i);
    if (!most_relevant_frame_sp) {
      Log *log = GetLog(LLDBLog::Unwind);
      LLDB_LOG(log,
               "Swift Runtime Instrumentation Failure Recognizer: Hit "
               "unwinding bound ({0} frames)!",
               i);
      return {};
    }
    auto &sc =
        most_relevant_frame_sp->GetSymbolContext(lldb::eSymbolContextFunction);
    ConstString module_name = TypeSystemSwiftTypeRef::GetSwiftModuleFor(&sc);
    if (!module_name)
      continue;
    if (module_name == swift::STDLIB_NAME)
      continue;
    if (i + 1 == max_depth)
      return {};

    break;
  }

  std::string runtime_error = thread_sp->GetStopDescriptionRaw();
  return lldb::RecognizedStackFrameSP(
      new SwiftRuntimeFailureRecognizedStackFrame(most_relevant_frame_sp,
                                                  runtime_error));
}

namespace lldb_private {

void RegisterSwiftRuntimeFailureRecognizer(Process &process) {
  RegularExpressionSP module_regex_sp = nullptr;
  {
    auto symbol_regex_sp =
        std::make_shared<RegularExpression>("Swift runtime failure");

    StackFrameRecognizerSP srf_recognizer_sp =
        std::make_shared<SwiftRuntimeFailureFrameRecognizer>();

    process.GetTarget().GetFrameRecognizerManager().AddRecognizer(
        srf_recognizer_sp, module_regex_sp, symbol_regex_sp, false);
  }
  {
    auto symbol_regex_sp =
        std::make_shared<RegularExpression>("_swift_runtime_on_report");

    StackFrameRecognizerSP srf_recognizer_sp =
        std::make_shared<SwiftRuntimeInstrumentedFrameRecognizer>();

    process.GetTarget().GetFrameRecognizerManager().AddRecognizer(
        srf_recognizer_sp, module_regex_sp, symbol_regex_sp, false);
  }
}

} // namespace lldb_private
