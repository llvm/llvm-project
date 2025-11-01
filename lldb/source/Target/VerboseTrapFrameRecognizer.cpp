#include "lldb/Target/VerboseTrapFrameRecognizer.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Target/Target.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "clang/CodeGen/ModuleBuilder.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

/// The 0th frame is the artificial inline frame generated to store
/// the verbose_trap message. So, starting with the current parent frame,
/// find the first frame that's not inside of the STL.
static StackFrameSP FindMostRelevantFrame(Thread &selected_thread) {
  // Defensive upper-bound of when we stop walking up the frames in
  // case we somehow ended up looking at an infinite recursion.
  const size_t max_stack_depth = 128;

  // Start at parent frame.
  size_t stack_idx = 1;
  StackFrameSP most_relevant_frame_sp =
      selected_thread.GetStackFrameAtIndex(stack_idx);

  while (most_relevant_frame_sp && stack_idx <= max_stack_depth) {
    auto const &sc =
        most_relevant_frame_sp->GetSymbolContext(eSymbolContextEverything);
    ConstString frame_name = sc.GetFunctionName();
    if (!frame_name)
      return nullptr;

    // Found a frame outside of the `std` namespace. That's the
    // first frame in user-code that ended up triggering the
    // verbose_trap. Hence that's the one we want to display.
    if (!frame_name.GetStringRef().starts_with("std::"))
      return most_relevant_frame_sp;

    ++stack_idx;
    most_relevant_frame_sp = selected_thread.GetStackFrameAtIndex(stack_idx);
  }

  return nullptr;
}

VerboseTrapRecognizedStackFrame::VerboseTrapRecognizedStackFrame(
    StackFrameSP most_relevant_frame_sp, std::string stop_desc)
    : m_most_relevant_frame(most_relevant_frame_sp) {
  m_stop_desc = std::move(stop_desc);
}

lldb::RecognizedStackFrameSP
VerboseTrapFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame_sp) {
  if (frame_sp->GetFrameIndex())
    return {};

  ThreadSP thread_sp = frame_sp->GetThread();
  ProcessSP process_sp = thread_sp->GetProcess();

  StackFrameSP most_relevant_frame_sp = FindMostRelevantFrame(*thread_sp);

  if (!most_relevant_frame_sp) {
    Log *log = GetLog(LLDBLog::Unwind);
    LLDB_LOG(
        log,
        "Failed to find most relevant frame: Hit unwinding bound (1 frame)!");
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

  auto func_name = inline_info->GetName().GetStringRef();
  if (func_name.empty())
    return {};

  auto MaybeTrapReason =
      clang::CodeGen::DemangleTrapReasonInDebugInfo(func_name);
  if (!MaybeTrapReason.has_value()) {
    LLDB_LOGF(GetLog(LLDBLog::Unwind), "Failed to demangle '%s' as trap reason",
              func_name.str().c_str());
    return {};
  }
  auto [category, message] = MaybeTrapReason.value();

  std::string stop_reason =
      category.empty() ? "<empty category>" : category.str();
  if (!message.empty()) {
    stop_reason += ": ";
    stop_reason += message.str();
  }

  return std::make_shared<VerboseTrapRecognizedStackFrame>(
      most_relevant_frame_sp, std::move(stop_reason));
}

lldb::StackFrameSP VerboseTrapRecognizedStackFrame::GetMostRelevantFrame() {
  return m_most_relevant_frame;
}

namespace lldb_private {

void RegisterVerboseTrapFrameRecognizer(Process &process) {
  RegularExpressionSP module_regex_sp = nullptr;
  auto symbol_regex_sp = std::make_shared<RegularExpression>(
      llvm::formatv("^{0}", ClangTrapPrefix).str());

  StackFrameRecognizerSP srf_recognizer_sp =
      std::make_shared<VerboseTrapFrameRecognizer>();

  process.GetTarget().GetFrameRecognizerManager().AddRecognizer(
      srf_recognizer_sp, module_regex_sp, symbol_regex_sp,
      Mangled::ePreferDemangled, false);
}

} // namespace lldb_private
