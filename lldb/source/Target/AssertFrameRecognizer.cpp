#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrameList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

#include "lldb/Target/AssertFrameRecognizer.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
/// Fetches the abort frame location depending on the current platform.
///
/// \param[in] process_sp
///    The process that is currently aborting. This will give us information on
///    the target and the platform.
/// \return
///    If the platform is supported, returns an optional tuple containing
///    the abort module as a \a FileSpec and two symbol names as two \a
///    StringRef. The second \a StringRef may be empty.
///    Otherwise, returns \a llvm::None.
llvm::Optional<std::tuple<FileSpec, StringRef, StringRef>>
GetAbortLocation(Process *process) {
  Target &target = process->GetTarget();

  FileSpec module_spec;
  StringRef symbol_name, alternate_symbol_name;

  switch (target.GetArchitecture().GetTriple().getOS()) {
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    module_spec = FileSpec("libsystem_kernel.dylib");
    symbol_name = "__pthread_kill";
    break;
  case llvm::Triple::Linux:
    module_spec = FileSpec("libc.so.6");
    symbol_name = "raise";
    alternate_symbol_name = "__GI_raise";
    break;
  default:
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_UNWIND));
    LLDB_LOG(log, "AssertFrameRecognizer::GetAbortLocation Unsupported OS");
    return llvm::None;
  }

  return std::make_tuple(module_spec, symbol_name, alternate_symbol_name);
}

/// Fetches the assert frame location depending on the current platform.
///
/// \param[in] process_sp
///    The process that is currently asserting. This will give us information on
///    the target and the platform.
/// \return
///    If the platform is supported, returns an optional tuple containing
///    the asserting frame module as a \a FileSpec and two possible symbol
///    names as two \a StringRef. The second \a StringRef may be empty.
///    Otherwise, returns \a llvm::None.
llvm::Optional<std::tuple<FileSpec, StringRef, StringRef>>
GetAssertLocation(Process *process) {
  Target &target = process->GetTarget();

  FileSpec module_spec;
  StringRef symbol_name, alternate_symbol_name;

  switch (target.GetArchitecture().GetTriple().getOS()) {
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    module_spec = FileSpec("libsystem_c.dylib");
    symbol_name = "__assert_rtn";
    break;
  case llvm::Triple::Linux:
    module_spec = FileSpec("libc.so.6");
    symbol_name = "__assert_fail";
    alternate_symbol_name = "__GI___assert_fail";
    break;
  default:
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_UNWIND));
    LLDB_LOG(log, "AssertFrameRecognizer::GetAssertLocation Unsupported OS");
    return llvm::None;
  }

  return std::make_tuple(module_spec, symbol_name, alternate_symbol_name);
}

void RegisterAssertFrameRecognizer(Process *process) {
  static llvm::once_flag g_once_flag;
  llvm::call_once(g_once_flag, [process]() {
    auto abort_location = GetAbortLocation(process);

    if (!abort_location.hasValue())
      return;

    FileSpec module_spec;
    StringRef function_name, alternate_function_name;
    std::tie(module_spec, function_name, alternate_function_name) =
        *abort_location;

    StackFrameRecognizerManager::AddRecognizer(
        StackFrameRecognizerSP(new AssertFrameRecognizer()),
        module_spec.GetFilename(), ConstString(function_name),
        ConstString(alternate_function_name), /*first_instruction_only*/ false);
  });
}

} // namespace lldb_private

lldb::RecognizedStackFrameSP
AssertFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame_sp) {
  ThreadSP thread_sp = frame_sp->GetThread();
  ProcessSP process_sp = thread_sp->GetProcess();

  auto assert_location = GetAssertLocation(process_sp.get());

  if (!assert_location.hasValue())
    return RecognizedStackFrameSP();

  FileSpec module_spec;
  StringRef function_name, alternate_function_name;
  std::tie(module_spec, function_name, alternate_function_name) =
      *assert_location;

  const uint32_t frames_to_fetch = 5;
  const uint32_t last_frame_index = frames_to_fetch - 1;
  StackFrameSP prev_frame_sp = nullptr;

  // Fetch most relevant frame
  for (uint32_t frame_index = 0; frame_index < frames_to_fetch; frame_index++) {
    prev_frame_sp = thread_sp->GetStackFrameAtIndex(frame_index);

    if (!prev_frame_sp) {
      Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_UNWIND));
      LLDB_LOG(log, "Abort Recognizer: Hit unwinding bound ({1} frames)!",
               frames_to_fetch);
      break;
    }

    SymbolContext sym_ctx =
        prev_frame_sp->GetSymbolContext(eSymbolContextEverything);

    if (!sym_ctx.module_sp->GetFileSpec().FileEquals(module_spec))
      continue;

    ConstString func_name = sym_ctx.GetFunctionName();
    if (func_name == ConstString(function_name) ||
        alternate_function_name.empty() ||
        func_name == ConstString(alternate_function_name)) {

      // We go a frame beyond the assert location because the most relevant
      // frame for the user is the one in which the assert function was called.
      // If the assert location is the last frame fetched, then it is set as
      // the most relevant frame.

      StackFrameSP most_relevant_frame_sp = thread_sp->GetStackFrameAtIndex(
          std::min(frame_index + 1, last_frame_index));

      // Pass assert location to AbortRecognizedStackFrame to set as most
      // relevant frame.
      return lldb::RecognizedStackFrameSP(
          new AssertRecognizedStackFrame(most_relevant_frame_sp));
    }
  }

  return RecognizedStackFrameSP();
}

AssertRecognizedStackFrame::AssertRecognizedStackFrame(
    StackFrameSP most_relevant_frame_sp)
    : m_most_relevant_frame(most_relevant_frame_sp) {
  m_stop_desc = "hit program assert";
}

lldb::StackFrameSP AssertRecognizedStackFrame::GetMostRelevantFrame() {
  return m_most_relevant_frame;
}
