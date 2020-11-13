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

/// Stores a function module spec, symbol name and possibly an alternate symbol
/// name.
struct SymbolLocation {
  FileSpec module_spec;
  std::vector<ConstString> symbols;
};

/// Fetches the abort frame location depending on the current platform.
///
/// \param[in] os
///    The target's os type.
/// \param[in,out] location
///    The struct that will contain the abort module spec and symbol names.
/// \return
///    \b true, if the platform is supported
///    \b false, otherwise.
bool GetAbortLocation(llvm::Triple::OSType os, SymbolLocation &location) {
  switch (os) {
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    location.module_spec = FileSpec("libsystem_kernel.dylib");
    location.symbols.push_back(ConstString("__pthread_kill"));
    break;
  case llvm::Triple::Linux:
    location.module_spec = FileSpec("libc.so.6");
    location.symbols.push_back(ConstString("raise"));
    location.symbols.push_back(ConstString("__GI_raise"));
    location.symbols.push_back(ConstString("gsignal"));
    break;
  default:
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_UNWIND));
    LLDB_LOG(log, "AssertFrameRecognizer::GetAbortLocation Unsupported OS");
    return false;
  }

  return true;
}

/// Fetches the assert frame location depending on the current platform.
///
/// \param[in] os
///    The target's os type.
/// \param[in,out] location
///    The struct that will contain the assert module spec and symbol names.
/// \return
///    \b true, if the platform is supported
///    \b false, otherwise.
bool GetAssertLocation(llvm::Triple::OSType os, SymbolLocation &location) {
  switch (os) {
  case llvm::Triple::Darwin:
  case llvm::Triple::MacOSX:
    location.module_spec = FileSpec("libsystem_c.dylib");
    location.symbols.push_back(ConstString("__assert_rtn"));
    break;
  case llvm::Triple::Linux:
    location.module_spec = FileSpec("libc.so.6");
    location.symbols.push_back(ConstString("__assert_fail"));
    location.symbols.push_back(ConstString("__GI___assert_fail"));
    break;
  default:
    Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_UNWIND));
    LLDB_LOG(log, "AssertFrameRecognizer::GetAssertLocation Unsupported OS");
    return false;
  }

  return true;
}

void RegisterAssertFrameRecognizer(Process *process) {
  Target &target = process->GetTarget();
  llvm::Triple::OSType os = target.GetArchitecture().GetTriple().getOS();
  SymbolLocation location;

  if (!GetAbortLocation(os, location))
    return;

  target.GetFrameRecognizerManager().AddRecognizer(
      StackFrameRecognizerSP(new AssertFrameRecognizer()),
      location.module_spec.GetFilename(), location.symbols,
      /*first_instruction_only*/ false);
}

} // namespace lldb_private

lldb::RecognizedStackFrameSP
AssertFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame_sp) {
  ThreadSP thread_sp = frame_sp->GetThread();
  ProcessSP process_sp = thread_sp->GetProcess();
  Target &target = process_sp->GetTarget();
  llvm::Triple::OSType os = target.GetArchitecture().GetTriple().getOS();
  SymbolLocation location;

  if (!GetAssertLocation(os, location))
    return RecognizedStackFrameSP();

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

    if (!sym_ctx.module_sp ||
        !sym_ctx.module_sp->GetFileSpec().FileEquals(location.module_spec))
      continue;

    ConstString func_name = sym_ctx.GetFunctionName();

    if (llvm::is_contained(location.symbols, func_name)) {
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
