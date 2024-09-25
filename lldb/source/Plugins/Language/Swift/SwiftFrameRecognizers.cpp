#include "SwiftFrameRecognizers.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "swift/Strings.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

/// Holds the stack frame that caused the runtime failure and the inlined stop
/// reason message.
class SwiftRuntimeFailureRecognizedStackFrame : public RecognizedStackFrame {
public:
  SwiftRuntimeFailureRecognizedStackFrame(StackFrameSP most_relevant_frame_sp,
                                          llvm::StringRef stop_desc)
      : m_most_relevant_frame(most_relevant_frame_sp) {
    m_stop_desc = std::string(stop_desc);
  }

  lldb::StackFrameSP GetMostRelevantFrame() override {
    return m_most_relevant_frame;
  }

private:
  lldb::StackFrameSP m_most_relevant_frame;
};

/// When a thread stops, it checks the current frame contains a swift runtime
/// failure diagnostic. If so, it returns a \a
/// SwiftRuntimeFailureRecognizedStackFrame holding the diagnostic a stop reason
/// description with  and the parent frame as the most relevant frame.
class SwiftRuntimeFailureFrameRecognizer : public StackFrameRecognizer {
public:
  std::string GetName() override {
    return "Swift runtime failure frame recognizer";
  }

  lldb::RecognizedStackFrameSP
  RecognizeFrame(lldb::StackFrameSP frame_sp) override {
    if (frame_sp->GetFrameIndex())
      return {};

    ThreadSP thread_sp = frame_sp->GetThread();
    if (!thread_sp)
      return {};
    ProcessSP process_sp = thread_sp->GetProcess();

    StackFrameSP most_relevant_frame_sp = thread_sp->GetStackFrameAtIndex(1);

    if (!most_relevant_frame_sp) {
      Log *log = GetLog(LLDBLog::Unwind);
      LLDB_LOG(log, "{0}: Hit unwinding bound (1 frame)!", GetName());
      return {};
    }

    SymbolContext sc = frame_sp->GetSymbolContext(eSymbolContextEverything);

    if (!sc.block)
      return {};

    // The runtime error is set as the function name in the inlined function
    // info of frame #0 by the compiler
    // (https://github.com/apple/swift/pull/29506)
    const InlineFunctionInfo *inline_info = nullptr;
    Block *inline_block = sc.block->GetContainingInlinedBlock();

    if (!inline_block)
      return {};

    inline_info = sc.block->GetInlinedFunctionInfo();

    if (!inline_info)
      return {};

    llvm::StringRef runtime_error = inline_info->GetName().AsCString();

    if (runtime_error.empty())
      return {};

    return lldb::RecognizedStackFrameSP(
        new SwiftRuntimeFailureRecognizedStackFrame(most_relevant_frame_sp,
                                                    runtime_error));
  }
};

/// Detect when a thread stops in _swift_runtime_on_report.
class SwiftRuntimeInstrumentedFrameRecognizer : public StackFrameRecognizer {
public:
  std::string GetName() override {
    return "Swift runtime instrumentation frame recognizer";
  }
  lldb::RecognizedStackFrameSP
  RecognizeFrame(lldb::StackFrameSP frame_sp) override {
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
        LLDB_LOG(log, "{0}: Hit unwinding bound ({1} frames)!", GetName(), i);
        return {};
      }
      auto &sc = most_relevant_frame_sp->GetSymbolContext(
          lldb::eSymbolContextFunction);
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
};

/// A frame recognizer that to hide Swift trampolines and thunks from
/// the backtrace.
class SwiftHiddenFrameRecognizer : public StackFrameRecognizer {
  RegularExpression m_hidden_function_regex;
  RecognizedStackFrameSP m_hidden_frame;

  struct SwiftHiddenFrame : public RecognizedStackFrame {
    bool ShouldHide() override { return true; }
  };

  /// Returns true if \ref root represents a Swift name
  /// that we want to mark hidden by this recognizer.
  ///
  /// Currently these are:
  /// * Async thunks
  /// * Auto-conformed protocols in the `std` module
  ///
  bool ShouldHideSwiftName(NodePointer root) {
    using namespace swift_demangle;

    if (NodeAtPath(root, {Node::Kind::Global,
                          Node::Kind::AsyncAwaitResumePartialFunction}) &&
        (ChildAtPath(root, {Node::Kind::BackDeploymentFallback}) ||
         ChildAtPath(root, {Node::Kind::PartialApplyForwarder})))
      return true;

    if (auto witness_node =
            NodeAtPath(root, {Node::Kind::Global, Node::Kind::ProtocolWitness,
                              Node::Kind::ProtocolConformance}))
      if (auto module_node = ChildAtPath(witness_node, {Node::Kind::Module});
          module_node && module_node->getText() == "__C_Synthesized")
        return true;
    ;

    return false;
  }

public:
  SwiftHiddenFrameRecognizer() : m_hidden_frame(new SwiftHiddenFrame()) {}

  std::string GetName() override { return "Swift hidden frame recognizer"; }

  lldb::RecognizedStackFrameSP
  RecognizeFrame(lldb::StackFrameSP frame_sp) override {
    if (!frame_sp)
      return {};

    // Hide compiler-generated frames.
    if (frame_sp->IsArtificial())
      return m_hidden_frame;

    const auto &sc = frame_sp->GetSymbolContext(lldb::eSymbolContextFunction);
    if (!sc.function)
      return {};

    FileSpec source_file;
    uint32_t line_no;
    sc.function->GetStartLineSourceInfo(source_file, line_no);
    // FIXME: these <compiler-generated> frames should be marked artificial
    // by the Swift compiler.
    if (source_file.GetFilename() == "<compiler-generated>"
        && line_no == 0)
      return m_hidden_frame;

    auto symbol_name =
        sc.function->GetMangled().GetMangledName().GetStringRef();

    using namespace swift::Demangle;
    using namespace swift_demangle;
    Context demangle_ctx;
    if (NodePointer nodes = SwiftLanguageRuntime::DemangleSymbolAsNode(
            symbol_name, demangle_ctx))
      if (ShouldHideSwiftName(nodes))
        return m_hidden_frame;

    return {};
  }
};

void RegisterSwiftFrameRecognizers(Process &process) {
  RegularExpressionSP module_regex_sp = nullptr;
  auto &manager = process.GetTarget().GetFrameRecognizerManager();
  {
    auto symbol_regex_sp =
        std::make_shared<RegularExpression>("Swift runtime failure");
    auto srf_sp = std::make_shared<SwiftRuntimeFailureFrameRecognizer>();
    manager.AddRecognizer(
        srf_sp, module_regex_sp, symbol_regex_sp,
        Mangled::NamePreference::ePreferDemangledWithoutArguments, false);
  }
  {
    auto symbol_regex_sp =
        std::make_shared<RegularExpression>("_swift_runtime_on_report");
    auto srf_sp = std::make_shared<SwiftRuntimeInstrumentedFrameRecognizer>();
    manager.AddRecognizer(
        srf_sp, module_regex_sp, symbol_regex_sp,
        Mangled::NamePreference::ePreferDemangledWithoutArguments, false);
  }
  {
    auto symbol_regex_sp = std::make_shared<RegularExpression>("^\\$s.*");
    auto srf_sp = std::make_shared<SwiftHiddenFrameRecognizer>();
    manager.AddRecognizer(srf_sp, module_regex_sp, symbol_regex_sp,
                          Mangled::NamePreference::ePreferMangled, false);
  }
}

} // namespace lldb_private
