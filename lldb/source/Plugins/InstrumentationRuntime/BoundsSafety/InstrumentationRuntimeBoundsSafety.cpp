//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrumentationRuntimeBoundsSafety.h"

#include "Plugins/Process/Utility/HistoryThread.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/InstrumentationRuntimeStopInfo.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/RegularExpression.h"
#include "clang/CodeGen/ModuleBuilder.h"

#include <memory>
#include <type_traits>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(InstrumentationRuntimeBoundsSafety)

constexpr llvm::StringLiteral
    BoundsSafetySoftTrapMinimal("__bounds_safety_soft_trap");
constexpr llvm::StringLiteral
    BoundsSafetySoftTrapStr("__bounds_safety_soft_trap_s");

constexpr std::array<llvm::StringLiteral, 2>
getBoundsSafetySoftTrapRuntimeFuncs() {
  return {BoundsSafetySoftTrapMinimal, BoundsSafetySoftTrapStr};
}

#define SOFT_TRAP_CATEGORY_PREFIX "Soft "
#define SOFT_TRAP_FALLBACK_CATEGORY                                            \
  SOFT_TRAP_CATEGORY_PREFIX "Bounds check failed"

using ComputedStopInfo =
    std::pair<std::optional<std::string>, std::optional<uint32_t>>;

class InstrumentationBoundsSafetyStopInfo : public StopInfo {
public:
  ~InstrumentationBoundsSafetyStopInfo() override = default;

  lldb::StopReason GetStopReason() const override {
    return lldb::eStopReasonInstrumentation;
  }

  std::optional<uint32_t>
  GetSuggestedStackFrameIndex(bool inlined_stack) override {
    return m_value;
  }

  const char *GetDescription() override { return m_description.c_str(); }

  bool DoShouldNotify(Event *event_ptr) override { return true; }

  static lldb::StopInfoSP
  CreateInstrumentationBoundsSafetyStopInfo(Thread &thread) {
    return StopInfoSP(new InstrumentationBoundsSafetyStopInfo(thread));
  }

private:
  InstrumentationBoundsSafetyStopInfo(Thread &thread);

  ComputedStopInfo
  ComputeStopReasonAndSuggestedStackFrame(bool &warning_emitted_for_failure);

  ComputedStopInfo ComputeStopReasonAndSuggestedStackFrameWithDebugInfo(
      lldb::StackFrameSP parent_sf, lldb::user_id_t debugger_id,
      bool &warning_emitted_for_failure);

  ComputedStopInfo ComputeStopReasonAndSuggestedStackFrameWithoutDebugInfo(
      ThreadSP thread_sp, lldb::user_id_t debugger_id,
      bool &warning_emitted_for_failure);
};

InstrumentationBoundsSafetyStopInfo::InstrumentationBoundsSafetyStopInfo(
    Thread &thread)
    : StopInfo(thread, 0) {
  // No additional data describing the reason for stopping.
  m_extended_info = nullptr;
  m_description = SOFT_TRAP_FALLBACK_CATEGORY;
  Log *log_category = GetLog(LLDBLog::InstrumentationRuntime);

  bool warning_emitted_for_failure = false;
  auto [MaybeDescription, MaybeSuggestedStackIndex] =
      ComputeStopReasonAndSuggestedStackFrame(warning_emitted_for_failure);
  if (MaybeDescription)
    m_description = MaybeDescription.value();
  else
    LLDB_LOG(log_category, "failed to compute description");

  if (MaybeSuggestedStackIndex)
    m_value = MaybeSuggestedStackIndex.value();
  else
    LLDB_LOG(log_category, "failed to compute suggested stack index");

  // Emit warning about the failure to compute the stop info if one wasn't
  // already emitted.
  if ((!MaybeDescription.has_value()) && !warning_emitted_for_failure) {
    if (ThreadSP thread_sp = GetThread()) {
      lldb::user_id_t debugger_id =
          thread_sp->GetProcess()->GetTarget().GetDebugger().GetID();
      Debugger::ReportWarning(
          "specific BoundsSafety trap reason could not be computed",
          debugger_id);
    }
  }

  LLDB_LOG(log_category,
           "computed InstrumentationBoundsSafetyStopInfo: stack index: {0}, "
           "description:\"{1}\"",
           m_value, m_description);
}

// Helper functions to make it convenient to log a failure and then return.
template <typename T, typename... ArgTys>
[[nodiscard]] T LogBeforeReturn(ArgTys &&...Args) {
  LLDB_LOG(GetLog(LLDBLog::InstrumentationRuntime), Args...);
  return T();
}

template <typename... ArgTys>
[[nodiscard]] ComputedStopInfo LogFailedCSI(ArgTys &&...Args) {
  return LogBeforeReturn<ComputedStopInfo>(Args...);
}

ComputedStopInfo
InstrumentationBoundsSafetyStopInfo::ComputeStopReasonAndSuggestedStackFrame(
    bool &warning_emitted_for_failure) {
  ThreadSP thread_sp = GetThread();
  Log *log_category = GetLog(LLDBLog::InstrumentationRuntime);
  if (!thread_sp)
    return LogFailedCSI("failed to get thread while stopped");

  lldb::user_id_t debugger_id =
      thread_sp->GetProcess()->GetTarget().GetDebugger().GetID();

  StackFrameSP parent_sf = thread_sp->GetStackFrameAtIndex(1);
  if (!parent_sf)
    return LogFailedCSI("got nullptr when fetching stackframe at index 1");

  if (parent_sf->HasDebugInformation()) {
    LLDB_LOG(log_category,
             "frame {0} has debug info so trying to compute "
             "BoundsSafety stop info from debug info",
             parent_sf->GetFrameIndex());
    return ComputeStopReasonAndSuggestedStackFrameWithDebugInfo(
        parent_sf, debugger_id, warning_emitted_for_failure);
  }

  // If the debug info is missing we can still get some information
  // from the parameter in the soft trap runtime call.
  LLDB_LOG(log_category,
           "frame {0} has no debug info so trying to compute "
           "BoundsSafety stop info from registers",
           parent_sf->GetFrameIndex());
  return ComputeStopReasonAndSuggestedStackFrameWithoutDebugInfo(
      thread_sp, debugger_id, warning_emitted_for_failure);
}

ComputedStopInfo InstrumentationBoundsSafetyStopInfo::
    ComputeStopReasonAndSuggestedStackFrameWithDebugInfo(
        lldb::StackFrameSP parent_sf, lldb::user_id_t debugger_id,
        bool &warning_emitted_for_failure) {
  // First try to use debug info to understand the reason for trapping. The
  // call stack will look something like this:
  //
  // ```
  // frame #0: `__bounds_safety_soft_trap_s(reason="")
  // frame #1: `__clang_trap_msg$Bounds check failed$<reason>'
  // frame #2: `bad_read(index=10)
  // ```
  // ....
  const char *TrapReasonFuncName = parent_sf->GetFunctionName();

  auto MaybeTrapReason =
      clang::CodeGen::DemangleTrapReasonInDebugInfo(TrapReasonFuncName);
  if (!MaybeTrapReason.has_value())
    return LogFailedCSI(
        "clang::CodeGen::DemangleTrapReasonInDebugInfo(\"{0}\") call failed",
        TrapReasonFuncName);

  llvm::StringRef category = MaybeTrapReason.value().first;
  llvm::StringRef message = MaybeTrapReason.value().second;

  // TODO: Clang should probably be changed to emit the "Soft " prefix itself
  std::string stop_reason;
  llvm::raw_string_ostream ss(stop_reason);
  ss << SOFT_TRAP_CATEGORY_PREFIX;
  if (category.empty())
    ss << "<empty category>";
  else
    ss << category;
  if (message.empty()) {
    // This is not a failure so leave `warning_emitted_for_failure` untouched.
    Debugger::ReportWarning(
        "specific BoundsSafety trap reason is not "
        "available because the compiler omitted it from the debug info",
        debugger_id);
  } else {
    ss << ": " << message;
  }
  // Use computed stop-reason and assume the parent of `parent_sf` is the
  // the place in the user's code where the call to the soft trap runtime
  // originated.
  return std::make_pair(stop_reason, parent_sf->GetFrameIndex() + 1);
}

ComputedStopInfo InstrumentationBoundsSafetyStopInfo::
    ComputeStopReasonAndSuggestedStackFrameWithoutDebugInfo(
        ThreadSP thread_sp, lldb::user_id_t debugger_id,
        bool &warning_emitted_for_failure) {

  StackFrameSP softtrap_sf = thread_sp->GetStackFrameAtIndex(0);
  if (!softtrap_sf)
    return LogFailedCSI("got nullptr when fetching stackframe at index 0");
  llvm::StringRef trap_reason_func_name = softtrap_sf->GetFunctionName();

  if (trap_reason_func_name == BoundsSafetySoftTrapMinimal) {
    // This function has no arguments so there's no additional information
    // that would allow us to identify the trap reason.
    //
    // Use the fallback stop reason and the current frame.
    // While we "could" set the suggested frame to our parent (where the
    // bounds check failed), doing this leads to very misleading output in
    // LLDB. E.g.:
    //
    // ```
    //     0x100003b40 <+104>: bl  0x100003d64    ; __bounds_safety_soft_trap
    // ->  0x100003b44 <+108>: b   0x100003b48    ; <+112>
    // ```
    //
    // This makes it look we stopped after finishing the call to
    // `__bounds_safety_soft_trap` but actually we are in the middle of the
    // call. To avoid this confusion just use the current frame.
    std::string warning;
    llvm::raw_string_ostream ss(warning);
    ss << "specific BoundsSafety trap reason is not available because debug "
          "info is missing on the caller of '"
       << BoundsSafetySoftTrapMinimal << "'";
    Debugger::ReportWarning(warning.c_str(), debugger_id);
    warning_emitted_for_failure = true;
    return {};
  }

  // __bounds_safety_soft_trap_s has one argument which is a pointer to a string
  // describing the trap or a nullptr.
  if (trap_reason_func_name != BoundsSafetySoftTrapStr) {
    assert(0 && "hit breakpoint for unexpected function name");
    return LogFailedCSI(
        "unexpected function name. Expected \"{0}\" but got \"{1}\"",
        BoundsSafetySoftTrapStr.data(), trap_reason_func_name.data());
  }

  RegisterContextSP rc = thread_sp->GetRegisterContext();
  if (!rc)
    return LogFailedCSI("failed to get register context");

  // FIXME: LLDB should have an API that tells us for the current target if
  // `LLDB_REGNUM_GENERIC_ARG1` can be used.
  // https://github.com/llvm/llvm-project/issues/168602
  // Don't try for architectures where examining the first register won't
  // work.
  ProcessSP process = thread_sp->GetProcess();
  if (!process)
    return LogFailedCSI("failed to get process");

  switch (process->GetTarget().GetArchitecture().GetCore()) {
  case ArchSpec::eCore_x86_32_i386:
  case ArchSpec::eCore_x86_32_i486:
  case ArchSpec::eCore_x86_32_i486sx:
  case ArchSpec::eCore_x86_32_i686: {
    // Technically some x86 calling conventions do use a register for
    // passing the first argument but let's ignore that for now.
    std::string warning;
    llvm::raw_string_ostream ss(warning);
    ss << "specific BoundsSafety trap reason cannot be inferred on x86 when "
          "the caller of '"
       << BoundsSafetySoftTrapStr << "' is missing debug info";
    Debugger::ReportWarning(warning.c_str(), debugger_id);
    warning_emitted_for_failure = true;
    return {};
  }
  default: {
  }
  };

  // Examine the register for the first argument.
  const RegisterInfo *arg0_info = rc->GetRegisterInfo(
      lldb::RegisterKind::eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1);
  if (!arg0_info)
    return LogFailedCSI(
        "failed to get register info for LLDB_REGNUM_GENERIC_ARG1");
  RegisterValue reg_value;
  if (!rc->ReadRegister(arg0_info, reg_value))
    return LogFailedCSI("failed to read register {0}", arg0_info->name);
  uint64_t reg_value_as_int = reg_value.GetAsUInt64(UINT64_MAX);
  if (reg_value_as_int == UINT64_MAX)
    return LogFailedCSI("failed to read register {0} as a UInt64",
                        arg0_info->name);

  if (reg_value_as_int == 0) {
    // nullptr arg. The compiler will pass that if no trap reason string was
    // available.
    Debugger::ReportWarning(
        "specific BoundsSafety trap reason cannot be inferred because the "
        "compiler omitted the reason",
        debugger_id);
    warning_emitted_for_failure = true;
    return {};
  }

  // The first argument to the call is a pointer to a global C string
  // containing the trap reason.
  std::string out_string;
  Status error_status;
  thread_sp->GetProcess()->ReadCStringFromMemory(reg_value_as_int, out_string,
                                                 error_status);
  if (error_status.Fail())
    return LogFailedCSI("failed to read C string from address {0}",
                        (void *)reg_value_as_int);

  LLDB_LOG(GetLog(LLDBLog::InstrumentationRuntime),
           "read C string from {0} found in register {1}: \"{2}\"",
           (void *)reg_value_as_int, arg0_info->name, out_string.c_str());
  std::string stop_reason;
  llvm::raw_string_ostream SS(stop_reason);
  SS << SOFT_TRAP_FALLBACK_CATEGORY;
  if (!stop_reason.empty()) {
    SS << ": " << out_string;
  }
  // Use the current frame as the suggested frame for the same reason as for
  // `__bounds_safety_soft_trap`.
  return {stop_reason, 0};
}

InstrumentationRuntimeBoundsSafety::~InstrumentationRuntimeBoundsSafety() {
  Deactivate();
}

lldb::InstrumentationRuntimeSP
InstrumentationRuntimeBoundsSafety::CreateInstance(
    const lldb::ProcessSP &process_sp) {
  return InstrumentationRuntimeSP(
      new InstrumentationRuntimeBoundsSafety(process_sp));
}

void InstrumentationRuntimeBoundsSafety::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "BoundsSafety instrumentation runtime plugin.",
                                CreateInstance, GetTypeStatic);
}

void InstrumentationRuntimeBoundsSafety::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb::InstrumentationRuntimeType
InstrumentationRuntimeBoundsSafety::GetTypeStatic() {
  return lldb::eInstrumentationRuntimeTypeBoundsSafety;
}

const RegularExpression &
InstrumentationRuntimeBoundsSafety::GetPatternForRuntimeLibrary() {
  static RegularExpression regex;
  return regex;
}

bool InstrumentationRuntimeBoundsSafety::CheckIfRuntimeIsValid(
    const lldb::ModuleSP module_sp) {
  Log *log_category = GetLog(LLDBLog::InstrumentationRuntime);
  for (const auto &SoftTrapFunc : getBoundsSafetySoftTrapRuntimeFuncs()) {
    ConstString test_sym(SoftTrapFunc);

    if (module_sp->FindFirstSymbolWithNameAndType(test_sym,
                                                  lldb::eSymbolTypeAny)) {
      LLDB_LOG(log_category, "found \"{0}\" in {1}",
               test_sym.AsCString("<unknown symbol>"),
               module_sp->GetFileSpec().GetPath());
      return true;
    }
  }
  LLDB_LOG(log_category,
           "did not find BoundsSafety soft trap functions in module {0}",
           module_sp->GetFileSpec().GetPath());
  return false;
}

bool InstrumentationRuntimeBoundsSafety::NotifyBreakpointHit(
    void *baton, StoppointCallbackContext *context, user_id_t break_id,
    user_id_t break_loc_id) {
  assert(baton && "null baton");
  if (!baton)
    return false; ///< false => resume execution.

  InstrumentationRuntimeBoundsSafety *const instance =
      static_cast<InstrumentationRuntimeBoundsSafety *>(baton);

  ProcessSP process_sp = instance->GetProcessSP();
  if (!process_sp)
    return LogBeforeReturn<bool>("failed to get process from baton");
  ThreadSP thread_sp = context->exe_ctx_ref.GetThreadSP();
  if (!thread_sp)
    return LogBeforeReturn<bool>(
        "failed to get thread from StoppointCallbackContext");

  if (process_sp != context->exe_ctx_ref.GetProcessSP())
    return LogBeforeReturn<bool>(
        "process from baton ({0}) and StoppointCallbackContext ({1}) do "
        "not match",
        (void *)process_sp.get(),
        (void *)context->exe_ctx_ref.GetProcessSP().get());

  if (process_sp->GetModIDRef().IsLastResumeForUserExpression())
    return LogBeforeReturn<bool>("IsLastResumeForUserExpression is true");

  // Maybe the stop reason and stackframe selection should be done by
  // a stackframe recognizer instead?
  thread_sp->SetStopInfo(
      InstrumentationBoundsSafetyStopInfo::
          CreateInstrumentationBoundsSafetyStopInfo(*thread_sp));
  return true;
}

void InstrumentationRuntimeBoundsSafety::Activate() {
  if (IsActive())
    return;

  ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return LogBeforeReturn<void>("could not get process during Activate()");

  std::vector<std::string> breakpoints;
  for (auto &breakpoint_func : getBoundsSafetySoftTrapRuntimeFuncs())
    breakpoints.emplace_back(breakpoint_func);

  BreakpointSP breakpoint = process_sp->GetTarget().CreateBreakpoint(
      /*containingModules=*/nullptr,
      /*containingSourceFiles=*/nullptr, breakpoints, eFunctionNameTypeFull,
      eLanguageTypeUnknown,
      /*m_offset=*/0,
      /*skip_prologue*/ eLazyBoolNo,
      /*internal=*/true,
      /*request_hardware*/ false);

  if (!breakpoint)
    return LogBeforeReturn<void>("failed to create breakpoint");

  if (!breakpoint->HasResolvedLocations()) {
    assert(0 && "breakpoint has no resolved locations");
    process_sp->GetTarget().RemoveBreakpointByID(breakpoint->GetID());
    return LogBeforeReturn<void>(
        "breakpoint {0} for BoundsSafety soft traps did not resolve to "
        "any locations",
        breakpoint->GetID());
  }

  // Note: When `sync=true` the suggested stackframe is completely ignored. So
  // we use `sync=false`. Is that a bug?
  breakpoint->SetCallback(
      InstrumentationRuntimeBoundsSafety::NotifyBreakpointHit, this,
      /*sync=*/false);
  breakpoint->SetBreakpointKind("bounds-safety-soft-trap");
  SetBreakpointID(breakpoint->GetID());
  LLDB_LOG(GetLog(LLDBLog::InstrumentationRuntime),
           "created breakpoint {0} for BoundsSafety soft traps",
           breakpoint->GetID());
  SetActive(true);
}

void InstrumentationRuntimeBoundsSafety::Deactivate() {
  SetActive(false);
  Log *log_category = GetLog(LLDBLog::InstrumentationRuntime);
  if (ProcessSP process_sp = GetProcessSP()) {
    bool success =
        process_sp->GetTarget().RemoveBreakpointByID(GetBreakpointID());
    // FIXME: GetBreakPointID() uses `lldb::user_id_t` which is an unsigned
    // type but it should be using `break_id_t` which is a signed type. For now
    // just use the right type in the format string so the breakpoint ID is
    // printed correctly.
    LLDB_LOG(log_category,
             "{0}removed breakpoint {1} for BoundsSafety soft traps",
             success ? "" : "failed to ",
             static_cast<break_id_t>(GetBreakpointID()));
  } else {
    LLDB_LOG(log_category, "no process available during Deactivate()");
  }

  SetBreakpointID(LLDB_INVALID_BREAK_ID);
}
