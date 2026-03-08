//===-- CPPLanguageRuntime.cpp---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <iostream>

#include <memory>

#include "CPPLanguageRuntime.h"
#include "CommandObjectCPlusPlus.h"
#include "VerboseTrapFrameRecognizer.h"

#include "llvm/ADT/StringRef.h"

#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Utility/Timer.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(CPPLanguageRuntime, CPPRuntime)

static ConstString g_this = ConstString("this");
// Artificial coroutine-related variables emitted by clang.
static ConstString g_promise = ConstString("__promise");
static ConstString g_coro_frame = ConstString("__coro_frame");

char CPPLanguageRuntime::ID = 0;

/// A frame recognizer that is installed to hide libc++ implementation
/// details from the backtrace.
class LibCXXFrameRecognizer : public StackFrameRecognizer {
  std::array<RegularExpression, 2> m_hidden_regex;
  RecognizedStackFrameSP m_hidden_frame;

  struct LibCXXHiddenFrame : public RecognizedStackFrame {
    bool ShouldHide() override { return true; }
  };

public:
  LibCXXFrameRecognizer()
      : m_hidden_regex{
            // internal implementation details in the `std::` namespace
            //    std::__1::__function::__alloc_func<void (*)(), std::__1::allocator<void (*)()>, void ()>::operator()[abi:ne200000]
            //    std::__1::__function::__func<void (*)(), std::__1::allocator<void (*)()>, void ()>::operator()
            //    std::__1::__function::__value_func<void ()>::operator()[abi:ne200000]() const
            //    std::__2::__function::__policy_invoker<void (int, int)>::__call_impl[abi:ne200000]<std::__2::__function::__default_alloc_func<int (*)(int, int), int (int, int)>>
            //    std::__1::__invoke[abi:ne200000]<void (*&)()>
            //    std::__1::__invoke_void_return_wrapper<void, true>::__call[abi:ne200000]<void (*&)()>
            RegularExpression{R"(^std::__[^:]*::__)"},
            // internal implementation details in the `std::ranges` namespace
            //    std::__1::ranges::__sort::__sort_fn_impl[abi:ne200000]<std::__1::__wrap_iter<int*>, std::__1::__wrap_iter<int*>, bool (*)(int, int), std::__1::identity>
            RegularExpression{R"(^std::__[^:]*::ranges::__)"},
        },
        m_hidden_frame(new LibCXXHiddenFrame()) {}

  std::string GetName() override { return "libc++ frame recognizer"; }

  lldb::RecognizedStackFrameSP
  RecognizeFrame(lldb::StackFrameSP frame_sp) override {
    if (!frame_sp)
      return {};
    const auto &sc = frame_sp->GetSymbolContext(lldb::eSymbolContextFunction);
    if (!sc.function)
      return {};

    // Check if we have a regex match
    for (RegularExpression &r : m_hidden_regex) {
      if (!r.Execute(sc.function->GetNameNoArguments()))
        continue;

      // Only hide this frame if the immediate caller is also within libc++.
      lldb::ThreadSP thread_sp = frame_sp->GetThread();
      if (!thread_sp)
        return {};
      lldb::StackFrameSP parent_frame_sp =
          thread_sp->GetStackFrameAtIndex(frame_sp->GetFrameIndex() + 1);
      if (!parent_frame_sp)
        return {};
      const auto &parent_sc =
          parent_frame_sp->GetSymbolContext(lldb::eSymbolContextFunction);
      if (!parent_sc.function)
        return {};
      if (parent_sc.function->GetNameNoArguments().GetStringRef().starts_with(
              "std::"))
        return m_hidden_frame;
    }

    return {};
  }
};

CPPLanguageRuntime::CPPLanguageRuntime(Process *process)
    : LanguageRuntime(process), m_itanium_runtime(process) {
  if (process) {
    process->GetTarget().GetFrameRecognizerManager().AddRecognizer(
        StackFrameRecognizerSP(new LibCXXFrameRecognizer()), {},
        std::make_shared<RegularExpression>("^std::__[^:]*::"),
        /*mangling_preference=*/Mangled::ePreferDemangledWithoutArguments,
        /*first_instruction_only=*/false);

    RegisterVerboseTrapFrameRecognizer(*process);
  }
}

bool CPPLanguageRuntime::IsAllowedRuntimeValue(ConstString name) {
  return name == g_this || name == g_promise || name == g_coro_frame;
}

llvm::Error CPPLanguageRuntime::GetObjectDescription(Stream &str,
                                                     ValueObject &object) {
  // C++ has no generic way to do this.
  return llvm::createStringError("C++ does not support object descriptions");
}

llvm::Error
CPPLanguageRuntime::GetObjectDescription(Stream &str, Value &value,
                                         ExecutionContextScope *exe_scope) {
  // C++ has no generic way to do this.
  return llvm::createStringError("C++ does not support object descriptions");
}

bool contains_lambda_identifier(llvm::StringRef &str_ref) {
  return str_ref.contains("$_") || str_ref.contains("'lambda'");
}

CPPLanguageRuntime::LibCppStdFunctionCallableInfo
line_entry_helper(Target &target, const SymbolContext &sc, Symbol *symbol,
                  llvm::StringRef first_template_param_sref, bool has_invoke) {

  CPPLanguageRuntime::LibCppStdFunctionCallableInfo optional_info;

  Address address = sc.GetFunctionOrSymbolAddress();

  Address addr;
  if (target.ResolveLoadAddress(address.GetCallableLoadAddress(&target),
                                addr)) {
    LineEntry line_entry;
    addr.CalculateSymbolContextLineEntry(line_entry);

    if (contains_lambda_identifier(first_template_param_sref) || has_invoke) {
      // Case 1 and 2
      optional_info.callable_case = lldb_private::CPPLanguageRuntime::
          LibCppStdFunctionCallableCase::Lambda;
    } else {
      // Case 3
      optional_info.callable_case = lldb_private::CPPLanguageRuntime::
          LibCppStdFunctionCallableCase::CallableObject;
    }

    optional_info.callable_symbol = *symbol;
    optional_info.callable_line_entry = line_entry;
    optional_info.callable_address = addr;
  }

  return optional_info;
}

CPPLanguageRuntime::LibCppStdFunctionCallableInfo
CPPLanguageRuntime::FindLibCppStdFunctionCallableInfo(
    lldb::ValueObjectSP &valobj_sp) {
  LLDB_SCOPED_TIMER();

  LibCppStdFunctionCallableInfo optional_info;

  if (!valobj_sp)
    return optional_info;

  // Member __f_ has type __base*, the contents of which will hold:
  // 1) a vtable entry which may hold type information needed to discover the
  //    lambda being called
  // 2) possibly hold a pointer to the callable object
  // e.g.
  //
  // (lldb) frame var -R  f_display
  // (std::__1::function<void (int)>) f_display = {
  //  __buf_ = {
  //  …
  // }
  //  __f_ = 0x00007ffeefbffa00
  // }
  // (lldb) memory read -fA 0x00007ffeefbffa00
  // 0x7ffeefbffa00: ... `vtable for std::__1::__function::__func<void (*) ...
  // 0x7ffeefbffa08: ... `print_num(int) at std_function_cppreference_exam ...
  //
  // We will be handling five cases below, std::function is wrapping:
  //
  // 1) a lambda we know at compile time. We will obtain the name of the lambda
  //    from the first template pameter from __func's vtable. We will look up
  //    the lambda's operator()() and obtain the line table entry.
  // 2) a lambda we know at runtime. A pointer to the lambdas __invoke method
  //    will be stored after the vtable. We will obtain the lambdas name from
  //    this entry and lookup operator()() and obtain the line table entry.
  // 3) a callable object via operator()(). We will obtain the name of the
  //    object from the first template parameter from __func's vtable. We will
  //    look up the objects operator()() and obtain the line table entry.
  // 4) a member function. A pointer to the function will stored after the
  //    we will obtain the name from this pointer.
  // 5) a free function. A pointer to the function will stored after the vtable
  //    we will obtain the name from this pointer.
  ValueObjectSP member_f_(valobj_sp->GetChildMemberWithName("__f_"));

  if (member_f_) {
    ValueObjectSP sub_member_f_(member_f_->GetChildMemberWithName("__f_"));

    if (sub_member_f_)
      member_f_ = sub_member_f_;
  }

  if (!member_f_)
    return optional_info;

  lldb::addr_t member_f_pointer_value = member_f_->GetValueAsUnsigned(0);

  optional_info.member_f_pointer_value = member_f_pointer_value;

  if (!member_f_pointer_value)
    return optional_info;

  ExecutionContext exe_ctx(valobj_sp->GetExecutionContextRef());
  Process *process = exe_ctx.GetProcessPtr();

  if (process == nullptr)
    return optional_info;

  uint32_t address_size = process->GetAddressByteSize();
  Status status;

  // First item pointed to by __f_ should be the pointer to the vtable for
  // a __base object.
  lldb::addr_t vtable_address =
      process->ReadPointerFromMemory(member_f_pointer_value, status);

  if (status.Fail())
    return optional_info;

  lldb::addr_t vtable_address_first_entry =
      process->ReadPointerFromMemory(vtable_address + address_size, status);

  if (status.Fail())
    return optional_info;

  lldb::addr_t address_after_vtable = member_f_pointer_value + address_size;
  // As commented above we may not have a function pointer but if we do we will
  // need it.
  lldb::addr_t possible_function_address =
      process->ReadPointerFromMemory(address_after_vtable, status);

  if (status.Fail())
    return optional_info;

  Target &target = process->GetTarget();

  if (!target.HasLoadedSections())
    return optional_info;

  Address vtable_first_entry_resolved;

  if (!target.ResolveLoadAddress(vtable_address_first_entry,
                                 vtable_first_entry_resolved))
    return optional_info;

  Address vtable_addr_resolved;
  SymbolContext sc;
  Symbol *symbol = nullptr;

  if (!target.ResolveLoadAddress(vtable_address, vtable_addr_resolved))
    return optional_info;

  target.GetImages().ResolveSymbolContextForAddress(
      vtable_addr_resolved, eSymbolContextEverything, sc);
  symbol = sc.symbol;

  if (symbol == nullptr)
    return optional_info;

  llvm::StringRef vtable_name(symbol->GetName().GetStringRef());
  bool found_expected_start_string =
      vtable_name.starts_with("vtable for std::__1::__function::__func<");

  if (!found_expected_start_string)
    return optional_info;

  // Given case 1 or 3 we have a vtable name, we are want to extract the first
  // template parameter
  //
  //  ... __func<main::$_0, std::__1::allocator<main::$_0> ...
  //             ^^^^^^^^^
  //
  // We could see names such as:
  //    main::$_0
  //    Bar::add_num2(int)::'lambda'(int)
  //    Bar
  //
  // We do this by find the first < and , and extracting in between.
  //
  // This covers the case of the lambda known at compile time.
  size_t first_open_angle_bracket = vtable_name.find('<') + 1;
  size_t first_comma = vtable_name.find(',');

  llvm::StringRef first_template_parameter =
      vtable_name.slice(first_open_angle_bracket, first_comma);

  Address function_address_resolved;

  // Setup for cases 2, 4 and 5 we have a pointer to a function after the
  // vtable. We will use a process of elimination to drop through each case
  // and obtain the data we need.
  if (target.ResolveLoadAddress(possible_function_address,
                                function_address_resolved)) {
    target.GetImages().ResolveSymbolContextForAddress(
        function_address_resolved, eSymbolContextEverything, sc);
    symbol = sc.symbol;
  }

  // These conditions are used several times to simplify statements later on.
  bool has_invoke =
      (symbol ? symbol->GetName().GetStringRef().contains("__invoke") : false);
  auto calculate_symbol_context_helper = [](auto &t,
                                            SymbolContextList &sc_list) {
    SymbolContext sc;
    t->CalculateSymbolContext(&sc);
    sc_list.Append(sc);
  };

  // Case 2
  if (has_invoke) {
    SymbolContextList scl;
    calculate_symbol_context_helper(symbol, scl);

    return line_entry_helper(target, scl[0], symbol, first_template_parameter,
                             has_invoke);
  }

  // Case 4 or 5
  if (symbol && !symbol->GetName().GetStringRef().starts_with("vtable for") &&
      !contains_lambda_identifier(first_template_parameter) && !has_invoke) {
    optional_info.callable_case =
        LibCppStdFunctionCallableCase::FreeOrMemberFunction;
    optional_info.callable_address = function_address_resolved;
    optional_info.callable_symbol = *symbol;

    return optional_info;
  }

  std::string func_to_match = first_template_parameter.str();

  auto it = CallableLookupCache.find(func_to_match);
  if (it != CallableLookupCache.end())
    return it->second;

  SymbolContextList scl;

  CompileUnit *vtable_cu =
      vtable_first_entry_resolved.CalculateSymbolContextCompileUnit();
  llvm::StringRef name_to_use = func_to_match;

  // Case 3, we have a callable object instead of a lambda
  //
  // TODO
  // We currently don't support this case a callable object may have multiple
  // operator()() varying on const/non-const and number of arguments and we
  // don't have a way to currently distinguish them so we will bail out now.
  if (!contains_lambda_identifier(name_to_use))
    return optional_info;

  if (vtable_cu && !has_invoke) {
    lldb::FunctionSP func_sp =
        vtable_cu->FindFunction([name_to_use](const FunctionSP &f) {
          auto name = f->GetName().GetStringRef();
          if (name.starts_with(name_to_use) && name.contains("operator"))
            return true;

          return false;
        });

    if (func_sp) {
      calculate_symbol_context_helper(func_sp, scl);
    }
  }

  if (symbol == nullptr)
    return optional_info;

  // Case 1 or 3
  if (scl.GetSize() >= 1) {
    optional_info = line_entry_helper(target, scl[0], symbol,
                                      first_template_parameter, has_invoke);
  }

  CallableLookupCache[func_to_match] = optional_info;

  return optional_info;
}

lldb::ThreadPlanSP
CPPLanguageRuntime::GetStepThroughTrampolinePlan(Thread &thread,
                                                 bool stop_others) {
  ThreadPlanSP ret_plan_sp;

  lldb::addr_t curr_pc = thread.GetRegisterContext()->GetPC();

  TargetSP target_sp(thread.CalculateTarget());

  if (!target_sp->HasLoadedSections())
    return ret_plan_sp;

  Address pc_addr_resolved;
  SymbolContext sc;
  Symbol *symbol;

  if (!target_sp->ResolveLoadAddress(curr_pc, pc_addr_resolved))
    return ret_plan_sp;

  target_sp->GetImages().ResolveSymbolContextForAddress(
      pc_addr_resolved, eSymbolContextEverything, sc);
  symbol = sc.symbol;

  if (symbol == nullptr)
    return ret_plan_sp;

  llvm::StringRef function_name(symbol->GetName().GetCString());

  // Handling the case where we are attempting to step into std::function.
  // The behavior will be that we will attempt to obtain the wrapped
  // callable via FindLibCppStdFunctionCallableInfo() and if we find it we
  // will return a ThreadPlanRunToAddress to the callable. Therefore we will
  // step into the wrapped callable.
  //
  bool found_expected_start_string =
      function_name.starts_with("std::__1::function<");

  if (!found_expected_start_string)
    return ret_plan_sp;

  AddressRange range_of_curr_func;
  sc.GetAddressRange(eSymbolContextEverything, 0, false, range_of_curr_func);

  StackFrameSP frame = thread.GetStackFrameAtIndex(0);

  if (frame) {
    ValueObjectSP value_sp = frame->FindVariable(g_this);

    CPPLanguageRuntime::LibCppStdFunctionCallableInfo callable_info =
        FindLibCppStdFunctionCallableInfo(value_sp);

    if (callable_info.callable_case != LibCppStdFunctionCallableCase::Invalid &&
        value_sp->GetValueIsValid()) {
      // We found the std::function wrapped callable and we have its address.
      // We now create a ThreadPlan to run to the callable.
      ret_plan_sp = std::make_shared<ThreadPlanRunToAddress>(
          thread, callable_info.callable_address, stop_others);
      return ret_plan_sp;
    } else {
      // We are in std::function but we could not obtain the callable.
      // We create a ThreadPlan to keep stepping through using the address range
      // of the current function.
      ret_plan_sp = std::make_shared<ThreadPlanStepInRange>(
          thread, range_of_curr_func, sc, nullptr, eOnlyThisThread,
          eLazyBoolYes, eLazyBoolYes);
      return ret_plan_sp;
    }
  }

  return ret_plan_sp;
}

bool CPPLanguageRuntime::IsSymbolARuntimeThunk(const Symbol &symbol) {
  llvm::StringRef mangled_name =
      symbol.GetMangled().GetMangledName().GetStringRef();
  // Virtual function overriding from a non-virtual base use a "Th" prefix.
  // Virtual function overriding from a virtual base must use a "Tv" prefix.
  // Virtual function overriding thunks with covariant returns use a "Tc"
  // prefix.
  return mangled_name.starts_with("_ZTh") || mangled_name.starts_with("_ZTv") ||
         mangled_name.starts_with("_ZTc");
}

bool CPPLanguageRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  const bool check_cxx = true;
  const bool check_objc = false;
  return in_value.GetCompilerType().IsPossibleDynamicType(nullptr, check_cxx,
                                                          check_objc);
}

bool CPPLanguageRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &dynamic_address,
    Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer) {
  class_type_or_name.Clear();
  value_type = Value::ValueType::Scalar;

  if (!CouldHaveDynamicValue(in_value))
    return false;

  return m_itanium_runtime.GetDynamicTypeAndAddress(
      in_value, use_dynamic, class_type_or_name, dynamic_address, value_type);
}

TypeAndOrName
CPPLanguageRuntime::FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                     ValueObject &static_value) {
  CompilerType static_type(static_value.GetCompilerType());
  Flags static_type_flags(static_type.GetTypeInfo());

  TypeAndOrName ret(type_and_or_name);
  if (type_and_or_name.HasType()) {
    // The type will always be the type of the dynamic object.  If our parent's
    // type was a pointer, then our type should be a pointer to the type of the
    // dynamic object.  If a reference, then the original type should be
    // okay...
    CompilerType orig_type = type_and_or_name.GetCompilerType();
    CompilerType corrected_type = orig_type;
    if (static_type_flags.AllSet(eTypeIsPointer))
      corrected_type = orig_type.GetPointerType();
    else if (static_type_flags.AllSet(eTypeIsReference))
      corrected_type = orig_type.GetLValueReferenceType();
    ret.SetCompilerType(corrected_type);
  } else {
    // If we are here we need to adjust our dynamic type name to include the
    // correct & or * symbol
    std::string corrected_name(type_and_or_name.GetName().GetCString());
    if (static_type_flags.AllSet(eTypeIsPointer))
      corrected_name.append(" *");
    else if (static_type_flags.AllSet(eTypeIsReference))
      corrected_name.append(" &");
    // the parent type should be a correctly pointer'ed or referenc'ed type
    ret.SetCompilerType(static_type);
    ret.SetName(corrected_name.c_str());
  }
  return ret;
}

LanguageRuntime *
CPPLanguageRuntime::CreateInstance(Process *process,
                                   lldb::LanguageType language) {
  if (language == eLanguageTypeC_plus_plus ||
      language == eLanguageTypeC_plus_plus_03 ||
      language == eLanguageTypeC_plus_plus_11 ||
      language == eLanguageTypeC_plus_plus_14)
    return new CPPLanguageRuntime(process);
  else
    return nullptr;
}

void CPPLanguageRuntime::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "C++ language runtime", CreateInstance,
      [](CommandInterpreter &interpreter) -> lldb::CommandObjectSP {
        return CommandObjectSP(new CommandObjectCPlusPlus(interpreter));
      });
}

void CPPLanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::Expected<LanguageRuntime::VTableInfo>
CPPLanguageRuntime::GetVTableInfo(ValueObject &in_value, bool check_type) {
  return m_itanium_runtime.GetVTableInfo(in_value, check_type);
}

BreakpointResolverSP
CPPLanguageRuntime::CreateExceptionResolver(const BreakpointSP &bkpt,
                                            bool catch_bp, bool throw_bp) {
  return CreateExceptionResolver(bkpt, catch_bp, throw_bp, false);
}

BreakpointResolverSP
CPPLanguageRuntime::CreateExceptionResolver(const BreakpointSP &bkpt,
                                            bool catch_bp, bool throw_bp,
                                            bool for_expressions) {
  std::vector<const char *> exception_names;
  m_itanium_runtime.AppendExceptionBreakpointFunctions(
      exception_names, catch_bp, throw_bp, for_expressions);

  BreakpointResolverSP resolver_sp(new BreakpointResolverName(
      bkpt, exception_names.data(), exception_names.size(),
      eFunctionNameTypeBase, eLanguageTypeUnknown, 0, eLazyBoolNo));

  return resolver_sp;
}

lldb::SearchFilterSP CPPLanguageRuntime::CreateExceptionSearchFilter() {
  Target &target = m_process->GetTarget();

  FileSpecList filter_modules;
  m_itanium_runtime.AppendExceptionBreakpointFilterModules(filter_modules,
                                                           target);
  return target.GetSearchFilterForModuleList(&filter_modules);
}

lldb::BreakpointSP CPPLanguageRuntime::CreateExceptionBreakpoint(
    bool catch_bp, bool throw_bp, bool for_expressions, bool is_internal) {
  Target &target = m_process->GetTarget();
  FileSpecList filter_modules;
  BreakpointResolverSP exception_resolver_sp =
      CreateExceptionResolver(nullptr, catch_bp, throw_bp, for_expressions);
  SearchFilterSP filter_sp(CreateExceptionSearchFilter());
  const bool hardware = false;
  const bool resolve_indirect_functions = false;
  return target.CreateBreakpoint(filter_sp, exception_resolver_sp, is_internal,
                                 hardware, resolve_indirect_functions);
}

void CPPLanguageRuntime::SetExceptionBreakpoints() {
  if (!m_process)
    return;

  const bool catch_bp = false;
  const bool throw_bp = true;
  const bool is_internal = true;
  const bool for_expressions = true;

  // For the exception breakpoints set by the Expression parser, we'll be a
  // little more aggressive and stop at exception allocation as well.

  if (m_cxx_exception_bp_sp) {
    m_cxx_exception_bp_sp->SetEnabled(true);
  } else {
    m_cxx_exception_bp_sp = CreateExceptionBreakpoint(
        catch_bp, throw_bp, for_expressions, is_internal);
    if (m_cxx_exception_bp_sp)
      m_cxx_exception_bp_sp->SetBreakpointKind("c++ exception");
  }
}

void CPPLanguageRuntime::ClearExceptionBreakpoints() {
  if (!m_process)
    return;

  if (m_cxx_exception_bp_sp) {
    m_cxx_exception_bp_sp->SetEnabled(false);
  }
}

bool CPPLanguageRuntime::ExceptionBreakpointsAreSet() {
  return m_cxx_exception_bp_sp && m_cxx_exception_bp_sp->IsEnabled();
}

bool CPPLanguageRuntime::ExceptionBreakpointsExplainStop(
    lldb::StopInfoSP stop_reason) {
  if (!m_process)
    return false;

  if (!stop_reason || stop_reason->GetStopReason() != eStopReasonBreakpoint)
    return false;

  uint64_t break_site_id = stop_reason->GetValue();
  return m_process->GetBreakpointSiteList().StopPointSiteContainsBreakpoint(
      break_site_id, m_cxx_exception_bp_sp->GetID());
}

lldb::ValueObjectSP
CPPLanguageRuntime::GetExceptionObjectForThread(lldb::ThreadSP thread_sp) {
  return m_itanium_runtime.GetExceptionObjectForThread(std::move(thread_sp));
}
