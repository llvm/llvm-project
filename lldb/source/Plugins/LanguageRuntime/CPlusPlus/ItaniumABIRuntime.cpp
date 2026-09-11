//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ItaniumABIRuntime.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/FunctionCaller.h"
#include "lldb/Utility/LLDBLog.h"
#include "llvm/Support/Error.h"

#include "lldb/Symbol/VariableList.h"
#include "lldb/ValueObject/ValueObjectVariable.h"

using namespace lldb;
using namespace lldb_private;

static const char *vtable_demangled_prefix = "vtable for ";

ItaniumABIRuntime::ItaniumABIRuntime(Process *process)
    : CommonABIRuntime(process) {}

llvm::StringRef ItaniumABIRuntime::GetName() const {
  return "Itanium ABI runtime";
}

bool ItaniumABIRuntime::IsVTableSymbol(Mangled &mangled) const {
  return mangled.GetDemangledName().GetStringRef().starts_with(
      vtable_demangled_prefix);
}

/// clang emits a `DW_TAG_variable` called `__clang_vtable` inside the
/// enclosing class this function uses that to resolve the dynamic type of
/// `in_value`
TypeAndOrName ItaniumABIRuntime::FindTypeInfoWithClangVTable(
    ValueObject &in_value,
    const LanguageRuntime::VTableInfo &vtable_info) const {
  ModuleSP module_sp = vtable_info.symbol->CalculateSymbolContextModule();
  if (!module_sp)
    return TypeAndOrName();

  auto vtableBase = vtable_info.symbol->GetLoadAddress(&m_process->GetTarget());

  VariableList vars;

  // SymbolFile doesn't provide an API to lookup a global variable with a
  // specific location, so we have to resort to an unbounded name lookup
  module_sp->FindGlobalVariables(ConstString("__clang_vtable"),
                                 CompilerDeclContext(), -1, vars);

  Log *log = GetLog(LLDBLog::Object);
  LLDB_LOG(log, "{0:x}: found {1} __clang_vtable variables",
           in_value.GetPointerValue().address, vars.GetSize());

  for (lldb::VariableSP var : vars) {
    auto valobj = ValueObjectVariable::Create(m_process, var);
    if (valobj->GetLoadAddress() != vtableBase)
      continue;

    auto type_sp = var->GetEnclosingType();
    if (!type_sp) {
      LLDB_LOG(
          log,
          "{0:x}: Found __clang_vtable at {1:x}, but not the enclosing type!",
          in_value.GetPointerValue().address, valobj->GetLoadAddress());

      // Failure to find the type is either an error in the debug info, or the
      // symptom of a module with debug kind info which doesn't support this
      // sort of lookup. Carry on
      continue;
    }

    if (!TypeSystemClang::IsCXXClassType(type_sp->GetForwardCompilerType())) {
      LLDB_LOG(log,
               "{0:x}: Found __clang_vtable at {1:x} for '{2}' which is not a "
               "CXXClassType. Ignoring",
               in_value.GetPointerValue().address, valobj->GetLoadAddress(),
               type_sp->GetQualifiedName().AsCString(""));
      continue;
    }

    LLDB_LOG(log,
             "{0:x}: static-type = '{1}' has dynamic type: uid={2:x}, "
             "type-name='{3}'",
             in_value.GetPointerValue().address,
             in_value.GetTypeName().AsCString(""), type_sp->GetID(),
             type_sp->GetName().GetCString());

    return TypeAndOrName(type_sp);
  }

  return TypeAndOrName();
}

TypeAndOrName
ItaniumABIRuntime::GetTypeInfo(ValueObject &in_value,
                               const LanguageRuntime::VTableInfo &vtable_info) {
  if (vtable_info.addr.IsSectionOffset()) {
    // See if we have cached info for this type already
    TypeAndOrName type_info = GetDynamicTypeInfo(vtable_info.addr);
    if (type_info)
      return type_info;

    if (vtable_info.symbol) {
      type_info = FindTypeInfoWithDemangling(in_value, vtable_info);
      if (type_info) {
        SetDynamicTypeInfo(vtable_info.addr, type_info);
        return type_info;
      }

      type_info = FindTypeInfoWithClangVTable(in_value, vtable_info);
      if (type_info) {
        SetDynamicTypeInfo(vtable_info.addr, type_info);
        return type_info;
      }
    }
  }
  return TypeAndOrName();
}

TypeAndOrName ItaniumABIRuntime::FindTypeInfoWithDemangling(
    ValueObject &in_value,
    const LanguageRuntime::VTableInfo &vtable_info) const {
  if (vtable_info.addr.IsSectionOffset()) {
    if (vtable_info.symbol) {
      TypeAndOrName type_info;

      Log *log = GetLog(LLDBLog::Object);
      llvm::StringRef symbol_name =
          vtable_info.symbol->GetMangled().GetDemangledName().GetStringRef();
      LLDB_LOGF(log,
                "0x%16.16" PRIx64
                ": static-type = '%s' has vtable symbol '%s'\n",
                in_value.GetPointerValue().address,
                in_value.GetTypeName().GetCString(), symbol_name.str().c_str());

      // We are a C++ class, that's good.  Get the class name and look it
      // up:
      llvm::StringRef class_name = symbol_name;
      class_name.consume_front(vtable_demangled_prefix);
      // We know the class name is absolute, so tell FindTypes that by
      // prefixing it with the root namespace:
      std::string lookup_name("::");
      lookup_name.append(class_name.data(), class_name.size());

      type_info.SetName(class_name);
      bool any_found = false;
      TypeSP type_sp = LookupTypeByName(
          class_name, vtable_info.symbol->CalculateSymbolContextModule(),
          any_found);
      if (!any_found)
        return TypeAndOrName(); // Type is not dynamic.

      if (type_sp) {
        LLDB_LOGF(log,
                  "0x%16.16" PRIx64
                  ": static-type = '%s' has dynamic type: uid={0x%" PRIx64
                  "}, type-name='%s'\n",
                  in_value.GetPointerValue().address,
                  in_value.GetTypeName().AsCString(""), type_sp->GetID(),
                  type_sp->GetName().GetCString());
        type_info.SetTypeSP(std::move(type_sp));
      }
      return type_info;
    }
  }
  return TypeAndOrName();
}

bool ItaniumABIRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    const LanguageRuntime::VTableInfo &vtable_info,
    TypeAndOrName &class_type_or_name, Address &dynamic_address) {
  // For Itanium, if the type has a vtable pointer in the object, it will be at
  // offset 0 in the object.  That will point to the "address point" within the
  // vtable (not the beginning of the vtable.)  We can then look up the symbol
  // containing this "address point" and that symbol's name demangled will
  // contain the full class name. The second pointer above the "address point"
  // is the "offset_to_top".  We'll use that to get the start of the value
  // object which holds the dynamic type.

  // Check if we have a vtable pointer in this value. If we don't it will
  // return an error, else it will return a valid resolved address. We don't
  // want GetVTableInfo to check the type since we accept void * as a possible
  // dynamic type and that won't pass the type check. We already checked the
  // type above in CouldHaveDynamicValue(...).
  class_type_or_name = GetTypeInfo(in_value, vtable_info);

  if (!class_type_or_name)
    return false;

  CompilerType type = class_type_or_name.GetCompilerType();
  // There can only be one type with a given name, so we've just found
  // duplicate definitions, and this one will do as well as any other. We
  // don't consider something to have a dynamic type if it is the same as
  // the static type.  So compare against the value we were handed.
  if (!type)
    return true;

  if (TypeSystemClang::AreTypesSame(in_value.GetCompilerType(), type)) {
    // The dynamic type we found was the same type, so we don't have a
    // dynamic type here...
    return false;
  }

  // The offset_to_top is two pointers above the vtable pointer.
  Target &target = m_process->GetTarget();
  const addr_t vtable_load_addr = vtable_info.addr.GetLoadAddress(&target);
  if (vtable_load_addr == LLDB_INVALID_ADDRESS)
    return false;
  const uint32_t addr_byte_size = m_process->GetAddressByteSize();
  const lldb::addr_t offset_to_top_location =
      vtable_load_addr - 2 * addr_byte_size;
  // Watch for underflow, offset_to_top_location should be less than
  // vtable_load_addr
  if (offset_to_top_location >= vtable_load_addr)
    return false;
  Status error;
  const int64_t offset_to_top = target.ReadSignedIntegerFromMemory(
      Address(offset_to_top_location), addr_byte_size, INT64_MIN, error);

  if (offset_to_top == INT64_MIN)
    return false;
  // So the dynamic type is a value that starts at offset_to_top above
  // the original address.
  lldb::addr_t dynamic_addr =
      in_value.GetPointerValue().address + offset_to_top;
  if (!m_process->GetTarget().ResolveLoadAddress(dynamic_addr,
                                                 dynamic_address)) {
    dynamic_address.SetRawAddress(dynamic_addr);
  }
  return true;
}

void ItaniumABIRuntime::AppendExceptionBreakpointFunctions(
    std::vector<const char *> &names, bool catch_bp, bool throw_bp,
    bool for_expressions) {
  // One complication here is that most users DON'T want to stop at
  // __cxa_allocate_expression, but until we can do anything better with
  // predicting unwinding the expression parser does.  So we have two forms of
  // the exception breakpoints, one for expressions that leaves out
  // __cxa_allocate_exception, and one that includes it. The
  // SetExceptionBreakpoints does the latter, the CreateExceptionBreakpoint in
  // the runtime the former.
  static const char *g_catch_name = "__cxa_begin_catch";
  static const char *g_throw_name1 = "__cxa_throw";
  static const char *g_throw_name2 = "__cxa_rethrow";
  static const char *g_exception_throw_name = "__cxa_allocate_exception";

  if (catch_bp)
    names.push_back(g_catch_name);

  if (throw_bp) {
    names.push_back(g_throw_name1);
    names.push_back(g_throw_name2);
  }

  if (for_expressions)
    names.push_back(g_exception_throw_name);
}

void ItaniumABIRuntime::AppendExceptionBreakpointFilterModules(
    FileSpecList &filter_modules, const Target &target) {
  if (target.GetArchitecture().GetTriple().getVendor() == llvm::Triple::Apple) {
    // Limit the number of modules that are searched for these breakpoints for
    // Apple binaries.
    filter_modules.EmplaceBack("libc++abi.dylib");
    filter_modules.EmplaceBack("libSystem.B.dylib");
    filter_modules.EmplaceBack("libc++abi.1.0.dylib");
    filter_modules.EmplaceBack("libc++abi.1.dylib");
  }
}

ValueObjectSP
ItaniumABIRuntime::GetExceptionObjectForThread(ThreadSP thread_sp) {
  if (!thread_sp->SafeToCallFunctions())
    return {};

  TypeSystemClangSP scratch_ts_sp =
      ScratchTypeSystemClang::GetForTarget(m_process->GetTarget());
  if (!scratch_ts_sp)
    return {};

  CompilerType voidstar =
      scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType();

  DiagnosticManager diagnostics;
  ExecutionContext exe_ctx;
  EvaluateExpressionOptions options;

  options.SetUnwindOnError(true);
  options.SetIgnoreBreakpoints(true);
  options.SetStopOthers(true);
  options.SetTimeout(m_process->GetUtilityExpressionTimeout());
  options.SetTryAllThreads(false);
  thread_sp->CalculateExecutionContext(exe_ctx);

  const ModuleList &modules = m_process->GetTarget().GetImages();
  SymbolContextList contexts;
  SymbolContext context;

  modules.FindSymbolsWithNameAndType(
      ConstString("__cxa_current_exception_type"), eSymbolTypeCode, contexts);
  contexts.GetContextAtIndex(0, context);
  if (!context.symbol) {
    return {};
  }
  Address addr = context.symbol->GetAddress();

  Status error;
  FunctionCaller *function_caller =
      m_process->GetTarget().GetFunctionCallerForLanguage(
          eLanguageTypeC, voidstar, addr, ValueList(), "caller", error);

  ExpressionResults func_call_ret;
  Value results;
  func_call_ret = function_caller->ExecuteFunction(exe_ctx, nullptr, options,
                                                   diagnostics, results);
  if (func_call_ret != eExpressionCompleted || !error.Success()) {
    return ValueObjectSP();
  }

  size_t ptr_size = m_process->GetAddressByteSize();
  addr_t result_ptr = results.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
  llvm::Expected<lldb::addr_t> exception_addr =
      m_process->ReadPointerFromMemory(result_ptr - ptr_size);

  if (!exception_addr) {
    llvm::consumeError(exception_addr.takeError());
    return ValueObjectSP();
  }

  lldb_private::formatters::InferiorSizedWord exception_isw(*exception_addr,
                                                            *m_process);
  ValueObjectSP exception = ValueObject::CreateValueObjectFromData(
      "exception", exception_isw.GetAsData(m_process->GetByteOrder()), exe_ctx,
      voidstar);
  ValueObjectSP dyn_exception =
      exception->GetDynamicValue(eDynamicDontRunTarget);
  // If we succeed in making a dynamic value, return that:
  if (dyn_exception)
    return dyn_exception;

  return exception;
}
