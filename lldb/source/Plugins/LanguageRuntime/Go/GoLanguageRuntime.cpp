//===-- GoLanguageRuntime.cpp --------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GoLanguageRuntime.h"

#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Symbol/GoASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/Twine.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

namespace {
ValueObjectSP GetChild(ValueObject &obj, const char *name,
                       bool dereference = true) {
  ConstString name_const_str(name);
  ValueObjectSP result = obj.GetChildMemberWithName(name_const_str, true);
  if (dereference && result && result->IsPointerType()) {
    Error err;
    result = result->Dereference(err);
    if (err.Fail())
      result.reset();
  }
  return result;
}

ConstString ReadString(ValueObject &str, Process *process) {
  ConstString result;
  ValueObjectSP data = GetChild(str, "str", false);
  ValueObjectSP len = GetChild(str, "len");
  if (len && data) {
    Error err;
    lldb::addr_t addr = data->GetPointerValue();
    if (addr == LLDB_INVALID_ADDRESS)
      return result;
    uint64_t byte_size = len->GetValueAsUnsigned(0);
    char *buf = new char[byte_size + 1];
    buf[byte_size] = 0;
    size_t bytes_read = process->ReadMemory(addr, buf, byte_size, err);
    if (!(err.Fail() || bytes_read != byte_size))
      result = ConstString(buf, bytes_read);
    delete[] buf;
  }
  return result;
}

ConstString ReadTypeName(ValueObjectSP type, Process *process) {
  if (ValueObjectSP uncommon = GetChild(*type, "x")) {
    ValueObjectSP name = GetChild(*uncommon, "name");
    ValueObjectSP package = GetChild(*uncommon, "pkgpath");
    if (name && name->GetPointerValue() != 0 && package &&
        package->GetPointerValue() != 0) {
      ConstString package_const_str = ReadString(*package, process);
      ConstString name_const_str = ReadString(*name, process);
      if (package_const_str.GetLength() == 0)
        return name_const_str;
      return ConstString((package_const_str.GetStringRef() + "." +
                          name_const_str.GetStringRef())
                             .str());
    }
  }
  ValueObjectSP name = GetChild(*type, "_string");
  if (name)
    return ReadString(*name, process);
  return ConstString("");
}

CompilerType LookupRuntimeType(ValueObjectSP type, ExecutionContext *exe_ctx,
                               bool *is_direct) {
  uint8_t kind = GetChild(*type, "kind")->GetValueAsUnsigned(0);
  *is_direct = GoASTContext::IsDirectIface(kind);
  if (GoASTContext::IsPointerKind(kind)) {
    CompilerType type_ptr = type->GetCompilerType().GetPointerType();
    Error err;
    ValueObjectSP elem =
        type->CreateValueObjectFromAddress("elem", type->GetAddressOf() +
                                                       type->GetByteSize(),
                                           *exe_ctx, type_ptr)
            ->Dereference(err);
    if (err.Fail())
      return CompilerType();
    bool tmp_direct;
    return LookupRuntimeType(elem, exe_ctx, &tmp_direct).GetPointerType();
  }
  Target *target = exe_ctx->GetTargetPtr();
  Process *process = exe_ctx->GetProcessPtr();

  ConstString const_typename = ReadTypeName(type, process);
  if (const_typename.GetLength() == 0)
    return CompilerType();

  SymbolContext sc;
  TypeList type_list;
  llvm::DenseSet<SymbolFile *> searched_symbol_files;
  uint32_t num_matches = target->GetImages().FindTypes(
      sc, const_typename, false, 2, searched_symbol_files, type_list);
  if (num_matches > 0) {
    return type_list.GetTypeAtIndex(0)->GetFullCompilerType();
  }
  return CompilerType();
}
}

bool GoLanguageRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  return GoASTContext::IsGoInterface(in_value.GetCompilerType());
}

bool GoLanguageRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &dynamic_address,
    Value::ValueType &value_type) {
  value_type = Value::eValueTypeScalar;
  class_type_or_name.Clear();
  if (CouldHaveDynamicValue(in_value)) {
    Error err;
    ValueObjectSP iface = in_value.GetStaticValue();
    ValueObjectSP data_sp = GetChild(*iface, "data", false);
    if (!data_sp)
      return false;

    if (ValueObjectSP tab = GetChild(*iface, "tab"))
      iface = tab;
    ValueObjectSP type = GetChild(*iface, "_type");
    if (!type) {
      return false;
    }

    bool direct;
    ExecutionContext exe_ctx(in_value.GetExecutionContextRef());
    CompilerType final_type = LookupRuntimeType(type, &exe_ctx, &direct);
    if (!final_type)
      return false;
    if (direct) {
      class_type_or_name.SetCompilerType(final_type);
    } else {
      // TODO: implement reference types or fix caller to support dynamic types
      // that aren't pointers
      // so we don't have to introduce this extra pointer.
      class_type_or_name.SetCompilerType(final_type.GetPointerType());
    }

    dynamic_address.SetLoadAddress(data_sp->GetPointerValue(),
                                   exe_ctx.GetTargetPtr());

    return true;
  }
  return false;
}

TypeAndOrName
GoLanguageRuntime::FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                    ValueObject &static_value) {
  return type_and_or_name;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
GoLanguageRuntime::CreateInstance(Process *process,
                                  lldb::LanguageType language) {
  if (language == eLanguageTypeGo)
    return new GoLanguageRuntime(process);
  else
    return NULL;
}

void GoLanguageRuntime::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Go Language Runtime",
                                CreateInstance);
}

void GoLanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString GoLanguageRuntime::GetPluginNameStatic() {
  static ConstString g_name("golang");
  return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString GoLanguageRuntime::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t GoLanguageRuntime::GetPluginVersion() { return 1; }
