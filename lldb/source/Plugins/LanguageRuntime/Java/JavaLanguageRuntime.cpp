//===-- JavaLanguageRuntime.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JavaLanguageRuntime.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/JavaASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

JavaLanguageRuntime::JavaLanguageRuntime(Process *process)
    : LanguageRuntime(process) {}

LanguageRuntime *
JavaLanguageRuntime::CreateInstance(Process *process,
                                    lldb::LanguageType language) {
  if (language == eLanguageTypeJava)
    return new JavaLanguageRuntime(process);
  return nullptr;
}

void JavaLanguageRuntime::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), "Java language runtime",
                                CreateInstance);
}

void JavaLanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString JavaLanguageRuntime::GetPluginNameStatic() {
  static ConstString g_name("java");
  return g_name;
}

lldb_private::ConstString JavaLanguageRuntime::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t JavaLanguageRuntime::GetPluginVersion() { return 1; }

bool JavaLanguageRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  return true;
}

static ConstString GetDynamicTypeId(ExecutionContext *exe_ctx, Target *target,
                                    ValueObject &in_value) {
  SymbolContext sc;
  TypeList class_types;
  llvm::DenseSet<SymbolFile *> searched_symbol_files;
  size_t num_matches = target->GetImages().FindTypes(
      sc, ConstString("Object"),
      true, // name_is_fully_qualified
      UINT32_MAX, searched_symbol_files, class_types);
  for (size_t i = 0; i < num_matches; ++i) {
    TypeSP type_sp = class_types.GetTypeAtIndex(i);
    CompilerType compiler_type = type_sp->GetFullCompilerType();

    if (compiler_type.GetMinimumLanguage() != eLanguageTypeJava ||
        compiler_type.GetTypeName() != ConstString("java::lang::Object"))
      continue;

    if (compiler_type.GetCompleteType() && compiler_type.IsCompleteType()) {
      uint64_t type_id = JavaASTContext::CalculateDynamicTypeId(
          exe_ctx, compiler_type, in_value);
      if (type_id != UINT64_MAX) {
        char id[32];
        snprintf(id, sizeof(id), "0x%" PRIX64, type_id);
        return ConstString(id);
      }
    }
  }
  return ConstString();
}

bool JavaLanguageRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &dynamic_address,
    Value::ValueType &value_type) {
  class_type_or_name.Clear();

  // null references don't have a dynamic type
  if (in_value.IsNilReference())
    return false;

  ExecutionContext exe_ctx(in_value.GetExecutionContextRef());
  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return false;

  ConstString linkage_name;
  CompilerType in_type = in_value.GetCompilerType();
  if (in_type.IsPossibleDynamicType(nullptr, false, false))
    linkage_name = GetDynamicTypeId(&exe_ctx, target, in_value);
  else
    linkage_name = JavaASTContext::GetLinkageName(in_type);

  if (!linkage_name)
    return false;

  class_type_or_name.SetName(in_type.GetNonReferenceType().GetTypeName());

  SymbolContext sc;
  TypeList class_types;
  llvm::DenseSet<SymbolFile *> searched_symbol_files;
  size_t num_matches = target->GetImages().FindTypes(
      sc, linkage_name,
      true, // name_is_fully_qualified
      UINT32_MAX, searched_symbol_files, class_types);

  for (size_t i = 0; i < num_matches; ++i) {
    TypeSP type_sp = class_types.GetTypeAtIndex(i);
    CompilerType compiler_type = type_sp->GetFullCompilerType();

    if (compiler_type.GetMinimumLanguage() != eLanguageTypeJava)
      continue;

    if (compiler_type.GetCompleteType() && compiler_type.IsCompleteType()) {
      class_type_or_name.SetTypeSP(type_sp);

      Value &value = in_value.GetValue();
      value_type = value.GetValueType();
      dynamic_address.SetRawAddress(value.GetScalar().ULongLong(0));
      return true;
    }
  }
  return false;
}

TypeAndOrName
JavaLanguageRuntime::FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                      ValueObject &static_value) {
  CompilerType static_type(static_value.GetCompilerType());

  TypeAndOrName ret(type_and_or_name);
  if (type_and_or_name.HasType()) {
    CompilerType orig_type = type_and_or_name.GetCompilerType();
    if (static_type.IsReferenceType())
      ret.SetCompilerType(orig_type.GetLValueReferenceType());
  }
  return ret;
}
