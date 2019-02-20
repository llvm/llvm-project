//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangPersistentVariables.h"
#include "lldb/Expression/IRExecutionUnit.h"

#include "lldb/Core/Value.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "lldb/Symbol/SwiftASTContext.h" // Needed for llvm::isa<SwiftASTContext>(...)
#include "lldb/Symbol/TypeSystem.h"

#include "swift/AST/Decl.h"
#include "swift/AST/Pattern.h"
#include "clang/AST/Decl.h"

#include "llvm/ADT/StringMap.h"

using namespace lldb;
using namespace lldb_private;

ClangPersistentVariables::ClangPersistentVariables()
    : lldb_private::PersistentExpressionState(LLVMCastKind::eKindClang),
      m_next_persistent_variable_id(0), m_next_persistent_error_id(0) {}

ExpressionVariableSP ClangPersistentVariables::CreatePersistentVariable(
    const lldb::ValueObjectSP &valobj_sp) {
  return AddNewlyConstructedVariable(new ClangExpressionVariable(valobj_sp));
}

ExpressionVariableSP ClangPersistentVariables::CreatePersistentVariable(
    ExecutionContextScope *exe_scope, const ConstString &name,
    const CompilerType &compiler_type, lldb::ByteOrder byte_order,
    uint32_t addr_byte_size) {
  return AddNewlyConstructedVariable(new ClangExpressionVariable(
      exe_scope, name, compiler_type, byte_order, addr_byte_size));
}

void ClangPersistentVariables::RemovePersistentVariable(
    lldb::ExpressionVariableSP variable) {
  if (!variable)
    return;

  RemoveVariable(variable);

  const char *name = variable->GetName().AsCString();

  if (*name != '$')
    return;
  name++;

  bool is_error = false;

  if (llvm::isa<SwiftASTContext>(variable->GetCompilerType().GetTypeSystem())) {
    switch (*name) {
    case 'R':
      break;
    case 'E':
      is_error = true;
      break;
    default:
      return;
    }
    name++;
  }

  uint32_t value = strtoul(name, NULL, 0);
  if (is_error) {
    if (value == m_next_persistent_error_id - 1)
      m_next_persistent_error_id--;
  } else {
    if (value == m_next_persistent_variable_id - 1)
      m_next_persistent_variable_id--;
  }
}

void ClangPersistentVariables::RegisterPersistentDecl(const ConstString &name,
                                                      clang::NamedDecl *decl) {
  m_persistent_decls.insert(
      std::pair<const char *, clang::NamedDecl *>(name.GetCString(), decl));

  if (clang::EnumDecl *enum_decl = llvm::dyn_cast<clang::EnumDecl>(decl)) {
    for (clang::EnumConstantDecl *enumerator_decl : enum_decl->enumerators()) {
      m_persistent_decls.insert(std::pair<const char *, clang::NamedDecl *>(
          ConstString(enumerator_decl->getNameAsString()).GetCString(),
          enumerator_decl));
    }
  }
}

clang::NamedDecl *
ClangPersistentVariables::GetPersistentDecl(const ConstString &name) {
  PersistentDeclMap::const_iterator i =
      m_persistent_decls.find(name.GetCString());

  if (i == m_persistent_decls.end())
    return NULL;
  else
    return i->second;
}
