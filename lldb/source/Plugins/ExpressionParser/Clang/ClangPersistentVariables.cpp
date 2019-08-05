//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangPersistentVariables.h"

#include "lldb/Core/Value.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "clang/AST/Decl.h"

#include "llvm/ADT/StringMap.h"

using namespace lldb;
using namespace lldb_private;

ClangPersistentVariables::ClangPersistentVariables()
    : lldb_private::PersistentExpressionState(LLVMCastKind::eKindClang) {}

ExpressionVariableSP ClangPersistentVariables::CreatePersistentVariable(
    const lldb::ValueObjectSP &valobj_sp) {
  return AddNewlyConstructedVariable(new ClangExpressionVariable(valobj_sp));
}

ExpressionVariableSP ClangPersistentVariables::CreatePersistentVariable(
    ExecutionContextScope *exe_scope, ConstString name,
    const CompilerType &compiler_type, lldb::ByteOrder byte_order,
    uint32_t addr_byte_size) {
  return AddNewlyConstructedVariable(new ClangExpressionVariable(
      exe_scope, name, compiler_type, byte_order, addr_byte_size));
}

void ClangPersistentVariables::RemovePersistentVariable(
    lldb::ExpressionVariableSP variable) {
  RemoveVariable(variable);

  // Check if the removed variable was the last one that was created. If yes,
  // reuse the variable id for the next variable.

  // Nothing to do if we have not assigned a variable id so far.
  if (m_next_persistent_variable_id == 0)
    return;

  llvm::StringRef name = variable->GetName().GetStringRef();
  // Remove the prefix from the variable that only the indes is left.
  if (!name.consume_front(GetPersistentVariablePrefix(false)))
    return;

  // Check if the variable contained a variable id.
  uint32_t variable_id;
  if (name.getAsInteger(10, variable_id))
    return;
  // If it's the most recent variable id that was assigned, make sure that this
  // variable id will be used for the next persistent variable.
  if (variable_id == m_next_persistent_variable_id - 1)
    m_next_persistent_variable_id--;
}

llvm::Optional<CompilerType>
ClangPersistentVariables::GetCompilerTypeFromPersistentDecl(
    ConstString type_name) {
  CompilerType compiler_type;
  if (clang::TypeDecl *tdecl = llvm::dyn_cast_or_null<clang::TypeDecl>(
          GetPersistentDecl(type_name))) {
    compiler_type.SetCompilerType(
        ClangASTContext::GetASTContext(&tdecl->getASTContext()),
        reinterpret_cast<lldb::opaque_compiler_type_t>(
            const_cast<clang::Type *>(tdecl->getTypeForDecl())));
    return compiler_type;
  }
  return llvm::None;
}

void ClangPersistentVariables::RegisterPersistentDecl(ConstString name,
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
ClangPersistentVariables::GetPersistentDecl(ConstString name) {
  PersistentDeclMap::const_iterator i =
      m_persistent_decls.find(name.GetCString());

  if (i == m_persistent_decls.end())
    return nullptr;
  else
    return i->second;
}
