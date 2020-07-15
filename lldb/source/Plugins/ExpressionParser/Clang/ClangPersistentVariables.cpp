//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangPersistentVariables.h"
#include "ClangASTImporter.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "Plugins/TypeSystem/Swift/SwiftASTContext.h" // Needed for llvm::isa<SwiftASTContext>(...)
#include "lldb/Symbol/TypeSystem.h"

#include "swift/AST/Decl.h"
#include "swift/AST/Pattern.h"
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
  if (!variable)
    return;

  RemoveVariable(variable);

  const char *name = variable->GetName().AsCString();

  if (*name != '$')
    return;
  name++;

  bool is_error = false;

  if (variable->GetCompilerType().GetTypeSystem()->SupportsLanguage(
          lldb::eLanguageTypeSwift)) {
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

  uint32_t value = strtoul(name, nullptr, 0);
  if (is_error) {
    if (value == m_next_persistent_error_id - 1)
      m_next_persistent_error_id--;
  } else {
    if (value == m_next_persistent_variable_id - 1)
      m_next_persistent_variable_id--;
  }
}

llvm::Optional<CompilerType>
ClangPersistentVariables::GetCompilerTypeFromPersistentDecl(
    ConstString type_name) {
  PersistentDecl p = m_persistent_decls.lookup(type_name.GetCString());

  if (p.m_decl == nullptr)
    return llvm::None;

  if (clang::TypeDecl *tdecl = llvm::dyn_cast<clang::TypeDecl>(p.m_decl)) {
    opaque_compiler_type_t t = static_cast<opaque_compiler_type_t>(
        const_cast<clang::Type *>(tdecl->getTypeForDecl()));
    return CompilerType(p.m_context, t);
  }
  return llvm::None;
}

void ClangPersistentVariables::RegisterPersistentDecl(ConstString name,
                                                      clang::NamedDecl *decl,
                                                      TypeSystemClang *ctx) {
  PersistentDecl p = {decl, ctx};
  m_persistent_decls.insert(std::make_pair(name.GetCString(), p));

  if (clang::EnumDecl *enum_decl = llvm::dyn_cast<clang::EnumDecl>(decl)) {
    for (clang::EnumConstantDecl *enumerator_decl : enum_decl->enumerators()) {
      p = {enumerator_decl, ctx};
      m_persistent_decls.insert(std::make_pair(
          ConstString(enumerator_decl->getNameAsString()).GetCString(), p));
    }
  }
}

clang::NamedDecl *
ClangPersistentVariables::GetPersistentDecl(ConstString name) {
  return m_persistent_decls.lookup(name.GetCString()).m_decl;
}
