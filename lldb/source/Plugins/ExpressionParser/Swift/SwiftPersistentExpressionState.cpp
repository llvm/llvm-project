//===-- SwiftPersistentExpressionState.cpp ----------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftPersistentExpressionState.h"
#include "SwiftExpressionVariable.h"

#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "swift/AST/Decl.h"
#include "swift/AST/ParameterList.h"
#include "swift/AST/Pattern.h"

#include "llvm/ADT/StringMap.h"

using namespace lldb;
using namespace lldb_private;

char SwiftPersistentExpressionState::ID;

SwiftPersistentExpressionState::SwiftPersistentExpressionState()
    : lldb_private::PersistentExpressionState(),
      m_next_persistent_variable_id(0), m_next_persistent_error_id(0) {}

ExpressionVariableSP SwiftPersistentExpressionState::CreatePersistentVariable(
    const lldb::ValueObjectSP &valobj_sp) {
  return AddNewlyConstructedVariable(new SwiftExpressionVariable(valobj_sp));
}

ExpressionVariableSP SwiftPersistentExpressionState::CreatePersistentVariable(
    ExecutionContextScope *exe_scope, ConstString name,
    const CompilerType &compiler_type, lldb::ByteOrder byte_order,
    uint32_t addr_byte_size) {
  return AddNewlyConstructedVariable(new SwiftExpressionVariable(
      exe_scope, name, compiler_type, byte_order, addr_byte_size));
}

void SwiftPersistentExpressionState::RemovePersistentVariable(
    lldb::ExpressionVariableSP variable) {
  if (!variable)
    return;

  RemoveVariable(variable);

  const char *name = variable->GetName().AsCString();

  if (*name != '$')
    return;
  name++;

  bool is_error = false;

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

  uint32_t value = strtoul(name, NULL, 0);
  if (is_error) {
    if (value == m_next_persistent_error_id - 1)
      m_next_persistent_error_id--;
  } else {
    if (value == m_next_persistent_variable_id - 1)
      m_next_persistent_variable_id--;
  }
}

llvm::Optional<CompilerType>
SwiftPersistentExpressionState::GetCompilerTypeFromPersistentDecl(
    ConstString type_name) {
  return llvm::None;
}

bool SwiftPersistentExpressionState::SwiftDeclMap::DeclsAreEquivalent(
    swift::Decl *lhs_decl, swift::Decl *rhs_decl) {
  swift::DeclKind lhs_kind = lhs_decl->getKind();
  swift::DeclKind rhs_kind = rhs_decl->getKind();
  if (lhs_kind != rhs_kind)
    return false;

  switch (lhs_kind) {
  // All the decls that define types of things should only allow one
  // instance, so in this case, erase what is there, and copy in the new
  // version.
  case swift::DeclKind::Class:
  case swift::DeclKind::Struct:
  case swift::DeclKind::Enum:
  case swift::DeclKind::TypeAlias:
  case swift::DeclKind::Protocol:
    return true;

  // For functions, we check that the signature is the same, and only replace it
  // if it
  // is, otherwise we just add it.
  case swift::DeclKind::Func: {
    swift::FuncDecl *lhs_func_decl = llvm::cast<swift::FuncDecl>(lhs_decl);
    swift::FuncDecl *rhs_func_decl = llvm::cast<swift::FuncDecl>(rhs_decl);

    return lhs_func_decl->getInterfaceType()
        ->isEqual(rhs_func_decl->getInterfaceType());
  }

  // Not really sure how to compare operators, so we just do last one wins...
  case swift::DeclKind::InfixOperator:
  case swift::DeclKind::PrefixOperator:
  case swift::DeclKind::PostfixOperator:
    return true;

  default:
    return true;
  }
}

void SwiftPersistentExpressionState::SwiftDeclMap::AddDecl(
    swift::ValueDecl *value_decl, bool check_existing, llvm::StringRef alias) {
  llvm::StringRef name;

  if (alias.empty())
    name = value_decl->getBaseName().getIdentifier().str();
  else
    name = alias;

  auto it = m_swift_decls.find(name);
  if (it == m_swift_decls.end()) {
    m_swift_decls.insert({name, {value_decl}});
    return;
  }

  llvm::SmallVectorImpl<swift::ValueDecl *> &decls = it->second;
  if (check_existing)
    decls.erase(std::remove_if(decls.begin(), decls.end(),
                               [&value_decl](swift::ValueDecl *cur_decl) {
                                 return DeclsAreEquivalent(cur_decl,
                                                           value_decl);
                               }),
                decls.end());

  decls.push_back(value_decl);
}

bool SwiftPersistentExpressionState::SwiftDeclMap::FindMatchingDecls(
    llvm::StringRef name,
    const std::vector<swift::ValueDecl *> &excluding_equivalents,
    std::vector<swift::ValueDecl *> &matches) {
  auto it = m_swift_decls.find(name);
  if (it == m_swift_decls.end())
    return false;
  llvm::SmallVectorImpl<swift::ValueDecl *> &decls = it->second;

  size_t start_num_items = matches.size();
  for (auto &cur_decl : decls)
    if (std::none_of(excluding_equivalents.begin(), excluding_equivalents.end(),
                     [&](swift::ValueDecl *decl) {
                       return DeclsAreEquivalent(cur_decl, decl);
                     }))
      matches.push_back(cur_decl);

  return start_num_items != matches.size();
}

void SwiftPersistentExpressionState::SwiftDeclMap::CopyDeclsTo(
    SwiftPersistentExpressionState::SwiftDeclMap &target_map) {
  for (auto &entry : m_swift_decls)
    for (auto &elem : entry.second)
      target_map.AddDecl(elem, true, {});
}

void SwiftPersistentExpressionState::RegisterSwiftPersistentDecl(
    swift::ValueDecl *value_decl) {
  m_swift_persistent_decls.AddDecl(value_decl, true, {});
}

void SwiftPersistentExpressionState::RegisterSwiftPersistentDeclAlias(
    swift::ValueDecl *value_decl, llvm::StringRef name) {
  m_swift_persistent_decls.AddDecl(value_decl, true, name);
}

void SwiftPersistentExpressionState::CopyInSwiftPersistentDecls(
    SwiftPersistentExpressionState::SwiftDeclMap &target_map) {
  target_map.CopyDeclsTo(m_swift_persistent_decls);
}

bool SwiftPersistentExpressionState::GetSwiftPersistentDecls(
    llvm::StringRef name,
    const std::vector<swift::ValueDecl *> &excluding_equivalents,
    std::vector<swift::ValueDecl *> &matches) {
  return m_swift_persistent_decls.FindMatchingDecls(name, excluding_equivalents,
                                                    matches);
}

ConstString
SwiftPersistentExpressionState::GetNextPersistentVariableName(bool is_error) {
  llvm::SmallString<64> name;
  {
    llvm::raw_svector_ostream os(name);
    uint32_t variable_num = is_error ? m_next_persistent_error_id++
                                     : m_next_persistent_variable_id++;
    os << GetPersistentVariablePrefix(is_error) << variable_num;
  }
  return ConstString(name);
}
