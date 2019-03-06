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

#include "lldb/Symbol/SwiftASTContext.h" // Needed for llvm::isa<SwiftASTContext>(...)
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

SwiftPersistentExpressionState::SwiftPersistentExpressionState()
    : lldb_private::PersistentExpressionState(LLVMCastKind::eKindClang),
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
    swift::ValueDecl *value_decl, bool check_existing, ConstString alias) {
  std::string name_str;

  if (alias.IsEmpty()) {
    name_str = (value_decl->getBaseName().getIdentifier().str());
  } else {
    name_str.assign(alias.GetCString());
  }

  if (!check_existing) {
    m_swift_decls.insert(std::make_pair(name_str, value_decl));
    return;
  }

  SwiftDeclMapTy::iterator map_end = m_swift_decls.end();
  std::pair<SwiftDeclMapTy::iterator, SwiftDeclMapTy::iterator> found_range =
      m_swift_decls.equal_range(name_str);

  if (found_range.first == map_end) {
    m_swift_decls.insert(std::make_pair(name_str, value_decl));
    return;
  } else {
    SwiftDeclMapTy::iterator cur_item;
    bool done = false;
    for (cur_item = found_range.first; !done && cur_item != found_range.second;
         cur_item++) {
      swift::ValueDecl *cur_decl = (*cur_item).second;
      if (DeclsAreEquivalent(cur_decl, value_decl)) {
        m_swift_decls.erase(cur_item);
        break;
      }
    }

    m_swift_decls.insert(std::make_pair(name_str, value_decl));
  }
}

bool SwiftPersistentExpressionState::SwiftDeclMap::FindMatchingDecls(
    ConstString name,
    const std::vector<swift::ValueDecl *> &excluding_equivalents,
    std::vector<swift::ValueDecl *> &matches) {
  std::string name_str(name.AsCString());
  size_t start_num_items = matches.size();
  size_t start_num_excluding_equivalents = excluding_equivalents.size();

  std::pair<SwiftDeclMapTy::iterator, SwiftDeclMapTy::iterator> found_range =
      m_swift_decls.equal_range(name_str);
  for (SwiftDeclMapTy::iterator cur_item = found_range.first;
       cur_item != found_range.second; cur_item++) {
    bool add_it = true;
    swift::ValueDecl *cur_decl = (*cur_item).second;

    // Iterate over only the elements of `excluding_equivalents` that were
    // originally there, in case `matches` aliases it.
    for (size_t idx = 0; idx < start_num_excluding_equivalents; idx++) {
      if (DeclsAreEquivalent(excluding_equivalents[idx], cur_decl)) {
        add_it = false;
        break;
      }
    }
    if (add_it)
      matches.push_back((*cur_item).second);
  }
  return start_num_items != matches.size();
}

void SwiftPersistentExpressionState::SwiftDeclMap::CopyDeclsTo(
    SwiftPersistentExpressionState::SwiftDeclMap &target_map) {
  for (auto elem : m_swift_decls)
    target_map.AddDecl(elem.second, true, ConstString());
}

void SwiftPersistentExpressionState::RegisterSwiftPersistentDecl(
    swift::ValueDecl *value_decl) {
  m_swift_persistent_decls.AddDecl(value_decl, true, ConstString());
}

void SwiftPersistentExpressionState::RegisterSwiftPersistentDeclAlias(
    swift::ValueDecl *value_decl, ConstString name) {
  m_swift_persistent_decls.AddDecl(value_decl, true, name);
}

void SwiftPersistentExpressionState::CopyInSwiftPersistentDecls(
    SwiftPersistentExpressionState::SwiftDeclMap &target_map) {
  target_map.CopyDeclsTo(m_swift_persistent_decls);
}

bool SwiftPersistentExpressionState::GetSwiftPersistentDecls(
    ConstString name,
    const std::vector<swift::ValueDecl *> &excluding_equivalents,
    std::vector<swift::ValueDecl *> &matches) {
  return m_swift_persistent_decls.FindMatchingDecls(name, excluding_equivalents,
                                                    matches);
}
