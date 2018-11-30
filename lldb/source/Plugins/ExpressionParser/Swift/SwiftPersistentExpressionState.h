//===-- SwiftPersistentExpressionState.h ------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftPersistentExpressionState_h_
#define liblldb_SwiftPersistentExpressionState_h_

#include "SwiftExpressionVariable.h"

#include "lldb/Expression/ExpressionVariable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <set>
#include <string>
#include <unordered_map>

namespace lldb_private {

//----------------------------------------------------------------------
/// @class SwiftPersistentExpressionState SwiftPersistentExpressionState.h
/// "lldb/Expression/SwiftPersistentExpressionState.h"
/// @brief Manages persistent values that need to be preserved between
/// expression invocations.
///
/// A list of variables that can be accessed and updated by any expression.  See
/// ClangPersistentVariable for more discussion.  Also provides an increasing,
/// 0-based counter for naming result variables.
//----------------------------------------------------------------------
class SwiftPersistentExpressionState : public PersistentExpressionState {
  typedef std::set<lldb_private::ConstString> HandLoadedModuleSet;

public:
  class SwiftDeclMap {
  public:
    void AddDecl(swift::ValueDecl *decl, bool check_existing,
                 const ConstString &name);

    /// Find decls matching `name`, excluding decls that are equivalent to
    /// decls in `excluding_equivalents`, and put the results in `matches`.
    /// Return true if there are any results.
    bool FindMatchingDecls(
        const ConstString &name,
        const std::vector<swift::ValueDecl *> &excluding_equivalents,
        std::vector<swift::ValueDecl *> &matches);

    void CopyDeclsTo(SwiftDeclMap &target_map);
    static bool DeclsAreEquivalent(swift::Decl *lhs, swift::Decl *rhs);

  private:
    typedef std::unordered_multimap<std::string, swift::ValueDecl *>
        SwiftDeclMapTy;
    typedef SwiftDeclMapTy::iterator iterator;
    SwiftDeclMapTy m_swift_decls;
  };

  //----------------------------------------------------------------------
  /// Constructor
  //----------------------------------------------------------------------
  SwiftPersistentExpressionState();

  ~SwiftPersistentExpressionState() {}

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const PersistentExpressionState *pv) {
    return pv->getKind() == PersistentExpressionState::eKindClang;
  }

  lldb::ExpressionVariableSP
  CreatePersistentVariable(const lldb::ValueObjectSP &valobj_sp) override;

  lldb::ExpressionVariableSP CreatePersistentVariable(
      ExecutionContextScope *exe_scope, const ConstString &name,
      const CompilerType &compiler_type, lldb::ByteOrder byte_order,
      uint32_t addr_byte_size) override;

  llvm::StringRef
  GetPersistentVariablePrefix(bool is_error) const override {
    return is_error ? "$E" : "$R";
  }

  void RemovePersistentVariable(lldb::ExpressionVariableSP variable) override;

  void RegisterSwiftPersistentDecl(swift::ValueDecl *value_decl);

  void RegisterSwiftPersistentDeclAlias(swift::ValueDecl *value_decl,
                                        const ConstString &name);

  void CopyInSwiftPersistentDecls(SwiftDeclMap &source_map);

  /// Find decls matching `name`, excluding decls that are equivalent to decls
  /// in `excluding_equivalents`, and put the results in `matches`.  Return true
  /// if there are any results.
  bool GetSwiftPersistentDecls(
      const ConstString &name,
      const std::vector<swift::ValueDecl *> &excluding_equivalents,
      std::vector<swift::ValueDecl *> &matches);

  // This just adds this module to the list of hand-loaded modules, it doesn't
  // actually load it.
  void AddHandLoadedModule(const ConstString &module_name) {
    m_hand_loaded_modules.insert(module_name);
  }

  /// This returns the list of hand-loaded modules.
  HandLoadedModuleSet GetHandLoadedModules() { return m_hand_loaded_modules; }

private:
  uint32_t m_next_persistent_variable_id; ///< The counter used by
                                          ///GetNextResultName().
  uint32_t m_next_persistent_error_id;    ///< The counter used by
                                       ///GetNextResultName() when is_error is
                                       ///true.

  SwiftDeclMap m_swift_persistent_decls; ///< The persistent functions declared
                                         ///by the user.

  HandLoadedModuleSet m_hand_loaded_modules; ///< These are the names of modules
                                             ///that we have loaded by
  ///< hand into the Contexts we make for parsing.
};
}

#endif
