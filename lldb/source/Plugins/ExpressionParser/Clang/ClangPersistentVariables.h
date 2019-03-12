//===-- ClangPersistentVariables.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangPersistentVariables_h_
#define liblldb_ClangPersistentVariables_h_

#include "llvm/ADT/DenseMap.h"

#include "ClangExpressionVariable.h"
#include "ClangModulesDeclVendor.h"

#include "lldb/Expression/ExpressionVariable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <set>
#include <string>
#include <unordered_map>

namespace lldb_private {

//----------------------------------------------------------------------
/// \class ClangPersistentVariables ClangPersistentVariables.h
/// "lldb/Expression/ClangPersistentVariables.h" Manages persistent values
/// that need to be preserved between expression invocations.
///
/// A list of variables that can be accessed and updated by any expression.  See
/// ClangPersistentVariable for more discussion.  Also provides an increasing,
/// 0-based counter for naming result variables.
//----------------------------------------------------------------------
class ClangPersistentVariables : public PersistentExpressionState {
public:
  //----------------------------------------------------------------------
  /// Constructor
  //----------------------------------------------------------------------
  ClangPersistentVariables();

  ~ClangPersistentVariables() override = default;

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const PersistentExpressionState *pv) {
    return pv->getKind() == PersistentExpressionState::eKindClang;
  }

  lldb::ExpressionVariableSP
  CreatePersistentVariable(const lldb::ValueObjectSP &valobj_sp) override;

  lldb::ExpressionVariableSP CreatePersistentVariable(
      ExecutionContextScope *exe_scope, ConstString name,
      const CompilerType &compiler_type, lldb::ByteOrder byte_order,
      uint32_t addr_byte_size) override;

  void RemovePersistentVariable(lldb::ExpressionVariableSP variable) override;
  llvm::StringRef
  GetPersistentVariablePrefix(bool is_error) const override {
    return "$";
  }

  // This just adds this module to the list of hand-loaded modules, it doesn't
  // actually load it.
  void AddHandLoadedModule(ConstString module_name) {
    m_hand_loaded_modules.insert(module_name);
  }

  using HandLoadedModuleCallback = std::function<bool(const ConstString)>;

  bool RunOverHandLoadedModules(HandLoadedModuleCallback callback) {
    for (ConstString name : m_hand_loaded_modules) {
      if (!callback(name))
        return false;
    }
    return true;
  }

  void RegisterPersistentDecl(ConstString name, clang::NamedDecl *decl);

  clang::NamedDecl *GetPersistentDecl(ConstString name);

  void AddHandLoadedClangModule(ClangModulesDeclVendor::ModuleID module) {
    m_hand_loaded_clang_modules.push_back(module);
  }

  const ClangModulesDeclVendor::ModuleVector &GetHandLoadedClangModules() {
    return m_hand_loaded_clang_modules;
  }

private:
  uint32_t m_next_persistent_variable_id; ///< The counter used by
                                          ///GetNextResultName().
  uint32_t m_next_persistent_error_id;    ///< The counter used by
                                       ///GetNextResultName() when is_error is
                                       ///true.

  typedef llvm::DenseMap<const char *, clang::TypeDecl *>
      ClangPersistentTypeMap;
  ClangPersistentTypeMap
      m_clang_persistent_types; ///< The persistent types declared by the user.

  typedef std::set<lldb::IRExecutionUnitSP> ExecutionUnitSet;
  ExecutionUnitSet
      m_execution_units; ///< The execution units that contain valuable symbols.

  typedef std::set<lldb_private::ConstString> HandLoadedModuleSet;
  HandLoadedModuleSet m_hand_loaded_modules; ///< These are the names of modules
                                             ///that we have loaded by
  ///< hand into the Contexts we make for parsing.

  typedef llvm::DenseMap<const char *, lldb::addr_t> SymbolMap;
  SymbolMap
      m_symbol_map; ///< The addresses of the symbols in m_execution_units.

  typedef llvm::DenseMap<const char *, clang::NamedDecl *> PersistentDeclMap;
  PersistentDeclMap
      m_persistent_decls; ///< Persistent entities declared by the user.

  ClangModulesDeclVendor::ModuleVector
      m_hand_loaded_clang_modules; ///< These are Clang modules we hand-loaded;
                                   ///these are the highest-
                                   ///< priority source for macros.
};

} // namespace lldb_private

#endif // liblldb_ClangPersistentVariables_h_
