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

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace lldb_private {

/// Manages persistent values that need to be preserved between
/// expression invocations.
///
/// A list of variables that can be accessed and updated by any
/// expression.  See \ref ClangPersistentVariable for more discussion.
/// Also provides an increasing, 0-based counter for naming result
/// variables.
class SwiftPersistentExpressionState : public PersistentExpressionState {
public:
  class SwiftDeclMap {
  public:
    void AddDecl(CompilerDecl value_decl, bool check_existing,
                 llvm::StringRef name);

    /// Find decls matching `name`, excluding decls that are equivalent to
    /// decls in `excluding_equivalents`, and put the results in `matches`.
    /// Return true if there are any results.
    bool FindMatchingDecls(
        llvm::StringRef name,
        const std::vector<CompilerDecl> &excluding_equivalents,
        std::vector<CompilerDecl> &matches);

    void CopyDeclsTo(SwiftDeclMap &target_map);
    static bool DeclsAreEquivalent(CompilerDecl lhs, CompilerDecl rhs);

  private:
    /// Each decl also stores the context it comes from.
    llvm::StringMap<llvm::SmallVector<CompilerDecl, 1>> m_swift_decls;
  };

  //----------------------------------------------------------------------
  /// Constructor
  //----------------------------------------------------------------------
  SwiftPersistentExpressionState();

  ~SwiftPersistentExpressionState() {}

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  // LLVM RTTI Support
  static char ID;

  lldb::ExpressionVariableSP
  CreatePersistentVariable(const lldb::ValueObjectSP &valobj_sp) override;

  lldb::ExpressionVariableSP
  CreatePersistentVariable(ExecutionContextScope *exe_scope, ConstString name,
                           const CompilerType &compiler_type,
                           lldb::ByteOrder byte_order,
                           uint32_t addr_byte_size) override;

  llvm::StringRef GetPersistentVariablePrefix(bool is_error) const override {
    return is_error ? "$E" : "$R";
  }

  void RemovePersistentVariable(lldb::ExpressionVariableSP variable) override;

  ConstString GetNextPersistentVariableName(bool is_error = false) override;

  std::optional<CompilerType>
  GetCompilerTypeFromPersistentDecl(ConstString type_name) override;

  void RegisterSwiftPersistentDecl(CompilerDecl value_decl);

  void RegisterSwiftPersistentDeclAlias(CompilerDecl value_decl,
                                        llvm::StringRef name);

  void CopyInSwiftPersistentDecls(SwiftDeclMap &source_map);

  /// Find decls matching `name`, excluding decls that are equivalent to decls
  /// in `excluding_equivalents`, and put the results in `matches`.  Return true
  /// if there are any results.
  bool GetSwiftPersistentDecls(
      llvm::StringRef name,
      const std::vector<CompilerDecl> &excluding_equivalents,
      std::vector<CompilerDecl> &matches);

private:
  /// The counter used by GetNextResultName().
  uint32_t m_next_persistent_variable_id;
  /// The counter used by GetNextResultName() when is_error is true.
  uint32_t m_next_persistent_error_id;
  /// The persistent functions declared by the user.
  SwiftDeclMap m_swift_persistent_decls;
};
} // namespace lldb_private

#endif
