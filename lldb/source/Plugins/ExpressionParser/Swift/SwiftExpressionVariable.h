//===-- SwiftExpressionVariable.h -------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftExpressionVariable_h_
#define liblldb_SwiftExpressionVariable_h_

// C Includes
#include <signal.h>
#include <stdint.h>
#include <string.h>

// C++ Includes
#include <map>
#include <string>
#include <vector>

// Other libraries and framework includes
#include "llvm/Support/Casting.h"

// Project includes
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-public.h"

namespace llvm {
class Value;
}

namespace lldb_private {

class SwiftExpressionVariable : public ExpressionVariable {
public:
  SwiftExpressionVariable(ExecutionContextScope *exe_scope,
                          lldb::ByteOrder byte_order, uint32_t addr_byte_size);

  SwiftExpressionVariable(const lldb::ValueObjectSP &valobj_sp);

  SwiftExpressionVariable(ExecutionContextScope *exe_scope,
                          const ConstString &name,
                          const TypeFromUser &user_type,
                          lldb::ByteOrder byte_order, uint32_t addr_byte_size);

  //----------------------------------------------------------------------
  /// Utility functions for dealing with ExpressionVariableLists in
  /// Clang-specific ways
  //----------------------------------------------------------------------

  static SwiftExpressionVariable *
  CreateVariableInList(ExpressionVariableList &list,
                       ExecutionContextScope *exe_scope,
                       const ConstString &name, const TypeFromUser &user_type,
                       lldb::ByteOrder byte_order, uint32_t addr_byte_size) {
    SwiftExpressionVariable *swift_var =
        new SwiftExpressionVariable(exe_scope, byte_order, addr_byte_size);
    lldb::ExpressionVariableSP var_sp(swift_var);
    swift_var->SetName(name);
    swift_var->SetCompilerType(user_type);
    list.AddVariable(var_sp);
    return swift_var;
  }

  bool GetIsModifiable() const { return (m_swift_flags & EVSIsModifiable); }

  void SetIsModifiable(bool is_modifiable) {
    if (is_modifiable)
      m_swift_flags |= EVSIsModifiable;
    else
      m_swift_flags &= ~EVSIsModifiable;
  }

  bool GetIsComputed() const { return (m_swift_flags & EVSIsComputed); }

  void SetIsComputed(bool is_computed) {
    if (is_computed)
      m_swift_flags |= EVSIsComputed;
    else
      m_swift_flags &= ~EVSIsComputed;
  }

  enum SwiftFlags {
    EVSNone = 0,
    EVSNeedsInit = 1 << 0, ///< This variable's contents are not yet initialized
                           ///(for result variables and new persistent
                           ///variables)
    EVSIsModifiable =
        1 << 1, ///< This variable is a "let" and should not be modified.
    EVSIsComputed =
        1
        << 2 ///< This variable is a computed property and has no backing store.
  };

  typedef uint8_t SwiftFlagType;

  SwiftFlagType m_swift_flags; // takes elements of Flags

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const ExpressionVariable *ev) {
    return ev->getKind() == ExpressionVariable::eKindSwift;
  }

  //----------------------------------------------------------------------
  /// Members
  //----------------------------------------------------------------------
  DISALLOW_COPY_AND_ASSIGN(SwiftExpressionVariable);
};

} // namespace lldb_private

#endif // liblldb_SwiftExpressionVariable_h_
