//===-- SwiftREPLMaterializer.h ---------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftREPLMaterializer_h
#define liblldb_SwiftREPLMaterializer_h

#include "lldb/Expression/Materializer.h"

namespace lldb_private {

class SwiftREPLMaterializer : public Materializer {
public:
  SwiftREPLMaterializer() : Materializer(eKindSwiftREPL) {}

  uint32_t AddREPLResultVariable(const CompilerType &type,
                                 swift::ValueDecl *decl,
                                 PersistentVariableDelegate *delegate,
                                 Status &err);

  uint32_t
  AddPersistentVariable(lldb::ExpressionVariableSP &persistent_variable_sp,
                        PersistentVariableDelegate *delegate,
                        Status &err) override;

  void RegisterExecutionUnit(IRExecutionUnit *execution_unit) {
    m_execution_unit = execution_unit;
  }

  IRExecutionUnit *GetExecutionUnit() { return m_execution_unit; }

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const Materializer *m) {
    return m->getKind() == Materializer::eKindSwiftREPL;
  }

private:
  IRExecutionUnit *m_execution_unit;
};
}

#endif /* SwiftREPLMaterializer_h */
