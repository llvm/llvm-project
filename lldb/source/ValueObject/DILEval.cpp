//===-- DILEval.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILEval.h"

#include <memory>

#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/ValueObject.h"
#include "llvm/Support/FormatAdapters.h"

namespace lldb_private {

namespace dil {

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<std::string> sm)
    : m_target(std::move(target)), m_sm(std::move(sm)) {
  m_default_dynamic = lldb::eNoDynamicValues;
}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<std::string> sm,
                               lldb::DynamicValueType use_dynamic)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_default_dynamic(use_dynamic) {}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<std::string> sm,
                               lldb::ValueObjectSP scope)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_scope(std::move(scope)) {
  m_default_dynamic = lldb::eNoDynamicValues;
  // If `m_scope` is a reference, dereference it. All operations on a reference
  // should be operations on the referent.
  if (m_scope->GetCompilerType().IsValid() &&
      m_scope->GetCompilerType().IsReferenceType()) {
    Status error;
    m_scope = m_scope->Dereference(error);
  }
}

lldb::ValueObjectSP DILInterpreter::DILEval(const DILASTNode *tree,
                                            lldb::TargetSP target_sp,
                                            Status &error) {
  m_error.Clear();
  // Evaluate an AST.
  DILEvalNode(tree);
  // Set the error.
  error = std::move(m_error);
  // Return the computed result. If there was an error, it will be invalid.
  return m_result;
}

lldb::ValueObjectSP DILInterpreter::DILEvalNode(const DILASTNode *node) {

  // Traverse an AST pointed by the `node`.
  node->Accept(this);

  // Return the computed value for convenience. The caller is responsible for
  // checking if an error occured during the evaluation.
  return m_result;
}

void DILInterpreter::SetError(ErrorCode code, std::string error, uint32_t loc) {
  assert(m_error.Success() && "interpreter can error only once");
  m_error = Status((uint32_t)code, lldb::eErrorTypeGeneric,
                   FormatDiagnostics(m_sm, error, loc));
}

void DILInterpreter::Visit(const ErrorNode *node) {
  // The AST is not valid.
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const IdentifierNode *node) {
  std::shared_ptr<ExecutionContextScope> exe_ctx_scope =
      node->get_exe_context();
  lldb::DynamicValueType use_dynamic = node->use_dynamic();

  std::unique_ptr<IdentifierInfo> identifier =
      LookupIdentifier(node->name(), exe_ctx_scope, use_dynamic);

  if (!identifier) {
    std::string errMsg;
    std::string name = node->name();
    if (name == "this")
      errMsg = "invalid use of 'this' outside of a non-static member function";
    else
      errMsg = llvm::formatv("use of undeclared identifier '{0}'", name);
    SetError(ErrorCode::kUndeclaredIdentifier, errMsg, node->GetLocation());
    m_result = lldb::ValueObjectSP();
    return;
  }
  lldb::ValueObjectSP val;
  lldb::TargetSP target_sp;
  Status error;

  assert(identifier->GetKind() == IdentifierInfo::Kind::eValue &&
         "Unrecognized identifier kind");

  val = identifier->GetValue();
  target_sp = val->GetTargetSP();
  assert(target_sp && target_sp->IsValid() &&
         "identifier doesn't resolve to a valid value");

  m_result = val;
}

} // namespace dil

} // namespace lldb_private
