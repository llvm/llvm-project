//===-- DILEval.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILEval.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectRegister.h"
#include "lldb/ValueObject/ValueObjectVariable.h"
#include "llvm/Support/FormatAdapters.h"
#include <memory>

namespace lldb_private::dil {

static lldb::ValueObjectSP
LookupStaticIdentifier(lldb::TargetSP target_sp,
                       const llvm::StringRef &name_ref,
                       ConstString unqualified_name) {
  // List global variable with the same "basename". There can be many matches
  // from other scopes (namespaces, classes), so we do additional filtering
  // later.
  VariableList variable_list;
  ConstString name(name_ref);
  target_sp->GetImages().FindGlobalVariables(name, 1, variable_list);
  if (variable_list.Empty())
    return nullptr;

  ExecutionContextScope *exe_scope = target_sp->GetProcessSP().get();
  if (exe_scope == nullptr)
    exe_scope = target_sp.get();
  for (const lldb::VariableSP &var_sp : variable_list) {
    lldb::ValueObjectSP valobj_sp(
        ValueObjectVariable::Create(exe_scope, var_sp));
    if (valobj_sp && valobj_sp->GetVariable() &&
        (valobj_sp->GetVariable()->NameMatches(unqualified_name) ||
         valobj_sp->GetVariable()->NameMatches(ConstString(name_ref))))
      return valobj_sp;
  }

  return nullptr;
}

static lldb::VariableSP DILFindVariable(ConstString name,
                                        VariableList *variable_list) {
  lldb::VariableSP exact_match;
  std::vector<lldb::VariableSP> possible_matches;

  typedef std::vector<lldb::VariableSP> collection;
  typedef collection::iterator iterator;

  iterator pos, end = variable_list->end();
  for (pos = variable_list->begin(); pos != end; ++pos) {
    llvm::StringRef str_ref_name = pos->get()->GetName().GetStringRef();
    // Check for global vars, which might start with '::'.
    str_ref_name.consume_front("::");

    if (str_ref_name == name.GetStringRef())
      possible_matches.push_back(*pos);
    else if (pos->get()->NameMatches(name))
      possible_matches.push_back(*pos);
  }

  // Look for exact matches (favors local vars over global vars)
  auto exact_match_it =
      llvm::find_if(possible_matches, [&](lldb::VariableSP var_sp) {
        return var_sp->GetName() == name;
      });

  if (exact_match_it != llvm::adl_end(possible_matches))
    exact_match = *exact_match_it;

  if (!exact_match)
    // Look for a global var exact match.
    for (auto var_sp : possible_matches) {
      llvm::StringRef str_ref_name = var_sp->GetName().GetStringRef();
      if (str_ref_name.size() > 2 && str_ref_name[0] == ':' &&
          str_ref_name[1] == ':')
        str_ref_name = str_ref_name.drop_front(2);
      ConstString tmp_name(str_ref_name);
      if (tmp_name == name) {
        exact_match = var_sp;
        break;
      }
    }

  // Take any match at this point.
  if (!exact_match && possible_matches.size() > 0)
    exact_match = possible_matches[0];

  return exact_match;
}

std::unique_ptr<IdentifierInfo>
LookupIdentifier(const std::string &name,
                 std::shared_ptr<ExecutionContextScope> ctx_scope,
                 lldb::DynamicValueType use_dynamic, CompilerType *scope_ptr) {
  ConstString name_str(name);
  llvm::StringRef name_ref = name_str.GetStringRef();

  // Support $rax as a special syntax for accessing registers.
  // Will return an invalid value in case the requested register doesn't exist.
  if (name_ref.starts_with("$")) {
    lldb::ValueObjectSP value_sp;
    const char *reg_name = name_ref.drop_front(1).data();
    Target *target = ctx_scope->CalculateTarget().get();
    Process *process = ctx_scope->CalculateProcess().get();
    if (!target || !process)
      return nullptr;

    StackFrame *stack_frame = ctx_scope->CalculateStackFrame().get();
    if (!stack_frame)
      return nullptr;

    lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
    if (!reg_ctx)
      return nullptr;

    if (const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(reg_name))
      value_sp = ValueObjectRegister::Create(stack_frame, reg_ctx, reg_info);

    if (value_sp)
      return IdentifierInfo::FromValue(*value_sp);

    return nullptr;
  }

  // Internally values don't have global scope qualifier in their names and
  // LLDB doesn't support queries with it too.
  bool global_scope = false;
  if (name_ref.starts_with("::")) {
    name_ref = name_ref.drop_front(2);
    global_scope = true;
  }

  // If the identifier doesn't refer to the global scope and doesn't have any
  // other scope qualifiers, try looking among the local and instance variables.
  if (!global_scope && !name_ref.contains("::")) {
    if (!scope_ptr || !scope_ptr->IsValid()) {
      // Lookup in the current frame.
      lldb::StackFrameSP frame = ctx_scope->CalculateStackFrame();
      // Try looking for a local variable in current scope.
      lldb::ValueObjectSP value_sp;
      lldb::VariableListSP var_list_sp(frame->GetInScopeVariableList(true));
      VariableList *variable_list = var_list_sp.get();
      if (variable_list) {
        lldb::VariableSP var_sp =
            DILFindVariable(ConstString(name_ref), variable_list);
        if (var_sp)
          value_sp = frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
      }
      if (!value_sp)
        value_sp = frame->FindVariable(ConstString(name_ref));

      if (value_sp)
        // Force static value, otherwise we can end up with the "real" type.
        return IdentifierInfo::FromValue(*value_sp);

      // Try looking for an instance variable (class member).
      SymbolContext sc = frame->GetSymbolContext(lldb::eSymbolContextFunction |
                                                 lldb::eSymbolContextBlock);
      llvm::StringRef ivar_name = sc.GetInstanceVariableName();
      value_sp = frame->FindVariable(ConstString(ivar_name));
      if (value_sp)
        value_sp = value_sp->GetChildMemberWithName(name_ref);

      if (value_sp)
        // Force static value, otherwise we can end up with the "real" type.
        return IdentifierInfo::FromValue(*(value_sp->GetStaticValue()));
    }
  }

  // Try looking for a global or static variable.

  lldb::ValueObjectSP value;
  if (!global_scope) {
    // Try looking for static member of the current scope value, e.g.
    // `ScopeType::NAME`. NAME can include nested struct (`Nested::SUBNAME`),
    // but it cannot be part of the global scope (start with "::").
    const char *type_name = "";
    if (scope_ptr)
      type_name = scope_ptr->GetCanonicalType().GetTypeName().AsCString();
    std::string name_with_type_prefix =
        llvm::formatv("{0}::{1}", type_name, name_ref).str();
    value = LookupStaticIdentifier(ctx_scope->CalculateTarget(),
                                   name_with_type_prefix, name_str);
  }

  // Lookup a regular global variable.
  if (!value)
    value = LookupStaticIdentifier(ctx_scope->CalculateTarget(), name_ref,
                                   name_str);

  // Last resort, lookup as a register (e.g. `rax` or `rip`).
  if (!value) {
    Target *target = ctx_scope->CalculateTarget().get();
    Process *process = ctx_scope->CalculateProcess().get();
    if (target && process) {
      StackFrame *stack_frame = ctx_scope->CalculateStackFrame().get();
      if (stack_frame) {
        lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
        if (reg_ctx) {
          if (const RegisterInfo *reg_info =
                  reg_ctx->GetRegisterInfoByName(name_ref))
            value = ValueObjectRegister::Create(stack_frame, reg_ctx, reg_info);
        }
      }
    }
  }

  if (value)
    return IdentifierInfo::FromValue(*value);

  return nullptr;
}

DILInterpreter::DILInterpreter(
    lldb::TargetSP target, llvm::StringRef expr,
    lldb::DynamicValueType use_dynamic,
    std::shared_ptr<ExecutionContextScope> exe_ctx_scope)
    : m_target(std::move(target)), m_expr(expr), m_default_dynamic(use_dynamic),
      m_exe_ctx_scope(exe_ctx_scope) {}

llvm::Expected<lldb::ValueObjectSP>
DILInterpreter::DILEvalNode(const DILASTNode *node) {

  // Traverse an AST pointed by the `node`.
  auto value_or_error = node->Accept(this);

  // Return the computed value or error.
  return value_or_error;
}

llvm::Expected<lldb::ValueObjectSP>
DILInterpreter::Visit(const ErrorNode *node) {
  // The AST is not valid.
  return lldb::ValueObjectSP();
}

llvm::Expected<lldb::ValueObjectSP>
DILInterpreter::Visit(const IdentifierNode *node) {
  lldb::DynamicValueType use_dynamic = node->GetUseDynamic();

  std::unique_ptr<IdentifierInfo> identifier =
      LookupIdentifier(node->GetName(), m_exe_ctx_scope, use_dynamic);

  if (!identifier) {
    std::string errMsg;
    std::string name = node->GetName();
    errMsg = llvm::formatv("use of undeclared identifier '{0}'", name);
    Status error = Status(
        (uint32_t)ErrorCode::kUndeclaredIdentifier, lldb::eErrorTypeGeneric,
        FormatDiagnostics(m_expr, errMsg, node->GetLocation()));
    return error.ToError();
  }
  lldb::ValueObjectSP val;
  lldb::TargetSP target_sp;
  Status error;

  assert(identifier->GetKind() == IdentifierInfo::Kind::eValue &&
         "Unrecognized identifier kind");

  val = identifier->GetValue();

  if (val->GetCompilerType().IsReferenceType()) {
    Status error;
    val = val->Dereference(error);
    if (error.Fail())
      return error.ToError();
  }

  target_sp = val->GetTargetSP();
  assert(target_sp && target_sp->IsValid() &&
         "identifier doesn't resolve to a valid value");

  return val;
}

} // namespace lldb_private::dil
