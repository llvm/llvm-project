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

static lldb::ValueObjectSP LookupStaticIdentifier(
    VariableList &variable_list, std::shared_ptr<StackFrame> exe_scope,
    llvm::StringRef name_ref, llvm::StringRef unqualified_name) {
  // First look for an exact match to the (possibly) qualified name.
  for (const lldb::VariableSP &var_sp : variable_list) {
    lldb::ValueObjectSP valobj_sp(
        ValueObjectVariable::Create(exe_scope.get(), var_sp));
    if (valobj_sp && valobj_sp->GetVariable() &&
        (valobj_sp->GetVariable()->NameMatches(ConstString(name_ref))))
      return valobj_sp;
  }

  // If the qualified name is the same as the unqualfied name, there's nothing
  // more to be done.
  if (name_ref == unqualified_name)
    return nullptr;

  // We didn't match the qualified name; try to match the unqualified name.
  for (const lldb::VariableSP &var_sp : variable_list) {
    lldb::ValueObjectSP valobj_sp(
        ValueObjectVariable::Create(exe_scope.get(), var_sp));
    if (valobj_sp && valobj_sp->GetVariable() &&
        (valobj_sp->GetVariable()->NameMatches(ConstString(unqualified_name))))
      return valobj_sp;
  }

  return nullptr;
}

static lldb::VariableSP DILFindVariable(ConstString name,
                                        lldb::VariableListSP variable_list) {
  lldb::VariableSP exact_match;
  std::vector<lldb::VariableSP> possible_matches;

  for (lldb::VariableSP var_sp : *variable_list) {
    llvm::StringRef str_ref_name = var_sp->GetName().GetStringRef();
    // Check for global vars, which might start with '::'.
    str_ref_name.consume_front("::");

    if (str_ref_name == name.GetStringRef())
      possible_matches.push_back(var_sp);
    else if (var_sp->NameMatches(name))
      possible_matches.push_back(var_sp);
  }

  // Look for exact matches (favors local vars over global vars)
  auto exact_match_it =
      llvm::find_if(possible_matches, [&](lldb::VariableSP var_sp) {
        return var_sp->GetName() == name;
      });

  if (exact_match_it != possible_matches.end())
    return *exact_match_it;

  // Look for a global var exact match.
  for (auto var_sp : possible_matches) {
    llvm::StringRef str_ref_name = var_sp->GetName().GetStringRef();
    str_ref_name.consume_front("::");
    if (str_ref_name == name.GetStringRef())
      return var_sp;
  }

  // If there's a single non-exact match, take it.
  if (possible_matches.size() == 1)
    return possible_matches[0];

  return nullptr;
}

lldb::ValueObjectSP LookupGlobalIdentifier(
    llvm::StringRef name_ref, std::shared_ptr<StackFrame> stack_frame,
    lldb::TargetSP target_sp, lldb::DynamicValueType use_dynamic,
    CompilerType *scope_ptr) {
  // First look for match in "local" global variables.
  lldb::VariableListSP variable_list(stack_frame->GetInScopeVariableList(true));
  name_ref.consume_front("::");

  lldb::ValueObjectSP value_sp;
  if (variable_list) {
    lldb::VariableSP var_sp =
        DILFindVariable(ConstString(name_ref), variable_list);
    if (var_sp)
      value_sp =
          stack_frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
  }

  if (value_sp)
    return value_sp;

  // Also check for static global vars.
  if (variable_list) {
    const char *type_name = "";
    if (scope_ptr)
      type_name = scope_ptr->GetCanonicalType().GetTypeName().AsCString();
    std::string name_with_type_prefix =
        llvm::formatv("{0}::{1}", type_name, name_ref).str();
    value_sp = LookupStaticIdentifier(*variable_list, stack_frame,
                                      name_with_type_prefix, name_ref);
    if (!value_sp)
      value_sp = LookupStaticIdentifier(*variable_list, stack_frame, name_ref,
                                        name_ref);
  }

  if (value_sp)
    return value_sp;

  // Check for match in modules global variables.
  VariableList modules_var_list;
  target_sp->GetImages().FindGlobalVariables(
      ConstString(name_ref), std::numeric_limits<uint32_t>::max(),
      modules_var_list);
  if (modules_var_list.Empty())
    return nullptr;

  for (const lldb::VariableSP &var_sp : modules_var_list) {
    std::string qualified_name = llvm::formatv("::{0}", name_ref).str();
    if (var_sp->NameMatches(ConstString(name_ref)) ||
        var_sp->NameMatches(ConstString(qualified_name))) {
      value_sp = ValueObjectVariable::Create(stack_frame.get(), var_sp);
      break;
    }
  }

  if (value_sp)
    return value_sp;

  return nullptr;
}

lldb::ValueObjectSP LookupIdentifier(llvm::StringRef name_ref,
                                     std::shared_ptr<StackFrame> stack_frame,
                                     lldb::DynamicValueType use_dynamic,
                                     CompilerType *scope_ptr) {
  // Support $rax as a special syntax for accessing registers.
  // Will return an invalid value in case the requested register doesn't exist.
  if (name_ref.consume_front("$")) {
    lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
    if (!reg_ctx)
      return nullptr;

    if (const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(name_ref))
      return ValueObjectRegister::Create(stack_frame.get(), reg_ctx, reg_info);

    return nullptr;
  }

  lldb::VariableListSP variable_list(
      stack_frame->GetInScopeVariableList(false));

  if (!name_ref.contains("::")) {
    if (!scope_ptr || !scope_ptr->IsValid()) {
      // Lookup in the current frame.
      // Try looking for a local variable in current scope.
      lldb::ValueObjectSP value_sp;
      if (variable_list) {
        lldb::VariableSP var_sp =
            DILFindVariable(ConstString(name_ref), variable_list);
        if (var_sp)
          value_sp =
              stack_frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
      }
      if (!value_sp)
        value_sp = stack_frame->FindVariable(ConstString(name_ref));

      if (value_sp)
        return value_sp;

      // Try looking for an instance variable (class member).
      SymbolContext sc = stack_frame->GetSymbolContext(
          lldb::eSymbolContextFunction | lldb::eSymbolContextBlock);
      llvm::StringRef ivar_name = sc.GetInstanceVariableName();
      value_sp = stack_frame->FindVariable(ConstString(ivar_name));
      if (value_sp)
        value_sp = value_sp->GetChildMemberWithName(name_ref);

      if (value_sp)
        return value_sp;
    }
  }
  return nullptr;
}

Interpreter::Interpreter(lldb::TargetSP target, llvm::StringRef expr,
                         lldb::DynamicValueType use_dynamic,
                         std::shared_ptr<StackFrame> frame_sp)
    : m_target(std::move(target)), m_expr(expr), m_default_dynamic(use_dynamic),
      m_exe_ctx_scope(frame_sp) {}

llvm::Expected<lldb::ValueObjectSP> Interpreter::Evaluate(const ASTNode *node) {
  // Evaluate an AST.
  auto value_or_error = node->Accept(this);
  // Return the computed value-or-error. The caller is responsible for
  // checking if an error occured during the evaluation.
  return value_or_error;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const IdentifierNode *node) {
  lldb::DynamicValueType use_dynamic = m_default_dynamic;

  lldb::ValueObjectSP identifier =
      LookupIdentifier(node->GetName(), m_exe_ctx_scope, use_dynamic);

  if (!identifier)
    identifier = LookupGlobalIdentifier(node->GetName(), m_exe_ctx_scope,
                                        m_target, use_dynamic);
  if (!identifier) {
    std::string errMsg =
        llvm::formatv("use of undeclared identifier '{0}'", node->GetName());
    return llvm::make_error<DILDiagnosticError>(
        m_expr, errMsg, node->GetLocation(), node->GetName().size());
  }

  return identifier;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const UnaryOpNode *node) {
  Status error;
  auto rhs_or_err = Evaluate(node->operand());
  if (!rhs_or_err)
    return rhs_or_err;

  lldb::ValueObjectSP rhs = *rhs_or_err;

  switch (node->kind()) {
  case UnaryOpKind::Deref: {
    lldb::ValueObjectSP dynamic_rhs = rhs->GetDynamicValue(m_default_dynamic);
    if (dynamic_rhs)
      rhs = dynamic_rhs;

    lldb::ValueObjectSP child_sp = rhs->Dereference(error);
    if (error.Fail())
      return llvm::make_error<DILDiagnosticError>(m_expr, error.AsCString(),
                                                  node->GetLocation());

    return child_sp;
  }
  case UnaryOpKind::AddrOf: {
    Status error;
    lldb::ValueObjectSP value = rhs->AddressOf(error);
    if (error.Fail())
      return llvm::make_error<DILDiagnosticError>(m_expr, error.AsCString(),
                                                  node->GetLocation());

    return value;
  }
  }

  // Unsupported/invalid operation.
  return llvm::make_error<DILDiagnosticError>(
      m_expr, "invalid ast: unexpected binary operator", node->GetLocation());
}

lldb::ValueObjectSP
Interpreter::EvaluateMemberOf(lldb::ValueObjectSP value,
                              const std::vector<uint32_t> &path,
                              bool use_synthetic, bool is_dynamic) {
  lldb::ValueObjectSP member_val_sp = value;

  lldb::DynamicValueType use_dynamic =
      (!is_dynamic) ? lldb::eNoDynamicValues : lldb::eDynamicDontRunTarget;
  // Walk the path from the base value to the value that contains the requested field.
  for (uint32_t idx : path) {
    member_val_sp = member_val_sp->GetChildAtIndex(idx, /*can_create*/ true);
  }
  // If that didn't work, try it with the dynamic value.
  if (!member_val_sp && is_dynamic) {
    lldb::ValueObjectSP dyn_val_sp = value->GetDynamicValue(use_dynamic);
    if (dyn_val_sp) {
      for (uint32_t idx : path) {
        dyn_val_sp = dyn_val_sp->GetChildAtIndex(idx, true);
      }
      member_val_sp = dyn_val_sp;
    }
  }
  assert(member_val_sp && "invalid ast: invalid member access");

  return member_val_sp;
}

static bool GetFieldIndex(CompilerType type, const std::string &name,
                          std::vector<uint32_t> *idx_path) {
  bool found = false;
  uint32_t num_fields = type.GetNumFields();
  for (uint32_t i = 0; i < num_fields; ++i) {
    uint64_t bit_offset = 0;
    uint32_t bitfield_bit_size = 0;
    bool is_bitfield = false;
    std::string name_sstr;
    CompilerType field_type(type.GetFieldAtIndex(
        i, name_sstr, &bit_offset, &bitfield_bit_size, &is_bitfield));
    auto field_name =
        name_sstr.length() == 0 ? std::optional<std::string>() : name_sstr;
    if (field_type.IsValid() && name_sstr == name) {
      idx_path->push_back(i + type.GetNumberOfNonEmptyBaseClasses());
      found = true;
      break;
    } else if (field_type.IsAnonymousType()) {
      found = GetFieldIndex(field_type, name, idx_path);
      if (found) {
        idx_path->push_back(i + type.GetNumberOfNonEmptyBaseClasses());
        break;
      }
    }
  }
  return found;
}

static bool SearchBaseClassesForField(lldb::ValueObjectSP base_sp,
                                      CompilerType base_type,
                                      const std::string &name,
                                      std::vector<uint32_t> *idx_path,
                                      bool use_synthetic, bool is_dynamic) {
  bool found = false;
  uint32_t num_non_empty_bases = 0;
  uint32_t num_direct_bases = base_type.GetNumDirectBaseClasses();
  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    uint32_t bit_offset;
    CompilerType base_class =
        base_type.GetDirectBaseClassAtIndex(i, &bit_offset);
    std::vector<uint32_t> field_idx_path;
    if (GetFieldIndex(base_class, name, &field_idx_path)) {
      for (uint32_t j : field_idx_path)
        idx_path->push_back(j + base_class.GetNumberOfNonEmptyBaseClasses());
      idx_path->push_back(i);
      return true;
    }

    found = SearchBaseClassesForField(base_sp, base_class, name, idx_path,
                                      use_synthetic, is_dynamic);
    if (found) {
      idx_path->push_back(i);
      return true;
    }

    if (base_class.GetNumFields() > 0)
      num_non_empty_bases += 1;
  }
  return false;
}

lldb::ValueObjectSP Interpreter::FindMemberWithName(lldb::ValueObjectSP base,
                                                    ConstString name,
                                                    bool is_arrow) {
  bool is_synthetic = false;
  bool is_dynamic = true;
  // See if GetChildMemberWithName works.
  lldb::ValueObjectSP field_obj =
      base->GetChildMemberWithName(name.GetStringRef());
  if (field_obj && field_obj->GetName() == name)
    return field_obj;

  // Check for synthetic member.
  lldb::ValueObjectSP child_sp = base->GetSyntheticValue();
  if (child_sp) {
    is_synthetic = true;
    field_obj = child_sp->GetChildMemberWithName(name);
    if (field_obj && field_obj->GetName() == name)
      return field_obj;
  }

  // Check indices of immediate member fields of base's type.
  CompilerType base_type = base->GetCompilerType();
  std::vector<uint32_t> field_idx_path;
  if (GetFieldIndex(base_type, name.GetString(), &field_idx_path)) {
    std::reverse(field_idx_path.begin(), field_idx_path.end());
    // Traverse the path & verify the final object is correct.
    field_obj = base;
    for (uint32_t i : field_idx_path)
      field_obj = field_obj->GetChildAtIndex(i, true);
    if (field_obj && field_obj->GetName() == name)
      return field_obj;
  }

  // Go through base classes and look for field there.
  std::vector<uint32_t> base_class_idx_path;
  bool found =
      SearchBaseClassesForField(base, base_type, name.GetString(),
                                &base_class_idx_path, is_synthetic, is_dynamic);
  if (found && !base_class_idx_path.empty()) {
    std::reverse(base_class_idx_path.begin(), base_class_idx_path.end());
    field_obj =
        EvaluateMemberOf(base, base_class_idx_path, is_synthetic, is_dynamic);
    if (field_obj && field_obj->GetName() == name)
      return field_obj;
  }

  // Field not found.
  return lldb::ValueObjectSP();
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const MemberOfNode *node) {
  Status error;
  auto base_or_err = Evaluate(node->base());
  if (!base_or_err) {
    return base_or_err;
  }
  lldb::ValueObjectSP base = *base_or_err;

  // Perform basic type checking.
  CompilerType base_type = base->GetCompilerType();
  // When using an arrow, make sure the base is a pointer or array type.
  // When using a period, make sure the base type is NOT a pointer type.
  if (node->IsArrow() && !base_type.IsPointerType() &&
      !base_type.IsArrayType()) {
    lldb::ValueObjectSP deref_sp = base->Dereference(error);
    if (error.Success()) {
      base = deref_sp;
      base_type = deref_sp->GetCompilerType().GetPointerType();
    } else {
      std::string errMsg =
          llvm::formatv("member reference type {0} is not a pointer; "
                        "did you mean to use '.'?",
                        base_type.TypeDescription());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, errMsg, node->GetLocation(), node->FieldName().GetLength());
    }
  } else if (!node->IsArrow() && base_type.IsPointerType()) {
    std::string errMsg =
        llvm::formatv("member reference type {0} is a pointer; "
                      "did you mean to use '->'?",
                      base_type.TypeDescription());
    return llvm::make_error<DILDiagnosticError>(
        m_expr, errMsg, node->GetLocation(), node->FieldName().GetLength());
  }

  // User specified array->elem; need to get to element[0] to look for fields.
  if (node->IsArrow() && base_type.IsArrayType())
    base = base->GetChildAtIndex(0);

  // Now look for the member with the specified name.
  lldb::ValueObjectSP field_obj =
      FindMemberWithName(base, node->FieldName(), node->IsArrow());
  if (field_obj) {
    if (field_obj->GetCompilerType().IsReferenceType()) {
      lldb::ValueObjectSP tmp_obj = field_obj->Dereference(error);
      if (error.Fail())
        return error.ToError();
      return tmp_obj;
    }
    return field_obj;
  }

  if (node->IsArrow() && base_type.IsPointerType())
    base_type = base_type.GetPointeeType();
  std::string errMsg = llvm::formatv(
      "no member named '{0}' in {1}", node->FieldName().GetStringRef(),
      base_type.GetFullyUnqualifiedType().TypeDescription());
  return llvm::make_error<DILDiagnosticError>(
      m_expr, errMsg, node->GetLocation(), node->FieldName().GetLength());
}

} // namespace lldb_private::dil
