//===-- DILEval.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILEval.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectRegister.h"
#include "lldb/ValueObject/ValueObjectVariable.h"
#include "llvm/Support/FormatAdapters.h"
#include <memory>

namespace lldb_private::dil {

static lldb::VariableSP DILFindVariable(ConstString name,
                                        VariableList &variable_list) {
  lldb::VariableSP exact_match;
  std::vector<lldb::VariableSP> possible_matches;

  for (lldb::VariableSP var_sp : variable_list) {
    llvm::StringRef str_ref_name = var_sp->GetName().GetStringRef();

    str_ref_name.consume_front("::");
    // Check for the exact same match
    if (str_ref_name == name.GetStringRef())
      return var_sp;

    // Check for possible matches by base name
    if (var_sp->NameMatches(name))
      possible_matches.push_back(var_sp);
  }

  // If there's a non-exact match, take it.
  if (possible_matches.size() > 0)
    return possible_matches[0];

  return nullptr;
}

lldb::ValueObjectSP LookupGlobalIdentifier(
    llvm::StringRef name_ref, std::shared_ptr<StackFrame> stack_frame,
    lldb::TargetSP target_sp, lldb::DynamicValueType use_dynamic) {
  // Get a global variables list without the locals from the current frame
  SymbolContext symbol_context =
      stack_frame->GetSymbolContext(lldb::eSymbolContextCompUnit);
  lldb::VariableListSP variable_list;
  if (symbol_context.comp_unit)
    variable_list = symbol_context.comp_unit->GetVariableList(true);

  name_ref.consume_front("::");
  lldb::ValueObjectSP value_sp;
  if (variable_list) {
    lldb::VariableSP var_sp =
        DILFindVariable(ConstString(name_ref), *variable_list);
    if (var_sp)
      value_sp =
          stack_frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
  }

  if (value_sp)
    return value_sp;

  // Check for match in modules global variables.
  VariableList modules_var_list;
  target_sp->GetImages().FindGlobalVariables(
      ConstString(name_ref), std::numeric_limits<uint32_t>::max(),
      modules_var_list);

  if (!modules_var_list.Empty()) {
    lldb::VariableSP var_sp =
        DILFindVariable(ConstString(name_ref), modules_var_list);
    if (var_sp)
      value_sp = ValueObjectVariable::Create(stack_frame.get(), var_sp);

    if (value_sp)
      return value_sp;
  }
  return nullptr;
}

lldb::ValueObjectSP LookupIdentifier(llvm::StringRef name_ref,
                                     std::shared_ptr<StackFrame> stack_frame,
                                     lldb::DynamicValueType use_dynamic) {
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

  if (!name_ref.contains("::")) {
    // Lookup in the current frame.
    // Try looking for a local variable in current scope.
    lldb::VariableListSP variable_list(
        stack_frame->GetInScopeVariableList(false));

    lldb::ValueObjectSP value_sp;
    if (variable_list) {
      lldb::VariableSP var_sp =
          variable_list->FindVariable(ConstString(name_ref));
      if (var_sp)
        value_sp =
            stack_frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
    }

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
  return nullptr;
}

Interpreter::Interpreter(lldb::TargetSP target, llvm::StringRef expr,
                         std::shared_ptr<StackFrame> frame_sp,
                         lldb::DynamicValueType use_dynamic, bool use_synthetic,
                         bool fragile_ivar, bool check_ptr_vs_member)
    : m_target(std::move(target)), m_expr(expr), m_exe_ctx_scope(frame_sp),
      m_use_dynamic(use_dynamic), m_use_synthetic(use_synthetic),
      m_fragile_ivar(fragile_ivar), m_check_ptr_vs_member(check_ptr_vs_member) {
}

llvm::Expected<lldb::ValueObjectSP> Interpreter::Evaluate(const ASTNode *node) {
  // Evaluate an AST.
  auto value_or_error = node->Accept(this);
  // Return the computed value-or-error. The caller is responsible for
  // checking if an error occured during the evaluation.
  return value_or_error;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const IdentifierNode *node) {
  lldb::DynamicValueType use_dynamic = m_use_dynamic;

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
  auto rhs_or_err = Evaluate(node->GetOperand());
  if (!rhs_or_err)
    return rhs_or_err;

  lldb::ValueObjectSP rhs = *rhs_or_err;

  switch (node->GetKind()) {
  case UnaryOpKind::Deref: {
    lldb::ValueObjectSP dynamic_rhs = rhs->GetDynamicValue(m_use_dynamic);
    if (dynamic_rhs)
      rhs = dynamic_rhs;

    lldb::ValueObjectSP child_sp = rhs->Dereference(error);
    if (!child_sp && m_use_synthetic) {
      if (lldb::ValueObjectSP synth_obj_sp = rhs->GetSyntheticValue()) {
        error.Clear();
        child_sp = synth_obj_sp->Dereference(error);
      }
    }
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

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const MemberOfNode *node) {
  auto base_or_err = Evaluate(node->GetBase());
  if (!base_or_err)
    return base_or_err;
  bool expr_is_ptr = node->GetIsArrow();
  lldb::ValueObjectSP base = *base_or_err;

  // Perform some basic type & correctness checking.
  if (node->GetIsArrow()) {
    if (!m_fragile_ivar) {
      // Make sure we aren't trying to deref an objective
      // C ivar if this is not allowed
      const uint32_t pointer_type_flags =
          base->GetCompilerType().GetTypeInfo(nullptr);
      if ((pointer_type_flags & lldb::eTypeIsObjC) &&
          (pointer_type_flags & lldb::eTypeIsPointer)) {
        // This was an objective C object pointer and it was requested we
        // skip any fragile ivars so return nothing here
        return lldb::ValueObjectSP();
      }
    }

    // If we have a non-pointer type with a synthetic value then lets check
    // if we have a synthetic dereference specified.
    if (!base->IsPointerType() && base->HasSyntheticValue()) {
      Status deref_error;
      if (lldb::ValueObjectSP synth_deref_sp =
              base->GetSyntheticValue()->Dereference(deref_error);
          synth_deref_sp && deref_error.Success()) {
        base = std::move(synth_deref_sp);
      }
      if (!base || deref_error.Fail()) {
        std::string errMsg = llvm::formatv(
            "Failed to dereference synthetic value: {0}", deref_error);
        return llvm::make_error<DILDiagnosticError>(
            m_expr, errMsg, node->GetLocation(), node->GetFieldName().size());
      }

      // Some synthetic plug-ins fail to set the error in Dereference
      if (!base) {
        std::string errMsg = "Failed to dereference synthetic value";
        return llvm::make_error<DILDiagnosticError>(
            m_expr, errMsg, node->GetLocation(), node->GetFieldName().size());
      }
      expr_is_ptr = false;
    }
  }

  if (m_check_ptr_vs_member) {
    bool base_is_ptr = base->IsPointerType();

    if (expr_is_ptr != base_is_ptr) {
      if (base_is_ptr) {
        std::string errMsg =
            llvm::formatv("member reference type {0} is a pointer; "
                          "did you mean to use '->'?",
                          base->GetCompilerType().TypeDescription());
        return llvm::make_error<DILDiagnosticError>(
            m_expr, errMsg, node->GetLocation(), node->GetFieldName().size());
      } else {
        std::string errMsg =
            llvm::formatv("member reference type {0} is not a pointer; "
                          "did you mean to use '.'?",
                          base->GetCompilerType().TypeDescription());
        return llvm::make_error<DILDiagnosticError>(
            m_expr, errMsg, node->GetLocation(), node->GetFieldName().size());
      }
    }
  }

  lldb::ValueObjectSP field_obj =
      base->GetChildMemberWithName(node->GetFieldName());
  if (!field_obj) {
    if (m_use_synthetic) {
      field_obj = base->GetSyntheticValue();
      if (field_obj)
        field_obj = field_obj->GetChildMemberWithName(node->GetFieldName());
    }

    if (!m_use_synthetic || !field_obj) {
      std::string errMsg = llvm::formatv(
          "\"{0}\" is not a member of \"({1}) {2}\"", node->GetFieldName(),
          base->GetTypeName().AsCString("<invalid type>"), base->GetName());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, errMsg, node->GetLocation(), node->GetFieldName().size());
    }
  }

  if (field_obj) {
    if (m_use_dynamic != lldb::eNoDynamicValues) {
      lldb::ValueObjectSP dynamic_val_sp =
          field_obj->GetDynamicValue(m_use_dynamic);
      if (dynamic_val_sp)
        field_obj = dynamic_val_sp;
    }
    return field_obj;
  }

  CompilerType base_type = base->GetCompilerType();
  if (node->GetIsArrow() && base->IsPointerType())
    base_type = base_type.GetPointeeType();
  std::string errMsg = llvm::formatv(
      "\"{0}\" is not a member of \"({1}) {2}\"", node->GetFieldName(),
      base->GetTypeName().AsCString("<invalid type>"), base->GetName());
  return llvm::make_error<DILDiagnosticError>(
      m_expr, errMsg, node->GetLocation(), node->GetFieldName().size());
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const ArraySubscriptNode *node) {
  auto lhs_or_err = Evaluate(node->GetBase());
  if (!lhs_or_err)
    return lhs_or_err;
  lldb::ValueObjectSP base = *lhs_or_err;

  StreamString var_expr_path_strm;
  uint64_t child_idx = node->GetIndex();
  lldb::ValueObjectSP child_valobj_sp;

  bool is_incomplete_array = false;
  CompilerType base_type = base->GetCompilerType().GetNonReferenceType();
  base->GetExpressionPath(var_expr_path_strm);

  if (base_type.IsPointerType()) {
    bool is_objc_pointer = true;

    if (base->GetCompilerType().GetMinimumLanguage() != lldb::eLanguageTypeObjC)
      is_objc_pointer = false;
    else if (!base->GetCompilerType().IsPointerType())
      is_objc_pointer = false;

    if (!m_use_synthetic && is_objc_pointer) {
      std::string err_msg = llvm::formatv(
          "\"({0}) {1}\" is an Objective-C pointer, and cannot be subscripted",
          base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation());
    }
    if (is_objc_pointer) {
      lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
      if (!synthetic || synthetic == base) {
        std::string err_msg =
            llvm::formatv("\"({0}) {1}\" is not an array type",
                          base->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetData());
        return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                    node->GetLocation());
      }
      if (static_cast<uint32_t>(child_idx) >=
          synthetic->GetNumChildrenIgnoringErrors()) {
        std::string err_msg = llvm::formatv(
            "array index {0} is not valid for \"({1}) {2}\"", child_idx,
            base->GetTypeName().AsCString("<invalid type>"),
            var_expr_path_strm.GetData());
        return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                    node->GetLocation());
      }
      child_valobj_sp = synthetic->GetChildAtIndex(child_idx);
      if (!child_valobj_sp) {
        std::string err_msg = llvm::formatv(
            "array index {0} is not valid for \"({1}) {2}\"", child_idx,
            base->GetTypeName().AsCString("<invalid type>"),
            var_expr_path_strm.GetData());
        return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                    node->GetLocation());
      }
      if (m_use_dynamic != lldb::eNoDynamicValues) {
        if (auto dynamic_sp = child_valobj_sp->GetDynamicValue(m_use_dynamic))
          child_valobj_sp = std::move(dynamic_sp);
      }
      return child_valobj_sp;
    }

    child_valobj_sp = base->GetSyntheticArrayMember(child_idx, true);
    if (!child_valobj_sp) {
      std::string err_msg = llvm::formatv(
          "failed to use pointer as array for index {0} for "
          "\"({1}) {2}\"",
          child_idx, base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      if (base_type.IsPointerToVoid())
        err_msg = "subscript of pointer to incomplete type 'void'";
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation());
    }
  } else if (base_type.IsArrayType(nullptr, nullptr, &is_incomplete_array)) {
    child_valobj_sp = base->GetChildAtIndex(child_idx);
    if (!child_valobj_sp && (is_incomplete_array || m_use_synthetic))
      child_valobj_sp = base->GetSyntheticArrayMember(child_idx, true);
    if (!child_valobj_sp) {
      std::string err_msg = llvm::formatv(
          "array index {0} is not valid for \"({1}) {2}\"", child_idx,
          base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation());
    }
  } else if (base_type.IsScalarType()) {
    child_valobj_sp =
        base->GetSyntheticBitFieldChild(child_idx, child_idx, true);
    if (!child_valobj_sp) {
      std::string err_msg = llvm::formatv(
          "bitfield range {0}-{1} is not valid for \"({2}) {3}\"", child_idx,
          child_idx, base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation(), 1);
    }
  } else {
    lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
    if (!m_use_synthetic || !synthetic || synthetic == base) {
      std::string err_msg =
          llvm::formatv("\"{0}\" is not an array type",
                        base->GetTypeName().AsCString("<invalid type>"));
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation(), 1);
    }
    if (static_cast<uint32_t>(child_idx) >=
        synthetic->GetNumChildrenIgnoringErrors(child_idx + 1)) {
      std::string err_msg = llvm::formatv(
          "array index {0} is not valid for \"({1}) {2}\"", child_idx,
          base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation(), 1);
    }
    child_valobj_sp = synthetic->GetChildAtIndex(child_idx);
    if (!child_valobj_sp) {
      std::string err_msg = llvm::formatv(
          "array index {0} is not valid for \"({1}) {2}\"", child_idx,
          base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node->GetLocation(), 1);
    }
  }

  if (child_valobj_sp) {
    if (m_use_dynamic != lldb::eNoDynamicValues) {
      if (auto dynamic_sp = child_valobj_sp->GetDynamicValue(m_use_dynamic))
        child_valobj_sp = std::move(dynamic_sp);
    }
    return child_valobj_sp;
  }

  int64_t signed_child_idx = node->GetIndex();
  return base->GetSyntheticArrayMember(signed_child_idx, true);
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const BitFieldExtractionNode *node) {
  auto lhs_or_err = Evaluate(node->GetBase());
  if (!lhs_or_err)
    return lhs_or_err;
  lldb::ValueObjectSP base = *lhs_or_err;
  int64_t first_index = node->GetFirstIndex();
  int64_t last_index = node->GetLastIndex();

  // if the format given is [high-low], swap range
  if (first_index > last_index)
    std::swap(first_index, last_index);

  Status error;
  if (base->GetCompilerType().IsReferenceType()) {
    base = base->Dereference(error);
    if (error.Fail())
      return error.ToError();
  }
  lldb::ValueObjectSP child_valobj_sp =
      base->GetSyntheticBitFieldChild(first_index, last_index, true);
  if (!child_valobj_sp) {
    std::string message = llvm::formatv(
        "bitfield range {0}-{1} is not valid for \"({2}) {3}\"", first_index,
        last_index, base->GetTypeName().AsCString("<invalid type>"),
        base->GetName().AsCString());
    return llvm::make_error<DILDiagnosticError>(m_expr, message,
                                                node->GetLocation());
  }
  return child_valobj_sp;
}

static llvm::Expected<lldb::TypeSystemSP>
GetTypeSystemFromCU(std::shared_ptr<StackFrame> ctx) {
  SymbolContext symbol_context =
      ctx->GetSymbolContext(lldb::eSymbolContextCompUnit);
  lldb::LanguageType language = symbol_context.comp_unit->GetLanguage();

  symbol_context = ctx->GetSymbolContext(lldb::eSymbolContextModule);
  return symbol_context.module_sp->GetTypeSystemForLanguage(language);
}

static CompilerType GetBasicType(lldb::TypeSystemSP type_system,
                                 lldb::BasicType basic_type) {
  if (type_system)
    return type_system.get()->GetBasicTypeFromAST(basic_type);

  return CompilerType();
}

llvm::Expected<CompilerType>
Interpreter::PickIntegerType(lldb::TypeSystemSP type_system,
                             std::shared_ptr<ExecutionContextScope> ctx,
                             const IntegerLiteralNode *literal) {
  // Binary, Octal, Hexadecimal and literals with a U suffix are allowed to be
  // an unsigned integer.
  bool unsigned_is_allowed = literal->IsUnsigned() || literal->GetRadix() != 10;
  llvm::APInt apint = literal->GetValue();

  llvm::SmallVector<std::pair<lldb::BasicType, lldb::BasicType>, 3> candidates;
  if (literal->GetTypeSuffix() <= IntegerTypeSuffix::None)
    candidates.emplace_back(lldb::eBasicTypeInt,
                            unsigned_is_allowed ? lldb::eBasicTypeUnsignedInt
                                                : lldb::eBasicTypeInvalid);
  if (literal->GetTypeSuffix() <= IntegerTypeSuffix::Long)
    candidates.emplace_back(lldb::eBasicTypeLong,
                            unsigned_is_allowed ? lldb::eBasicTypeUnsignedLong
                                                : lldb::eBasicTypeInvalid);
  candidates.emplace_back(lldb::eBasicTypeLongLong,
                          lldb::eBasicTypeUnsignedLongLong);
  for (auto [signed_, unsigned_] : candidates) {
    CompilerType signed_type = type_system->GetBasicTypeFromAST(signed_);
    if (!signed_type)
      continue;
    llvm::Expected<uint64_t> size = signed_type.GetBitSize(ctx.get());
    if (!size)
      return size.takeError();
    if (!literal->IsUnsigned() && apint.isIntN(*size - 1))
      return signed_type;
    if (unsigned_ != lldb::eBasicTypeInvalid && apint.isIntN(*size))
      return type_system->GetBasicTypeFromAST(unsigned_);
  }

  return llvm::make_error<DILDiagnosticError>(
      m_expr,
      "integer literal is too large to be represented in any integer type",
      literal->GetLocation());
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const IntegerLiteralNode *node) {
  llvm::Expected<lldb::TypeSystemSP> type_system =
      GetTypeSystemFromCU(m_exe_ctx_scope);
  if (!type_system)
    return type_system.takeError();

  llvm::Expected<CompilerType> type =
      PickIntegerType(*type_system, m_exe_ctx_scope, node);
  if (!type)
    return type.takeError();

  Scalar scalar = node->GetValue();
  // APInt from StringRef::getAsInteger comes with just enough bitwidth to
  // hold the value. This adjusts APInt bitwidth to match the compiler type.
  llvm::Expected<uint64_t> type_bitsize =
      type->GetBitSize(m_exe_ctx_scope.get());
  if (!type_bitsize)
    return type_bitsize.takeError();
  scalar.TruncOrExtendTo(*type_bitsize, false);
  return ValueObject::CreateValueObjectFromScalar(m_target, scalar, *type,
                                                  "result");
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const FloatLiteralNode *node) {
  llvm::Expected<lldb::TypeSystemSP> type_system =
      GetTypeSystemFromCU(m_exe_ctx_scope);
  if (!type_system)
    return type_system.takeError();

  bool isFloat =
      &node->GetValue().getSemantics() == &llvm::APFloat::IEEEsingle();
  lldb::BasicType basic_type =
      isFloat ? lldb::eBasicTypeFloat : lldb::eBasicTypeDouble;
  CompilerType type = GetBasicType(*type_system, basic_type);

  if (!type)
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "unable to create a const literal", node->GetLocation());

  Scalar scalar = node->GetValue();
  return ValueObject::CreateValueObjectFromScalar(m_target, scalar, type,
                                                  "result");
}

} // namespace lldb_private::dil
