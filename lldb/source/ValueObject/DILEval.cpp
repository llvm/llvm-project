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

lldb::ValueObjectSP
GetDynamicOrSyntheticValue(lldb::ValueObjectSP value_sp,
                           lldb::DynamicValueType use_dynamic,
                           bool use_synthetic) {
  if (!value_sp)
    return nullptr;

  if (use_dynamic != lldb::eNoDynamicValues) {
    lldb::ValueObjectSP dynamic_sp = value_sp->GetDynamicValue(use_dynamic);
    if (dynamic_sp)
      value_sp = dynamic_sp;
  }

  if (use_synthetic) {
    lldb::ValueObjectSP synthetic_sp = value_sp->GetSyntheticValue();
    if (synthetic_sp)
      value_sp = synthetic_sp;
  }

  return value_sp;
}

static llvm::Expected<lldb::TypeSystemSP>
GetTypeSystemFromCU(std::shared_ptr<ExecutionContextScope> ctx) {
  auto stack_frame = ctx->CalculateStackFrame();
  if (!stack_frame)
    return llvm::createStringError("no stack frame in this context");
  SymbolContext symbol_context =
      stack_frame->GetSymbolContext(lldb::eSymbolContextCompUnit);

  if (!symbol_context.comp_unit)
    return llvm::createStringError("no compile unit in this context");
  lldb::LanguageType language = symbol_context.comp_unit->GetLanguage();

  symbol_context = stack_frame->GetSymbolContext(lldb::eSymbolContextModule);
  return symbol_context.module_sp->GetTypeSystemForLanguage(language);
}

static CompilerType GetBasicType(lldb::TypeSystemSP type_system,
                                 lldb::BasicType basic_type) {
  if (type_system)
    return type_system.get()->GetBasicTypeFromAST(basic_type);

  return CompilerType();
}

static lldb::ValueObjectSP ArrayToPointerConversion(ValueObject &valobj,
                                                    ExecutionContextScope &ctx,
                                                    llvm::StringRef name) {
  uint64_t addr = valobj.GetLoadAddress();
  ExecutionContext exe_ctx;
  ctx.CalculateExecutionContext(exe_ctx);
  return ValueObject::CreateValueObjectFromAddress(
      name, addr, exe_ctx,
      valobj.GetCompilerType().GetArrayElementType(&ctx).GetPointerType(),
      /* do_deref */ false);
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::UnaryConversion(lldb::ValueObjectSP valobj, uint32_t location) {
  if (!valobj)
    return llvm::make_error<DILDiagnosticError>(m_expr, "invalid value object",
                                                location);
  llvm::Expected<lldb::TypeSystemSP> type_system =
      GetTypeSystemFromCU(m_exe_ctx_scope);
  if (!type_system)
    return type_system.takeError();

  CompilerType in_type = valobj->GetCompilerType();
  if (valobj->IsBitfield()) {
    // Promote bitfields. If `int` can represent the bitfield value, it is
    // converted to `int`. Otherwise, if `unsigned int` can represent it, it
    // is converted to `unsigned int`. Otherwise, it is treated as its
    // underlying type.
    uint32_t bitfield_size = valobj->GetBitfieldBitSize();
    // Some bitfields have undefined size (e.g. result of ternary operation).
    // The AST's `bitfield_size` of those is 0, and no promotion takes place.
    if (bitfield_size > 0 && in_type.IsInteger()) {
      CompilerType int_type = GetBasicType(*type_system, lldb::eBasicTypeInt);
      CompilerType uint_type =
          GetBasicType(*type_system, lldb::eBasicTypeUnsignedInt);
      llvm::Expected<uint64_t> int_bit_size =
          int_type.GetBitSize(m_exe_ctx_scope.get());
      if (!int_bit_size)
        return int_bit_size.takeError();
      llvm::Expected<uint64_t> uint_bit_size =
          uint_type.GetBitSize(m_exe_ctx_scope.get());
      if (!uint_bit_size)
        return int_bit_size.takeError();
      if (bitfield_size < *int_bit_size ||
          (in_type.IsSigned() && bitfield_size == *int_bit_size))
        return valobj->CastToBasicType(int_type);
      if (bitfield_size <= *uint_bit_size)
        return valobj->CastToBasicType(uint_type);
      // Re-create as a const value with the same underlying type
      Scalar scalar;
      bool resolved = valobj->ResolveValue(scalar);
      if (!resolved)
        return llvm::createStringError("invalid scalar value");
      return ValueObject::CreateValueObjectFromScalar(m_target, scalar, in_type,
                                                      "result");
    }
  }

  if (in_type.IsArrayType())
    valobj = ArrayToPointerConversion(*valobj, *m_exe_ctx_scope, "result");

  if (valobj->GetCompilerType().IsInteger() ||
      valobj->GetCompilerType().IsUnscopedEnumerationType()) {
    llvm::Expected<CompilerType> promoted_type =
        type_system.get()->DoIntegralPromotion(valobj->GetCompilerType(),
                                               m_exe_ctx_scope.get());
    if (!promoted_type)
      return promoted_type.takeError();
    if (!promoted_type->CompareTypes(valobj->GetCompilerType()))
      return valobj->CastToBasicType(*promoted_type);
  }

  return valobj;
}

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

llvm::Expected<lldb::ValueObjectSP> Interpreter::Evaluate(const ASTNode &node) {
  // Evaluate an AST.
  auto value_or_error = node.Accept(this);
  // Convert SP with a nullptr to an error.
  if (value_or_error && !*value_or_error)
    return llvm::make_error<DILDiagnosticError>(m_expr, "invalid value object",
                                                node.GetLocation());
  // Return the computed value-or-error. The caller is responsible for
  // checking if an error occurred during the evaluation.
  return value_or_error;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::EvaluateAndDereference(const ASTNode &node) {
  auto valobj_or_err = Evaluate(node);
  if (!valobj_or_err)
    return valobj_or_err;
  lldb::ValueObjectSP valobj = *valobj_or_err;

  Status error;
  if (valobj->GetCompilerType().IsReferenceType()) {
    valobj = valobj->Dereference(error);
    if (error.Fail())
      return error.ToError();
  }
  return valobj;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const IdentifierNode &node) {
  lldb::DynamicValueType use_dynamic = m_use_dynamic;

  lldb::ValueObjectSP identifier =
      LookupIdentifier(node.GetName(), m_exe_ctx_scope, use_dynamic);

  if (!identifier)
    identifier = LookupGlobalIdentifier(node.GetName(), m_exe_ctx_scope,
                                        m_target, use_dynamic);
  if (!identifier) {
    std::string errMsg =
        llvm::formatv("use of undeclared identifier '{0}'", node.GetName());
    return llvm::make_error<DILDiagnosticError>(
        m_expr, errMsg, node.GetLocation(), node.GetName().size());
  }

  return identifier;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const UnaryOpNode &node) {
  Status error;
  auto op_or_err = Evaluate(node.GetOperand());
  if (!op_or_err)
    return op_or_err;

  lldb::ValueObjectSP operand = *op_or_err;

  switch (node.GetKind()) {
  case UnaryOpKind::Deref: {
    lldb::ValueObjectSP dynamic_op = operand->GetDynamicValue(m_use_dynamic);
    if (dynamic_op)
      operand = dynamic_op;

    lldb::ValueObjectSP child_sp = operand->Dereference(error);
    if (!child_sp && m_use_synthetic) {
      if (lldb::ValueObjectSP synth_obj_sp = operand->GetSyntheticValue()) {
        error.Clear();
        child_sp = synth_obj_sp->Dereference(error);
      }
    }
    if (error.Fail())
      return llvm::make_error<DILDiagnosticError>(m_expr, error.AsCString(),
                                                  node.GetLocation());

    return child_sp;
  }
  case UnaryOpKind::AddrOf: {
    Status error;
    lldb::ValueObjectSP value = operand->AddressOf(error);
    if (error.Fail())
      return llvm::make_error<DILDiagnosticError>(m_expr, error.AsCString(),
                                                  node.GetLocation());

    return value;
  }
  case UnaryOpKind::Minus: {
    if (operand->GetCompilerType().IsReferenceType()) {
      operand = operand->Dereference(error);
      if (error.Fail())
        return error.ToError();
    }
    llvm::Expected<lldb::ValueObjectSP> conv_op =
        UnaryConversion(operand, node.GetOperand().GetLocation());
    if (!conv_op)
      return conv_op;
    operand = *conv_op;
    CompilerType operand_type = operand->GetCompilerType();
    if (!operand_type.IsScalarType()) {
      std::string errMsg =
          llvm::formatv("invalid argument type '{0}' to unary expression",
                        operand_type.GetTypeName());
      return llvm::make_error<DILDiagnosticError>(m_expr, errMsg,
                                                  node.GetLocation());
    }
    Scalar scalar;
    bool resolved = operand->ResolveValue(scalar);
    if (!resolved)
      break;

    bool negated = scalar.UnaryNegate();
    if (negated)
      return ValueObject::CreateValueObjectFromScalar(
          m_target, scalar, operand->GetCompilerType(), "result");
    break;
  }
  case UnaryOpKind::Plus: {
    if (operand->GetCompilerType().IsReferenceType()) {
      operand = operand->Dereference(error);
      if (error.Fail())
        return error.ToError();
    }
    llvm::Expected<lldb::ValueObjectSP> conv_op =
        UnaryConversion(operand, node.GetOperand().GetLocation());
    if (!conv_op)
      return conv_op;
    operand = *conv_op;
    CompilerType operand_type = operand->GetCompilerType();
    if (!operand_type.IsScalarType() &&
        // Unary plus is allowed for pointers.
        !operand_type.IsPointerType()) {
      std::string errMsg =
          llvm::formatv("invalid argument type '{0}' to unary expression",
                        operand_type.GetTypeName());
      return llvm::make_error<DILDiagnosticError>(m_expr, errMsg,
                                                  node.GetLocation());
    }
    return operand;
  }
  }
  return llvm::make_error<DILDiagnosticError>(m_expr, "invalid unary operation",
                                              node.GetLocation());
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const MemberOfNode &node) {
  auto base_or_err = Evaluate(node.GetBase());
  if (!base_or_err)
    return base_or_err;
  bool expr_is_ptr = node.GetIsArrow();
  lldb::ValueObjectSP base = *base_or_err;

  // Perform some basic type & correctness checking.
  if (node.GetIsArrow()) {
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
            m_expr, errMsg, node.GetLocation(), node.GetFieldName().size());
      }

      // Some synthetic plug-ins fail to set the error in Dereference
      if (!base) {
        std::string errMsg = "Failed to dereference synthetic value";
        return llvm::make_error<DILDiagnosticError>(
            m_expr, errMsg, node.GetLocation(), node.GetFieldName().size());
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
            m_expr, errMsg, node.GetLocation(), node.GetFieldName().size());
      } else {
        std::string errMsg =
            llvm::formatv("member reference type {0} is not a pointer; "
                          "did you mean to use '.'?",
                          base->GetCompilerType().TypeDescription());
        return llvm::make_error<DILDiagnosticError>(
            m_expr, errMsg, node.GetLocation(), node.GetFieldName().size());
      }
    }
  }

  lldb::ValueObjectSP field_obj =
      base->GetChildMemberWithName(node.GetFieldName());
  if (!field_obj) {
    if (m_use_synthetic) {
      field_obj = base->GetSyntheticValue();
      if (field_obj)
        field_obj = field_obj->GetChildMemberWithName(node.GetFieldName());
    }

    if (!m_use_synthetic || !field_obj) {
      std::string errMsg = llvm::formatv(
          "\"{0}\" is not a member of \"({1}) {2}\"", node.GetFieldName(),
          base->GetTypeName().AsCString("<invalid type>"), base->GetName());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, errMsg, node.GetLocation(), node.GetFieldName().size());
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
  if (node.GetIsArrow() && base->IsPointerType())
    base_type = base_type.GetPointeeType();
  std::string errMsg = llvm::formatv(
      "\"{0}\" is not a member of \"({1}) {2}\"", node.GetFieldName(),
      base->GetTypeName().AsCString("<invalid type>"), base->GetName());
  return llvm::make_error<DILDiagnosticError>(
      m_expr, errMsg, node.GetLocation(), node.GetFieldName().size());
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const ArraySubscriptNode &node) {
  auto idx_or_err = EvaluateAndDereference(node.GetIndex());
  if (!idx_or_err)
    return idx_or_err;
  lldb::ValueObjectSP idx = *idx_or_err;

  if (!idx->GetCompilerType().IsIntegerOrUnscopedEnumerationType()) {
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "array subscript is not an integer", node.GetLocation());
  }

  StreamString var_expr_path_strm;
  uint64_t child_idx = idx->GetValueAsUnsigned(0);
  lldb::ValueObjectSP child_valobj_sp;

  auto base_or_err = Evaluate(node.GetBase());
  if (!base_or_err)
    return base_or_err;
  lldb::ValueObjectSP base = *base_or_err;

  CompilerType base_type = base->GetCompilerType().GetNonReferenceType();
  base->GetExpressionPath(var_expr_path_strm);
  bool is_incomplete_array = false;
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
                                                  node.GetLocation());
    }
    if (is_objc_pointer) {
      lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
      if (!synthetic || synthetic == base) {
        std::string err_msg =
            llvm::formatv("\"({0}) {1}\" is not an array type",
                          base->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetData());
        return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                    node.GetLocation());
      }
      if (static_cast<uint32_t>(child_idx) >=
          synthetic->GetNumChildrenIgnoringErrors()) {
        std::string err_msg = llvm::formatv(
            "array index {0} is not valid for \"({1}) {2}\"", child_idx,
            base->GetTypeName().AsCString("<invalid type>"),
            var_expr_path_strm.GetData());
        return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                    node.GetLocation());
      }
      child_valobj_sp = synthetic->GetChildAtIndex(child_idx);
      if (!child_valobj_sp) {
        std::string err_msg = llvm::formatv(
            "array index {0} is not valid for \"({1}) {2}\"", child_idx,
            base->GetTypeName().AsCString("<invalid type>"),
            var_expr_path_strm.GetData());
        return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                    node.GetLocation());
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
                                                  node.GetLocation());
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
                                                  node.GetLocation());
    }
  } else if (base_type.IsScalarType()) {
    child_valobj_sp =
        base->GetSyntheticBitFieldChild(child_idx, child_idx, true);
    if (!child_valobj_sp) {
      std::string err_msg = llvm::formatv(
          "bitfield range {0}:{1} is not valid for \"({2}) {3}\"", child_idx,
          child_idx, base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node.GetLocation(), 1);
    }
  } else {
    lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
    if (!m_use_synthetic || !synthetic || synthetic == base) {
      std::string err_msg =
          llvm::formatv("\"{0}\" is not an array type",
                        base->GetTypeName().AsCString("<invalid type>"));
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node.GetLocation(), 1);
    }
    if (static_cast<uint32_t>(child_idx) >=
        synthetic->GetNumChildrenIgnoringErrors(child_idx + 1)) {
      std::string err_msg = llvm::formatv(
          "array index {0} is not valid for \"({1}) {2}\"", child_idx,
          base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node.GetLocation(), 1);
    }
    child_valobj_sp = synthetic->GetChildAtIndex(child_idx);
    if (!child_valobj_sp) {
      std::string err_msg = llvm::formatv(
          "array index {0} is not valid for \"({1}) {2}\"", child_idx,
          base->GetTypeName().AsCString("<invalid type>"),
          var_expr_path_strm.GetData());
      return llvm::make_error<DILDiagnosticError>(m_expr, std::move(err_msg),
                                                  node.GetLocation(), 1);
    }
  }

  if (child_valobj_sp) {
    if (m_use_dynamic != lldb::eNoDynamicValues) {
      if (auto dynamic_sp = child_valobj_sp->GetDynamicValue(m_use_dynamic))
        child_valobj_sp = std::move(dynamic_sp);
    }
    return child_valobj_sp;
  }

  bool success;
  int64_t signed_child_idx = idx->GetValueAsSigned(0, &success);
  if (!success)
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "could not get the index as an integer",
        node.GetIndex().GetLocation());
  return base->GetSyntheticArrayMember(signed_child_idx, true);
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const BitFieldExtractionNode &node) {
  auto first_idx_or_err = EvaluateAndDereference(node.GetFirstIndex());
  if (!first_idx_or_err)
    return first_idx_or_err;
  lldb::ValueObjectSP first_idx = *first_idx_or_err;
  auto last_idx_or_err = EvaluateAndDereference(node.GetLastIndex());
  if (!last_idx_or_err)
    return last_idx_or_err;
  lldb::ValueObjectSP last_idx = *last_idx_or_err;

  if (!first_idx->GetCompilerType().IsIntegerOrUnscopedEnumerationType() ||
      !last_idx->GetCompilerType().IsIntegerOrUnscopedEnumerationType()) {
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "bit index is not an integer", node.GetLocation());
  }

  bool success_first, success_last;
  int64_t first_index = first_idx->GetValueAsSigned(0, &success_first);
  int64_t last_index = last_idx->GetValueAsSigned(0, &success_last);
  if (!success_first || !success_last)
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "could not get the index as an integer", node.GetLocation());

  // if the format given is [high-low], swap range
  if (first_index > last_index)
    std::swap(first_index, last_index);

  auto base_or_err = EvaluateAndDereference(node.GetBase());
  if (!base_or_err)
    return base_or_err;
  lldb::ValueObjectSP base = *base_or_err;
  lldb::ValueObjectSP child_valobj_sp =
      base->GetSyntheticBitFieldChild(first_index, last_index, true);
  if (!child_valobj_sp) {
    std::string message = llvm::formatv(
        "bitfield range {0}:{1} is not valid for \"({2}) {3}\"", first_index,
        last_index, base->GetTypeName().AsCString("<invalid type>"),
        base->GetName().AsCString());
    return llvm::make_error<DILDiagnosticError>(m_expr, message,
                                                node.GetLocation());
  }
  return child_valobj_sp;
}

llvm::Expected<CompilerType>
Interpreter::PickIntegerType(lldb::TypeSystemSP type_system,
                             std::shared_ptr<ExecutionContextScope> ctx,
                             const IntegerLiteralNode &literal) {
  // Binary, Octal, Hexadecimal and literals with a U suffix are allowed to be
  // an unsigned integer.
  bool unsigned_is_allowed = literal.IsUnsigned() || literal.GetRadix() != 10;
  llvm::APInt apint = literal.GetValue();

  llvm::SmallVector<std::pair<lldb::BasicType, lldb::BasicType>, 3> candidates;
  if (literal.GetTypeSuffix() <= IntegerTypeSuffix::None)
    candidates.emplace_back(lldb::eBasicTypeInt,
                            unsigned_is_allowed ? lldb::eBasicTypeUnsignedInt
                                                : lldb::eBasicTypeInvalid);
  if (literal.GetTypeSuffix() <= IntegerTypeSuffix::Long)
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
    if (!literal.IsUnsigned() && apint.isIntN(*size - 1))
      return signed_type;
    if (unsigned_ != lldb::eBasicTypeInvalid && apint.isIntN(*size))
      return type_system->GetBasicTypeFromAST(unsigned_);
  }

  return llvm::make_error<DILDiagnosticError>(
      m_expr,
      "integer literal is too large to be represented in any integer type",
      literal.GetLocation());
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const IntegerLiteralNode &node) {
  llvm::Expected<lldb::TypeSystemSP> type_system =
      GetTypeSystemFromCU(m_exe_ctx_scope);
  if (!type_system)
    return type_system.takeError();

  llvm::Expected<CompilerType> type =
      PickIntegerType(*type_system, m_exe_ctx_scope, node);
  if (!type)
    return type.takeError();

  Scalar scalar = node.GetValue();
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
Interpreter::Visit(const FloatLiteralNode &node) {
  llvm::Expected<lldb::TypeSystemSP> type_system =
      GetTypeSystemFromCU(m_exe_ctx_scope);
  if (!type_system)
    return type_system.takeError();

  bool isFloat =
      &node.GetValue().getSemantics() == &llvm::APFloat::IEEEsingle();
  lldb::BasicType basic_type =
      isFloat ? lldb::eBasicTypeFloat : lldb::eBasicTypeDouble;
  CompilerType type = GetBasicType(*type_system, basic_type);

  if (!type)
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "unable to create a const literal", node.GetLocation());

  Scalar scalar = node.GetValue();
  return ValueObject::CreateValueObjectFromScalar(m_target, scalar, type,
                                                  "result");
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const BooleanLiteralNode &node) {
  bool value = node.GetValue();
  return ValueObject::CreateValueObjectFromBool(m_target, value, "result");
}

llvm::Expected<CastKind>
Interpreter::VerifyArithmeticCast(CompilerType source_type,
                                  CompilerType target_type, int location) {
  if (source_type.IsPointerType() || source_type.IsNullPtrType()) {
    // Cast from pointer to float/double is not allowed.
    if (target_type.IsFloat()) {
      std::string errMsg = llvm::formatv("Cast from {0} to {1} is not allowed",
                                         source_type.TypeDescription(),
                                         target_type.TypeDescription());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, std::move(errMsg), location,
          source_type.TypeDescription().length());
    }

    // Casting from pointer to bool is always valid.
    if (target_type.IsBoolean())
      return CastKind::eArithmetic;

    // Otherwise check if the result type is at least as big as the pointer
    // size.
    uint64_t type_byte_size = 0;
    uint64_t rhs_type_byte_size = 0;
    if (auto temp = target_type.GetByteSize(m_exe_ctx_scope.get())) {
      type_byte_size = *temp;
    } else {
      std::string errMsg = llvm::formatv("unable to get byte size for type {0}",
                                         target_type.TypeDescription());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, std::move(errMsg), location,
          target_type.TypeDescription().length());
    }

    if (auto temp = source_type.GetByteSize(m_exe_ctx_scope.get())) {
      rhs_type_byte_size = *temp;
    } else {
      std::string errMsg = llvm::formatv("unable to get byte size for type {0}",
                                         source_type.TypeDescription());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, std::move(errMsg), location,
          source_type.TypeDescription().length());
    }

    if (type_byte_size < rhs_type_byte_size) {
      std::string errMsg = llvm::formatv(
          "cast from pointer to smaller type {0} loses information",
          target_type.TypeDescription());
      return llvm::make_error<DILDiagnosticError>(
          m_expr, std::move(errMsg), location,
          source_type.TypeDescription().length());
    }
  } else if (!source_type.IsScalarType() && !source_type.IsEnumerationType()) {
    // Otherwise accept only arithmetic types and enums.
    std::string errMsg = llvm::formatv("cannot convert {0} to {1}",
                                       source_type.TypeDescription(),
                                       target_type.TypeDescription());

    return llvm::make_error<DILDiagnosticError>(
        m_expr, std::move(errMsg), location,
        source_type.TypeDescription().length());
  }
  return CastKind::eArithmetic;
}

llvm::Expected<CastKind>
Interpreter::VerifyCastType(lldb::ValueObjectSP operand,
                            CompilerType source_type, CompilerType target_type,
                            int location) {

  if (target_type.IsScalarType())
    return VerifyArithmeticCast(source_type, target_type, location);

  if (target_type.IsEnumerationType()) {
    // Cast to enum type.
    if (!source_type.IsScalarType() && !source_type.IsEnumerationType()) {
      std::string errMsg = llvm::formatv("Cast from {0} to {1} is not allowed",
                                         source_type.TypeDescription(),
                                         target_type.TypeDescription());

      return llvm::make_error<DILDiagnosticError>(
          m_expr, std::move(errMsg), location,
          source_type.TypeDescription().length());
    }
    return CastKind::eEnumeration;
  }

  if (target_type.IsPointerType()) {
    if (!source_type.IsInteger() && !source_type.IsEnumerationType() &&
        !source_type.IsArrayType() && !source_type.IsPointerType() &&
        !source_type.IsNullPtrType()) {
      std::string errMsg = llvm::formatv(
          "cannot cast from type {0} to pointer type {1}",
          source_type.TypeDescription(), target_type.TypeDescription());

      return llvm::make_error<DILDiagnosticError>(
          m_expr, std::move(errMsg), location,
          source_type.TypeDescription().length());
    }
    return CastKind::ePointer;
  }

  // Unsupported cast.
  std::string errMsg = llvm::formatv(
      "casting of {0} to {1} is not implemented yet",
      source_type.TypeDescription(), target_type.TypeDescription());
  return llvm::make_error<DILDiagnosticError>(
      m_expr, std::move(errMsg), location,
      source_type.TypeDescription().length());
}

llvm::Expected<lldb::ValueObjectSP> Interpreter::Visit(const CastNode &node) {
  auto operand_or_err = Evaluate(node.GetOperand());

  if (!operand_or_err)
    return operand_or_err;

  lldb::ValueObjectSP operand = *operand_or_err;
  CompilerType op_type = operand->GetCompilerType();
  CompilerType target_type = node.GetType();

  if (op_type.IsReferenceType())
    op_type = op_type.GetNonReferenceType();
  if (target_type.IsScalarType() && op_type.IsArrayType()) {
    operand = ArrayToPointerConversion(*operand, *m_exe_ctx_scope,
                                       operand->GetName().GetStringRef());
    op_type = operand->GetCompilerType();
  }
  auto type_or_err =
      VerifyCastType(operand, op_type, target_type, node.GetLocation());
  if (!type_or_err)
    return type_or_err.takeError();

  CastKind cast_kind = *type_or_err;
  if (operand->GetCompilerType().IsReferenceType()) {
    Status error;
    operand = operand->Dereference(error);
    if (error.Fail())
      return llvm::make_error<DILDiagnosticError>(m_expr, error.AsCString(),
                                                  node.GetLocation());
  }

  switch (cast_kind) {
  case CastKind::eEnumeration: {
    if (op_type.IsFloat() || op_type.IsInteger() || op_type.IsEnumerationType())
      return operand->CastToEnumType(target_type);
    break;
  }
  case CastKind::eArithmetic: {
    if (op_type.IsPointerType() || op_type.IsNullPtrType() ||
        op_type.IsScalarType() || op_type.IsEnumerationType())
      return operand->CastToBasicType(target_type);
    break;
  }
  case CastKind::ePointer: {
    uint64_t addr = op_type.IsArrayType()
                        ? operand->GetLoadAddress()
                        : (op_type.IsSigned() ? operand->GetValueAsSigned(0)
                                              : operand->GetValueAsUnsigned(0));
    llvm::StringRef name = "result";
    ExecutionContext exe_ctx(m_target.get(), false);
    return ValueObject::CreateValueObjectFromAddress(name, addr, exe_ctx,
                                                     target_type,
                                                     /* do_deref */ false);
  }
  case CastKind::eNone: {
    return lldb::ValueObjectSP();
  }
  } // switch

  std::string errMsg =
      llvm::formatv("unable to cast from '{0}' to '{1}'",
                    op_type.TypeDescription(), target_type.TypeDescription());
  return llvm::make_error<DILDiagnosticError>(m_expr, std::move(errMsg),
                                              node.GetLocation());
}

} // namespace lldb_private::dil
