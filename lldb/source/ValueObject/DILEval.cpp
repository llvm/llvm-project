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
  lldb::VariableListSP variable_list =
      symbol_context.comp_unit->GetVariableList(true);

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

  if (field_obj && field_obj->GetName() == node->GetFieldName()) {
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

  // Check to see if 'base' has a synthetic value; if so, try using that.
  uint64_t child_idx = node->GetIndex();
  if (lldb::ValueObjectSP synthetic = base->GetSyntheticValue()) {
    llvm::Expected<uint32_t> num_children =
        synthetic->GetNumChildren(child_idx + 1);
    if (!num_children)
      return llvm::make_error<DILDiagnosticError>(
          m_expr, toString(num_children.takeError()), node->GetLocation());
    if (child_idx >= *num_children) {
      std::string message = llvm::formatv(
          "array index {0} is not valid for \"({1}) {2}\"", child_idx,
          base->GetTypeName().AsCString("<invalid type>"),
          base->GetName().AsCString());
      return llvm::make_error<DILDiagnosticError>(m_expr, message,
                                                  node->GetLocation());
    }
    if (lldb::ValueObjectSP child_valobj_sp =
            synthetic->GetChildAtIndex(child_idx))
      return child_valobj_sp;
  }

  auto base_type = base->GetCompilerType().GetNonReferenceType();
  if (!base_type.IsPointerType() && !base_type.IsArrayType())
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "subscripted value is not an array or pointer",
        node->GetLocation());
  if (base_type.IsPointerToVoid())
    return llvm::make_error<DILDiagnosticError>(
        m_expr, "subscript of pointer to incomplete type 'void'",
        node->GetLocation());

  if (base_type.IsArrayType()) {
    if (lldb::ValueObjectSP child_valobj_sp = base->GetChildAtIndex(child_idx))
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

static CompilerType GetBasicTypeFromCU(std::shared_ptr<StackFrame> ctx,
                                       lldb::BasicType basic_type) {
  SymbolContext symbol_context =
      ctx->GetSymbolContext(lldb::eSymbolContextCompUnit);
  auto language = symbol_context.comp_unit->GetLanguage();

  symbol_context = ctx->GetSymbolContext(lldb::eSymbolContextModule);
  auto type_system =
      symbol_context.module_sp->GetTypeSystemForLanguage(language);

  if (type_system)
    if (auto compiler_type = type_system.get()->GetBasicTypeFromAST(basic_type))
      return compiler_type;

  CompilerType empty_type;
  return empty_type;
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const ScalarLiteralNode *node) {
  CompilerType result_type =
      GetBasicTypeFromCU(m_exe_ctx_scope, node->GetType());
  Scalar value = node->GetValue();

  // Scalar later could be float or bool
  if (result_type.IsInteger() || result_type.IsNullPtrType() ||
      result_type.IsPointerType()) {
    llvm::APInt val = value.GetAPSInt();
    return ValueObject::CreateValueObjectFromAPInt(m_target, val, result_type,
                                                   "result");
  }

  return lldb::ValueObjectSP();
}

static CompilerType GetBasicType(lldb::TypeSystemSP type_system,
                                 lldb::BasicType basic_type) {
  if (type_system)
    if (auto compiler_type = type_system.get()->GetBasicTypeFromAST(basic_type))
      return compiler_type;

  CompilerType empty_type;
  return empty_type;
}

static CompilerType DoIntegralPromotion(CompilerType from,
                                        std::shared_ptr<StackFrame> ctx) {
  if (!from.IsInteger() && !from.IsUnscopedEnumerationType())
    return from;

  if (!from.IsPromotableIntegerType())
    return from;

  if (from.IsUnscopedEnumerationType())
    return DoIntegralPromotion(from.GetEnumerationIntegerType(), ctx);
  lldb::BasicType builtin_type =
      from.GetCanonicalType().GetBasicTypeEnumeration();
  lldb::TypeSystemSP type_system = from.GetTypeSystem().GetSharedPointer();

  uint64_t from_size = 0;
  if (builtin_type == lldb::eBasicTypeWChar ||
      builtin_type == lldb::eBasicTypeSignedWChar ||
      builtin_type == lldb::eBasicTypeUnsignedWChar ||
      builtin_type == lldb::eBasicTypeChar16 ||
      builtin_type == lldb::eBasicTypeChar32) {
    // Find the type that can hold the entire range of values for our type.
    bool is_signed = from.IsSigned();
    if (auto temp = from.GetByteSize(ctx.get()))
      from_size = *temp;

    CompilerType promote_types[] = {
        GetBasicType(type_system, lldb::eBasicTypeInt),
        GetBasicType(type_system, lldb::eBasicTypeUnsignedInt),
        GetBasicType(type_system, lldb::eBasicTypeLong),
        GetBasicType(type_system, lldb::eBasicTypeUnsignedLong),
        GetBasicType(type_system, lldb::eBasicTypeLongLong),
        GetBasicType(type_system, lldb::eBasicTypeUnsignedLongLong),
    };
    for (auto &type : promote_types) {
      uint64_t byte_size = 0;
      if (auto temp = type.GetByteSize(ctx.get()))
        byte_size = *temp;
      if (from_size < byte_size ||
          (from_size == byte_size &&
           is_signed == (bool)(type.GetTypeInfo() & lldb::eTypeIsSigned))) {
        return type;
      }
    }

    llvm_unreachable("char type should fit into long long");
  }

  // Here we can promote only to "int" or "unsigned int".
  CompilerType int_type = GetBasicType(type_system, lldb::eBasicTypeInt);
  uint64_t int_byte_size = 0;
  if (auto temp = int_type.GetByteSize(ctx.get()))
    int_byte_size = *temp;

  // Signed integer types can be safely promoted to "int".
  if (from.IsSigned()) {
    return int_type;
  }
  // Unsigned integer types are promoted to "unsigned int" if "int" cannot hold
  // their entire value range.
  return (from_size == int_byte_size)
             ? GetBasicType(type_system, lldb::eBasicTypeUnsignedInt)
             : int_type;
}

static lldb::ValueObjectSP UnaryConversion(lldb::ValueObjectSP valobj,
                                           std::shared_ptr<StackFrame> ctx) {
  // Perform usual conversions for unary operators.
  CompilerType in_type = valobj->GetCompilerType();
  CompilerType result_type;

  if (valobj->GetCompilerType().IsInteger() ||
      valobj->GetCompilerType().IsUnscopedEnumerationType()) {
    CompilerType promoted_type =
        DoIntegralPromotion(valobj->GetCompilerType(), ctx);
    if (!promoted_type.CompareTypes(valobj->GetCompilerType()))
      return valobj->CastToBasicType(promoted_type);
  }

  return valobj;
}

static size_t ConversionRank(CompilerType type) {
  // Get integer conversion rank
  // https://eel.is/c++draft/conv.rank
  switch (type.GetCanonicalType().GetBasicTypeEnumeration()) {
  case lldb::eBasicTypeBool:
    return 1;
  case lldb::eBasicTypeChar:
  case lldb::eBasicTypeSignedChar:
  case lldb::eBasicTypeUnsignedChar:
    return 2;
  case lldb::eBasicTypeShort:
  case lldb::eBasicTypeUnsignedShort:
    return 3;
  case lldb::eBasicTypeInt:
  case lldb::eBasicTypeUnsignedInt:
    return 4;
  case lldb::eBasicTypeLong:
  case lldb::eBasicTypeUnsignedLong:
    return 5;
  case lldb::eBasicTypeLongLong:
  case lldb::eBasicTypeUnsignedLongLong:
    return 6;

    // The ranks of char16_t, char32_t, and wchar_t are equal to the
    // ranks of their underlying types.
  case lldb::eBasicTypeWChar:
  case lldb::eBasicTypeSignedWChar:
  case lldb::eBasicTypeUnsignedWChar:
    return 3;
  case lldb::eBasicTypeChar16:
    return 3;
  case lldb::eBasicTypeChar32:
    return 4;

  default:
    break;
  }
  return 0;
}

static lldb::BasicType BasicTypeToUnsigned(lldb::BasicType basic_type) {
  switch (basic_type) {
  case lldb::eBasicTypeInt:
    return lldb::eBasicTypeUnsignedInt;
  case lldb::eBasicTypeLong:
    return lldb::eBasicTypeUnsignedLong;
  case lldb::eBasicTypeLongLong:
    return lldb::eBasicTypeUnsignedLongLong;
  default:
    return basic_type;
  }
}

static void PerformIntegerConversions(std::shared_ptr<StackFrame> ctx,
                                      lldb::ValueObjectSP &lhs,
                                      lldb::ValueObjectSP &rhs,
                                      bool convert_lhs, bool convert_rhs) {
  CompilerType l_type = lhs->GetCompilerType();
  CompilerType r_type = rhs->GetCompilerType();
  if (r_type.IsSigned() && !l_type.IsSigned()) {
    uint64_t l_size = 0;
    uint64_t r_size = 0;
    if (auto temp = l_type.GetByteSize(ctx.get()))
      l_size = *temp;
    ;
    if (auto temp = r_type.GetByteSize(ctx.get()))
      r_size = *temp;
    if (l_size <= r_size) {
      if (r_size == l_size) {
        lldb::TypeSystemSP type_system =
            l_type.GetTypeSystem().GetSharedPointer();
        auto r_type_unsigned = GetBasicType(
            type_system,
            BasicTypeToUnsigned(
                r_type.GetCanonicalType().GetBasicTypeEnumeration()));
        if (convert_rhs)
          rhs = rhs->CastToBasicType(r_type_unsigned);
      }
    }
  }
  if (convert_lhs)
    lhs = lhs->CastToBasicType(rhs->GetCompilerType());
}

static CompilerType ArithmeticConversions(lldb::ValueObjectSP &lhs,
                                          lldb::ValueObjectSP &rhs,
                                          std::shared_ptr<StackFrame> ctx) {
  // Apply unary conversion (e.g. intergal promotion) for both operands.
  lhs = UnaryConversion(lhs, ctx);
  rhs = UnaryConversion(rhs, ctx);

  CompilerType lhs_type = lhs->GetCompilerType();
  CompilerType rhs_type = rhs->GetCompilerType();

  if (lhs_type.CompareTypes(rhs_type))
    return lhs_type;

  // If either of the operands is not arithmetic (e.g. pointer), we're done.
  if (!lhs_type.IsScalarType() || !rhs_type.IsScalarType()) {
    CompilerType bad_type;
    return bad_type;
  }

  // Removed floating point conversions
  if (lhs_type.IsFloat() || rhs_type.IsFloat()) {
    CompilerType bad_type;
    return bad_type;
  }

  if (lhs_type.IsInteger() && rhs_type.IsInteger()) {
    using Rank = std::tuple<size_t, bool>;
    Rank l_rank = {ConversionRank(lhs_type), !lhs_type.IsSigned()};
    Rank r_rank = {ConversionRank(rhs_type), !rhs_type.IsSigned()};

    if (l_rank < r_rank) {
      PerformIntegerConversions(ctx, lhs, rhs, true, true);
    } else if (l_rank > r_rank) {
      PerformIntegerConversions(ctx, rhs, lhs, true, true);
    }
  }

  return rhs_type;
}

llvm::Error Interpreter::PrepareBinaryAddition(lldb::ValueObjectSP &lhs,
                                               lldb::ValueObjectSP &rhs,
                                               uint32_t location) {
  // Operation '+' works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  {integer,unscoped_enum} <-> pointer
  //  pointer <-> {integer,unscoped_enum}
  auto orig_lhs_type = lhs->GetCompilerType();
  auto orig_rhs_type = rhs->GetCompilerType();
  auto result_type = ArithmeticConversions(lhs, rhs, m_exe_ctx_scope);

  if (result_type.IsScalarType())
    return llvm::Error::success();

  // Removed pointer arithmetics
  return llvm::make_error<DILDiagnosticError>(m_expr, "unimplemented",
                                              location);
}

static lldb::ValueObjectSP EvaluateArithmeticOpInteger(lldb::TargetSP target,
                                                       BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs,
                                                       CompilerType rtype) {
  assert(lhs->GetCompilerType().IsInteger() &&
         rhs->GetCompilerType().IsInteger() &&
         "invalid ast: both operands must be integers");

  auto wrap = [target, rtype](auto value) {
    return ValueObject::CreateValueObjectFromAPInt(target, value, rtype,
                                                   "result");
  };

  llvm::Expected<llvm::APSInt> l_value = lhs->GetValueAsAPSInt();
  llvm::Expected<llvm::APSInt> r_value = rhs->GetValueAsAPSInt();

  if (l_value && r_value) {
    llvm::APSInt l = *l_value;
    llvm::APSInt r = *r_value;

    switch (kind) {
    case BinaryOpKind::Add:
      return wrap(l + r);

    default:
      assert(false && "invalid ast: invalid arithmetic operation");
      return lldb::ValueObjectSP();
    }
  } else {
    return lldb::ValueObjectSP();
  }
}

static lldb::ValueObjectSP EvaluateArithmeticOp(lldb::TargetSP target,
                                                BinaryOpKind kind,
                                                lldb::ValueObjectSP lhs,
                                                lldb::ValueObjectSP rhs,
                                                CompilerType rtype) {
  assert((rtype.IsInteger() || rtype.IsFloat()) &&
         "invalid ast: result type must either integer or floating point");

  // Evaluate arithmetic operation for two integral values.
  if (rtype.IsInteger()) {
    return EvaluateArithmeticOpInteger(target, kind, lhs, rhs, rtype);
  }

  // Removed floating point arithmetics

  return lldb::ValueObjectSP();
}

llvm::Expected<lldb::ValueObjectSP> Interpreter::EvaluateBinaryAddition(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs, uint32_t location) {
  // Addition of two arithmetic types.
  if (lhs->GetCompilerType().IsScalarType() &&
      rhs->GetCompilerType().IsScalarType()) {
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Add, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }

  // Removed pointer arithmetics
  return llvm::make_error<DILDiagnosticError>(m_expr, "unimplemented",
                                              location);
}

lldb::ValueObjectSP
Interpreter::ConvertValueObjectToTypeSystem(lldb::ValueObjectSP valobj,
                                            lldb::TypeSystemSP type_system) {
  auto apsint = valobj->GetValueAsAPSInt();
  if (apsint) {
    llvm::APInt value = *apsint;
    if (type_system) {
      lldb::BasicType basic_type = valobj->GetCompilerType()
                                       .GetCanonicalType()
                                       .GetBasicTypeEnumeration();
      if (auto compiler_type =
              type_system.get()->GetBasicTypeFromAST(basic_type)) {
        valobj->GetValue().SetCompilerType(compiler_type);
        return ValueObject::CreateValueObjectFromAPInt(m_target, value,
                                                       compiler_type, "result");
      }
    }
  }

  return lldb::ValueObjectSP();
}

llvm::Expected<lldb::ValueObjectSP>
Interpreter::Visit(const BinaryOpNode *node) {
  auto lhs_or_err = Evaluate(node->GetLHS());
  if (!lhs_or_err)
    return lhs_or_err;
  lldb::ValueObjectSP lhs = *lhs_or_err;
  auto rhs_or_err = Evaluate(node->GetRHS());
  if (!rhs_or_err)
    return rhs_or_err;
  lldb::ValueObjectSP rhs = *rhs_or_err;

  lldb::TypeSystemSP lhs_system =
      lhs->GetCompilerType().GetTypeSystem().GetSharedPointer();
  lldb::TypeSystemSP rhs_system =
      rhs->GetCompilerType().GetTypeSystem().GetSharedPointer();

  // Is this a correct way to check if type systems are the same?
  if (lhs_system != rhs_system) {
    // If one of the nodes is a scalar const, convert it to the
    // type system of another one
    if (node->GetLHS()->GetKind() == NodeKind::eScalarLiteralNode)
      lhs = ConvertValueObjectToTypeSystem(lhs, rhs_system);
    else if (node->GetRHS()->GetKind() == NodeKind::eScalarLiteralNode)
      rhs = ConvertValueObjectToTypeSystem(rhs, lhs_system);
    else
      return llvm::make_error<DILDiagnosticError>(
          m_expr, "incompatible type systems", node->GetLocation());
  }

  switch (node->GetKind()) {
  case BinaryOpKind::Add:
    if (auto err = PrepareBinaryAddition(lhs, rhs, node->GetLocation()))
      return err;
    return EvaluateBinaryAddition(lhs, rhs, node->GetLocation());

    // Other ops

  default:
    break;
  }

  return llvm::make_error<DILDiagnosticError>(m_expr, "unimplemented",
                                              node->GetLocation());
}

} // namespace lldb_private::dil
