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
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"

namespace lldb_private {

namespace dil {

template <typename T>
bool Compare(BinaryOpKind kind, const T& l, const T& r) {
  switch (kind) {
    case BinaryOpKind::EQ:
      return l == r;
    case BinaryOpKind::NE:
      return l != r;
    case BinaryOpKind::LT:
      return l < r;
    case BinaryOpKind::LE:
      return l <= r;
    case BinaryOpKind::GT:
      return l > r;
    case BinaryOpKind::GE:
      return l >= r;

    default:
      assert(false && "invalid ast: invalid comparison operation");
      return false;
  }
}

static uint64_t GetUInt64(lldb::ValueObjectSP value_sp) {
  // GetValueAsUnsigned performs overflow according to the underlying type. Fo\
r
  // example, if the underlying type is `int32_t` and the value is `-1`,
  // GetValueAsUnsigned will return 4294967295.
  return value_sp->GetCompilerType().IsSigned()
      ? value_sp->GetValueAsSigned(0)
      : value_sp->GetValueAsUnsigned(0);
}

static lldb::ValueObjectSP EvaluateArithmeticOpInteger(lldb::TargetSP target,
                                                       BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs,
                                                       CompilerType rtype)
{
  assert(lhs->GetCompilerType().IsInteger() &&
         rhs->GetCompilerType().IsInteger() &&
         "invalid ast: both operands must be integers");
  assert((kind == BinaryOpKind::Shl || kind == BinaryOpKind::Shr ||
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must have the same type");

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
      case BinaryOpKind::Sub:
        return wrap(l - r);
      case BinaryOpKind::Div:
        return wrap(l / r);
      case BinaryOpKind::Mul:
        return wrap(l * r);
      case BinaryOpKind::Rem:
        return wrap(l % r);
      case BinaryOpKind::And:
        return wrap(l & r);
      case BinaryOpKind::Or:
        return wrap(l | r);
      case BinaryOpKind::Xor:
        return wrap(l ^ r);
      case BinaryOpKind::Shl:
        return wrap(l.shl(r));
      case BinaryOpKind::Shr:
        // Apply arithmetic shift on signed values and logical shift operation
        // on unsigned values.
        return wrap(l.isSigned() ? l.ashr(r) : l.lshr(r));

      default:
        assert(false && "invalid ast: invalid arithmetic operation");
        return lldb::ValueObjectSP();
    }
  } else {
    return lldb::ValueObjectSP();
  }
}

static lldb::ValueObjectSP EvaluateArithmeticOpFloat(lldb::TargetSP target,
                                                     BinaryOpKind kind,
                                                     lldb::ValueObjectSP lhs,
                                                     lldb::ValueObjectSP rhs,
                                                     CompilerType rtype) {
  assert((lhs->GetCompilerType().IsFloat() &&
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must be floats and have the same type");

  auto wrap = [target, rtype](auto value) {
    return ValueObject::CreateValueObjectFromAPFloat(target, value, rtype, "result");
  };

  auto lval_or_err = lhs->GetValueAsAPFloat();
  auto rval_or_err = rhs->GetValueAsAPFloat();
  if (lval_or_err && rval_or_err) {
    llvm::APFloat l = *lval_or_err;
    llvm::APFloat r = *rval_or_err;

    switch (kind) {
      case BinaryOpKind::Add:
        return wrap(l + r);
      case BinaryOpKind::Sub:
        return wrap(l - r);
      case BinaryOpKind::Div:
        return wrap(l / r);
      case BinaryOpKind::Mul:
        return wrap(l * r);

      default:
        assert(false && "invalid ast: invalid arithmetic operation");
        return lldb::ValueObjectSP();
    }
  }
  return lldb::ValueObjectSP();
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

  // Evaluate arithmetic operation for two floating point values.
  if (rtype.IsFloat()) {
    return EvaluateArithmeticOpFloat(target, kind, lhs, rhs, rtype);
  }

  return lldb::ValueObjectSP();
}

static bool IsInvalidDivisionByMinusOne(lldb::ValueObjectSP lhs_sp,
                                        lldb::ValueObjectSP rhs_sp)
{
  assert(lhs_sp->GetCompilerType().IsInteger() &&
         rhs_sp->GetCompilerType().IsInteger() && "operands should be integers");

  // The result type should be signed integer.
  auto basic_type =
      rhs_sp->GetCompilerType().GetCanonicalType().GetBasicTypeEnumeration();
  if (basic_type != lldb::eBasicTypeInt && basic_type != lldb::eBasicTypeLong &&
      basic_type != lldb::eBasicTypeLongLong) {
    return false;
  }

  // The RHS should be equal to -1.
  if (rhs_sp->GetValueAsSigned(0) != -1) {
    return false;
  }

  // The LHS should be equal to the minimum value the result type can hold.
  uint64_t byte_size = 0;
  if (auto temp = rhs_sp->GetCompilerType().GetByteSize(rhs_sp->GetTargetSP().get()))
    byte_size = temp.value();
  auto bit_size = byte_size * CHAR_BIT;
  return lhs_sp->GetValueAsSigned(0) + (1LLU << (bit_size - 1)) == 0;
}

lldb::ValueObjectSP DILInterpreter::EvaluateMemberOf(lldb::ValueObjectSP value,
                                            const std::vector<uint32_t>& path,
                                            bool use_synthetic,
                                            bool is_dynamic) {
  // The given `value` can be a pointer, but GetChildAtIndex works for pointers
  // too, so we don't need to dereference it explicitely. This also avoid having
  // an "ephemeral" parent lldb::ValueObjectSP, representing the dereferenced
  // value.
  lldb::ValueObjectSP member_val_sp = value;
  // Objects from the standard library (e.g. containers, smart pointers) have
  // synthetic children (e.g. stored values for containers, wrapped object for
  // smart pointers), but the indexes in `member_index()` array refer to the
  // actual type members.
  lldb::DynamicValueType use_dynamic = (!is_dynamic)
                                       ? lldb::eNoDynamicValues
                                       : lldb::eDynamicDontRunTarget;
  for (uint32_t idx : path) {
    // Force static value, otherwise we can end up with the "real" type.
    member_val_sp = member_val_sp->GetChildAtIndex(idx, /*can_create*/ true);
  }
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

  // If value is a reference, derefernce it to get to the underlying type. All
  // operations on a reference should be actually operations on the referent.
  Status error;
  if (member_val_sp->GetCompilerType().IsReferenceType()) {
    member_val_sp = member_val_sp->Dereference(error);
    assert(member_val_sp && error.Success() && "unable to dereference member val");
  }

  return member_val_sp;
}

void SetUbStatus(Status& error, ErrorCode code) {
  llvm::StringRef err_str;
  switch ((int) code) {
    case (int) ErrorCode::kUBDivisionByZero:
      err_str ="Error: Division by zero detected.";
      break;
    case (int) ErrorCode::kUBDivisionByMinusOne:
      // If "a / b" isn't representable in its result type, then results of
      // "a / b" and "a % b" are undefined behaviour. This happens when "a"
      // is equal to the minimum value of the result type and "b" is equal
      // to -1.
      err_str ="Error: Invalid division by negative one  detected.";
      break;
    case (int) ErrorCode::kUBInvalidCast:
      err_str ="Error: Invalid type cast detected.";
      break;
    case (int) ErrorCode::kUBInvalidShift:
      err_str ="Error: Invalid shift detected.";
      break;
    case (int) ErrorCode::kUBNullPtrArithmetic:
      err_str ="Error: Attempt to perform arithmetic with null ptr  detected.";
      break;
    case (int) ErrorCode::kUBInvalidPtrDiff:
      err_str ="Error: Attempt to perform invalid ptr arithmetic detected.";
      break;
    default:
      err_str ="Error: Unknown undefined behavior error.";
      break;
  }
  error = Status(err_str.str());
}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm)
    : m_target(std::move(target)), m_sm(std::move(sm))
{
  m_default_dynamic = lldb::eNoDynamicValues;
}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm,
                               lldb::DynamicValueType use_dynamic)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_default_dynamic(use_dynamic) {}

DILInterpreter::DILInterpreter(lldb::TargetSP target,
                               std::shared_ptr<DILSourceManager> sm,
                               lldb::ValueObjectSP scope)
    : m_target(std::move(target)), m_sm(std::move(sm)),
      m_scope(std::move(scope))
{
  m_default_dynamic = lldb::eNoDynamicValues;
  // If `m_scope` is a reference, dereference it. All operations on a reference
  // should be operations on the referent.
  if (m_scope->GetCompilerType().IsValid() &&
      m_scope->GetCompilerType().IsReferenceType()) {
    Status error;
    m_scope = m_scope->Dereference(error);
  }
}

void DILInterpreter::SetContextVars(
    std::unordered_map<std::string, lldb::ValueObjectSP> context_vars) {
  m_context_vars = std::move(context_vars);
}

lldb::ValueObjectSP DILInterpreter::DILEval(const DILASTNode* tree,
                                            lldb::TargetSP target_sp,
                                            Status& error)
{
  m_error.Clear();
  // Evaluate an AST.
  DILEvalNode(tree);
  // Set the error.
  error = std::move(m_error);
  // Return the computed result. If there was an error, it will be invalid.
  return m_result;
}

lldb::ValueObjectSP DILInterpreter::DILEvalNode(const DILASTNode* node,
                                             FlowAnalysis* flow) {
  // Set up the evaluation context for the current node.
  m_flow_analysis_chain.push_back(flow);
  // Traverse an AST pointed by the `node`.
  node->Accept(this);
  // Cleanup the context.
  m_flow_analysis_chain.pop_back();
  // Return the computed value for convenience. The caller is responsible for
  // checking if an error occured during the evaluation.
  return m_result;
}

void DILInterpreter::SetError(ErrorCode code, std::string error,
                              uint32_t loc) {
  assert(m_error.Success() && "interpreter can error only once");
  m_error = Status((uint32_t) code, lldb::eErrorTypeGeneric,
                   FormatDiagnostics(*m_sm, error, loc));
}

void DILInterpreter::Visit(const ErrorNode* node) {
  // The AST is not valid.
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const ScalarLiteralNode* node) {
  CompilerType result_type = node->result_type();
  Scalar value = node->GetValue();
  if (result_type.IsBoolean()) {
    unsigned int int_val = value.UInt();
    bool b_val = false;
    if (int_val == 1)
      b_val = true;
    m_result = ValueObject::CreateValueObjectFromBool(m_target, b_val,
                                                      "result");
  } else if (result_type.IsFloat()) {
    llvm::APFloat val = value.GetAPFloat();
    m_result = ValueObject::CreateValueObjectFromAPFloat(m_target, val,
                                                     result_type,
                                                     "result");
  } else if (result_type.IsInteger() ||
             result_type.IsNullPtrType() ||
             result_type.IsPointerType()) {
    llvm::APInt val = value.GetAPSInt();
    m_result = ValueObject::CreateValueObjectFromAPInt(m_target, val,
                                                       result_type,
                                                       "result");
  } else {
    m_result =  lldb::ValueObjectSP();
  }
}

void DILInterpreter::Visit(const StringLiteralNode* node) {
  CompilerType result_type = node->result_type();
  std::string val = node->GetValue();
  ExecutionContext exe_ctx(m_target.get(), false);
  uint64_t byte_size = 0;
  if (auto temp = result_type.GetByteSize(m_target.get()))
    byte_size = temp.value();
  lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
      reinterpret_cast<const void*>(val.data()), byte_size,
      exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
  m_result = ValueObject::CreateValueObjectFromData("result", *data_sp,
                                                    exe_ctx,
                                                    result_type);
}

void DILInterpreter::Visit(const IdentifierNode* node) {
  auto identifier = static_cast<const IdentifierInfo&>(node->info());

  lldb::ValueObjectSP val;
  lldb::TargetSP target_sp;
  Status error;
  switch (identifier.GetKind()) {
    using Kind = IdentifierInfo::Kind;
    case Kind::eValue:
      val = identifier.GetValue();
      target_sp = val->GetTargetSP();
      assert(target_sp && target_sp->IsValid()
             && "invalid ast: invalid identifier value");
      break;

    case Kind::eContextArg:
      assert(node->is_context_var() && "invalid ast: context var expected");
      val = ResolveContextVar(node->name());
      target_sp = val->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUndeclaredIdentifier,
            llvm::formatv("use of undeclared identifier '{0}'", node->name()),
            node->GetLocation());
        m_result = lldb::ValueObjectSP();
        return;
      }
      if (!node->GetDereferencedResultType().CompareTypes(val->GetCompilerType())) {
        SetError(ErrorCode::kInvalidOperandType,
                 llvm::formatv("unexpected type of context variable '{0}' "
                               "(expected {1}, got {2})",
                               node->name(),
                               node->GetDereferencedResultType().TypeDescription(),
                               val->GetCompilerType().TypeDescription()),
                 node->GetLocation());
        m_result = lldb::ValueObjectSP();
        return;
      }
      break;

    case Kind::eMemberPath:
      target_sp = m_scope->GetTargetSP();
      if (!target_sp || !target_sp->IsValid()) {
        SetError(
            ErrorCode::kUnknown,
            llvm::formatv(
                "unable to resolve '{0}', evaluation requires a value context",
                node->name()),
            node->GetLocation());
        m_result = lldb::ValueObjectSP();
        return;
      }
      val = EvaluateMemberOf(m_scope, identifier.GetPath(), false, false);
      break;

    default:
      assert(false && "invalid ast: invalid identifier kind");
  }

  target_sp = val->GetTargetSP();
  assert(target_sp && target_sp->IsValid() &&
         "identifier doesn't resolve to a valid value");

  m_result = val;
}

void DILInterpreter::Visit(const SizeOfNode* node) {
  auto operand = node->operand();

  uint64_t deref_byte_size = 0;
  uint64_t other_byte_size = 0;
  if (auto temp = operand.GetNonReferenceType().GetByteSize(m_target.get()))
    deref_byte_size = temp.value();
  if (auto temp = operand.GetByteSize(m_target.get()))
    other_byte_size = temp.value();
  // For reference type (int&) we need to look at the referenced type.
  size_t size = operand.IsReferenceType()
                ? deref_byte_size
                : other_byte_size;
  CompilerType type = node->GetDereferencedResultType();
  ExecutionContext exe_ctx(m_target.get(), false);
  uint64_t byte_size = 0;
  if (auto temp = type.GetByteSize(m_target.get()))
    byte_size = temp.value();
  lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
      reinterpret_cast<const void*>(&size), byte_size,
      exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
  m_result = ValueObject::CreateValueObjectFromData("result", *data_sp, exe_ctx,
                                                    type);
}

void DILInterpreter::Visit(const BuiltinFunctionCallNode* node) {
  if (node->name() == "__log2") {
    assert(node->arguments().size() == 1 &&
           "invalid ast: expected exactly one argument to `__log2`");
    // Get the first (and the only) argument and evaluate it.
    auto& arg = node->arguments()[0];
    lldb::ValueObjectSP val = DILEvalNode(arg.get());
    if (!val) {
      return;
    }
    assert(val->GetCompilerType().IsInteger() &&
           "invalid ast: argument to __log2 must be an interger");

    // Use Log2_32 to match the behaviour of Visual Studio debugger.
    uint32_t ret =
        llvm::Log2_32(static_cast<uint32_t>(val->GetValueAsUnsigned(0)));
    CompilerType target_type;
    for (auto type_system_sp : m_target->GetScratchTypeSystems())
      if (auto compiler_type =
          type_system_sp->GetBasicTypeFromAST(lldb::eBasicTypeUnsignedInt)) {
        target_type = compiler_type;
        break;
      }

    ExecutionContext exe_ctx(m_target.get(), false);
    uint64_t byte_size = 0;
    if (auto temp = target_type.GetByteSize(m_target.get()))
      byte_size = temp.value();
    lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
        reinterpret_cast<const void*>(&ret), byte_size,
        exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
    m_result =
        ValueObject::CreateValueObjectFromData("result", *data_sp,
                                               exe_ctx, target_type);
    return;
  }

  if (node->name() == "__findnonnull") {
    assert(node->arguments().size() == 2 &&
           "invalid ast: expected exactly two arguments to `__findnonnull`");

    auto& arg1 = node->arguments()[0];
    lldb::ValueObjectSP val1_sp = DILEvalNode(arg1.get());
    if (!val1_sp) {
      return;
    }

    // Resolve data address for the first argument.
    uint64_t addr;

    if (val1_sp->GetCompilerType().IsPointerType()) {
      addr = val1_sp->GetValueAsUnsigned(0);
    } else if (val1_sp->GetCompilerType().IsArrayType()) {
      addr = val1_sp->GetLoadAddress();
    } else {
      SetError(ErrorCode::kInvalidOperandType,
               llvm::formatv("no known conversion from '{0}' to 'T*' for 1st "
                             "argument of __findnonnull()",
                             val1_sp->GetCompilerType().GetTypeName()),
               arg1->GetLocation());
      return;
    }

    auto& arg2 = node->arguments()[1];
    lldb::ValueObjectSP val2_sp = DILEvalNode(arg2.get());
    if (!val2_sp) {
      return;
    }
    int64_t size = val2_sp->GetValueAsSigned(0);

    if (size < 0 || size > 100000000) {
      SetError(ErrorCode::kInvalidOperandType,
               llvm::formatv(
                   "passing in a buffer size ('{0}') that is negative or in "
                   "excess of 100 million to __findnonnull() is not allowed.",
                   size),
               arg2->GetLocation());
      return;
    }

    lldb::ProcessSP process = m_target->GetProcessSP();
    size_t ptr_size = m_target->GetArchitecture().GetAddressByteSize();

    uint64_t memory = 0;
    Status error;

    CompilerType target_type;
    for (auto type_system_sp : m_target->GetScratchTypeSystems())
      if (auto compiler_type =
          type_system_sp->GetBasicTypeFromAST(lldb::eBasicTypeInt)) {
        target_type = compiler_type;
        break;
      }
    ExecutionContext exe_ctx(m_target.get(), false);
    uint64_t byte_size = 0;
    if (auto temp = target_type.GetByteSize(m_target.get()))
      byte_size = temp.value();

    for (int i = 0; i < size; ++i) {
      size_t read =
          process->ReadMemory(addr + i * ptr_size, &memory, ptr_size, error);

      if (error.Fail() || read != ptr_size) {
        SetError(ErrorCode::kUnknown,
                 llvm::formatv("error calling __findnonnull(): {0}",
                               error.AsCString() ? error.AsCString()
                                                 : "cannot read memory"),
                 node->GetLocation());
        return;
      }

      if (memory != 0) {
        lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
            reinterpret_cast<const void*>(&i), byte_size,
            exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
        m_result = ValueObject::CreateValueObjectFromData("result", *data_sp,
                                                          exe_ctx,
                                                          target_type);
        return;
      }
    }

    int ret = -1;

    lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
        reinterpret_cast<const void*>(&ret), byte_size,
        exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
    m_result = ValueObject::CreateValueObjectFromData("result", *data_sp,
                                                      exe_ctx,
                                                      target_type);
    return;
  }

  assert(false && "invalid ast: unknown builtin function");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const CStyleCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->operand());
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType()) {
    Status error;
    rhs = rhs->Dereference(error);
  }

  switch (node->cast_kind()) {
    case CStyleCastKind::eEnumeration: {
      assert(type.IsEnumerationType() &&
             "invalid ast: target type should be an enumeration.");

      if (rhs->GetCompilerType().IsFloat()) {
        m_result = rhs->CastToEnumType(type);
      } else if (rhs->GetCompilerType().IsInteger() ||
                 rhs->GetCompilerType().IsEnumerationType()) {
        m_result = rhs->CastToEnumType(type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to enumeration type");
      }
      return;
    }
    case CStyleCastKind::eNullptr: {
      assert(
          (type.GetCanonicalType().GetBasicTypeEnumeration() ==
           lldb::eBasicTypeNullPtr)
          && "invalid ast: target type should be a nullptr_t.");
      m_result = ValueObject::CreateValueObjectFromNullptr(m_target, type, "result");
      return;
    }
    case CStyleCastKind::eReference: {
      lldb::ValueObjectSP rhs_sp(GetDynamicOrSyntheticValue(rhs));
      m_result =
          lldb::ValueObjectSP(rhs_sp->Cast(type.GetNonReferenceType()));
      return;
    }
    case CStyleCastKind::eNone: {

      switch (node->promo_kind()) {

        case TypePromotionCastKind::eArithmetic: {
          assert((type.GetCanonicalType().GetBasicTypeEnumeration() !=
                  lldb::eBasicTypeInvalid) &&
                 "invalid ast: target type should be a basic type.");
          // Pick an appropriate cast.
          if (rhs->GetCompilerType().IsPointerType()
              || rhs->GetCompilerType().IsNullPtrType()) {
            m_result = rhs->CastToBasicType(type);
          } else if (rhs->GetCompilerType().IsScalarType()) {
            m_result = rhs->CastToBasicType(type);
            //m_result = rhs->CastScalarToBasicType(type, m_error);
          } else if (rhs->GetCompilerType().IsEnumerationType()) {
            m_result = rhs->CastToBasicType(type);
          } else {
            assert(false &&
                   "invalid ast: operand is not convertible to arithmetic type");
          }
          return;
        }
        case TypePromotionCastKind::ePointer: {
          assert(type.IsPointerType() &&
                 "invalid ast: target type should be a pointer.");
          uint64_t addr = rhs->GetCompilerType().IsArrayType()
                          ? rhs->GetLoadAddress()
                          : GetUInt64(rhs);
          llvm::StringRef name = "result";
          ExecutionContext exe_ctx(m_target.get(), false);
          m_result =
              ValueObject::CreateValueObjectFromAddress(name, addr, exe_ctx,
                                                        type,
                                                        /* do_deref */ false);
          return;
        }
        case TypePromotionCastKind::eNone:
          return;
      }
    }
  }

  assert(false && "invalid ast: unexpected c-style cast kind");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const CxxStaticCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->operand());
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType()) {
    Status error;
    rhs = rhs->Dereference(error);
  }

  switch (node->cast_kind()) {
    case CxxStaticCastKind::eNoOp: {
      assert(type.CompareTypes(rhs->GetCompilerType()) &&
             "invalid ast: types should be the same");
      lldb::ValueObjectSP rhs_sp(GetDynamicOrSyntheticValue(rhs));
      m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
      return;
    }

    case CxxStaticCastKind::eEnumeration: {
      if (rhs->GetCompilerType().IsFloat()) {
        m_result = rhs->CastToEnumType(type);
      } else if (rhs->GetCompilerType().IsInteger() ||
                 rhs->GetCompilerType().IsEnumerationType()) {
        m_result = rhs->CastToEnumType(type);
      } else {
        assert(false &&
               "invalid ast: operand is not convertible to enumeration type");
      }
      return;
    }

    case CxxStaticCastKind::eNullptr: {
      m_result = ValueObject::CreateValueObjectFromNullptr(m_target, type, "result");
      return;
    }

    case CxxStaticCastKind::eDerivedToBase: {
      llvm::Expected<lldb::ValueObjectSP> result =
          rhs->CastDerivedToBaseType(type, node->idx());
      if (result)
        m_result = *result;
      return;
    }

    case CxxStaticCastKind::eBaseToDerived: {
      llvm::Expected<lldb::ValueObjectSP> result =
          rhs->CastBaseToDerivedType(type, node->offset());
      if (result)
        m_result = *result;
      return;
    }
    case CxxStaticCastKind::eNone: {

      switch (node->promo_kind()) {

        case TypePromotionCastKind::eArithmetic: {
          assert(type.IsScalarType());
          if (rhs->GetCompilerType().IsPointerType()
              || rhs->GetCompilerType().IsNullPtrType()) {
            assert(type.IsBoolean() && "invalid ast: target type should be bool");
            m_result = rhs->CastToBasicType(type);
          } else if (rhs->GetCompilerType().IsScalarType()) {
            //m_result = rhs->CastScalarToBasicType(type, m_error);
            m_result = rhs->CastToBasicType(type);
          } else if (rhs->GetCompilerType().IsEnumerationType()) {
            m_result = rhs->CastToBasicType(type);
          } else {
            assert(false &&
                   "invalid ast: operand is not convertible to arithmetic type");
          }
          return;
        }

        case TypePromotionCastKind::ePointer: {
          assert(type.IsPointerType() &&
                 "invalid ast: target type should be a pointer.");

          uint64_t addr = rhs->GetCompilerType().IsArrayType()
                          ? rhs-> GetLoadAddress()
                          : rhs->GetValueAsUnsigned(0);
          llvm::StringRef name = "result";
          ExecutionContext exe_ctx(m_target.get(), false);
          m_result =
              ValueObject::CreateValueObjectFromAddress(name, addr, exe_ctx,
                                                        type,
                                                        /* do_deref */ false);
          return;
        }
        case TypePromotionCastKind::eNone:
          return;
      }
    }
  }
}

void DILInterpreter::Visit(const CxxReinterpretCastNode* node) {
  // Get the type and the value we need to cast.
  auto type = node->type();
  auto rhs = DILEvalNode(node->operand());
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType()) {
    Status error;
    rhs = rhs->Dereference(error);
  }

  if (type.IsInteger()) {
    if (rhs->GetCompilerType().IsPointerType()
        || rhs->GetCompilerType().IsNullPtrType()) {
      m_result = rhs->CastToBasicType(type);
    } else {
      CompilerType base_type = type.IsTypedefType() ? type.GetTypedefedType()
                               : type;
      CompilerType rhs_base_type = rhs->GetCompilerType().IsTypedefType() ?
                                   rhs->GetCompilerType().GetTypedefedType() :
                                   rhs->GetCompilerType();
      assert(base_type.CompareTypes(rhs_base_type) &&
             "invalid ast: operands should have the same type");
      // Cast value to handle type aliases.
      lldb::ValueObjectSP rhs_sp(GetDynamicOrSyntheticValue(rhs));
      m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
    }
  } else if (type.IsEnumerationType()) {
    CompilerType base_type = type.IsTypedefType() ? type.GetTypedefedType()
                             : type;
    CompilerType rhs_base_type = rhs->GetCompilerType().IsTypedefType() ?
                                 rhs->GetCompilerType().GetTypedefedType() :
                                 rhs->GetCompilerType();
    assert(base_type.CompareTypes(rhs_base_type) &&
           "invalid ast: operands should have the same type");
    // Cast value to handle type aliases.
    lldb::ValueObjectSP rhs_sp(GetDynamicOrSyntheticValue(rhs));
    m_result = lldb::ValueObjectSP(rhs_sp->Cast(type));
  } else if (type.IsPointerType()) {
    assert((rhs->GetCompilerType().IsInteger() ||
            rhs->GetCompilerType().IsEnumerationType() ||
            rhs->GetCompilerType().IsPointerType() ||
            rhs->GetCompilerType().IsArrayType()) &&
           "invalid ast: unexpected operand to reinterpret_cast");
    uint64_t addr = rhs->GetCompilerType().IsArrayType()
                    ? rhs->GetLoadAddress()
                    : rhs->GetValueAsUnsigned(0);
    llvm::StringRef name = "result";
    ExecutionContext exe_ctx(m_target.get(), false);
    m_result =
        ValueObject::CreateValueObjectFromAddress(name, addr, exe_ctx,
                                                  type,
                                                  /* do_deref */ false);
  } else if (type.IsReferenceType()) {
    lldb::ValueObjectSP rhs_sp(GetDynamicOrSyntheticValue(rhs));
    m_result =
        lldb::ValueObjectSP(rhs_sp->Cast(type.GetNonReferenceType()));
  } else {
    assert(false && "invalid ast: unexpected reinterpret_cast kind");
    m_result = lldb::ValueObjectSP();
  }
}

void DILInterpreter::Visit(const MemberOfNode* node) {
  // TODO: Implement address-of elision for member-of:
  //
  //  &(*ptr).foo -> (ptr + foo_offset)
  //  &ptr->foo -> (ptr + foo_offset)
  //
  // This requires calculating the offset of "foo" and generally possible only
  // for members from non-virtual bases.

  Status error;
  lldb::ValueObjectSP base = DILEvalNode(node->base());
  if (!base) {
    return;
  }

  if (node->valobj()) {
    m_result = node->valobj()->GetSP();
  } else {
    if (base->GetCompilerType().IsReferenceType())
      base = base->Dereference(error);
    m_result = EvaluateMemberOf(base, node->member_index(),
                                node->is_synthetic(),
                                node->is_dynamic());
  }
}

void DILInterpreter::Visit(const ArraySubscriptNode* node) {
  auto base = DILEvalNode(node->base());
  if (!base) {
    return;
  }
  auto index = DILEvalNode(node->index());
  if (!index) {
    return;
  }

  // Check to see if either the base or the index are references; if they
  // are, dereference them.
  Status error;
  if (base->GetCompilerType().IsReferenceType())
    base = base->Dereference(error);
  if (index->GetCompilerType().IsReferenceType())
    index = index->Dereference(error);

  // Check to see if 'base' has a synthetic value; if so, try using that.
  if (base->HasSyntheticValue()) {
    lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
    if (synthetic && synthetic != base) {
      uint64_t child_idx = index->GetValueAsUnsigned(0);
      if (static_cast<uint32_t>(child_idx) <
          synthetic->GetNumChildrenIgnoringErrors()) {
        lldb::ValueObjectSP child_valobj_sp =
            synthetic->GetChildAtIndex(child_idx);
        if (child_valobj_sp) {
          m_result = child_valobj_sp;
          return;
        }
      }
    }
  }

  // Verify that the 'index' is not out-of-range for the declared type.
  lldb::ValueObjectSP synthetic = base->GetSyntheticValue();
  lldb::VariableSP base_var = base->GetVariable();
  if (synthetic) {
    uint32_t num_children = synthetic->GetNumChildrenIgnoringErrors();
    if (index->GetValueAsSigned(0) >= num_children) {
      SetError(ErrorCode::kSubscriptOutOfRange,
               llvm::formatv("array index {0} is not valid for \"({1}) {2}\"",
                             index->GetValueAsSigned(0),
                             base->GetTypeName().AsCString("<invalid type>"),
                             base->GetName().AsCString()),
               node->GetLocation());
      m_result = lldb::ValueObjectSP();
      return;
    }
  }

  std::string base_name(base->GetCompilerType().GetTypeName(false).AsCString());
  assert((base->GetCompilerType().IsPointerType() ||
          base_name.find("std::tuple") != std::string::npos)
         && "array subscript: base must be a pointer or a tuple");
  assert(index->GetCompilerType().IsIntegerOrUnscopedEnumerationType() &&
         "array subscript: index must be integer or unscoped enum");

  CompilerType item_type = base->GetCompilerType().GetPointeeType();
  lldb::addr_t base_addr = base->GetValueAsUnsigned(0);

  llvm::StringRef name = "result";
  ExecutionContext exe_ctx(m_target.get(), false);
  // Create a pointer and add the index, i.e. "base + index".
  lldb::ValueObjectSP value =
      PointerAdd(ValueObject::CreateValueObjectFromAddress(
          name, base_addr, exe_ctx, item_type.GetPointerType(),
          /* do_deref */ false),
                 index->GetValueAsSigned(0));

  // If we're in the address-of context, skip the dereference and cancel the
  // pending address-of operation as well.
  if (flow_analysis() && flow_analysis()->AddressOfIsPending()) {
    flow_analysis()->DiscardAddressOf();
    m_result = value;
  } else {
    Status error;
    m_result = value->Dereference(error);
  }
}

void DILInterpreter::Visit(const BinaryOpNode* node) {
  // Short-circuit logical operators.
  if (node->kind() == BinaryOpKind::LAnd || node->kind() == BinaryOpKind::LOr) {
    Status error;
    auto lhs = DILEvalNode(node->lhs());
    if (!lhs) {
      return;
    }
    if (lhs->GetCompilerType().IsReferenceType())
      lhs = lhs->Dereference(error);
    assert(lhs->GetCompilerType().IsContextuallyConvertibleToBool() &&
           "invalid ast: must be convertible to bool");

    // For "&&" break if LHS is "false", for "||" if LHS is "true".
    auto lvalue_or_err = lhs->GetValueAsBool();
    if (!lvalue_or_err)
      return;

    bool lhs_val = *lvalue_or_err;
    bool break_early =
        (node->kind() == BinaryOpKind::LAnd) ? !lhs_val : lhs_val;

    if (break_early) {
      m_result = ValueObject::CreateValueObjectFromBool(m_target, lhs_val, "result");
      return;
    }

    // Breaking early didn't happen, evaluate the RHS and use it as a result.
    auto rhs = DILEvalNode(node->rhs());
    if (!rhs) {
      return;
    }
    if (rhs->GetCompilerType().IsReferenceType())
      rhs = rhs->Dereference(error);
    assert(rhs->GetCompilerType().IsContextuallyConvertibleToBool() &&
           "invalid ast: must be convertible to bool");

    auto rvalue_or_err = rhs->GetValueAsBool();
    if (!rvalue_or_err)
      return;

    m_result = ValueObject::CreateValueObjectFromBool(m_target,
                                                      *rvalue_or_err,
                                                      "result");
    return;
  }

  // All other binary operations require evaluating both operands.
  auto lhs = DILEvalNode(node->lhs());
  if (!lhs) {
    return;
  }
  auto rhs = DILEvalNode(node->rhs());
  if (!rhs) {
    return;
  }

  // For math operations, be sure to dereference the operands.
  if ((node->kind() == BinaryOpKind::Add)
      || (node->kind() == BinaryOpKind::Sub)
      || (node->kind() == BinaryOpKind::Mul)
      || (node->kind() == BinaryOpKind::Div)
      || (node->kind() == BinaryOpKind::Rem)) {
    Status error;
    if (lhs->GetCompilerType().IsReferenceType())
      lhs = lhs->Dereference(error);
    if (rhs->GetCompilerType().IsReferenceType())
      rhs = rhs->Dereference(error);
  }

  switch (node->kind()) {
    case BinaryOpKind::Add:
      m_result = EvaluateBinaryAddition(lhs, rhs);
      return;
    case BinaryOpKind::Sub:
      // The result type of subtraction is required because it holds the
      // correct "ptrdiff_t" type in the case of subtracting two pointers.
      m_result = EvaluateBinarySubtraction(lhs, rhs, node->GetDereferencedResultType());
      return;
    case BinaryOpKind::Mul:
      m_result = EvaluateBinaryMultiplication(lhs, rhs);
      return;
    case BinaryOpKind::Div:
      m_result = EvaluateBinaryDivision(lhs, rhs);
      return;
    case BinaryOpKind::Rem:
      m_result = EvaluateBinaryRemainder(lhs, rhs);
      return;
    case BinaryOpKind::And:
    case BinaryOpKind::Or:
    case BinaryOpKind::Xor:
      m_result = EvaluateBinaryBitwise(node->kind(), lhs, rhs);
      return;
    case BinaryOpKind::Shl:
    case BinaryOpKind::Shr:
      m_result = EvaluateBinaryShift(node->kind(), lhs, rhs);
      return;

    // Comparison operations.
    case BinaryOpKind::EQ:
    case BinaryOpKind::NE:
    case BinaryOpKind::LT:
    case BinaryOpKind::LE:
    case BinaryOpKind::GT:
    case BinaryOpKind::GE:
      m_result = EvaluateComparison(node->kind(), lhs, rhs);
      return;

    case BinaryOpKind::Assign:
      m_result = EvaluateAssignment(lhs, rhs);
      return;

    case BinaryOpKind::AddAssign:
      m_result = EvaluateBinaryAddAssign(lhs, rhs);
      return;
    case BinaryOpKind::SubAssign:
      m_result = EvaluateBinarySubAssign(lhs, rhs);
      return;
    case BinaryOpKind::MulAssign:
      m_result = EvaluateBinaryMulAssign(lhs, rhs);
      return;
    case BinaryOpKind::DivAssign:
      m_result = EvaluateBinaryDivAssign(lhs, rhs);
      return;
    case BinaryOpKind::RemAssign:
      m_result = EvaluateBinaryRemAssign(lhs, rhs);
      return;

    case BinaryOpKind::AndAssign:
    case BinaryOpKind::OrAssign:
    case BinaryOpKind::XorAssign:
      m_result = EvaluateBinaryBitwiseAssign(node->kind(), lhs, rhs);
      return;
    case BinaryOpKind::ShlAssign:
    case BinaryOpKind::ShrAssign:
      m_result = EvaluateBinaryShiftAssign(node->kind(), lhs, rhs,
                                          node->comp_assign_type());
      return;

    default:
      break;
  }

  // Unsupported/invalid operation.
  assert(false && "invalid ast: unexpected binary operator");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const UnaryOpNode* node) {
  FlowAnalysis rhs_flow(
      /* address_of_is_pending */ node->kind() == UnaryOpKind::AddrOf);

  Status error;
  auto rhs = DILEvalNode(node->rhs(), &rhs_flow);
  if (!rhs) {
    return;
  }

  if (rhs->GetCompilerType().IsReferenceType())
    rhs = rhs->Dereference(error);

  switch (node->kind()) {
    case UnaryOpKind::Deref:
      {
        lldb::ValueObjectSP dynamic_rhs =
            rhs->GetDynamicValue(m_default_dynamic);
        if (dynamic_rhs)
          rhs = dynamic_rhs;
      }
      if (rhs->GetCompilerType().IsPointerType())
        m_result = EvaluateDereference(rhs);
      else {
        lldb::ValueObjectSP child_sp = rhs->Dereference(error);
        if (error.Success())
          rhs = child_sp;
        m_result = rhs;
      }
      return;

    case UnaryOpKind::AddrOf:
      // If the address-of operation wasn't cancelled during the evaluation of
      // RHS (e.g. because of the address-of-a-dereference elision), apply it
      // here.
      if (rhs_flow.AddressOfIsPending()) {
        Status error;
        m_result = rhs->AddressOf(error);
      } else {
        m_result = rhs;
      }
      return;
    case UnaryOpKind::Plus:
      m_result = rhs;
      return;
    case UnaryOpKind::Minus:
      m_result = EvaluateUnaryMinus(rhs);
      return;
    case UnaryOpKind::LNot:
      m_result = EvaluateUnaryNegation(rhs);
      return;
    case UnaryOpKind::Not:
      m_result = EvaluateUnaryBitwiseNot(rhs);
      return;
    case UnaryOpKind::PreInc:
      m_result = EvaluateUnaryPrefixIncrement(rhs);
      return;
    case UnaryOpKind::PreDec:
      m_result = EvaluateUnaryPrefixDecrement(rhs);
      return;
    case UnaryOpKind::PostInc:
      // In postfix inc/dec the result is the original value.
      m_result = rhs->Clone(ConstString("cloned-object"));
      EvaluateUnaryPrefixIncrement(rhs);
      return;
    case UnaryOpKind::PostDec:
      // In postfix inc/dec the result is the original value.
      m_result = rhs->Clone(ConstString("cloned-object"));
      EvaluateUnaryPrefixDecrement(rhs);
      return;

    default:
      break;
  }

  // Unsupported/invalid operation.
  assert(false && "invalid ast: unexpected binary operator");
  m_result = lldb::ValueObjectSP();
}

void DILInterpreter::Visit(const TernaryOpNode* node) {
  auto cond = DILEvalNode(node->cond());
  if (!cond) {
    return;
  }
  assert(cond->GetCompilerType().IsContextuallyConvertibleToBool() &&
         "invalid ast: must be convertible to bool");

  // Pass down the flow analysis because the conditional operator is a "flow
  // control" construct -- LHS/RHS might be lvalues and eligible for some
  // optimizations (e.g. "&*" elision).
  auto value_or_err = cond->GetValueAsBool();
  if (value_or_err) {
    if (*value_or_err)
      m_result = DILEvalNode(node->lhs(), flow_analysis());
    else
      m_result = DILEvalNode(node->rhs(), flow_analysis());
  }
}

lldb::ValueObjectSP DILInterpreter::EvaluateComparison(BinaryOpKind kind,
                                                       lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs) {
  // Evaluate arithmetic operation for two integral values.
  if (lhs->GetCompilerType().IsInteger() && rhs->GetCompilerType().IsInteger()) {
    llvm::Expected<llvm::APSInt> l = lhs->GetValueAsAPSInt();
    llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();

    if (l && r) {
      bool ret = Compare(kind, *l, *r);
      return ValueObject::CreateValueObjectFromBool(m_target, ret, "result");
    }
  }

  // Evaluate arithmetic operation for two floating point values.
  if (lhs->GetCompilerType().IsFloat() && rhs->GetCompilerType().IsFloat()) {
    llvm::Expected<llvm::APFloat> l = lhs->GetValueAsAPFloat();
    llvm::Expected<llvm::APFloat> r = rhs->GetValueAsAPFloat();
    if (l && r) {
      bool ret = Compare(kind, *l, *r);
      return ValueObject::CreateValueObjectFromBool(m_target, ret, "result");
    }
  }

  // Evaluate arithmetic operation for two scoped enum values.
  if (lhs->GetCompilerType().IsScopedEnumerationType()
      && rhs->GetCompilerType().IsScopedEnumerationType()) {
    llvm::Expected<llvm::APSInt> l = lhs->GetValueAsAPSInt();
    llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();
    if (l && r) {
      bool ret = Compare(kind, *l, *r);
      return ValueObject::CreateValueObjectFromBool(m_target, ret, "result");
    }
  }

  // Must be pointer/integer and/or nullptr comparison.
  size_t ptr_size = m_target->GetArchitecture().GetAddressByteSize() * 8;
  llvm::Expected<llvm::APSInt> l = lhs->GetValueAsAPSInt();
  llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();

  if (l && r) {
    bool ret =
        Compare(kind, llvm::APSInt(l->sextOrTrunc(ptr_size), true),
                llvm::APSInt(r->sextOrTrunc(ptr_size), true));
    return ValueObject::CreateValueObjectFromBool(m_target, ret, "result");
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateDereference(lldb::ValueObjectSP rhs)
{
  // If rhs is a reference, dereference it first.
  Status error;
  if (rhs->GetCompilerType().IsReferenceType())
    rhs = rhs->Dereference(error);

  assert(rhs->GetCompilerType().IsPointerType()
         && "invalid ast: must be a pointer type");

  if (rhs->GetDerefValobj())
    return rhs->GetDerefValobj()->GetSP();

  CompilerType pointer_type = rhs->GetCompilerType();
  lldb::addr_t base_addr = rhs->GetValueAsUnsigned(0);

  llvm::StringRef name = "result";
  ExecutionContext exe_ctx(m_target.get(), false);
  lldb::ValueObjectSP value =
          ValueObject::CreateValueObjectFromAddress(name, base_addr, exe_ctx,
                                                    pointer_type,
                                                    /* do_deref */ false);

  // If we're in the address-of context, skip the dereference and cancel the
  // pending address-of operation as well.
  if (flow_analysis() && flow_analysis()->AddressOfIsPending()) {
    flow_analysis()->DiscardAddressOf();
    return value;
  }

  return value->Dereference(error);
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryMinus(lldb::ValueObjectSP rhs)
{
  assert((rhs->GetCompilerType().IsInteger() || rhs->GetCompilerType().IsFloat())
         && "invalid ast: must be an arithmetic type");

  if (rhs->GetCompilerType().IsInteger()) {
    llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();
    if (r) {
      llvm::APSInt v = *r;
      v.negate();
      return ValueObject::CreateValueObjectFromAPInt(m_target, v,
                                                     rhs->GetCompilerType(),
                                                     "result");
    }
  }
  if (rhs->GetCompilerType().IsFloat()) {
    auto value_or_err = rhs->GetValueAsAPFloat();
    if (value_or_err) {
      llvm::APFloat v = *value_or_err;
      v.changeSign();
      return ValueObject::CreateValueObjectFromAPFloat(m_target, v,
                                                       rhs->GetCompilerType(),
                                                       "result");
    }
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryNegation(
    lldb::ValueObjectSP rhs)
{
  assert(rhs->GetCompilerType().IsContextuallyConvertibleToBool() &&
         "invalid ast: must be convertible to bool");
  auto value_or_err = rhs->GetValueAsBool();
  if (value_or_err)
    return ValueObject::CreateValueObjectFromBool(m_target,
                                                  !(*value_or_err),
                                                  "result");
  else
    return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryBitwiseNot(
    lldb::ValueObjectSP rhs) {
  assert(rhs->GetCompilerType().IsInteger() && "invalid ast: must be an integer");
  auto value_or_err = rhs->GetValueAsAPSInt();
  if (value_or_err) {
    llvm::APSInt v = *value_or_err;
    v.flipAllBits();
    return ValueObject::CreateValueObjectFromAPInt(m_target, v,
                                                   rhs->GetCompilerType(),
                                                   "result");
  }
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryPrefixIncrement(
    lldb::ValueObjectSP rhs)
{
  assert((rhs->GetCompilerType().IsInteger() || rhs->GetCompilerType().IsFloat()
          || rhs->GetCompilerType().IsPointerType()) &&
         "invalid ast: must be either arithmetic type or pointer");

  Status status;
  if (rhs->GetCompilerType().IsInteger()) {
    llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();
    if (r) {
      llvm::APSInt v = *r;
      ++v;  // Do the increment.

      rhs->SetValueFromInteger(v, status);
      if (status.Success())
        return rhs;
      return lldb::ValueObjectSP();
    }
  }
  if (rhs->GetCompilerType().IsFloat()) {
    auto value_or_err = rhs->GetValueAsAPFloat();
    if (value_or_err) {
      llvm::APFloat v = *value_or_err;
      // Do the increment.
      v = v + llvm::APFloat(v.getSemantics(), 1ULL);

      rhs->SetValueFromInteger(v.bitcastToAPInt(), status);
      if (status.Success())
        return rhs;
      return lldb::ValueObjectSP();
    }
  }
  if (rhs->GetCompilerType().IsPointerType()) {
    uint64_t v = rhs->GetValueAsUnsigned(0);
    uint64_t byte_size = 0;
    if (auto temp =
            rhs->GetCompilerType().GetPointeeType().GetByteSize(
                rhs->GetTargetSP().get()))
      byte_size = temp.value();
    v += byte_size;;  // Do the increment.

    rhs->SetValueFromInteger(llvm::APInt(64, v), status);
    if (status.Success())
      return rhs;
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateUnaryPrefixDecrement(
    lldb::ValueObjectSP rhs) {
  assert((rhs->GetCompilerType().IsInteger() ||
          rhs->GetCompilerType().IsFloat() ||
          rhs->GetCompilerType().IsPointerType()) &&
         "invalid ast: must be either arithmetic type or pointer");

  Status status;
  if (rhs->GetCompilerType().IsInteger()) {
    llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();
    if (r) {
      llvm::APSInt v = *r;
      --v;  // Do the decrement.

      rhs->SetValueFromInteger(v, status);
      if (status.Success())
        return rhs;
      return lldb::ValueObjectSP();
    }
  }
  if (rhs->GetCompilerType().IsFloat()) {
    auto value_or_err = rhs->GetValueAsAPFloat();
    if (value_or_err) {
      llvm::APFloat v = *value_or_err;
      // Do the decrement.
      v = v - llvm::APFloat(v.getSemantics(), 1ULL);

      rhs->SetValueFromInteger(v.bitcastToAPInt(), status);
      if (status.Success())
        return rhs;
      return lldb::ValueObjectSP();
    }
  }
  if (rhs->GetCompilerType().IsPointerType()) {
    uint64_t v = rhs->GetValueAsUnsigned(0);
    uint64_t byte_size = 0;
    if (auto temp =
            rhs->GetCompilerType().GetPointeeType().GetByteSize(
                rhs->GetTargetSP().get()))
      byte_size = temp.value();
    v -= byte_size;  // Do the decrement.

    rhs->SetValueFromInteger(llvm::APInt(64, v), status);
    if (status.Success())
      return rhs;
  }

  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryAddition(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  // Addition of two arithmetic types.
  if (lhs->GetCompilerType().IsScalarType()
      && rhs->GetCompilerType().IsScalarType()) {
    assert(lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()) &&
           "invalid ast: operand must have the same type");
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Add, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }

  // Here one of the operands must be a pointer and the other one an integer.
  lldb::ValueObjectSP ptr, offset;
  if (lhs->GetCompilerType().IsPointerType()) {
    ptr = lhs;
    offset = rhs;
  } else {
    ptr = rhs;
    offset = lhs;
  }
  assert(ptr->GetCompilerType().IsPointerType() &&
         "invalid ast: ptr must be a pointer");
  assert(offset->GetCompilerType().IsInteger() &&
         "invalid ast: offset must be an integer");

  if (ptr->GetValueAsUnsigned(0) == 0 && offset->GetValueAsUnsigned(0) != 0) {
    // Binary addition with null pointer causes mismatches between LLDB and
    // lldb-eval if the offset different than zero.
    SetUbStatus(m_error, ErrorCode::kUBNullPtrArithmetic);
  }

  return PointerAdd(ptr, offset->GetValueAsUnsigned(0));
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinarySubtraction(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs,
    CompilerType result_type)
{
  if (lhs->GetCompilerType().IsScalarType()
      && rhs->GetCompilerType().IsScalarType()) {
    assert(lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()) &&
           "invalid ast: operand must have the same type");
    return EvaluateArithmeticOp(m_target, BinaryOpKind::Sub, lhs, rhs,
                                lhs->GetCompilerType().GetCanonicalType());
  }
  assert(lhs->GetCompilerType().IsPointerType()
         && "invalid ast: lhs must be a pointer");

  // "pointer - integer" operation.
  if (rhs->GetCompilerType().IsInteger()) {
    return PointerAdd(lhs, - rhs->GetValueAsUnsigned(0));
  }

  uint64_t lhs_byte_size = 0;
  uint64_t rhs_byte_size = 0;
  if (auto temp =
      lhs->GetCompilerType().GetPointeeType().GetByteSize(
          lhs->GetTargetSP().get()))
    lhs_byte_size = temp.value();
  if (auto temp = rhs->GetCompilerType().GetPointeeType().GetByteSize(
          rhs->GetTargetSP().get()))
    rhs_byte_size = temp.value();
  // "pointer - pointer" operation.
  assert(rhs->GetCompilerType().IsPointerType()
         && "invalid ast: rhs must an integer or a pointer");
  assert((lhs_byte_size == rhs_byte_size) &&
         "invalid ast: pointees should be the same size");

  // Since pointers have compatible types, both have the same pointee size.
  uint64_t item_size = lhs_byte_size;
  // Pointer difference is a signed value.
  int64_t diff = static_cast<int64_t>(lhs->GetValueAsUnsigned(0) -
                                      rhs->GetValueAsUnsigned(0));

  if (diff % item_size != 0 && diff < 0) {
    // If address difference isn't divisible by pointee size then performing
    // the operation is undefined behaviour. Note: mismatches were encountered
    // only for negative difference (diff < 0).
    SetUbStatus(m_error, ErrorCode::kUBInvalidPtrDiff);
  }

  diff /= static_cast<int64_t>(item_size);

  // Pointer difference is ptrdiff_t.
  ExecutionContext exe_ctx(m_target.get(), false);
  uint64_t byte_size = 0;
  if (auto temp = result_type.GetByteSize(m_target.get()))
    byte_size = temp.value();
  lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
      reinterpret_cast<const void*>(&diff), byte_size,
      exe_ctx.GetByteOrder(), exe_ctx.GetAddressByteSize());
  return ValueObject::CreateValueObjectFromData("result", *data_sp, exe_ctx,
                                                result_type);
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryMultiplication(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((lhs->GetCompilerType().IsScalarType() &&
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must be arithmetic and have the same type");

  return EvaluateArithmeticOp(m_target, BinaryOpKind::Mul, lhs, rhs,
                              lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryDivision(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert((lhs->GetCompilerType().IsScalarType() &&
          lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType())) &&
         "invalid ast: operands must be arithmetic and have the same type");

  // Check for zero only for integer division.
  if (rhs->GetCompilerType().IsInteger() && rhs->GetValueAsUnsigned(0) == 0) {
    // This is UB and the compiler would generate a warning:
    //
    //  warning: division by zero is undefined [-Wdivision-by-zero]
    //
    SetUbStatus(m_error, ErrorCode::kUBDivisionByZero);

    return rhs;
  }

  if (rhs->GetCompilerType().IsInteger() && IsInvalidDivisionByMinusOne(lhs, rhs))
  {
    SetUbStatus(m_error, ErrorCode::kUBDivisionByMinusOne);
  }

  return EvaluateArithmeticOp(m_target, BinaryOpKind::Div, lhs, rhs,
                              lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryRemainder(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  assert((lhs->GetCompilerType().IsInteger()
          && lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()))
         && "invalid ast: operands must be integers and have the same type");

  if (rhs->GetValueAsUnsigned(0) == 0) {
    // This is UB and the compiler would generate a warning:
    //
    //  warning: remainder by zero is undefined [-Wdivision-by-zero]
    //
    SetUbStatus(m_error, ErrorCode::kUBDivisionByZero);

    return rhs;
  }

  if (IsInvalidDivisionByMinusOne(lhs, rhs)) {
    SetUbStatus(m_error, ErrorCode::kUBDivisionByMinusOne);
  }

  return EvaluateArithmeticOpInteger(m_target, BinaryOpKind::Rem, lhs, rhs,
                                     lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryBitwise(
    BinaryOpKind kind,
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  assert((lhs->GetCompilerType().IsInteger()
          && lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()))
         &&"invalid ast: operands must be integers and have the same type");
  assert((kind == BinaryOpKind::And || kind == BinaryOpKind::Or ||
          kind == BinaryOpKind::Xor) &&
         "invalid ast: operation must be '&', '|' or '^'");

  return EvaluateArithmeticOpInteger(m_target, kind, lhs, rhs,
                                     lhs->GetCompilerType().GetCanonicalType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryShift(BinaryOpKind kind,
                                                        lldb::ValueObjectSP lhs,
                                                        lldb::ValueObjectSP rhs)
{
  assert(lhs->GetCompilerType().IsInteger() &&
         rhs->GetCompilerType().IsInteger() &&
         "invalid ast: operands must be integers");
  assert((kind == BinaryOpKind::Shl || kind == BinaryOpKind::Shr) &&
         "invalid ast: operation must be '<<' or '>>'");

  uint64_t lhs_byte_size = 0;
  if (auto temp = lhs->GetCompilerType().GetByteSize(lhs->GetTargetSP().get()))
    lhs_byte_size = temp.value();
  // Performing shift operation is undefined behaviour if the right operand
  // isn't in interval [0, bit-size of the left operand).
  llvm::Expected<llvm::APSInt> r = rhs->GetValueAsAPSInt();
  if (r && (r->isNegative() ||
            (rhs->GetValueAsUnsigned(0) >= lhs_byte_size * CHAR_BIT))) {
    SetUbStatus(m_error, ErrorCode::kUBInvalidShift);
  }

  return EvaluateArithmeticOpInteger(m_target, kind, lhs, rhs,
                                     lhs->GetCompilerType());
}

lldb::ValueObjectSP DILInterpreter::EvaluateAssignment(lldb::ValueObjectSP lhs,
                                                       lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().CompareTypes(rhs->GetCompilerType()) &&
         "invalid ast: operands must have the same type");

  Status status;
  lhs->SetValueFromInteger(rhs, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryAddAssign(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs) {
  lldb::ValueObjectSP ret;

  if (lhs->GetCompilerType().IsPointerType()) {
    assert(rhs->GetCompilerType().IsInteger() &&
           "invalid ast: rhs must be an integer");
    ret = EvaluateBinaryAddition(lhs, rhs);
  } else {
    assert(lhs->GetCompilerType().IsScalarType()
           && "invalid ast: lhs must be an arithmetic type");
    assert(rhs->GetCompilerType().IsBasicType() &&
           "invalid ast: rhs must be a basic type");
    ret = lhs->CastToBasicType(rhs->GetCompilerType());
    ret = EvaluateBinaryAddition(ret, rhs);
    ret = ret->CastToBasicType(lhs->GetCompilerType());
  }

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinarySubAssign(
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs)
{
  lldb::ValueObjectSP ret;

  if (lhs->GetCompilerType().IsPointerType()) {
    assert(rhs->GetCompilerType().IsInteger() &&
           "invalid ast: rhs must be an integer");
    ret = EvaluateBinarySubtraction(lhs, rhs, lhs->GetCompilerType());
  } else {
    assert(lhs->GetCompilerType().IsScalarType()
           && "invalid ast: lhs must be an arithmetic type");
    assert(rhs->GetCompilerType().IsBasicType() &&
           "invalid ast: rhs must be a basic type");
    ret = lhs->CastToBasicType(rhs->GetCompilerType());
    ret = EvaluateBinarySubtraction(ret, rhs, ret->GetCompilerType());
    ret = ret->CastToBasicType(lhs->GetCompilerType());
  }

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryMulAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType()
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastToBasicType(rhs->GetCompilerType());
  ret = EvaluateBinaryMultiplication(ret, rhs);
  ret = ret->CastToBasicType(lhs->GetCompilerType());

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryDivAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType()
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastToBasicType(rhs->GetCompilerType());
  ret = EvaluateBinaryDivision(ret, rhs);
  ret = ret->CastToBasicType(lhs->GetCompilerType());

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryRemAssign(
    lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs) {
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType()
         && "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastToBasicType(rhs->GetCompilerType());

  ret = EvaluateBinaryRemainder(ret, rhs);
  ret = ret->CastToBasicType(lhs->GetCompilerType());

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryBitwiseAssign(
    BinaryOpKind kind, lldb::ValueObjectSP lhs, lldb::ValueObjectSP rhs)
{
  switch (kind) {
    case BinaryOpKind::AndAssign:
      kind = BinaryOpKind::And;
      break;
    case BinaryOpKind::OrAssign:
      kind = BinaryOpKind::Or;
      break;
    case BinaryOpKind::XorAssign:
      kind = BinaryOpKind::Xor;
      break;
    default:
      assert(false && "invalid BinaryOpKind: must be '&=', '|=' or '^='");
      break;
  }
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType() &&
         "invalid ast: rhs must be a basic type");

  lldb::ValueObjectSP ret = lhs->CastToBasicType(rhs->GetCompilerType());
  ret = EvaluateBinaryBitwise(kind, ret, rhs);
  ret = ret->CastToBasicType(lhs->GetCompilerType());

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::EvaluateBinaryShiftAssign(
    BinaryOpKind kind,
    lldb::ValueObjectSP lhs,
    lldb::ValueObjectSP rhs,
    CompilerType comp_assign_type)
{
  switch (kind) {
    case BinaryOpKind::ShlAssign:
      kind = BinaryOpKind::Shl;
      break;
    case BinaryOpKind::ShrAssign:
      kind = BinaryOpKind::Shr;
      break;
    default:
      assert(false && "invalid BinaryOpKind: must be '<<=' or '>>='");
      break;
  }
  assert(lhs->GetCompilerType().IsScalarType()
         && "invalid ast: lhs must be an arithmetic type");
  assert(rhs->GetCompilerType().IsBasicType() &&
         "invalid ast: rhs must be a basic type");
  assert(comp_assign_type.IsInteger() &&
         "invalid ast: comp_assign_type must be an integer");

  lldb::ValueObjectSP ret = lhs->CastToBasicType(comp_assign_type);
  ret = EvaluateBinaryShift(kind, ret, rhs);
  ret = ret->CastToBasicType(lhs->GetCompilerType());

  Status status;
  lhs->SetValueFromInteger(ret, status);
  if (status.Success())
    return lhs;
  return lldb::ValueObjectSP();
}

lldb::ValueObjectSP DILInterpreter::PointerAdd(lldb::ValueObjectSP lhs,
                                               int64_t offset) {
  uint64_t byte_size = 0;
  if (auto temp = lhs->GetCompilerType().GetPointeeType().GetByteSize(
          lhs->GetTargetSP().get()))
    byte_size = temp.value();
  uintptr_t addr = lhs->GetValueAsUnsigned(0) + offset * byte_size;

  llvm::StringRef name = "result";
  ExecutionContext exe_ctx(m_target.get(), false);
  return ValueObject::CreateValueObjectFromAddress(name, addr, exe_ctx,
                                                   lhs->GetCompilerType(),
                                                    /* do_deref */ false);
}

lldb::ValueObjectSP DILInterpreter::ResolveContextVar(
    const std::string& name) const
{
  auto it = m_context_vars.find(name);
  return it != m_context_vars.end() ? it->second : lldb::ValueObjectSP();
}

}  // namespace dil

}  // namespace lldb_private
