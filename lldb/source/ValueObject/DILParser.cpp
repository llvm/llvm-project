//===-- DILParser.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This implements the recursive descent parser for the Data Inspection
// Language (DIL), and its helper functions, which will eventually underlie the
// 'frame variable' command. The language that this parser recognizes is
// described in lldb/docs/dil-expr-lang.ebnf
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILParser.h"

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <limits.h>

#include "lldb/lldb-enumerations.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILEval.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Host.h"

namespace {

const char* kInvalidOperandsToUnaryExpression =
    "invalid argument type {0} to unary expression";

const char* kInvalidOperandsToBinaryExpression =
    "invalid operands to binary expression ({0} and {1})";

const char* kValueIsNotConvertibleToBool =
    "value of type {0} is not contextually convertible to 'bool'";

template <typename T>
constexpr unsigned type_width() {
  return static_cast<unsigned>(sizeof(T)) * CHAR_BIT;
}

}  // namespace

namespace lldb_private {

namespace dil {

inline void TokenKindsJoinImpl(std::ostringstream& os,
                               dil::TokenKind k) {
  os << "'" << DILToken::getTokenName(k) << "'";
}

template <typename... Ts>
inline void TokenKindsJoinImpl(std::ostringstream& os, dil::TokenKind k,
                               Ts... ks) {
  TokenKindsJoinImpl(os, k);
  os << ", ";
  TokenKindsJoinImpl(os, ks...);
}

template <typename... Ts>
inline std::string TokenKindsJoin(dil::TokenKind k, Ts... ks) {
  std::ostringstream os;
  TokenKindsJoinImpl(os, k, ks...);

  return os.str();
}

//static std::unordered_map<std::string, CompilerType> context_args;

/// Quick lookup to check if a type name already exists in a
/// name-to-CompilerType map the DIL parser keeps of previously found
/// name/type pairs, for LLDB convenience var. -- look for 'persistent'
/// vars in Expressions directory.
static bool IsContextVar(const std::string &name) {
  return false;
}

static CompilerType GetBasicType(std::shared_ptr<ExecutionContextScope> ctx,
                                 lldb::BasicType basic_type) {
  static std::unordered_map<lldb::BasicType, CompilerType> basic_types;
  auto type = basic_types.find(basic_type);
  if (type != basic_types.end()) {
    std::string type_name((type->second).GetTypeName().AsCString());
    // Only return the found type if it's valid.
    if (type_name != "<invalid>")
      return type->second;
  }

  lldb::TargetSP target_sp = ctx->CalculateTarget();
  if (target_sp) {
    for (auto type_system_sp : target_sp->GetScratchTypeSystems())
      if (auto compiler_type = type_system_sp->GetBasicTypeFromAST(basic_type)){
        basic_types.insert({basic_type, compiler_type});
        return compiler_type;
      }
  }
  CompilerType empty_type;
  return empty_type;
}

static lldb::BasicType GetPtrDiffType(std::shared_ptr<ExecutionContextScope> ctx)
{
  lldb::TargetSP target_sp = ctx->CalculateTarget();
  llvm::Triple triple(
      llvm::Twine(target_sp->GetArchitecture().GetTriple().str()));

  if (triple.isOSWindows()) {
    return triple.isArch64Bit() ? lldb::eBasicTypeLongLong
                                : lldb::eBasicTypeInt;
  } else {
    return triple.isArch64Bit() ? lldb::eBasicTypeLong : lldb::eBasicTypeInt;
  }
}

static std::unique_ptr<BuiltinFunctionDef> GetBuiltinFunctionDef(
    std::shared_ptr<ExecutionContextScope> ctx, const std::string& identifier) {
  //
  // __log2(unsigned int x) -> unsigned int
  //
  //   Calculates the log2(x).
  //
  if (identifier == "__log2") {
    CompilerType return_type = GetBasicType(ctx, lldb::eBasicTypeUnsignedInt);
    std::vector<CompilerType> arguments = {
      GetBasicType(ctx, lldb::eBasicTypeUnsignedInt),
    };
    return std::make_unique<BuiltinFunctionDef>(identifier, return_type,
                                                std::move(arguments));
  }
  //
  // __findnonnull(T* ptr, long long buffer_size) -> int
  //
  //   Finds the first non-null object pointed by `ptr`. `ptr` is treated as an
  //   array of pointers of size `buffer_size`.
  //
  if (identifier == "__findnonnull") {
    auto return_type = GetBasicType(ctx, lldb::eBasicTypeInt);
    std::vector<CompilerType> arguments = {
        // The first argument should actually be "T*", but we don't support
        // templates here.
        // HACK: Void means "any" and we'll check in runtime. The argument will
        // be passed as is without any conversions.
      GetBasicType(ctx, lldb::eBasicTypeVoid),
      GetBasicType(ctx,lldb::eBasicTypeLongLong),
    };
    return std::make_unique<BuiltinFunctionDef>(identifier, return_type,
                                                std::move(arguments));
  }
  // Not a builtin function.
  return nullptr;
}

static bool GetPathToBaseType(CompilerType type, CompilerType target_base,
                              std::vector<uint32_t>* path,
                              uint64_t* offset) {
  if (type.CompareTypes(target_base)) {
    return true;
  }

  uint32_t bit_offset = 0;
  uint32_t num_non_empty_bases = 0;
  uint32_t num_direct_bases =
      type.GetNumDirectBaseClasses();
  for (uint32_t i = 0; i < num_direct_bases; ++i) {
    auto member_base_type =
        type.GetDirectBaseClassAtIndex(i, &bit_offset);
    if (GetPathToBaseType(member_base_type, target_base, path, offset)) {
      if (path) {
        path->push_back(num_non_empty_bases);
      }
      if (offset) {
        *offset += bit_offset / 8u;
      }
      return true;
    }
    if (member_base_type.GetNumFields() > 0) {
      num_non_empty_bases++;
    }
  }

  return false;
}

std::shared_ptr<DILSourceManager> DILSourceManager::Create(std::string expr) {
  return std::shared_ptr<DILSourceManager>(new DILSourceManager(
      std::move(expr)));
};

static const char* ToString(TypeDeclaration::TypeSpecifier type_spec) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  switch (type_spec) {
    case TypeSpecifier::kVoid:       return "void";
    case TypeSpecifier::kBool:       return "bool";
    case TypeSpecifier::kChar:       return "char";
    case TypeSpecifier::kShort:      return "short";
    case TypeSpecifier::kInt:        return "int";
    case TypeSpecifier::kLong:       return "long";
    case TypeSpecifier::kLongLong:   return "long long";
    case TypeSpecifier::kFloat:      return "float";
    case TypeSpecifier::kDouble:     return "double";
    case TypeSpecifier::kLongDouble: return "long double";
    case TypeSpecifier::kWChar:      return "wchar_t";
    case TypeSpecifier::kChar16:     return "char16_t";
    case TypeSpecifier::kChar32:     return "char32_t";
    default:
      assert(false && "invalid type specifier");
      return nullptr;
  }
}

static const char* ToString(TypeDeclaration::SignSpecifier sign_spec) {
  using SignSpecifier = TypeDeclaration::SignSpecifier;
  switch (sign_spec) {
    case SignSpecifier::kSigned:   return "signed";
    case SignSpecifier::kUnsigned: return "unsigned";
    default:
      assert(false && "invalid sign specifier");
      return nullptr;
  }
}

static bool TokenEndsTemplateArgumentList(const DILToken& token) {
  // Note: in C++11 ">>" can be treated as "> >" and thus be a valid token
  // for the template argument list.
  return token.isOneOf(dil::TokenKind::comma, dil::TokenKind::greater,
                       dil::TokenKind::greatergreater);
}

static DILASTNodeUP InsertArrayToPointerConversion(DILASTNodeUP expr) {
  assert(expr->GetDereferencedResultType().IsArrayType() &&
         "an argument to array-to-pointer conversion must be an array");

  // TODO: Make this an explicit array-to-pointer conversion instead of
  // using a "generic" CStyleCastNode.
  return std::make_unique<CStyleCastNode>(
      expr->GetLocation(),
      expr->GetDereferencedResultType().GetArrayElementType(nullptr).GetPointerType(),
      std::move(expr), TypePromotionCastKind::ePointer);
}

static CompilerType DoIntegralPromotion(
    std::shared_ptr<ExecutionContextScope> ctx, CompilerType from) {
  assert((from.IsInteger() || from.IsUnscopedEnumerationType()) &&
         "Integral promotion works only for integers and unscoped enums.");

  // Don't do anything if the type doesn't need to be promoted.
  if (!from.IsPromotableIntegerType()) {
    return from;
  }

  if (from.IsUnscopedEnumerationType()) {
    // Get the enumeration underlying type and promote it.
    return DoIntegralPromotion(ctx, from.GetEnumerationIntegerType());
  }

  // At this point the type should an integer.
  assert(from.IsInteger() && "invalid type: must be an integer");

  // Get the underlying builtin representation.
  lldb::BasicType builtin_type =
      from.GetCanonicalType().GetBasicTypeEnumeration();

  uint64_t from_size = 0;
  if (builtin_type == lldb::eBasicTypeWChar ||
      builtin_type == lldb::eBasicTypeSignedWChar ||
      builtin_type == lldb::eBasicTypeUnsignedWChar ||
      builtin_type == lldb::eBasicTypeChar16 ||
      builtin_type == lldb::eBasicTypeChar32) {
    // Find the type that can hold the entire range of values for our type.
    bool is_signed = from.IsSigned();
    if (auto temp = from.GetByteSize(ctx.get()))
      from_size = temp.value();

    CompilerType promote_types[] = {
      GetBasicType(ctx, lldb::eBasicTypeInt),
      GetBasicType(ctx, lldb::eBasicTypeUnsignedInt),
      GetBasicType(ctx, lldb::eBasicTypeLong),
      GetBasicType(ctx, lldb::eBasicTypeUnsignedLong),
      GetBasicType(ctx, lldb::eBasicTypeLongLong),
      GetBasicType(ctx, lldb::eBasicTypeUnsignedLongLong),
    };
    for (auto& type : promote_types) {
      uint64_t byte_size = 0;
      if (auto temp = type.GetByteSize(ctx.get()))
        byte_size = temp.value();
      if (from_size < byte_size ||
          (from_size == byte_size &&
           is_signed ==(bool)(
               type.GetTypeInfo() & lldb::eTypeIsSigned)))
      {
        return type;
      }
    }

    llvm_unreachable("char type should fit into long long");
  }

  // Here we can promote only to "int" or "unsigned int".
  CompilerType int_type = GetBasicType(ctx, lldb::eBasicTypeInt);
  uint64_t int_byte_size = 0;
  if (auto temp = int_type.GetByteSize(ctx.get()))
    int_byte_size = temp.value();

  // Signed integer types can be safely promoted to "int".
  if (from.IsSigned()) {
    return int_type;
  }
  // Unsigned integer types are promoted to "unsigned int" if "int" cannot hold
  // their entire value range.
  return (from_size == int_byte_size)
      ? GetBasicType(ctx, lldb::eBasicTypeUnsignedInt)
      : int_type;
}

static DILASTNodeUP UsualUnaryConversions(
    std::shared_ptr<ExecutionContextScope>  ctx, DILASTNodeUP expr) {
  // Perform usual conversions for unary operators. At the moment this includes
  // array-to-pointer and the integral promotion for eligible types.
  auto result_type = expr->GetDereferencedResultType();

  if (expr->is_bitfield()) {
    // Promote bitfields. If `int` can represent the bitfield value, it is
    // converted to `int`. Otherwise, if `unsigned int` can represent it, it
    // is converted to `unsigned int`. Otherwise, it is treated as its
    // underlying type.

    uint32_t bitfield_size = expr->bitfield_size();
    // Some bitfields have undefined size (e.g. result of ternary operation).
    // The AST's `bitfield_size` of those is 0, and no promotion takes place.
    if (bitfield_size > 0 && result_type.IsInteger()) {
      auto int_type = GetBasicType(ctx, lldb::eBasicTypeInt);
      auto uint_type = GetBasicType(ctx, lldb::eBasicTypeUnsignedInt);
      uint64_t int_byte_size = 0;
      uint64_t uint_byte_size = 0;
      if (auto temp = int_type.GetByteSize(ctx.get()))
        int_byte_size = temp.value();
      if (auto temp = uint_type.GetByteSize(ctx.get()))
        uint_byte_size = temp.value();
      uint32_t int_bit_size = int_byte_size * CHAR_BIT;
      if (bitfield_size < int_bit_size ||
          (result_type.IsSigned() && bitfield_size == int_bit_size)) {
        expr = std::make_unique<CStyleCastNode>(expr->GetLocation(), int_type,
                                                std::move(expr),
                                                TypePromotionCastKind::eArithmetic);
      } else if (bitfield_size <= uint_byte_size * CHAR_BIT) {
        expr = std::make_unique<CStyleCastNode>(expr->GetLocation(), uint_type,
                                                std::move(expr),
                                                TypePromotionCastKind::eArithmetic);
      }
    }
  }

  if (result_type.IsArrayType()) {
    expr = InsertArrayToPointerConversion(std::move(expr));
  }

  if (result_type.IsInteger() || result_type.IsUnscopedEnumerationType()) {
    auto promoted_type = DoIntegralPromotion(ctx, result_type);

    // Insert a cast if the type promotion is happening.
    // TODO: Make this an implicit static_cast.
    if (!promoted_type.CompareTypes(result_type)) {
      expr = std::make_unique<CStyleCastNode>(expr->GetLocation(), promoted_type,
                                              std::move(expr),
                                              TypePromotionCastKind::eArithmetic);
    }
  }

  return expr;
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

      // TODO: The ranks of char16_t, char32_t, and wchar_t are equal to the
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

static void PerformIntegerConversions(std::shared_ptr<ExecutionContextScope> ctx,
                                      DILASTNodeUP& l,
                                      DILASTNodeUP& r, bool convert_lhs,
                                      bool convert_rhs) {
  // Assert that rank(l) < rank(r).
  auto l_type = l->GetDereferencedResultType();
  auto r_type = r->GetDereferencedResultType();

  // if `r` is signed and `l` is unsigned, check whether it can represent all
  // of the values of the type of the `l`. If not, then promote `r` to the
  // unsigned version of its type.
  if (r_type.IsSigned() && !l_type.IsSigned()) {
    uint64_t l_size = 0;
    uint64_t r_size = 0;
    if (auto temp = l_type.GetByteSize(ctx.get()))
      l_size = temp.value();;
    if (auto temp = r_type.GetByteSize(ctx.get()))
      r_size = temp.value();

    assert(l_size <= r_size && "left value must not be larger then the right!");

    if (r_size == l_size) {
      auto r_type_unsigned = GetBasicType(
          ctx,
          BasicTypeToUnsigned(r_type.GetCanonicalType()
                                  .GetBasicTypeEnumeration()));
      if (convert_rhs) {
        r = std::make_unique<CStyleCastNode>(r->GetLocation(), r_type_unsigned,
                                             std::move(r),
                                             TypePromotionCastKind::eArithmetic);
      }
    }
  }

  if (convert_lhs) {
    l = std::make_unique<CStyleCastNode>(l->GetLocation(), r->result_type(),
                                         std::move(l),
                                         TypePromotionCastKind::eArithmetic);
  }
}

static CompilerType UsualArithmeticConversions(
    std::shared_ptr<ExecutionContextScope> ctx,
    DILASTNodeUP& lhs,
    DILASTNodeUP& rhs,
    bool is_comp_assign = false) {
  // Apply unary conversions (e.g. intergal promotion) for both operands.
  // In case of a composite assignment operator LHS shouldn't get promoted.
  if (!is_comp_assign) {
    lhs = UsualUnaryConversions(ctx, std::move(lhs));
  }
  rhs = UsualUnaryConversions(ctx, std::move(rhs));

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  if (lhs_type.CompareTypes(rhs_type)) {
    return lhs_type;
  }

  // If either of the operands is not arithmetic (e.g. pointer), we're done.
  if (!lhs_type.IsScalarType() || !rhs_type.IsScalarType()) {
    CompilerType bad_type;
    return bad_type;
  }

  // Handle conversions for floating types (float, double).
  if (lhs_type.IsFloat() || rhs_type.IsFloat()) {
    // If both are floats, convert the smaller operand to the bigger.
    if (lhs_type.IsFloat() && rhs_type.IsFloat()) {
      int order = lhs_type.GetBasicTypeEnumeration() -
                  rhs_type.GetBasicTypeEnumeration();
      if (order > 0) {
        rhs = std::make_unique<CStyleCastNode>(rhs->GetLocation(), lhs_type,
                                               std::move(rhs),
                                               TypePromotionCastKind::eArithmetic);
        return lhs_type;
      }
      assert(order < 0 && "illegal operands: must not be of the same type");
      if (!is_comp_assign) {
        lhs = std::make_unique<CStyleCastNode>(lhs->GetLocation(), rhs_type,
                                               std::move(lhs),
                                               TypePromotionCastKind::eArithmetic);
      }
      return rhs_type;
    }

    if (lhs_type.IsFloat()) {
      assert(rhs_type.IsInteger() && "illegal operand: must be an integer");
      rhs = std::make_unique<CStyleCastNode>(rhs->GetLocation(), lhs_type,
                                             std::move(rhs),
                                             TypePromotionCastKind::eArithmetic);
      return lhs_type;
    }
    assert(rhs_type.IsFloat() && "illegal operand: must be a float");
    if (!is_comp_assign) {
      lhs = std::make_unique<CStyleCastNode>(lhs->GetLocation(), rhs_type,
                                             std::move(lhs),
                                             TypePromotionCastKind::eArithmetic);
    }
    return rhs_type;
  }

  // Handle conversion for integer types.
  assert((lhs_type.IsInteger() && rhs_type.IsInteger()) &&
         "illegal operands: must be both integers");

  using Rank = std::tuple<size_t, bool>;
  Rank l_rank = {ConversionRank(lhs_type), !lhs_type.IsSigned()};
  Rank r_rank = {ConversionRank(rhs_type), !rhs_type.IsSigned()};

  if (l_rank < r_rank) {
    PerformIntegerConversions(ctx, lhs, rhs, !is_comp_assign, true);
  } else if (l_rank > r_rank) {
    PerformIntegerConversions(ctx, rhs, lhs, true, !is_comp_assign);
  }

  if (!is_comp_assign) {
    assert(lhs->GetDereferencedResultType().CompareTypes(rhs->GetDereferencedResultType()) &&
           "integral promotion error: operands result types must be the same");
  }

  return lhs->GetDereferencedResultType().GetCanonicalType();
}

static TypeDeclaration::TypeSpecifier ToTypeSpecifier(
    dil::TokenKind kind) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  switch (kind) {
    case dil::TokenKind::kw_void:     return TypeSpecifier::kVoid;
    case dil::TokenKind::kw_bool:     return TypeSpecifier::kBool;
    case dil::TokenKind::kw_char:     return TypeSpecifier::kChar;
    case dil::TokenKind::kw_short:    return TypeSpecifier::kShort;
    case dil::TokenKind::kw_int:      return TypeSpecifier::kInt;
    case dil::TokenKind::kw_long:     return TypeSpecifier::kLong;
    case dil::TokenKind::kw_float:    return TypeSpecifier::kFloat;
    case dil::TokenKind::kw_double:   return TypeSpecifier::kDouble;
    case dil::TokenKind::kw_wchar_t:  return TypeSpecifier::kWChar;
    case dil::TokenKind::kw_char16_t: return TypeSpecifier::kChar16;
    case dil::TokenKind::kw_char32_t: return TypeSpecifier::kChar32;
    default:
      assert(false && "invalid type specifier token");
      return TypeSpecifier::kUnknown;
  }
}

std::tuple<lldb::BasicType, bool> PickIntegerType(
    std::shared_ptr<ExecutionContextScope> ctx,
    const dil::NumericLiteralParser& literal,
    const llvm::APInt& value) {
  uint64_t int_byte_size = 0;
  uint64_t long_byte_size = 0;
  uint64_t long_long_byte_size = 0;
  if (auto temp = GetBasicType(ctx, lldb::eBasicTypeInt).GetByteSize(nullptr))
    int_byte_size = temp.value();

  if (auto temp = GetBasicType(ctx, lldb::eBasicTypeLong).GetByteSize(nullptr))
    long_byte_size = temp.value();

  if (auto temp = GetBasicType(ctx,
                               lldb::eBasicTypeLongLong).GetByteSize(nullptr))
    long_long_byte_size = temp.value();

  unsigned int_size = int_byte_size * CHAR_BIT;
  unsigned long_size = long_byte_size * CHAR_BIT;
  unsigned long_long_size = long_long_byte_size * CHAR_BIT;


  // Binary, Octal, Hexadecimal and literals with a U suffix are allowed to be
  // an unsigned integer.
  bool unsigned_is_allowed = literal.isUnsigned || literal.getRadix() != 10;

  // Try int/unsigned int.
  if (!literal.isLong && !literal.isLongLong && value.isIntN(int_size)) {
    if (!literal.isUnsigned && value.isIntN(int_size - 1)) {
      return {lldb::eBasicTypeInt, false};
    }
    if (unsigned_is_allowed) {
      return {lldb::eBasicTypeUnsignedInt, true};
    }
  }
  // Try long/unsigned long.
  if (!literal.isLongLong && value.isIntN(long_size)) {
    if (!literal.isUnsigned && value.isIntN(long_size - 1)) {
      return {lldb::eBasicTypeLong, false};
    }
    if (unsigned_is_allowed) {
      return {lldb::eBasicTypeUnsignedLong, true};
    }
  }
  // Try long long/unsigned long long.
  if (value.isIntN(long_long_size)) {
    if (!literal.isUnsigned && value.isIntN(long_long_size - 1)) {
      return {lldb::eBasicTypeLongLong, false};
    }
    if (unsigned_is_allowed) {
      return {lldb::eBasicTypeUnsignedLongLong, true};
    }
  }

  // If we still couldn't decide a type, we probably have something that does
  // not fit in a signed long long, but has no U suffix. Also known as:
  //
  //  warning: integer literal is too large to be represented in a signed
  //  integer type, interpreting as unsigned [-Wimplicitly-unsigned-literal]
  //
  return {lldb::eBasicTypeUnsignedLongLong, true};
}

lldb::BasicType PickCharType(const dil::CharLiteralParser& literal) {
  if (literal.isMultiChar()) {
    return lldb::eBasicTypeInt;
#if LLVM_VERSION_MAJOR < 15
  } else if (literal.isAscii()) {
#else
  } else if (literal.isOrdinary()) {
#endif
    return lldb::eBasicTypeChar;
  } else if (literal.isWide()) {
    return lldb::eBasicTypeWChar;
  } else if (literal.isUTF8()) {
    // TODO: Change to eBasicTypeChar8 when support for u8 is added
    return lldb::eBasicTypeChar;
  }
  return lldb::eBasicTypeChar;
}

lldb::BasicType PickCharType(const dil::StringLiteralParser& literal) {
#if LLVM_VERSION_MAJOR < 15
  if (literal.isAscii()) {
#else
  if (literal.isOrdinary()) {
#endif
    return lldb::eBasicTypeChar;
  } else if (literal.isWide()) {
    return lldb::eBasicTypeWChar;
  } else if (literal.isUTF8()) {
    // TODO: Change to eBasicTypeChar8 when support for u8 is added.
    return lldb::eBasicTypeChar;
  }
  return lldb::eBasicTypeChar;
}

std::string FormatDiagnostics(DILSourceManager& sm, const std::string& message,
                              uint32_t loc) {
  // Get the source buffer and the location of the current token.
  llvm::StringRef text = sm.GetSource();
  size_t loc_offset = (size_t) loc;

  // Look for the start of the line.
  size_t line_start = text.rfind('\n', loc_offset);
  line_start = line_start == llvm::StringRef::npos ? 0 : line_start + 1;

  // Look for the end of the line.
  size_t line_end = text.find('\n', loc_offset);
  line_end = line_end == llvm::StringRef::npos ? text.size() : line_end;

  // Get a view of the current line in the source code and the position of the
  // diagnostics pointer.
  llvm::StringRef line = text.slice(line_start, line_end);
  int32_t arrow = loc + 1; // Column offset starts at 1, not 0.

  // Calculate the padding in case we point outside of the expression (this can
  // happen if the parser expected something, but got EOF).˚
  size_t expr_rpad = std::max(0, arrow - static_cast<int32_t>(line.size()));
  size_t arrow_rpad = std::max(0, static_cast<int32_t>(line.size()) - arrow);

  return llvm::formatv("<expr:1:{0}>: {1}\n{2}\n{3}", loc,
                       message, llvm::fmt_pad(line, 0, expr_rpad),
                       llvm::fmt_pad("^", arrow - 1, arrow_rpad));
}

DILParser::DILParser(std::shared_ptr<DILSourceManager> dil_sm,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member)
    : m_ctx_scope(exe_ctx_scope), m_sm(dil_sm), m_use_dynamic(use_dynamic),
    m_use_synthetic(use_synthetic), m_fragile_ivar(fragile_ivar),
    m_check_ptr_vs_member(check_ptr_vs_member),
    m_dil_lexer(DILLexer(dil_sm))
{
  // Initialize the token.
  m_dil_token.setKind(dil::TokenKind::unknown);
}

DILASTNodeUP DILParser::Run(Status& error) {
  ConsumeToken();

  DILASTNodeUP expr;

  if (m_dil_lexer.isStringLiteral(m_dil_token.getKind()) &&
      m_dil_lexer.LookAhead(0).is(dil::TokenKind::eof)) {
    // A special case to handle a single string-literal token.
    expr = ParseStringLiteral();
  } else {
    expr = ParseExpression();
  }


  Expect(dil::TokenKind::eof);

  error = std::move(m_error);
  m_error.Clear();

  // Explicitly return ErrorNode if there was an error during the parsing.
  // Some routines raise an error, but don't change the return value (e.g.
  // Expect).
  if (error.Fail()) {
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }
  return expr;
}

CompilerType DILParser::ResolveTypeDeclarators(
    CompilerType type, const std::vector<PtrOperator>& ptr_operators)
{
  CompilerType bad_type;
  // Resolve pointers/references.
  for (auto& [tk, loc] : ptr_operators) {
    if (tk == dil::TokenKind::star) {
      // Pointers to reference types are forbidden.
      if (type.IsReferenceType()) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("'type name' declared as a pointer to a "
                              "reference of type {0}",
                              type.TypeDescription()),
                loc);
        return bad_type;
      }
      // Get pointer type for the base type: e.g. int* -> int**.
      type = type.GetPointerType();

    } else if (tk == dil::TokenKind::amp) {
      // References to references are forbidden.
      if (type.IsReferenceType()) {
        BailOut(ErrorCode::kInvalidOperandType,
                "type name declared as a reference to a reference", loc);
        return bad_type;
      }
      // Get reference type for the base type: e.g. int -> int&.
      type = type.GetLValueReferenceType();
    }
  }

  return type;
}

bool DILParser::IsSimpleTypeSpecifierKeyword(DILToken token) const {
  return token.isOneOf(
      dil::TokenKind::kw_char, dil::TokenKind::kw_char16_t, dil::TokenKind::kw_char32_t,
      dil::TokenKind::kw_wchar_t, dil::TokenKind::kw_bool, dil::TokenKind::kw_short,
      dil::TokenKind::kw_int, dil::TokenKind::kw_long, dil::TokenKind::kw_signed,
      dil::TokenKind::kw_unsigned, dil::TokenKind::kw_float, dil::TokenKind::kw_double,
      dil::TokenKind::kw_void);
}

bool DILParser::IsCvQualifier(DILToken token) const {
  return token.isOneOf(dil::TokenKind::kw_const, dil::TokenKind::kw_volatile);
}

bool DILParser::IsPtrOperator(DILToken token) const {
  return token.isOneOf(dil::TokenKind::star, dil::TokenKind::amp);
}

bool DILParser::HandleSimpleTypeSpecifier(TypeDeclaration* type_decl) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  using SignSpecifier = TypeDeclaration::SignSpecifier;

  TypeSpecifier type_spec = type_decl->m_type_specifier;
  uint32_t loc = m_dil_token.getLocation();
  dil::TokenKind kind = m_dil_token.getKind();

  switch (kind) {
    case dil::TokenKind::kw_int: {
      // "int" can have signedness and be combined with "short", "long" and
      // "long long" (but not with another "int").
      if (type_decl->m_has_int_specifier) {
        BailOut(ErrorCode::kInvalidOperandType,
                "cannot combine with previous 'int' declaration specifier",
                loc);
        return false;
      }
      if (type_spec == TypeSpecifier::kShort ||
          type_spec == TypeSpecifier::kLong ||
          type_spec == TypeSpecifier::kLongLong) {
        type_decl->m_has_int_specifier = true;
        return true;
      } else if (type_spec == TypeSpecifier::kUnknown) {
        type_decl->m_type_specifier = TypeSpecifier::kInt;
        type_decl->m_has_int_specifier = true;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case dil::TokenKind::kw_long: {
      // "long" can have signedness and be combined with "int" or "long" to
      // form "long long".
      if (type_spec == TypeSpecifier::kUnknown ||
          type_spec == TypeSpecifier::kInt) {
        type_decl->m_type_specifier = TypeSpecifier::kLong;
        return true;
      } else if (type_spec == TypeSpecifier::kLong) {
        type_decl->m_type_specifier = TypeSpecifier::kLongLong;
        return true;
      } else if (type_spec == TypeSpecifier::kDouble) {
        type_decl->m_type_specifier = TypeSpecifier::kLongDouble;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case dil::TokenKind::kw_short: {
      // "short" can have signedness and be combined with "int".
      if (type_spec == TypeSpecifier::kUnknown ||
          type_spec == TypeSpecifier::kInt) {
        type_decl->m_type_specifier = TypeSpecifier::kShort;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case dil::TokenKind::kw_char: {
      // "char" can have signedness, but it cannot be combined with any other
      // type specifier.
      if (type_spec == TypeSpecifier::kUnknown) {
        type_decl->m_type_specifier = TypeSpecifier::kChar;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case dil::TokenKind::kw_double: {
      // "double" can be combined with "long" to form "long double", but it
      // cannot be combined with signedness specifier.
      if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                "'double' cannot be signed or unsigned", loc);
        return false;
      }
      if (type_spec == TypeSpecifier::kUnknown) {
        type_decl->m_type_specifier = TypeSpecifier::kDouble;
        return true;
      } else if (type_spec == TypeSpecifier::kLong) {
        type_decl->m_type_specifier = TypeSpecifier::kLongDouble;
        return true;
      }
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  ToString(type_spec)),
              loc);
      return false;
    }

    case dil::TokenKind::kw_bool:
    case dil::TokenKind::kw_void:
    case dil::TokenKind::kw_float:
    case dil::TokenKind::kw_wchar_t:
    case dil::TokenKind::kw_char16_t:
    case dil::TokenKind::kw_char32_t: {
      // These types cannot have signedness or be combined with any other type
      // specifiers.
      if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("'{0}' cannot be signed or unsigned",
                              ToString(ToTypeSpecifier(kind))),
                loc);
        return false;
      }
      if (type_spec != TypeSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cannot combine with previous '{0}' declaration specifier",
                    ToString(type_spec)),
                loc);
      }
      type_decl->m_type_specifier = ToTypeSpecifier(kind);
      return true;
    }

    case dil::TokenKind::kw_signed:
    case dil::TokenKind::kw_unsigned: {
      // "signed" and "unsigned" cannot be combined with another signedness
      // specifier.
      if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cannot combine with previous '{0}' declaration specifier",
                    ToString(type_decl->m_sign_specifier)),
                loc);
        return false;
      }
      if (type_spec == TypeSpecifier::kVoid ||
          type_spec == TypeSpecifier::kBool ||
          type_spec == TypeSpecifier::kFloat ||
          type_spec == TypeSpecifier::kDouble ||
          type_spec == TypeSpecifier::kLongDouble ||
          type_spec == TypeSpecifier::kWChar ||
          type_spec == TypeSpecifier::kChar16 ||
          type_spec == TypeSpecifier::kChar32) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("'{0}' cannot be signed or unsigned",
                              ToString(type_spec)),
                loc);
        return false;
      }

      type_decl->m_sign_specifier = (kind == dil::TokenKind::kw_signed)
                                       ? SignSpecifier::kSigned
                                       : SignSpecifier::kUnsigned;
      return true;
    }

    default:
      assert(false && "invalid simple type specifier kind");
      return false;
  }
}

DILASTNodeUP DILParser::ParseStringLiteral() {
  ExpectOneOf(dil::TokenKind::string_literal, dil::TokenKind::wide_string_literal,
              dil::TokenKind::utf8_string_literal);
  uint32_t loc = m_dil_token.getLocation();

  // TODO: Support parsing of joined string-literals (e.g. "abc" "def").
  // Currently, only a single token can be parsed into a string.
  dil::StringLiteralParser string_literal(
      llvm::ArrayRef<DILToken>(m_dil_token), m_dil_lexer);

  if (string_literal.hadError) {
    // TODO: Use ErrorCode::kInvalidStringLiteral in the future.
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("Failed to parse token as string-literal: {0}",
                          TokenDescription(m_dil_token)),
            loc);
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto char_type = GetBasicType(m_ctx_scope, PickCharType(string_literal));
  // Strings are terminated by a null value (add +1).
  CompilerType compiler_type = char_type;
  uint64_t byte_size = 0;
  if (auto temp = compiler_type.GetByteSize(nullptr))
    byte_size = temp.value();
  uint64_t array_size = string_literal.GetStringLength() / byte_size + 1;
  auto array_type = compiler_type.GetArrayType(array_size);

  llvm::StringRef value = string_literal.GetString();
  std::string data(value.data());

  assert(data.size() == array_type.GetByteSize(nullptr) &&
         "invalid string literal: unexpected data size");
  ConsumeToken();
  return std::make_unique<StringLiteralNode>(loc, array_type, std::move(data));
}

// Parse an expression.
//
//  expression:
//    assignment_expression
//
DILASTNodeUP DILParser::ParseExpression() { return ParseAssignmentExpression(); }

// Parse an assignment_expression.
//
//  assignment_expression:
//    conditional_expression
//    logical_or_expression assignment_operator assignment_expression
//
//  assignment_operator:
//    "="
//    "*="
//    "/="
//    "%="
//    "+="
//    "-="
//    ">>="
//    "<<="
//    "&="
//    "^="
//    "|="
//
//  conditional_expression:
//    logical_or_expression
//    logical_or_expression "?" expression ":" assignment_expression
//
DILASTNodeUP DILParser::ParseAssignmentExpression() {
  auto lhs = ParseLogicalOrExpression();

  // Check if it's an assignment expression.
  if (m_dil_token.isOneOf(dil::TokenKind::equal, dil::TokenKind::starequal,
                     dil::TokenKind::slashequal, dil::TokenKind::percentequal,
                     dil::TokenKind::plusequal, dil::TokenKind::minusequal,
                     dil::TokenKind::greatergreaterequal, dil::TokenKind::lesslessequal,
                     dil::TokenKind::ampequal, dil::TokenKind::caretequal,
                     dil::TokenKind::pipeequal)) {
    // That's an assignment!
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseAssignmentExpression();
    lhs = BuildBinaryOp(dil_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  // Check if it's a conditional expression.
  if (m_dil_token.is(dil::TokenKind::question)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto true_val = ParseExpression();
    Expect(dil::TokenKind::colon);
    ConsumeToken();
    auto false_val = ParseAssignmentExpression();
    lhs = BuildTernaryOp(std::move(lhs), std::move(true_val),
                         std::move(false_val), token.getLocation());
  }

  return lhs;
}

// Parse a logical_or_expression.
//
//  logical_or_expression:
//    logical_and_expression {"||" logical_and_expression}
//
DILASTNodeUP DILParser::ParseLogicalOrExpression() {
  auto lhs = ParseLogicalAndExpression();

  while (m_dil_token.is(dil::TokenKind::pipepipe)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseLogicalAndExpression();
    lhs = BuildBinaryOp(BinaryOpKind::LOr, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse a logical_and_expression.
//
//  logical_and_expression:
//    inclusive_or_expression {"&&" inclusive_or_expression}
//
DILASTNodeUP DILParser::ParseLogicalAndExpression() {
  auto lhs = ParseInclusiveOrExpression();

  while (m_dil_token.is(dil::TokenKind::ampamp)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseInclusiveOrExpression();
    lhs = BuildBinaryOp(BinaryOpKind::LAnd, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an inclusive_or_expression.
//
//  inclusive_or_expression:
//    exclusive_or_expression {"|" exclusive_or_expression}
//
DILASTNodeUP DILParser::ParseInclusiveOrExpression() {
  auto lhs = ParseExclusiveOrExpression();

  while (m_dil_token.is(dil::TokenKind::pipe)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseExclusiveOrExpression();
    lhs = BuildBinaryOp(BinaryOpKind::Or, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an exclusive_or_expression.
//
//  exclusive_or_expression:
//    and_expression {"^" and_expression}
//
DILASTNodeUP DILParser::ParseExclusiveOrExpression() {
  auto lhs = ParseAndExpression();

  while (m_dil_token.is(dil::TokenKind::caret)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseAndExpression();
    lhs = BuildBinaryOp(BinaryOpKind::Xor, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an and_expression.
//
//  and_expression:
//    equality_expression {"&" equality_expression}
//
DILASTNodeUP DILParser::ParseAndExpression() {
  auto lhs = ParseEqualityExpression();

  while (m_dil_token.is(dil::TokenKind::amp)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseEqualityExpression();
    lhs = BuildBinaryOp(BinaryOpKind::And, std::move(lhs), std::move(rhs),
                        token.getLocation());
  }

  return lhs;
}

// Parse an equality_expression.
//
//  equality_expression:
//    relational_expression {"==" relational_expression}
//    relational_expression {"!=" relational_expression}
//
DILASTNodeUP DILParser::ParseEqualityExpression() {
  auto lhs = ParseRelationalExpression();

  while (m_dil_token.isOneOf(dil::TokenKind::equalequal, dil::TokenKind::exclaimequal)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseRelationalExpression();
    lhs = BuildBinaryOp(dil_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse a relational_expression.
//
//  relational_expression:
//    shift_expression {"<" shift_expression}
//    shift_expression {">" shift_expression}
//    shift_expression {"<=" shift_expression}
//    shift_expression {">=" shift_expression}
//
DILASTNodeUP DILParser::ParseRelationalExpression() {
  auto lhs = ParseShiftExpression();

  while (m_dil_token.isOneOf(dil::TokenKind::less, dil::TokenKind::greater,
                        dil::TokenKind::lessequal, dil::TokenKind::greaterequal)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseShiftExpression();
    lhs = BuildBinaryOp(dil_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse a shift_expression.
//
//  shift_expression:
//    additive_expression {"<<" additive_expression}
//    additive_expression {">>" additive_expression}
//
DILASTNodeUP DILParser::ParseShiftExpression() {
  auto lhs = ParseAdditiveExpression();

  while (m_dil_token.isOneOf(dil::TokenKind::lessless, dil::TokenKind::greatergreater)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseAdditiveExpression();
    lhs = BuildBinaryOp(dil_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse an additive_expression.
//
//  additive_expression:
//    multiplicative_expression {"+" multiplicative_expression}
//    multiplicative_expression {"-" multiplicative_expression}
//
DILASTNodeUP DILParser::ParseAdditiveExpression() {
  auto lhs = ParseMultiplicativeExpression();

  while (m_dil_token.isOneOf(dil::TokenKind::plus, dil::TokenKind::minus)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseMultiplicativeExpression();
    lhs = BuildBinaryOp(dil_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}

// Parse a multiplicative_expression.
//
//  multiplicative_expression:
//    cast_expression {"*" cast_expression}
//    cast_expression {"/" cast_expression}
//    cast_expression {"%" cast_expression}
//
DILASTNodeUP DILParser::ParseMultiplicativeExpression() {
  auto lhs = ParseCastExpression();

  while (m_dil_token.isOneOf(dil::TokenKind::star, dil::TokenKind::slash,
                        dil::TokenKind::percent)) {
    DILToken token = m_dil_token;
    ConsumeToken();
    auto rhs = ParseCastExpression();
    lhs = BuildBinaryOp(dil_token_kind_to_binary_op_kind(token.getKind()),
                        std::move(lhs), std::move(rhs), token.getLocation());
  }

  return lhs;
}



// Parse a cast_expression.
//
//  cast_expression:
//    unary_expression
//    "(" type_id ")" cast_expression
//
DILASTNodeUP DILParser::ParseCastExpression() {
  // This can be a C-style cast, try parsing the contents as a type declaration.
  if (m_dil_token.is(dil::TokenKind::l_paren)) {
    DILToken token = m_dil_token;

    // Enable lexer backtracking, so that we can rollback in case it's not
    // actually a type declaration.

    // Start tentative parsing (save token location/idx, for possible rollback).
    uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

    // Consume the token only after enabling the backtracking.
    ConsumeToken();

    // Try parsing the type declaration. If the returned value is not valid,
    // then we should rollback and try parsing the expression.
    auto type_id = ParseTypeId();
    if (type_id) {
      // Successfully parsed the type declaration. Commit the backtracked
      // tokens and parse the cast_expression.

      if (!type_id.value().IsValid()) {
        CompilerType bad_type;
        return std::make_unique<ErrorNode>(bad_type);
      }

      Expect(dil::TokenKind::r_paren);
      ConsumeToken();
      auto rhs = ParseCastExpression();

      return BuildCStyleCast(type_id.value(), std::move(rhs),
                             token.getLocation());
    }

    // Failed to parse the contents of the parentheses as a type declaration.
    // Rollback the lexer and try parsing it as unary_expression.

    TentativeParsingRollback(save_token_idx);
  }

  return ParseUnaryExpression();
}


// Parse an unary_expression.
//
//  unary_expression:
//    postfix_expression
//    "++" cast_expression
//    "--" cast_expression
//    unary_operator cast_expression
//    sizeof unary_expression
//    sizeof "(" type_id ")"
//
//  unary_operator:
//    "&"
//    "*"
//    "+"
//    "-"
//    "~"
//    "!"
//
DILASTNodeUP DILParser::ParseUnaryExpression() {
  if (m_dil_token.isOneOf(dil::TokenKind::plusplus, dil::TokenKind::minusminus,
                     dil::TokenKind::star, dil::TokenKind::amp, dil::TokenKind::plus,
                     dil::TokenKind::minus, dil::TokenKind::exclaim,
                     dil::TokenKind::tilde)) {
    DILToken token = m_dil_token;
    uint32_t loc = token.getLocation();
    ConsumeToken();
    auto rhs = ParseCastExpression();

    switch (token.getKind()) {
      case dil::TokenKind::plusplus:
        return BuildUnaryOp(UnaryOpKind::PreInc, std::move(rhs), loc);
      case dil::TokenKind::minusminus:
        return BuildUnaryOp(UnaryOpKind::PreDec, std::move(rhs), loc);
      case dil::TokenKind::star:
        return BuildUnaryOp(UnaryOpKind::Deref, std::move(rhs), loc);
      case dil::TokenKind::amp:
        return BuildUnaryOp(UnaryOpKind::AddrOf, std::move(rhs), loc);
      case dil::TokenKind::plus:
        return BuildUnaryOp(UnaryOpKind::Plus, std::move(rhs), loc);
      case dil::TokenKind::minus:
        return BuildUnaryOp(UnaryOpKind::Minus, std::move(rhs), loc);
      case dil::TokenKind::tilde:
        return BuildUnaryOp(UnaryOpKind::Not, std::move(rhs), loc);
      case dil::TokenKind::exclaim:
        return BuildUnaryOp(UnaryOpKind::LNot, std::move(rhs), loc);

      default:
        llvm_unreachable("invalid token kind");
    }
  }

  if (m_dil_token.is(dil::TokenKind::kw_sizeof)) {
    uint32_t sizeof_loc = m_dil_token.getLocation();
    ConsumeToken();

    // [expr.sizeof](http://eel.is/c++draft/expr.sizeof#1)
    //
    // The operand is either an expression, which is an unevaluated operand,
    // or a parenthesized type-id.

    // Either operand itself (if it's a type_id), or an operand return type
    // (if it's an expression).
    CompilerType operand;

    // `(` can mean either a type_id or a parenthesized expression.
    if (m_dil_token.is(dil::TokenKind::l_paren)) {
      // Start tentative parsing (save token location/idx, for possible
      // rollback).
      uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

      Expect(dil::TokenKind::l_paren);
      ConsumeToken();

      // Parse the type definition and resolve the type.
      auto type_id = ParseTypeId();
      if (type_id) {
        // type_id requires parentheses, so there must be a closing one.
        Expect(dil::TokenKind::r_paren);
        ConsumeToken();

        operand = type_id.value();

      } else {
        TentativeParsingRollback(save_token_idx);

        // Failed to parse type_id, fallback to parsing an unary_expression.
        operand = ParseUnaryExpression()->GetDereferencedResultType();
      }

    } else {
      // No opening parenthesis means this must be an unary_expression.
      operand = ParseUnaryExpression()->GetDereferencedResultType();
    }
    if (!operand.IsValid()) {
      CompilerType bad_type;
      return std::make_unique<ErrorNode>(bad_type);
    }

    lldb::BasicType size_type;
    llvm::Triple triple(llvm::Twine(
        m_ctx_scope->CalculateTarget()->GetArchitecture().GetTriple().str()));
    if (triple.isOSWindows()) {
      size_type = triple.isArch64Bit() ? lldb::eBasicTypeUnsignedLongLong
          : lldb::eBasicTypeUnsignedInt;
    } else {
      size_type = triple.isArch64Bit() ? lldb::eBasicTypeUnsignedLong
          : lldb::eBasicTypeUnsignedInt;
  }

    auto result_type = GetBasicType(m_ctx_scope, size_type);
    return std::make_unique<SizeOfNode>(sizeof_loc, result_type, operand);
  }

  return ParsePostfixExpression();
}

// Parse a postfix_expression.
//
//  postfix_expression:
//    primary_expression
//    postfix_expression "[" expression "]"
//    postfix_expression "." id_expression
//    postfix_expression "->" id_expression
//    postfix_expression "++"
//    postfix_expression "--"
//    static_cast "<" type_id ">" "(" expression ")" ;
//    dynamic_cast "<" type_id ">" "(" expression ")" ;
//    reinterpret_cast "<" type_id ">" "(" expression ")" ;
//
DILASTNodeUP DILParser::ParsePostfixExpression() {
  // Parse the first part of the postfix_expression. This could be either a
  // primary_expression, or a postfix_expression itself.
  DILASTNodeUP lhs;
  CompilerType bad_type;

  // C++-style cast.
  if (m_dil_token.isOneOf(dil::TokenKind::kw_static_cast, dil::TokenKind::kw_dynamic_cast,
                     dil::TokenKind::kw_reinterpret_cast)) {
    dil::TokenKind cast_kind = m_dil_token.getKind();
    uint32_t cast_loc = m_dil_token.getLocation();
    ConsumeToken();

    Expect(dil::TokenKind::less);
    ConsumeToken();

    uint32_t loc = m_dil_token.getLocation();

    // Parse the type definition and resolve the type.
    auto type_id = ParseTypeId(/*must_be_type_id*/ true);
    if (!type_id) {
      BailOut(ErrorCode::kInvalidOperandType,
              "type name requires a specifier or qualifier", loc);
      return std::make_unique<ErrorNode>(bad_type);
    }
    if (!type_id.value().IsValid()) {
      return std::make_unique<ErrorNode>(bad_type);
    }

    Expect(dil::TokenKind::greater);
    ConsumeToken();

    Expect(dil::TokenKind::l_paren);
    ConsumeToken();
    auto rhs = ParseExpression();
    Expect(dil::TokenKind::r_paren);
    ConsumeToken();

    lhs = BuildCxxCast(cast_kind, type_id.value(), std::move(rhs), cast_loc);

  } else {
    // Otherwise it's a primary_expression.
    lhs = ParsePrimaryExpression();
  }
  assert(lhs && "LHS of the postfix_expression can't be NULL.");

  while (m_dil_token.isOneOf(dil::TokenKind::l_square, dil::TokenKind::period,
                        dil::TokenKind::arrow, dil::TokenKind::plusplus,
                        dil::TokenKind::minusminus)) {
    DILToken token = m_dil_token;
    switch (token.getKind()) {
      case dil::TokenKind::period:
      case dil::TokenKind::arrow: {
        ConsumeToken();
        DILToken member_token = m_dil_token;
        auto member_id = ParseIdExpression();
        // Check if this is a function call.
        if (m_dil_token.is(dil::TokenKind::l_paren)) {
          // TODO: Check if `member_id` is actually a member function of `lhs`.
          // If not, produce a more accurate diagnostic.
          BailOut(ErrorCode::kNotImplemented,
                  "member function calls are not supported",
                  m_dil_token.getLocation());
        }
        lhs = BuildMemberOf(std::move(lhs), std::move(member_id),
                            token.getKind() == dil::TokenKind::arrow,
                            member_token.getLocation());
        break;
      }
      case dil::TokenKind::plusplus: {
        ConsumeToken();
        return BuildUnaryOp(UnaryOpKind::PostInc, std::move(lhs),
                            token.getLocation());
      }
      case dil::TokenKind::minusminus: {
        ConsumeToken();
        return BuildUnaryOp(UnaryOpKind::PostDec, std::move(lhs),
                            token.getLocation());
      }
      case dil::TokenKind::l_square: {
        ConsumeToken();
        auto rhs = ParseExpression();
        Expect(dil::TokenKind::r_square);
        ConsumeToken();
        lhs = BuildBinarySubscript(std::move(lhs), std::move(rhs),
                                   token.getLocation());
        break;
      }

      default:
        llvm_unreachable("invalid token");
    }
  }

  return lhs;
}

// Parse a primary_expression.
//
//  primary_expression:
//    numeric_literal
//    boolean_literal
//    pointer_literal
//    id_expression
//    "this"
//    "(" expression ")"
//    builtin_func
//
DILASTNodeUP DILParser::ParsePrimaryExpression() {
  CompilerType bad_type;
  if (m_dil_token.is(dil::TokenKind::numeric_constant)) {
    return ParseNumericLiteral();
  } else if (m_dil_token.isOneOf(dil::TokenKind::kw_true, dil::TokenKind::kw_false)) {
    return ParseBooleanLiteral();
  } else if (m_dil_token.isOneOf(dil::TokenKind::char_constant,
                            dil::TokenKind::wide_char_constant,
                            dil::TokenKind::utf8_char_constant)) {
    return ParseCharLiteral();
  } else if (m_dil_lexer.isStringLiteral(m_dil_token.getKind())) {
    // Note: Only expressions that consist of a single string literal can be
    // handled by DIL.
    BailOut(ErrorCode::kNotImplemented, "string literals are not supported",
            m_dil_token.getLocation());
    return std::make_unique<ErrorNode>(bad_type);
  } else if (m_dil_token.is(dil::TokenKind::kw_nullptr)) {
    return ParsePointerLiteral();
  } else if (m_dil_token.isOneOf(dil::TokenKind::coloncolon, dil::TokenKind::identifier)) {
    // Save the source location for the diagnostics message.
    uint32_t loc = m_dil_token.getLocation();
    auto identifier = ParseIdExpression();
    // Check if this is a function call.
    if (m_dil_token.is(dil::TokenKind::l_paren)) {
      auto func_def = GetBuiltinFunctionDef(m_ctx_scope, identifier);
      if (!func_def) {
        BailOut(
            ErrorCode::kNotImplemented,
            llvm::formatv("function '{0}' is not a supported builtin intrinsic",
                          identifier),
            loc);
        return std::make_unique<ErrorNode>(bad_type);
      }
      return ParseBuiltinFunction(loc, std::move(func_def));
    }
    // Otherwise look for an identifier.
    // TODO: Handle bitfield identifiers when evaluating in the value context.
    auto value = LookupIdentifier(identifier, m_ctx_scope, m_use_dynamic,
                                  nullptr);
    if (!value) {
      BailOut(ErrorCode::kUndeclaredIdentifier,
              llvm::formatv("use of undeclared identifier '{0}'", identifier),
              loc);
      return std::make_unique<ErrorNode>(bad_type);
    }
    return std::make_unique<IdentifierNode>(loc, identifier, std::move(value),
                                            /*is_rvalue*/ false,
                                            IsContextVar(identifier));
  } else if (m_dil_token.is(dil::TokenKind::kw_this)) {
    // Save the source location for the diagnostics message.
    uint32_t loc = m_dil_token.getLocation();
    ConsumeToken();
    auto value = LookupIdentifier("this", m_ctx_scope, m_use_dynamic);
    if (!value) {
      BailOut(ErrorCode::kUndeclaredIdentifier,
              "invalid use of 'this' outside of a non-static member function",
              loc);
      return std::make_unique<ErrorNode>(bad_type);
    }
    // Special case for "this" pointer. As per C++ standard, it's a prvalue.
    return std::make_unique<IdentifierNode>(loc, "this", std::move(value),
                                            /*is_rvalue*/ true,
                                            /*is_context_var*/ false);
  } else if (m_dil_token.is(dil::TokenKind::l_paren)) {
    // Check in case this is an anonynmous namespace
    if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::identifier)
        && (((DILToken)m_dil_lexer.LookAhead(0)).getSpelling() == "anonymous")
        && m_dil_lexer.LookAhead(1).is(dil::TokenKind::kw_namespace)
        && m_dil_lexer.LookAhead(2).is(dil::TokenKind::r_paren)
        && m_dil_lexer.LookAhead(3).is(dil::TokenKind::coloncolon)) {
      m_dil_token = m_dil_lexer.AcceptLookAhead(3);
      std::string identifier = "(anonymous namespace)";
      Expect(dil::TokenKind::coloncolon);
      // Save the source location for the diagnostics message.
      uint32_t loc = m_dil_token.getLocation();
      ConsumeToken();
      assert ((m_dil_token.is(dil::TokenKind::identifier) ||
               m_dil_token.is(dil::TokenKind::l_paren)) &&
              "Expected an identifier or anonymous namespeace, but not found.");
      std::string identifier2 = ParseNestedNameSpecifier();
      if (identifier2.empty()) {
        // There was only an identifer, no more levels of nesting. Or there
        // was an invalid expression starting with a left parenthesis.
        Expect(dil::TokenKind::identifier);
        identifier2 = m_dil_token.getSpelling();
        ConsumeToken();
      }
      identifier = identifier + "::" + identifier2;
      auto value = LookupIdentifier(identifier, m_ctx_scope, m_use_dynamic);
      if (!value) {
        BailOut(ErrorCode::kUndeclaredIdentifier,
                llvm::formatv("use of undeclared identifier '{0}'", identifier),
                loc);
        return std::make_unique<ErrorNode>(bad_type);
      }
      return std::make_unique<IdentifierNode>(loc, identifier, std::move(value),
                                              /*is_rvalue*/ false,
                                              IsContextVar(identifier));
    } else {
      ConsumeToken();
      auto expr = ParseExpression();
      Expect(dil::TokenKind::r_paren);
      ConsumeToken();
      return expr;
    }
  }

  BailOut(ErrorCode::kInvalidExpressionSyntax,
          llvm::formatv("Unexpected token: {0}", TokenDescription(m_dil_token)),
          m_dil_token.getLocation());
  return std::make_unique<ErrorNode>(bad_type);
}

// Parse a type_id.
//
//  type_id:
//    type_specifier_seq [abstract_declarator]
//
std::optional<CompilerType> DILParser::ParseTypeId(bool must_be_type_id) {
  uint32_t type_loc = m_dil_token.getLocation();
  TypeDeclaration type_decl;
  CompilerType bad_type;

  // type_specifier_seq is required here, start with trying to parse it.
  ParseTypeSpecifierSeq(&type_decl);

  if (type_decl.IsEmpty()) {
    // TODO: Should we bail out if `must_be_type_id` is set?
    return {};
  }

  if (type_decl.m_has_error) {
    if (type_decl.m_is_builtin) {
      return bad_type;
    }

    assert(type_decl.m_is_user_type && "type_decl must be a user type");
    // Found something looking like a user type, but failed to parse it.
    // Return invalid type if we expect to have a type here, otherwise nullopt.
    if (must_be_type_id) {
      return bad_type;
    }
    return {};
  }

  // Try to resolve the base type.
  CompilerType type;
  if (type_decl.m_is_builtin) {
    type = GetBasicType(m_ctx_scope, type_decl.GetBasicType());
    assert(type.IsValid() && "cannot resolve basic type");

  } else {
    assert(type_decl.m_is_user_type && "type_decl must be a user type");
    type = ResolveTypeByName(type_decl.m_user_typename, m_ctx_scope);
    if (!type.IsValid()) {
      if (must_be_type_id) {
        BailOut(
            ErrorCode::kUndeclaredIdentifier,
            llvm::formatv("unknown type name '{0}'",
                          type_decl.m_user_typename),
            type_loc);
        return bad_type;
      }
      return {};
    }

    if (LookupIdentifier(type_decl.m_user_typename, m_ctx_scope,
                         m_use_dynamic)) {
      // Same-name identifiers should be preferred over typenames.
      // TODO: Make type accessible with 'class', 'struct' and 'union' keywords.
      if (must_be_type_id) {
        BailOut(ErrorCode::kUndeclaredIdentifier,
                llvm::formatv(
                    "must use '{0}' tag to refer to type '{1}' in this scope",
                    type.GetTypeTag(), type_decl.m_user_typename),
                type_loc);
        return bad_type;
      }
      return {};
    }
  }

  //
  //  abstract_declarator:
  //    ptr_operator [abstract_declarator]
  //
  std::vector<DILParser::PtrOperator> ptr_operators;
  while (IsPtrOperator(m_dil_token)) {
    ptr_operators.push_back(ParsePtrOperator());
  }
  type = ResolveTypeDeclarators(type, ptr_operators);

  return type;
}

// Parse a type_specifier_seq.
//
//  type_specifier_seq:
//    type_specifier [type_specifier_seq]
//
void DILParser::ParseTypeSpecifierSeq(TypeDeclaration* type_decl) {
  while (true) {
    bool type_specifier = ParseTypeSpecifier(type_decl);
    if (!type_specifier) {
      break;
    }
  }
}

// Parse a type_specifier.
//
//  type_specifier:
//    simple_type_specifier
//    cv_qualifier
//
//  simple_type_specifier:
//    ["::"] [nested_name_specifier] type_name
//    "char"
//    "char16_t"
//    "char32_t"
//    "wchar_t"
//    "bool"
//    "short"
//    "int"
//    "long"
//    "signed"
//    "unsigned"
//    "float"
//    "double"
//    "void"
//
// Returns TRUE if a type_specifier was successfully parsed at this location.
//
bool DILParser::ParseTypeSpecifier(TypeDeclaration* type_decl) {
  if (IsCvQualifier(m_dil_token)) {
    // Just ignore CV quialifiers, we don't use them in type casting.
    ConsumeToken();
    return true;
  }

  if (IsSimpleTypeSpecifierKeyword(m_dil_token)) {
    // User-defined typenames can't be combined with builtin keywords.
    if (type_decl->m_is_user_type) {
      BailOut(ErrorCode::kInvalidOperandType,
              "cannot combine with previous declaration specifier",
              m_dil_token.getLocation());
      type_decl->m_has_error = true;
      return false;
    }

    // From now on this type declaration must describe a builtin type.
    // TODO: Should this be allowed -- `unsigned myint`?
    type_decl->m_is_builtin = true;

    if (!HandleSimpleTypeSpecifier(type_decl)) {
      type_decl->m_has_error = true;
      return false;
    }
    ConsumeToken();
    return true;
  }

  // The type_specifier must be a user-defined type. Try parsing a
  // simple_type_specifier.
  {
    // Try parsing optional global scope operator.
    bool global_scope = false;
    if (m_dil_token.is(dil::TokenKind::coloncolon)) {
      global_scope = true;
      ConsumeToken();
    }

    uint32_t loc = m_dil_token.getLocation();

    // Try parsing optional nested_name_specifier.
    auto nested_name_specifier = ParseNestedNameSpecifier();

    // Try parsing required type_name.
    auto type_name = ParseTypeName();

    // If there is a type_name, then this is indeed a simple_type_specifier.
    // Global and qualified (namespace/class) scopes can be empty, since they're
    // optional. In this case type_name is type we're looking for.
    if (!type_name.empty()) {
      // User-defined typenames can't be combined with builtin keywords.
      if (type_decl->m_is_builtin) {
        BailOut(ErrorCode::kInvalidOperandType,
                "cannot combine with previous declaration specifier", loc);
        type_decl->m_has_error = true;
        return false;
      }
      // There should be only one user-defined typename.
      if (type_decl->m_is_user_type) {
        BailOut(ErrorCode::kInvalidOperandType,
                "two or more data types in declaration of 'type name'", loc);
        type_decl->m_has_error = true;
        return false;
      }

      // Construct the fully qualified typename.
      type_decl->m_is_user_type = true;
      type_decl->m_user_typename =
          llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                        nested_name_specifier, type_name);
      return true;
    }
  }

  // No type_specifier was found here.
  return false;
}

// Parse nested_name_specifier.
//
//  nested_name_specifier:
//    type_name "::"
//    namespace_name '::'
//    nested_name_specifier identifier "::"
//    nested_name_specifier simple_template_id "::"
//
std::string DILParser::ParseNestedNameSpecifier() {
  // The first token in nested_name_specifier is always an identifier, or
  // '(anonymous namespace)'.
  if (m_dil_token.isNot(dil::TokenKind::identifier) &&
      m_dil_token.isNot(dil::TokenKind::l_paren)) {
    return "";
  }

  // Anonymous namespaces need to be treated specially: They are represented
  // the the string '(anonymous namespace)', which has a space in it (throwing
  // off normal parsing) and is not actually proper C++> Check to see if we're
  // looking at '(anonymous namespace)::...'
  if (m_dil_token.is(dil::TokenKind::l_paren)) {
    // Look for all the pieces, in order:
    // l_paren 'anonymous' 'namespace' r_paren coloncolon
    if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::identifier)
        && (((DILToken)m_dil_lexer.LookAhead(0)).getSpelling() == "anonymous")
        && m_dil_lexer.LookAhead(1).is(dil::TokenKind::kw_namespace)
        && m_dil_lexer.LookAhead(2).is(dil::TokenKind::r_paren)
        && m_dil_lexer.LookAhead(3).is(dil::TokenKind::coloncolon)) {
      m_dil_token = m_dil_lexer.AcceptLookAhead(3);

      assert ((m_dil_token.is(dil::TokenKind::identifier)
               || m_dil_token.is(dil::TokenKind::l_paren)) &&
              "Expected an identifier or anonymous namespace, but not found.");
      // Continue parsing the nested_namespace_specifier.
      std::string identifier2 = ParseNestedNameSpecifier();
      if (identifier2.empty()) {
        Expect(dil::TokenKind::identifier);
        identifier2 = m_dil_token.getSpelling();
        ConsumeToken();
      }
      return "(anonymous namespace)::" + identifier2;
    } else {
      return "";
    }
  } // end of special handling for '(anonymous namespace)'

  // If the next token is scope ("::"), then this is indeed a
  // nested_name_specifier
  if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::coloncolon)) {
    // This nested_name_specifier is a single identifier.
    std::string identifier = m_dil_token.getSpelling();
    m_dil_token = m_dil_lexer.AcceptLookAhead(0);
    Expect(dil::TokenKind::coloncolon);
    ConsumeToken();
    // Continue parsing the nested_name_specifier.
    return identifier + "::" + ParseNestedNameSpecifier();
  }

  // If the next token starts a template argument list, then we have a
  // simple_template_id here.
  if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::less)) {
    // We don't know whether this will be a nested_name_identifier or just a
    // type_name. Prepare to rollback if this is not a nested_name_identifier.

    // Start tentative parsing (save token location/idx, for possible rollback).
    uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

    // TODO: Parse just the simple_template_id?
    auto type_name = ParseTypeName();

    // If we did parse the type_name successfully and it's followed by the scope
    // operator ("::"), then this is indeed a nested_name_specifier. Continue
    // parsing nested_name_specifier.
    if (!type_name.empty() && m_dil_token.is(dil::TokenKind::coloncolon)) {
      m_dil_lexer.IncrementTokenIdx();
      m_dil_token = m_dil_lexer.GetCurrentToken();
      // Continue parsing the nested_name_specifier.
      return type_name + "::" + ParseNestedNameSpecifier();
    }

    // Not a nested_name_specifier, but could be just a type_name or something
    // else entirely. Rollback the parser and try a different path.

    TentativeParsingRollback(save_token_idx);
  }

  return "";
}

// Parse a type_name.
//
//  type_name:
//    class_name
//    enum_name
//    typedef_name
//    simple_template_id
//
//  class_name
//    identifier
//
//  enum_name
//    identifier
//
//  typedef_name
//    identifier
//
//  simple_template_id:
//    template_name "<" [template_argument_list] ">"
//
std::string DILParser::ParseTypeName() {
  // Typename always starts with an identifier.
  if (m_dil_token.isNot(dil::TokenKind::identifier)) {
    return "";
  }

  // If the next token starts a template argument list, parse this type_name as
  // a simple_template_id.
  if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::less)) {
    // Parse the template_name. In this case it's just an identifier.
    std::string template_name = m_dil_token.getSpelling();
    m_dil_token = m_dil_lexer.AcceptLookAhead(0);
    // Consume the "<" token.
    ConsumeToken();

    // Short-circuit for missing template_argument_list.
    if (m_dil_token.is(dil::TokenKind::greater)) {
      ConsumeToken();
      return llvm::formatv("{0}<>", template_name);
    }

    // Try parsing template_argument_list.
    auto template_argument_list = ParseTemplateArgumentList();

    if (m_dil_token.is(dil::TokenKind::greater)) {
      // Single closing angle bracket is a valid end of the template argument
      // list, just consume it.
      ConsumeToken();

    } else if (m_dil_token.is(dil::TokenKind::greatergreater)) {
      // C++11 allows using ">>" in nested template argument lists and C++-style
      // casts. In this case we alter change the token type to ">", but don't
      // consume it -- it will be done on the outer level when completing the
      // outer template argument list or C++-style cast.
      m_dil_token.setKind(dil::TokenKind::greater);

    } else {
      // Not a valid end of the template argument list, failed to parse a
      // simple_template_id
      return "";
    }

    return llvm::formatv("{0}<{1}>", template_name, template_argument_list);
  }

  // Otherwise look for a class_name, enum_name or a typedef_name.
  std::string identifier = m_dil_token.getSpelling();
  ConsumeToken();

  return identifier;
}

// Parse a template_argument_list.
//
//  template_argument_list:
//    template_argument
//    template_argument_list "," template_argument
//
std::string DILParser::ParseTemplateArgumentList() {
  // Parse template arguments one by one.
  std::vector<std::string> arguments;

  do {
    // Eat the comma if this is not the first iteration.
    if (arguments.size() > 0) {
      ConsumeToken();
    }

    // Try parsing a template_argument. If this fails, then this is actually not
    // a template_argument_list.
    auto argument = ParseTemplateArgument();
    if (argument.empty()) {
      return "";
    }

    arguments.push_back(argument);

  } while (m_dil_token.is(dil::TokenKind::comma));

  // Internally in LLDB/Clang nested template type names have extra spaces to
  // avoid having ">>". Add the extra space before the closing ">" if the
  // template argument is also a template.
  if (arguments.back().back() == '>') {
    arguments.back().push_back(' ');
  }

  return llvm::formatv("{0:$[, ]}",
                       llvm::make_range(arguments.begin(), arguments.end()));
}

// Parse a template_argument.
//
//  template_argument:
//    type_id
//    numeric_literal
//    id_expression
//
std::string DILParser::ParseTemplateArgument() {
  // There is no way to know at this point whether there is going to be a
  // type_id or something else. Try different options one by one.

  {
    // [temp.arg](http://eel.is/c++draft/temp.arg#2)
    //
    // In a template-argument, an ambiguity between a type-id and an expression
    // is resolved to a type-id, regardless of the form of the corresponding
    // template-parameter.

    // Therefore, first try parsing type_id.
    // Start tentative parsing (save token location/idx, for possible rollback).
    uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

    auto type_id = ParseTypeId();
    if (type_id) {

      CompilerType type = type_id.value();
      return type.IsValid()
          ? std::string(type.GetTypeName().AsCString())
          : "";

    } else {
      // Failed to parse a type_id. Rollback the parser and try something else.
      TentativeParsingRollback(save_token_idx);
    }
  }

  {
    // The next candidate is a numeric_literal.
    // Start tentative parsing (save token location/idx, for possible rollback).
    uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

    // Parse a numeric_literal.
    if (m_dil_token.is(dil::TokenKind::numeric_constant)) {
      // TODO: Actually parse the literal, check if it's valid and
      // canonize it (e.g. 8LL -> 8).
      std::string numeric_literal = m_dil_token.getSpelling();
      ConsumeToken();

      if (TokenEndsTemplateArgumentList(m_dil_token)) {
        return numeric_literal;
      }
    }

    // Failed to parse a numeric_literal.
    TentativeParsingRollback(save_token_idx);
  }

  {
    // The next candidate is an id_expression.
    // Start tentative parsing (save token location/idx, for possible rollback).
    uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

    // Parse an id_expression.
    auto id_expression = ParseIdExpression();

    // If we've parsed the id_expression successfully and the next token can
    // finish the template_argument, then we're done here.
    if (!id_expression.empty() && TokenEndsTemplateArgumentList(m_dil_token)) {
      return id_expression;
    }
    // Failed to parse a id_expression.
    TentativeParsingRollback(save_token_idx);
  }

  // TODO: Another valid option here is a constant_expression, but
  // we definitely don't want to support constant arithmetic like "Foo<1+2>".
  // We can probably use ParsePrimaryExpression here, but need to figure out the
  // "stringification", since ParsePrimaryExpression returns DILASTNodeUP (and
  // potentially a whole expression, not just a single constant.)

  // This is not a template_argument.
  return "";
}

// Parse a ptr_operator.
//
//  ptr_operator:
//    "*" [cv_qualifier_seq]
//    "&"
//
DILParser::PtrOperator DILParser::ParsePtrOperator() {
  ExpectOneOf(dil::TokenKind::star, dil::TokenKind::amp);

  PtrOperator ptr_operator;
  if (m_dil_token.is(dil::TokenKind::star)) {
    ptr_operator = std::make_tuple(dil::TokenKind::star, m_dil_token.getLocation());
    ConsumeToken();

    //
    //  cv_qualifier_seq:
    //    cv_qualifier [cv_qualifier_seq]
    //
    //  cv_qualifier:
    //    "const"
    //    "volatile"
    //
    while (IsCvQualifier(m_dil_token)) {
      // Just ignore CV quialifiers, we don't use them in type casting.
      ConsumeToken();
    }

  } else if (m_dil_token.is(dil::TokenKind::amp)) {
    ptr_operator = std::make_tuple(dil::TokenKind::amp, m_dil_token.getLocation());
    ConsumeToken();
  }

  return ptr_operator;
}

// Parse an id_expression.
//
//  id_expression:
//    unqualified_id
//    qualified_id
//
//  qualified_id:
//    ["::"] [nested_name_specifier] unqualified_id
//    ["::"] identifier
//
//  identifier:
//    ? dil::TokenKind::identifier ?
//
std::string DILParser::ParseIdExpression() {
  // Try parsing optional global scope operator.
  bool global_scope = false;
  if (m_dil_token.is(dil::TokenKind::coloncolon)) {
    global_scope = true;
    ConsumeToken();
  }

  // Try parsing optional nested_name_specifier.
  auto nested_name_specifier = ParseNestedNameSpecifier();

  // If nested_name_specifier is present, then it's qualified_id production.
  // Follow the first production rule.
  if (!nested_name_specifier.empty()) {
    // Parse unqualified_id and construct a fully qualified id expression.
    auto unqualified_id = ParseUnqualifiedId();

    return llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                         nested_name_specifier, unqualified_id);
  }

  // No nested_name_specifier, but with global scope -- this is also a
  // qualified_id production. Follow the second production rule.
  else if (global_scope) {
    Expect(dil::TokenKind::identifier);
    std::string identifier = m_dil_token.getSpelling();
    ConsumeToken();
    return llvm::formatv("{0}{1}", global_scope ? "::" : "", identifier);
  }

  // This is unqualified_id production.
  return ParseUnqualifiedId();
}

// Parse an unqualified_id.
//
//  unqualified_id:
//    identifier
//
//  identifier:
//    ? dil::TokenKind::identifier ?
//
std::string DILParser::ParseUnqualifiedId() {
  Expect(dil::TokenKind::identifier);
  std::string identifier = m_dil_token.getSpelling();
  ConsumeToken();
  return identifier;
}

// Parse a numeric_literal.
//
//  numeric_literal:
//    ? dil::TokenKind::numeric_constant ?
//
DILASTNodeUP DILParser::ParseNumericLiteral() {
  Expect(dil::TokenKind::numeric_constant);
  DILASTNodeUP numeric_constant = ParseNumericConstant();
  ConsumeToken();
  return numeric_constant;
}

// Parse an boolean_literal.
//
//  boolean_literal:
//    "true"
//    "false"
//
DILASTNodeUP DILParser::ParseBooleanLiteral() {
  ExpectOneOf(dil::TokenKind::kw_true, dil::TokenKind::kw_false);
  uint32_t loc = m_dil_token.getLocation();
  bool literal_value = m_dil_token.is(dil::TokenKind::kw_true);
  ConsumeToken();
  Scalar scalar_value(static_cast<int>(literal_value));
  return std::make_unique<ScalarLiteralNode>(
      loc, GetBasicType(m_ctx_scope, lldb::eBasicTypeBool), scalar_value);
}

DILASTNodeUP DILParser::ParseCharLiteral() {
  ExpectOneOf(dil::TokenKind::char_constant,
              dil::TokenKind::wide_char_constant,
              dil::TokenKind::utf8_char_constant);
  uint32_t loc = m_dil_token.getLocation();

  std::string token_spelling = m_dil_token.getSpelling();

  const char* token_begin = token_spelling.c_str();
  dil::CharLiteralParser char_literal(token_begin,
                                      token_begin + token_spelling.size(),
                                      loc, m_dil_lexer, m_dil_token.getKind());

  if (char_literal.hadError()) {
    // TODO: Add new ErrorCode kInvalidCharLiteral and use it
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("Failed to parse token as char-constant: {0}",
                          TokenDescription(m_dil_token)),
            m_dil_token.getLocation());
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto ctx_basic_type = GetBasicType(m_ctx_scope, PickCharType(char_literal));
  uint64_t byte_size = 0;
  if (auto temp = ctx_basic_type.GetByteSize(nullptr))
    byte_size = temp.value();
  llvm::APInt literal_value(byte_size * CHAR_BIT,
                            char_literal.getValue());

  ConsumeToken();
  Scalar scalar_value(literal_value);
  return std::make_unique<ScalarLiteralNode>(loc, ctx_basic_type, scalar_value);
}

// Parse an pointer_literal.
//
//  pointer_literal:
//    "nullptr"
//
DILASTNodeUP DILParser::ParsePointerLiteral() {
  Expect(dil::TokenKind::kw_nullptr);
  uint32_t loc = m_dil_token.getLocation();
  ConsumeToken();
  llvm::APInt raw_value(type_width<uintmax_t>(), 0);
  Scalar scalar_value(raw_value);
  return std::make_unique<ScalarLiteralNode>(
      loc, GetBasicType(m_ctx_scope, lldb::eBasicTypeNullPtr), scalar_value);
}

DILASTNodeUP DILParser::ParseNumericConstant() {
  CompilerType bad_type;
  // Parse numeric constant, it can be either integer or float.
  std::string tok_spelling = m_dil_token.getSpelling();
  llvm::StringRef tok_spelling_ref(tok_spelling);

  lldb::TargetSP target_sp = m_ctx_scope->CalculateTarget();
  dil::NumericLiteralParser literal(
      tok_spelling_ref, m_dil_token.getLocation(), /*AllowHalfType=*/true,
      m_dil_lexer,
      /*AllowMicrosoftExt=*/true);

  if (literal.hadError) {
    BailOut(
        ErrorCode::kInvalidNumericLiteral,
        "Failed to parse token as numeric-constant: " +
        TokenDescription(m_dil_token),
        m_dil_token.getLocation());
    return std::make_unique<ErrorNode>(bad_type);
  }

  // Check for floating-literal and integer-literal. Fail on anything else (i.e.
  // fixed-point literal, who needs them anyway??).
  if (literal.isFloatingLiteral()) {
    return ParseFloatingLiteral(literal, m_dil_token);
  }
  if (literal.isIntegerLiteral()) {
    return ParseIntegerLiteral(literal, m_dil_token);
  }

  // Don't care about anything else.
  BailOut(ErrorCode::kInvalidNumericLiteral,
          "numeric-constant should be either float or integer literal: " +
              TokenDescription(m_dil_token),
          m_dil_token.getLocation());
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::ParseFloatingLiteral(
    dil::NumericLiteralParser& literal,
    DILToken& token) {
  const llvm::fltSemantics& format = literal.isFloat
                                         ? llvm::APFloat::IEEEsingle()
                                         : llvm::APFloat::IEEEdouble();
  llvm::APFloat raw_value(format);
  llvm::RoundingMode rm = llvm::RoundingMode::NearestTiesToEven;
  llvm::APFloat::opStatus result = literal.GetFloatValue(raw_value, rm);

  // Overflow is always an error, but underflow is only an error if we
  // underflowed to zero (APFloat reports denormals as underflow).
  if ((result & llvm::APFloat::opOverflow) ||
      ((result & llvm::APFloat::opUnderflow) && raw_value.isZero())) {
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("float underflow/overflow happened: {0}",
                          TokenDescription(token)),
            token.getLocation());
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto basic_type =
      literal.isFloat ? lldb::eBasicTypeFloat : lldb::eBasicTypeDouble;
  Scalar scalar_value(raw_value);
  return std::make_unique<ScalarLiteralNode>(
      token.getLocation(), GetBasicType(m_ctx_scope, basic_type), scalar_value);
}

DILASTNodeUP DILParser::ParseIntegerLiteral(dil::NumericLiteralParser& literal,
                                            DILToken& token) {
  // Create a value big enough to fit all valid numbers.
  llvm::APInt raw_value(type_width<uintmax_t>(), 0);

  if (literal.GetIntegerValue(raw_value)) {
    BailOut(ErrorCode::kInvalidNumericLiteral,
            llvm::formatv("integer literal is too large to be represented in "
                          "any integer type: {0}",
                          TokenDescription(token)),
            token.getLocation());
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto [type, is_unsigned] = PickIntegerType(m_ctx_scope, literal, raw_value);

  Scalar scalar_value(raw_value);
  return std::make_unique<ScalarLiteralNode>(token.getLocation(),
                                             GetBasicType(m_ctx_scope, type),
                                             scalar_value);
}

// Parse a builtin_func.
//
//  builtin_func:
//    builtin_func_name "(" [builtin_func_argument_list] ")"
//
//  builtin_func_name:
//    "__log2"
//
//  builtin_func_argument_list:
//    builtin_func_argument
//    builtin_func_argument_list "," builtin_func_argument
//
//  builtin_func_argument:
//    expression
//
DILASTNodeUP DILParser::ParseBuiltinFunction(
    uint32_t loc, std::unique_ptr<BuiltinFunctionDef> func_def) {
  Expect(dil::TokenKind::l_paren);
  ConsumeToken();

  std::vector<DILASTNodeUP> arguments;
  CompilerType bad_type;

  if (m_dil_token.is(dil::TokenKind::r_paren)) {
    // Empty argument list, nothing to do here.
    ConsumeToken();
  } else {
    // Non-empty argument list, parse all the arguments.
    do {
      // Eat the comma if this is not the first iteration.
      if (arguments.size() > 0) {
        ConsumeToken();
      }

      // Parse a builtin_func_argument. If failed to parse, bail out early and
      // don't try parsing the rest of the arguments.
      auto argument = ParseExpression();
      if (llvm::isa<ErrorNode>(argument)) {
        return std::make_unique<ErrorNode>(bad_type);
      }

      arguments.push_back(std::move(argument));

    } while (m_dil_token.is(dil::TokenKind::comma));

    Expect(dil::TokenKind::r_paren);
    ConsumeToken();
  }

  // Check we have the correct number of arguments.
  if (arguments.size() != func_def->m_arguments.size()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv(
                "no matching function for call to '{0}': requires {1} "
                "argument(s), but {2} argument(s) were provided",
                func_def->m_name, func_def->m_arguments.size(),
                arguments.size()),
            loc);
    return std::make_unique<ErrorNode>(bad_type);
  }

  // Now check that all arguments are correct types and perform implicit
  // conversions if possible.
  for (size_t i = 0; i < arguments.size(); ++i) {
    // HACK: Void means "any" and we'll check in runtime. The argument will be
    // passed as is without any conversions.
    if (func_def->m_arguments[i].GetBasicTypeEnumeration()
        == lldb::eBasicTypeVoid) {
      continue;
    }
    arguments[i] = InsertImplicitConversion(std::move(arguments[i]),
                                            func_def->m_arguments[i]);
    if (llvm::isa<ErrorNode>(arguments[i])) {
      return std::make_unique<ErrorNode>(bad_type);
    }
  }

  return std::make_unique<BuiltinFunctionCallNode>(
      loc, func_def->m_return_type, func_def->m_name, std::move(arguments));
}

DILASTNodeUP DILParser::BuildCStyleCast(CompilerType type, DILASTNodeUP rhs,
                                      uint32_t location) {
  CStyleCastKind cast_kind = dil::CStyleCastKind::eNone;
  TypePromotionCastKind promo_kind = dil::TypePromotionCastKind::eNone;
  CompilerType bad_type;
  auto rhs_type = rhs->GetDereferencedResultType();

  // Cast to basic type (integer/float).
  if (type.IsScalarType()) {
    // Before casting arrays to scalar types, array-to-pointer conversion
    // should be performed.
    if (rhs_type.IsArrayType()) {
      rhs = InsertArrayToPointerConversion(std::move(rhs));
      rhs_type = rhs->GetDereferencedResultType();
    }
    // Pointers can be cast to integers of the same or larger size.
    if (rhs_type.IsPointerType() || rhs_type.IsNullPtrType()) {
      // C-style cast from pointer to float/double is not allowed.
      if (type.IsFloat()) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("C-style cast from {0} to {1} is not allowed",
                              rhs_type.TypeDescription(),
                              type.TypeDescription()),
                location);
        return std::make_unique<ErrorNode>(bad_type);
      }
      // Casting pointer to bool is valid. Otherwise check if the result type
      // is at least as big as the pointer size.
      uint64_t type_byte_size = 0;
      uint64_t rhs_type_byte_size = 0;
      if (auto temp = type.GetByteSize(m_ctx_scope.get()))
        type_byte_size = temp.value();
      if (auto temp = rhs_type.GetByteSize(m_ctx_scope.get()))
        rhs_type_byte_size = temp.value();
      if (!type.IsBoolean() && type_byte_size < rhs_type_byte_size) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cast from pointer to smaller type {0} loses information",
                    type.TypeDescription()),
                location);
        return std::make_unique<ErrorNode>(bad_type);
      }
    } else if (!rhs_type.IsScalarType() && !rhs_type.IsEnumerationType()) {
      // Otherwise accept only arithmetic types and enums.
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv(
                  "cannot convert {0} to {1} without a conversion operator",
                  rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
    promo_kind = TypePromotionCastKind::eArithmetic;

  } else if (type.IsEnumerationType()) {
    // Cast to enum type.
    if (!rhs_type.IsScalarType() && !rhs_type.IsEnumerationType()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("C-style cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
    cast_kind = CStyleCastKind::eEnumeration;

  } else if (type.IsPointerType()) {
    // Cast to pointer type.
    if (!rhs_type.IsInteger() && !rhs_type.IsEnumerationType() &&
        !rhs_type.IsArrayType() && !rhs_type.IsPointerType() &&
        !rhs_type.IsNullPtrType()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("cannot cast from type {0} to pointer type {1}",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
    promo_kind = TypePromotionCastKind::ePointer;

  } else if (type.IsNullPtrType()) {
    // Cast to nullptr type.
    if (!type.IsNullPtrType() && !rhs->is_literal_zero()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("C-style cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
    cast_kind = CStyleCastKind::eNullptr;

  } else if (type.IsReferenceType()) {
    // Cast to a reference type.
    if (rhs->is_rvalue()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("C-style cast from rvalue to reference type {0}",
                            type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
    cast_kind = CStyleCastKind::eReference;

  } else {
    // Unsupported cast.
    BailOut(ErrorCode::kNotImplemented,
            llvm::formatv("casting of {0} to {1} is not implemented yet",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  if (cast_kind != dil::CStyleCastKind::eNone)
    return std::make_unique<CStyleCastNode>(location, type, std::move(rhs), cast_kind);
  return std::make_unique<CStyleCastNode>(location, type, std::move(rhs), promo_kind);
}

DILASTNodeUP DILParser::BuildCxxCast(dil::TokenKind kind, CompilerType type,
                                   DILASTNodeUP rhs,
                                   uint32_t location) {
  assert((kind == dil::TokenKind::kw_static_cast ||
          kind == dil::TokenKind::kw_dynamic_cast ||
          kind == dil::TokenKind::kw_reinterpret_cast) &&
         "invalid C++-style cast type");

  // TODO: Implement custom builders for all C++-style casts.
  if (kind == dil::TokenKind::kw_dynamic_cast) {
    return BuildCxxDynamicCast(type, std::move(rhs), location);
  }
  if (kind == dil::TokenKind::kw_reinterpret_cast) {
    return BuildCxxReinterpretCast(type, std::move(rhs), location);
  }
  if (kind == dil::TokenKind::kw_static_cast) {
    return BuildCxxStaticCast(type, std::move(rhs), location);
  }
  return BuildCStyleCast(type, std::move(rhs), location);
}

DILASTNodeUP DILParser::BuildCxxStaticCast(CompilerType type, DILASTNodeUP rhs,
                                         uint32_t location) {
  auto rhs_type = rhs->GetDereferencedResultType();

  // Perform implicit array-to-pointer conversion.
  if (rhs_type.IsArrayType()) {
    rhs = InsertArrayToPointerConversion(std::move(rhs));
    rhs_type = rhs->GetDereferencedResultType();
  }

  if (rhs_type.CompareTypes(type)) {
    return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                               CxxStaticCastKind::eNoOp,
                                               /*is_rvalue*/ true);
  }

  if (type.IsScalarType()) {
    return BuildCxxStaticCastToScalar(type, std::move(rhs), location);
  } else if (type.IsEnumerationType()) {
    return BuildCxxStaticCastToEnum(type, std::move(rhs), location);
  } else if (type.IsPointerType()) {
    return BuildCxxStaticCastToPointer(type, std::move(rhs), location);
  } else if (type.IsNullPtrType()) {
    return BuildCxxStaticCastToNullPtr(type, std::move(rhs), location);
  } else if (type.IsReferenceType()) {
    return BuildCxxStaticCastToReference(type, std::move(rhs), location);
  }

  // Unsupported cast.
  BailOut(ErrorCode::kNotImplemented,
          llvm::formatv("casting of {0} to {1} is not implemented yet",
                        rhs_type.TypeDescription(), type.TypeDescription()),
          location);
  CompilerType bad_type;
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::BuildCxxStaticCastToScalar(CompilerType type,
                                                 DILASTNodeUP rhs,
                                                 uint32_t location)
{
  auto rhs_type = rhs->GetDereferencedResultType();
  CompilerType bad_type;

  if (rhs_type.IsPointerType() || rhs_type.IsNullPtrType()) {
    // Pointers can be casted to bools.
    if (!type.IsBoolean()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("static_cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
  } else if (!rhs_type.IsScalarType() && !rhs_type.IsEnumerationType()) {
    // Otherwise accept only arithmetic types and enums.
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("cannot convert {0} to {1} without a conversion operator",
                      rhs_type.TypeDescription(), type.TypeDescription()),
        location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             TypePromotionCastKind::eArithmetic,
                                             /*is_rvalue*/ true);
}

DILASTNodeUP DILParser::BuildCxxStaticCastToEnum(CompilerType type, DILASTNodeUP rhs,
                                               uint32_t location) {
  auto rhs_type = rhs->GetDereferencedResultType();

  if (!rhs_type.IsScalarType() && !rhs_type.IsEnumerationType()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("static_cast from {0} to {1} is not allowed",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             CxxStaticCastKind::eEnumeration,
                                             /*is_rvalue*/ true);
}

DILASTNodeUP DILParser::BuildCxxStaticCastToPointer(
    CompilerType type, DILASTNodeUP rhs, uint32_t location)
{
  CompilerType bad_type;
  auto rhs_type = rhs->GetDereferencedResultType();

  if (rhs_type.IsPointerType()) {
    auto type_pointee = type.GetPointeeType();
    auto rhs_type_pointee = rhs_type.GetPointeeType();

    if (type_pointee.IsRecordType() && rhs_type_pointee.IsRecordType()) {
      return BuildCxxStaticCastForInheritedTypes(type, std::move(rhs),
                                                 location);
    }

    if (!type.IsPointerToVoid() && !rhs_type.IsPointerToVoid()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("static_cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
  } else if (!rhs_type.IsNullPtrType() && !rhs->is_literal_zero()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot cast from type {0} to pointer type '{1}'",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             TypePromotionCastKind::ePointer,
                                             /*is_rvalue*/ true);
}

DILASTNodeUP DILParser::BuildCxxStaticCastToNullPtr(CompilerType type,
                                                  DILASTNodeUP rhs,
                                                  uint32_t location)
{
  auto rhs_type = rhs->GetDereferencedResultType();

  if (!rhs_type.IsNullPtrType() && !rhs->is_literal_zero()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("static_cast from {0} to {1} is not allowed",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                             CxxStaticCastKind::eNullptr,
                                             /*is_rvalue*/ true);
}

DILASTNodeUP DILParser::BuildCxxStaticCastToReference(
    CompilerType type,
    DILASTNodeUP rhs,
    uint32_t location) {
  CompilerType bad_type;
  auto rhs_type = rhs->GetDereferencedResultType();
  auto type_deref = type.GetNonReferenceType();

  if (rhs->is_rvalue()) {
    BailOut(ErrorCode::kNotImplemented,
            llvm::formatv("static_cast from rvalue of type {0} to reference "
                          "type {1} is not implemented yet",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  if (type_deref.CompareTypes(rhs_type)) {
    return std::make_unique<CxxStaticCastNode>(
        location, type_deref, std::move(rhs), CxxStaticCastKind::eNoOp,
        /*is_rvalue*/ false);
  }

  if (type_deref.IsRecordType() && rhs_type.IsRecordType()) {
    return BuildCxxStaticCastForInheritedTypes(type, std::move(rhs), location);
  }

  BailOut(ErrorCode::kNotImplemented,
          llvm::formatv("static_cast from {0} to {1} is not implemented yet",
                        rhs_type.TypeDescription(), type.TypeDescription()),
          location);
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::BuildCxxStaticCastForInheritedTypes(
    CompilerType type, DILASTNodeUP rhs, uint32_t location) {
  assert((type.IsPointerType() || type.IsReferenceType()) &&
         "target type should either be a pointer or a reference");

  CompilerType bad_type;
  auto rhs_type = rhs->GetDereferencedResultType();
  auto record_type = type.IsPointerType()
                     ? type.GetPointeeType()
                     : type.GetNonReferenceType();
  auto rhs_record_type =
      rhs_type.IsPointerType() ? rhs_type.GetPointeeType() : rhs_type;

  assert(record_type.IsRecordType() && rhs_record_type.IsRecordType() &&
         "underlying RHS and target types should be record types");
  assert(!record_type.CompareTypes(rhs_record_type) &&
         "underlying RHS and target types should be different");

  // Result of cast to reference type is an lvalue.
  bool is_rvalue = !type.IsReferenceType();

  // Handle derived-to-base conversion.
  std::vector<uint32_t> idx;
  if (GetPathToBaseType(rhs_record_type, record_type, &idx,
                        /*offset*/ nullptr)) {
    std::reverse(idx.begin(), idx.end());
    // At this point `idx` represents indices of direct base classes on path
    // from the `rhs` type to the target `type`.
    return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                               llvm::ArrayRef(std::move(idx)),
                                               is_rvalue);
  }

  // Handle base-to-derived conversion.
  uint64_t offset = 0;
  if (GetPathToBaseType(record_type, rhs_record_type, /*path*/ nullptr,
                        &offset)) {
    CompilerType virtual_base;
    if (record_type.IsVirtualBase(rhs_record_type, &virtual_base)) {
      // Base-to-derived conversion isn't possible for virtually inherited
      // types (either directly or indirectly).
      assert(virtual_base.IsValid() && "virtual base should be valid");
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("cannot cast {0} to {1} via virtual base {2}",
                            rhs_type.TypeDescription(), type.TypeDescription(),
                            virtual_base.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }

    return std::make_unique<CxxStaticCastNode>(location, type, std::move(rhs),
                                               offset, is_rvalue);
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("static_cast from {0} to {1}, which are not "
                        "related by inheritance, is not allowed",
                        rhs_type.TypeDescription(), type.TypeDescription()),
          location);
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::BuildCxxReinterpretCast(CompilerType type, DILASTNodeUP rhs,
                                              uint32_t location) {
  CompilerType bad_type;
  auto rhs_type = rhs->GetDereferencedResultType();
  bool is_rvalue = true;

  if (type.IsScalarType()) {
    // reinterpret_cast doesn't support non-integral scalar types.
    if (!type.IsInteger()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }

    // Perform implicit conversions.
    if (rhs_type.IsArrayType()) {
      rhs = InsertArrayToPointerConversion(std::move(rhs));
      rhs_type = rhs->GetDereferencedResultType();
    }

    if (rhs_type.IsPointerType() || rhs_type.IsNullPtrType()) {
      // A pointer can be converted to any integral type large enough to hold
      // its value.
      uint64_t type_byte_size = 0;
      uint64_t rhs_type_byte_size = 0;
      if (auto temp = type.GetByteSize(m_ctx_scope.get()))
        type_byte_size = temp.value();
      if (auto temp = rhs_type.GetByteSize(m_ctx_scope.get()))
        rhs_type_byte_size = temp.value();
      if (type_byte_size < rhs_type_byte_size) {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv(
                    "cast from pointer to smaller type {0} loses information",
                    type.TypeDescription()),
                location);
        return std::make_unique<ErrorNode>(bad_type);
      }
    } else if (type.IsTypedefType() || rhs_type.IsTypedefType()) {
      CompilerType base_type = type.IsTypedefType() ? type.GetTypedefedType()
                               :  type;
      CompilerType rhs_base_type = rhs_type.IsTypedefType() ?
                                   rhs_type.GetTypedefedType() : rhs_type;
      if (!base_type.CompareTypes(rhs_base_type)) {
        // Integral type can be converted to its own type.
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                              rhs_type.TypeDescription(), type.TypeDescription()),
                location);
        return std::make_unique<ErrorNode>(bad_type);
      }
    } else if (!type.CompareTypes(rhs_type)) {
      // Integral type can be converted to its own type.
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
  } else if (type.IsEnumerationType()) {
    // Enumeration type can be converted to its own type.
    CompilerType base_type = type.IsTypedefType() ? type.GetTypedefedType()
                             :  type;
    CompilerType rhs_base_type = rhs_type.IsTypedefType() ?
                                 rhs_type.GetTypedefedType() : rhs_type;
    if (!base_type.CompareTypes(rhs_base_type)) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }

  } else if (type.IsPointerType()) {
    // Integral, enumeration and other pointer types can be converted to any
    // pointer type.
    // TODO: Implement an explicit node for array-to-pointer conversions.
    if (!rhs_type.IsInteger() && !rhs_type.IsEnumerationType() &&
        !rhs_type.IsArrayType() && !rhs_type.IsPointerType()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                            rhs_type.TypeDescription(), type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }

  } else if (type.IsNullPtrType()) {
    // reinterpret_cast to nullptr_t isn't allowed (even for nullptr_t).
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("reinterpret_cast from {0} to {1} is not allowed",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);

  } else if (type.IsReferenceType()) {
    // L-values can be converted to any reference type.
    if (rhs->is_rvalue()) {
      BailOut(
          ErrorCode::kInvalidOperandType,
          llvm::formatv("reinterpret_cast from rvalue to reference type {0}",
                        type.TypeDescription()),
          location);
      return std::make_unique<ErrorNode>(bad_type);
    }
    // Casting to reference types gives an L-value result.
    is_rvalue = false;

  } else {
    // Unsupported cast.
    BailOut(ErrorCode::kNotImplemented,
            llvm::formatv("casting of {0} to {1} is not implemented yet",
                          rhs_type.TypeDescription(), type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<CxxReinterpretCastNode>(location, type,
                                                  std::move(rhs), is_rvalue);
}

DILASTNodeUP DILParser::BuildCxxDynamicCast(CompilerType type, DILASTNodeUP rhs,
                                          uint32_t location) {
  CompilerType pointee_type;
  CompilerType bad_type;
  if (type.IsPointerType()) {
    pointee_type = type.GetPointeeType();
  } else if (type.IsReferenceType()) {
    pointee_type = type.GetNonReferenceType();
  } else {
    // Dynamic casts are allowed only for pointers and references.
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("invalid target type {0} for dynamic_cast; target type "
                      "must be a reference or pointer type to a defined class",
                      type.TypeDescription()),
        location);
    return std::make_unique<ErrorNode>(bad_type);
  }
  // Dynamic casts are allowed only for record types.
  if (!pointee_type.IsRecordType()) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("{0} is not a class type", pointee_type.TypeDescription()),
        location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto expr_type = rhs->result_type();
  if (expr_type.IsPointerType()) {
    expr_type = expr_type.GetPointeeType();
  } else if (expr_type.IsReferenceType()) {
    expr_type = expr_type.GetNonReferenceType();
  } else {
    // Expression type must be a pointer or a reference.
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot use dynamic_cast to convert from {0} to {1}",
                          expr_type.TypeDescription(), type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }
  // Dynamic casts are allowed only for record types.
  if (!expr_type.IsRecordType()) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv("{0} is not a class type", expr_type.TypeDescription()),
        location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  // Expr type must be polymorphic.
  if (!expr_type.IsPolymorphicClass()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("{0} is not polymorphic", expr_type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  // LLDB doesn't support dynamic_cast in the expression evaluator. We disable
  // it too to match the behaviour, but theoretically it can be implemented.
  BailOut(ErrorCode::kInvalidOperandType,
          "dynamic_cast is not supported in this context", location);
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::BuildUnaryOp(UnaryOpKind kind, DILASTNodeUP rhs,
                                uint32_t location) {
  CompilerType result_type;
  auto rhs_type = rhs->GetDereferencedResultType();
  CompilerType bad_type;

  switch (kind) {
    case UnaryOpKind::Deref: {
      if (rhs_type.IsPointerType()) {
        result_type = rhs_type.GetPointeeType();
      } else if (rhs_type.IsArrayType()) {
        rhs = InsertArrayToPointerConversion(std::move(rhs));
        result_type = rhs->GetDereferencedResultType().GetPointeeType();
      } else {
        bool synthetic_child = false;
        lldb::ValueObjectSP rhs_valobj_sp;
        DILASTNode *ast_node = rhs.get();
        if (llvm::isa<IdentifierNode>(ast_node)) {
          IdentifierNode *id_node =
              (IdentifierNode*)ast_node;
          rhs_valobj_sp = id_node->valobj()->GetSP();
          Status deref_error;
          if (!rhs_valobj_sp->IsPointerType()) {
            lldb::ValueObjectSP child_sp =
                rhs_valobj_sp->Dereference(deref_error);
            if (child_sp && deref_error.Success()) {
              rhs_type = child_sp->GetCompilerType().GetPointerType();
              std::unique_ptr<IdentifierInfo> new_info =
                  IdentifierInfo::FromValue(*child_sp);
              DILASTNodeUP new_rhs = std::make_unique<IdentifierNode>(
                  id_node->GetLocation(), id_node->name(), std::move(new_info),
                  id_node->is_rvalue(), id_node->is_context_var());
              rhs = std::move(new_rhs);
              result_type = rhs->GetDereferencedResultType();
              synthetic_child = true;
            }
          }
        } else if (llvm::isa<MemberOfNode>(ast_node)) {
          const MemberOfNode *member_node =
              static_cast<const MemberOfNode*>(ast_node);
          rhs_valobj_sp = member_node->valobj()->GetSP();
          ValueObject *rhs_valobj = member_node->base()->valobj();
          if (rhs_valobj)
            rhs_valobj_sp = rhs_valobj->GetSP();
          if (rhs_valobj_sp)
            rhs_type = rhs_valobj_sp->GetCompilerType();
          result_type = rhs_type;
          synthetic_child = true;
        }

        if (!synthetic_child) {
          BailOut(
              ErrorCode::kInvalidOperandType,
              llvm::formatv("indirection requires pointer operand ({0} invalid)",
                            rhs_type.TypeDescription()),
              location);
          return std::make_unique<ErrorNode>(bad_type);
	}
      }
      break;
    }
    case UnaryOpKind::AddrOf: {
      if (rhs->is_rvalue()) {
        BailOut(
            ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot take the address of an rvalue of type {0}",
                          rhs_type.TypeDescription()),
            location);
        return std::make_unique<ErrorNode>(bad_type);
      }
      if (rhs->is_bitfield()) {
        BailOut(ErrorCode::kInvalidOperandType,
                "address of bit-field requested", location);
        return std::make_unique<ErrorNode>(bad_type);
      }
      result_type = rhs_type.GetPointerType();
      break;
    }
    case UnaryOpKind::Plus:
    case UnaryOpKind::Minus: {
      rhs = UsualUnaryConversions(m_ctx_scope, std::move(rhs));
      rhs_type = rhs->GetDereferencedResultType();
      if (rhs_type.IsScalarType() ||
          // Unary plus is allowed for pointers.
          (kind == UnaryOpKind::Plus && rhs_type.IsPointerType())) {
        result_type = rhs->result_type();
      }
      break;
    }
    case UnaryOpKind::Not: {
      rhs = UsualUnaryConversions(m_ctx_scope, std::move(rhs));
      rhs_type = rhs->GetDereferencedResultType();
      if (rhs_type.IsInteger()) {
        result_type = rhs->result_type();
      }
      break;
    }
    case UnaryOpKind::LNot: {
      if (rhs_type.IsContextuallyConvertibleToBool()) {
        result_type = GetBasicType(m_ctx_scope, lldb::eBasicTypeBool);
      }
      break;
    }
    case UnaryOpKind::PostInc:
    case UnaryOpKind::PostDec:
    case UnaryOpKind::PreInc:
    case UnaryOpKind::PreDec: {
      return BuildIncrementDecrement(kind, std::move(rhs), location);
    }

    default:
      llvm_unreachable("invalid unary op kind");
  }

  if (!result_type) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv(kInvalidOperandsToUnaryExpression,
                          rhs_type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<UnaryOpNode>(location, result_type, kind,
                                       std::move(rhs));
}

DILASTNodeUP DILParser::BuildBinaryOp(BinaryOpKind kind, DILASTNodeUP lhs,
                                 DILASTNodeUP rhs,
                                 uint32_t location) {
  // TODO: Get the "original" type (i.e. the one before implicit casts)
  // from the DILASTNodeUP.
  auto orig_lhs_type = lhs->GetDereferencedResultType();
  auto orig_rhs_type = rhs->GetDereferencedResultType();

  // Result type of the binary expression. For example, for `char + int` the
  // result type is `int`, but for `char += int` the result type is `char`.
  CompilerType result_type;

  // In case binary operation is a composite assignment, the type of the binary
  // expression _before_ the assignment. For example, for `char += int`
  // composite assignment type is `int`, because `char + int` is promoted to
  // `int + int`.
  CompilerType comp_assign_type;

  switch (kind) {
    case BinaryOpKind::Add:
      result_type =
          PrepareBinaryAddition(lhs, rhs, location, /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::Sub:
      result_type = PrepareBinarySubtraction(lhs, rhs, location,
                                             /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::Mul:
    case BinaryOpKind::Div:
      result_type = PrepareBinaryMulDiv(lhs, rhs,
                                        /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::Rem:
      result_type = PrepareBinaryRemainder(lhs, rhs, /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::And:
    case BinaryOpKind::Or:
    case BinaryOpKind::Xor:
      result_type = PrepareBinaryBitwise(lhs, rhs,
                                         /*is_comp_assign*/ false);
      break;
    case BinaryOpKind::Shl:
    case BinaryOpKind::Shr:
      result_type = PrepareBinaryShift(lhs, rhs,
                                       /*is_comp_assign*/ false);
      break;

    case BinaryOpKind::EQ:
    case BinaryOpKind::NE:
    case BinaryOpKind::LT:
    case BinaryOpKind::LE:
    case BinaryOpKind::GT:
    case BinaryOpKind::GE:
      result_type = PrepareBinaryComparison(kind, lhs, rhs, location);
      break;

    case BinaryOpKind::LAnd:
    case BinaryOpKind::LOr:
      result_type = PrepareBinaryLogical(lhs, rhs);
      break;

    case BinaryOpKind::Assign:
      // For plain assignment try to implicitly convert RHS to the type of LHS.
      // Later we'll check if the assignment is actually possible.
      rhs = InsertImplicitConversion(std::move(rhs), lhs->GetDereferencedResultType());
      // Shortcut for the case when the implicit conversion is not possible.
      if (llvm::isa<ErrorNode>(rhs)) {
        CompilerType bad_type;
        return std::make_unique<ErrorNode>(bad_type);
      }
      comp_assign_type = rhs->GetDereferencedResultType();
      break;

    case BinaryOpKind::AddAssign:
      comp_assign_type = PrepareBinaryAddition(lhs, rhs, location,
                                               /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::SubAssign:
      comp_assign_type = PrepareBinarySubtraction(lhs, rhs, location,
                                                  /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::MulAssign:
    case BinaryOpKind::DivAssign:
      comp_assign_type = PrepareBinaryMulDiv(lhs, rhs,
                                             /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::RemAssign:
      comp_assign_type =
          PrepareBinaryRemainder(lhs, rhs, /*is_comp_assign*/ true);
      break;

    case BinaryOpKind::AndAssign:
    case BinaryOpKind::OrAssign:
    case BinaryOpKind::XorAssign:
      comp_assign_type = PrepareBinaryBitwise(lhs, rhs,
                                              /*is_comp_assign*/ true);
      break;
    case BinaryOpKind::ShlAssign:
    case BinaryOpKind::ShrAssign:
      comp_assign_type = PrepareBinaryShift(lhs, rhs,
                                            /*is_comp_assign*/ true);
      break;

    default:
      llvm_unreachable("invalid binary op kind");
  }

  // If we're building a composite assignment, check for composite assignments
  // constraints: if the LHS is assignable, if the type of the binary operation
  // can be assigned to it, etc.
  if (comp_assign_type.IsValid()) {
    result_type = PrepareCompositeAssignment(comp_assign_type, lhs, location);
  }

  // If the result type is valid, then the binary operation is valid!
  if (result_type.IsValid()) {
    ValueObject *valobj_ptr = nullptr;
    if (result_type.IsPointerType()) {
      // Check to see if either lhs or rhs is an IdentifierNode. If so,
      // extract the valobj from the IdentifierNode & pass it to
      // BinaryOpNode (for help with looking up members of dynamic types).
      const DILASTNode* lhs_ast_node = lhs.get();
      const DILASTNode* rhs_ast_node = rhs.get();
      lldb::ValueObjectSP valobj_sp;
      if (llvm::isa<IdentifierNode>(lhs_ast_node)) {
        const IdentifierNode *id_node =
            static_cast<const IdentifierNode*>(lhs_ast_node);
        valobj_sp = id_node->valobj()->GetSP();
      } else if (llvm::isa<IdentifierNode>(rhs_ast_node)) {
        const IdentifierNode *id_node =
            static_cast<const IdentifierNode*>(rhs_ast_node);
        valobj_sp = id_node->valobj()->GetSP();
      }
      if (valobj_sp)
        valobj_ptr = valobj_sp.get();
    }
    return std::make_unique<BinaryOpNode>(location, result_type, kind,
                                          std::move(lhs), std::move(rhs),
                                          comp_assign_type,
                                          valobj_ptr);
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv(kInvalidOperandsToBinaryExpression,
                        orig_lhs_type.TypeDescription(),
                        orig_rhs_type.TypeDescription()),
          location);
  CompilerType bad_type;
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::BuildTernaryOp(DILASTNodeUP cond, DILASTNodeUP lhs,
                                  DILASTNodeUP rhs,
                                  uint32_t location) {
  CompilerType bad_type;
  // First check if the condition contextually converted to bool.
  auto cond_type = cond->GetDereferencedResultType();
  if (!cond_type.IsContextuallyConvertibleToBool()) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv(kValueIsNotConvertibleToBool, cond_type.TypeDescription()),
        location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  // If operands have the same type, don't do any promotions.
  if (lhs_type.CompareTypes(rhs_type)) {
    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  // If both operands have arithmetic type, apply the usual arithmetic
  // conversions to bring them to a common type.
  if (lhs_type.IsScalarOrUnscopedEnumerationType() &&
      rhs_type.IsScalarOrUnscopedEnumerationType()) {
    auto result_type = UsualArithmeticConversions(m_ctx_scope, lhs, rhs);
    return std::make_unique<TernaryOpNode>(
        location, result_type, std::move(cond), std::move(lhs), std::move(rhs));
  }

  // Apply array-to-pointer implicit conversions.
  if (lhs_type.IsArrayType()) {
    lhs = InsertArrayToPointerConversion(std::move(lhs));
    lhs_type = lhs->GetDereferencedResultType();
  }
  if (rhs_type.IsArrayType()) {
    rhs = InsertArrayToPointerConversion(std::move(rhs));
    rhs_type = rhs->GetDereferencedResultType();
  }

  // Check if operands have the same pointer type.
  if (lhs_type.CompareTypes(rhs_type)) {
    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  // If one operand is a pointer and the other is a nullptr or literal zero,
  // convert the nullptr operand to pointer type.
  if (lhs_type.IsPointerType() &&
      (rhs->is_literal_zero() || rhs_type.IsNullPtrType())) {
    rhs = std::make_unique<CStyleCastNode>(
        rhs->GetLocation(), lhs_type, std::move(rhs), TypePromotionCastKind::ePointer);

    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }
  if ((lhs->is_literal_zero() || lhs_type.IsNullPtrType()) &&
      rhs_type.IsPointerType()) {
    lhs = std::make_unique<CStyleCastNode>(
        lhs->GetLocation(), rhs_type, std::move(lhs), TypePromotionCastKind::ePointer);

    return std::make_unique<TernaryOpNode>(location, rhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  // If one operand is nullptr and the other one is literal zero, convert
  // the literal zero to a nullptr type.
  if (lhs_type.IsNullPtrType() && rhs->is_literal_zero()) {
    rhs = std::make_unique<CStyleCastNode>(
        rhs->GetLocation(), lhs_type, std::move(rhs), CStyleCastKind::eNullptr);

    return std::make_unique<TernaryOpNode>(location, lhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }
  if (lhs->is_literal_zero() && rhs_type.IsNullPtrType()) {
    lhs = std::make_unique<CStyleCastNode>(
        lhs->GetLocation(), rhs_type, std::move(lhs), CStyleCastKind::eNullptr);

    return std::make_unique<TernaryOpNode>(location, rhs_type, std::move(cond),
                                           std::move(lhs), std::move(rhs));
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("incompatible operand types ({0} and {1})",
                        lhs_type.TypeDescription(), rhs_type.TypeDescription()),
          location);
  return std::make_unique<ErrorNode>(bad_type);
}

DILASTNodeUP DILParser::BuildBinarySubscript(DILASTNodeUP lhs, DILASTNodeUP rhs,
                                        uint32_t location) {
  // C99 6.5.2.1p2: the expression e1[e2] is by definition precisely
  // equivalent to the expression *((e1)+(e2)).
  // We need to figure out which expression is "base" and which is "index".

  DILASTNodeUP base;
  DILASTNodeUP index;
  CompilerType bad_type;

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  if (lhs_type.IsArrayType()) {
    base = InsertArrayToPointerConversion(std::move(lhs));
    index = std::move(rhs);
  } else if (lhs_type.IsPointerType()) {
    base = std::move(lhs);
    index = std::move(rhs);
  } else if (rhs_type.IsArrayType()) {
    base = InsertArrayToPointerConversion(std::move(rhs));
    index = std::move(lhs);
  } else if (rhs_type.IsPointerType()) {
    base = std::move(rhs);
    index = std::move(lhs);
  } else {
    // Check to see if this might be a synthetic value.
    const DILASTNode* ast_node = lhs.get();
    if (llvm::isa<IdentifierNode>(ast_node)) {
      const IdentifierNode* id_node =
          static_cast<const IdentifierNode*>(ast_node);
      lldb::ValueObjectSP lhs_valobj_sp = id_node->valobj()->GetSP();
      if (lhs_valobj_sp->HasSyntheticValue()) {
        base = std::move(lhs);
        index = std::move(rhs);
      } else {
        BailOut(ErrorCode::kInvalidOperandType,
                "subscripted value is not an array or pointer", location);
        return std::make_unique<ErrorNode>(bad_type);
      }
    } else {
      BailOut(ErrorCode::kInvalidOperandType,
              "subscripted value is not an array or pointer", location);
      return std::make_unique<ErrorNode>(bad_type);
    }
  }

  // Index can be a typedef of a typedef of a typedef of a typedef...
  // Get canonical underlying type.
  auto index_type = index->GetDereferencedResultType();

  // Check if the index is of an integral type.
  if (!index_type.IsIntegerOrUnscopedEnumerationType()) {
    BailOut(ErrorCode::kInvalidOperandType, "array subscript is not an integer",
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto base_type = base->GetDereferencedResultType();
  if (base_type.IsPointerToVoid()) {
    BailOut(ErrorCode::kInvalidOperandType,
            "subscript of pointer to incomplete type 'void'", location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<ArraySubscriptNode>(
      location, base->GetDereferencedResultType().GetPointeeType(), std::move(base),
      std::move(index));
}

DILASTNodeUP DILParser::BuildMemberOf(DILASTNodeUP lhs, std::string member_id,
                                     bool is_arrow,
                                     uint32_t location) {
  CompilerType bad_type;
  auto lhs_type = lhs->GetDereferencedResultType();
  lldb::ValueObjectSP lhs_valobj_sp;
  lldb::ValueObjectSP deref_sp;

  const DILASTNode *ast_node = lhs.get();
  ValueObject *valobj = ast_node->valobj();
  if (valobj)
    lhs_valobj_sp = valobj->GetSP();

  if (is_arrow) {
    if (!lhs_type.IsPointerType() && !lhs_type.IsArrayType()) {
      Status deref_error;
      deref_sp = lhs_valobj_sp->Dereference(deref_error);
      if (deref_error.Success()) {
        lhs_valobj_sp = deref_sp;
        lhs_type = lhs_valobj_sp->GetCompilerType().GetPointerType();
      } else  {
        BailOut(ErrorCode::kInvalidOperandType,
                llvm::formatv("member reference type {0} is not a pointer; "
                              "did you mean to use '.'?",
                              lhs_type.TypeDescription()),
                location);
        return std::make_unique<ErrorNode>(bad_type);
      }
    } else if (lhs_type.IsArrayType()) {
      // If LHS is an array, convert it to pointer.
      lhs = InsertArrayToPointerConversion(std::move(lhs));
      lhs_type = lhs->GetDereferencedResultType();
      lhs_valobj_sp = lhs_valobj_sp->GetChildAtIndex(0);
    }

    lhs_type = lhs_type.GetPointeeType();
  } else {
    // "member of object" operator, check that LHS is an object.
    if (lhs_type.IsPointerType()) {
      BailOut(ErrorCode::kInvalidOperandType,
              llvm::formatv("member reference type {0} is a pointer; "
                            "did you mean to use '->'?",
                            lhs_type.TypeDescription()),
              location);
      return std::make_unique<ErrorNode>(bad_type);
    }
  }

  // Check if LHS is a record type, i.e. class/struct or union.
  if (!lhs_type.IsRecordType()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv(
                "member reference base type {0} is not a structure or union",
                lhs_type.TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  auto [opt_member, idx] = GetMemberInfo(lhs_valobj_sp, lhs_type, member_id,
                                         UseSynthetic(), is_arrow);

  if (!opt_member) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("no member named '{0}' in {1}", member_id,
                          lhs_type.GetFullyUnqualifiedType().TypeDescription()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  MemberInfo member = opt_member.value();
  std::optional<uint32_t> bitfield_size = member.bitfield_size_in_bits;
  uint64_t byte_size = 0;
  if (auto temp =
      member.type.GetByteSize(m_ctx_scope.get()))
    byte_size = temp.value();
  if (bitfield_size && bitfield_size.value() > byte_size * CHAR_BIT) {
    // If the declared bitfield size is exceeding the type size, shrink
    // the bitfield size to the size of the type in bits.
    bitfield_size = byte_size * CHAR_BIT;
  }

  std::string tmp_name= "";
  if (member.name)
    tmp_name = member.name.value();
  ConstString field_name(tmp_name.c_str());
  return std::make_unique<MemberOfNode>(location, member.type, std::move(lhs),
                                        bitfield_size,
                                        std::move(idx), is_arrow,
                                        member.is_synthetic, member.is_dynamic,
                                        field_name,
                                        member.val_obj_sp);
}

void DILParser::Expect(dil::TokenKind kind) {
  if (m_dil_token.isNot(kind)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected {0}, got: {1}", TokenKindsJoin(kind),
                          TokenDescription(m_dil_token)),
            m_dil_token.getLocation());
  }
}

template <typename... Ts>
void DILParser::ExpectOneOf(dil::TokenKind k, Ts... ks) {
  static_assert((std::is_same_v<Ts, dil::TokenKind> && ...),
                "ExpectOneOf can be only called with values of type "
                "dil::TokenKind");

  if (!m_dil_token.isOneOf(k, ks...)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected any of ({0}), got: {1}",
                          TokenKindsJoin(k, ks...), TokenDescription(m_dil_token)),
            m_dil_token.getLocation());
  }
}

DILASTNodeUP DILParser::BuildIncrementDecrement(UnaryOpKind kind, DILASTNodeUP rhs,
                                              uint32_t location) {
  assert((kind == UnaryOpKind::PreInc || kind == UnaryOpKind::PreDec ||
          kind == UnaryOpKind::PostInc || kind == UnaryOpKind::PostDec) &&
         "illegal unary op kind, expected inc/dec");

  CompilerType bad_type;
  auto rhs_type = rhs->GetDereferencedResultType();

  // In C++ the requirement here is that the expression is "assignable". However
  // in the debugger context side-effects are not allowed and the only case
  // where increment/decrement are permitted is when modifying the "context
  // variable".
  // Technically, `++(++$var)` could be allowed too, since both increments
  // modify the context variable. However, MSVC debugger doesn't allow it, so we
  // don't implement it too.
  if (rhs->is_rvalue()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("expression is not assignable"), location);
    return std::make_unique<ErrorNode>(bad_type);
  }
  if (!rhs->is_context_var() && !AllowSideEffects()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("side effects are not supported in this context: "
                          "trying to modify data at the target process"),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }
  llvm::StringRef op_name =
      (kind == UnaryOpKind::PreInc || kind == UnaryOpKind::PostInc)
          ? "increment"
          : "decrement";
  if (rhs_type.IsEnumerationType()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot {0} expression of enum type '{1}'", op_name,
                          rhs_type.GetTypeName()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }
  if (!rhs_type.IsScalarType() && !rhs_type.IsPointerType()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("cannot {0} value of type '{1}'", op_name,
                          rhs_type.GetTypeName()),
            location);
    return std::make_unique<ErrorNode>(bad_type);
  }

  return std::make_unique<UnaryOpNode>(location, rhs->result_type(), kind,
                                       std::move(rhs));
}

CompilerType DILParser::PrepareBinaryAddition(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                              uint32_t location,
                                              bool is_comp_assign) {
  // Operation '+' works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  {integer,unscoped_enum} <-> pointer
  //  pointer <-> {integer,unscoped_enum}

  CompilerType bad_type;
  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  if (result_type.IsScalarType()) {
    return result_type;
  }

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  // Check for pointer arithmetic operation.
  CompilerType ptr_type, integer_type;

  if (lhs_type.IsPointerType()) {
    ptr_type = lhs_type;
    integer_type = rhs_type;
  } else if (rhs_type.IsPointerType()) {
    integer_type = lhs_type;
    ptr_type = rhs_type;
  }

  if (!ptr_type || !integer_type.IsInteger()) {
    return bad_type;
  }

  if (ptr_type.IsPointerToVoid()) {
    BailOut(ErrorCode::kInvalidOperandType, "arithmetic on a pointer to void",
            location);
    return bad_type;
  }

  return ptr_type;
}

CompilerType DILParser::PrepareBinarySubtraction(DILASTNodeUP& lhs,
                                                 DILASTNodeUP& rhs,
                                                 uint32_t location,
                                                 bool is_comp_assign) {
  // Operation '-' works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  pointer <-> {integer,unscoped_enum}
  //  pointer <-> pointer (if pointee types are compatible)

  CompilerType bad_type;
  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  if (result_type.IsScalarType()) {
    return result_type;
  }

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  if (lhs_type.IsPointerType() && rhs_type.IsInteger()) {
    if (lhs_type.IsPointerToVoid()) {
      BailOut(ErrorCode::kInvalidOperandType, "arithmetic on a pointer to void",
              location);
      return bad_type;
    }

    return lhs_type;
  }

  if (lhs_type.IsPointerType() && rhs_type.IsPointerType()) {
    if (lhs_type.IsPointerToVoid() && rhs_type.IsPointerToVoid()) {
      BailOut(ErrorCode::kInvalidOperandType, "arithmetic on pointers to void",
              location);
      return bad_type;
    }

    // Compare canonical unqualified pointer types.
    CompilerType lhs_unqualified_type =
        lhs_type.GetCanonicalType().GetFullyUnqualifiedType();
    CompilerType rhs_unqualified_type =
        rhs_type.GetCanonicalType().GetFullyUnqualifiedType();
    bool comparable =
        lhs_unqualified_type.CompareTypes(rhs_unqualified_type);

    if (!comparable) {
      BailOut(
          ErrorCode::kInvalidOperandType,
          llvm::formatv("{0} and {1} are not pointers to compatible types",
                        lhs_type.TypeDescription(), rhs_type.TypeDescription()),
          location);
      return bad_type;
    }

    // Pointer difference is ptrdiff_t.
    return GetBasicType(m_ctx_scope, GetPtrDiffType(m_ctx_scope));
  }

  // Invalid operands.
  return bad_type;
}

CompilerType DILParser::PrepareBinaryMulDiv(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                            bool is_comp_assign) {
  // Operations {'*', '/'} work for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //

  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  // TODO: Check for arithmetic zero division.
  if (result_type.IsScalarType()) {
    return result_type;
  }

  // Invalid operands.
  CompilerType bad_type;
  return bad_type;
}

CompilerType DILParser::PrepareBinaryRemainder(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                               bool is_comp_assign) {
  // Operation '%' works for:
  //
  //  {integer,unscoped_enum} <-> {integer,unscoped_enum}

  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  // TODO: Check for arithmetic zero division.
  if (result_type.IsInteger()) {
    return result_type;
  }

  // Invalid operands.
  CompilerType bad_type;
  return bad_type;;
}

CompilerType DILParser::PrepareBinaryBitwise(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                             bool is_comp_assign) {
  // Operations {'&', '|', '^'} work for:
  //
  //  {integer,unscoped_enum} <-> {integer,unscoped_enum}
  //
  // Note that {'<<', '>>'} are handled in a separate method.

  auto result_type =
      UsualArithmeticConversions(m_ctx_scope, lhs, rhs, is_comp_assign);

  if (result_type.IsInteger()) {
    return result_type;
  }

  // Invalid operands.
  CompilerType bad_type;
  return bad_type;;
}

CompilerType DILParser::PrepareBinaryShift(DILASTNodeUP& lhs, DILASTNodeUP& rhs,
                                           bool is_comp_assign) {
  // Operations {'<<', '>>'} work for:
  //
  //  {integer,unscoped_enum} <-> {integer,unscoped_enum}
  CompilerType bad_type;

  if (!is_comp_assign) {
    lhs = UsualUnaryConversions(m_ctx_scope, std::move(lhs));
  }
  rhs = UsualUnaryConversions(m_ctx_scope, std::move(rhs));

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  if (!lhs_type.IsInteger() || !rhs_type.IsInteger()) {
    return bad_type;
  }

  // The type of the result is that of the promoted left operand.
  return DoIntegralPromotion(m_ctx_scope, lhs_type);
}

CompilerType DILParser::PrepareBinaryComparison(BinaryOpKind kind,
                                                DILASTNodeUP& lhs,
                                                DILASTNodeUP& rhs,
                                                uint32_t location)
{
  // Comparison works for:
  //
  //  {scalar,unscoped_enum} <-> {scalar,unscoped_enum}
  //  scoped_enum <-> scoped_enum (if the same type)
  //  pointer <-> pointer (if pointee types are compatible)
  //  pointer <-> {integer,unscoped_enum,nullptr_t}
  //  {integer,unscoped_enum,nullptr_t} <-> pointer
  //  nullptr_t <-> {nullptr_t,integer} (if integer is literal zero)
  //  {nullptr_t,integer} <-> nullptr_t (if integer is literal zero)

  // If the operands has arithmetic or enumeration type (scoped or unscoped),
  // usual arithmetic conversions are performed on both operands following the
  // rules for arithmetic operators.
  CompilerType bad_type;
  auto _ = UsualArithmeticConversions(m_ctx_scope, lhs, rhs);

  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  // The result of the comparison is always bool.
  auto boolean_ty = GetBasicType(m_ctx_scope, lldb::eBasicTypeBool);

  if (lhs_type.IsScalarOrUnscopedEnumerationType() &&
      rhs_type.IsScalarOrUnscopedEnumerationType()) {
    return boolean_ty;
  }

  // Scoped enums can be compared only to the instances of the same type.
  if (lhs_type.IsScopedEnumerationType() || rhs_type.IsScopedEnumerationType()) {
    if (lhs_type.CompareTypes(rhs_type)) {
      return boolean_ty;
    }
    // Invalid operands.
    return bad_type;;
  }

  bool is_ordered = (kind == BinaryOpKind::LT || kind == BinaryOpKind::LE ||
                     kind == BinaryOpKind::GT || kind == BinaryOpKind::GE);

  // Check if the value can be compared to a pointer. We allow all pointers,
  // integers, unscoped enumerations and a nullptr literal if it's an
  // equality/inequality comparison. For "pointer <-> integer" C++ allows only
  // equality/inequality comparison against literal zero and nullptr. However in
  // the debugger context it's often useful to compare a pointer with an integer
  // representing an address. That said, this also allows comparing nullptr and
  // any integer, not just literal zero, e.g. "nullptr == 1 -> false". C++
  // doesn't allow it, but we implement this for convenience.
  auto comparable_to_pointer = [&](CompilerType t) {
    return t.IsPointerType() || t.IsInteger() || t.IsUnscopedEnumerationType() ||
           (!is_ordered && t.IsNullPtrType());
  };

  if ((lhs_type.IsPointerType() && comparable_to_pointer(rhs_type)) ||
      (comparable_to_pointer(lhs_type) && rhs_type.IsPointerType())) {
    // If both are pointers, check if they have comparable types. Comparing
    // pointers to void is always allowed.
    if ((lhs_type.IsPointerType() && !lhs_type.IsPointerToVoid()) &&
        (rhs_type.IsPointerType() && !rhs_type.IsPointerToVoid())) {
      // Compare canonical unqualified pointer types.
      CompilerType lhs_unqualified_type =
          lhs_type.GetCanonicalType().GetFullyUnqualifiedType();
      CompilerType rhs_unqualified_type =
          rhs_type.GetCanonicalType().GetFullyUnqualifiedType();
      bool comparable =
          lhs_unqualified_type.CompareTypes(rhs_unqualified_type);

      if (!comparable) {
        BailOut(
            ErrorCode::kInvalidOperandType,
            llvm::formatv("comparison of distinct pointer types ({0} and {1})",
                          lhs_type.TypeDescription(), rhs_type.TypeDescription()),
            location);
        return bad_type;;
      }
    }

    return boolean_ty;
  }

  bool lhs_nullptr_or_zero =
      lhs_type.IsNullPtrType() || lhs->is_literal_zero();
  bool rhs_nullptr_or_zero =
      rhs_type.IsNullPtrType() || rhs->is_literal_zero();

  if (!is_ordered && ((lhs_type.IsNullPtrType() && rhs_nullptr_or_zero) ||
                      (lhs_nullptr_or_zero && rhs_type.IsNullPtrType()))) {
    return boolean_ty;
  }

  // Invalid operands.
  return bad_type;
}

CompilerType DILParser::PrepareBinaryLogical(const DILASTNodeUP& lhs,
                                             const DILASTNodeUP& rhs) {
  CompilerType bad_type;
  auto lhs_type = lhs->GetDereferencedResultType();
  auto rhs_type = rhs->GetDereferencedResultType();

  if (!lhs_type.IsContextuallyConvertibleToBool()) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv(kValueIsNotConvertibleToBool, lhs_type.TypeDescription()),
        lhs->GetLocation());
    return bad_type;
  }

  if (!rhs_type.IsContextuallyConvertibleToBool()) {
    BailOut(
        ErrorCode::kInvalidOperandType,
        llvm::formatv(kValueIsNotConvertibleToBool, rhs_type.TypeDescription()),
        rhs->GetLocation());
    return bad_type;
  }

  // The result of the logical operator is always bool.
  return GetBasicType(m_ctx_scope, lldb::eBasicTypeBool);
}

CompilerType DILParser::PrepareCompositeAssignment(
    CompilerType comp_assign_type,
    const DILASTNodeUP& lhs,
    uint32_t location) {
  // In C++ the requirement here is that the expression is "assignable".
  // However in the debugger context side-effects are not allowed and the only
  // case where composite assignments are permitted is when modifying the
  // "context variable".
  // Technically, `($var += 1) += 1` could be allowed too, since both
  // operations modify the context variable. However, MSVC debugger doesn't
  // allow it, so we don't implement it too.
  CompilerType bad_type;
  if (lhs->is_rvalue()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("expression is not assignable"), location);
    return bad_type;
  }
  if (!lhs->is_context_var() && !AllowSideEffects()) {
    BailOut(ErrorCode::kInvalidOperandType,
            llvm::formatv("side effects are not supported in this context: "
                          "trying to modify data at the target process"),
            location);
    return bad_type;
  }

  // Check if we can assign the result of the binary operation back to LHS.
  auto lhs_type = lhs->GetDereferencedResultType();

  if (comp_assign_type.CompareTypes(lhs_type) ||
      ImplicitConversionIsAllowed(comp_assign_type, lhs_type)) {
    return lhs_type;
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("no known conversion from {0} to {1}",
                        comp_assign_type.TypeDescription(),
                        lhs_type.TypeDescription()),
          location);
  return bad_type;
}

void DILParser::BailOut(ErrorCode code, const std::string& error,
                        uint32_t loc) {
  if (m_error.Fail()) {
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;
  }

  m_error = Status((uint32_t) code, lldb::eErrorTypeGeneric,
                   FormatDiagnostics(*m_sm, error, loc));
  m_dil_token.setKind(dil::TokenKind::eof);
}

void DILParser::BailOut(Status error) {
  if (m_error.Fail()) {
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;
  }
  m_error = std::move(error);
  m_dil_token.setKind(dil::TokenKind::eof);
}

void DILParser::ConsumeToken() {
  if (m_dil_token.is(dil::TokenKind::eof)) {
    // Don't do anything if we're already at eof. This can happen if an error
    // occurred during parsing and we're trying to bail out.
    return;
  }
  bool all_ok;
  m_dil_lexer.Lex(m_dil_token);
  if (m_dil_lexer.GetCurrentTokenIdx() == UINT_MAX)
    all_ok = m_dil_lexer.ResetTokenIdx(0);
  else
    all_ok = m_dil_lexer.IncrementTokenIdx();
  if (!all_ok)
    BailOut(ErrorCode::kUnknown, "Invalid lexer token index", 0);
}

std::string DILParser::TokenDescription(const DILToken& token) {
  const auto& spelling = ((DILToken)token).getSpelling();
  const std::string kind_name = DILToken::getTokenName(((DILToken)token).getKind());
  return llvm::formatv("<'{0}' ({1})>", spelling, kind_name);
}

bool DILParser::ImplicitConversionIsAllowed(CompilerType src, CompilerType dst,
                                            bool is_src_literal_zero) {
  if (dst.IsInteger() || dst.IsFloat()) {
    // Arithmetic types and enumerations can be implicitly converted to integers
    // and floating point types.
    if (src.IsScalarOrUnscopedEnumerationType()
        || src.IsScopedEnumerationType()) {
      return true;
    }
  }

  if (dst.IsPointerType()) {
    // Literal zero, `nullptr_t` and arrays can be implicitly converted to
    // pointers.
    if (is_src_literal_zero || src.IsNullPtrType()) {
      return true;
    }
    if (src.IsArrayType() &&
        src.GetArrayElementType(nullptr).CompareTypes(dst.GetPointeeType())) {
      return true;
    }
  }

  return false;
}

DILASTNodeUP DILParser::InsertImplicitConversion(DILASTNodeUP expr,
                                                CompilerType type) {
  auto expr_type = expr->GetDereferencedResultType();

  // If the expression already has the required type, nothing to do here.
  if (expr_type.CompareTypes(type)) {
    return expr;
  }

  // Check if the implicit conversion is possible and insert a cast.
  if (ImplicitConversionIsAllowed(expr_type, type, expr->is_literal_zero())) {
    if (type.GetCanonicalType().GetBasicTypeEnumeration() !=
        lldb::eBasicTypeInvalid) {
      return std::make_unique<CStyleCastNode>(
          expr->GetLocation(), type, std::move(expr), TypePromotionCastKind::eArithmetic);
    }

    if (type.IsPointerType()) {
      return std::make_unique<CStyleCastNode>(
          expr->GetLocation(), type, std::move(expr), TypePromotionCastKind::ePointer);
    }

    // TODO: What about if the conversion is not `kArithmetic` or
    // `kPointer`?
    llvm_unreachable("invalid implicit cast kind");
  }

  BailOut(ErrorCode::kInvalidOperandType,
          llvm::formatv("no known conversion from {0} to {1}",
                        expr_type.TypeDescription(), type.TypeDescription()),
          expr->GetLocation());
  CompilerType bad_type;
  return std::make_unique<ErrorNode>(bad_type);
}


lldb::BasicType TypeDeclaration::GetBasicType() const {
  assert(m_is_builtin && "type declaration doesn't describe a builtin type");

  if (m_sign_specifier == SignSpecifier::kSigned &&
      m_type_specifier == TypeSpecifier::kChar) {
    // "signed char" isn't the same as "char".
    return lldb::eBasicTypeSignedChar;
  }

  if (m_sign_specifier == SignSpecifier::kUnsigned) {
    switch (m_type_specifier) {
      // "unsigned" is "unsigned int"
      case TypeSpecifier::kUnknown:  return lldb::eBasicTypeUnsignedInt;
      case TypeSpecifier::kChar:     return lldb::eBasicTypeUnsignedChar;
      case TypeSpecifier::kShort:    return lldb::eBasicTypeUnsignedShort;
      case TypeSpecifier::kInt:      return lldb::eBasicTypeUnsignedInt;
      case TypeSpecifier::kLong:     return lldb::eBasicTypeUnsignedLong;
      case TypeSpecifier::kLongLong: return lldb::eBasicTypeUnsignedLongLong;
      default:
        assert(false && "unknown unsigned basic type");
        return lldb::eBasicTypeInvalid;
    }
  }

  switch (m_type_specifier) {
    case TypeSpecifier::kUnknown:
      // "signed" is "signed int"
      assert(m_sign_specifier == SignSpecifier::kSigned &&
             "invalid basic type declaration");
      return lldb::eBasicTypeInt;
    case TypeSpecifier::kVoid:       return lldb::eBasicTypeVoid;
    case TypeSpecifier::kBool:       return lldb::eBasicTypeBool;
    case TypeSpecifier::kChar:       return lldb::eBasicTypeChar;
    case TypeSpecifier::kShort:      return lldb::eBasicTypeShort;
    case TypeSpecifier::kInt:        return lldb::eBasicTypeInt;
    case TypeSpecifier::kLong:       return lldb::eBasicTypeLong;
    case TypeSpecifier::kLongLong:   return lldb::eBasicTypeLongLong;
    case TypeSpecifier::kFloat:      return lldb::eBasicTypeFloat;
    case TypeSpecifier::kDouble:     return lldb::eBasicTypeDouble;
    case TypeSpecifier::kLongDouble: return lldb::eBasicTypeLongDouble;
    case TypeSpecifier::kWChar:      return lldb::eBasicTypeWChar;
    case TypeSpecifier::kChar16:     return lldb::eBasicTypeChar16;
    case TypeSpecifier::kChar32:     return lldb::eBasicTypeChar32;
  }

  return lldb::eBasicTypeInvalid;
}

}  // namespace dil

}  // namespace lldb_private
