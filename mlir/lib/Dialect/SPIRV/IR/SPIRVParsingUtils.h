//===- SPIRVParsingUtils.h - MLIR SPIR-V Dialect Parsing Utilities --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <type_traits>

namespace mlir::spirv {
namespace AttrNames {
// TODO: generate these strings using ODS.
inline constexpr char kAlignmentAttrName[] = "alignment";
inline constexpr char kBranchWeightAttrName[] = "branch_weights";
inline constexpr char kCallee[] = "callee";
inline constexpr char kClusterSize[] = "cluster_size";
inline constexpr char kControl[] = "control";
inline constexpr char kDefaultValueAttrName[] = "default_value";
inline constexpr char kEqualSemanticsAttrName[] = "equal_semantics";
inline constexpr char kExecutionScopeAttrName[] = "execution_scope";
inline constexpr char kFnNameAttrName[] = "fn";
inline constexpr char kGroupOperationAttrName[] = "group_operation";
inline constexpr char kIndicesAttrName[] = "indices";
inline constexpr char kInitializerAttrName[] = "initializer";
inline constexpr char kInterfaceAttrName[] = "interface";
inline constexpr char kKhrCooperativeMatrixLayoutAttrName[] = "matrix_layout";
inline constexpr char kMemoryAccessAttrName[] = "memory_access";
inline constexpr char kMemoryOperandAttrName[] = "memory_operand";
inline constexpr char kMemoryScopeAttrName[] = "memory_scope";
inline constexpr char kPackedVectorFormatAttrName[] = "format";
inline constexpr char kSemanticsAttrName[] = "semantics";
inline constexpr char kSourceAlignmentAttrName[] = "source_alignment";
inline constexpr char kSourceMemoryAccessAttrName[] = "source_memory_access";
inline constexpr char kSpecIdAttrName[] = "spec_id";
inline constexpr char kTypeAttrName[] = "type";
inline constexpr char kUnequalSemanticsAttrName[] = "unequal_semantics";
inline constexpr char kValueAttrName[] = "value";
inline constexpr char kValuesAttrName[] = "values";
inline constexpr char kCompositeSpecConstituentsName[] = "constituents";
} // namespace AttrNames

template <typename Ty>
ArrayAttr getStrArrayAttrForEnumList(Builder &builder, ArrayRef<Ty> enumValues,
                                     function_ref<StringRef(Ty)> stringifyFn) {
  if (enumValues.empty()) {
    return nullptr;
  }
  SmallVector<StringRef, 1> enumValStrs;
  enumValStrs.reserve(enumValues.size());
  for (auto val : enumValues) {
    enumValStrs.emplace_back(stringifyFn(val));
  }
  return builder.getStrArrayAttr(enumValStrs);
}

/// Parses the next keyword in `parser` as an enumerant of the given
/// `EnumClass`.
template <typename EnumClass, typename ParserType>
ParseResult
parseEnumKeywordAttr(EnumClass &value, ParserType &parser,
                     StringRef attrName = spirv::attributeName<EnumClass>()) {
  StringRef keyword;
  SmallVector<NamedAttribute, 1> attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword))
    return failure();

  if (std::optional<EnumClass> attr =
          spirv::symbolizeEnum<EnumClass>(keyword)) {
    value = *attr;
    return success();
  }
  return parser.emitError(loc, "invalid ")
         << attrName << " attribute specification: " << keyword;
}

/// Parses the next string attribute in `parser` as an enumerant of the given
/// `EnumClass`.
template <typename EnumClass>
ParseResult
parseEnumStrAttr(EnumClass &value, OpAsmParser &parser,
                 StringRef attrName = spirv::attributeName<EnumClass>()) {
  static_assert(std::is_enum_v<EnumClass>);
  Attribute attrVal;
  NamedAttrList attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                            attrName, attr))
    return failure();
  if (!llvm::isa<StringAttr>(attrVal))
    return parser.emitError(loc, "expected ")
           << attrName << " attribute specified as string";
  auto attrOptional = spirv::symbolizeEnum<EnumClass>(
      llvm::cast<StringAttr>(attrVal).getValue());
  if (!attrOptional)
    return parser.emitError(loc, "invalid ")
           << attrName << " attribute specification: " << attrVal;
  value = *attrOptional;
  return success();
}

/// Parses the next string attribute in `parser` as an enumerant of the given
/// `EnumClass` and inserts the enumerant into `state` as an 32-bit integer
/// attribute with the enum class's name as attribute name.
template <typename EnumAttrClass,
          typename EnumClass = typename EnumAttrClass::ValueType>
ParseResult
parseEnumStrAttr(EnumClass &value, OpAsmParser &parser, OperationState &state,
                 StringRef attrName = spirv::attributeName<EnumClass>()) {
  static_assert(std::is_enum_v<EnumClass>);
  if (parseEnumStrAttr(value, parser, attrName))
    return failure();
  state.addAttribute(attrName,
                     parser.getBuilder().getAttr<EnumAttrClass>(value));
  return success();
}

/// Parses the next keyword in `parser` as an enumerant of the given `EnumClass`
/// and inserts the enumerant into `state` as an 32-bit integer attribute with
/// the enum class's name as attribute name.
template <typename EnumAttrClass,
          typename EnumClass = typename EnumAttrClass::ValueType>
ParseResult
parseEnumKeywordAttr(EnumClass &value, OpAsmParser &parser,
                     OperationState &state,
                     StringRef attrName = spirv::attributeName<EnumClass>()) {
  static_assert(std::is_enum_v<EnumClass>);
  if (parseEnumKeywordAttr(value, parser))
    return failure();
  state.addAttribute(attrName,
                     parser.getBuilder().getAttr<EnumAttrClass>(value));
  return success();
}

/// Parses optional memory access (a.k.a. memory operand) attributes attached to
/// a memory access operand/pointer. Specifically, parses the following syntax:
///     (`[` memory-access `]`)?
/// where:
///     memory-access ::= `"None"` | `"Volatile"` | `"Aligned", `
///         integer-literal | `"NonTemporal"`
ParseResult parseMemoryAccessAttributes(
    OpAsmParser &parser, OperationState &state,
    StringRef attrName = AttrNames::kMemoryAccessAttrName);

ParseResult parseVariableDecorations(OpAsmParser &parser,
                                     OperationState &state);

} // namespace mlir::spirv
