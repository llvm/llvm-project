//===- SPIRVOps.cpp - MLIR SPIR-V operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "mlir/Dialect/SPIRV/IR/ParserUtils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"

using namespace mlir;

// TODO: generate these strings using ODS.
static constexpr const char kMemoryAccessAttrName[] = "memory_access";
static constexpr const char kSourceMemoryAccessAttrName[] =
    "source_memory_access";
static constexpr const char kAlignmentAttrName[] = "alignment";
static constexpr const char kSourceAlignmentAttrName[] = "source_alignment";
static constexpr const char kBranchWeightAttrName[] = "branch_weights";
static constexpr const char kCallee[] = "callee";
static constexpr const char kClusterSize[] = "cluster_size";
static constexpr const char kControl[] = "control";
static constexpr const char kDefaultValueAttrName[] = "default_value";
static constexpr const char kExecutionScopeAttrName[] = "execution_scope";
static constexpr const char kEqualSemanticsAttrName[] = "equal_semantics";
static constexpr const char kFnNameAttrName[] = "fn";
static constexpr const char kGroupOperationAttrName[] = "group_operation";
static constexpr const char kIndicesAttrName[] = "indices";
static constexpr const char kInitializerAttrName[] = "initializer";
static constexpr const char kInterfaceAttrName[] = "interface";
static constexpr const char kMemoryScopeAttrName[] = "memory_scope";
static constexpr const char kSemanticsAttrName[] = "semantics";
static constexpr const char kSpecIdAttrName[] = "spec_id";
static constexpr const char kTypeAttrName[] = "type";
static constexpr const char kUnequalSemanticsAttrName[] = "unequal_semantics";
static constexpr const char kValueAttrName[] = "value";
static constexpr const char kValuesAttrName[] = "values";
static constexpr const char kCompositeSpecConstituentsName[] = "constituents";

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

/// Returns true if the given op is a function-like op or nested in a
/// function-like op without a module-like op in the middle.
static bool isNestedInFunctionLikeOp(Operation *op) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::SymbolTable>())
    return false;
  if (op->hasTrait<OpTrait::FunctionLike>())
    return true;
  return isNestedInFunctionLikeOp(op->getParentOp());
}

/// Returns true if the given op is an module-like op that maintains a symbol
/// table.
static bool isDirectInModuleLikeOp(Operation *op) {
  return op && op->hasTrait<OpTrait::SymbolTable>();
}

static LogicalResult extractValueFromConstOp(Operation *op, int32_t &value) {
  auto constOp = dyn_cast_or_null<spirv::ConstantOp>(op);
  if (!constOp) {
    return failure();
  }
  auto valueAttr = constOp.value();
  auto integerValueAttr = valueAttr.dyn_cast<IntegerAttr>();
  if (!integerValueAttr) {
    return failure();
  }
  value = integerValueAttr.getInt();
  return success();
}

template <typename Ty>
static ArrayAttr
getStrArrayAttrForEnumList(Builder &builder, ArrayRef<Ty> enumValues,
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

/// Parses the next string attribute in `parser` as an enumerant of the given
/// `EnumClass`.
template <typename EnumClass>
static ParseResult
parseEnumStrAttr(EnumClass &value, OpAsmParser &parser,
                 StringRef attrName = spirv::attributeName<EnumClass>()) {
  Attribute attrVal;
  NamedAttrList attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                            attrName, attr)) {
    return failure();
  }
  if (!attrVal.isa<StringAttr>()) {
    return parser.emitError(loc, "expected ")
           << attrName << " attribute specified as string";
  }
  auto attrOptional =
      spirv::symbolizeEnum<EnumClass>(attrVal.cast<StringAttr>().getValue());
  if (!attrOptional) {
    return parser.emitError(loc, "invalid ")
           << attrName << " attribute specification: " << attrVal;
  }
  value = attrOptional.getValue();
  return success();
}

/// Parses the next string attribute in `parser` as an enumerant of the given
/// `EnumClass` and inserts the enumerant into `state` as an 32-bit integer
/// attribute with the enum class's name as attribute name.
template <typename EnumClass>
static ParseResult
parseEnumStrAttr(EnumClass &value, OpAsmParser &parser, OperationState &state,
                 StringRef attrName = spirv::attributeName<EnumClass>()) {
  if (parseEnumStrAttr(value, parser)) {
    return failure();
  }
  state.addAttribute(attrName, parser.getBuilder().getI32IntegerAttr(
                                   llvm::bit_cast<int32_t>(value)));
  return success();
}

/// Parses the next keyword in `parser` as an enumerant of the given `EnumClass`
/// and inserts the enumerant into `state` as an 32-bit integer attribute with
/// the enum class's name as attribute name.
template <typename EnumClass>
static ParseResult
parseEnumKeywordAttr(EnumClass &value, OpAsmParser &parser,
                     OperationState &state,
                     StringRef attrName = spirv::attributeName<EnumClass>()) {
  if (parseEnumKeywordAttr(value, parser)) {
    return failure();
  }
  state.addAttribute(attrName, parser.getBuilder().getI32IntegerAttr(
                                   llvm::bit_cast<int32_t>(value)));
  return success();
}

/// Parses Function, Selection and Loop control attributes. If no control is
/// specified, "None" is used as a default.
template <typename EnumClass>
static ParseResult
parseControlAttribute(OpAsmParser &parser, OperationState &state,
                      StringRef attrName = spirv::attributeName<EnumClass>()) {
  if (succeeded(parser.parseOptionalKeyword(kControl))) {
    EnumClass control;
    if (parser.parseLParen() || parseEnumKeywordAttr(control, parser, state) ||
        parser.parseRParen())
      return failure();
    return success();
  }
  // Set control to "None" otherwise.
  Builder builder = parser.getBuilder();
  state.addAttribute(attrName, builder.getI32IntegerAttr(0));
  return success();
}

/// Parses optional memory access attributes attached to a memory access
/// operand/pointer. Specifically, parses the following syntax:
///     (`[` memory-access `]`)?
/// where:
///     memory-access ::= `"None"` | `"Volatile"` | `"Aligned", `
///         integer-literal | `"NonTemporal"`
static ParseResult parseMemoryAccessAttributes(OpAsmParser &parser,
                                               OperationState &state) {
  // Parse an optional list of attributes staring with '['
  if (parser.parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  if (parseEnumStrAttr(memoryAccessAttr, parser, state,
                       kMemoryAccessAttrName)) {
    return failure();
  }

  if (spirv::bitEnumContains(memoryAccessAttr, spirv::MemoryAccess::Aligned)) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseComma() ||
        parser.parseAttribute(alignmentAttr, i32Type, kAlignmentAttrName,
                              state.attributes)) {
      return failure();
    }
  }
  return parser.parseRSquare();
}

// TODO Make sure to merge this and the previous function into one template
// parameterized by memory access attribute name and alignment. Doing so now
// results in VS2017 in producing an internal error (at the call site) that's
// not detailed enough to understand what is happening.
static ParseResult parseSourceMemoryAccessAttributes(OpAsmParser &parser,
                                                     OperationState &state) {
  // Parse an optional list of attributes staring with '['
  if (parser.parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  if (parseEnumStrAttr(memoryAccessAttr, parser, state,
                       kSourceMemoryAccessAttrName)) {
    return failure();
  }

  if (spirv::bitEnumContains(memoryAccessAttr, spirv::MemoryAccess::Aligned)) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseComma() ||
        parser.parseAttribute(alignmentAttr, i32Type, kSourceAlignmentAttrName,
                              state.attributes)) {
      return failure();
    }
  }
  return parser.parseRSquare();
}

template <typename MemoryOpTy>
static void printMemoryAccessAttribute(
    MemoryOpTy memoryOp, OpAsmPrinter &printer,
    SmallVectorImpl<StringRef> &elidedAttrs,
    Optional<spirv::MemoryAccess> memoryAccessAtrrValue = None,
    Optional<uint32_t> alignmentAttrValue = None) {
  // Print optional memory access attribute.
  if (auto memAccess = (memoryAccessAtrrValue ? memoryAccessAtrrValue
                                              : memoryOp.memory_access())) {
    elidedAttrs.push_back(kMemoryAccessAttrName);

    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    if (spirv::bitEnumContains(*memAccess, spirv::MemoryAccess::Aligned)) {
      // Print integer alignment attribute.
      if (auto alignment = (alignmentAttrValue ? alignmentAttrValue
                                               : memoryOp.alignment())) {
        elidedAttrs.push_back(kAlignmentAttrName);
        printer << ", " << alignment;
      }
    }
    printer << "]";
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
}

// TODO Make sure to merge this and the previous function into one template
// parameterized by memory access attribute name and alignment. Doing so now
// results in VS2017 in producing an internal error (at the call site) that's
// not detailed enough to understand what is happening.
template <typename MemoryOpTy>
static void printSourceMemoryAccessAttribute(
    MemoryOpTy memoryOp, OpAsmPrinter &printer,
    SmallVectorImpl<StringRef> &elidedAttrs,
    Optional<spirv::MemoryAccess> memoryAccessAtrrValue = None,
    Optional<uint32_t> alignmentAttrValue = None) {

  printer << ", ";

  // Print optional memory access attribute.
  if (auto memAccess = (memoryAccessAtrrValue ? memoryAccessAtrrValue
                                              : memoryOp.memory_access())) {
    elidedAttrs.push_back(kSourceMemoryAccessAttrName);

    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"";

    if (spirv::bitEnumContains(*memAccess, spirv::MemoryAccess::Aligned)) {
      // Print integer alignment attribute.
      if (auto alignment = (alignmentAttrValue ? alignmentAttrValue
                                               : memoryOp.alignment())) {
        elidedAttrs.push_back(kSourceAlignmentAttrName);
        printer << ", " << alignment;
      }
    }
    printer << "]";
  }
  elidedAttrs.push_back(spirv::attributeName<spirv::StorageClass>());
}

static LogicalResult verifyCastOp(Operation *op,
                                  bool requireSameBitWidth = true,
                                  bool skipBitWidthCheck = false) {
  // Some CastOps have no limit on bit widths for result and operand type.
  if (skipBitWidthCheck)
    return success();

  Type operandType = op->getOperand(0).getType();
  Type resultType = op->getResult(0).getType();

  // ODS checks that result type and operand type have the same shape.
  if (auto vectorType = operandType.dyn_cast<VectorType>()) {
    operandType = vectorType.getElementType();
    resultType = resultType.cast<VectorType>().getElementType();
  }

  if (auto coopMatrixType =
          operandType.dyn_cast<spirv::CooperativeMatrixNVType>()) {
    operandType = coopMatrixType.getElementType();
    resultType =
        resultType.cast<spirv::CooperativeMatrixNVType>().getElementType();
  }

  auto operandTypeBitWidth = operandType.getIntOrFloatBitWidth();
  auto resultTypeBitWidth = resultType.getIntOrFloatBitWidth();
  auto isSameBitWidth = operandTypeBitWidth == resultTypeBitWidth;

  if (requireSameBitWidth) {
    if (!isSameBitWidth) {
      return op->emitOpError(
                 "expected the same bit widths for operand type and result "
                 "type, but provided ")
             << operandType << " and " << resultType;
    }
    return success();
  }

  if (isSameBitWidth) {
    return op->emitOpError(
               "expected the different bit widths for operand type and result "
               "type, but provided ")
           << operandType << " and " << resultType;
  }
  return success();
}

template <typename MemoryOpTy>
static LogicalResult verifyMemoryAccessAttribute(MemoryOpTy memoryOp) {
  // ODS checks for attributes values. Just need to verify that if the
  // memory-access attribute is Aligned, then the alignment attribute must be
  // present.
  auto *op = memoryOp.getOperation();
  auto memAccessAttr = op->getAttr(kMemoryAccessAttrName);
  if (!memAccessAttr) {
    // Alignment attribute shouldn't be present if memory access attribute is
    // not present.
    if (op->getAttr(kAlignmentAttrName)) {
      return memoryOp.emitOpError(
          "invalid alignment specification without aligned memory access "
          "specification");
    }
    return success();
  }

  auto memAccessVal = memAccessAttr.template cast<IntegerAttr>();
  auto memAccess = spirv::symbolizeMemoryAccess(memAccessVal.getInt());

  if (!memAccess) {
    return memoryOp.emitOpError("invalid memory access specifier: ")
           << memAccessVal;
  }

  if (spirv::bitEnumContains(*memAccess, spirv::MemoryAccess::Aligned)) {
    if (!op->getAttr(kAlignmentAttrName)) {
      return memoryOp.emitOpError("missing alignment value");
    }
  } else {
    if (op->getAttr(kAlignmentAttrName)) {
      return memoryOp.emitOpError(
          "invalid alignment specification with non-aligned memory access "
          "specification");
    }
  }
  return success();
}

// TODO Make sure to merge this and the previous function into one template
// parameterized by memory access attribute name and alignment. Doing so now
// results in VS2017 in producing an internal error (at the call site) that's
// not detailed enough to understand what is happening.
template <typename MemoryOpTy>
static LogicalResult verifySourceMemoryAccessAttribute(MemoryOpTy memoryOp) {
  // ODS checks for attributes values. Just need to verify that if the
  // memory-access attribute is Aligned, then the alignment attribute must be
  // present.
  auto *op = memoryOp.getOperation();
  auto memAccessAttr = op->getAttr(kSourceMemoryAccessAttrName);
  if (!memAccessAttr) {
    // Alignment attribute shouldn't be present if memory access attribute is
    // not present.
    if (op->getAttr(kSourceAlignmentAttrName)) {
      return memoryOp.emitOpError(
          "invalid alignment specification without aligned memory access "
          "specification");
    }
    return success();
  }

  auto memAccessVal = memAccessAttr.template cast<IntegerAttr>();
  auto memAccess = spirv::symbolizeMemoryAccess(memAccessVal.getInt());

  if (!memAccess) {
    return memoryOp.emitOpError("invalid memory access specifier: ")
           << memAccessVal;
  }

  if (spirv::bitEnumContains(*memAccess, spirv::MemoryAccess::Aligned)) {
    if (!op->getAttr(kSourceAlignmentAttrName)) {
      return memoryOp.emitOpError("missing alignment value");
    }
  } else {
    if (op->getAttr(kSourceAlignmentAttrName)) {
      return memoryOp.emitOpError(
          "invalid alignment specification with non-aligned memory access "
          "specification");
    }
  }
  return success();
}

template <typename BarrierOp>
static LogicalResult verifyMemorySemantics(BarrierOp op) {
  // According to the SPIR-V specification:
  // "Despite being a mask and allowing multiple bits to be combined, it is
  // invalid for more than one of these four bits to be set: Acquire, Release,
  // AcquireRelease, or SequentiallyConsistent. Requesting both Acquire and
  // Release semantics is done by setting the AcquireRelease bit, not by setting
  // two bits."
  auto memorySemantics = op.memory_semantics();
  auto atMostOneInSet = spirv::MemorySemantics::Acquire |
                        spirv::MemorySemantics::Release |
                        spirv::MemorySemantics::AcquireRelease |
                        spirv::MemorySemantics::SequentiallyConsistent;

  auto bitCount = llvm::countPopulation(
      static_cast<uint32_t>(memorySemantics & atMostOneInSet));
  if (bitCount > 1) {
    return op.emitError("expected at most one of these four memory constraints "
                        "to be set: `Acquire`, `Release`,"
                        "`AcquireRelease` or `SequentiallyConsistent`");
  }
  return success();
}

template <typename LoadStoreOpTy>
static LogicalResult verifyLoadStorePtrAndValTypes(LoadStoreOpTy op, Value ptr,
                                                   Value val) {
  // ODS already checks ptr is spirv::PointerType. Just check that the pointee
  // type of the pointer and the type of the value are the same
  //
  // TODO: Check that the value type satisfies restrictions of
  // SPIR-V OpLoad/OpStore operations
  if (val.getType() !=
      ptr.getType().cast<spirv::PointerType>().getPointeeType()) {
    return op.emitOpError("mismatch in result type and pointer type");
  }
  return success();
}

template <typename BlockReadWriteOpTy>
static LogicalResult verifyBlockReadWritePtrAndValTypes(BlockReadWriteOpTy op,
                                                        Value ptr, Value val) {
  auto valType = val.getType();
  if (auto valVecTy = valType.dyn_cast<VectorType>())
    valType = valVecTy.getElementType();

  if (valType != ptr.getType().cast<spirv::PointerType>().getPointeeType()) {
    return op.emitOpError("mismatch in result type and pointer type");
  }
  return success();
}

static ParseResult parseVariableDecorations(OpAsmParser &parser,
                                            OperationState &state) {
  auto builtInName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::BuiltIn));
  if (succeeded(parser.parseOptionalKeyword("bind"))) {
    Attribute set, binding;
    // Parse optional descriptor binding
    auto descriptorSetName = llvm::convertToSnakeFromCamelCase(
        stringifyDecoration(spirv::Decoration::DescriptorSet));
    auto bindingName = llvm::convertToSnakeFromCamelCase(
        stringifyDecoration(spirv::Decoration::Binding));
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseLParen() ||
        parser.parseAttribute(set, i32Type, descriptorSetName,
                              state.attributes) ||
        parser.parseComma() ||
        parser.parseAttribute(binding, i32Type, bindingName,
                              state.attributes) ||
        parser.parseRParen()) {
      return failure();
    }
  } else if (succeeded(parser.parseOptionalKeyword(builtInName))) {
    StringAttr builtIn;
    if (parser.parseLParen() ||
        parser.parseAttribute(builtIn, builtInName, state.attributes) ||
        parser.parseRParen()) {
      return failure();
    }
  }

  // Parse other attributes
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

static void printVariableDecorations(Operation *op, OpAsmPrinter &printer,
                                     SmallVectorImpl<StringRef> &elidedAttrs) {
  // Print optional descriptor binding
  auto descriptorSetName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::Binding));
  auto descriptorSet = op->getAttrOfType<IntegerAttr>(descriptorSetName);
  auto binding = op->getAttrOfType<IntegerAttr>(bindingName);
  if (descriptorSet && binding) {
    elidedAttrs.push_back(descriptorSetName);
    elidedAttrs.push_back(bindingName);
    printer << " bind(" << descriptorSet.getInt() << ", " << binding.getInt()
            << ")";
  }

  // Print BuiltIn attribute if present
  auto builtInName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::BuiltIn));
  if (auto builtin = op->getAttrOfType<StringAttr>(builtInName)) {
    printer << " " << builtInName << "(\"" << builtin.getValue() << "\")";
    elidedAttrs.push_back(builtInName);
  }

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

// Get bit width of types.
static unsigned getBitWidth(Type type) {
  if (type.isa<spirv::PointerType>()) {
    // Just return 64 bits for pointer types for now.
    // TODO: Make sure not caller relies on the actual pointer width value.
    return 64;
  }

  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  if (auto vectorType = type.dyn_cast<VectorType>()) {
    assert(vectorType.getElementType().isIntOrFloat());
    return vectorType.getNumElements() *
           vectorType.getElementType().getIntOrFloatBitWidth();
  }
  llvm_unreachable("unhandled bit width computation for type");
}

/// Walks the given type hierarchy with the given indices, potentially down
/// to component granularity, to select an element type. Returns null type and
/// emits errors with the given loc on failure.
static Type
getElementType(Type type, ArrayRef<int32_t> indices,
               function_ref<InFlightDiagnostic(StringRef)> emitErrorFn) {
  if (indices.empty()) {
    emitErrorFn("expected at least one index for spv.CompositeExtract");
    return nullptr;
  }

  for (auto index : indices) {
    if (auto cType = type.dyn_cast<spirv::CompositeType>()) {
      if (cType.hasCompileTimeKnownNumElements() &&
          (index < 0 ||
           static_cast<uint64_t>(index) >= cType.getNumElements())) {
        emitErrorFn("index ") << index << " out of bounds for " << type;
        return nullptr;
      }
      type = cType.getElementType(index);
    } else {
      emitErrorFn("cannot extract from non-composite type ")
          << type << " with index " << index;
      return nullptr;
    }
  }
  return type;
}

static Type
getElementType(Type type, Attribute indices,
               function_ref<InFlightDiagnostic(StringRef)> emitErrorFn) {
  auto indicesArrayAttr = indices.dyn_cast<ArrayAttr>();
  if (!indicesArrayAttr) {
    emitErrorFn("expected a 32-bit integer array attribute for 'indices'");
    return nullptr;
  }
  if (!indicesArrayAttr.size()) {
    emitErrorFn("expected at least one index for spv.CompositeExtract");
    return nullptr;
  }

  SmallVector<int32_t, 2> indexVals;
  for (auto indexAttr : indicesArrayAttr) {
    auto indexIntAttr = indexAttr.dyn_cast<IntegerAttr>();
    if (!indexIntAttr) {
      emitErrorFn("expected an 32-bit integer for index, but found '")
          << indexAttr << "'";
      return nullptr;
    }
    indexVals.push_back(indexIntAttr.getInt());
  }
  return getElementType(type, indexVals, emitErrorFn);
}

static Type getElementType(Type type, Attribute indices, Location loc) {
  auto errorFn = [&](StringRef err) -> InFlightDiagnostic {
    return ::mlir::emitError(loc, err);
  };
  return getElementType(type, indices, errorFn);
}

static Type getElementType(Type type, Attribute indices, OpAsmParser &parser,
                           llvm::SMLoc loc) {
  auto errorFn = [&](StringRef err) -> InFlightDiagnostic {
    return parser.emitError(loc, err);
  };
  return getElementType(type, indices, errorFn);
}

/// Returns true if the given `block` only contains one `spv.mlir.merge` op.
static inline bool isMergeBlock(Block &block) {
  return !block.empty() && std::next(block.begin()) == block.end() &&
         isa<spirv::MergeOp>(block.front());
}

//===----------------------------------------------------------------------===//
// Common parsers and printers
//===----------------------------------------------------------------------===//

// Parses an atomic update op. If the update op does not take a value (like
// AtomicIIncrement) `hasValue` must be false.
static ParseResult parseAtomicUpdateOp(OpAsmParser &parser,
                                       OperationState &state, bool hasValue) {
  spirv::Scope scope;
  spirv::MemorySemantics memoryScope;
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  OpAsmParser::OperandType ptrInfo, valueInfo;
  Type type;
  llvm::SMLoc loc;
  if (parseEnumStrAttr(scope, parser, state, kMemoryScopeAttrName) ||
      parseEnumStrAttr(memoryScope, parser, state, kSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, (hasValue ? 2 : 1)) ||
      parser.getCurrentLocation(&loc) || parser.parseColonType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  SmallVector<Type, 2> operandTypes;
  operandTypes.push_back(ptrType);
  if (hasValue)
    operandTypes.push_back(ptrType.getPointeeType());
  if (parser.resolveOperands(operandInfo, operandTypes, parser.getNameLoc(),
                             state.operands))
    return failure();
  return parser.addTypeToList(ptrType.getPointeeType(), state.types);
}

// Prints an atomic update op.
static void printAtomicUpdateOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName() << " \"";
  auto scopeAttr = op->getAttrOfType<IntegerAttr>(kMemoryScopeAttrName);
  printer << spirv::stringifyScope(
                 static_cast<spirv::Scope>(scopeAttr.getInt()))
          << "\" \"";
  auto memorySemanticsAttr = op->getAttrOfType<IntegerAttr>(kSemanticsAttrName);
  printer << spirv::stringifyMemorySemantics(
                 static_cast<spirv::MemorySemantics>(
                     memorySemanticsAttr.getInt()))
          << "\" " << op->getOperands() << " : " << op->getOperand(0).getType();
}

// Verifies an atomic update op.
static LogicalResult verifyAtomicUpdateOp(Operation *op) {
  auto ptrType = op->getOperand(0).getType().cast<spirv::PointerType>();
  auto elementType = ptrType.getPointeeType();
  if (!elementType.isa<IntegerType>())
    return op->emitOpError(
               "pointer operand must point to an integer value, found ")
           << elementType;

  if (op->getNumOperands() > 1) {
    auto valueType = op->getOperand(1).getType();
    if (valueType != elementType)
      return op->emitOpError("expected value to have the same type as the "
                             "pointer operand's pointee type ")
             << elementType << ", but found " << valueType;
  }
  return success();
}

static ParseResult parseGroupNonUniformArithmeticOp(OpAsmParser &parser,
                                                    OperationState &state) {
  spirv::Scope executionScope;
  spirv::GroupOperation groupOperation;
  OpAsmParser::OperandType valueInfo;
  if (parseEnumStrAttr(executionScope, parser, state,
                       kExecutionScopeAttrName) ||
      parseEnumStrAttr(groupOperation, parser, state,
                       kGroupOperationAttrName) ||
      parser.parseOperand(valueInfo))
    return failure();

  Optional<OpAsmParser::OperandType> clusterSizeInfo;
  if (succeeded(parser.parseOptionalKeyword(kClusterSize))) {
    clusterSizeInfo = OpAsmParser::OperandType();
    if (parser.parseLParen() || parser.parseOperand(*clusterSizeInfo) ||
        parser.parseRParen())
      return failure();
  }

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();

  if (parser.resolveOperand(valueInfo, resultType, state.operands))
    return failure();

  if (clusterSizeInfo.hasValue()) {
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.resolveOperand(*clusterSizeInfo, i32Type, state.operands))
      return failure();
  }

  return parser.addTypeToList(resultType, state.types);
}

static void printGroupNonUniformArithmeticOp(Operation *groupOp,
                                             OpAsmPrinter &printer) {
  printer << groupOp->getName() << " \""
          << stringifyScope(static_cast<spirv::Scope>(
                 groupOp->getAttrOfType<IntegerAttr>(kExecutionScopeAttrName)
                     .getInt()))
          << "\" \""
          << stringifyGroupOperation(static_cast<spirv::GroupOperation>(
                 groupOp->getAttrOfType<IntegerAttr>(kGroupOperationAttrName)
                     .getInt()))
          << "\" " << groupOp->getOperand(0);

  if (groupOp->getNumOperands() > 1)
    printer << " " << kClusterSize << '(' << groupOp->getOperand(1) << ')';
  printer << " : " << groupOp->getResult(0).getType();
}

static LogicalResult verifyGroupNonUniformArithmeticOp(Operation *groupOp) {
  spirv::Scope scope = static_cast<spirv::Scope>(
      groupOp->getAttrOfType<IntegerAttr>(kExecutionScopeAttrName).getInt());
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return groupOp->emitOpError(
        "execution scope must be 'Workgroup' or 'Subgroup'");

  spirv::GroupOperation operation = static_cast<spirv::GroupOperation>(
      groupOp->getAttrOfType<IntegerAttr>(kGroupOperationAttrName).getInt());
  if (operation == spirv::GroupOperation::ClusteredReduce &&
      groupOp->getNumOperands() == 1)
    return groupOp->emitOpError("cluster size operand must be provided for "
                                "'ClusteredReduce' group operation");
  if (groupOp->getNumOperands() > 1) {
    Operation *sizeOp = groupOp->getOperand(1).getDefiningOp();
    int32_t clusterSize = 0;

    // TODO: support specialization constant here.
    if (failed(extractValueFromConstOp(sizeOp, clusterSize)))
      return groupOp->emitOpError(
          "cluster size operand must come from a constant op");

    if (!llvm::isPowerOf2_32(clusterSize))
      return groupOp->emitOpError(
          "cluster size operand must be a power of two");
  }
  return success();
}

static ParseResult parseUnaryOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType operandInfo;
  Type type;
  if (parser.parseOperand(operandInfo) || parser.parseColonType(type) ||
      parser.resolveOperands(operandInfo, type, state.operands)) {
    return failure();
  }
  state.addTypes(type);
  return success();
}

static void printUnaryOp(Operation *unaryOp, OpAsmPrinter &printer) {
  printer << unaryOp->getName() << ' ' << unaryOp->getOperand(0) << " : "
          << unaryOp->getOperand(0).getType();
}

/// Result of a logical op must be a scalar or vector of boolean type.
static Type getUnaryOpResultType(Builder &builder, Type operandType) {
  Type resultType = builder.getIntegerType(1);
  if (auto vecType = operandType.dyn_cast<VectorType>()) {
    return VectorType::get(vecType.getNumElements(), resultType);
  }
  return resultType;
}

static ParseResult parseLogicalUnaryOp(OpAsmParser &parser,
                                       OperationState &state) {
  OpAsmParser::OperandType operandInfo;
  Type type;
  if (parser.parseOperand(operandInfo) || parser.parseColonType(type) ||
      parser.resolveOperand(operandInfo, type, state.operands)) {
    return failure();
  }
  state.addTypes(getUnaryOpResultType(parser.getBuilder(), type));
  return success();
}

static ParseResult parseLogicalBinaryOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  if (parser.parseOperandList(ops, 2) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands)) {
    return failure();
  }
  result.addTypes(getUnaryOpResultType(parser.getBuilder(), type));
  return success();
}

static void printLogicalOp(Operation *logicalOp, OpAsmPrinter &printer) {
  printer << logicalOp->getName() << ' ' << logicalOp->getOperands() << " : "
          << logicalOp->getOperand(0).getType();
}

static ParseResult parseShiftOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  Type baseType;
  Type shiftType;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operandInfo, 2) || parser.parseColon() ||
      parser.parseType(baseType) || parser.parseComma() ||
      parser.parseType(shiftType) ||
      parser.resolveOperands(operandInfo, {baseType, shiftType}, loc,
                             state.operands)) {
    return failure();
  }
  state.addTypes(baseType);
  return success();
}

static void printShiftOp(Operation *op, OpAsmPrinter &printer) {
  Value base = op->getOperand(0);
  Value shift = op->getOperand(1);
  printer << op->getName() << ' ' << base << ", " << shift << " : "
          << base.getType() << ", " << shift.getType();
}

static LogicalResult verifyShiftOp(Operation *op) {
  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    return op->emitError("expected the same type for the first operand and "
                         "result, but provided ")
           << op->getOperand(0).getType() << " and "
           << op->getResult(0).getType();
  }
  return success();
}

static void buildLogicalBinaryOp(OpBuilder &builder, OperationState &state,
                                 Value lhs, Value rhs) {
  assert(lhs.getType() == rhs.getType());

  Type boolType = builder.getI1Type();
  if (auto vecType = lhs.getType().dyn_cast<VectorType>())
    boolType = VectorType::get(vecType.getShape(), boolType);
  state.addTypes(boolType);

  state.addOperands({lhs, rhs});
}

static void buildLogicalUnaryOp(OpBuilder &builder, OperationState &state,
                                Value value) {
  Type boolType = builder.getI1Type();
  if (auto vecType = value.getType().dyn_cast<VectorType>())
    boolType = VectorType::get(vecType.getShape(), boolType);
  state.addTypes(boolType);

  state.addOperands(value);
}

//===----------------------------------------------------------------------===//
// spv.AccessChainOp
//===----------------------------------------------------------------------===//

static Type getElementPtrType(Type type, ValueRange indices, Location baseLoc) {
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    emitError(baseLoc, "'spv.AccessChain' op expected a pointer "
                       "to composite type, but provided ")
        << type;
    return nullptr;
  }

  auto resultType = ptrType.getPointeeType();
  auto resultStorageClass = ptrType.getStorageClass();
  int32_t index = 0;

  for (auto indexSSA : indices) {
    auto cType = resultType.dyn_cast<spirv::CompositeType>();
    if (!cType) {
      emitError(baseLoc,
                "'spv.AccessChain' op cannot extract from non-composite type ")
          << resultType << " with index " << index;
      return nullptr;
    }
    index = 0;
    if (resultType.isa<spirv::StructType>()) {
      Operation *op = indexSSA.getDefiningOp();
      if (!op) {
        emitError(baseLoc, "'spv.AccessChain' op index must be an "
                           "integer spv.Constant to access "
                           "element of spv.struct");
        return nullptr;
      }

      // TODO: this should be relaxed to allow
      // integer literals of other bitwidths.
      if (failed(extractValueFromConstOp(op, index))) {
        emitError(baseLoc,
                  "'spv.AccessChain' index must be an integer spv.Constant to "
                  "access element of spv.struct, but provided ")
            << op->getName();
        return nullptr;
      }
      if (index < 0 || static_cast<uint64_t>(index) >= cType.getNumElements()) {
        emitError(baseLoc, "'spv.AccessChain' op index ")
            << index << " out of bounds for " << resultType;
        return nullptr;
      }
    }
    resultType = cType.getElementType(index);
  }
  return spirv::PointerType::get(resultType, resultStorageClass);
}

void spirv::AccessChainOp::build(OpBuilder &builder, OperationState &state,
                                 Value basePtr, ValueRange indices) {
  auto type = getElementPtrType(basePtr.getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, indices);
}

static ParseResult parseAccessChainOp(OpAsmParser &parser,
                                      OperationState &state) {
  OpAsmParser::OperandType ptrInfo;
  SmallVector<OpAsmParser::OperandType, 4> indicesInfo;
  Type type;
  auto loc = parser.getCurrentLocation();
  SmallVector<Type, 4> indicesTypes;

  if (parser.parseOperand(ptrInfo) ||
      parser.parseOperandList(indicesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptrInfo, type, state.operands)) {
    return failure();
  }

  // Check that the provided indices list is not empty before parsing their
  // type list.
  if (indicesInfo.empty()) {
    return emitError(state.location, "'spv.AccessChain' op expected at "
                                     "least one index ");
  }

  if (parser.parseComma() || parser.parseTypeList(indicesTypes))
    return failure();

  // Check that the indices types list is not empty and that it has a one-to-one
  // mapping to the provided indices.
  if (indicesTypes.size() != indicesInfo.size()) {
    return emitError(state.location, "'spv.AccessChain' op indices "
                                     "types' count must be equal to indices "
                                     "info count");
  }

  if (parser.resolveOperands(indicesInfo, indicesTypes, loc, state.operands))
    return failure();

  auto resultType = getElementPtrType(
      type, llvm::makeArrayRef(state.operands).drop_front(), state.location);
  if (!resultType) {
    return failure();
  }

  state.addTypes(resultType);
  return success();
}

static void print(spirv::AccessChainOp op, OpAsmPrinter &printer) {
  printer << spirv::AccessChainOp::getOperationName() << ' ' << op.base_ptr()
          << '[' << op.indices() << "] : " << op.base_ptr().getType() << ", "
          << op.indices().getTypes();
}

static LogicalResult verify(spirv::AccessChainOp accessChainOp) {
  SmallVector<Value, 4> indices(accessChainOp.indices().begin(),
                                accessChainOp.indices().end());
  auto resultType = getElementPtrType(accessChainOp.base_ptr().getType(),
                                      indices, accessChainOp.getLoc());
  if (!resultType) {
    return failure();
  }

  auto providedResultType =
      accessChainOp.getType().dyn_cast<spirv::PointerType>();
  if (!providedResultType) {
    return accessChainOp.emitOpError(
               "result type must be a pointer, but provided")
           << providedResultType;
  }

  if (resultType != providedResultType) {
    return accessChainOp.emitOpError("invalid result type: expected ")
           << resultType << ", but provided " << providedResultType;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.mlir.addressof
//===----------------------------------------------------------------------===//

void spirv::AddressOfOp::build(OpBuilder &builder, OperationState &state,
                               spirv::GlobalVariableOp var) {
  build(builder, state, var.type(), builder.getSymbolRefAttr(var));
}

static LogicalResult verify(spirv::AddressOfOp addressOfOp) {
  auto varOp = dyn_cast_or_null<spirv::GlobalVariableOp>(
      SymbolTable::lookupNearestSymbolFrom(addressOfOp->getParentOp(),
                                           addressOfOp.variable()));
  if (!varOp) {
    return addressOfOp.emitOpError("expected spv.GlobalVariable symbol");
  }
  if (addressOfOp.pointer().getType() != varOp.type()) {
    return addressOfOp.emitOpError(
        "result type mismatch with the referenced global variable's type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.AtomicCompareExchangeWeak
//===----------------------------------------------------------------------===//

static ParseResult parseAtomicCompareExchangeWeakOp(OpAsmParser &parser,
                                                    OperationState &state) {
  spirv::Scope memoryScope;
  spirv::MemorySemantics equalSemantics, unequalSemantics;
  SmallVector<OpAsmParser::OperandType, 3> operandInfo;
  Type type;
  if (parseEnumStrAttr(memoryScope, parser, state, kMemoryScopeAttrName) ||
      parseEnumStrAttr(equalSemantics, parser, state,
                       kEqualSemanticsAttrName) ||
      parseEnumStrAttr(unequalSemantics, parser, state,
                       kUnequalSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, 3))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  if (parser.resolveOperands(
          operandInfo,
          {ptrType, ptrType.getPointeeType(), ptrType.getPointeeType()},
          parser.getNameLoc(), state.operands))
    return failure();

  return parser.addTypeToList(ptrType.getPointeeType(), state.types);
}

static void print(spirv::AtomicCompareExchangeWeakOp atomOp,
                  OpAsmPrinter &printer) {
  printer << spirv::AtomicCompareExchangeWeakOp::getOperationName() << " \""
          << stringifyScope(atomOp.memory_scope()) << "\" \""
          << stringifyMemorySemantics(atomOp.equal_semantics()) << "\" \""
          << stringifyMemorySemantics(atomOp.unequal_semantics()) << "\" "
          << atomOp.getOperands() << " : " << atomOp.pointer().getType();
}

static LogicalResult verify(spirv::AtomicCompareExchangeWeakOp atomOp) {
  // According to the spec:
  // "The type of Value must be the same as Result Type. The type of the value
  // pointed to by Pointer must be the same as Result Type. This type must also
  // match the type of Comparator."
  if (atomOp.getType() != atomOp.value().getType())
    return atomOp.emitOpError("value operand must have the same type as the op "
                              "result, but found ")
           << atomOp.value().getType() << " vs " << atomOp.getType();

  if (atomOp.getType() != atomOp.comparator().getType())
    return atomOp.emitOpError(
               "comparator operand must have the same type as the op "
               "result, but found ")
           << atomOp.comparator().getType() << " vs " << atomOp.getType();

  Type pointeeType =
      atomOp.pointer().getType().cast<spirv::PointerType>().getPointeeType();
  if (atomOp.getType() != pointeeType)
    return atomOp.emitOpError(
               "pointer operand's pointee type must have the same "
               "as the op result type, but found ")
           << pointeeType << " vs " << atomOp.getType();

  // TODO: Unequal cannot be set to Release or Acquire and Release.
  // In addition, Unequal cannot be set to a stronger memory-order then Equal.

  return success();
}

//===----------------------------------------------------------------------===//
// spv.BitcastOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::BitcastOp bitcastOp) {
  // TODO: The SPIR-V spec validation rules are different for different
  // versions.
  auto operandType = bitcastOp.operand().getType();
  auto resultType = bitcastOp.result().getType();
  if (operandType == resultType) {
    return bitcastOp.emitError(
        "result type must be different from operand type");
  }
  if (operandType.isa<spirv::PointerType>() &&
      !resultType.isa<spirv::PointerType>()) {
    return bitcastOp.emitError(
        "unhandled bit cast conversion from pointer type to non-pointer type");
  }
  if (!operandType.isa<spirv::PointerType>() &&
      resultType.isa<spirv::PointerType>()) {
    return bitcastOp.emitError(
        "unhandled bit cast conversion from non-pointer type to pointer type");
  }
  auto operandBitWidth = getBitWidth(operandType);
  auto resultBitWidth = getBitWidth(resultType);
  if (operandBitWidth != resultBitWidth) {
    return bitcastOp.emitOpError("mismatch in result type bitwidth ")
           << resultBitWidth << " and operand type bitwidth "
           << operandBitWidth;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.BranchOp
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange>
spirv::BranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return targetOperandsMutable();
}

//===----------------------------------------------------------------------===//
// spv.BranchConditionalOp
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange>
spirv::BranchConditionalOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < 2 && "invalid successor index");
  return index == kTrueIndex ? trueTargetOperandsMutable()
                             : falseTargetOperandsMutable();
}

static ParseResult parseBranchConditionalOp(OpAsmParser &parser,
                                            OperationState &state) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType condInfo;
  Block *dest;

  // Parse the condition.
  Type boolTy = builder.getI1Type();
  if (parser.parseOperand(condInfo) ||
      parser.resolveOperand(condInfo, boolTy, state.operands))
    return failure();

  // Parse the optional branch weights.
  if (succeeded(parser.parseOptionalLSquare())) {
    IntegerAttr trueWeight, falseWeight;
    NamedAttrList weights;

    auto i32Type = builder.getIntegerType(32);
    if (parser.parseAttribute(trueWeight, i32Type, "weight", weights) ||
        parser.parseComma() ||
        parser.parseAttribute(falseWeight, i32Type, "weight", weights) ||
        parser.parseRSquare())
      return failure();

    state.addAttribute(kBranchWeightAttrName,
                       builder.getArrayAttr({trueWeight, falseWeight}));
  }

  // Parse the true branch.
  SmallVector<Value, 4> trueOperands;
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, trueOperands))
    return failure();
  state.addSuccessors(dest);
  state.addOperands(trueOperands);

  // Parse the false branch.
  SmallVector<Value, 4> falseOperands;
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, falseOperands))
    return failure();
  state.addSuccessors(dest);
  state.addOperands(falseOperands);
  state.addAttribute(
      spirv::BranchConditionalOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({1, static_cast<int32_t>(trueOperands.size()),
                                static_cast<int32_t>(falseOperands.size())}));

  return success();
}

static void print(spirv::BranchConditionalOp branchOp, OpAsmPrinter &printer) {
  printer << spirv::BranchConditionalOp::getOperationName() << ' '
          << branchOp.condition();

  if (auto weights = branchOp.branch_weights()) {
    printer << " [";
    llvm::interleaveComma(weights->getValue(), printer, [&](Attribute a) {
      printer << a.cast<IntegerAttr>().getInt();
    });
    printer << "]";
  }

  printer << ", ";
  printer.printSuccessorAndUseList(branchOp.getTrueBlock(),
                                   branchOp.getTrueBlockArguments());
  printer << ", ";
  printer.printSuccessorAndUseList(branchOp.getFalseBlock(),
                                   branchOp.getFalseBlockArguments());
}

static LogicalResult verify(spirv::BranchConditionalOp branchOp) {
  if (auto weights = branchOp.branch_weights()) {
    if (weights->getValue().size() != 2) {
      return branchOp.emitOpError("must have exactly two branch weights");
    }
    if (llvm::all_of(*weights, [](Attribute attr) {
          return attr.cast<IntegerAttr>().getValue().isNullValue();
        }))
      return branchOp.emitOpError("branch weights cannot both be zero");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CompositeConstruct
//===----------------------------------------------------------------------===//

static ParseResult parseCompositeConstructOp(OpAsmParser &parser,
                                             OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  Type type;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(operands) || parser.parseColonType(type)) {
    return failure();
  }
  auto cType = type.dyn_cast<spirv::CompositeType>();
  if (!cType) {
    return parser.emitError(
               loc, "result type must be a composite type, but provided ")
           << type;
  }

  if (cType.hasCompileTimeKnownNumElements() &&
      operands.size() != cType.getNumElements()) {
    return parser.emitError(loc, "has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided " << operands.size();
  }
  // TODO: Add support for constructing a vector type from the vector operands.
  // According to the spec: "for constructing a vector, the operands may
  // also be vectors with the same component type as the Result Type component
  // type".
  SmallVector<Type, 4> elementTypes;
  elementTypes.reserve(operands.size());
  for (auto index : llvm::seq<uint32_t>(0, operands.size())) {
    elementTypes.push_back(cType.getElementType(index));
  }
  state.addTypes(type);
  return parser.resolveOperands(operands, elementTypes, loc, state.operands);
}

static void print(spirv::CompositeConstructOp compositeConstructOp,
                  OpAsmPrinter &printer) {
  printer << spirv::CompositeConstructOp::getOperationName() << " "
          << compositeConstructOp.constituents() << " : "
          << compositeConstructOp.getResult().getType();
}

static LogicalResult verify(spirv::CompositeConstructOp compositeConstructOp) {
  auto cType = compositeConstructOp.getType().cast<spirv::CompositeType>();
  SmallVector<Value, 4> constituents(compositeConstructOp.constituents());

  if (cType.isa<spirv::CooperativeMatrixNVType>()) {
    if (constituents.size() != 1)
      return compositeConstructOp.emitError(
                 "has incorrect number of operands: expected ")
             << "1, but provided " << constituents.size();
  } else if (constituents.size() != cType.getNumElements()) {
    return compositeConstructOp.emitError(
               "has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided "
           << constituents.size();
  }

  for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
    if (constituents[index].getType() != cType.getElementType(index)) {
      return compositeConstructOp.emitError(
                 "operand type mismatch: expected operand type ")
             << cType.getElementType(index) << ", but provided "
             << constituents[index].getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CompositeExtractOp
//===----------------------------------------------------------------------===//

void spirv::CompositeExtractOp::build(OpBuilder &builder, OperationState &state,
                                      Value composite,
                                      ArrayRef<int32_t> indices) {
  auto indexAttr = builder.getI32ArrayAttr(indices);
  auto elementType =
      getElementType(composite.getType(), indexAttr, state.location);
  if (!elementType) {
    return;
  }
  build(builder, state, elementType, composite, indexAttr);
}

static ParseResult parseCompositeExtractOp(OpAsmParser &parser,
                                           OperationState &state) {
  OpAsmParser::OperandType compositeInfo;
  Attribute indicesAttr;
  Type compositeType;
  llvm::SMLoc attrLocation;

  if (parser.parseOperand(compositeInfo) ||
      parser.getCurrentLocation(&attrLocation) ||
      parser.parseAttribute(indicesAttr, kIndicesAttrName, state.attributes) ||
      parser.parseColonType(compositeType) ||
      parser.resolveOperand(compositeInfo, compositeType, state.operands)) {
    return failure();
  }

  Type resultType =
      getElementType(compositeType, indicesAttr, parser, attrLocation);
  if (!resultType) {
    return failure();
  }
  state.addTypes(resultType);
  return success();
}

static void print(spirv::CompositeExtractOp compositeExtractOp,
                  OpAsmPrinter &printer) {
  printer << spirv::CompositeExtractOp::getOperationName() << ' '
          << compositeExtractOp.composite() << compositeExtractOp.indices()
          << " : " << compositeExtractOp.composite().getType();
}

static LogicalResult verify(spirv::CompositeExtractOp compExOp) {
  auto indicesArrayAttr = compExOp.indices().dyn_cast<ArrayAttr>();
  auto resultType = getElementType(compExOp.composite().getType(),
                                   indicesArrayAttr, compExOp.getLoc());
  if (!resultType)
    return failure();

  if (resultType != compExOp.getType()) {
    return compExOp.emitOpError("invalid result type: expected ")
           << resultType << " but provided " << compExOp.getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CompositeInsert
//===----------------------------------------------------------------------===//

void spirv::CompositeInsertOp::build(OpBuilder &builder, OperationState &state,
                                     Value object, Value composite,
                                     ArrayRef<int32_t> indices) {
  auto indexAttr = builder.getI32ArrayAttr(indices);
  build(builder, state, composite.getType(), object, composite, indexAttr);
}

static ParseResult parseCompositeInsertOp(OpAsmParser &parser,
                                          OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  Type objectType, compositeType;
  Attribute indicesAttr;
  auto loc = parser.getCurrentLocation();

  return failure(
      parser.parseOperandList(operands, 2) ||
      parser.parseAttribute(indicesAttr, kIndicesAttrName, state.attributes) ||
      parser.parseColonType(objectType) ||
      parser.parseKeywordType("into", compositeType) ||
      parser.resolveOperands(operands, {objectType, compositeType}, loc,
                             state.operands) ||
      parser.addTypesToList(compositeType, state.types));
}

static LogicalResult verify(spirv::CompositeInsertOp compositeInsertOp) {
  auto indicesArrayAttr = compositeInsertOp.indices().dyn_cast<ArrayAttr>();
  auto objectType =
      getElementType(compositeInsertOp.composite().getType(), indicesArrayAttr,
                     compositeInsertOp.getLoc());
  if (!objectType)
    return failure();

  if (objectType != compositeInsertOp.object().getType()) {
    return compositeInsertOp.emitOpError("object operand type should be ")
           << objectType << ", but found "
           << compositeInsertOp.object().getType();
  }

  if (compositeInsertOp.composite().getType() != compositeInsertOp.getType()) {
    return compositeInsertOp.emitOpError("result type should be the same as "
                                         "the composite type, but found ")
           << compositeInsertOp.composite().getType() << " vs "
           << compositeInsertOp.getType();
  }

  return success();
}

static void print(spirv::CompositeInsertOp compositeInsertOp,
                  OpAsmPrinter &printer) {
  printer << spirv::CompositeInsertOp::getOperationName() << " "
          << compositeInsertOp.object() << ", " << compositeInsertOp.composite()
          << compositeInsertOp.indices() << " : "
          << compositeInsertOp.object().getType() << " into "
          << compositeInsertOp.composite().getType();
}

//===----------------------------------------------------------------------===//
// spv.Constant
//===----------------------------------------------------------------------===//

static ParseResult parseConstantOp(OpAsmParser &parser, OperationState &state) {
  Attribute value;
  if (parser.parseAttribute(value, kValueAttrName, state.attributes))
    return failure();

  Type type = value.getType();
  if (type.isa<NoneType, TensorType>()) {
    if (parser.parseColonType(type))
      return failure();
  }

  return parser.addTypeToList(type, state.types);
}

static void print(spirv::ConstantOp constOp, OpAsmPrinter &printer) {
  printer << spirv::ConstantOp::getOperationName() << ' ' << constOp.value();
  if (constOp.getType().isa<spirv::ArrayType>())
    printer << " : " << constOp.getType();
}

static LogicalResult verify(spirv::ConstantOp constOp) {
  auto opType = constOp.getType();
  auto value = constOp.value();
  auto valueType = value.getType();

  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  if (value.isa<IntegerAttr, FloatAttr>()) {
    if (valueType != opType)
      return constOp.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  }
  if (value.isa<DenseIntOrFPElementsAttr, SparseElementsAttr>()) {
    if (valueType == opType)
      return success();
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    auto shapedType = valueType.dyn_cast<ShapedType>();
    if (!arrayType) {
      return constOp.emitOpError(
          "must have spv.array result type for array value");
    }

    int numElements = arrayType.getNumElements();
    auto opElemType = arrayType.getElementType();
    while (auto t = opElemType.dyn_cast<spirv::ArrayType>()) {
      numElements *= t.getNumElements();
      opElemType = t.getElementType();
    }
    if (!opElemType.isIntOrFloat())
      return constOp.emitOpError("only support nested array result type");

    auto valueElemType = shapedType.getElementType();
    if (valueElemType != opElemType) {
      return constOp.emitOpError("result element type (")
             << opElemType << ") does not match value element type ("
             << valueElemType << ")";
    }

    if (numElements != shapedType.getNumElements()) {
      return constOp.emitOpError("result number of elements (")
             << numElements << ") does not match value number of elements ("
             << shapedType.getNumElements() << ")";
    }
    return success();
  }
  if (auto attayAttr = value.dyn_cast<ArrayAttr>()) {
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    if (!arrayType)
      return constOp.emitOpError(
          "must have spv.array result type for array value");
    Type elemType = arrayType.getElementType();
    for (Attribute element : attayAttr.getValue()) {
      if (element.getType() != elemType)
        return constOp.emitOpError("has array element whose type (")
               << element.getType()
               << ") does not match the result element type (" << elemType
               << ')';
    }
    return success();
  }
  return constOp.emitOpError("cannot have value of type ") << valueType;
}

bool spirv::ConstantOp::isBuildableWith(Type type) {
  // Must be valid SPIR-V type first.
  if (!type.isa<spirv::SPIRVType>())
    return false;

  if (isa<SPIRVDialect>(type.getDialect())) {
    // TODO: support constant struct
    return type.isa<spirv::ArrayType>();
  }

  return true;
}

spirv::ConstantOp spirv::ConstantOp::getZero(Type type, Location loc,
                                             OpBuilder &builder) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    if (width == 1)
      return builder.create<spirv::ConstantOp>(loc, type,
                                               builder.getBoolAttr(false));
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getIntegerAttr(type, APInt(width, 0)));
  }
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getFloatAttr(floatType, 0.0));
  }
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    Type elemType = vectorType.getElementType();
    if (elemType.isa<IntegerType>()) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseElementsAttr::get(vectorType,
                                 IntegerAttr::get(elemType, 0.0).getValue()));
    }
    if (elemType.isa<FloatType>()) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseFPElementsAttr::get(vectorType,
                                   FloatAttr::get(elemType, 0.0).getValue()));
    }
  }

  llvm_unreachable("unimplemented types for ConstantOp::getZero()");
}

spirv::ConstantOp spirv::ConstantOp::getOne(Type type, Location loc,
                                            OpBuilder &builder) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    unsigned width = intType.getWidth();
    if (width == 1)
      return builder.create<spirv::ConstantOp>(loc, type,
                                               builder.getBoolAttr(true));
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getIntegerAttr(type, APInt(width, 1)));
  }
  if (auto floatType = type.dyn_cast<FloatType>()) {
    return builder.create<spirv::ConstantOp>(
        loc, type, builder.getFloatAttr(floatType, 1.0));
  }
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    Type elemType = vectorType.getElementType();
    if (elemType.isa<IntegerType>()) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseElementsAttr::get(vectorType,
                                 IntegerAttr::get(elemType, 1.0).getValue()));
    }
    if (elemType.isa<FloatType>()) {
      return builder.create<spirv::ConstantOp>(
          loc, type,
          DenseFPElementsAttr::get(vectorType,
                                   FloatAttr::get(elemType, 1.0).getValue()));
    }
  }

  llvm_unreachable("unimplemented types for ConstantOp::getOne()");
}

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

void spirv::EntryPointOp::build(OpBuilder &builder, OperationState &state,
                                spirv::ExecutionModel executionModel,
                                spirv::FuncOp function,
                                ArrayRef<Attribute> interfaceVars) {
  build(builder, state,
        spirv::ExecutionModelAttr::get(builder.getContext(), executionModel),
        builder.getSymbolRefAttr(function),
        builder.getArrayAttr(interfaceVars));
}

static ParseResult parseEntryPointOp(OpAsmParser &parser,
                                     OperationState &state) {
  spirv::ExecutionModel execModel;
  SmallVector<OpAsmParser::OperandType, 0> identifiers;
  SmallVector<Type, 0> idTypes;
  SmallVector<Attribute, 4> interfaceVars;

  FlatSymbolRefAttr fn;
  if (parseEnumStrAttr(execModel, parser, state) ||
      parser.parseAttribute(fn, Type(), kFnNameAttrName, state.attributes)) {
    return failure();
  }

  if (!parser.parseOptionalComma()) {
    // Parse the interface variables
    do {
      // The name of the interface variable attribute isnt important
      auto attrName = "var_symbol";
      FlatSymbolRefAttr var;
      NamedAttrList attrs;
      if (parser.parseAttribute(var, Type(), attrName, attrs)) {
        return failure();
      }
      interfaceVars.push_back(var);
    } while (!parser.parseOptionalComma());
  }
  state.addAttribute(kInterfaceAttrName,
                     parser.getBuilder().getArrayAttr(interfaceVars));
  return success();
}

static void print(spirv::EntryPointOp entryPointOp, OpAsmPrinter &printer) {
  printer << spirv::EntryPointOp::getOperationName() << " \""
          << stringifyExecutionModel(entryPointOp.execution_model()) << "\" ";
  printer.printSymbolName(entryPointOp.fn());
  auto interfaceVars = entryPointOp.interface().getValue();
  if (!interfaceVars.empty()) {
    printer << ", ";
    llvm::interleaveComma(interfaceVars, printer);
  }
}

static LogicalResult verify(spirv::EntryPointOp entryPointOp) {
  // Checks for fn and interface symbol reference are done in spirv::ModuleOp
  // verification.
  return success();
}

//===----------------------------------------------------------------------===//
// spv.ExecutionMode
//===----------------------------------------------------------------------===//

void spirv::ExecutionModeOp::build(OpBuilder &builder, OperationState &state,
                                   spirv::FuncOp function,
                                   spirv::ExecutionMode executionMode,
                                   ArrayRef<int32_t> params) {
  build(builder, state, builder.getSymbolRefAttr(function),
        spirv::ExecutionModeAttr::get(builder.getContext(), executionMode),
        builder.getI32ArrayAttr(params));
}

static ParseResult parseExecutionModeOp(OpAsmParser &parser,
                                        OperationState &state) {
  spirv::ExecutionMode execMode;
  Attribute fn;
  if (parser.parseAttribute(fn, kFnNameAttrName, state.attributes) ||
      parseEnumStrAttr(execMode, parser, state)) {
    return failure();
  }

  SmallVector<int32_t, 4> values;
  Type i32Type = parser.getBuilder().getIntegerType(32);
  while (!parser.parseOptionalComma()) {
    NamedAttrList attr;
    Attribute value;
    if (parser.parseAttribute(value, i32Type, "value", attr)) {
      return failure();
    }
    values.push_back(value.cast<IntegerAttr>().getInt());
  }
  state.addAttribute(kValuesAttrName,
                     parser.getBuilder().getI32ArrayAttr(values));
  return success();
}

static void print(spirv::ExecutionModeOp execModeOp, OpAsmPrinter &printer) {
  printer << spirv::ExecutionModeOp::getOperationName() << " ";
  printer.printSymbolName(execModeOp.fn());
  printer << " \"" << stringifyExecutionMode(execModeOp.execution_mode())
          << "\"";
  auto values = execModeOp.values();
  if (!values.size())
    return;
  printer << ", ";
  llvm::interleaveComma(values, printer, [&](Attribute a) {
    printer << a.cast<IntegerAttr>().getInt();
  });
}

//===----------------------------------------------------------------------===//
// spv.func
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (impl::parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                                   argTypes, argAttrs, isVariadic, resultTypes,
                                   resultAttrs))
    return failure();

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  state.addAttribute(impl::getTypeAttrName(), TypeAttr::get(fnType));

  // Parse the optional function control keyword.
  spirv::FunctionControl fnControl;
  if (parseEnumStrAttr(fnControl, parser, state))
    return failure();

  // If additional attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  // Add the attributes to the function arguments.
  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());
  impl::addArgAndResultAttrs(builder, state, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = state.addRegion();
  OptionalParseResult result = parser.parseOptionalRegion(
      *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes);
  return failure(result.hasValue() && failed(*result));
}

static void print(spirv::FuncOp fnOp, OpAsmPrinter &printer) {
  // Print function name, signature, and control.
  printer << spirv::FuncOp::getOperationName() << " ";
  printer.printSymbolName(fnOp.sym_name());
  auto fnType = fnOp.getType();
  impl::printFunctionSignature(printer, fnOp, fnType.getInputs(),
                               /*isVariadic=*/false, fnType.getResults());
  printer << " \"" << spirv::stringifyFunctionControl(fnOp.function_control())
          << "\"";
  impl::printFunctionAttributes(
      printer, fnOp, fnType.getNumInputs(), fnType.getNumResults(),
      {spirv::attributeName<spirv::FunctionControl>()});

  // Print the body if this is not an external function.
  Region &body = fnOp.body();
  if (!body.empty())
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
}

LogicalResult spirv::FuncOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  if (getType().getNumResults() > 1)
    return emitOpError("cannot have more than one result");
  return success();
}

LogicalResult spirv::FuncOp::verifyBody() {
  FunctionType fnType = getType();

  auto walkResult = walk([fnType](Operation *op) -> WalkResult {
    if (auto retOp = dyn_cast<spirv::ReturnOp>(op)) {
      if (fnType.getNumResults() != 0)
        return retOp.emitOpError("cannot be used in functions returning value");
    } else if (auto retOp = dyn_cast<spirv::ReturnValueOp>(op)) {
      if (fnType.getNumResults() != 1)
        return retOp.emitOpError(
                   "returns 1 value but enclosing function requires ")
               << fnType.getNumResults() << " results";

      auto retOperandType = retOp.value().getType();
      auto fnResultType = fnType.getResult(0);
      if (retOperandType != fnResultType)
        return retOp.emitOpError(" return value's type (")
               << retOperandType << ") mismatch with function's result type ("
               << fnResultType << ")";
    }
    return WalkResult::advance();
  });

  // TODO: verify other bits like linkage type.

  return failure(walkResult.wasInterrupted());
}

void spirv::FuncOp::build(OpBuilder &builder, OperationState &state,
                          StringRef name, FunctionType type,
                          spirv::FunctionControl control,
                          ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.addAttribute(spirv::attributeName<spirv::FunctionControl>(),
                     builder.getI32IntegerAttr(static_cast<uint32_t>(control)));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
}

// CallableOpInterface
Region *spirv::FuncOp::getCallableRegion() {
  return isExternal() ? nullptr : &body();
}

// CallableOpInterface
ArrayRef<Type> spirv::FuncOp::getCallableResults() {
  return getType().getResults();
}

//===----------------------------------------------------------------------===//
// spv.FunctionCall
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::FunctionCallOp functionCallOp) {
  auto fnName = functionCallOp.callee();

  auto funcOp =
      dyn_cast_or_null<spirv::FuncOp>(SymbolTable::lookupNearestSymbolFrom(
          functionCallOp->getParentOp(), fnName));
  if (!funcOp) {
    return functionCallOp.emitOpError("callee function '")
           << fnName << "' not found in nearest symbol table";
  }

  auto functionType = funcOp.getType();

  if (functionCallOp.getNumResults() > 1) {
    return functionCallOp.emitOpError(
               "expected callee function to have 0 or 1 result, but provided ")
           << functionCallOp.getNumResults();
  }

  if (functionType.getNumInputs() != functionCallOp.getNumOperands()) {
    return functionCallOp.emitOpError(
               "has incorrect number of operands for callee: expected ")
           << functionType.getNumInputs() << ", but provided "
           << functionCallOp.getNumOperands();
  }

  for (uint32_t i = 0, e = functionType.getNumInputs(); i != e; ++i) {
    if (functionCallOp.getOperand(i).getType() != functionType.getInput(i)) {
      return functionCallOp.emitOpError(
                 "operand type mismatch: expected operand type ")
             << functionType.getInput(i) << ", but provided "
             << functionCallOp.getOperand(i).getType() << " for operand number "
             << i;
    }
  }

  if (functionType.getNumResults() != functionCallOp.getNumResults()) {
    return functionCallOp.emitOpError(
               "has incorrect number of results has for callee: expected ")
           << functionType.getNumResults() << ", but provided "
           << functionCallOp.getNumResults();
  }

  if (functionCallOp.getNumResults() &&
      (functionCallOp.getResult(0).getType() != functionType.getResult(0))) {
    return functionCallOp.emitOpError("result type mismatch: expected ")
           << functionType.getResult(0) << ", but provided "
           << functionCallOp.getResult(0).getType();
  }

  return success();
}

CallInterfaceCallable spirv::FunctionCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(kCallee);
}

Operation::operand_range spirv::FunctionCallOp::getArgOperands() {
  return arguments();
}

//===----------------------------------------------------------------------===//
// spv.GlobalVariable
//===----------------------------------------------------------------------===//

void spirv::GlobalVariableOp::build(OpBuilder &builder, OperationState &state,
                                    Type type, StringRef name,
                                    unsigned descriptorSet, unsigned binding) {
  build(builder, state, TypeAttr::get(type), builder.getStringAttr(name),
        nullptr);
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::DescriptorSet),
      builder.getI32IntegerAttr(descriptorSet));
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::Binding),
      builder.getI32IntegerAttr(binding));
}

void spirv::GlobalVariableOp::build(OpBuilder &builder, OperationState &state,
                                    Type type, StringRef name,
                                    spirv::BuiltIn builtin) {
  build(builder, state, TypeAttr::get(type), builder.getStringAttr(name),
        nullptr);
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::BuiltIn),
      builder.getStringAttr(spirv::stringifyBuiltIn(builtin)));
}

static ParseResult parseGlobalVariableOp(OpAsmParser &parser,
                                         OperationState &state) {
  // Parse variable name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes)) {
    return failure();
  }

  // Parse optional initializer
  if (succeeded(parser.parseOptionalKeyword(kInitializerAttrName))) {
    FlatSymbolRefAttr initSymbol;
    if (parser.parseLParen() ||
        parser.parseAttribute(initSymbol, Type(), kInitializerAttrName,
                              state.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, state)) {
    return failure();
  }

  Type type;
  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type)) {
    return failure();
  }
  if (!type.isa<spirv::PointerType>()) {
    return parser.emitError(loc, "expected spv.ptr type");
  }
  state.addAttribute(kTypeAttrName, TypeAttr::get(type));

  return success();
}

static void print(spirv::GlobalVariableOp varOp, OpAsmPrinter &printer) {
  auto *op = varOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  printer << spirv::GlobalVariableOp::getOperationName();

  // Print variable name.
  printer << ' ';
  printer.printSymbolName(varOp.sym_name());
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());

  // Print optional initializer
  if (auto initializer = varOp.initializer()) {
    printer << " " << kInitializerAttrName << '(';
    printer.printSymbolName(initializer.getValue());
    printer << ')';
    elidedAttrs.push_back(kInitializerAttrName);
  }

  elidedAttrs.push_back(kTypeAttrName);
  printVariableDecorations(op, printer, elidedAttrs);
  printer << " : " << varOp.type();
}

static LogicalResult verify(spirv::GlobalVariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  // Also, Function storage class is reserved by spv.Variable.
  auto storageClass = varOp.storageClass();
  if (storageClass == spirv::StorageClass::Generic ||
      storageClass == spirv::StorageClass::Function) {
    return varOp.emitOpError("storage class cannot be '")
           << stringifyStorageClass(storageClass) << "'";
  }

  if (auto init =
          varOp->getAttrOfType<FlatSymbolRefAttr>(kInitializerAttrName)) {
    Operation *initOp = SymbolTable::lookupNearestSymbolFrom(
        varOp->getParentOp(), init.getValue());
    // TODO: Currently only variable initialization with specialization
    // constants and other variables is supported. They could be normal
    // constants in the module scope as well.
    if (!initOp ||
        !isa<spirv::GlobalVariableOp, spirv::SpecConstantOp>(initOp)) {
      return varOp.emitOpError("initializer must be result of a "
                               "spv.SpecConstant or spv.GlobalVariable op");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupBroadcast
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::GroupBroadcastOp broadcastOp) {
  spirv::Scope scope = broadcastOp.execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return broadcastOp.emitOpError(
        "execution scope must be 'Workgroup' or 'Subgroup'");

  if (auto localIdTy = broadcastOp.localid().getType().dyn_cast<VectorType>())
    if (!(localIdTy.getNumElements() == 2 || localIdTy.getNumElements() == 3))
      return broadcastOp.emitOpError("localid is a vector and can be with only "
                                     " 2 or 3 components, actual number is ")
             << localIdTy.getNumElements();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformBallotOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::GroupNonUniformBallotOp ballotOp) {
  spirv::Scope scope = ballotOp.execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return ballotOp.emitOpError(
        "execution scope must be 'Workgroup' or 'Subgroup'");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformBroadcast
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::GroupNonUniformBroadcastOp broadcastOp) {
  spirv::Scope scope = broadcastOp.execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return broadcastOp.emitOpError(
        "execution scope must be 'Workgroup' or 'Subgroup'");

  // SPIR-V spec: "Before version 1.5, Id must come from a
  // constant instruction.
  auto targetEnv = spirv::getDefaultTargetEnv(broadcastOp.getContext());
  if (auto spirvModule = broadcastOp->getParentOfType<spirv::ModuleOp>())
    targetEnv = spirv::lookupTargetEnvOrDefault(spirvModule);

  if (targetEnv.getVersion() < spirv::Version::V_1_5) {
    auto *idOp = broadcastOp.id().getDefiningOp();
    if (!idOp || !isa<spirv::ConstantOp,           // for normal constant
                      spirv::ReferenceOfOp>(idOp)) // for spec constant
      return broadcastOp.emitOpError("id must be the result of a constant op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SubgroupBlockReadINTEL
//===----------------------------------------------------------------------===//

static ParseResult parseSubgroupBlockReadINTELOp(OpAsmParser &parser,
                                                 OperationState &state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::OperandType ptrInfo;
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) || parser.parseOperand(ptrInfo) ||
      parser.parseColon() || parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (auto valVecTy = elementType.dyn_cast<VectorType>())
    ptrType = spirv::PointerType::get(valVecTy.getElementType(), storageClass);

  if (parser.resolveOperand(ptrInfo, ptrType, state.operands)) {
    return failure();
  }

  state.addTypes(elementType);
  return success();
}

static void print(spirv::SubgroupBlockReadINTELOp blockReadOp,
                  OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  printer << spirv::SubgroupBlockReadINTELOp::getOperationName() << " "
          << blockReadOp.ptr();
  printer << " : " << blockReadOp.getType();
}

static LogicalResult verify(spirv::SubgroupBlockReadINTELOp blockReadOp) {
  if (failed(verifyBlockReadWritePtrAndValTypes(blockReadOp, blockReadOp.ptr(),
                                                blockReadOp.value())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SubgroupBlockWriteINTEL
//===----------------------------------------------------------------------===//

static ParseResult parseSubgroupBlockWriteINTELOp(OpAsmParser &parser,
                                                  OperationState &state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  auto loc = parser.getCurrentLocation();
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) ||
      parser.parseOperandList(operandInfo, 2) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (auto valVecTy = elementType.dyn_cast<VectorType>())
    ptrType = spirv::PointerType::get(valVecTy.getElementType(), storageClass);

  if (parser.resolveOperands(operandInfo, {ptrType, elementType}, loc,
                             state.operands)) {
    return failure();
  }
  return success();
}

static void print(spirv::SubgroupBlockWriteINTELOp blockWriteOp,
                  OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  printer << spirv::SubgroupBlockWriteINTELOp::getOperationName() << " "
          << blockWriteOp.ptr() << ", " << blockWriteOp.value();
  printer << " : " << blockWriteOp.value().getType();
}

static LogicalResult verify(spirv::SubgroupBlockWriteINTELOp blockWriteOp) {
  if (failed(verifyBlockReadWritePtrAndValTypes(
          blockWriteOp, blockWriteOp.ptr(), blockWriteOp.value())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformElectOp
//===----------------------------------------------------------------------===//

void spirv::GroupNonUniformElectOp::build(OpBuilder &builder,
                                          OperationState &state,
                                          spirv::Scope scope) {
  build(builder, state, builder.getI1Type(), scope);
}

static LogicalResult verify(spirv::GroupNonUniformElectOp groupOp) {
  spirv::Scope scope = groupOp.execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return groupOp.emitOpError(
        "execution scope must be 'Workgroup' or 'Subgroup'");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.LoadOp
//===----------------------------------------------------------------------===//

void spirv::LoadOp::build(OpBuilder &builder, OperationState &state,
                          Value basePtr, MemoryAccessAttr memoryAccess,
                          IntegerAttr alignment) {
  auto ptrType = basePtr.getType().cast<spirv::PointerType>();
  build(builder, state, ptrType.getPointeeType(), basePtr, memoryAccess,
        alignment);
}

static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  OpAsmParser::OperandType ptrInfo;
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) || parser.parseOperand(ptrInfo) ||
      parseMemoryAccessAttributes(parser, state) ||
      parser.parseOptionalAttrDict(state.attributes) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser.resolveOperand(ptrInfo, ptrType, state.operands)) {
    return failure();
  }

  state.addTypes(elementType);
  return success();
}

static void print(spirv::LoadOp loadOp, OpAsmPrinter &printer) {
  auto *op = loadOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      loadOp.ptr().getType().cast<spirv::PointerType>().getStorageClass());
  printer << spirv::LoadOp::getOperationName() << " \"" << sc << "\" "
          << loadOp.ptr();

  printMemoryAccessAttribute(loadOp, printer, elidedAttrs);

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  printer << " : " << loadOp.getType();
}

static LogicalResult verify(spirv::LoadOp loadOp) {
  // SPIR-V spec : "Result Type is the type of the loaded object. It must be a
  // type with fixed size; i.e., it cannot be, nor include, any
  // OpTypeRuntimeArray types."
  if (failed(verifyLoadStorePtrAndValTypes(loadOp, loadOp.ptr(),
                                           loadOp.value()))) {
    return failure();
  }
  return verifyMemoryAccessAttribute(loadOp);
}

//===----------------------------------------------------------------------===//
// spv.mlir.loop
//===----------------------------------------------------------------------===//

void spirv::LoopOp::build(OpBuilder &builder, OperationState &state) {
  state.addAttribute("loop_control",
                     builder.getI32IntegerAttr(
                         static_cast<uint32_t>(spirv::LoopControl::None)));
  state.addRegion();
}

static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &state) {
  if (parseControlAttribute<spirv::LoopControl>(parser, state))
    return failure();
  return parser.parseRegion(*state.addRegion(), /*arguments=*/{},
                            /*argTypes=*/{});
}

static void print(spirv::LoopOp loopOp, OpAsmPrinter &printer) {
  auto *op = loopOp.getOperation();

  printer << spirv::LoopOp::getOperationName();
  auto control = loopOp.loop_control();
  if (control != spirv::LoopControl::None)
    printer << " control(" << spirv::stringifyLoopControl(control) << ")";
  printer.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

/// Returns true if the given `srcBlock` contains only one `spv.Branch` to the
/// given `dstBlock`.
static inline bool hasOneBranchOpTo(Block &srcBlock, Block &dstBlock) {
  // Check that there is only one op in the `srcBlock`.
  if (!llvm::hasSingleElement(srcBlock))
    return false;

  auto branchOp = dyn_cast<spirv::BranchOp>(srcBlock.back());
  return branchOp && branchOp.getSuccessor() == &dstBlock;
}

static LogicalResult verify(spirv::LoopOp loopOp) {
  auto *op = loopOp.getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +-------------+
  //                     | entry block |
  //                     +-------------+
  //                            |
  //                            v
  //                     +-------------+
  //                     | loop header | <-----+
  //                     +-------------+       |
  //                                           |
  //                           ...             |
  //                          \ | /            |
  //                            v              |
  //                    +---------------+      |
  //                    | loop continue | -----+
  //                    +---------------+
  //
  //                           ...
  //                          \ | /
  //                            v
  //                     +-------------+
  //                     | merge block |
  //                     +-------------+

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerated case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block is the merge block.
  Block &merge = region.back();
  if (!isMergeBlock(merge))
    return loopOp.emitOpError(
        "last block must be the merge block with only one 'spv.mlir.merge' op");

  if (std::next(region.begin()) == region.end())
    return loopOp.emitOpError(
        "must have an entry block branching to the loop header block");
  // The first block is the entry block.
  Block &entry = region.front();

  if (std::next(region.begin(), 2) == region.end())
    return loopOp.emitOpError(
        "must have a loop header block branched from the entry block");
  // The second block is the loop header block.
  Block &header = *std::next(region.begin(), 1);

  if (!hasOneBranchOpTo(entry, header))
    return loopOp.emitOpError(
        "entry block must only have one 'spv.Branch' op to the second block");

  if (std::next(region.begin(), 3) == region.end())
    return loopOp.emitOpError(
        "requires a loop continue block branching to the loop header block");
  // The second to last block is the loop continue block.
  Block &cont = *std::prev(region.end(), 2);

  // Make sure that we have a branch from the loop continue block to the loop
  // header block.
  if (llvm::none_of(
          llvm::seq<unsigned>(0, cont.getNumSuccessors()),
          [&](unsigned index) { return cont.getSuccessor(index) == &header; }))
    return loopOp.emitOpError("second to last block must be the loop continue "
                              "block that branches to the loop header block");

  // Make sure that no other blocks (except the entry and loop continue block)
  // branches to the loop header block.
  for (auto &block : llvm::make_range(std::next(region.begin(), 2),
                                      std::prev(region.end(), 2))) {
    for (auto i : llvm::seq<unsigned>(0, block.getNumSuccessors())) {
      if (block.getSuccessor(i) == &header) {
        return loopOp.emitOpError("can only have the entry and loop continue "
                                  "block branching to the loop header block");
      }
    }
  }

  return success();
}

Block *spirv::LoopOp::getEntryBlock() {
  assert(!body().empty() && "op region should not be empty!");
  return &body().front();
}

Block *spirv::LoopOp::getHeaderBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The second block is the loop header block.
  return &*std::next(body().begin());
}

Block *spirv::LoopOp::getContinueBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The second to last block is the loop continue block.
  return &*std::prev(body().end(), 2);
}

Block *spirv::LoopOp::getMergeBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The last block is the loop merge block.
  return &body().back();
}

void spirv::LoopOp::addEntryAndMergeBlock() {
  assert(body().empty() && "entry and merge block already exist");
  body().push_back(new Block());
  auto *mergeBlock = new Block();
  body().push_back(mergeBlock);
  OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

  // Add a spv.mlir.merge op into the merge block.
  builder.create<spirv::MergeOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// spv.mlir.merge
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::MergeOp mergeOp) {
  auto *parentOp = mergeOp->getParentOp();
  if (!parentOp || !isa<spirv::SelectionOp, spirv::LoopOp>(parentOp))
    return mergeOp.emitOpError(
        "expected parent op to be 'spv.mlir.selection' or 'spv.mlir.loop'");

  Block &parentLastBlock = mergeOp->getParentRegion()->back();
  if (mergeOp.getOperation() != parentLastBlock.getTerminator())
    return mergeOp.emitOpError("can only be used in the last block of "
                               "'spv.mlir.selection' or 'spv.mlir.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

void spirv::ModuleOp::build(OpBuilder &builder, OperationState &state,
                            Optional<StringRef> name) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  if (name) {
    state.attributes.append(mlir::SymbolTable::getSymbolAttrName(),
                            builder.getStringAttr(*name));
  }
}

void spirv::ModuleOp::build(OpBuilder &builder, OperationState &state,
                            spirv::AddressingModel addressingModel,
                            spirv::MemoryModel memoryModel,
                            Optional<StringRef> name) {
  state.addAttribute(
      "addressing_model",
      builder.getI32IntegerAttr(static_cast<int32_t>(addressingModel)));
  state.addAttribute("memory_model", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(memoryModel)));
  ensureTerminator(*state.addRegion(), builder, state.location);
  if (name) {
    state.attributes.append(mlir::SymbolTable::getSymbolAttrName(),
                            builder.getStringAttr(*name));
  }
}

static ParseResult parseModuleOp(OpAsmParser &parser, OperationState &state) {
  Region *body = state.addRegion();

  // If the name is present, parse it.
  StringAttr nameAttr;
  parser.parseOptionalSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                                 state.attributes);

  // Parse attributes
  spirv::AddressingModel addrModel;
  spirv::MemoryModel memoryModel;
  if (parseEnumKeywordAttr(addrModel, parser, state) ||
      parseEnumKeywordAttr(memoryModel, parser, state))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("requires"))) {
    spirv::VerCapExtAttr vceTriple;
    if (parser.parseAttribute(vceTriple,
                              spirv::ModuleOp::getVCETripleAttrName(),
                              state.attributes))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  spirv::ModuleOp::ensureTerminator(*body, parser.getBuilder(), state.location);
  return success();
}

static void print(spirv::ModuleOp moduleOp, OpAsmPrinter &printer) {
  printer << spirv::ModuleOp::getOperationName();

  if (Optional<StringRef> name = moduleOp.getName()) {
    printer << ' ';
    printer.printSymbolName(*name);
  }

  SmallVector<StringRef, 2> elidedAttrs;

  printer << " " << spirv::stringifyAddressingModel(moduleOp.addressing_model())
          << " " << spirv::stringifyMemoryModel(moduleOp.memory_model());
  auto addressingModelAttrName = spirv::attributeName<spirv::AddressingModel>();
  auto memoryModelAttrName = spirv::attributeName<spirv::MemoryModel>();
  elidedAttrs.assign({addressingModelAttrName, memoryModelAttrName,
                      SymbolTable::getSymbolAttrName()});

  if (Optional<spirv::VerCapExtAttr> triple = moduleOp.vce_triple()) {
    printer << " requires " << *triple;
    elidedAttrs.push_back(spirv::ModuleOp::getVCETripleAttrName());
  }

  printer.printOptionalAttrDictWithKeyword(moduleOp->getAttrs(), elidedAttrs);
  printer.printRegion(moduleOp.body(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

static LogicalResult verify(spirv::ModuleOp moduleOp) {
  auto &op = *moduleOp.getOperation();
  auto *dialect = op.getDialect();
  DenseMap<std::pair<spirv::FuncOp, spirv::ExecutionModel>, spirv::EntryPointOp>
      entryPoints;
  SymbolTable table(moduleOp);

  for (auto &op : moduleOp.getBlock()) {
    if (op.getDialect() != dialect)
      return op.emitError("'spv.module' can only contain spv.* ops");

    // For EntryPoint op, check that the function and execution model is not
    // duplicated in EntryPointOps. Also verify that the interface specified
    // comes from globalVariables here to make this check cheaper.
    if (auto entryPointOp = dyn_cast<spirv::EntryPointOp>(op)) {
      auto funcOp = table.lookup<spirv::FuncOp>(entryPointOp.fn());
      if (!funcOp) {
        return entryPointOp.emitError("function '")
               << entryPointOp.fn() << "' not found in 'spv.module'";
      }
      if (auto interface = entryPointOp.interface()) {
        for (Attribute varRef : interface) {
          auto varSymRef = varRef.dyn_cast<FlatSymbolRefAttr>();
          if (!varSymRef) {
            return entryPointOp.emitError(
                       "expected symbol reference for interface "
                       "specification instead of '")
                   << varRef;
          }
          auto variableOp =
              table.lookup<spirv::GlobalVariableOp>(varSymRef.getValue());
          if (!variableOp) {
            return entryPointOp.emitError("expected spv.GlobalVariable "
                                          "symbol reference instead of'")
                   << varSymRef << "'";
          }
        }
      }

      auto key = std::pair<spirv::FuncOp, spirv::ExecutionModel>(
          funcOp, entryPointOp.execution_model());
      auto entryPtIt = entryPoints.find(key);
      if (entryPtIt != entryPoints.end()) {
        return entryPointOp.emitError("duplicate of a previous EntryPointOp");
      }
      entryPoints[key] = entryPointOp;
    } else if (auto funcOp = dyn_cast<spirv::FuncOp>(op)) {
      if (funcOp.isExternal())
        return op.emitError("'spv.module' cannot contain external functions");

      // TODO: move this check to spv.func.
      for (auto &block : funcOp)
        for (auto &op : block) {
          if (op.getDialect() != dialect)
            return op.emitError(
                "functions in 'spv.module' can only contain spv.* ops");
        }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.mlir.referenceof
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::ReferenceOfOp referenceOfOp) {
  auto *specConstSym = SymbolTable::lookupNearestSymbolFrom(
      referenceOfOp->getParentOp(), referenceOfOp.spec_const());
  Type constType;

  auto specConstOp = dyn_cast_or_null<spirv::SpecConstantOp>(specConstSym);
  if (specConstOp)
    constType = specConstOp.default_value().getType();

  auto specConstCompositeOp =
      dyn_cast_or_null<spirv::SpecConstantCompositeOp>(specConstSym);
  if (specConstCompositeOp)
    constType = specConstCompositeOp.type();

  if (!specConstOp && !specConstCompositeOp)
    return referenceOfOp.emitOpError(
        "expected spv.SpecConstant or spv.SpecConstantComposite symbol");

  if (referenceOfOp.reference().getType() != constType)
    return referenceOfOp.emitOpError("result type mismatch with the referenced "
                                     "specialization constant's type");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::ReturnOp returnOp) {
  // Verification is performed in spv.func op.
  return success();
}

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::ReturnValueOp retValOp) {
  // Verification is performed in spv.func op.
  return success();
}

//===----------------------------------------------------------------------===//
// spv.Select
//===----------------------------------------------------------------------===//

void spirv::SelectOp::build(OpBuilder &builder, OperationState &state,
                            Value cond, Value trueValue, Value falseValue) {
  build(builder, state, trueValue.getType(), cond, trueValue, falseValue);
}

static LogicalResult verify(spirv::SelectOp op) {
  if (auto conditionTy = op.condition().getType().dyn_cast<VectorType>()) {
    auto resultVectorTy = op.result().getType().dyn_cast<VectorType>();
    if (!resultVectorTy) {
      return op.emitOpError("result expected to be of vector type when "
                            "condition is of vector type");
    }
    if (resultVectorTy.getNumElements() != conditionTy.getNumElements()) {
      return op.emitOpError("result should have the same number of elements as "
                            "the condition when condition is of vector type");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.mlir.selection
//===----------------------------------------------------------------------===//

static ParseResult parseSelectionOp(OpAsmParser &parser,
                                    OperationState &state) {
  if (parseControlAttribute<spirv::SelectionControl>(parser, state))
    return failure();
  return parser.parseRegion(*state.addRegion(), /*arguments=*/{},
                            /*argTypes=*/{});
}

static void print(spirv::SelectionOp selectionOp, OpAsmPrinter &printer) {
  auto *op = selectionOp.getOperation();

  printer << spirv::SelectionOp::getOperationName();
  auto control = selectionOp.selection_control();
  if (control != spirv::SelectionControl::None)
    printer << " control(" << spirv::stringifySelectionControl(control) << ")";
  printer.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

static LogicalResult verify(spirv::SelectionOp selectionOp) {
  auto *op = selectionOp.getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +--------------+
  //                     | header block |
  //                     +--------------+
  //                          / | \
  //                           ...
  //
  //
  //         +---------+   +---------+   +---------+
  //         | case #0 |   | case #1 |   | case #2 |  ...
  //         +---------+   +---------+   +---------+
  //
  //
  //                           ...
  //                          \ | /
  //                            v
  //                     +-------------+
  //                     | merge block |
  //                     +-------------+

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerated case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block is the merge block.
  if (!isMergeBlock(region.back()))
    return selectionOp.emitOpError(
        "last block must be the merge block with only one 'spv.mlir.merge' op");

  if (std::next(region.begin()) == region.end())
    return selectionOp.emitOpError("must have a selection header block");

  return success();
}

Block *spirv::SelectionOp::getHeaderBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The first block is the loop header block.
  return &body().front();
}

Block *spirv::SelectionOp::getMergeBlock() {
  assert(!body().empty() && "op region should not be empty!");
  // The last block is the loop merge block.
  return &body().back();
}

void spirv::SelectionOp::addMergeBlock() {
  assert(body().empty() && "entry and merge block already exist");
  auto *mergeBlock = new Block();
  body().push_back(mergeBlock);
  OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

  // Add a spv.mlir.merge op into the merge block.
  builder.create<spirv::MergeOp>(getLoc());
}

spirv::SelectionOp spirv::SelectionOp::createIfThen(
    Location loc, Value condition,
    function_ref<void(OpBuilder &builder)> thenBody, OpBuilder &builder) {
  auto selectionOp =
      builder.create<spirv::SelectionOp>(loc, spirv::SelectionControl::None);

  selectionOp.addMergeBlock();
  Block *mergeBlock = selectionOp.getMergeBlock();
  Block *thenBlock = nullptr;

  // Build the "then" block.
  {
    OpBuilder::InsertionGuard guard(builder);
    thenBlock = builder.createBlock(mergeBlock);
    thenBody(builder);
    builder.create<spirv::BranchOp>(loc, mergeBlock);
  }

  // Build the header block.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(thenBlock);
    builder.create<spirv::BranchConditionalOp>(
        loc, condition, thenBlock,
        /*trueArguments=*/ArrayRef<Value>(), mergeBlock,
        /*falseArguments=*/ArrayRef<Value>());
  }

  return selectionOp;
}

//===----------------------------------------------------------------------===//
// spv.SpecConstant
//===----------------------------------------------------------------------===//

static ParseResult parseSpecConstantOp(OpAsmParser &parser,
                                       OperationState &state) {
  StringAttr nameAttr;
  Attribute valueAttr;

  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse optional spec_id.
  if (succeeded(parser.parseOptionalKeyword(kSpecIdAttrName))) {
    IntegerAttr specIdAttr;
    if (parser.parseLParen() ||
        parser.parseAttribute(specIdAttr, kSpecIdAttrName, state.attributes) ||
        parser.parseRParen())
      return failure();
  }

  if (parser.parseEqual() ||
      parser.parseAttribute(valueAttr, kDefaultValueAttrName, state.attributes))
    return failure();

  return success();
}

static void print(spirv::SpecConstantOp constOp, OpAsmPrinter &printer) {
  printer << spirv::SpecConstantOp::getOperationName() << ' ';
  printer.printSymbolName(constOp.sym_name());
  if (auto specID = constOp->getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    printer << ' ' << kSpecIdAttrName << '(' << specID.getInt() << ')';
  printer << " = " << constOp.default_value();
}

static LogicalResult verify(spirv::SpecConstantOp constOp) {
  if (auto specID = constOp->getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    if (specID.getValue().isNegative())
      return constOp.emitOpError("SpecId cannot be negative");

  auto value = constOp.default_value();
  if (value.isa<IntegerAttr, FloatAttr>()) {
    // Make sure bitwidth is allowed.
    if (!value.getType().isa<spirv::SPIRVType>())
      return constOp.emitOpError("default value bitwidth disallowed");
    return success();
  }
  return constOp.emitOpError(
      "default value can only be a bool, integer, or float scalar");
}

//===----------------------------------------------------------------------===//
// spv.StoreOp
//===----------------------------------------------------------------------===//

static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &state) {
  // Parse the storage class specification
  spirv::StorageClass storageClass;
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  auto loc = parser.getCurrentLocation();
  Type elementType;
  if (parseEnumStrAttr(storageClass, parser) ||
      parser.parseOperandList(operandInfo, 2) ||
      parseMemoryAccessAttributes(parser, state) || parser.parseColon() ||
      parser.parseType(elementType)) {
    return failure();
  }

  auto ptrType = spirv::PointerType::get(elementType, storageClass);
  if (parser.resolveOperands(operandInfo, {ptrType, elementType}, loc,
                             state.operands)) {
    return failure();
  }
  return success();
}

static void print(spirv::StoreOp storeOp, OpAsmPrinter &printer) {
  auto *op = storeOp.getOperation();
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      storeOp.ptr().getType().cast<spirv::PointerType>().getStorageClass());
  printer << spirv::StoreOp::getOperationName() << " \"" << sc << "\" "
          << storeOp.ptr() << ", " << storeOp.value();

  printMemoryAccessAttribute(storeOp, printer, elidedAttrs);

  printer << " : " << storeOp.value().getType();
  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

static LogicalResult verify(spirv::StoreOp storeOp) {
  // SPIR-V spec : "Pointer is the pointer to store through. Its type must be an
  // OpTypePointer whose Type operand is the same as the type of Object."
  if (failed(verifyLoadStorePtrAndValTypes(storeOp, storeOp.ptr(),
                                           storeOp.value()))) {
    return failure();
  }
  return verifyMemoryAccessAttribute(storeOp);
}

//===----------------------------------------------------------------------===//
// spv.Unreachable
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::UnreachableOp unreachableOp) {
  auto *op = unreachableOp.getOperation();
  auto *block = op->getBlock();
  // Fast track: if this is in entry block, its invalid. Otherwise, if no
  // predecessors, it's valid.
  if (block->isEntryBlock())
    return unreachableOp.emitOpError("cannot be used in reachable block");
  if (block->hasNoPredecessors())
    return success();

  // TODO: further verification needs to analyze reachability from
  // the entry block.

  return success();
}

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

static ParseResult parseVariableOp(OpAsmParser &parser, OperationState &state) {
  // Parse optional initializer
  Optional<OpAsmParser::OperandType> initInfo;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    initInfo = OpAsmParser::OperandType();
    if (parser.parseLParen() || parser.parseOperand(*initInfo) ||
        parser.parseRParen())
      return failure();
  }

  if (parseVariableDecorations(parser, state)) {
    return failure();
  }

  // Parse result pointer type
  Type type;
  if (parser.parseColon())
    return failure();
  auto loc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser.emitError(loc, "expected spv.ptr type");
  state.addTypes(ptrType);

  // Resolve the initializer operand
  if (initInfo) {
    if (parser.resolveOperand(*initInfo, ptrType.getPointeeType(),
                              state.operands))
      return failure();
  }

  auto attr = parser.getBuilder().getI32IntegerAttr(
      llvm::bit_cast<int32_t>(ptrType.getStorageClass()));
  state.addAttribute(spirv::attributeName<spirv::StorageClass>(), attr);

  return success();
}

static void print(spirv::VariableOp varOp, OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  printer << spirv::VariableOp::getOperationName();

  // Print optional initializer
  if (varOp.getNumOperands() != 0)
    printer << " init(" << varOp.initializer() << ")";

  printVariableDecorations(varOp, printer, elidedAttrs);
  printer << " : " << varOp.getType();
}

static LogicalResult verify(spirv::VariableOp varOp) {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (varOp.storage_class() != spirv::StorageClass::Function) {
    return varOp.emitOpError(
        "can only be used to model function-level variables. Use "
        "spv.GlobalVariable for module-level variables.");
  }

  auto pointerType = varOp.pointer().getType().cast<spirv::PointerType>();
  if (varOp.storage_class() != pointerType.getStorageClass())
    return varOp.emitOpError(
        "storage class must match result pointer's storage class");

  if (varOp.getNumOperands() != 0) {
    // SPIR-V spec: "Initializer must be an <id> from a constant instruction or
    // a global (module scope) OpVariable instruction".
    auto *initOp = varOp.getOperand(0).getDefiningOp();
    if (!initOp || !isa<spirv::ConstantOp,    // for normal constant
                        spirv::ReferenceOfOp, // for spec constant
                        spirv::AddressOfOp>(initOp))
      return varOp.emitOpError("initializer must be the result of a "
                               "constant or spv.GlobalVariable op");
  }

  // TODO: generate these strings using ODS.
  auto *op = varOp.getOperation();
  auto descriptorSetName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::Binding));
  auto builtInName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::BuiltIn));

  for (const auto &attr : {descriptorSetName, bindingName, builtInName}) {
    if (op->getAttr(attr))
      return varOp.emitOpError("cannot have '")
             << attr << "' attribute (only allowed in spv.GlobalVariable)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.VectorShuffle
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::VectorShuffleOp shuffleOp) {
  VectorType resultType = shuffleOp.getType().cast<VectorType>();

  size_t numResultElements = resultType.getNumElements();
  if (numResultElements != shuffleOp.components().size())
    return shuffleOp.emitOpError("result type element count (")
           << numResultElements
           << ") mismatch with the number of component selectors ("
           << shuffleOp.components().size() << ")";

  size_t totalSrcElements =
      shuffleOp.vector1().getType().cast<VectorType>().getNumElements() +
      shuffleOp.vector2().getType().cast<VectorType>().getNumElements();

  for (const auto &selector :
       shuffleOp.components().getAsValueRange<IntegerAttr>()) {
    uint32_t index = selector.getZExtValue();
    if (index >= totalSrcElements &&
        index != std::numeric_limits<uint32_t>().max())
      return shuffleOp.emitOpError("component selector ")
             << index << " out of range: expected to be in [0, "
             << totalSrcElements << ") or 0xffffffff";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.CooperativeMatrixLoadNV
//===----------------------------------------------------------------------===//

static ParseResult parseCooperativeMatrixLoadNVOp(OpAsmParser &parser,
                                                  OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 3> operandInfo;
  Type strideType = parser.getBuilder().getIntegerType(32);
  Type columnMajorType = parser.getBuilder().getIntegerType(1);
  Type ptrType;
  Type elementType;
  if (parser.parseOperandList(operandInfo, 3) ||
      parseMemoryAccessAttributes(parser, state) || parser.parseColon() ||
      parser.parseType(ptrType) || parser.parseKeywordType("as", elementType)) {
    return failure();
  }
  if (parser.resolveOperands(operandInfo,
                             {ptrType, strideType, columnMajorType},
                             parser.getNameLoc(), state.operands)) {
    return failure();
  }

  state.addTypes(elementType);
  return success();
}

static void print(spirv::CooperativeMatrixLoadNVOp M, OpAsmPrinter &printer) {
  printer << spirv::CooperativeMatrixLoadNVOp::getOperationName() << " "
          << M.pointer() << ", " << M.stride() << ", " << M.columnmajor();
  // Print optional memory access attribute.
  if (auto memAccess = M.memory_access())
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"]";
  printer << " : " << M.pointer().getType() << " as " << M.getType();
}

static LogicalResult verifyPointerAndCoopMatrixType(Operation *op, Type pointer,
                                                    Type coopMatrix) {
  Type pointeeType = pointer.cast<spirv::PointerType>().getPointeeType();
  if (!pointeeType.isa<spirv::ScalarType>() && !pointeeType.isa<VectorType>())
    return op->emitError(
               "Pointer must point to a scalar or vector type but provided ")
           << pointeeType;
  spirv::StorageClass storage =
      pointer.cast<spirv::PointerType>().getStorageClass();
  if (storage != spirv::StorageClass::Workgroup &&
      storage != spirv::StorageClass::StorageBuffer &&
      storage != spirv::StorageClass::PhysicalStorageBuffer)
    return op->emitError(
               "Pointer storage class must be Workgroup, StorageBuffer or "
               "PhysicalStorageBufferEXT but provided ")
           << stringifyStorageClass(storage);
  return success();
}

//===----------------------------------------------------------------------===//
// spv.CooperativeMatrixStoreNV
//===----------------------------------------------------------------------===//

static ParseResult parseCooperativeMatrixStoreNVOp(OpAsmParser &parser,
                                                   OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 4> operandInfo;
  Type strideType = parser.getBuilder().getIntegerType(32);
  Type columnMajorType = parser.getBuilder().getIntegerType(1);
  Type ptrType;
  Type elementType;
  if (parser.parseOperandList(operandInfo, 4) ||
      parseMemoryAccessAttributes(parser, state) || parser.parseColon() ||
      parser.parseType(ptrType) || parser.parseComma() ||
      parser.parseType(elementType)) {
    return failure();
  }
  if (parser.resolveOperands(
          operandInfo, {ptrType, elementType, strideType, columnMajorType},
          parser.getNameLoc(), state.operands)) {
    return failure();
  }

  return success();
}

static void print(spirv::CooperativeMatrixStoreNVOp coopMatrix,
                  OpAsmPrinter &printer) {
  printer << spirv::CooperativeMatrixStoreNVOp::getOperationName() << " "
          << coopMatrix.pointer() << ", " << coopMatrix.object() << ", "
          << coopMatrix.stride() << ", " << coopMatrix.columnmajor();
  // Print optional memory access attribute.
  if (auto memAccess = coopMatrix.memory_access())
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"]";
  printer << " : " << coopMatrix.pointer().getType() << ", "
          << coopMatrix.getOperand(1).getType();
}

//===----------------------------------------------------------------------===//
// spv.CooperativeMatrixMulAddNV
//===----------------------------------------------------------------------===//

static LogicalResult
verifyCoopMatrixMulAdd(spirv::CooperativeMatrixMulAddNVOp op) {
  if (op.c().getType() != op.result().getType())
    return op.emitOpError("result and third operand must have the same type");
  auto typeA = op.a().getType().cast<spirv::CooperativeMatrixNVType>();
  auto typeB = op.b().getType().cast<spirv::CooperativeMatrixNVType>();
  auto typeC = op.c().getType().cast<spirv::CooperativeMatrixNVType>();
  auto typeR = op.result().getType().cast<spirv::CooperativeMatrixNVType>();
  if (typeA.getRows() != typeR.getRows() ||
      typeA.getColumns() != typeB.getRows() ||
      typeB.getColumns() != typeR.getColumns())
    return op.emitOpError("matrix size must match");
  if (typeR.getScope() != typeA.getScope() ||
      typeR.getScope() != typeB.getScope() ||
      typeR.getScope() != typeC.getScope())
    return op.emitOpError("matrix scope must match");
  if (typeA.getElementType() != typeB.getElementType() ||
      typeR.getElementType() != typeC.getElementType())
    return op.emitOpError("matrix element type must match");
  return success();
}

//===----------------------------------------------------------------------===//
// spv.MatrixTimesScalar
//===----------------------------------------------------------------------===//

static LogicalResult verifyMatrixTimesScalar(spirv::MatrixTimesScalarOp op) {
  // We already checked that result and matrix are both of matrix type in the
  // auto-generated verify method.

  auto inputMatrix = op.matrix().getType().cast<spirv::MatrixType>();
  auto resultMatrix = op.result().getType().cast<spirv::MatrixType>();

  // Check that the scalar type is the same as the matrix element type.
  if (op.scalar().getType() != inputMatrix.getElementType())
    return op.emitError("input matrix components' type and scaling value must "
                        "have the same type");

  // Note that the next three checks could be done using the AllTypesMatch
  // trait in the Op definition file but it generates a vague error message.

  // Check that the input and result matrices have the same columns' count
  if (inputMatrix.getNumColumns() != resultMatrix.getNumColumns())
    return op.emitError("input and result matrices must have the same "
                        "number of columns");

  // Check that the input and result matrices' have the same rows count
  if (inputMatrix.getNumRows() != resultMatrix.getNumRows())
    return op.emitError("input and result matrices' columns must have "
                        "the same size");

  // Check that the input and result matrices' have the same component type
  if (inputMatrix.getElementType() != resultMatrix.getElementType())
    return op.emitError("input and result matrices' columns must have "
                        "the same component type");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CopyMemory
//===----------------------------------------------------------------------===//

static void print(spirv::CopyMemoryOp copyMemory, OpAsmPrinter &printer) {
  auto *op = copyMemory.getOperation();
  printer << spirv::CopyMemoryOp::getOperationName() << ' ';

  StringRef targetStorageClass =
      stringifyStorageClass(copyMemory.target()
                                .getType()
                                .cast<spirv::PointerType>()
                                .getStorageClass());
  printer << " \"" << targetStorageClass << "\" " << copyMemory.target()
          << ", ";

  StringRef sourceStorageClass =
      stringifyStorageClass(copyMemory.source()
                                .getType()
                                .cast<spirv::PointerType>()
                                .getStorageClass());
  printer << " \"" << sourceStorageClass << "\" " << copyMemory.source();

  SmallVector<StringRef, 4> elidedAttrs;
  printMemoryAccessAttribute(copyMemory, printer, elidedAttrs);
  printSourceMemoryAccessAttribute(copyMemory, printer, elidedAttrs,
                                   copyMemory.source_memory_access(),
                                   copyMemory.source_alignment());

  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  Type pointeeType =
      copyMemory.target().getType().cast<spirv::PointerType>().getPointeeType();
  printer << " : " << pointeeType;
}

static ParseResult parseCopyMemoryOp(OpAsmParser &parser,
                                     OperationState &state) {
  spirv::StorageClass targetStorageClass;
  OpAsmParser::OperandType targetPtrInfo;

  spirv::StorageClass sourceStorageClass;
  OpAsmParser::OperandType sourcePtrInfo;

  Type elementType;

  if (parseEnumStrAttr(targetStorageClass, parser) ||
      parser.parseOperand(targetPtrInfo) || parser.parseComma() ||
      parseEnumStrAttr(sourceStorageClass, parser) ||
      parser.parseOperand(sourcePtrInfo) ||
      parseMemoryAccessAttributes(parser, state)) {
    return failure();
  }

  if (!parser.parseOptionalComma()) {
    // Parse 2nd memory access attributes.
    if (parseSourceMemoryAccessAttributes(parser, state)) {
      return failure();
    }
  }

  if (parser.parseColon() || parser.parseType(elementType))
    return failure();

  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  auto targetPtrType = spirv::PointerType::get(elementType, targetStorageClass);
  auto sourcePtrType = spirv::PointerType::get(elementType, sourceStorageClass);

  if (parser.resolveOperand(targetPtrInfo, targetPtrType, state.operands) ||
      parser.resolveOperand(sourcePtrInfo, sourcePtrType, state.operands)) {
    return failure();
  }

  return success();
}

static LogicalResult verifyCopyMemory(spirv::CopyMemoryOp copyMemory) {
  Type targetType =
      copyMemory.target().getType().cast<spirv::PointerType>().getPointeeType();

  Type sourceType =
      copyMemory.source().getType().cast<spirv::PointerType>().getPointeeType();

  if (targetType != sourceType) {
    return copyMemory.emitOpError(
        "both operands must be pointers to the same type");
  }

  if (failed(verifyMemoryAccessAttribute(copyMemory))) {
    return failure();
  }

  // TODO - According to the spec:
  //
  // If two masks are present, the first applies to Target and cannot include
  // MakePointerVisible, and the second applies to Source and cannot include
  // MakePointerAvailable.
  //
  // Add such verification here.

  return verifySourceMemoryAccessAttribute(copyMemory);
}

//===----------------------------------------------------------------------===//
// spv.Transpose
//===----------------------------------------------------------------------===//

static LogicalResult verifyTranspose(spirv::TransposeOp op) {
  auto inputMatrix = op.matrix().getType().cast<spirv::MatrixType>();
  auto resultMatrix = op.result().getType().cast<spirv::MatrixType>();

  // Verify that the input and output matrices have correct shapes.
  if (inputMatrix.getNumRows() != resultMatrix.getNumColumns())
    return op.emitError("input matrix rows count must be equal to "
                        "output matrix columns count");

  if (inputMatrix.getNumColumns() != resultMatrix.getNumRows())
    return op.emitError("input matrix columns count must be equal to "
                        "output matrix rows count");

  // Verify that the input and output matrices have the same component type
  if (inputMatrix.getElementType() != resultMatrix.getElementType())
    return op.emitError("input and output matrices must have the same "
                        "component type");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.MatrixTimesMatrix
//===----------------------------------------------------------------------===//

static LogicalResult verifyMatrixTimesMatrix(spirv::MatrixTimesMatrixOp op) {
  auto leftMatrix = op.leftmatrix().getType().cast<spirv::MatrixType>();
  auto rightMatrix = op.rightmatrix().getType().cast<spirv::MatrixType>();
  auto resultMatrix = op.result().getType().cast<spirv::MatrixType>();

  // left matrix columns' count and right matrix rows' count must be equal
  if (leftMatrix.getNumColumns() != rightMatrix.getNumRows())
    return op.emitError("left matrix columns' count must be equal to "
                        "the right matrix rows' count");

  // right and result matrices columns' count must be the same
  if (rightMatrix.getNumColumns() != resultMatrix.getNumColumns())
    return op.emitError(
        "right and result matrices must have equal columns' count");

  // right and result matrices component type must be the same
  if (rightMatrix.getElementType() != resultMatrix.getElementType())
    return op.emitError("right and result matrices' component type must"
                        " be the same");

  // left and result matrices component type must be the same
  if (leftMatrix.getElementType() != resultMatrix.getElementType())
    return op.emitError("left and result matrices' component type"
                        " must be the same");

  // left and result matrices rows count must be the same
  if (leftMatrix.getNumRows() != resultMatrix.getNumRows())
    return op.emitError("left and result matrices must have equal rows'"
                        " count");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite
//===----------------------------------------------------------------------===//

static ParseResult parseSpecConstantCompositeOp(OpAsmParser &parser,
                                                OperationState &state) {

  StringAttr compositeName;
  if (parser.parseSymbolName(compositeName, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  if (parser.parseLParen())
    return failure();

  SmallVector<Attribute, 4> constituents;

  do {
    // The name of the constituent attribute isn't important
    const char *attrName = "spec_const";
    FlatSymbolRefAttr specConstRef;
    NamedAttrList attrs;

    if (parser.parseAttribute(specConstRef, Type(), attrName, attrs))
      return failure();

    constituents.push_back(specConstRef);
  } while (!parser.parseOptionalComma());

  if (parser.parseRParen())
    return failure();

  state.addAttribute(kCompositeSpecConstituentsName,
                     parser.getBuilder().getArrayAttr(constituents));

  Type type;
  if (parser.parseColonType(type))
    return failure();

  state.addAttribute(kTypeAttrName, TypeAttr::get(type));

  return success();
}

static void print(spirv::SpecConstantCompositeOp op, OpAsmPrinter &printer) {
  printer << spirv::SpecConstantCompositeOp::getOperationName() << " ";
  printer.printSymbolName(op.sym_name());
  printer << " (";
  auto constituents = op.constituents().getValue();

  if (!constituents.empty())
    llvm::interleaveComma(constituents, printer);

  printer << ") : " << op.type();
}

static LogicalResult verify(spirv::SpecConstantCompositeOp constOp) {
  auto cType = constOp.type().dyn_cast<spirv::CompositeType>();
  auto constituents = constOp.constituents().getValue();

  if (!cType)
    return constOp.emitError(
               "result type must be a composite type, but provided ")
           << constOp.type();

  if (cType.isa<spirv::CooperativeMatrixNVType>())
    return constOp.emitError("unsupported composite type  ") << cType;
  else if (constituents.size() != cType.getNumElements())
    return constOp.emitError("has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided "
           << constituents.size();

  for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
    auto constituent = constituents[index].dyn_cast<FlatSymbolRefAttr>();

    auto constituentSpecConstOp =
        dyn_cast<spirv::SpecConstantOp>(SymbolTable::lookupNearestSymbolFrom(
            constOp->getParentOp(), constituent.getValue()));

    if (constituentSpecConstOp.default_value().getType() !=
        cType.getElementType(index))
      return constOp.emitError("has incorrect types of operands: expected ")
             << cType.getElementType(index) << ", but provided "
             << constituentSpecConstOp.default_value().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantOperation
//===----------------------------------------------------------------------===//

static ParseResult parseSpecConstantOperationOp(OpAsmParser &parser,
                                                OperationState &state) {
  Region *body = state.addRegion();

  if (parser.parseKeyword("wraps"))
    return failure();

  body->push_back(new Block);
  Block &block = body->back();
  Operation *wrappedOp = parser.parseGenericOperation(&block, block.begin());

  if (!wrappedOp)
    return failure();

  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<spirv::YieldOp>(wrappedOp->getLoc(), wrappedOp->getResult(0));
  state.location = wrappedOp->getLoc();

  state.addTypes(wrappedOp->getResult(0).getType());

  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

static void print(spirv::SpecConstantOperationOp op, OpAsmPrinter &printer) {
  printer << op.getOperationName() << " wraps ";
  printer.printGenericOp(&op.body().front().front());
}

static LogicalResult verify(spirv::SpecConstantOperationOp constOp) {
  Block &block = constOp.getRegion().getBlocks().front();

  if (block.getOperations().size() != 2)
    return constOp.emitOpError("expected exactly 2 nested ops");

  Operation &enclosedOp = block.getOperations().front();

  if (!enclosedOp.hasTrait<OpTrait::spirv::UsableInSpecConstantOp>())
    return constOp.emitOpError("invalid enclosed op");

  for (auto operand : enclosedOp.getOperands())
    if (!isa<spirv::ConstantOp, spirv::ReferenceOfOp,
             spirv::SpecConstantOperationOp>(operand.getDefiningOp()))
      return constOp.emitOpError(
          "invalid operand, must be defined by a constant operation");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FrexpStruct
//===----------------------------------------------------------------------===//
static LogicalResult
verifyGLSLFrexpStructOp(spirv::GLSLFrexpStructOp frexpStructOp) {
  spirv::StructType structTy =
      frexpStructOp.result().getType().dyn_cast<spirv::StructType>();

  if (structTy.getNumElements() != 2)
    return frexpStructOp.emitError("result type must be a struct  type "
                                   "with two memebers");

  Type significandTy = structTy.getElementType(0);
  Type exponentTy = structTy.getElementType(1);
  VectorType exponentVecTy = exponentTy.dyn_cast<VectorType>();
  IntegerType exponentIntTy = exponentTy.dyn_cast<IntegerType>();

  Type operandTy = frexpStructOp.operand().getType();
  VectorType operandVecTy = operandTy.dyn_cast<VectorType>();
  FloatType operandFTy = operandTy.dyn_cast<FloatType>();

  if (significandTy != operandTy)
    return frexpStructOp.emitError("member zero of the resulting struct type "
                                   "must be the same type as the operand");

  if (exponentVecTy) {
    IntegerType componentIntTy =
        exponentVecTy.getElementType().dyn_cast<IntegerType>();
    if (!(componentIntTy && componentIntTy.getWidth() == 32))
      return frexpStructOp.emitError(
          "member one of the resulting struct type must"
          "be a scalar or vector of 32 bit integer type");
  } else if (!(exponentIntTy && exponentIntTy.getWidth() == 32)) {
    return frexpStructOp.emitError(
        "member one of the resulting struct type "
        "must be a scalar or vector of 32 bit integer type");
  }

  // Check that the two member types have the same number of components
  if (operandVecTy && exponentVecTy &&
      (exponentVecTy.getNumElements() == operandVecTy.getNumElements()))
    return success();

  if (operandFTy && exponentIntTy)
    return success();

  return frexpStructOp.emitError(
      "member one of the resulting struct type "
      "must have the same number of components as the operand type");
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Ldexp
//===----------------------------------------------------------------------===//

static LogicalResult verify(spirv::GLSLLdexpOp ldexpOp) {
  Type significandType = ldexpOp.x().getType();
  Type exponentType = ldexpOp.exp().getType();

  if (significandType.isa<FloatType>() != exponentType.isa<IntegerType>())
    return ldexpOp.emitOpError("operands must both be scalars or vectors");

  auto getNumElements = [](Type type) -> unsigned {
    if (auto vectorType = type.dyn_cast<VectorType>())
      return vectorType.getNumElements();
    return 1;
  };

  if (getNumElements(significandType) != getNumElements(exponentType))
    return ldexpOp.emitOpError(
        "operands must have the same number of elements");

  return success();
}

namespace mlir {
namespace spirv {

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/IR/SPIRVAvailability.cpp.inc"
} // namespace spirv
} // namespace mlir

// TablenGen'erated operation definitions.
#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.cpp.inc"

namespace mlir {
namespace spirv {
// TableGen'erated operation availability interface implementations.
#include "mlir/Dialect/SPIRV/IR/SPIRVOpAvailabilityImpl.inc"

} // namespace spirv
} // namespace mlir
