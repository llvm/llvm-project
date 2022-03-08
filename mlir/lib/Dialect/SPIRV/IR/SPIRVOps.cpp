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

static ParseResult parseOneResultSameOperandTypeOp(OpAsmParser &parser,
                                                   OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  // If the operand list is in-between parentheses, then we have a generic form.
  // (see the fallback in `printOneResultOp`).
  SMLoc loc = parser.getCurrentLocation();
  if (!parser.parseOptionalLParen()) {
    if (parser.parseOperandList(ops) || parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(type))
      return failure();
    auto fnType = type.dyn_cast<FunctionType>();
    if (!fnType) {
      parser.emitError(loc, "expected function type");
      return failure();
    }
    if (parser.resolveOperands(ops, fnType.getInputs(), loc, result.operands))
      return failure();
    result.addTypes(fnType.getResults());
    return success();
  }
  return failure(parser.parseOperandList(ops) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperands(ops, type, result.operands) ||
                 parser.addTypeToList(type, result.types));
}

static void printOneResultOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumResults() == 1 && "op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (llvm::any_of(op->getOperandTypes(),
                   [&](Type type) { return type != resultType; })) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }

  p << ' ';
  p.printOperands(op->getOperands());
  p.printOptionalAttrDict(op->getAttrs());
  // Now we can output only one type for all operands and the result.
  p << " : " << resultType;
}

/// Returns true if the given op is a function-like op or nested in a
/// function-like op without a module-like op in the middle.
static bool isNestedInFunctionOpInterface(Operation *op) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::SymbolTable>())
    return false;
  if (isa<FunctionOpInterface>(op))
    return true;
  return isNestedInFunctionOpInterface(op->getParentOp());
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

  if (integerValueAttr.getType().isSignlessInteger())
    value = integerValueAttr.getInt();
  else
    value = integerValueAttr.getSInt();

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

static ParseResult parseImageOperands(OpAsmParser &parser,
                                      spirv::ImageOperandsAttr &attr) {
  // Expect image operands
  if (parser.parseOptionalLSquare())
    return success();

  spirv::ImageOperands imageOperands;
  if (parseEnumStrAttr(imageOperands, parser))
    return failure();

  attr = spirv::ImageOperandsAttr::get(parser.getContext(), imageOperands);

  return parser.parseRSquare();
}

static void printImageOperands(OpAsmPrinter &printer, Operation *imageOp,
                               spirv::ImageOperandsAttr attr) {
  if (attr) {
    auto strImageOperands = stringifyImageOperands(attr.getValue());
    printer << "[\"" << strImageOperands << "\"]";
  }
}

template <typename Op>
static LogicalResult verifyImageOperands(Op imageOp,
                                         spirv::ImageOperandsAttr attr,
                                         Operation::operand_range operands) {
  if (!attr) {
    if (operands.empty())
      return success();

    return imageOp.emitError("the Image Operands should encode what operands "
                             "follow, as per Image Operands");
  }

  // TODO: Add the validation rules for the following Image Operands.
  spirv::ImageOperands noSupportOperands =
      spirv::ImageOperands::Bias | spirv::ImageOperands::Lod |
      spirv::ImageOperands::Grad | spirv::ImageOperands::ConstOffset |
      spirv::ImageOperands::Offset | spirv::ImageOperands::ConstOffsets |
      spirv::ImageOperands::Sample | spirv::ImageOperands::MinLod |
      spirv::ImageOperands::MakeTexelAvailable |
      spirv::ImageOperands::MakeTexelVisible |
      spirv::ImageOperands::SignExtend | spirv::ImageOperands::ZeroExtend;

  if (spirv::bitEnumContains(attr.getValue(), noSupportOperands))
    llvm_unreachable("unimplemented operands of Image Operands");

  return success();
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

static LogicalResult
verifyMemorySemantics(Operation *op, spirv::MemorySemantics memorySemantics) {
  // According to the SPIR-V specification:
  // "Despite being a mask and allowing multiple bits to be combined, it is
  // invalid for more than one of these four bits to be set: Acquire, Release,
  // AcquireRelease, or SequentiallyConsistent. Requesting both Acquire and
  // Release semantics is done by setting the AcquireRelease bit, not by setting
  // two bits."
  auto atMostOneInSet = spirv::MemorySemantics::Acquire |
                        spirv::MemorySemantics::Release |
                        spirv::MemorySemantics::AcquireRelease |
                        spirv::MemorySemantics::SequentiallyConsistent;

  auto bitCount = llvm::countPopulation(
      static_cast<uint32_t>(memorySemantics & atMostOneInSet));
  if (bitCount > 1) {
    return op->emitError(
        "expected at most one of these four memory constraints "
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
  if (indicesArrayAttr.empty()) {
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
                           SMLoc loc) {
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
  SMLoc loc;
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
  printer << " \"";
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

template <typename T>
static StringRef stringifyTypeName();

template <>
StringRef stringifyTypeName<IntegerType>() {
  return "integer";
}

template <>
StringRef stringifyTypeName<FloatType>() {
  return "float";
}

// Verifies an atomic update op.
template <typename ExpectedElementType>
static LogicalResult verifyAtomicUpdateOp(Operation *op) {
  auto ptrType = op->getOperand(0).getType().cast<spirv::PointerType>();
  auto elementType = ptrType.getPointeeType();
  if (!elementType.isa<ExpectedElementType>())
    return op->emitOpError() << "pointer operand must point to an "
                             << stringifyTypeName<ExpectedElementType>()
                             << " value, found " << elementType;

  if (op->getNumOperands() > 1) {
    auto valueType = op->getOperand(1).getType();
    if (valueType != elementType)
      return op->emitOpError("expected value to have the same type as the "
                             "pointer operand's pointee type ")
             << elementType << ", but found " << valueType;
  }
  auto memorySemantics = static_cast<spirv::MemorySemantics>(
      op->getAttrOfType<IntegerAttr>(kSemanticsAttrName).getInt());
  if (failed(verifyMemorySemantics(op, memorySemantics))) {
    return failure();
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
  printer << " \""
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

/// Result of a logical op must be a scalar or vector of boolean type.
static Type getUnaryOpResultType(Type operandType) {
  Builder builder(operandType.getContext());
  Type resultType = builder.getIntegerType(1);
  if (auto vecType = operandType.dyn_cast<VectorType>())
    return VectorType::get(vecType.getNumElements(), resultType);
  return resultType;
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

ParseResult spirv::AccessChainOp::parse(OpAsmParser &parser,
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
    return mlir::emitError(state.location, "'spv.AccessChain' op expected at "
                                           "least one index ");
  }

  if (parser.parseComma() || parser.parseTypeList(indicesTypes))
    return failure();

  // Check that the indices types list is not empty and that it has a one-to-one
  // mapping to the provided indices.
  if (indicesTypes.size() != indicesInfo.size()) {
    return mlir::emitError(state.location,
                           "'spv.AccessChain' op indices types' count must be "
                           "equal to indices info count");
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

template <typename Op>
static void printAccessChain(Op op, ValueRange indices, OpAsmPrinter &printer) {
  printer << ' ' << op.base_ptr() << '[' << indices
          << "] : " << op.base_ptr().getType() << ", " << indices.getTypes();
}

void spirv::AccessChainOp::print(OpAsmPrinter &printer) {
  printAccessChain(*this, indices(), printer);
}

template <typename Op>
static LogicalResult verifyAccessChain(Op accessChainOp, ValueRange indices) {
  auto resultType = getElementPtrType(accessChainOp.base_ptr().getType(),
                                      indices, accessChainOp.getLoc());
  if (!resultType)
    return failure();

  auto providedResultType =
      accessChainOp.getType().template dyn_cast<spirv::PointerType>();
  if (!providedResultType)
    return accessChainOp.emitOpError(
               "result type must be a pointer, but provided")
           << providedResultType;

  if (resultType != providedResultType)
    return accessChainOp.emitOpError("invalid result type: expected ")
           << resultType << ", but provided " << providedResultType;

  return success();
}

LogicalResult spirv::AccessChainOp::verify() {
  return verifyAccessChain(*this, indices());
}

//===----------------------------------------------------------------------===//
// spv.mlir.addressof
//===----------------------------------------------------------------------===//

void spirv::AddressOfOp::build(OpBuilder &builder, OperationState &state,
                               spirv::GlobalVariableOp var) {
  build(builder, state, var.type(), SymbolRefAttr::get(var));
}

LogicalResult spirv::AddressOfOp::verify() {
  auto varOp = dyn_cast_or_null<spirv::GlobalVariableOp>(
      SymbolTable::lookupNearestSymbolFrom((*this)->getParentOp(),
                                           variableAttr()));
  if (!varOp) {
    return emitOpError("expected spv.GlobalVariable symbol");
  }
  if (pointer().getType() != varOp.type()) {
    return emitOpError(
        "result type mismatch with the referenced global variable's type");
  }
  return success();
}

template <typename T>
static void printAtomicCompareExchangeImpl(T atomOp, OpAsmPrinter &printer) {
  printer << " \"" << stringifyScope(atomOp.memory_scope()) << "\" \""
          << stringifyMemorySemantics(atomOp.equal_semantics()) << "\" \""
          << stringifyMemorySemantics(atomOp.unequal_semantics()) << "\" "
          << atomOp.getOperands() << " : " << atomOp.pointer().getType();
}

static ParseResult parseAtomicCompareExchangeImpl(OpAsmParser &parser,
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

template <typename T>
static LogicalResult verifyAtomicCompareExchangeImpl(T atomOp) {
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

  Type pointeeType = atomOp.pointer()
                         .getType()
                         .template cast<spirv::PointerType>()
                         .getPointeeType();
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
// spv.AtomicAndOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicAndOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicAndOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicAndOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicCompareExchangeOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicCompareExchangeOp::verify() {
  return ::verifyAtomicCompareExchangeImpl(*this);
}

ParseResult spirv::AtomicCompareExchangeOp::parse(OpAsmParser &parser,
                                                  OperationState &result) {
  return ::parseAtomicCompareExchangeImpl(parser, result);
}
void spirv::AtomicCompareExchangeOp::print(OpAsmPrinter &p) {
  ::printAtomicCompareExchangeImpl(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicCompareExchangeWeakOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicCompareExchangeWeakOp::verify() {
  return ::verifyAtomicCompareExchangeImpl(*this);
}

ParseResult spirv::AtomicCompareExchangeWeakOp::parse(OpAsmParser &parser,
                                                      OperationState &result) {
  return ::parseAtomicCompareExchangeImpl(parser, result);
}
void spirv::AtomicCompareExchangeWeakOp::print(OpAsmPrinter &p) {
  ::printAtomicCompareExchangeImpl(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicExchange
//===----------------------------------------------------------------------===//

void spirv::AtomicExchangeOp::print(OpAsmPrinter &printer) {
  printer << " \"" << stringifyScope(memory_scope()) << "\" \""
          << stringifyMemorySemantics(semantics()) << "\" " << getOperands()
          << " : " << pointer().getType();
}

ParseResult spirv::AtomicExchangeOp::parse(OpAsmParser &parser,
                                           OperationState &state) {
  spirv::Scope memoryScope;
  spirv::MemorySemantics semantics;
  SmallVector<OpAsmParser::OperandType, 2> operandInfo;
  Type type;
  if (parseEnumStrAttr(memoryScope, parser, state, kMemoryScopeAttrName) ||
      parseEnumStrAttr(semantics, parser, state, kSemanticsAttrName) ||
      parser.parseOperandList(operandInfo, 2))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parser.parseColonType(type))
    return failure();

  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return parser.emitError(loc, "expected pointer type");

  if (parser.resolveOperands(operandInfo, {ptrType, ptrType.getPointeeType()},
                             parser.getNameLoc(), state.operands))
    return failure();

  return parser.addTypeToList(ptrType.getPointeeType(), state.types);
}

LogicalResult spirv::AtomicExchangeOp::verify() {
  if (getType() != value().getType())
    return emitOpError("value operand must have the same type as the op "
                       "result, but found ")
           << value().getType() << " vs " << getType();

  Type pointeeType =
      pointer().getType().cast<spirv::PointerType>().getPointeeType();
  if (getType() != pointeeType)
    return emitOpError("pointer operand's pointee type must have the same "
                       "as the op result type, but found ")
           << pointeeType << " vs " << getType();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.AtomicIAddOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicIAddOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicIAddOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicIAddOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicFAddEXTOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicFAddEXTOp::verify() {
  return ::verifyAtomicUpdateOp<FloatType>(getOperation());
}

ParseResult spirv::AtomicFAddEXTOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicFAddEXTOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicIDecrementOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicIDecrementOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicIDecrementOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, false);
}
void spirv::AtomicIDecrementOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicIIncrementOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicIIncrementOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicIIncrementOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, false);
}
void spirv::AtomicIIncrementOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicISubOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicISubOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicISubOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicISubOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicOrOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicOrOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicOrOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicOrOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicSMaxOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicSMaxOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicSMaxOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicSMaxOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicSMinOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicSMinOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicSMinOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicSMinOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicUMaxOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicUMaxOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicUMaxOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicUMaxOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicUMinOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicUMinOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicUMinOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicUMinOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.AtomicXorOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::AtomicXorOp::verify() {
  return ::verifyAtomicUpdateOp<IntegerType>(getOperation());
}

ParseResult spirv::AtomicXorOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return ::parseAtomicUpdateOp(parser, result, true);
}
void spirv::AtomicXorOp::print(OpAsmPrinter &p) {
  ::printAtomicUpdateOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.BitcastOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::BitcastOp::verify() {
  // TODO: The SPIR-V spec validation rules are different for different
  // versions.
  auto operandType = operand().getType();
  auto resultType = result().getType();
  if (operandType == resultType) {
    return emitError("result type must be different from operand type");
  }
  if (operandType.isa<spirv::PointerType>() &&
      !resultType.isa<spirv::PointerType>()) {
    return emitError(
        "unhandled bit cast conversion from pointer type to non-pointer type");
  }
  if (!operandType.isa<spirv::PointerType>() &&
      resultType.isa<spirv::PointerType>()) {
    return emitError(
        "unhandled bit cast conversion from non-pointer type to pointer type");
  }
  auto operandBitWidth = getBitWidth(operandType);
  auto resultBitWidth = getBitWidth(resultType);
  if (operandBitWidth != resultBitWidth) {
    return emitOpError("mismatch in result type bitwidth ")
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

ParseResult spirv::BranchConditionalOp::parse(OpAsmParser &parser,
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

void spirv::BranchConditionalOp::print(OpAsmPrinter &printer) {
  printer << ' ' << condition();

  if (auto weights = branch_weights()) {
    printer << " [";
    llvm::interleaveComma(weights->getValue(), printer, [&](Attribute a) {
      printer << a.cast<IntegerAttr>().getInt();
    });
    printer << "]";
  }

  printer << ", ";
  printer.printSuccessorAndUseList(getTrueBlock(), getTrueBlockArguments());
  printer << ", ";
  printer.printSuccessorAndUseList(getFalseBlock(), getFalseBlockArguments());
}

LogicalResult spirv::BranchConditionalOp::verify() {
  if (auto weights = branch_weights()) {
    if (weights->getValue().size() != 2) {
      return emitOpError("must have exactly two branch weights");
    }
    if (llvm::all_of(*weights, [](Attribute attr) {
          return attr.cast<IntegerAttr>().getValue().isNullValue();
        }))
      return emitOpError("branch weights cannot both be zero");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CompositeConstruct
//===----------------------------------------------------------------------===//

ParseResult spirv::CompositeConstructOp::parse(OpAsmParser &parser,
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

void spirv::CompositeConstructOp::print(OpAsmPrinter &printer) {
  printer << " " << constituents() << " : " << getResult().getType();
}

LogicalResult spirv::CompositeConstructOp::verify() {
  auto cType = getType().cast<spirv::CompositeType>();
  operand_range constituents = this->constituents();

  if (cType.isa<spirv::CooperativeMatrixNVType>()) {
    if (constituents.size() != 1)
      return emitError("has incorrect number of operands: expected ")
             << "1, but provided " << constituents.size();
  } else if (constituents.size() != cType.getNumElements()) {
    return emitError("has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided "
           << constituents.size();
  }

  for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
    if (constituents[index].getType() != cType.getElementType(index)) {
      return emitError("operand type mismatch: expected operand type ")
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

ParseResult spirv::CompositeExtractOp::parse(OpAsmParser &parser,
                                             OperationState &state) {
  OpAsmParser::OperandType compositeInfo;
  Attribute indicesAttr;
  Type compositeType;
  SMLoc attrLocation;

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

void spirv::CompositeExtractOp::print(OpAsmPrinter &printer) {
  printer << ' ' << composite() << indices() << " : " << composite().getType();
}

LogicalResult spirv::CompositeExtractOp::verify() {
  auto indicesArrayAttr = indices().dyn_cast<ArrayAttr>();
  auto resultType =
      getElementType(composite().getType(), indicesArrayAttr, getLoc());
  if (!resultType)
    return failure();

  if (resultType != getType()) {
    return emitOpError("invalid result type: expected ")
           << resultType << " but provided " << getType();
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

ParseResult spirv::CompositeInsertOp::parse(OpAsmParser &parser,
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

LogicalResult spirv::CompositeInsertOp::verify() {
  auto indicesArrayAttr = indices().dyn_cast<ArrayAttr>();
  auto objectType =
      getElementType(composite().getType(), indicesArrayAttr, getLoc());
  if (!objectType)
    return failure();

  if (objectType != object().getType()) {
    return emitOpError("object operand type should be ")
           << objectType << ", but found " << object().getType();
  }

  if (composite().getType() != getType()) {
    return emitOpError("result type should be the same as "
                       "the composite type, but found ")
           << composite().getType() << " vs " << getType();
  }

  return success();
}

void spirv::CompositeInsertOp::print(OpAsmPrinter &printer) {
  printer << " " << object() << ", " << composite() << indices() << " : "
          << object().getType() << " into " << composite().getType();
}

//===----------------------------------------------------------------------===//
// spv.Constant
//===----------------------------------------------------------------------===//

ParseResult spirv::ConstantOp::parse(OpAsmParser &parser,
                                     OperationState &state) {
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

void spirv::ConstantOp::print(OpAsmPrinter &printer) {
  printer << ' ' << value();
  if (getType().isa<spirv::ArrayType>())
    printer << " : " << getType();
}

static LogicalResult verifyConstantType(spirv::ConstantOp op, Attribute value,
                                        Type opType) {
  auto valueType = value.getType();

  if (value.isa<IntegerAttr, FloatAttr>()) {
    if (valueType != opType)
      return op.emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  }
  if (value.isa<DenseIntOrFPElementsAttr, SparseElementsAttr>()) {
    if (valueType == opType)
      return success();
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    auto shapedType = valueType.dyn_cast<ShapedType>();
    if (!arrayType)
      return op.emitOpError("result or element type (")
             << opType << ") does not match value type (" << valueType
             << "), must be the same or spv.array";

    int numElements = arrayType.getNumElements();
    auto opElemType = arrayType.getElementType();
    while (auto t = opElemType.dyn_cast<spirv::ArrayType>()) {
      numElements *= t.getNumElements();
      opElemType = t.getElementType();
    }
    if (!opElemType.isIntOrFloat())
      return op.emitOpError("only support nested array result type");

    auto valueElemType = shapedType.getElementType();
    if (valueElemType != opElemType) {
      return op.emitOpError("result element type (")
             << opElemType << ") does not match value element type ("
             << valueElemType << ")";
    }

    if (numElements != shapedType.getNumElements()) {
      return op.emitOpError("result number of elements (")
             << numElements << ") does not match value number of elements ("
             << shapedType.getNumElements() << ")";
    }
    return success();
  }
  if (auto arrayAttr = value.dyn_cast<ArrayAttr>()) {
    auto arrayType = opType.dyn_cast<spirv::ArrayType>();
    if (!arrayType)
      return op.emitOpError("must have spv.array result type for array value");
    Type elemType = arrayType.getElementType();
    for (Attribute element : arrayAttr.getValue()) {
      // Verify array elements recursively.
      if (failed(verifyConstantType(op, element, elemType)))
        return failure();
    }
    return success();
  }
  return op.emitOpError("cannot have value of type ") << valueType;
}

LogicalResult spirv::ConstantOp::verify() {
  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  return verifyConstantType(*this, valueAttr(), getType());
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

void mlir::spirv::ConstantOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  Type type = getType();

  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "cst";

  IntegerType intTy = type.dyn_cast<IntegerType>();

  if (IntegerAttr intCst = value().dyn_cast<IntegerAttr>()) {
    if (intTy && intTy.getWidth() == 1) {
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));
    }

    if (intTy.isSignless()) {
      specialName << intCst.getInt();
    } else {
      specialName << intCst.getSInt();
    }
  }

  if (intTy || type.isa<FloatType>()) {
    specialName << '_' << type;
  }

  if (auto vecType = type.dyn_cast<VectorType>()) {
    specialName << "_vec_";
    specialName << vecType.getDimSize(0);

    Type elementType = vecType.getElementType();

    if (elementType.isa<IntegerType>() || elementType.isa<FloatType>()) {
      specialName << "x" << elementType;
    }
  }

  setNameFn(getResult(), specialName.str());
}

void mlir::spirv::AddressOfOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << variable() << "_addr";
  setNameFn(getResult(), specialName.str());
}

//===----------------------------------------------------------------------===//
// spv.ControlBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ControlBarrierOp::verify() {
  return verifyMemorySemantics(getOperation(), memory_semantics());
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToSOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ConvertFToSOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false,
                      /*skipBitWidthCheck=*/true);
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToUOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ConvertFToUOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false,
                      /*skipBitWidthCheck=*/true);
}

//===----------------------------------------------------------------------===//
// spv.ConvertSToFOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ConvertSToFOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false,
                      /*skipBitWidthCheck=*/true);
}

//===----------------------------------------------------------------------===//
// spv.ConvertUToFOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ConvertUToFOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false,
                      /*skipBitWidthCheck=*/true);
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
        SymbolRefAttr::get(function), builder.getArrayAttr(interfaceVars));
}

ParseResult spirv::EntryPointOp::parse(OpAsmParser &parser,
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
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          // The name of the interface variable attribute isnt important
          FlatSymbolRefAttr var;
          NamedAttrList attrs;
          if (parser.parseAttribute(var, Type(), "var_symbol", attrs))
            return failure();
          interfaceVars.push_back(var);
          return success();
        }))
      return failure();
  }
  state.addAttribute(kInterfaceAttrName,
                     parser.getBuilder().getArrayAttr(interfaceVars));
  return success();
}

void spirv::EntryPointOp::print(OpAsmPrinter &printer) {
  printer << " \"" << stringifyExecutionModel(execution_model()) << "\" ";
  printer.printSymbolName(fn());
  auto interfaceVars = interface().getValue();
  if (!interfaceVars.empty()) {
    printer << ", ";
    llvm::interleaveComma(interfaceVars, printer);
  }
}

LogicalResult spirv::EntryPointOp::verify() {
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
  build(builder, state, SymbolRefAttr::get(function),
        spirv::ExecutionModeAttr::get(builder.getContext(), executionMode),
        builder.getI32ArrayAttr(params));
}

ParseResult spirv::ExecutionModeOp::parse(OpAsmParser &parser,
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

void spirv::ExecutionModeOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(fn());
  printer << " \"" << stringifyExecutionMode(execution_mode()) << "\"";
  auto values = this->values();
  if (values.empty())
    return;
  printer << ", ";
  llvm::interleaveComma(values, printer, [&](Attribute a) {
    printer << a.cast<IntegerAttr>().getInt();
  });
}

//===----------------------------------------------------------------------===//
// spv.FConvertOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::FConvertOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false);
}

//===----------------------------------------------------------------------===//
// spv.SConvertOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::SConvertOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false);
}

//===----------------------------------------------------------------------===//
// spv.UConvertOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::UConvertOp::verify() {
  return verifyCastOp(*this, /*requireSameBitWidth=*/false);
}

//===----------------------------------------------------------------------===//
// spv.func
//===----------------------------------------------------------------------===//

ParseResult spirv::FuncOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType> entryArgs;
  SmallVector<NamedAttrList> argAttrs;
  SmallVector<NamedAttrList> resultAttrs;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;
  SmallVector<Location> argLocations;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs,
          argLocations, isVariadic, resultTypes, resultAttrs))
    return failure();

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  state.addAttribute(FunctionOpInterface::getTypeAttrName(),
                     TypeAttr::get(fnType));

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
  function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                                resultAttrs);

  // Parse the optional function body.
  auto *body = state.addRegion();
  OptionalParseResult result = parser.parseOptionalRegion(
      *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes);
  return failure(result.hasValue() && failed(*result));
}

void spirv::FuncOp::print(OpAsmPrinter &printer) {
  // Print function name, signature, and control.
  printer << " ";
  printer.printSymbolName(sym_name());
  auto fnType = getType();
  function_interface_impl::printFunctionSignature(
      printer, *this, fnType.getInputs(),
      /*isVariadic=*/false, fnType.getResults());
  printer << " \"" << spirv::stringifyFunctionControl(function_control())
          << "\"";
  function_interface_impl::printFunctionAttributes(
      printer, *this, fnType.getNumInputs(), fnType.getNumResults(),
      {spirv::attributeName<spirv::FunctionControl>()});

  // Print the body if this is not an external function.
  Region &body = this->body();
  if (!body.empty()) {
    printer << ' ';
    printer.printRegion(body, /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
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

LogicalResult spirv::FunctionCallOp::verify() {
  auto fnName = calleeAttr();

  auto funcOp = dyn_cast_or_null<spirv::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom((*this)->getParentOp(), fnName));
  if (!funcOp) {
    return emitOpError("callee function '")
           << fnName.getValue() << "' not found in nearest symbol table";
  }

  auto functionType = funcOp.getType();

  if (getNumResults() > 1) {
    return emitOpError(
               "expected callee function to have 0 or 1 result, but provided ")
           << getNumResults();
  }

  if (functionType.getNumInputs() != getNumOperands()) {
    return emitOpError("has incorrect number of operands for callee: expected ")
           << functionType.getNumInputs() << ", but provided "
           << getNumOperands();
  }

  for (uint32_t i = 0, e = functionType.getNumInputs(); i != e; ++i) {
    if (getOperand(i).getType() != functionType.getInput(i)) {
      return emitOpError("operand type mismatch: expected operand type ")
             << functionType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }
  }

  if (functionType.getNumResults() != getNumResults()) {
    return emitOpError(
               "has incorrect number of results has for callee: expected ")
           << functionType.getNumResults() << ", but provided "
           << getNumResults();
  }

  if (getNumResults() &&
      (getResult(0).getType() != functionType.getResult(0))) {
    return emitOpError("result type mismatch: expected ")
           << functionType.getResult(0) << ", but provided "
           << getResult(0).getType();
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
// spv.GLSLFClampOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLSLFClampOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLSLFClampOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spv.GLSLUClampOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLSLUClampOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLSLUClampOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spv.GLSLSClampOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLSLSClampOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLSLSClampOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spv.GLSLFmaOp
//===----------------------------------------------------------------------===//

ParseResult spirv::GLSLFmaOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseOneResultSameOperandTypeOp(parser, result);
}
void spirv::GLSLFmaOp::print(OpAsmPrinter &p) { printOneResultOp(*this, p); }

//===----------------------------------------------------------------------===//
// spv.GlobalVariable
//===----------------------------------------------------------------------===//

void spirv::GlobalVariableOp::build(OpBuilder &builder, OperationState &state,
                                    Type type, StringRef name,
                                    unsigned descriptorSet, unsigned binding) {
  build(builder, state, TypeAttr::get(type), builder.getStringAttr(name));
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
  build(builder, state, TypeAttr::get(type), builder.getStringAttr(name));
  state.addAttribute(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::BuiltIn),
      builder.getStringAttr(spirv::stringifyBuiltIn(builtin)));
}

ParseResult spirv::GlobalVariableOp::parse(OpAsmParser &parser,
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

void spirv::GlobalVariableOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};

  // Print variable name.
  printer << ' ';
  printer.printSymbolName(sym_name());
  elidedAttrs.push_back(SymbolTable::getSymbolAttrName());

  // Print optional initializer
  if (auto initializer = this->initializer()) {
    printer << " " << kInitializerAttrName << '(';
    printer.printSymbolName(initializer.getValue());
    printer << ')';
    elidedAttrs.push_back(kInitializerAttrName);
  }

  elidedAttrs.push_back(kTypeAttrName);
  printVariableDecorations(*this, printer, elidedAttrs);
  printer << " : " << type();
}

LogicalResult spirv::GlobalVariableOp::verify() {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  // Also, Function storage class is reserved by spv.Variable.
  auto storageClass = this->storageClass();
  if (storageClass == spirv::StorageClass::Generic ||
      storageClass == spirv::StorageClass::Function) {
    return emitOpError("storage class cannot be '")
           << stringifyStorageClass(storageClass) << "'";
  }

  if (auto init =
          (*this)->getAttrOfType<FlatSymbolRefAttr>(kInitializerAttrName)) {
    Operation *initOp = SymbolTable::lookupNearestSymbolFrom(
        (*this)->getParentOp(), init.getAttr());
    // TODO: Currently only variable initialization with specialization
    // constants and other variables is supported. They could be normal
    // constants in the module scope as well.
    if (!initOp ||
        !isa<spirv::GlobalVariableOp, spirv::SpecConstantOp>(initOp)) {
      return emitOpError("initializer must be result of a "
                         "spv.SpecConstant or spv.GlobalVariable op");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupBroadcast
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupBroadcastOp::verify() {
  spirv::Scope scope = execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return emitOpError("execution scope must be 'Workgroup' or 'Subgroup'");

  if (auto localIdTy = localid().getType().dyn_cast<VectorType>())
    if (!(localIdTy.getNumElements() == 2 || localIdTy.getNumElements() == 3))
      return emitOpError("localid is a vector and can be with only "
                         " 2 or 3 components, actual number is ")
             << localIdTy.getNumElements();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformBallotOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformBallotOp::verify() {
  spirv::Scope scope = execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return emitOpError("execution scope must be 'Workgroup' or 'Subgroup'");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformBroadcast
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformBroadcastOp::verify() {
  spirv::Scope scope = execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return emitOpError("execution scope must be 'Workgroup' or 'Subgroup'");

  // SPIR-V spec: "Before version 1.5, Id must come from a
  // constant instruction.
  auto targetEnv = spirv::getDefaultTargetEnv(getContext());
  if (auto spirvModule = (*this)->getParentOfType<spirv::ModuleOp>())
    targetEnv = spirv::lookupTargetEnvOrDefault(spirvModule);

  if (targetEnv.getVersion() < spirv::Version::V_1_5) {
    auto *idOp = id().getDefiningOp();
    if (!idOp || !isa<spirv::ConstantOp,           // for normal constant
                      spirv::ReferenceOfOp>(idOp)) // for spec constant
      return emitOpError("id must be the result of a constant op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SubgroupBlockReadINTEL
//===----------------------------------------------------------------------===//

ParseResult spirv::SubgroupBlockReadINTELOp::parse(OpAsmParser &parser,
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

void spirv::SubgroupBlockReadINTELOp::print(OpAsmPrinter &printer) {
  printer << " " << ptr() << " : " << getType();
}

LogicalResult spirv::SubgroupBlockReadINTELOp::verify() {
  if (failed(verifyBlockReadWritePtrAndValTypes(*this, ptr(), value())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SubgroupBlockWriteINTEL
//===----------------------------------------------------------------------===//

ParseResult spirv::SubgroupBlockWriteINTELOp::parse(OpAsmParser &parser,
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

void spirv::SubgroupBlockWriteINTELOp::print(OpAsmPrinter &printer) {
  printer << " " << ptr() << ", " << value() << " : " << value().getType();
}

LogicalResult spirv::SubgroupBlockWriteINTELOp::verify() {
  if (failed(verifyBlockReadWritePtrAndValTypes(*this, ptr(), value())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformElectOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformElectOp::verify() {
  spirv::Scope scope = execution_scope();
  if (scope != spirv::Scope::Workgroup && scope != spirv::Scope::Subgroup)
    return emitOpError("execution scope must be 'Workgroup' or 'Subgroup'");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFAddOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformFAddOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformFAddOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformFAddOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFMaxOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformFMaxOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformFMaxOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformFMaxOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFMinOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformFMinOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformFMinOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformFMinOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformFMulOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformFMulOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformFMulOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformFMulOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformIAddOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformIAddOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformIAddOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformIAddOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformIMulOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformIMulOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformIMulOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformIMulOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformSMaxOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformSMaxOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformSMaxOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformSMaxOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformSMinOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformSMinOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformSMinOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformSMinOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformUMaxOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformUMaxOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformUMaxOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformUMaxOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
}

//===----------------------------------------------------------------------===//
// spv.GroupNonUniformUMinOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GroupNonUniformUMinOp::verify() {
  return verifyGroupNonUniformArithmeticOp(*this);
}

ParseResult spirv::GroupNonUniformUMinOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  return parseGroupNonUniformArithmeticOp(parser, result);
}
void spirv::GroupNonUniformUMinOp::print(OpAsmPrinter &p) {
  printGroupNonUniformArithmeticOp(*this, p);
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

ParseResult spirv::LoadOp::parse(OpAsmParser &parser, OperationState &state) {
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

void spirv::LoadOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      ptr().getType().cast<spirv::PointerType>().getStorageClass());
  printer << " \"" << sc << "\" " << ptr();

  printMemoryAccessAttribute(*this, printer, elidedAttrs);

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << " : " << getType();
}

LogicalResult spirv::LoadOp::verify() {
  // SPIR-V spec : "Result Type is the type of the loaded object. It must be a
  // type with fixed size; i.e., it cannot be, nor include, any
  // OpTypeRuntimeArray types."
  if (failed(verifyLoadStorePtrAndValTypes(*this, ptr(), value()))) {
    return failure();
  }
  return verifyMemoryAccessAttribute(*this);
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

ParseResult spirv::LoopOp::parse(OpAsmParser &parser, OperationState &state) {
  if (parseControlAttribute<spirv::LoopControl>(parser, state))
    return failure();
  return parser.parseRegion(*state.addRegion(), /*arguments=*/{},
                            /*argTypes=*/{});
}

void spirv::LoopOp::print(OpAsmPrinter &printer) {
  auto control = loop_control();
  if (control != spirv::LoopControl::None)
    printer << " control(" << spirv::stringifyLoopControl(control) << ")";
  printer << ' ';
  printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
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

LogicalResult spirv::LoopOp::verify() {
  auto *op = getOperation();

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
    return emitOpError(
        "last block must be the merge block with only one 'spv.mlir.merge' op");

  if (std::next(region.begin()) == region.end())
    return emitOpError(
        "must have an entry block branching to the loop header block");
  // The first block is the entry block.
  Block &entry = region.front();

  if (std::next(region.begin(), 2) == region.end())
    return emitOpError(
        "must have a loop header block branched from the entry block");
  // The second block is the loop header block.
  Block &header = *std::next(region.begin(), 1);

  if (!hasOneBranchOpTo(entry, header))
    return emitOpError(
        "entry block must only have one 'spv.Branch' op to the second block");

  if (std::next(region.begin(), 3) == region.end())
    return emitOpError(
        "requires a loop continue block branching to the loop header block");
  // The second to last block is the loop continue block.
  Block &cont = *std::prev(region.end(), 2);

  // Make sure that we have a branch from the loop continue block to the loop
  // header block.
  if (llvm::none_of(
          llvm::seq<unsigned>(0, cont.getNumSuccessors()),
          [&](unsigned index) { return cont.getSuccessor(index) == &header; }))
    return emitOpError("second to last block must be the loop continue "
                       "block that branches to the loop header block");

  // Make sure that no other blocks (except the entry and loop continue block)
  // branches to the loop header block.
  for (auto &block : llvm::make_range(std::next(region.begin(), 2),
                                      std::prev(region.end(), 2))) {
    for (auto i : llvm::seq<unsigned>(0, block.getNumSuccessors())) {
      if (block.getSuccessor(i) == &header) {
        return emitOpError("can only have the entry and loop continue "
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
// spv.MemoryBarrierOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::MemoryBarrierOp::verify() {
  return verifyMemorySemantics(getOperation(), memory_semantics());
}

//===----------------------------------------------------------------------===//
// spv.mlir.merge
//===----------------------------------------------------------------------===//

LogicalResult spirv::MergeOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (!parentOp || !isa<spirv::SelectionOp, spirv::LoopOp>(parentOp))
    return emitOpError(
        "expected parent op to be 'spv.mlir.selection' or 'spv.mlir.loop'");

  Block &parentLastBlock = (*this)->getParentRegion()->back();
  if (getOperation() != parentLastBlock.getTerminator())
    return emitOpError("can only be used in the last block of "
                       "'spv.mlir.selection' or 'spv.mlir.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

void spirv::ModuleOp::build(OpBuilder &builder, OperationState &state,
                            Optional<StringRef> name) {
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(state.addRegion());
  if (name) {
    state.attributes.append(mlir::SymbolTable::getSymbolAttrName(),
                            builder.getStringAttr(*name));
  }
}

void spirv::ModuleOp::build(OpBuilder &builder, OperationState &state,
                            spirv::AddressingModel addressingModel,
                            spirv::MemoryModel memoryModel,
                            Optional<VerCapExtAttr> vceTriple,
                            Optional<StringRef> name) {
  state.addAttribute(
      "addressing_model",
      builder.getI32IntegerAttr(static_cast<int32_t>(addressingModel)));
  state.addAttribute("memory_model", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(memoryModel)));
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(state.addRegion());
  if (vceTriple)
    state.addAttribute(getVCETripleAttrName(), *vceTriple);
  if (name)
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(*name));
}

ParseResult spirv::ModuleOp::parse(OpAsmParser &parser, OperationState &state) {
  Region *body = state.addRegion();

  // If the name is present, parse it.
  StringAttr nameAttr;
  parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), state.attributes);

  // Parse attributes
  spirv::AddressingModel addrModel;
  spirv::MemoryModel memoryModel;
  if (::parseEnumKeywordAttr(addrModel, parser, state) ||
      ::parseEnumKeywordAttr(memoryModel, parser, state))
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

  // Make sure we have at least one block.
  if (body->empty())
    body->push_back(new Block());

  return success();
}

void spirv::ModuleOp::print(OpAsmPrinter &printer) {
  if (Optional<StringRef> name = getName()) {
    printer << ' ';
    printer.printSymbolName(*name);
  }

  SmallVector<StringRef, 2> elidedAttrs;

  printer << " " << spirv::stringifyAddressingModel(addressing_model()) << " "
          << spirv::stringifyMemoryModel(memory_model());
  auto addressingModelAttrName = spirv::attributeName<spirv::AddressingModel>();
  auto memoryModelAttrName = spirv::attributeName<spirv::MemoryModel>();
  elidedAttrs.assign({addressingModelAttrName, memoryModelAttrName,
                      mlir::SymbolTable::getSymbolAttrName()});

  if (Optional<spirv::VerCapExtAttr> triple = vce_triple()) {
    printer << " requires " << *triple;
    elidedAttrs.push_back(spirv::ModuleOp::getVCETripleAttrName());
  }

  printer.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  printer << ' ';
  printer.printRegion(getRegion());
}

LogicalResult spirv::ModuleOp::verify() {
  Dialect *dialect = (*this)->getDialect();
  DenseMap<std::pair<spirv::FuncOp, spirv::ExecutionModel>, spirv::EntryPointOp>
      entryPoints;
  mlir::SymbolTable table(*this);

  for (auto &op : *getBody()) {
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

LogicalResult spirv::ReferenceOfOp::verify() {
  auto *specConstSym = SymbolTable::lookupNearestSymbolFrom(
      (*this)->getParentOp(), spec_constAttr());
  Type constType;

  auto specConstOp = dyn_cast_or_null<spirv::SpecConstantOp>(specConstSym);
  if (specConstOp)
    constType = specConstOp.default_value().getType();

  auto specConstCompositeOp =
      dyn_cast_or_null<spirv::SpecConstantCompositeOp>(specConstSym);
  if (specConstCompositeOp)
    constType = specConstCompositeOp.type();

  if (!specConstOp && !specConstCompositeOp)
    return emitOpError(
        "expected spv.SpecConstant or spv.SpecConstantComposite symbol");

  if (reference().getType() != constType)
    return emitOpError("result type mismatch with the referenced "
                       "specialization constant's type");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

LogicalResult spirv::ReturnOp::verify() {
  // Verification is performed in spv.func op.
  return success();
}

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

LogicalResult spirv::ReturnValueOp::verify() {
  // Verification is performed in spv.func op.
  return success();
}

//===----------------------------------------------------------------------===//
// spv.Select
//===----------------------------------------------------------------------===//

LogicalResult spirv::SelectOp::verify() {
  if (auto conditionTy = condition().getType().dyn_cast<VectorType>()) {
    auto resultVectorTy = result().getType().dyn_cast<VectorType>();
    if (!resultVectorTy) {
      return emitOpError("result expected to be of vector type when "
                         "condition is of vector type");
    }
    if (resultVectorTy.getNumElements() != conditionTy.getNumElements()) {
      return emitOpError("result should have the same number of elements as "
                         "the condition when condition is of vector type");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.mlir.selection
//===----------------------------------------------------------------------===//

ParseResult spirv::SelectionOp::parse(OpAsmParser &parser,
                                      OperationState &state) {
  if (parseControlAttribute<spirv::SelectionControl>(parser, state))
    return failure();
  return parser.parseRegion(*state.addRegion(), /*arguments=*/{},
                            /*argTypes=*/{});
}

void spirv::SelectionOp::print(OpAsmPrinter &printer) {
  auto control = selection_control();
  if (control != spirv::SelectionControl::None)
    printer << " control(" << spirv::stringifySelectionControl(control) << ")";
  printer << ' ';
  printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

LogicalResult spirv::SelectionOp::verify() {
  auto *op = getOperation();

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
    return emitOpError(
        "last block must be the merge block with only one 'spv.mlir.merge' op");

  if (std::next(region.begin()) == region.end())
    return emitOpError("must have a selection header block");

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

ParseResult spirv::SpecConstantOp::parse(OpAsmParser &parser,
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

void spirv::SpecConstantOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(sym_name());
  if (auto specID = (*this)->getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    printer << ' ' << kSpecIdAttrName << '(' << specID.getInt() << ')';
  printer << " = " << default_value();
}

LogicalResult spirv::SpecConstantOp::verify() {
  if (auto specID = (*this)->getAttrOfType<IntegerAttr>(kSpecIdAttrName))
    if (specID.getValue().isNegative())
      return emitOpError("SpecId cannot be negative");

  auto value = default_value();
  if (value.isa<IntegerAttr, FloatAttr>()) {
    // Make sure bitwidth is allowed.
    if (!value.getType().isa<spirv::SPIRVType>())
      return emitOpError("default value bitwidth disallowed");
    return success();
  }
  return emitOpError(
      "default value can only be a bool, integer, or float scalar");
}

//===----------------------------------------------------------------------===//
// spv.StoreOp
//===----------------------------------------------------------------------===//

ParseResult spirv::StoreOp::parse(OpAsmParser &parser, OperationState &state) {
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

void spirv::StoreOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs;
  StringRef sc = stringifyStorageClass(
      ptr().getType().cast<spirv::PointerType>().getStorageClass());
  printer << " \"" << sc << "\" " << ptr() << ", " << value();

  printMemoryAccessAttribute(*this, printer, elidedAttrs);

  printer << " : " << value().getType();
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

LogicalResult spirv::StoreOp::verify() {
  // SPIR-V spec : "Pointer is the pointer to store through. Its type must be an
  // OpTypePointer whose Type operand is the same as the type of Object."
  if (failed(verifyLoadStorePtrAndValTypes(*this, ptr(), value())))
    return failure();
  return verifyMemoryAccessAttribute(*this);
}

//===----------------------------------------------------------------------===//
// spv.Unreachable
//===----------------------------------------------------------------------===//

LogicalResult spirv::UnreachableOp::verify() {
  auto *block = (*this)->getBlock();
  // Fast track: if this is in entry block, its invalid. Otherwise, if no
  // predecessors, it's valid.
  if (block->isEntryBlock())
    return emitOpError("cannot be used in reachable block");
  if (block->hasNoPredecessors())
    return success();

  // TODO: further verification needs to analyze reachability from
  // the entry block.

  return success();
}

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

ParseResult spirv::VariableOp::parse(OpAsmParser &parser,
                                     OperationState &state) {
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

void spirv::VariableOp::print(OpAsmPrinter &printer) {
  SmallVector<StringRef, 4> elidedAttrs{
      spirv::attributeName<spirv::StorageClass>()};
  // Print optional initializer
  if (getNumOperands() != 0)
    printer << " init(" << initializer() << ")";

  printVariableDecorations(*this, printer, elidedAttrs);
  printer << " : " << getType();
}

LogicalResult spirv::VariableOp::verify() {
  // SPIR-V spec: "Storage Class is the Storage Class of the memory holding the
  // object. It cannot be Generic. It must be the same as the Storage Class
  // operand of the Result Type."
  if (storage_class() != spirv::StorageClass::Function) {
    return emitOpError(
        "can only be used to model function-level variables. Use "
        "spv.GlobalVariable for module-level variables.");
  }

  auto pointerType = pointer().getType().cast<spirv::PointerType>();
  if (storage_class() != pointerType.getStorageClass())
    return emitOpError(
        "storage class must match result pointer's storage class");

  if (getNumOperands() != 0) {
    // SPIR-V spec: "Initializer must be an <id> from a constant instruction or
    // a global (module scope) OpVariable instruction".
    auto *initOp = getOperand(0).getDefiningOp();
    if (!initOp || !isa<spirv::ConstantOp,    // for normal constant
                        spirv::ReferenceOfOp, // for spec constant
                        spirv::AddressOfOp>(initOp))
      return emitOpError("initializer must be the result of a "
                         "constant or spv.GlobalVariable op");
  }

  // TODO: generate these strings using ODS.
  auto *op = getOperation();
  auto descriptorSetName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::DescriptorSet));
  auto bindingName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::Binding));
  auto builtInName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::BuiltIn));

  for (const auto &attr : {descriptorSetName, bindingName, builtInName}) {
    if (op->getAttr(attr))
      return emitOpError("cannot have '")
             << attr << "' attribute (only allowed in spv.GlobalVariable)";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.VectorShuffle
//===----------------------------------------------------------------------===//

LogicalResult spirv::VectorShuffleOp::verify() {
  VectorType resultType = getType().cast<VectorType>();

  size_t numResultElements = resultType.getNumElements();
  if (numResultElements != components().size())
    return emitOpError("result type element count (")
           << numResultElements
           << ") mismatch with the number of component selectors ("
           << components().size() << ")";

  size_t totalSrcElements =
      vector1().getType().cast<VectorType>().getNumElements() +
      vector2().getType().cast<VectorType>().getNumElements();

  for (const auto &selector : components().getAsValueRange<IntegerAttr>()) {
    uint32_t index = selector.getZExtValue();
    if (index >= totalSrcElements &&
        index != std::numeric_limits<uint32_t>().max())
      return emitOpError("component selector ")
             << index << " out of range: expected to be in [0, "
             << totalSrcElements << ") or 0xffffffff";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spv.CooperativeMatrixLoadNV
//===----------------------------------------------------------------------===//

ParseResult spirv::CooperativeMatrixLoadNVOp::parse(OpAsmParser &parser,
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

void spirv::CooperativeMatrixLoadNVOp::print(OpAsmPrinter &printer) {
  printer << " " << pointer() << ", " << stride() << ", " << columnmajor();
  // Print optional memory access attribute.
  if (auto memAccess = memory_access())
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"]";
  printer << " : " << pointer().getType() << " as " << getType();
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

LogicalResult spirv::CooperativeMatrixLoadNVOp::verify() {
  return verifyPointerAndCoopMatrixType(*this, pointer().getType(),
                                        result().getType());
}

//===----------------------------------------------------------------------===//
// spv.CooperativeMatrixStoreNV
//===----------------------------------------------------------------------===//

ParseResult spirv::CooperativeMatrixStoreNVOp::parse(OpAsmParser &parser,
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

void spirv::CooperativeMatrixStoreNVOp::print(OpAsmPrinter &printer) {
  printer << " " << pointer() << ", " << object() << ", " << stride() << ", "
          << columnmajor();
  // Print optional memory access attribute.
  if (auto memAccess = memory_access())
    printer << " [\"" << stringifyMemoryAccess(*memAccess) << "\"]";
  printer << " : " << pointer().getType() << ", " << getOperand(1).getType();
}

LogicalResult spirv::CooperativeMatrixStoreNVOp::verify() {
  return verifyPointerAndCoopMatrixType(*this, pointer().getType(),
                                        object().getType());
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

LogicalResult spirv::CooperativeMatrixMulAddNVOp::verify() {
  return verifyCoopMatrixMulAdd(*this);
}

//===----------------------------------------------------------------------===//
// spv.MatrixTimesScalar
//===----------------------------------------------------------------------===//

LogicalResult spirv::MatrixTimesScalarOp::verify() {
  // We already checked that result and matrix are both of matrix type in the
  // auto-generated verify method.

  auto inputMatrix = matrix().getType().cast<spirv::MatrixType>();
  auto resultMatrix = result().getType().cast<spirv::MatrixType>();

  // Check that the scalar type is the same as the matrix element type.
  if (scalar().getType() != inputMatrix.getElementType())
    return emitError("input matrix components' type and scaling value must "
                     "have the same type");

  // Note that the next three checks could be done using the AllTypesMatch
  // trait in the Op definition file but it generates a vague error message.

  // Check that the input and result matrices have the same columns' count
  if (inputMatrix.getNumColumns() != resultMatrix.getNumColumns())
    return emitError("input and result matrices must have the same "
                     "number of columns");

  // Check that the input and result matrices' have the same rows count
  if (inputMatrix.getNumRows() != resultMatrix.getNumRows())
    return emitError("input and result matrices' columns must have "
                     "the same size");

  // Check that the input and result matrices' have the same component type
  if (inputMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("input and result matrices' columns must have "
                     "the same component type");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.CopyMemory
//===----------------------------------------------------------------------===//

void spirv::CopyMemoryOp::print(OpAsmPrinter &printer) {
  printer << ' ';

  StringRef targetStorageClass = stringifyStorageClass(
      target().getType().cast<spirv::PointerType>().getStorageClass());
  printer << " \"" << targetStorageClass << "\" " << target() << ", ";

  StringRef sourceStorageClass = stringifyStorageClass(
      source().getType().cast<spirv::PointerType>().getStorageClass());
  printer << " \"" << sourceStorageClass << "\" " << source();

  SmallVector<StringRef, 4> elidedAttrs;
  printMemoryAccessAttribute(*this, printer, elidedAttrs);
  printSourceMemoryAccessAttribute(*this, printer, elidedAttrs,
                                   source_memory_access(), source_alignment());

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  Type pointeeType =
      target().getType().cast<spirv::PointerType>().getPointeeType();
  printer << " : " << pointeeType;
}

ParseResult spirv::CopyMemoryOp::parse(OpAsmParser &parser,
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

LogicalResult spirv::CopyMemoryOp::verify() {
  Type targetType =
      target().getType().cast<spirv::PointerType>().getPointeeType();

  Type sourceType =
      source().getType().cast<spirv::PointerType>().getPointeeType();

  if (targetType != sourceType)
    return emitOpError("both operands must be pointers to the same type");

  if (failed(verifyMemoryAccessAttribute(*this)))
    return failure();

  // TODO - According to the spec:
  //
  // If two masks are present, the first applies to Target and cannot include
  // MakePointerVisible, and the second applies to Source and cannot include
  // MakePointerAvailable.
  //
  // Add such verification here.

  return verifySourceMemoryAccessAttribute(*this);
}

//===----------------------------------------------------------------------===//
// spv.Transpose
//===----------------------------------------------------------------------===//

LogicalResult spirv::TransposeOp::verify() {
  auto inputMatrix = matrix().getType().cast<spirv::MatrixType>();
  auto resultMatrix = result().getType().cast<spirv::MatrixType>();

  // Verify that the input and output matrices have correct shapes.
  if (inputMatrix.getNumRows() != resultMatrix.getNumColumns())
    return emitError("input matrix rows count must be equal to "
                     "output matrix columns count");

  if (inputMatrix.getNumColumns() != resultMatrix.getNumRows())
    return emitError("input matrix columns count must be equal to "
                     "output matrix rows count");

  // Verify that the input and output matrices have the same component type
  if (inputMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("input and output matrices must have the same "
                     "component type");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.MatrixTimesMatrix
//===----------------------------------------------------------------------===//

LogicalResult spirv::MatrixTimesMatrixOp::verify() {
  auto leftMatrix = leftmatrix().getType().cast<spirv::MatrixType>();
  auto rightMatrix = rightmatrix().getType().cast<spirv::MatrixType>();
  auto resultMatrix = result().getType().cast<spirv::MatrixType>();

  // left matrix columns' count and right matrix rows' count must be equal
  if (leftMatrix.getNumColumns() != rightMatrix.getNumRows())
    return emitError("left matrix columns' count must be equal to "
                     "the right matrix rows' count");

  // right and result matrices columns' count must be the same
  if (rightMatrix.getNumColumns() != resultMatrix.getNumColumns())
    return emitError(
        "right and result matrices must have equal columns' count");

  // right and result matrices component type must be the same
  if (rightMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("right and result matrices' component type must"
                     " be the same");

  // left and result matrices component type must be the same
  if (leftMatrix.getElementType() != resultMatrix.getElementType())
    return emitError("left and result matrices' component type"
                     " must be the same");

  // left and result matrices rows count must be the same
  if (leftMatrix.getNumRows() != resultMatrix.getNumRows())
    return emitError("left and result matrices must have equal rows' count");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantComposite
//===----------------------------------------------------------------------===//

ParseResult spirv::SpecConstantCompositeOp::parse(OpAsmParser &parser,
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

void spirv::SpecConstantCompositeOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(sym_name());
  printer << " (";
  auto constituents = this->constituents().getValue();

  if (!constituents.empty())
    llvm::interleaveComma(constituents, printer);

  printer << ") : " << type();
}

LogicalResult spirv::SpecConstantCompositeOp::verify() {
  auto cType = type().dyn_cast<spirv::CompositeType>();
  auto constituents = this->constituents().getValue();

  if (!cType)
    return emitError("result type must be a composite type, but provided ")
           << type();

  if (cType.isa<spirv::CooperativeMatrixNVType>())
    return emitError("unsupported composite type  ") << cType;
  if (constituents.size() != cType.getNumElements())
    return emitError("has incorrect number of operands: expected ")
           << cType.getNumElements() << ", but provided "
           << constituents.size();

  for (auto index : llvm::seq<uint32_t>(0, constituents.size())) {
    auto constituent = constituents[index].cast<FlatSymbolRefAttr>();

    auto constituentSpecConstOp =
        dyn_cast<spirv::SpecConstantOp>(SymbolTable::lookupNearestSymbolFrom(
            (*this)->getParentOp(), constituent.getAttr()));

    if (constituentSpecConstOp.default_value().getType() !=
        cType.getElementType(index))
      return emitError("has incorrect types of operands: expected ")
             << cType.getElementType(index) << ", but provided "
             << constituentSpecConstOp.default_value().getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spv.SpecConstantOperation
//===----------------------------------------------------------------------===//

ParseResult spirv::SpecConstantOperationOp::parse(OpAsmParser &parser,
                                                  OperationState &state) {
  Region *body = state.addRegion();

  if (parser.parseKeyword("wraps"))
    return failure();

  body->push_back(new Block);
  Block &block = body->back();
  Operation *wrappedOp = parser.parseGenericOperation(&block, block.begin());

  if (!wrappedOp)
    return failure();

  OpBuilder builder(parser.getContext());
  builder.setInsertionPointToEnd(&block);
  builder.create<spirv::YieldOp>(wrappedOp->getLoc(), wrappedOp->getResult(0));
  state.location = wrappedOp->getLoc();

  state.addTypes(wrappedOp->getResult(0).getType());

  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

void spirv::SpecConstantOperationOp::print(OpAsmPrinter &printer) {
  printer << " wraps ";
  printer.printGenericOp(&body().front().front());
}

LogicalResult spirv::SpecConstantOperationOp::verify() {
  Block &block = getRegion().getBlocks().front();

  if (block.getOperations().size() != 2)
    return emitOpError("expected exactly 2 nested ops");

  Operation &enclosedOp = block.getOperations().front();

  if (!enclosedOp.hasTrait<OpTrait::spirv::UsableInSpecConstantOp>())
    return emitOpError("invalid enclosed op");

  for (auto operand : enclosedOp.getOperands())
    if (!isa<spirv::ConstantOp, spirv::ReferenceOfOp,
             spirv::SpecConstantOperationOp>(operand.getDefiningOp()))
      return emitOpError(
          "invalid operand, must be defined by a constant operation");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.GLSL.FrexpStruct
//===----------------------------------------------------------------------===//

LogicalResult spirv::GLSLFrexpStructOp::verify() {
  spirv::StructType structTy = result().getType().dyn_cast<spirv::StructType>();

  if (structTy.getNumElements() != 2)
    return emitError("result type must be a struct type with two memebers");

  Type significandTy = structTy.getElementType(0);
  Type exponentTy = structTy.getElementType(1);
  VectorType exponentVecTy = exponentTy.dyn_cast<VectorType>();
  IntegerType exponentIntTy = exponentTy.dyn_cast<IntegerType>();

  Type operandTy = operand().getType();
  VectorType operandVecTy = operandTy.dyn_cast<VectorType>();
  FloatType operandFTy = operandTy.dyn_cast<FloatType>();

  if (significandTy != operandTy)
    return emitError("member zero of the resulting struct type must be the "
                     "same type as the operand");

  if (exponentVecTy) {
    IntegerType componentIntTy =
        exponentVecTy.getElementType().dyn_cast<IntegerType>();
    if (!(componentIntTy && componentIntTy.getWidth() == 32))
      return emitError("member one of the resulting struct type must"
                       "be a scalar or vector of 32 bit integer type");
  } else if (!(exponentIntTy && exponentIntTy.getWidth() == 32)) {
    return emitError("member one of the resulting struct type "
                     "must be a scalar or vector of 32 bit integer type");
  }

  // Check that the two member types have the same number of components
  if (operandVecTy && exponentVecTy &&
      (exponentVecTy.getNumElements() == operandVecTy.getNumElements()))
    return success();

  if (operandFTy && exponentIntTy)
    return success();

  return emitError("member one of the resulting struct type must have the same "
                   "number of components as the operand type");
}

//===----------------------------------------------------------------------===//
// spv.GLSL.Ldexp
//===----------------------------------------------------------------------===//

LogicalResult spirv::GLSLLdexpOp::verify() {
  Type significandType = x().getType();
  Type exponentType = exp().getType();

  if (significandType.isa<FloatType>() != exponentType.isa<IntegerType>())
    return emitOpError("operands must both be scalars or vectors");

  auto getNumElements = [](Type type) -> unsigned {
    if (auto vectorType = type.dyn_cast<VectorType>())
      return vectorType.getNumElements();
    return 1;
  };

  if (getNumElements(significandType) != getNumElements(exponentType))
    return emitOpError("operands must have the same number of elements");

  return success();
}

//===----------------------------------------------------------------------===//
// spv.ImageDrefGather
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageDrefGatherOp::verify() {
  VectorType resultType = result().getType().cast<VectorType>();
  auto sampledImageType =
      sampledimage().getType().cast<spirv::SampledImageType>();
  auto imageType = sampledImageType.getImageType().cast<spirv::ImageType>();

  if (resultType.getNumElements() != 4)
    return emitOpError("result type must be a vector of four components");

  Type elementType = resultType.getElementType();
  Type sampledElementType = imageType.getElementType();
  if (!sampledElementType.isa<NoneType>() && elementType != sampledElementType)
    return emitOpError(
        "the component type of result must be the same as sampled type of the "
        "underlying image type");

  spirv::Dim imageDim = imageType.getDim();
  spirv::ImageSamplingInfo imageMS = imageType.getSamplingInfo();

  if (imageDim != spirv::Dim::Dim2D && imageDim != spirv::Dim::Cube &&
      imageDim != spirv::Dim::Rect)
    return emitOpError(
        "the Dim operand of the underlying image type must be 2D, Cube, or "
        "Rect");

  if (imageMS != spirv::ImageSamplingInfo::SingleSampled)
    return emitOpError("the MS operand of the underlying image type must be 0");

  spirv::ImageOperandsAttr attr = imageoperandsAttr();
  auto operandArguments = operand_arguments();

  return verifyImageOperands(*this, attr, operandArguments);
}

//===----------------------------------------------------------------------===//
// spv.ShiftLeftLogicalOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ShiftLeftLogicalOp::verify() {
  return verifyShiftOp(*this);
}

//===----------------------------------------------------------------------===//
// spv.ShiftRightArithmeticOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ShiftRightArithmeticOp::verify() {
  return verifyShiftOp(*this);
}

//===----------------------------------------------------------------------===//
// spv.ShiftRightLogicalOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ShiftRightLogicalOp::verify() {
  return verifyShiftOp(*this);
}

//===----------------------------------------------------------------------===//
// spv.ImageQuerySize
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageQuerySizeOp::verify() {
  spirv::ImageType imageType = image().getType().cast<spirv::ImageType>();
  Type resultType = result().getType();

  spirv::Dim dim = imageType.getDim();
  spirv::ImageSamplingInfo samplingInfo = imageType.getSamplingInfo();
  spirv::ImageSamplerUseInfo samplerInfo = imageType.getSamplerUseInfo();
  switch (dim) {
  case spirv::Dim::Dim1D:
  case spirv::Dim::Dim2D:
  case spirv::Dim::Dim3D:
  case spirv::Dim::Cube:
    if (!(samplingInfo == spirv::ImageSamplingInfo::MultiSampled ||
          samplerInfo == spirv::ImageSamplerUseInfo::SamplerUnknown ||
          samplerInfo == spirv::ImageSamplerUseInfo::NoSampler))
      return emitError(
          "if Dim is 1D, 2D, 3D, or Cube, "
          "it must also have either an MS of 1 or a Sampled of 0 or 2");
    break;
  case spirv::Dim::Buffer:
  case spirv::Dim::Rect:
    break;
  default:
    return emitError("the Dim operand of the image type must "
                     "be 1D, 2D, 3D, Buffer, Cube, or Rect");
  }

  unsigned componentNumber = 0;
  switch (dim) {
  case spirv::Dim::Dim1D:
  case spirv::Dim::Buffer:
    componentNumber = 1;
    break;
  case spirv::Dim::Dim2D:
  case spirv::Dim::Cube:
  case spirv::Dim::Rect:
    componentNumber = 2;
    break;
  case spirv::Dim::Dim3D:
    componentNumber = 3;
    break;
  default:
    break;
  }

  if (imageType.getArrayedInfo() == spirv::ImageArrayedInfo::Arrayed)
    componentNumber += 1;

  unsigned resultComponentNumber = 1;
  if (auto resultVectorType = resultType.dyn_cast<VectorType>())
    resultComponentNumber = resultVectorType.getNumElements();

  if (componentNumber != resultComponentNumber)
    return emitError("expected the result to have ")
           << componentNumber << " component(s), but found "
           << resultComponentNumber << " component(s)";

  return success();
}

static ParseResult parsePtrAccessChainOpImpl(StringRef opName,
                                             OpAsmParser &parser,
                                             OperationState &state) {
  OpAsmParser::OperandType ptrInfo;
  SmallVector<OpAsmParser::OperandType, 4> indicesInfo;
  Type type;
  auto loc = parser.getCurrentLocation();
  SmallVector<Type, 4> indicesTypes;

  if (parser.parseOperand(ptrInfo) ||
      parser.parseOperandList(indicesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptrInfo, type, state.operands))
    return failure();

  // Check that the provided indices list is not empty before parsing their
  // type list.
  if (indicesInfo.empty())
    return emitError(state.location) << opName << " expected element";

  if (parser.parseComma() || parser.parseTypeList(indicesTypes))
    return failure();

  // Check that the indices types list is not empty and that it has a one-to-one
  // mapping to the provided indices.
  if (indicesTypes.size() != indicesInfo.size())
    return emitError(state.location)
           << opName
           << " indices types' count must be equal to indices info count";

  if (parser.resolveOperands(indicesInfo, indicesTypes, loc, state.operands))
    return failure();

  auto resultType = getElementPtrType(
      type, llvm::makeArrayRef(state.operands).drop_front(2), state.location);
  if (!resultType)
    return failure();

  state.addTypes(resultType);
  return success();
}

template <typename Op>
static auto concatElemAndIndices(Op op) {
  SmallVector<Value> ret(op.indices().size() + 1);
  ret[0] = op.element();
  llvm::copy(op.indices(), ret.begin() + 1);
  return ret;
}

//===----------------------------------------------------------------------===//
// spv.InBoundsPtrAccessChainOp
//===----------------------------------------------------------------------===//

void spirv::InBoundsPtrAccessChainOp::build(OpBuilder &builder,
                                            OperationState &state,
                                            Value basePtr, Value element,
                                            ValueRange indices) {
  auto type = getElementPtrType(basePtr.getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, element, indices);
}

ParseResult spirv::InBoundsPtrAccessChainOp::parse(OpAsmParser &parser,
                                                   OperationState &state) {
  return parsePtrAccessChainOpImpl(
      spirv::InBoundsPtrAccessChainOp::getOperationName(), parser, state);
}

void spirv::InBoundsPtrAccessChainOp::print(OpAsmPrinter &printer) {
  printAccessChain(*this, concatElemAndIndices(*this), printer);
}

LogicalResult spirv::InBoundsPtrAccessChainOp::verify() {
  return verifyAccessChain(*this, indices());
}

//===----------------------------------------------------------------------===//
// spv.PtrAccessChainOp
//===----------------------------------------------------------------------===//

void spirv::PtrAccessChainOp::build(OpBuilder &builder, OperationState &state,
                                    Value basePtr, Value element,
                                    ValueRange indices) {
  auto type = getElementPtrType(basePtr.getType(), indices, state.location);
  assert(type && "Unable to deduce return type based on basePtr and indices");
  build(builder, state, type, basePtr, element, indices);
}

ParseResult spirv::PtrAccessChainOp::parse(OpAsmParser &parser,
                                           OperationState &state) {
  return parsePtrAccessChainOpImpl(spirv::PtrAccessChainOp::getOperationName(),
                                   parser, state);
}

void spirv::PtrAccessChainOp::print(OpAsmPrinter &printer) {
  printAccessChain(*this, concatElemAndIndices(*this), printer);
}

LogicalResult spirv::PtrAccessChainOp::verify() {
  return verifyAccessChain(*this, indices());
}

//===----------------------------------------------------------------------===//
// spv.VectorTimesScalarOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::VectorTimesScalarOp::verify() {
  if (vector().getType() != getType())
    return emitOpError("vector operand and result type mismatch");
  auto scalarType = getType().cast<VectorType>().getElementType();
  if (scalar().getType() != scalarType)
    return emitOpError("scalar operand and result element type match");
  return success();
}

// TableGen'erated operation interfaces for querying versions, extensions, and
// capabilities.
#include "mlir/Dialect/SPIRV/IR/SPIRVAvailability.cpp.inc"

// TablenGen'erated operation definitions.
#define GET_OP_CLASSES
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.cpp.inc"

namespace mlir {
namespace spirv {
// TableGen'erated operation availability interface implementations.
#include "mlir/Dialect/SPIRV/IR/SPIRVOpAvailabilityImpl.inc"
} // namespace spirv
} // namespace mlir
