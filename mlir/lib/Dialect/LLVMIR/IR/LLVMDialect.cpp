//===- LLVMDialect.cpp - LLVM IR Ops and Dialect registration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the LLVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "TypeDetail.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SourceMgr.h"

#include <numeric>

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::cconv::getMaxEnumValForCConv;
using mlir::LLVM::linkage::getMaxEnumValForLinkage;

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.cpp.inc"

static constexpr const char kVolatileAttrName[] = "volatile_";
static constexpr const char kNonTemporalAttrName[] = "nontemporal";
static constexpr const char kElemTypeAttrName[] = "elem_type";

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"
#include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"

static auto processFMFAttr(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
        if (attr.getName() == "fastmathFlags") {
          auto defAttr = FMFAttr::get(attr.getValue().getContext(), {});
          return defAttr != attr.getValue();
        }
        return true;
      }));
  return filteredAttrs;
}

static ParseResult parseLLVMOpAttrs(OpAsmParser &parser,
                                    NamedAttrList &result) {
  return parser.parseOptionalAttrDict(result);
}

static void printLLVMOpAttrs(OpAsmPrinter &printer, Operation *op,
                             DictionaryAttr attrs) {
  printer.printOptionalAttrDict(processFMFAttr(attrs.getValue()));
}

/// Verifies `symbol`'s use in `op` to ensure the symbol is a valid and
/// fully defined llvm.func.
static LogicalResult verifySymbolAttrUse(FlatSymbolRefAttr symbol,
                                         Operation *op,
                                         SymbolTableCollection &symbolTable) {
  StringRef name = symbol.getValue();
  auto func =
      symbolTable.lookupNearestSymbolFrom<LLVMFuncOp>(op, symbol.getAttr());
  if (!func)
    return op->emitOpError("'")
           << name << "' does not reference a valid LLVM function";
  if (func.isExternal())
    return op->emitOpError("'") << name << "' does not have a definition";
  return success();
}

//===----------------------------------------------------------------------===//
// Printing, parsing and builder for LLVM::CmpOp.
//===----------------------------------------------------------------------===//

void ICmpOp::build(OpBuilder &builder, OperationState &result,
                   ICmpPredicate predicate, Value lhs, Value rhs) {
  auto boolType = IntegerType::get(lhs.getType().getContext(), 1);
  if (LLVM::isCompatibleVectorType(lhs.getType()) ||
      LLVM::isCompatibleVectorType(rhs.getType())) {
    int64_t numLHSElements = 1, numRHSElements = 1;
    if (LLVM::isCompatibleVectorType(lhs.getType()))
      numLHSElements =
          LLVM::getVectorNumElements(lhs.getType()).getFixedValue();
    if (LLVM::isCompatibleVectorType(rhs.getType()))
      numRHSElements =
          LLVM::getVectorNumElements(rhs.getType()).getFixedValue();
    build(builder, result,
          VectorType::get({std::max(numLHSElements, numRHSElements)}, boolType),
          predicate, lhs, rhs);
  } else {
    build(builder, result, boolType, predicate, lhs, rhs);
  }
}

void ICmpOp::print(OpAsmPrinter &p) {
  p << " \"" << stringifyICmpPredicate(getPredicate()) << "\" " << getOperand(0)
    << ", " << getOperand(1);
  p.printOptionalAttrDict((*this)->getAttrs(), {"predicate"});
  p << " : " << getLhs().getType();
}

void FCmpOp::print(OpAsmPrinter &p) {
  p << " \"" << stringifyFCmpPredicate(getPredicate()) << "\" " << getOperand(0)
    << ", " << getOperand(1);
  p.printOptionalAttrDict(processFMFAttr((*this)->getAttrs()), {"predicate"});
  p << " : " << getLhs().getType();
}

// <operation> ::= `llvm.icmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
// <operation> ::= `llvm.fcmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
template <typename CmpPredicateType>
static ParseResult parseCmpOp(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();

  StringAttr predicateAttr;
  OpAsmParser::UnresolvedOperand lhs, rhs;
  Type type;
  SMLoc predicateLoc, trailingTypeLoc;
  if (parser.getCurrentLocation(&predicateLoc) ||
      parser.parseAttribute(predicateAttr, "predicate", result.attributes) ||
      parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type) ||
      parser.resolveOperand(lhs, type, result.operands) ||
      parser.resolveOperand(rhs, type, result.operands))
    return failure();

  // Replace the string attribute `predicate` with an integer attribute.
  int64_t predicateValue = 0;
  if (std::is_same<CmpPredicateType, ICmpPredicate>()) {
    Optional<ICmpPredicate> predicate =
        symbolizeICmpPredicate(predicateAttr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateAttr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(predicate.getValue());
  } else {
    Optional<FCmpPredicate> predicate =
        symbolizeFCmpPredicate(predicateAttr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateAttr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(predicate.getValue());
  }

  result.attributes.set("predicate",
                        parser.getBuilder().getI64IntegerAttr(predicateValue));

  // The result type is either i1 or a vector type <? x i1> if the inputs are
  // vectors.
  Type resultType = IntegerType::get(builder.getContext(), 1);
  if (!isCompatibleType(type))
    return parser.emitError(trailingTypeLoc,
                            "expected LLVM dialect-compatible type");
  if (LLVM::isCompatibleVectorType(type)) {
    if (LLVM::isScalableVectorType(type)) {
      resultType = LLVM::getVectorType(
          resultType, LLVM::getVectorNumElements(type).getKnownMinValue(),
          /*isScalable=*/true);
    } else {
      resultType = LLVM::getVectorType(
          resultType, LLVM::getVectorNumElements(type).getFixedValue(),
          /*isScalable=*/false);
    }
  }

  result.addTypes({resultType});
  return success();
}

ParseResult ICmpOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseCmpOp<ICmpPredicate>(parser, result);
}

ParseResult FCmpOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseCmpOp<FCmpPredicate>(parser, result);
}

//===----------------------------------------------------------------------===//
// Printing, parsing and verification for LLVM::AllocaOp.
//===----------------------------------------------------------------------===//

void AllocaOp::print(OpAsmPrinter &p) {
  Type elemTy = getType().cast<LLVM::LLVMPointerType>().getElementType();
  if (!elemTy)
    elemTy = *getElemType();

  auto funcTy =
      FunctionType::get(getContext(), {getArraySize().getType()}, {getType()});

  p << ' ' << getArraySize() << " x " << elemTy;
  if (getAlignment().hasValue() && *getAlignment() != 0)
    p.printOptionalAttrDict((*this)->getAttrs(), {kElemTypeAttrName});
  else
    p.printOptionalAttrDict((*this)->getAttrs(),
                            {"alignment", kElemTypeAttrName});
  p << " : " << funcTy;
}

// <operation> ::= `llvm.alloca` ssa-use `x` type attribute-dict?
//                 `:` type `,` type
ParseResult AllocaOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand arraySize;
  Type type, elemType;
  SMLoc trailingTypeLoc;
  if (parser.parseOperand(arraySize) || parser.parseKeyword("x") ||
      parser.parseType(elemType) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  Optional<NamedAttribute> alignmentAttr =
      result.attributes.getNamed("alignment");
  if (alignmentAttr.hasValue()) {
    auto alignmentInt =
        alignmentAttr.getValue().getValue().dyn_cast<IntegerAttr>();
    if (!alignmentInt)
      return parser.emitError(parser.getNameLoc(),
                              "expected integer alignment");
    if (alignmentInt.getValue().isNullValue())
      result.attributes.erase("alignment");
  }

  // Extract the result type from the trailing function type.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumInputs() != 1 ||
      funcType.getNumResults() != 1)
    return parser.emitError(
        trailingTypeLoc,
        "expected trailing function type with one argument and one result");

  if (parser.resolveOperand(arraySize, funcType.getInput(0), result.operands))
    return failure();

  Type resultType = funcType.getResult(0);
  if (auto ptrResultType = resultType.dyn_cast<LLVMPointerType>()) {
    if (ptrResultType.isOpaque())
      result.addAttribute(kElemTypeAttrName, TypeAttr::get(elemType));
  }

  result.addTypes({funcType.getResult(0)});
  return success();
}

/// Checks that the elemental type is present in either the pointer type or
/// the attribute, but not both.
static LogicalResult verifyOpaquePtr(Operation *op, LLVMPointerType ptrType,
                                     Optional<Type> ptrElementType) {
  if (ptrType.isOpaque() && !ptrElementType.hasValue()) {
    return op->emitOpError() << "expected '" << kElemTypeAttrName
                             << "' attribute if opaque pointer type is used";
  }
  if (!ptrType.isOpaque() && ptrElementType.hasValue()) {
    return op->emitOpError()
           << "unexpected '" << kElemTypeAttrName
           << "' attribute when non-opaque pointer type is used";
  }
  return success();
}

LogicalResult AllocaOp::verify() {
  return verifyOpaquePtr(getOperation(), getType().cast<LLVMPointerType>(),
                         getElemType());
}

//===----------------------------------------------------------------------===//
// LLVM::BrOp
//===----------------------------------------------------------------------===//

SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// LLVM::CondBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands CondBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getTrueDestOperandsMutable()
                                      : getFalseDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// LLVM::SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<int32_t> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  ElementsAttr caseValuesAttr;
  if (!caseValues.empty())
    caseValuesAttr = builder.getI32VectorAttr(caseValues);

  ElementsAttr weightsAttr;
  if (!branchWeights.empty())
    weightsAttr = builder.getI32VectorAttr(llvm::to_vector<4>(branchWeights));

  build(builder, result, value, defaultOperands, caseOperands, caseValuesAttr,
        weightsAttr, defaultDestination, caseDestinations);
}

/// <cases> ::= integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
///             ( `,` integer `:` bb-id (`(` ssa-use-and-type-list `)`)? )?
static ParseResult parseSwitchOpCases(
    OpAsmParser &parser, Type flagType, ElementsAttr &caseValues,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &caseOperands,
    SmallVectorImpl<SmallVector<Type>> &caseOperandTypes) {
  SmallVector<APInt> values;
  unsigned bitWidth = flagType.getIntOrFloatBitWidth();
  do {
    int64_t value = 0;
    OptionalParseResult integerParseResult = parser.parseOptionalInteger(value);
    if (values.empty() && !integerParseResult.hasValue())
      return success();

    if (!integerParseResult.hasValue() || integerParseResult.getValue())
      return failure();
    values.push_back(APInt(bitWidth, value));

    Block *destination;
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    SmallVector<Type> operandTypes;
    if (parser.parseColon() || parser.parseSuccessor(destination))
      return failure();
    if (!parser.parseOptionalLParen()) {
      if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None,
                                  /*allowResultNumber=*/false) ||
          parser.parseColonTypeList(operandTypes) || parser.parseRParen())
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.emplace_back(operands);
    caseOperandTypes.emplace_back(operandTypes);
  } while (!parser.parseOptionalComma());

  ShapedType caseValueType =
      VectorType::get(static_cast<int64_t>(values.size()), flagType);
  caseValues = DenseIntElementsAttr::get(caseValueType, values);
  return success();
}

static void printSwitchOpCases(OpAsmPrinter &p, SwitchOp op, Type flagType,
                               ElementsAttr caseValues,
                               SuccessorRange caseDestinations,
                               OperandRangeRange caseOperands,
                               const TypeRangeRange &caseOperandTypes) {
  if (!caseValues)
    return;

  size_t index = 0;
  llvm::interleave(
      llvm::zip(caseValues.cast<DenseIntElementsAttr>(), caseDestinations),
      [&](auto i) {
        p << "  ";
        p << std::get<0>(i).getLimitedValue();
        p << ": ";
        p.printSuccessorAndUseList(std::get<1>(i), caseOperands[index++]);
      },
      [&] {
        p << ',';
        p.printNewline();
      });
  p.printNewline();
}

LogicalResult SwitchOp::verify() {
  if ((!getCaseValues() && !getCaseDestinations().empty()) ||
      (getCaseValues() &&
       getCaseValues()->size() !=
           static_cast<int64_t>(getCaseDestinations().size())))
    return emitOpError("expects number of case values to match number of "
                       "case destinations");
  if (getBranchWeights() && getBranchWeights()->size() != getNumSuccessors())
    return emitError("expects number of branch weights to match number of "
                     "successors: ")
           << getBranchWeights()->size() << " vs " << getNumSuccessors();
  return success();
}

SuccessorOperands SwitchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDefaultOperandsMutable()
                                      : getCaseOperandsMutable(index - 1));
}

//===----------------------------------------------------------------------===//
// Code for LLVM::GEPOp.
//===----------------------------------------------------------------------===//

constexpr int GEPOp::kDynamicIndex;

namespace {
/// Base class for llvm::Error related to GEP index.
class GEPIndexError : public llvm::ErrorInfo<GEPIndexError> {
protected:
  unsigned indexPos;

public:
  static char ID;

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  explicit GEPIndexError(unsigned pos) : indexPos(pos) {}
};

/// llvm::Error for out-of-bound GEP index.
struct GEPIndexOutOfBoundError
    : public llvm::ErrorInfo<GEPIndexOutOfBoundError, GEPIndexError> {
  static char ID;

  using ErrorInfo::ErrorInfo;

  void log(llvm::raw_ostream &os) const override {
    os << "index " << indexPos << " indexing a struct is out of bounds";
  }
};

/// llvm::Error for non-static GEP index indexing a struct.
struct GEPStaticIndexError
    : public llvm::ErrorInfo<GEPStaticIndexError, GEPIndexError> {
  static char ID;

  using ErrorInfo::ErrorInfo;

  void log(llvm::raw_ostream &os) const override {
    os << "expected index " << indexPos << " indexing a struct "
       << "to be constant";
  }
};
} // end anonymous namespace

char GEPIndexError::ID = 0;
char GEPIndexOutOfBoundError::ID = 0;
char GEPStaticIndexError::ID = 0;

/// For the given `structIndices` and `indices`, check if they're complied
/// with `baseGEPType`, especially check against LLVMStructTypes nested within,
/// and refine/promote struct index from `indices` to `updatedStructIndices`
/// if the latter argument is not null.
static llvm::Error
recordStructIndices(Type baseGEPType, unsigned indexPos,
                    ArrayRef<int32_t> structIndices, ValueRange indices,
                    SmallVectorImpl<int32_t> *updatedStructIndices,
                    SmallVectorImpl<Value> *remainingIndices) {
  if (indexPos >= structIndices.size())
    // Stop searching
    return llvm::Error::success();

  int32_t gepIndex = structIndices[indexPos];
  bool isStaticIndex = gepIndex != GEPOp::kDynamicIndex;

  unsigned dynamicIndexPos = indexPos;
  if (!isStaticIndex)
    dynamicIndexPos = llvm::count(structIndices.take_front(indexPos + 1),
                                  LLVM::GEPOp::kDynamicIndex) - 1;

  return llvm::TypeSwitch<Type, llvm::Error>(baseGEPType)
      .Case<LLVMStructType>([&](LLVMStructType structType) -> llvm::Error {
        // We don't always want to refine the index (e.g. when performing
        // verification), so we only refine when updatedStructIndices is not
        // null.
        if (!isStaticIndex && updatedStructIndices) {
          // Try to refine.
          APInt staticIndexValue;
          isStaticIndex = matchPattern(indices[dynamicIndexPos],
                                       m_ConstantInt(&staticIndexValue));
          if (isStaticIndex) {
            assert(staticIndexValue.getBitWidth() <= 64 &&
                   llvm::isInt<32>(staticIndexValue.getLimitedValue()) &&
                   "struct index can't fit within int32_t");
            gepIndex = static_cast<int32_t>(staticIndexValue.getSExtValue());
          }
        }
        if (!isStaticIndex)
          return llvm::make_error<GEPStaticIndexError>(indexPos);

        ArrayRef<Type> elementTypes = structType.getBody();
        if (gepIndex < 0 ||
            static_cast<size_t>(gepIndex) >= elementTypes.size())
          return llvm::make_error<GEPIndexOutOfBoundError>(indexPos);

        if (updatedStructIndices)
          (*updatedStructIndices)[indexPos] = gepIndex;

        // Instead of recusively going into every children types, we only
        // dive into the one indexed by gepIndex.
        return recordStructIndices(elementTypes[gepIndex], indexPos + 1,
                                   structIndices, indices, updatedStructIndices,
                                   remainingIndices);
      })
      .Case<VectorType, LLVMScalableVectorType, LLVMFixedVectorType,
            LLVMArrayType>([&](auto containerType) -> llvm::Error {
        // Currently we don't refine non-struct index even if it's static.
        if (remainingIndices)
          remainingIndices->push_back(indices[dynamicIndexPos]);
        return recordStructIndices(containerType.getElementType(), indexPos + 1,
                                   structIndices, indices, updatedStructIndices,
                                   remainingIndices);
      })
      .Default(
          [](auto otherType) -> llvm::Error { return llvm::Error::success(); });
}

/// Driver function around `recordStructIndices`. Note that we always check
/// from the second GEP index since the first one is always dynamic.
static llvm::Error
findStructIndices(Type baseGEPType, ArrayRef<int32_t> structIndices,
                  ValueRange indices,
                  SmallVectorImpl<int32_t> *updatedStructIndices = nullptr,
                  SmallVectorImpl<Value> *remainingIndices = nullptr) {
  if (remainingIndices)
    // The first GEP index is always dynamic.
    remainingIndices->push_back(indices[0]);
  return recordStructIndices(baseGEPType, /*indexPos=*/1, structIndices,
                             indices, updatedStructIndices, remainingIndices);
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Value basePtr, ValueRange operands,
                  ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultType, basePtr, operands,
        SmallVector<int32_t>(operands.size(), kDynamicIndex), attributes);
}

/// Returns the elemental type of any LLVM-compatible vector type or self.
static Type extractVectorElementType(Type type) {
  if (auto vectorType = type.dyn_cast<VectorType>())
    return vectorType.getElementType();
  if (auto scalableVectorType = type.dyn_cast<LLVMScalableVectorType>())
    return scalableVectorType.getElementType();
  if (auto fixedVectorType = type.dyn_cast<LLVMFixedVectorType>())
    return fixedVectorType.getElementType();
  return type;
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Value basePtr, ValueRange indices,
                  ArrayRef<int32_t> structIndices,
                  ArrayRef<NamedAttribute> attributes) {
  auto ptrType =
      extractVectorElementType(basePtr.getType()).cast<LLVMPointerType>();
  assert(!ptrType.isOpaque() &&
         "expected non-opaque pointer, provide elementType explicitly when "
         "opaque pointers are used");
  build(builder, result, resultType, ptrType.getElementType(), basePtr, indices,
        structIndices, attributes);
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Type elementType, Value basePtr, ValueRange indices,
                  ArrayRef<int32_t> structIndices,
                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<Value> remainingIndices;
  SmallVector<int32_t> updatedStructIndices(structIndices.begin(),
                                            structIndices.end());
  if (llvm::Error err =
          findStructIndices(elementType, structIndices, indices,
                            &updatedStructIndices, &remainingIndices))
    llvm::report_fatal_error(StringRef(llvm::toString(std::move(err))));

  assert(remainingIndices.size() == static_cast<size_t>(llvm::count(
                                        updatedStructIndices, kDynamicIndex)) &&
         "expected as many index operands as dynamic index attr elements");

  result.addTypes(resultType);
  result.addAttributes(attributes);
  result.addAttribute("structIndices",
                      builder.getI32TensorAttr(updatedStructIndices));
  if (extractVectorElementType(basePtr.getType())
          .cast<LLVMPointerType>()
          .isOpaque())
    result.addAttribute(kElemTypeAttrName, TypeAttr::get(elementType));
  result.addOperands(basePtr);
  result.addOperands(remainingIndices);
}

static ParseResult
parseGEPIndices(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
                DenseIntElementsAttr &structIndices) {
  SmallVector<int32_t> constantIndices;

  auto idxParser = [&]() -> ParseResult {
    int32_t constantIndex;
    OptionalParseResult parsedInteger =
        parser.parseOptionalInteger(constantIndex);
    if (parsedInteger.hasValue()) {
      if (failed(parsedInteger.getValue()))
        return failure();
      constantIndices.push_back(constantIndex);
      return success();
    }

    constantIndices.push_back(LLVM::GEPOp::kDynamicIndex);
    return parser.parseOperand(indices.emplace_back());
  };
  if (parser.parseCommaSeparatedList(idxParser))
    return failure();

  structIndices = parser.getBuilder().getI32TensorAttr(constantIndices);
  return success();
}

static void printGEPIndices(OpAsmPrinter &printer, LLVM::GEPOp gepOp,
                            OperandRange indices,
                            DenseIntElementsAttr structIndices) {
  unsigned operandIdx = 0;
  llvm::interleaveComma(structIndices.getValues<int32_t>(), printer,
                        [&](int32_t cst) {
                          if (cst == LLVM::GEPOp::kDynamicIndex)
                            printer.printOperand(indices[operandIdx++]);
                          else
                            printer << cst;
                        });
}

LogicalResult LLVM::GEPOp::verify() {
  if (failed(verifyOpaquePtr(
          getOperation(),
          extractVectorElementType(getType()).cast<LLVMPointerType>(),
          getElemType())))
    return failure();

  auto structIndexRange = getStructIndices().getValues<int32_t>();
  // structIndexRange is a kind of iterator, which cannot be converted
  // to ArrayRef directly.
  SmallVector<int32_t> structIndices(structIndexRange.size());
  for (unsigned i : llvm::seq<unsigned>(0, structIndexRange.size()))
    structIndices[i] = structIndexRange[i];
  if (llvm::Error err = findStructIndices(getSourceElementType(), structIndices,
                                          getIndices()))
    return emitOpError() << llvm::toString(std::move(err));

  return success();
}

Type LLVM::GEPOp::getSourceElementType() {
  if (Optional<Type> elemType = getElemType())
    return *elemType;

  return extractVectorElementType(getBase().getType())
      .cast<LLVMPointerType>()
      .getElementType();
}

//===----------------------------------------------------------------------===//
// Builder, printer and parser for for LLVM::LoadOp.
//===----------------------------------------------------------------------===//

LogicalResult verifySymbolAttribute(
    Operation *op, StringRef attributeName,
    llvm::function_ref<LogicalResult(Operation *, SymbolRefAttr)>
        verifySymbolType) {
  if (Attribute attribute = op->getAttr(attributeName)) {
    // The attribute is already verified to be a symbol ref array attribute via
    // a constraint in the operation definition.
    for (SymbolRefAttr symbolRef :
         attribute.cast<ArrayAttr>().getAsRange<SymbolRefAttr>()) {
      StringAttr metadataName = symbolRef.getRootReference();
      StringAttr symbolName = symbolRef.getLeafReference();
      // We want @metadata::@symbol, not just @symbol
      if (metadataName == symbolName) {
        return op->emitOpError() << "expected '" << symbolRef
                                 << "' to specify a fully qualified reference";
      }
      auto metadataOp = SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
          op->getParentOp(), metadataName);
      if (!metadataOp)
        return op->emitOpError()
               << "expected '" << symbolRef << "' to reference a metadata op";
      Operation *symbolOp =
          SymbolTable::lookupNearestSymbolFrom(metadataOp, symbolName);
      if (!symbolOp)
        return op->emitOpError()
               << "expected '" << symbolRef << "' to be a valid reference";
      if (failed(verifySymbolType(symbolOp, symbolRef))) {
        return failure();
      }
    }
  }
  return success();
}

// Verifies that metadata ops are wired up properly.
template <typename OpTy>
static LogicalResult verifyOpMetadata(Operation *op, StringRef attributeName) {
  auto verifySymbolType = [op](Operation *symbolOp,
                               SymbolRefAttr symbolRef) -> LogicalResult {
    if (!isa<OpTy>(symbolOp)) {
      return op->emitOpError()
             << "expected '" << symbolRef << "' to resolve to a "
             << OpTy::getOperationName();
    }
    return success();
  };

  return verifySymbolAttribute(op, attributeName, verifySymbolType);
}

static LogicalResult verifyMemoryOpMetadata(Operation *op) {
  // access_groups
  if (failed(verifyOpMetadata<LLVM::AccessGroupMetadataOp>(
          op, LLVMDialect::getAccessGroupsAttrName())))
    return failure();

  // alias_scopes
  if (failed(verifyOpMetadata<LLVM::AliasScopeMetadataOp>(
          op, LLVMDialect::getAliasScopesAttrName())))
    return failure();

  // noalias_scopes
  if (failed(verifyOpMetadata<LLVM::AliasScopeMetadataOp>(
          op, LLVMDialect::getNoAliasScopesAttrName())))
    return failure();

  return success();
}

LogicalResult LoadOp::verify() { return verifyMemoryOpMetadata(*this); }

void LoadOp::build(OpBuilder &builder, OperationState &result, Type t,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal) {
  result.addOperands(addr);
  result.addTypes(t);
  if (isVolatile)
    result.addAttribute(kVolatileAttrName, builder.getUnitAttr());
  if (isNonTemporal)
    result.addAttribute(kNonTemporalAttrName, builder.getUnitAttr());
  if (alignment != 0)
    result.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
}

void LoadOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getVolatile_())
    p << "volatile ";
  p << getAddr();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {kVolatileAttrName, kElemTypeAttrName});
  p << " : " << getAddr().getType();
  if (getAddr().getType().cast<LLVMPointerType>().isOpaque())
    p << " -> " << getType();
}

// Extract the pointee type from the LLVM pointer type wrapped in MLIR. Return
// the resulting type if any, null type if opaque pointers are used, and None
// if the given type is not the pointer type.
static Optional<Type> getLoadStoreElementType(OpAsmParser &parser, Type type,
                                              SMLoc trailingTypeLoc) {
  auto llvmTy = type.dyn_cast<LLVM::LLVMPointerType>();
  if (!llvmTy) {
    parser.emitError(trailingTypeLoc, "expected LLVM pointer type");
    return llvm::None;
  }
  return llvmTy.getElementType();
}

// <operation> ::= `llvm.load` `volatile` ssa-use attribute-dict? `:` type
//                 (`->` type)?
ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand addr;
  Type type;
  SMLoc trailingTypeLoc;

  if (succeeded(parser.parseOptionalKeyword("volatile")))
    result.addAttribute(kVolatileAttrName, parser.getBuilder().getUnitAttr());

  if (parser.parseOperand(addr) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure();

  Optional<Type> elemTy =
      getLoadStoreElementType(parser, type, trailingTypeLoc);
  if (!elemTy)
    return failure();
  if (*elemTy) {
    result.addTypes(*elemTy);
    return success();
  }

  Type trailingType;
  if (parser.parseArrow() || parser.parseType(trailingType))
    return failure();
  result.addTypes(trailingType);
  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and parser for LLVM::StoreOp.
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() { return verifyMemoryOpMetadata(*this); }

void StoreOp::build(OpBuilder &builder, OperationState &result, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal) {
  result.addOperands({value, addr});
  result.addTypes({});
  if (isVolatile)
    result.addAttribute(kVolatileAttrName, builder.getUnitAttr());
  if (isNonTemporal)
    result.addAttribute(kNonTemporalAttrName, builder.getUnitAttr());
  if (alignment != 0)
    result.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
}

void StoreOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getVolatile_())
    p << "volatile ";
  p << getValue() << ", " << getAddr();
  p.printOptionalAttrDict((*this)->getAttrs(), {kVolatileAttrName});
  p << " : ";
  if (getAddr().getType().cast<LLVMPointerType>().isOpaque())
    p << getValue().getType() << ", ";
  p << getAddr().getType();
}

// <operation> ::= `llvm.store` `volatile` ssa-use `,` ssa-use
//                 attribute-dict? `:` type (`,` type)?
ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand addr, value;
  Type type;
  SMLoc trailingTypeLoc;

  if (succeeded(parser.parseOptionalKeyword("volatile")))
    result.addAttribute(kVolatileAttrName, parser.getBuilder().getUnitAttr());

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(addr) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  Type operandType;
  if (succeeded(parser.parseOptionalComma())) {
    operandType = type;
    if (parser.parseType(type))
      return failure();
  } else {
    Optional<Type> maybeOperandType =
        getLoadStoreElementType(parser, type, trailingTypeLoc);
    if (!maybeOperandType)
      return failure();
    operandType = *maybeOperandType;
  }

  if (parser.resolveOperand(value, operandType, result.operands) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure();

  return success();
}

///===---------------------------------------------------------------------===//
/// LLVM::InvokeOp
///===---------------------------------------------------------------------===//

SuccessorOperands InvokeOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getNormalDestOperandsMutable()
                                      : getUnwindDestOperandsMutable());
}

LogicalResult InvokeOp::verify() {
  if (getNumResults() > 1)
    return emitOpError("must have 0 or 1 result");

  Block *unwindDest = getUnwindDest();
  if (unwindDest->empty())
    return emitError("must have at least one operation in unwind destination");

  // In unwind destination, first operation must be LandingpadOp
  if (!isa<LandingpadOp>(unwindDest->front()))
    return emitError("first operation in unwind destination should be a "
                     "llvm.landingpad operation");

  return success();
}

void InvokeOp::print(OpAsmPrinter &p) {
  auto callee = getCallee();
  bool isDirect = callee.hasValue();

  p << ' ';

  // Either function name or pointer
  if (isDirect)
    p.printSymbolName(callee.getValue());
  else
    p << getOperand(0);

  p << '(' << getOperands().drop_front(isDirect ? 0 : 1) << ')';
  p << " to ";
  p.printSuccessorAndUseList(getNormalDest(), getNormalDestOperands());
  p << " unwind ";
  p.printSuccessorAndUseList(getUnwindDest(), getUnwindDestOperands());

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {InvokeOp::getOperandSegmentSizeAttr(), "callee"});
  p << " : ";
  p.printFunctionalType(llvm::drop_begin(getOperandTypes(), isDirect ? 0 : 1),
                        getResultTypes());
}

/// <operation> ::= `llvm.invoke` (function-id | ssa-use) `(` ssa-use-list `)`
///                  `to` bb-id (`[` ssa-use-and-type-list `]`)?
///                  `unwind` bb-id (`[` ssa-use-and-type-list `]`)?
///                  attribute-dict? `:` function-type
ParseResult InvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  FunctionType funcType;
  SymbolRefAttr funcAttr;
  SMLoc trailingTypeLoc;
  Block *normalDest, *unwindDest;
  SmallVector<Value, 4> normalOperands, unwindOperands;
  Builder &builder = parser.getBuilder();

  // Parse an operand list that will, in practice, contain 0 or 1 operand.  In
  // case of an indirect call, there will be 1 operand before `(`.  In case of a
  // direct call, there will be no operands and the parser will stop at the
  // function identifier without complaining.
  if (parser.parseOperandList(operands))
    return failure();
  bool isDirect = operands.empty();

  // Optionally parse a function identifier.
  if (isDirect && parser.parseAttribute(funcAttr, "callee", result.attributes))
    return failure();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("to") ||
      parser.parseSuccessorAndUseList(normalDest, normalOperands) ||
      parser.parseKeyword("unwind") ||
      parser.parseSuccessorAndUseList(unwindDest, unwindOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(funcType))
    return failure();

  if (isDirect) {
    // Make sure types match.
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();
    result.addTypes(funcType.getResults());
  } else {
    // Construct the LLVM IR Dialect function type that the first operand
    // should match.
    if (funcType.getNumResults() > 1)
      return parser.emitError(trailingTypeLoc,
                              "expected function with 0 or 1 result");

    Type llvmResultType;
    if (funcType.getNumResults() == 0) {
      llvmResultType = LLVM::LLVMVoidType::get(builder.getContext());
    } else {
      llvmResultType = funcType.getResult(0);
      if (!isCompatibleType(llvmResultType))
        return parser.emitError(trailingTypeLoc,
                                "expected result to have LLVM type");
    }

    SmallVector<Type, 8> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    for (Type ty : funcType.getInputs()) {
      if (isCompatibleType(ty))
        argTypes.push_back(ty);
      else
        return parser.emitError(trailingTypeLoc,
                                "expected LLVM types as inputs");
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(llvmResultType, argTypes);
    auto wrappedFuncType = LLVM::LLVMPointerType::get(llvmFuncType);

    auto funcArguments = llvm::makeArrayRef(operands).drop_front();

    // Make sure that the first operand (indirect callee) matches the wrapped
    // LLVM IR function type, and that the types of the other call operands
    // match the types of the function arguments.
    if (parser.resolveOperand(operands[0], wrappedFuncType, result.operands) ||
        parser.resolveOperands(funcArguments, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();

    result.addTypes(llvmResultType);
  }
  result.addSuccessors({normalDest, unwindDest});
  result.addOperands(normalOperands);
  result.addOperands(unwindOperands);

  result.addAttribute(
      InvokeOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(operands.size()),
                                static_cast<int32_t>(normalOperands.size()),
                                static_cast<int32_t>(unwindOperands.size())}));
  return success();
}

///===----------------------------------------------------------------------===//
/// Verifying/Printing/Parsing for LLVM::LandingpadOp.
///===----------------------------------------------------------------------===//

LogicalResult LandingpadOp::verify() {
  Value value;
  if (LLVMFuncOp func = (*this)->getParentOfType<LLVMFuncOp>()) {
    if (!func.getPersonality().hasValue())
      return emitError(
          "llvm.landingpad needs to be in a function with a personality");
  }

  if (!getCleanup() && getOperands().empty())
    return emitError("landingpad instruction expects at least one clause or "
                     "cleanup attribute");

  for (unsigned idx = 0, ie = getNumOperands(); idx < ie; idx++) {
    value = getOperand(idx);
    bool isFilter = value.getType().isa<LLVMArrayType>();
    if (isFilter) {
      // FIXME: Verify filter clauses when arrays are appropriately handled
    } else {
      // catch - global addresses only.
      // Bitcast ops should have global addresses as their args.
      if (auto bcOp = value.getDefiningOp<BitcastOp>()) {
        if (auto addrOp = bcOp.getArg().getDefiningOp<AddressOfOp>())
          continue;
        return emitError("constant clauses expected").attachNote(bcOp.getLoc())
               << "global addresses expected as operand to "
                  "bitcast used in clauses for landingpad";
      }
      // NullOp and AddressOfOp allowed
      if (value.getDefiningOp<NullOp>())
        continue;
      if (value.getDefiningOp<AddressOfOp>())
        continue;
      return emitError("clause #")
             << idx << " is not a known constant - null, addressof, bitcast";
    }
  }
  return success();
}

void LandingpadOp::print(OpAsmPrinter &p) {
  p << (getCleanup() ? " cleanup " : " ");

  // Clauses
  for (auto value : getOperands()) {
    // Similar to llvm - if clause is an array type then it is filter
    // clause else catch clause
    bool isArrayTy = value.getType().isa<LLVMArrayType>();
    p << '(' << (isArrayTy ? "filter " : "catch ") << value << " : "
      << value.getType() << ") ";
  }

  p.printOptionalAttrDict((*this)->getAttrs(), {"cleanup"});

  p << ": " << getType();
}

/// <operation> ::= `llvm.landingpad` `cleanup`?
///                 ((`catch` | `filter`) operand-type ssa-use)* attribute-dict?
ParseResult LandingpadOp::parse(OpAsmParser &parser, OperationState &result) {
  // Check for cleanup
  if (succeeded(parser.parseOptionalKeyword("cleanup")))
    result.addAttribute("cleanup", parser.getBuilder().getUnitAttr());

  // Parse clauses with types
  while (succeeded(parser.parseOptionalLParen()) &&
         (succeeded(parser.parseOptionalKeyword("filter")) ||
          succeeded(parser.parseOptionalKeyword("catch")))) {
    OpAsmParser::UnresolvedOperand operand;
    Type ty;
    if (parser.parseOperand(operand) || parser.parseColon() ||
        parser.parseType(ty) ||
        parser.resolveOperand(operand, ty, result.operands) ||
        parser.parseRParen())
      return failure();
  }

  Type type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  result.addTypes(type);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifying/Printing/parsing for LLVM::CallOp.
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verify() {
  if (getNumResults() > 1)
    return emitOpError("must have 0 or 1 result");

  // Type for the callee, we'll get it differently depending if it is a direct
  // or indirect call.
  Type fnType;

  bool isIndirect = false;

  // If this is an indirect call, the callee attribute is missing.
  FlatSymbolRefAttr calleeName = getCalleeAttr();
  if (!calleeName) {
    isIndirect = true;
    if (!getNumOperands())
      return emitOpError(
          "must have either a `callee` attribute or at least an operand");
    auto ptrType = getOperand(0).getType().dyn_cast<LLVMPointerType>();
    if (!ptrType)
      return emitOpError("indirect call expects a pointer as callee: ")
             << ptrType;
    fnType = ptrType.getElementType();
  } else {
    Operation *callee =
        SymbolTable::lookupNearestSymbolFrom(*this, calleeName.getAttr());
    if (!callee)
      return emitOpError()
             << "'" << calleeName.getValue()
             << "' does not reference a symbol in the current scope";
    auto fn = dyn_cast<LLVMFuncOp>(callee);
    if (!fn)
      return emitOpError() << "'" << calleeName.getValue()
                           << "' does not reference a valid LLVM function";

    fnType = fn.getFunctionType();
  }

  LLVMFunctionType funcType = fnType.dyn_cast<LLVMFunctionType>();
  if (!funcType)
    return emitOpError("callee does not have a functional type: ") << fnType;

  // Verify that the operand and result types match the callee.

  if (!funcType.isVarArg() &&
      funcType.getNumParams() != (getNumOperands() - isIndirect))
    return emitOpError() << "incorrect number of operands ("
                         << (getNumOperands() - isIndirect)
                         << ") for callee (expecting: "
                         << funcType.getNumParams() << ")";

  if (funcType.getNumParams() > (getNumOperands() - isIndirect))
    return emitOpError() << "incorrect number of operands ("
                         << (getNumOperands() - isIndirect)
                         << ") for varargs callee (expecting at least: "
                         << funcType.getNumParams() << ")";

  for (unsigned i = 0, e = funcType.getNumParams(); i != e; ++i)
    if (getOperand(i + isIndirect).getType() != funcType.getParamType(i))
      return emitOpError() << "operand type mismatch for operand " << i << ": "
                           << getOperand(i + isIndirect).getType()
                           << " != " << funcType.getParamType(i);

  if (getNumResults() == 0 &&
      !funcType.getReturnType().isa<LLVM::LLVMVoidType>())
    return emitOpError() << "expected function call to produce a value";

  if (getNumResults() != 0 &&
      funcType.getReturnType().isa<LLVM::LLVMVoidType>())
    return emitOpError()
           << "calling function with void result must not produce values";

  if (getNumResults() > 1)
    return emitOpError()
           << "expected LLVM function call to produce 0 or 1 result";

  if (getNumResults() && getResult(0).getType() != funcType.getReturnType())
    return emitOpError() << "result type mismatch: " << getResult(0).getType()
                         << " != " << funcType.getReturnType();

  return success();
}

void CallOp::print(OpAsmPrinter &p) {
  auto callee = getCallee();
  bool isDirect = callee.hasValue();

  // Print the direct callee if present as a function attribute, or an indirect
  // callee (first operand) otherwise.
  p << ' ';
  if (isDirect)
    p.printSymbolName(callee.getValue());
  else
    p << getOperand(0);

  auto args = getOperands().drop_front(isDirect ? 0 : 1);
  p << '(' << args << ')';
  p.printOptionalAttrDict(processFMFAttr((*this)->getAttrs()), {"callee"});

  // Reconstruct the function MLIR function type from operand and result types.
  p << " : ";
  p.printFunctionalType(args.getTypes(), getResultTypes());
}

// <operation> ::= `llvm.call` (function-id | ssa-use) `(` ssa-use-list `)`
//                 attribute-dict? `:` function-type
ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  Type type;
  SymbolRefAttr funcAttr;
  SMLoc trailingTypeLoc;

  // Parse an operand list that will, in practice, contain 0 or 1 operand.  In
  // case of an indirect call, there will be 1 operand before `(`.  In case of a
  // direct call, there will be no operands and the parser will stop at the
  // function identifier without complaining.
  if (parser.parseOperandList(operands))
    return failure();
  bool isDirect = operands.empty();

  // Optionally parse a function identifier.
  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", result.attributes))
      return failure();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType)
    return parser.emitError(trailingTypeLoc, "expected function type");
  if (funcType.getNumResults() > 1)
    return parser.emitError(trailingTypeLoc,
                            "expected function with 0 or 1 result");
  if (isDirect) {
    // Make sure types match.
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();
    if (funcType.getNumResults() != 0 &&
        !funcType.getResult(0).isa<LLVM::LLVMVoidType>())
      result.addTypes(funcType.getResults());
  } else {
    Builder &builder = parser.getBuilder();
    Type llvmResultType;
    if (funcType.getNumResults() == 0) {
      llvmResultType = LLVM::LLVMVoidType::get(builder.getContext());
    } else {
      llvmResultType = funcType.getResult(0);
      if (!isCompatibleType(llvmResultType))
        return parser.emitError(trailingTypeLoc,
                                "expected result to have LLVM type");
    }

    SmallVector<Type, 8> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    for (int i = 0, e = funcType.getNumInputs(); i < e; ++i) {
      auto argType = funcType.getInput(i);
      if (!isCompatibleType(argType))
        return parser.emitError(trailingTypeLoc,
                                "expected LLVM types as inputs");
      argTypes.push_back(argType);
    }
    auto llvmFuncType = LLVM::LLVMFunctionType::get(llvmResultType, argTypes);
    auto wrappedFuncType = LLVM::LLVMPointerType::get(llvmFuncType);

    auto funcArguments =
        ArrayRef<OpAsmParser::UnresolvedOperand>(operands).drop_front();

    // Make sure that the first operand (indirect callee) matches the wrapped
    // LLVM IR function type, and that the types of the other call operands
    // match the types of the function arguments.
    if (parser.resolveOperand(operands[0], wrappedFuncType, result.operands) ||
        parser.resolveOperands(funcArguments, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();

    if (!llvmResultType.isa<LLVM::LLVMVoidType>())
      result.addTypes(llvmResultType);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractElementOp.
//===----------------------------------------------------------------------===//
// Expects vector to be of wrapped LLVM vector type and position to be of
// wrapped LLVM i32 type.
void LLVM::ExtractElementOp::build(OpBuilder &b, OperationState &result,
                                   Value vector, Value position,
                                   ArrayRef<NamedAttribute> attrs) {
  auto vectorType = vector.getType();
  auto llvmType = LLVM::getVectorElementType(vectorType);
  build(b, result, llvmType, vector, position);
  result.addAttributes(attrs);
}

void ExtractElementOp::print(OpAsmPrinter &p) {
  p << ' ' << getVector() << "[" << getPosition() << " : "
    << getPosition().getType() << "]";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getVector().getType();
}

// <operation> ::= `llvm.extractelement` ssa-use `, ` ssa-use
//                 attribute-dict? `:` type
ParseResult ExtractElementOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  SMLoc loc;
  OpAsmParser::UnresolvedOperand vector, position;
  Type type, positionType;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(vector) ||
      parser.parseLSquare() || parser.parseOperand(position) ||
      parser.parseColonType(positionType) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(vector, type, result.operands) ||
      parser.resolveOperand(position, positionType, result.operands))
    return failure();
  if (!LLVM::isCompatibleVectorType(type))
    return parser.emitError(
        loc, "expected LLVM dialect-compatible vector type for operand #1");
  result.addTypes(LLVM::getVectorElementType(type));
  return success();
}

LogicalResult ExtractElementOp::verify() {
  Type vectorType = getVector().getType();
  if (!LLVM::isCompatibleVectorType(vectorType))
    return emitOpError("expected LLVM dialect-compatible vector type for "
                       "operand #1, got")
           << vectorType;
  Type valueType = LLVM::getVectorElementType(vectorType);
  if (valueType != getRes().getType())
    return emitOpError() << "Type mismatch: extracting from " << vectorType
                         << " should produce " << valueType
                         << " but this op returns " << getRes().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractValueOp.
//===----------------------------------------------------------------------===//

void ExtractValueOp::print(OpAsmPrinter &p) {
  p << ' ' << getContainer() << getPosition();
  p.printOptionalAttrDict((*this)->getAttrs(), {"position"});
  p << " : " << getContainer().getType();
}

// Extract the type at `position` in the wrapped LLVM IR aggregate type
// `containerType`.  Position is an integer array attribute where each value
// is a zero-based position of the element in the aggregate type.  Return the
// resulting type wrapped in MLIR, or nullptr on error.
static Type getInsertExtractValueElementType(OpAsmParser &parser,
                                             Type containerType,
                                             ArrayAttr positionAttr,
                                             SMLoc attributeLoc,
                                             SMLoc typeLoc) {
  Type llvmType = containerType;
  if (!isCompatibleType(containerType))
    return parser.emitError(typeLoc, "expected LLVM IR Dialect type"), nullptr;

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // structures.  Check the position index before accessing, it is supposed to
  // be in bounds.
  for (Attribute subAttr : positionAttr) {
    auto positionElementAttr = subAttr.dyn_cast<IntegerAttr>();
    if (!positionElementAttr)
      return parser.emitError(attributeLoc,
                              "expected an array of integer literals"),
             nullptr;
    int position = positionElementAttr.getInt();
    if (auto arrayType = llvmType.dyn_cast<LLVMArrayType>()) {
      if (position < 0 ||
          static_cast<unsigned>(position) >= arrayType.getNumElements())
        return parser.emitError(attributeLoc, "position out of bounds"),
               nullptr;
      llvmType = arrayType.getElementType();
    } else if (auto structType = llvmType.dyn_cast<LLVMStructType>()) {
      if (position < 0 ||
          static_cast<unsigned>(position) >= structType.getBody().size())
        return parser.emitError(attributeLoc, "position out of bounds"),
               nullptr;
      llvmType = structType.getBody()[position];
    } else {
      return parser.emitError(typeLoc, "expected LLVM IR structure/array type"),
             nullptr;
    }
  }
  return llvmType;
}

// Extract the type at `position` in the wrapped LLVM IR aggregate type
// `containerType`. Returns null on failure.
static Type getInsertExtractValueElementType(Type containerType,
                                             ArrayAttr positionAttr,
                                             Operation *op) {
  Type llvmType = containerType;
  if (!isCompatibleType(containerType)) {
    op->emitError("expected LLVM IR Dialect type, got ") << containerType;
    return {};
  }

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // structures.  Check the position index before accessing, it is supposed to
  // be in bounds.
  for (Attribute subAttr : positionAttr) {
    auto positionElementAttr = subAttr.dyn_cast<IntegerAttr>();
    if (!positionElementAttr) {
      op->emitOpError("expected an array of integer literals, got: ")
          << subAttr;
      return {};
    }
    int position = positionElementAttr.getInt();
    if (auto arrayType = llvmType.dyn_cast<LLVMArrayType>()) {
      if (position < 0 ||
          static_cast<unsigned>(position) >= arrayType.getNumElements()) {
        op->emitOpError("position out of bounds: ") << position;
        return {};
      }
      llvmType = arrayType.getElementType();
    } else if (auto structType = llvmType.dyn_cast<LLVMStructType>()) {
      if (position < 0 ||
          static_cast<unsigned>(position) >= structType.getBody().size()) {
        op->emitOpError("position out of bounds") << position;
        return {};
      }
      llvmType = structType.getBody()[position];
    } else {
      op->emitOpError("expected LLVM IR structure/array type, got: ")
          << llvmType;
      return {};
    }
  }
  return llvmType;
}

// <operation> ::= `llvm.extractvalue` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
ParseResult ExtractValueOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand container;
  Type containerType;
  ArrayAttr positionAttr;
  SMLoc attributeLoc, trailingTypeLoc;

  if (parser.parseOperand(container) ||
      parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(positionAttr, "position", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(containerType) ||
      parser.resolveOperand(container, containerType, result.operands))
    return failure();

  auto elementType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!elementType)
    return failure();

  result.addTypes(elementType);
  return success();
}

OpFoldResult LLVM::ExtractValueOp::fold(ArrayRef<Attribute> operands) {
  auto insertValueOp = getContainer().getDefiningOp<InsertValueOp>();
  OpFoldResult result = {};
  while (insertValueOp) {
    if (getPosition() == insertValueOp.getPosition())
      return insertValueOp.getValue();
    unsigned min =
        std::min(getPosition().size(), insertValueOp.getPosition().size());
    // If one is fully prefix of the other, stop propagating back as it will
    // miss dependencies. For instance, %3 should not fold to %f0 in the
    // following example:
    // ```
    //   %1 = llvm.insertvalue %f0, %0[0, 0] :
    //     !llvm.array<4 x !llvm.array<4xf32>>
    //   %2 = llvm.insertvalue %arr, %1[0] :
    //     !llvm.array<4 x !llvm.array<4xf32>>
    //   %3 = llvm.extractvalue %2[0, 0] : !llvm.array<4 x !llvm.array<4xf32>>
    // ```
    if (getPosition().getValue().take_front(min) ==
        insertValueOp.getPosition().getValue().take_front(min))
      return result;

    // If neither a prefix, nor the exact position, we can extract out of the
    // value being inserted into. Moreover, we can try again if that operand
    // is itself an insertvalue expression.
    getContainerMutable().assign(insertValueOp.getContainer());
    result = getResult();
    insertValueOp = insertValueOp.getContainer().getDefiningOp<InsertValueOp>();
  }
  return result;
}

LogicalResult ExtractValueOp::verify() {
  Type valueType = getInsertExtractValueElementType(getContainer().getType(),
                                                    getPositionAttr(), *this);
  if (!valueType)
    return failure();

  if (getRes().getType() != valueType)
    return emitOpError() << "Type mismatch: extracting from "
                         << getContainer().getType() << " should produce "
                         << valueType << " but this op returns "
                         << getRes().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertElementOp.
//===----------------------------------------------------------------------===//

void InsertElementOp::print(OpAsmPrinter &p) {
  p << ' ' << getValue() << ", " << getVector() << "[" << getPosition() << " : "
    << getPosition().getType() << "]";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getVector().getType();
}

// <operation> ::= `llvm.insertelement` ssa-use `,` ssa-use `,` ssa-use
//                 attribute-dict? `:` type
ParseResult InsertElementOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SMLoc loc;
  OpAsmParser::UnresolvedOperand vector, value, position;
  Type vectorType, positionType;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(value) ||
      parser.parseComma() || parser.parseOperand(vector) ||
      parser.parseLSquare() || parser.parseOperand(position) ||
      parser.parseColonType(positionType) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(vectorType))
    return failure();

  if (!LLVM::isCompatibleVectorType(vectorType))
    return parser.emitError(
        loc, "expected LLVM dialect-compatible vector type for operand #1");
  Type valueType = LLVM::getVectorElementType(vectorType);
  if (!valueType)
    return failure();

  if (parser.resolveOperand(vector, vectorType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands) ||
      parser.resolveOperand(position, positionType, result.operands))
    return failure();

  result.addTypes(vectorType);
  return success();
}

LogicalResult InsertElementOp::verify() {
  Type valueType = LLVM::getVectorElementType(getVector().getType());
  if (valueType != getValue().getType())
    return emitOpError() << "Type mismatch: cannot insert "
                         << getValue().getType() << " into "
                         << getVector().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertValueOp.
//===----------------------------------------------------------------------===//

void InsertValueOp::print(OpAsmPrinter &p) {
  p << ' ' << getValue() << ", " << getContainer() << getPosition();
  p.printOptionalAttrDict((*this)->getAttrs(), {"position"});
  p << " : " << getContainer().getType();
}

// <operation> ::= `llvm.insertvaluevalue` ssa-use `,` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
ParseResult InsertValueOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand container, value;
  Type containerType;
  ArrayAttr positionAttr;
  SMLoc attributeLoc, trailingTypeLoc;

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(container) ||
      parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(positionAttr, "position", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(containerType))
    return failure();

  auto valueType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!valueType)
    return failure();

  if (parser.resolveOperand(container, containerType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands))
    return failure();

  result.addTypes(containerType);
  return success();
}

LogicalResult InsertValueOp::verify() {
  Type valueType = getInsertExtractValueElementType(getContainer().getType(),
                                                    getPositionAttr(), *this);
  if (!valueType)
    return failure();

  if (getValue().getType() != valueType)
    return emitOpError() << "Type mismatch: cannot insert "
                         << getValue().getType() << " into "
                         << getContainer().getType();

  return success();
}

//===----------------------------------------------------------------------===//
// Printing, parsing and verification for LLVM::ReturnOp.
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  if (getNumOperands() > 1)
    return emitOpError("expected at most 1 operand");

  if (auto parent = (*this)->getParentOfType<LLVMFuncOp>()) {
    Type expectedType = parent.getFunctionType().getReturnType();
    if (expectedType.isa<LLVMVoidType>()) {
      if (getNumOperands() == 0)
        return success();
      InFlightDiagnostic diag = emitOpError("expected no operands");
      diag.attachNote(parent->getLoc()) << "when returning from function";
      return diag;
    }
    if (getNumOperands() == 0) {
      if (expectedType.isa<LLVMVoidType>())
        return success();
      InFlightDiagnostic diag = emitOpError("expected 1 operand");
      diag.attachNote(parent->getLoc()) << "when returning from function";
      return diag;
    }
    if (expectedType != getOperand(0).getType()) {
      InFlightDiagnostic diag = emitOpError("mismatching result types");
      diag.attachNote(parent->getLoc()) << "when returning from function";
      return diag;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ResumeOp
//===----------------------------------------------------------------------===//

LogicalResult ResumeOp::verify() {
  if (!getValue().getDefiningOp<LandingpadOp>())
    return emitOpError("expects landingpad value as operand");
  // No check for personality of function - landingpad op verifies it.
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier for LLVM::AddressOfOp.
//===----------------------------------------------------------------------===//

template <typename OpTy>
static OpTy lookupSymbolInModule(Operation *parent, StringRef name) {
  Operation *module = parent;
  while (module && !satisfiesLLVMModule(module))
    module = module->getParentOp();
  assert(module && "unexpected operation outside of a module");
  return dyn_cast_or_null<OpTy>(
      mlir::SymbolTable::lookupSymbolIn(module, name));
}

GlobalOp AddressOfOp::getGlobal() {
  return lookupSymbolInModule<LLVM::GlobalOp>((*this)->getParentOp(),
                                              getGlobalName());
}

LLVMFuncOp AddressOfOp::getFunction() {
  return lookupSymbolInModule<LLVM::LLVMFuncOp>((*this)->getParentOp(),
                                                getGlobalName());
}

LogicalResult AddressOfOp::verify() {
  auto global = getGlobal();
  auto function = getFunction();
  if (!global && !function)
    return emitOpError(
        "must reference a global defined by 'llvm.mlir.global' or 'llvm.func'");

  LLVMPointerType type = getType();
  if (global && global.getAddrSpace() != type.getAddressSpace())
    return emitOpError("pointer address space must match address space of the "
                       "referenced global");

  if (type.isOpaque())
    return success();

  if (global && type.getElementType() != global.getType())
    return emitOpError(
        "the type must be a pointer to the type of the referenced global");

  if (function && type.getElementType() != function.getFunctionType())
    return emitOpError(
        "the type must be a pointer to the type of the referenced function");

  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::GlobalOp.
//===----------------------------------------------------------------------===//

void GlobalOp::build(OpBuilder &builder, OperationState &result, Type type,
                     bool isConstant, Linkage linkage, StringRef name,
                     Attribute value, uint64_t alignment, unsigned addrSpace,
                     bool dsoLocal, bool threadLocal,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(name));
  result.addAttribute(getGlobalTypeAttrName(result.name), TypeAttr::get(type));
  if (isConstant)
    result.addAttribute(getConstantAttrName(result.name),
                        builder.getUnitAttr());
  if (value)
    result.addAttribute(getValueAttrName(result.name), value);
  if (dsoLocal)
    result.addAttribute(getDsoLocalAttrName(result.name),
                        builder.getUnitAttr());
  if (threadLocal)
    result.addAttribute(getThreadLocal_AttrName(result.name),
                        builder.getUnitAttr());

  // Only add an alignment attribute if the "alignment" input
  // is different from 0. The value must also be a power of two, but
  // this is tested in GlobalOp::verify, not here.
  if (alignment != 0)
    result.addAttribute(getAlignmentAttrName(result.name),
                        builder.getI64IntegerAttr(alignment));

  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(builder.getContext(), linkage));
  if (addrSpace != 0)
    result.addAttribute(getAddrSpaceAttrName(result.name),
                        builder.getI32IntegerAttr(addrSpace));
  result.attributes.append(attrs.begin(), attrs.end());
  result.addRegion();
}

void GlobalOp::print(OpAsmPrinter &p) {
  p << ' ' << stringifyLinkage(getLinkage()) << ' ';
  if (auto unnamedAddr = getUnnamedAddr()) {
    StringRef str = stringifyUnnamedAddr(*unnamedAddr);
    if (!str.empty())
      p << str << ' ';
  }
  if (getThreadLocal_())
    p << "thread_local ";
  if (getConstant())
    p << "constant ";
  p.printSymbolName(getSymName());
  p << '(';
  if (auto value = getValueOrNull())
    p.printAttribute(value);
  p << ')';
  // Note that the alignment attribute is printed using the
  // default syntax here, even though it is an inherent attribute
  // (as defined in https://mlir.llvm.org/docs/LangRef/#attributes)
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      {SymbolTable::getSymbolAttrName(), getGlobalTypeAttrName(),
       getConstantAttrName(), getValueAttrName(), getLinkageAttrName(),
       getUnnamedAddrAttrName(), getThreadLocal_AttrName()});

  // Print the trailing type unless it's a string global.
  if (getValueOrNull().dyn_cast_or_null<StringAttr>())
    return;
  p << " : " << getType();

  Region &initializer = getInitializerRegion();
  if (!initializer.empty()) {
    p << ' ';
    p.printRegion(initializer, /*printEntryBlockArgs=*/false);
  }
}

// Parses one of the keywords provided in the list `keywords` and returns the
// position of the parsed keyword in the list. If none of the keywords from the
// list is parsed, returns -1.
static int parseOptionalKeywordAlternative(OpAsmParser &parser,
                                           ArrayRef<StringRef> keywords) {
  for (const auto &en : llvm::enumerate(keywords)) {
    if (succeeded(parser.parseOptionalKeyword(en.value())))
      return en.index();
  }
  return -1;
}

namespace {
template <typename Ty>
struct EnumTraits {};

#define REGISTER_ENUM_TYPE(Ty)                                                 \
  template <>                                                                  \
  struct EnumTraits<Ty> {                                                      \
    static StringRef stringify(Ty value) { return stringify##Ty(value); }      \
    static unsigned getMaxEnumVal() { return getMaxEnumValFor##Ty(); }         \
  }

REGISTER_ENUM_TYPE(Linkage);
REGISTER_ENUM_TYPE(UnnamedAddr);
REGISTER_ENUM_TYPE(CConv);
} // namespace

/// Parse an enum from the keyword, or default to the provided default value.
/// The return type is the enum type by default, unless overriden with the
/// second template argument.
template <typename EnumTy, typename RetTy = EnumTy>
static RetTy parseOptionalLLVMKeyword(OpAsmParser &parser,
                                      OperationState &result,
                                      EnumTy defaultValue) {
  SmallVector<StringRef, 10> names;
  for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return static_cast<RetTy>(defaultValue);
  return static_cast<RetTy>(index);
}

// operation ::= `llvm.mlir.global` linkage? `constant`? `@` identifier
//               `(` attribute? `)` align? attribute-list? (`:` type)? region?
// align     ::= `align` `=` UINT64
//
// The type can be omitted for string attributes, in which case it will be
// inferred from the value of the string as [strlen(value) x i8].
ParseResult GlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  // Parse optional linkage, default to External.
  result.addAttribute(getLinkageAttrName(result.name),
                      LLVM::LinkageAttr::get(
                          ctx, parseOptionalLLVMKeyword<Linkage>(
                                   parser, result, LLVM::Linkage::External)));

  if (succeeded(parser.parseOptionalKeyword("thread_local")))
    result.addAttribute(getThreadLocal_AttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  // Parse optional UnnamedAddr, default to None.
  result.addAttribute(getUnnamedAddrAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<UnnamedAddr, int64_t>(
                              parser, result, LLVM::UnnamedAddr::None)));

  if (succeeded(parser.parseOptionalKeyword("constant")))
    result.addAttribute(getConstantAttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  StringAttr name;
  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes) ||
      parser.parseLParen())
    return failure();

  Attribute value;
  if (parser.parseOptionalRParen()) {
    if (parser.parseAttribute(value, getValueAttrName(result.name),
                              result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  Region &initRegion = *result.addRegion();
  if (types.empty()) {
    if (auto strAttr = value.dyn_cast_or_null<StringAttr>()) {
      MLIRContext *context = parser.getContext();
      auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(context, 8),
                                                strAttr.getValue().size());
      types.push_back(arrayType);
    } else {
      return parser.emitError(parser.getNameLoc(),
                              "type can only be omitted for string globals");
    }
  } else {
    OptionalParseResult parseResult =
        parser.parseOptionalRegion(initRegion, /*arguments=*/{},
                                   /*argTypes=*/{});
    if (parseResult.hasValue() && failed(*parseResult))
      return failure();
  }

  result.addAttribute(getGlobalTypeAttrName(result.name),
                      TypeAttr::get(types[0]));
  return success();
}

static bool isZeroAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isNullValue();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isZero();
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

LogicalResult GlobalOp::verify() {
  if (!LLVMPointerType::isValidElementType(getType()))
    return emitOpError(
        "expects type to be a valid element type for an LLVM pointer");
  if ((*this)->getParentOp() && !satisfiesLLVMModule((*this)->getParentOp()))
    return emitOpError("must appear at the module level");

  if (auto strAttr = getValueOrNull().dyn_cast_or_null<StringAttr>()) {
    auto type = getType().dyn_cast<LLVMArrayType>();
    IntegerType elementType =
        type ? type.getElementType().dyn_cast<IntegerType>() : nullptr;
    if (!elementType || elementType.getWidth() != 8 ||
        type.getNumElements() != strAttr.getValue().size())
      return emitOpError(
          "requires an i8 array type of the length equal to that of the string "
          "attribute");
  }

  if (getLinkage() == Linkage::Common) {
    if (Attribute value = getValueOrNull()) {
      if (!isZeroAttribute(value)) {
        return emitOpError()
               << "expected zero value for '"
               << stringifyLinkage(Linkage::Common) << "' linkage";
      }
    }
  }

  if (getLinkage() == Linkage::Appending) {
    if (!getType().isa<LLVMArrayType>()) {
      return emitOpError() << "expected array type for '"
                           << stringifyLinkage(Linkage::Appending)
                           << "' linkage";
    }
  }

  Optional<uint64_t> alignAttr = getAlignment();
  if (alignAttr.hasValue()) {
    uint64_t value = alignAttr.getValue();
    if (!llvm::isPowerOf2_64(value))
      return emitError() << "alignment attribute is not a power of 2";
  }

  return success();
}

LogicalResult GlobalOp::verifyRegions() {
  if (Block *b = getInitializerBlock()) {
    ReturnOp ret = cast<ReturnOp>(b->getTerminator());
    if (ret.operand_type_begin() == ret.operand_type_end())
      return emitOpError("initializer region cannot return void");
    if (*ret.operand_type_begin() != getType())
      return emitOpError("initializer region type ")
             << *ret.operand_type_begin() << " does not match global type "
             << getType();

    for (Operation &op : *b) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect())
        return op.emitError()
               << "ops with side effects not allowed in global initializers";
    }

    if (getValueOrNull())
      return emitOpError("cannot have both initializer value and region");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::GlobalCtorsOp
//===----------------------------------------------------------------------===//

LogicalResult
GlobalCtorsOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute ctor : getCtors()) {
    if (failed(verifySymbolAttrUse(ctor.cast<FlatSymbolRefAttr>(), *this,
                                   symbolTable)))
      return failure();
  }
  return success();
}

LogicalResult GlobalCtorsOp::verify() {
  if (getCtors().size() != getPriorities().size())
    return emitError(
        "mismatch between the number of ctors and the number of priorities");
  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::GlobalDtorsOp
//===----------------------------------------------------------------------===//

LogicalResult
GlobalDtorsOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute dtor : getDtors()) {
    if (failed(verifySymbolAttrUse(dtor.cast<FlatSymbolRefAttr>(), *this,
                                   symbolTable)))
      return failure();
  }
  return success();
}

LogicalResult GlobalDtorsOp::verify() {
  if (getDtors().size() != getPriorities().size())
    return emitError(
        "mismatch between the number of dtors and the number of priorities");
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ShuffleVectorOp.
//===----------------------------------------------------------------------===//
// Expects vector to be of wrapped LLVM vector type and position to be of
// wrapped LLVM i32 type.
void LLVM::ShuffleVectorOp::build(OpBuilder &b, OperationState &result,
                                  Value v1, Value v2, ArrayAttr mask,
                                  ArrayRef<NamedAttribute> attrs) {
  auto containerType = v1.getType();
  auto vType = LLVM::getVectorType(LLVM::getVectorElementType(containerType),
                                   mask.size(),
                                   LLVM::isScalableVectorType(containerType));
  build(b, result, vType, v1, v2, mask);
  result.addAttributes(attrs);
}

void ShuffleVectorOp::print(OpAsmPrinter &p) {
  p << ' ' << getV1() << ", " << getV2() << " " << getMask();
  p.printOptionalAttrDict((*this)->getAttrs(), {"mask"});
  p << " : " << getV1().getType() << ", " << getV2().getType();
}

// <operation> ::= `llvm.shufflevector` ssa-use `, ` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
ParseResult ShuffleVectorOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  SMLoc loc;
  OpAsmParser::UnresolvedOperand v1, v2;
  ArrayAttr maskAttr;
  Type typeV1, typeV2;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(v1) ||
      parser.parseComma() || parser.parseOperand(v2) ||
      parser.parseAttribute(maskAttr, "mask", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(typeV1) || parser.parseComma() ||
      parser.parseType(typeV2) ||
      parser.resolveOperand(v1, typeV1, result.operands) ||
      parser.resolveOperand(v2, typeV2, result.operands))
    return failure();
  if (!LLVM::isCompatibleVectorType(typeV1))
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  auto vType =
      LLVM::getVectorType(LLVM::getVectorElementType(typeV1), maskAttr.size(),
                          LLVM::isScalableVectorType(typeV1));
  result.addTypes(vType);
  return success();
}

LogicalResult ShuffleVectorOp::verify() {
  Type type1 = getV1().getType();
  Type type2 = getV2().getType();
  if (LLVM::getVectorElementType(type1) != LLVM::getVectorElementType(type2))
    return emitOpError("expected matching LLVM IR Dialect element types");
  if (LLVM::isScalableVectorType(type1))
    if (llvm::any_of(getMask(), [](Attribute attr) {
          return attr.cast<IntegerAttr>().getInt() != 0;
        }))
      return emitOpError("expected a splat operation for scalable vectors");
  return success();
}

//===----------------------------------------------------------------------===//
// Implementations for LLVM::LLVMFuncOp.
//===----------------------------------------------------------------------===//

// Add the entry block to the function.
Block *LLVMFuncOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  assert(!isVarArg() && "unimplemented: non-external variadic functions");

  auto *entry = new Block;
  push_back(entry);

  // FIXME: Allow passing in proper locations for the entry arguments.
  LLVMFunctionType type = getFunctionType();
  for (unsigned i = 0, e = type.getNumParams(); i < e; ++i)
    entry->addArgument(type.getParamType(i), getLoc());
  return entry;
}

void LLVMFuncOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, Type type, LLVM::Linkage linkage,
                       bool dsoLocal, CConv cconv,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<DictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(builder.getContext(), linkage));
  result.addAttribute(getCConvAttrName(result.name),
                      CConvAttr::get(builder.getContext(), cconv));
  result.attributes.append(attrs.begin(), attrs.end());
  if (dsoLocal)
    result.addAttribute("dso_local", builder.getUnitAttr());
  if (argAttrs.empty())
    return;

  assert(type.cast<LLVMFunctionType>().getNumParams() == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                /*resultAttrs=*/llvm::None);
}

// Builds an LLVM function type from the given lists of input and output types.
// Returns a null type if any of the types provided are non-LLVM types, or if
// there is more than one output type.
static Type
buildLLVMFunctionType(OpAsmParser &parser, SMLoc loc, ArrayRef<Type> inputs,
                      ArrayRef<Type> outputs,
                      function_interface_impl::VariadicFlag variadicFlag) {
  Builder &b = parser.getBuilder();
  if (outputs.size() > 1) {
    parser.emitError(loc, "failed to construct function type: expected zero or "
                          "one function result");
    return {};
  }

  // Convert inputs to LLVM types, exit early on error.
  SmallVector<Type, 4> llvmInputs;
  for (auto t : inputs) {
    if (!isCompatibleType(t)) {
      parser.emitError(loc, "failed to construct function type: expected LLVM "
                            "type for function arguments");
      return {};
    }
    llvmInputs.push_back(t);
  }

  // No output is denoted as "void" in LLVM type system.
  Type llvmOutput =
      outputs.empty() ? LLVMVoidType::get(b.getContext()) : outputs.front();
  if (!isCompatibleType(llvmOutput)) {
    parser.emitError(loc, "failed to construct function type: expected LLVM "
                          "type for function results")
        << llvmOutput;
    return {};
  }
  return LLVMFunctionType::get(llvmOutput, llvmInputs,
                               variadicFlag.isVariadic());
}

// Parses an LLVM function.
//
// operation ::= `llvm.func` linkage? cconv? function-signature
// function-attributes?
//               function-body
//
ParseResult LLVMFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Default to external linkage if no keyword is provided.
  result.addAttribute(
      getLinkageAttrName(result.name),
      LinkageAttr::get(parser.getContext(),
                       parseOptionalLLVMKeyword<Linkage>(
                           parser, result, LLVM::Linkage::External)));

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(), parseOptionalLLVMKeyword<CConv>(
                                              parser, result, LLVM::CConv::C)));

  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  auto signatureLocation = parser.getCurrentLocation();
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  auto type =
      buildLLVMFunctionType(parser, signatureLocation, argTypes, resultTypes,
                            function_interface_impl::VariadicFlag(isVariadic));
  if (!type)
    return failure();
  result.addAttribute(FunctionOpInterface::getTypeAttrName(),
                      TypeAttr::get(type));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  function_interface_impl::addArgAndResultAttrs(parser.getBuilder(), result,
                                                entryArgs, resultAttrs);

  auto *body = result.addRegion();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs);
  return failure(parseResult.hasValue() && failed(*parseResult));
}

// Print the LLVMFuncOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back. Skips
// the external linkage since it is the default value.
void LLVMFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getLinkage() != LLVM::Linkage::External)
    p << stringifyLinkage(getLinkage()) << ' ';
  if (getCConv() != LLVM::CConv::C)
    p << stringifyCConv(getCConv()) << ' ';

  p.printSymbolName(getName());

  LLVMFunctionType fnType = getFunctionType();
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 1> resTypes;
  argTypes.reserve(fnType.getNumParams());
  for (unsigned i = 0, e = fnType.getNumParams(); i < e; ++i)
    argTypes.push_back(fnType.getParamType(i));

  Type returnType = fnType.getReturnType();
  if (!returnType.isa<LLVMVoidType>())
    resTypes.push_back(returnType);

  function_interface_impl::printFunctionSignature(p, *this, argTypes,
                                                  isVarArg(), resTypes);
  function_interface_impl::printFunctionAttributes(
      p, *this, argTypes.size(), resTypes.size(),
      {getLinkageAttrName(), getCConvAttrName()});

  // Print the body if this is not an external function.
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

// Verifies LLVM- and implementation-specific properties of the LLVM func Op:
// - functions don't have 'common' linkage
// - external functions have 'external' or 'extern_weak' linkage;
// - vararg is (currently) only supported for external functions;
LogicalResult LLVMFuncOp::verify() {
  if (getLinkage() == LLVM::Linkage::Common)
    return emitOpError() << "functions cannot have '"
                         << stringifyLinkage(LLVM::Linkage::Common)
                         << "' linkage";

  // Check to see if this function has a void return with a result attribute to
  // it. It isn't clear what semantics we would assign to that.
  if (getFunctionType().getReturnType().isa<LLVMVoidType>() &&
      !getResultAttrs(0).empty()) {
    return emitOpError()
           << "cannot attach result attributes to functions with a void return";
  }

  if (isExternal()) {
    if (getLinkage() != LLVM::Linkage::External &&
        getLinkage() != LLVM::Linkage::ExternWeak)
      return emitOpError() << "external functions must have '"
                           << stringifyLinkage(LLVM::Linkage::External)
                           << "' or '"
                           << stringifyLinkage(LLVM::Linkage::ExternWeak)
                           << "' linkage";
    return success();
  }

  if (isVarArg())
    return emitOpError("only external functions can be variadic");

  return success();
}

/// Verifies LLVM- and implementation-specific properties of the LLVM func Op:
/// - entry block arguments are of LLVM types.
LogicalResult LLVMFuncOp::verifyRegions() {
  if (isExternal())
    return success();

  unsigned numArguments = getFunctionType().getNumParams();
  Block &entryBlock = front();
  for (unsigned i = 0; i < numArguments; ++i) {
    Type argType = entryBlock.getArgument(i).getType();
    if (!isCompatibleType(argType))
      return emitOpError("entry block argument #")
             << i << " is not of LLVM type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Verification for LLVM::ConstantOp.
//===----------------------------------------------------------------------===//

LogicalResult LLVM::ConstantOp::verify() {
  if (StringAttr sAttr = getValue().dyn_cast<StringAttr>()) {
    auto arrayType = getType().dyn_cast<LLVMArrayType>();
    if (!arrayType || arrayType.getNumElements() != sAttr.getValue().size() ||
        !arrayType.getElementType().isInteger(8)) {
      return emitOpError() << "expected array type of "
                           << sAttr.getValue().size()
                           << " i8 elements for the string constant";
    }
    return success();
  }
  if (auto structType = getType().dyn_cast<LLVMStructType>()) {
    if (structType.getBody().size() != 2 ||
        structType.getBody()[0] != structType.getBody()[1]) {
      return emitError() << "expected struct type with two elements of the "
                            "same type, the type of a complex constant";
    }

    auto arrayAttr = getValue().dyn_cast<ArrayAttr>();
    if (!arrayAttr || arrayAttr.size() != 2 ||
        arrayAttr[0].getType() != arrayAttr[1].getType()) {
      return emitOpError() << "expected array attribute with two elements, "
                              "representing a complex constant";
    }

    Type elementType = structType.getBody()[0];
    if (!elementType
             .isa<IntegerType, Float16Type, Float32Type, Float64Type>()) {
      return emitError()
             << "expected struct element types to be floating point type or "
                "integer type";
    }
    return success();
  }
  if (!getValue().isa<IntegerAttr, ArrayAttr, FloatAttr, ElementsAttr>())
    return emitOpError()
           << "only supports integer, float, string or elements attributes";
  return success();
}

// Constant op constant-folds to its value.
OpFoldResult LLVM::ConstantOp::fold(ArrayRef<Attribute>) { return getValue(); }

//===----------------------------------------------------------------------===//
// Utility functions for parsing atomic ops
//===----------------------------------------------------------------------===//

// Helper function to parse a keyword into the specified attribute named by
// `attrName`. The keyword must match one of the string values defined by the
// AtomicBinOp enum. The resulting I64 attribute is added to the `result`
// state.
static ParseResult parseAtomicBinOp(OpAsmParser &parser, OperationState &result,
                                    StringRef attrName) {
  SMLoc loc;
  StringRef keyword;
  if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword))
    return failure();

  // Replace the keyword `keyword` with an integer attribute.
  auto kind = symbolizeAtomicBinOp(keyword);
  if (!kind) {
    return parser.emitError(loc)
           << "'" << keyword << "' is an incorrect value of the '" << attrName
           << "' attribute";
  }

  auto value = static_cast<int64_t>(kind.getValue());
  auto attr = parser.getBuilder().getI64IntegerAttr(value);
  result.addAttribute(attrName, attr);

  return success();
}

// Helper function to parse a keyword into the specified attribute named by
// `attrName`. The keyword must match one of the string values defined by the
// AtomicOrdering enum. The resulting I64 attribute is added to the `result`
// state.
static ParseResult parseAtomicOrdering(OpAsmParser &parser,
                                       OperationState &result,
                                       StringRef attrName) {
  SMLoc loc;
  StringRef ordering;
  if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&ordering))
    return failure();

  // Replace the keyword `ordering` with an integer attribute.
  auto kind = symbolizeAtomicOrdering(ordering);
  if (!kind) {
    return parser.emitError(loc)
           << "'" << ordering << "' is an incorrect value of the '" << attrName
           << "' attribute";
  }

  auto value = static_cast<int64_t>(kind.getValue());
  auto attr = parser.getBuilder().getI64IntegerAttr(value);
  result.addAttribute(attrName, attr);

  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::AtomicRMWOp.
//===----------------------------------------------------------------------===//

void AtomicRMWOp::print(OpAsmPrinter &p) {
  p << ' ' << stringifyAtomicBinOp(getBinOp()) << ' ' << getPtr() << ", "
    << getVal() << ' ' << stringifyAtomicOrdering(getOrdering()) << ' ';
  p.printOptionalAttrDict((*this)->getAttrs(), {"bin_op", "ordering"});
  p << " : " << getRes().getType();
}

// <operation> ::= `llvm.atomicrmw` keyword ssa-use `,` ssa-use keyword
//                 attribute-dict? `:` type
ParseResult AtomicRMWOp::parse(OpAsmParser &parser, OperationState &result) {
  Type type;
  OpAsmParser::UnresolvedOperand ptr, val;
  if (parseAtomicBinOp(parser, result, "bin_op") || parser.parseOperand(ptr) ||
      parser.parseComma() || parser.parseOperand(val) ||
      parseAtomicOrdering(parser, result, "ordering") ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptr, LLVM::LLVMPointerType::get(type),
                            result.operands) ||
      parser.resolveOperand(val, type, result.operands))
    return failure();

  result.addTypes(type);
  return success();
}

LogicalResult AtomicRMWOp::verify() {
  auto ptrType = getPtr().getType().cast<LLVM::LLVMPointerType>();
  auto valType = getVal().getType();
  if (valType != ptrType.getElementType())
    return emitOpError("expected LLVM IR element type for operand #0 to "
                       "match type for operand #1");
  auto resType = getRes().getType();
  if (resType != valType)
    return emitOpError(
        "expected LLVM IR result type to match type for operand #1");
  if (getBinOp() == AtomicBinOp::fadd || getBinOp() == AtomicBinOp::fsub) {
    if (!mlir::LLVM::isCompatibleFloatingPointType(valType))
      return emitOpError("expected LLVM IR floating point type");
  } else if (getBinOp() == AtomicBinOp::xchg) {
    auto intType = valType.dyn_cast<IntegerType>();
    unsigned intBitWidth = intType ? intType.getWidth() : 0;
    if (intBitWidth != 8 && intBitWidth != 16 && intBitWidth != 32 &&
        intBitWidth != 64 && !valType.isa<BFloat16Type>() &&
        !valType.isa<Float16Type>() && !valType.isa<Float32Type>() &&
        !valType.isa<Float64Type>())
      return emitOpError("unexpected LLVM IR type for 'xchg' bin_op");
  } else {
    auto intType = valType.dyn_cast<IntegerType>();
    unsigned intBitWidth = intType ? intType.getWidth() : 0;
    if (intBitWidth != 8 && intBitWidth != 16 && intBitWidth != 32 &&
        intBitWidth != 64)
      return emitOpError("expected LLVM IR integer type");
  }

  if (static_cast<unsigned>(getOrdering()) <
      static_cast<unsigned>(AtomicOrdering::monotonic))
    return emitOpError() << "expected at least '"
                         << stringifyAtomicOrdering(AtomicOrdering::monotonic)
                         << "' ordering";

  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::AtomicCmpXchgOp.
//===----------------------------------------------------------------------===//

void AtomicCmpXchgOp::print(OpAsmPrinter &p) {
  p << ' ' << getPtr() << ", " << getCmp() << ", " << getVal() << ' '
    << stringifyAtomicOrdering(getSuccessOrdering()) << ' '
    << stringifyAtomicOrdering(getFailureOrdering());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"success_ordering", "failure_ordering"});
  p << " : " << getVal().getType();
}

// <operation> ::= `llvm.cmpxchg` ssa-use `,` ssa-use `,` ssa-use
//                 keyword keyword attribute-dict? `:` type
ParseResult AtomicCmpXchgOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  Type type;
  OpAsmParser::UnresolvedOperand ptr, cmp, val;
  if (parser.parseOperand(ptr) || parser.parseComma() ||
      parser.parseOperand(cmp) || parser.parseComma() ||
      parser.parseOperand(val) ||
      parseAtomicOrdering(parser, result, "success_ordering") ||
      parseAtomicOrdering(parser, result, "failure_ordering") ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(ptr, LLVM::LLVMPointerType::get(type),
                            result.operands) ||
      parser.resolveOperand(cmp, type, result.operands) ||
      parser.resolveOperand(val, type, result.operands))
    return failure();

  auto boolType = IntegerType::get(builder.getContext(), 1);
  auto resultType =
      LLVMStructType::getLiteral(builder.getContext(), {type, boolType});
  result.addTypes(resultType);

  return success();
}

LogicalResult AtomicCmpXchgOp::verify() {
  auto ptrType = getPtr().getType().cast<LLVM::LLVMPointerType>();
  if (!ptrType)
    return emitOpError("expected LLVM IR pointer type for operand #0");
  auto cmpType = getCmp().getType();
  auto valType = getVal().getType();
  if (cmpType != ptrType.getElementType() || cmpType != valType)
    return emitOpError("expected LLVM IR element type for operand #0 to "
                       "match type for all other operands");
  auto intType = valType.dyn_cast<IntegerType>();
  unsigned intBitWidth = intType ? intType.getWidth() : 0;
  if (!valType.isa<LLVMPointerType>() && intBitWidth != 8 &&
      intBitWidth != 16 && intBitWidth != 32 && intBitWidth != 64 &&
      !valType.isa<BFloat16Type>() && !valType.isa<Float16Type>() &&
      !valType.isa<Float32Type>() && !valType.isa<Float64Type>())
    return emitOpError("unexpected LLVM IR type");
  if (getSuccessOrdering() < AtomicOrdering::monotonic ||
      getFailureOrdering() < AtomicOrdering::monotonic)
    return emitOpError("ordering must be at least 'monotonic'");
  if (getFailureOrdering() == AtomicOrdering::release ||
      getFailureOrdering() == AtomicOrdering::acq_rel)
    return emitOpError("failure ordering cannot be 'release' or 'acq_rel'");
  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::FenceOp.
//===----------------------------------------------------------------------===//

// <operation> ::= `llvm.fence` (`syncscope(`strAttr`)`)? keyword
// attribute-dict?
ParseResult FenceOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr sScope;
  StringRef syncscopeKeyword = "syncscope";
  if (!failed(parser.parseOptionalKeyword(syncscopeKeyword))) {
    if (parser.parseLParen() ||
        parser.parseAttribute(sScope, syncscopeKeyword, result.attributes) ||
        parser.parseRParen())
      return failure();
  } else {
    result.addAttribute(syncscopeKeyword,
                        parser.getBuilder().getStringAttr(""));
  }
  if (parseAtomicOrdering(parser, result, "ordering") ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void FenceOp::print(OpAsmPrinter &p) {
  StringRef syncscopeKeyword = "syncscope";
  p << ' ';
  if (!(*this)->getAttr(syncscopeKeyword).cast<StringAttr>().getValue().empty())
    p << "syncscope(" << (*this)->getAttr(syncscopeKeyword) << ") ";
  p << stringifyAtomicOrdering(getOrdering());
}

LogicalResult FenceOp::verify() {
  if (getOrdering() == AtomicOrdering::not_atomic ||
      getOrdering() == AtomicOrdering::unordered ||
      getOrdering() == AtomicOrdering::monotonic)
    return emitOpError("can be given only acquire, release, acq_rel, "
                       "and seq_cst orderings");
  return success();
}

//===----------------------------------------------------------------------===//
// Folder for LLVM::BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::BitcastOp::fold(ArrayRef<Attribute> operands) {
  // bitcast(x : T0, T0) -> x
  if (getArg().getType() == getType())
    return getArg();
  // bitcast(bitcast(x : T0, T1), T0) -> x
  if (auto prev = getArg().getDefiningOp<BitcastOp>())
    if (prev.getArg().getType() == getType())
      return prev.getArg();
  return {};
}

//===----------------------------------------------------------------------===//
// Folder for LLVM::AddrSpaceCastOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::AddrSpaceCastOp::fold(ArrayRef<Attribute> operands) {
  // addrcast(x : T0, T0) -> x
  if (getArg().getType() == getType())
    return getArg();
  // addrcast(addrcast(x : T0, T1), T0) -> x
  if (auto prev = getArg().getDefiningOp<AddrSpaceCastOp>())
    if (prev.getArg().getType() == getType())
      return prev.getArg();
  return {};
}

//===----------------------------------------------------------------------===//
// Folder for LLVM::GEPOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::GEPOp::fold(ArrayRef<Attribute> operands) {
  // gep %x:T, 0 -> %x
  if (getBase().getType() == getType() && getIndices().size() == 1 &&
      matchPattern(getIndices()[0], m_Zero()))
    return getBase();
  return {};
}

//===----------------------------------------------------------------------===//
// LLVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void LLVMDialect::initialize() {
  addAttributes<FMFAttr, LinkageAttr, CConvAttr, LoopOptionsAttr>();

  // clang-format off
  addTypes<LLVMVoidType,
           LLVMPPCFP128Type,
           LLVMX86MMXType,
           LLVMTokenType,
           LLVMLabelType,
           LLVMMetadataType,
           LLVMFunctionType,
           LLVMPointerType,
           LLVMFixedVectorType,
           LLVMScalableVectorType,
           LLVMArrayType,
           LLVMStructType>();
  // clang-format on
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"
      ,
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.cpp.inc"
      >();

  // Support unknown operations because not all LLVM operations are registered.
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"

/// Parse a type registered to this dialect.
Type LLVMDialect::parseType(DialectAsmParser &parser) const {
  return detail::parseType(parser);
}

/// Print a type registered to this dialect.
void LLVMDialect::printType(Type type, DialectAsmPrinter &os) const {
  return detail::printType(type, os);
}

LogicalResult LLVMDialect::verifyDataLayoutString(
    StringRef descr, llvm::function_ref<void(const Twine &)> reportError) {
  llvm::Expected<llvm::DataLayout> maybeDataLayout =
      llvm::DataLayout::parse(descr);
  if (maybeDataLayout)
    return success();

  std::string message;
  llvm::raw_string_ostream messageStream(message);
  llvm::logAllUnhandledErrors(maybeDataLayout.takeError(), messageStream);
  reportError("invalid data layout descriptor: " + messageStream.str());
  return failure();
}

/// Verify LLVM dialect attributes.
LogicalResult LLVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // If the `llvm.loop` attribute is present, enforce the following structure,
  // which the module translation can assume.
  if (attr.getName() == LLVMDialect::getLoopAttrName()) {
    auto loopAttr = attr.getValue().dyn_cast<DictionaryAttr>();
    if (!loopAttr)
      return op->emitOpError() << "expected '" << LLVMDialect::getLoopAttrName()
                               << "' to be a dictionary attribute";
    Optional<NamedAttribute> parallelAccessGroup =
        loopAttr.getNamed(LLVMDialect::getParallelAccessAttrName());
    if (parallelAccessGroup.hasValue()) {
      auto accessGroups = parallelAccessGroup->getValue().dyn_cast<ArrayAttr>();
      if (!accessGroups)
        return op->emitOpError()
               << "expected '" << LLVMDialect::getParallelAccessAttrName()
               << "' to be an array attribute";
      for (Attribute attr : accessGroups) {
        auto accessGroupRef = attr.dyn_cast<SymbolRefAttr>();
        if (!accessGroupRef)
          return op->emitOpError()
                 << "expected '" << attr << "' to be a symbol reference";
        StringAttr metadataName = accessGroupRef.getRootReference();
        auto metadataOp =
            SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
                op->getParentOp(), metadataName);
        if (!metadataOp)
          return op->emitOpError()
                 << "expected '" << attr << "' to reference a metadata op";
        StringAttr accessGroupName = accessGroupRef.getLeafReference();
        Operation *accessGroupOp =
            SymbolTable::lookupNearestSymbolFrom(metadataOp, accessGroupName);
        if (!accessGroupOp)
          return op->emitOpError()
                 << "expected '" << attr << "' to reference an access_group op";
      }
    }

    Optional<NamedAttribute> loopOptions =
        loopAttr.getNamed(LLVMDialect::getLoopOptionsAttrName());
    if (loopOptions.hasValue() &&
        !loopOptions->getValue().isa<LoopOptionsAttr>())
      return op->emitOpError()
             << "expected '" << LLVMDialect::getLoopOptionsAttrName()
             << "' to be a `loopopts` attribute";
  }

  if (attr.getName() == LLVMDialect::getStructAttrsAttrName()) {
    return op->emitOpError()
           << "'" << LLVM::LLVMDialect::getStructAttrsAttrName()
           << "' is permitted only in argument or result attributes";
  }

  // If the data layout attribute is present, it must use the LLVM data layout
  // syntax. Try parsing it and report errors in case of failure. Users of this
  // attribute may assume it is well-formed and can pass it to the (asserting)
  // llvm::DataLayout constructor.
  if (attr.getName() != LLVM::LLVMDialect::getDataLayoutAttrName())
    return success();
  if (auto stringAttr = attr.getValue().dyn_cast<StringAttr>())
    return verifyDataLayoutString(
        stringAttr.getValue(),
        [op](const Twine &message) { op->emitOpError() << message.str(); });

  return op->emitOpError() << "expected '"
                           << LLVM::LLVMDialect::getDataLayoutAttrName()
                           << "' to be a string attributes";
}

LogicalResult LLVMDialect::verifyStructAttr(Operation *op, Attribute attr,
                                            Type annotatedType) {
  auto structType = annotatedType.dyn_cast<LLVMStructType>();
  if (!structType) {
    const auto emitIncorrectAnnotatedType = [&op]() {
      return op->emitError()
             << "expected '" << LLVMDialect::getStructAttrsAttrName()
             << "' to annotate '!llvm.struct' or '!llvm.ptr<struct<...>>'";
    };
    const auto ptrType = annotatedType.dyn_cast<LLVMPointerType>();
    if (!ptrType)
      return emitIncorrectAnnotatedType();
    structType = ptrType.getElementType().dyn_cast<LLVMStructType>();
    if (!structType)
      return emitIncorrectAnnotatedType();
  }

  const auto arrAttrs = attr.dyn_cast<ArrayAttr>();
  if (!arrAttrs)
    return op->emitError() << "expected '"
                           << LLVMDialect::getStructAttrsAttrName()
                           << "' to be an array attribute";

  if (structType.getBody().size() != arrAttrs.size())
    return op->emitError()
           << "size of '" << LLVMDialect::getStructAttrsAttrName()
           << "' must match the size of the annotated '!llvm.struct'";
  return success();
}

static LogicalResult verifyFuncOpInterfaceStructAttr(
    Operation *op, Attribute attr,
    const std::function<Type(FunctionOpInterface)> &getAnnotatedType) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op))
    return LLVMDialect::verifyStructAttr(op, attr, getAnnotatedType(funcOp));
  return op->emitError() << "expected '"
                         << LLVMDialect::getStructAttrsAttrName()
                         << "' to be used on function-like operations";
}

/// Verify LLVMIR function argument attributes.
LogicalResult LLVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIdx,
                                                    unsigned argIdx,
                                                    NamedAttribute argAttr) {
  // Check that llvm.noalias is a unit attribute.
  if (argAttr.getName() == LLVMDialect::getNoAliasAttrName() &&
      !argAttr.getValue().isa<UnitAttr>())
    return op->emitError()
           << "expected llvm.noalias argument attribute to be a unit attribute";
  // Check that llvm.align is an integer attribute.
  if (argAttr.getName() == LLVMDialect::getAlignAttrName() &&
      !argAttr.getValue().isa<IntegerAttr>())
    return op->emitError()
           << "llvm.align argument attribute of non integer type";
  if (argAttr.getName() == LLVMDialect::getStructAttrsAttrName()) {
    return verifyFuncOpInterfaceStructAttr(
        op, argAttr.getValue(), [argIdx](FunctionOpInterface funcOp) {
          return funcOp.getArgumentTypes()[argIdx];
        });
  }
  return success();
}

LogicalResult LLVMDialect::verifyRegionResultAttribute(Operation *op,
                                                       unsigned regionIdx,
                                                       unsigned resIdx,
                                                       NamedAttribute resAttr) {
  if (resAttr.getName() == LLVMDialect::getStructAttrsAttrName()) {
    return verifyFuncOpInterfaceStructAttr(
        op, resAttr.getValue(), [resIdx](FunctionOpInterface funcOp) {
          return funcOp.getResultTypes()[resIdx];
        });
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

Value mlir::LLVM::createGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     LLVM::Linkage linkage) {
  assert(builder.getInsertionBlock() &&
         builder.getInsertionBlock()->getParentOp() &&
         "expected builder to point to a block constrained in an op");
  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(module && "builder points to an op outside of a module");

  // Create the global at the entry of the module.
  OpBuilder moduleBuilder(module.getBodyRegion(), builder.getListener());
  MLIRContext *ctx = builder.getContext();
  auto type = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), value.size());
  auto global = moduleBuilder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, linkage, name,
      builder.getStringAttr(value), /*alignment=*/0);

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(ctx, 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), globalPtr,
      ValueRange{cst0, cst0});
}

bool mlir::LLVM::satisfiesLLVMModule(Operation *op) {
  return op->hasTrait<OpTrait::SymbolTable>() &&
         op->hasTrait<OpTrait::IsIsolatedFromAbove>();
}

void FMFAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer << stringifyFastmathFlags(this->getFlags());
  printer << ">";
}

Attribute FMFAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  FastmathFlags flags = {};
  if (failed(parser.parseOptionalGreater())) {
    auto parseFlags = [&]() -> ParseResult {
      StringRef elemName;
      if (failed(parser.parseKeyword(&elemName)))
        return failure();

      auto elem = symbolizeFastmathFlags(elemName);
      if (!elem)
        return parser.emitError(parser.getNameLoc(), "Unknown fastmath flag: ")
               << elemName;

      flags = flags | *elem;
      return success();
    };
    if (failed(parser.parseCommaSeparatedList(parseFlags)) ||
        parser.parseGreater())
      return {};
  }

  return FMFAttr::get(parser.getContext(), flags);
}

void LinkageAttr::print(AsmPrinter &printer) const {
  printer << "<";
  if (static_cast<uint64_t>(getLinkage()) <= getMaxEnumValForLinkage())
    printer << stringifyEnum(getLinkage());
  else
    printer << static_cast<uint64_t>(getLinkage());
  printer << ">";
}

Attribute LinkageAttr::parse(AsmParser &parser, Type type) {
  StringRef elemName;
  if (parser.parseLess() || parser.parseKeyword(&elemName) ||
      parser.parseGreater())
    return {};
  auto elem = linkage::symbolizeLinkage(elemName);
  if (!elem) {
    parser.emitError(parser.getNameLoc(), "Unknown linkage: ") << elemName;
    return {};
  }
  Linkage linkage = *elem;
  return LinkageAttr::get(parser.getContext(), linkage);
}

void CConvAttr::print(AsmPrinter &printer) const {
  printer << "<";
  if (static_cast<uint64_t>(getCallingConv()) <= cconv::getMaxEnumValForCConv())
    printer << stringifyEnum(getCallingConv());
  else
    printer << "INVALID_cc_" << static_cast<uint64_t>(getCallingConv());
  printer << ">";
}

Attribute CConvAttr::parse(AsmParser &parser, Type type) {
  StringRef convName;

  if (parser.parseLess() || parser.parseKeyword(&convName) ||
      parser.parseGreater())
    return {};
  auto cconv = cconv::symbolizeCConv(convName);
  if (!cconv) {
    parser.emitError(parser.getNameLoc(), "unknown calling convention: ")
        << convName;
    return {};
  }
  CConv cconvVal = *cconv;
  return CConvAttr::get(parser.getContext(), cconvVal);
}

LoopOptionsAttrBuilder::LoopOptionsAttrBuilder(LoopOptionsAttr attr)
    : options(attr.getOptions().begin(), attr.getOptions().end()) {}

template <typename T>
LoopOptionsAttrBuilder &LoopOptionsAttrBuilder::setOption(LoopOptionCase tag,
                                                          Optional<T> value) {
  auto option = llvm::find_if(
      options, [tag](auto option) { return option.first == tag; });
  if (option != options.end()) {
    if (value.hasValue())
      option->second = *value;
    else
      options.erase(option);
  } else {
    options.push_back(LoopOptionsAttr::OptionValuePair(tag, *value));
  }
  return *this;
}

LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisableLICM(Optional<bool> value) {
  return setOption(LoopOptionCase::disable_licm, value);
}

/// Set the `interleave_count` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setInterleaveCount(Optional<uint64_t> count) {
  return setOption(LoopOptionCase::interleave_count, count);
}

/// Set the `disable_unroll` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisableUnroll(Optional<bool> value) {
  return setOption(LoopOptionCase::disable_unroll, value);
}

/// Set the `disable_pipeline` option to the provided value. If no value
/// is provided the option is deleted.
LoopOptionsAttrBuilder &
LoopOptionsAttrBuilder::setDisablePipeline(Optional<bool> value) {
  return setOption(LoopOptionCase::disable_pipeline, value);
}

/// Set the `pipeline_initiation_interval` option to the provided value.
/// If no value is provided the option is deleted.
LoopOptionsAttrBuilder &LoopOptionsAttrBuilder::setPipelineInitiationInterval(
    Optional<uint64_t> count) {
  return setOption(LoopOptionCase::pipeline_initiation_interval, count);
}

template <typename T>
static Optional<T>
getOption(ArrayRef<std::pair<LoopOptionCase, int64_t>> options,
          LoopOptionCase option) {
  auto it =
      lower_bound(options, option, [](auto optionPair, LoopOptionCase option) {
        return optionPair.first < option;
      });
  if (it == options.end())
    return {};
  return static_cast<T>(it->second);
}

Optional<bool> LoopOptionsAttr::disableUnroll() {
  return getOption<bool>(getOptions(), LoopOptionCase::disable_unroll);
}

Optional<bool> LoopOptionsAttr::disableLICM() {
  return getOption<bool>(getOptions(), LoopOptionCase::disable_licm);
}

Optional<int64_t> LoopOptionsAttr::interleaveCount() {
  return getOption<int64_t>(getOptions(), LoopOptionCase::interleave_count);
}

/// Build the LoopOptions Attribute from a sorted array of individual options.
LoopOptionsAttr LoopOptionsAttr::get(
    MLIRContext *context,
    ArrayRef<std::pair<LoopOptionCase, int64_t>> sortedOptions) {
  assert(llvm::is_sorted(sortedOptions, llvm::less_first()) &&
         "LoopOptionsAttr ctor expects a sorted options array");
  return Base::get(context, sortedOptions);
}

/// Build the LoopOptions Attribute from a sorted array of individual options.
LoopOptionsAttr LoopOptionsAttr::get(MLIRContext *context,
                                     LoopOptionsAttrBuilder &optionBuilders) {
  llvm::sort(optionBuilders.options, llvm::less_first());
  return Base::get(context, optionBuilders.options);
}

void LoopOptionsAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getOptions(), printer, [&](auto option) {
    printer << stringifyEnum(option.first) << " = ";
    switch (option.first) {
    case LoopOptionCase::disable_licm:
    case LoopOptionCase::disable_unroll:
    case LoopOptionCase::disable_pipeline:
      printer << (option.second ? "true" : "false");
      break;
    case LoopOptionCase::interleave_count:
    case LoopOptionCase::pipeline_initiation_interval:
      printer << option.second;
      break;
    }
  });
  printer << ">";
}

Attribute LoopOptionsAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};

  SmallVector<std::pair<LoopOptionCase, int64_t>> options;
  llvm::SmallDenseSet<LoopOptionCase> seenOptions;
  auto parseLoopOptions = [&]() -> ParseResult {
    StringRef optionName;
    if (parser.parseKeyword(&optionName))
      return failure();

    auto option = symbolizeLoopOptionCase(optionName);
    if (!option)
      return parser.emitError(parser.getNameLoc(), "unknown loop option: ")
             << optionName;
    if (!seenOptions.insert(*option).second)
      return parser.emitError(parser.getNameLoc(), "loop option present twice");
    if (failed(parser.parseEqual()))
      return failure();

    int64_t value;
    switch (*option) {
    case LoopOptionCase::disable_licm:
    case LoopOptionCase::disable_unroll:
    case LoopOptionCase::disable_pipeline:
      if (succeeded(parser.parseOptionalKeyword("true")))
        value = 1;
      else if (succeeded(parser.parseOptionalKeyword("false")))
        value = 0;
      else {
        return parser.emitError(parser.getNameLoc(),
                                "expected boolean value 'true' or 'false'");
      }
      break;
    case LoopOptionCase::interleave_count:
    case LoopOptionCase::pipeline_initiation_interval:
      if (failed(parser.parseInteger(value)))
        return parser.emitError(parser.getNameLoc(), "expected integer value");
      break;
    }
    options.push_back(std::make_pair(*option, value));
    return success();
  };
  if (parser.parseCommaSeparatedList(parseLoopOptions) || parser.parseGreater())
    return {};

  llvm::sort(options, llvm::less_first());
  return get(parser.getContext(), options);
}
