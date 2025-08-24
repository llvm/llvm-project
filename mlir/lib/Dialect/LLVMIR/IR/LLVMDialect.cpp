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
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"

#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::cconv::getMaxEnumValForCConv;
using mlir::LLVM::linkage::getMaxEnumValForLinkage;
using mlir::LLVM::tailcallkind::getMaxEnumValForTailCallKind;

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute Helpers
//===----------------------------------------------------------------------===//

static constexpr const char kElemTypeAttrName[] = "elem_type";

static auto processFMFAttr(ArrayRef<NamedAttribute> attrs) {
  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range(attrs, [&](NamedAttribute attr) {
        if (attr.getName() == "fastmathFlags") {
          auto defAttr =
              FastmathFlagsAttr::get(attr.getValue().getContext(), {});
          return defAttr != attr.getValue();
        }
        return true;
      }));
  return filteredAttrs;
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

/// Returns a boolean type that has the same shape as `type`. It supports both
/// fixed size vectors as well as scalable vectors.
static Type getI1SameShape(Type type) {
  Type i1Type = IntegerType::get(type.getContext(), 1);
  if (LLVM::isCompatibleVectorType(type))
    return LLVM::getVectorType(i1Type, LLVM::getVectorNumElements(type));
  return i1Type;
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
REGISTER_ENUM_TYPE(TailCallKind);
REGISTER_ENUM_TYPE(Visibility);
} // namespace

/// Parse an enum from the keyword, or default to the provided default value.
/// The return type is the enum type by default, unless overridden with the
/// second template argument.
template <typename EnumTy, typename RetTy = EnumTy>
static RetTy parseOptionalLLVMKeyword(OpAsmParser &parser,
                                      EnumTy defaultValue) {
  SmallVector<StringRef, 10> names;
  for (unsigned i = 0, e = EnumTraits<EnumTy>::getMaxEnumVal(); i <= e; ++i)
    names.push_back(EnumTraits<EnumTy>::stringify(static_cast<EnumTy>(i)));

  int index = parseOptionalKeywordAlternative(parser, names);
  if (index == -1)
    return static_cast<RetTy>(defaultValue);
  return static_cast<RetTy>(index);
}

static void printLLVMLinkage(OpAsmPrinter &p, Operation *, LinkageAttr val) {
  p << stringifyLinkage(val.getLinkage());
}

static ParseResult parseLLVMLinkage(OpAsmParser &p, LinkageAttr &val) {
  val = LinkageAttr::get(
      p.getContext(),
      parseOptionalLLVMKeyword<LLVM::Linkage>(p, LLVM::Linkage::External));
  return success();
}

static ArrayAttr getLLVMAlignParamForCompressExpand(OpBuilder &builder,
                                                    bool isExpandLoad,
                                                    uint64_t alignment = 1) {
  // From
  // https://llvm.org/docs/LangRef.html#llvm-masked-expandload-intrinsics
  // https://llvm.org/docs/LangRef.html#llvm-masked-compressstore-intrinsics
  //
  //   The pointer alignment defaults to 1.
  if (alignment == 1) {
    return nullptr;
  }

  auto emptyDictAttr = builder.getDictionaryAttr({});
  auto alignmentAttr = builder.getI64IntegerAttr(alignment);
  auto namedAttr =
      builder.getNamedAttr(LLVMDialect::getAlignAttrName(), alignmentAttr);
  SmallVector<mlir::NamedAttribute> attrs = {namedAttr};
  auto alignDictAttr = builder.getDictionaryAttr(attrs);
  // From
  // https://llvm.org/docs/LangRef.html#llvm-masked-expandload-intrinsics
  // https://llvm.org/docs/LangRef.html#llvm-masked-compressstore-intrinsics
  //
  //   The align parameter attribute can be provided for [expandload]'s first
  //   argument. The align parameter attribute can be provided for
  //   [compressstore]'s second argument.
  int pos = isExpandLoad ? 0 : 1;
  return pos == 0 ? builder.getArrayAttr(
                        {alignDictAttr, emptyDictAttr, emptyDictAttr})
                  : builder.getArrayAttr(
                        {emptyDictAttr, alignDictAttr, emptyDictAttr});
}

//===----------------------------------------------------------------------===//
// Operand bundle helpers.
//===----------------------------------------------------------------------===//

static void printOneOpBundle(OpAsmPrinter &p, OperandRange operands,
                             TypeRange operandTypes, StringRef tag) {
  p.printString(tag);
  p << "(";

  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operandTypes, p);
  }

  p << ")";
}

static void printOpBundles(OpAsmPrinter &p, Operation *op,
                           OperandRangeRange opBundleOperands,
                           TypeRangeRange opBundleOperandTypes,
                           std::optional<ArrayAttr> opBundleTags) {
  if (opBundleOperands.empty())
    return;
  assert(opBundleTags && "expect operand bundle tags");

  p << "[";
  llvm::interleaveComma(
      llvm::zip(opBundleOperands, opBundleOperandTypes, *opBundleTags), p,
      [&p](auto bundle) {
        auto bundleTag = cast<StringAttr>(std::get<2>(bundle)).getValue();
        printOneOpBundle(p, std::get<0>(bundle), std::get<1>(bundle),
                         bundleTag);
      });
  p << "]";
}

static ParseResult parseOneOpBundle(
    OpAsmParser &p,
    SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> &opBundleOperands,
    SmallVector<SmallVector<Type>> &opBundleOperandTypes,
    SmallVector<Attribute> &opBundleTags) {
  SMLoc currentParserLoc = p.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;
  std::string tag;

  if (p.parseString(&tag))
    return p.emitError(currentParserLoc, "expect operand bundle tag");

  if (p.parseLParen())
    return failure();

  if (p.parseOptionalRParen()) {
    if (p.parseOperandList(operands) || p.parseColon() ||
        p.parseTypeList(types) || p.parseRParen())
      return failure();
  }

  opBundleOperands.push_back(std::move(operands));
  opBundleOperandTypes.push_back(std::move(types));
  opBundleTags.push_back(StringAttr::get(p.getContext(), tag));

  return success();
}

static std::optional<ParseResult> parseOpBundles(
    OpAsmParser &p,
    SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> &opBundleOperands,
    SmallVector<SmallVector<Type>> &opBundleOperandTypes,
    ArrayAttr &opBundleTags) {
  if (p.parseOptionalLSquare())
    return std::nullopt;

  if (succeeded(p.parseOptionalRSquare()))
    return success();

  SmallVector<Attribute> opBundleTagAttrs;
  auto bundleParser = [&] {
    return parseOneOpBundle(p, opBundleOperands, opBundleOperandTypes,
                            opBundleTagAttrs);
  };
  if (p.parseCommaSeparatedList(bundleParser))
    return failure();

  if (p.parseRSquare())
    return failure();

  opBundleTags = ArrayAttr::get(p.getContext(), opBundleTagAttrs);

  return success();
}

//===----------------------------------------------------------------------===//
// Printing, parsing, folding and builder for LLVM::CmpOp.
//===----------------------------------------------------------------------===//

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
    std::optional<ICmpPredicate> predicate =
        symbolizeICmpPredicate(predicateAttr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateAttr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(*predicate);
  } else {
    std::optional<FCmpPredicate> predicate =
        symbolizeFCmpPredicate(predicateAttr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateAttr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(*predicate);
  }

  result.attributes.set("predicate",
                        parser.getBuilder().getI64IntegerAttr(predicateValue));

  // The result type is either i1 or a vector type <? x i1> if the inputs are
  // vectors.
  if (!isCompatibleType(type))
    return parser.emitError(trailingTypeLoc,
                            "expected LLVM dialect-compatible type");
  result.addTypes(getI1SameShape(type));
  return success();
}

ParseResult ICmpOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseCmpOp<ICmpPredicate>(parser, result);
}

ParseResult FCmpOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseCmpOp<FCmpPredicate>(parser, result);
}

/// Returns a scalar or vector boolean attribute of the given type.
static Attribute getBoolAttribute(Type type, MLIRContext *ctx, bool value) {
  auto boolAttr = BoolAttr::get(ctx, value);
  ShapedType shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return boolAttr;
  return DenseElementsAttr::get(shapedType, boolAttr);
}

OpFoldResult ICmpOp::fold(FoldAdaptor adaptor) {
  if (getPredicate() != ICmpPredicate::eq &&
      getPredicate() != ICmpPredicate::ne)
    return {};

  // cmpi(eq/ne, x, x) -> true/false
  if (getLhs() == getRhs())
    return getBoolAttribute(getType(), getContext(),
                            getPredicate() == ICmpPredicate::eq);

  // cmpi(eq/ne, alloca, null) -> false/true
  if (getLhs().getDefiningOp<AllocaOp>() && getRhs().getDefiningOp<ZeroOp>())
    return getBoolAttribute(getType(), getContext(),
                            getPredicate() == ICmpPredicate::ne);

  // cmpi(eq/ne, null, alloca) -> cmpi(eq/ne, alloca, null)
  if (getLhs().getDefiningOp<ZeroOp>() && getRhs().getDefiningOp<AllocaOp>()) {
    Value lhs = getLhs();
    Value rhs = getRhs();
    getLhsMutable().assign(rhs);
    getRhsMutable().assign(lhs);
    return getResult();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Printing, parsing and verification for LLVM::AllocaOp.
//===----------------------------------------------------------------------===//

void AllocaOp::print(OpAsmPrinter &p) {
  auto funcTy =
      FunctionType::get(getContext(), {getArraySize().getType()}, {getType()});

  if (getInalloca())
    p << " inalloca";

  p << ' ' << getArraySize() << " x " << getElemType();
  if (getAlignment() && *getAlignment() != 0)
    p.printOptionalAttrDict((*this)->getAttrs(),
                            {kElemTypeAttrName, getInallocaAttrName()});
  else
    p.printOptionalAttrDict(
        (*this)->getAttrs(),
        {getAlignmentAttrName(), kElemTypeAttrName, getInallocaAttrName()});
  p << " : " << funcTy;
}

// <operation> ::= `llvm.alloca` `inalloca`? ssa-use `x` type
//                  attribute-dict? `:` type `,` type
ParseResult AllocaOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand arraySize;
  Type type, elemType;
  SMLoc trailingTypeLoc;

  if (succeeded(parser.parseOptionalKeyword("inalloca")))
    result.addAttribute(getInallocaAttrName(result.name),
                        UnitAttr::get(parser.getContext()));

  if (parser.parseOperand(arraySize) || parser.parseKeyword("x") ||
      parser.parseType(elemType) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  std::optional<NamedAttribute> alignmentAttr =
      result.attributes.getNamed("alignment");
  if (alignmentAttr.has_value()) {
    auto alignmentInt = llvm::dyn_cast<IntegerAttr>(alignmentAttr->getValue());
    if (!alignmentInt)
      return parser.emitError(parser.getNameLoc(),
                              "expected integer alignment");
    if (alignmentInt.getValue().isZero())
      result.attributes.erase("alignment");
  }

  // Extract the result type from the trailing function type.
  auto funcType = llvm::dyn_cast<FunctionType>(type);
  if (!funcType || funcType.getNumInputs() != 1 ||
      funcType.getNumResults() != 1)
    return parser.emitError(
        trailingTypeLoc,
        "expected trailing function type with one argument and one result");

  if (parser.resolveOperand(arraySize, funcType.getInput(0), result.operands))
    return failure();

  Type resultType = funcType.getResult(0);
  if (auto ptrResultType = llvm::dyn_cast<LLVMPointerType>(resultType))
    result.addAttribute(kElemTypeAttrName, TypeAttr::get(elemType));

  result.addTypes({funcType.getResult(0)});
  return success();
}

LogicalResult AllocaOp::verify() {
  // Only certain target extension types can be used in 'alloca'.
  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getElemType());
      targetExtType && !targetExtType.supportsMemOps())
    return emitOpError()
           << "this target extension type cannot be used in alloca";

  return success();
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

void CondBrOp::build(OpBuilder &builder, OperationState &result,
                     Value condition, Block *trueDest, ValueRange trueOperands,
                     Block *falseDest, ValueRange falseOperands,
                     std::optional<std::pair<uint32_t, uint32_t>> weights) {
  DenseI32ArrayAttr weightsAttr;
  if (weights)
    weightsAttr =
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(weights->first),
                                      static_cast<int32_t>(weights->second)});

  build(builder, result, condition, trueOperands, falseOperands, weightsAttr,
        /*loop_annotation=*/{}, trueDest, falseDest);
}

//===----------------------------------------------------------------------===//
// LLVM::SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     DenseIntElementsAttr caseValues,
                     BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseI32ArrayAttr weightsAttr;
  if (!branchWeights.empty())
    weightsAttr = builder.getDenseI32ArrayAttr(branchWeights);

  build(builder, result, value, defaultOperands, caseOperands, caseValues,
        weightsAttr, defaultDestination, caseDestinations);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<APInt> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }

  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands, branchWeights);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<int32_t> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }

  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands, branchWeights);
}

/// <cases> ::= `[` (case (`,` case )* )? `]`
/// <case>  ::= integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
static ParseResult parseSwitchOpCases(
    OpAsmParser &parser, Type flagType, DenseIntElementsAttr &caseValues,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &caseOperands,
    SmallVectorImpl<SmallVector<Type>> &caseOperandTypes) {
  if (failed(parser.parseLSquare()))
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  SmallVector<APInt> values;
  unsigned bitWidth = flagType.getIntOrFloatBitWidth();
  auto parseCase = [&]() {
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();
    values.push_back(APInt(bitWidth, value, /*isSigned=*/true));

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
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(parseCase)))
    return failure();

  ShapedType caseValueType =
      VectorType::get(static_cast<int64_t>(values.size()), flagType);
  caseValues = DenseIntElementsAttr::get(caseValueType, values);
  return parser.parseRSquare();
}

static void printSwitchOpCases(OpAsmPrinter &p, SwitchOp op, Type flagType,
                               DenseIntElementsAttr caseValues,
                               SuccessorRange caseDestinations,
                               OperandRangeRange caseOperands,
                               const TypeRangeRange &caseOperandTypes) {
  p << '[';
  p.printNewline();
  if (!caseValues) {
    p << ']';
    return;
  }

  size_t index = 0;
  llvm::interleave(
      llvm::zip(caseValues, caseDestinations),
      [&](auto i) {
        p << "  ";
        p << std::get<0>(i);
        p << ": ";
        p.printSuccessorAndUseList(std::get<1>(i), caseOperands[index++]);
      },
      [&] {
        p << ',';
        p.printNewline();
      });
  p.printNewline();
  p << ']';
}

LogicalResult SwitchOp::verify() {
  if ((!getCaseValues() && !getCaseDestinations().empty()) ||
      (getCaseValues() &&
       getCaseValues()->size() !=
           static_cast<int64_t>(getCaseDestinations().size())))
    return emitOpError("expects number of case values to match number of "
                       "case destinations");
  if (getCaseValues() &&
      getValue().getType() != getCaseValues()->getElementType())
    return emitError("expects case value type to match condition value type");
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

constexpr int32_t GEPOp::kDynamicIndex;

GEPIndicesAdaptor<ValueRange> GEPOp::getIndices() {
  return GEPIndicesAdaptor<ValueRange>(getRawConstantIndicesAttr(),
                                       getDynamicIndices());
}

/// Returns the elemental type of any LLVM-compatible vector type or self.
static Type extractVectorElementType(Type type) {
  if (auto vectorType = llvm::dyn_cast<VectorType>(type))
    return vectorType.getElementType();
  return type;
}

/// Destructures the 'indices' parameter into 'rawConstantIndices' and
/// 'dynamicIndices', encoding the former in the process. In the process,
/// dynamic indices which are used to index into a structure type are converted
/// to constant indices when possible. To do this, the GEPs element type should
/// be passed as first parameter.
static void destructureIndices(Type currType, ArrayRef<GEPArg> indices,
                               SmallVectorImpl<int32_t> &rawConstantIndices,
                               SmallVectorImpl<Value> &dynamicIndices) {
  for (const GEPArg &iter : indices) {
    // If the thing we are currently indexing into is a struct we must turn
    // any integer constants into constant indices. If this is not possible
    // we don't do anything here. The verifier will catch it and emit a proper
    // error. All other canonicalization is done in the fold method.
    bool requiresConst = !rawConstantIndices.empty() &&
                         isa_and_nonnull<LLVMStructType>(currType);
    if (Value val = llvm::dyn_cast_if_present<Value>(iter)) {
      APInt intC;
      if (requiresConst && matchPattern(val, m_ConstantInt(&intC)) &&
          intC.isSignedIntN(kGEPConstantBitWidth)) {
        rawConstantIndices.push_back(intC.getSExtValue());
      } else {
        rawConstantIndices.push_back(GEPOp::kDynamicIndex);
        dynamicIndices.push_back(val);
      }
    } else {
      rawConstantIndices.push_back(cast<GEPConstantIndex>(iter));
    }

    // Skip for very first iteration of this loop. First index does not index
    // within the aggregates, but is just a pointer offset.
    if (rawConstantIndices.size() == 1 || !currType)
      continue;

    currType = TypeSwitch<Type, Type>(currType)
                   .Case<VectorType, LLVMArrayType>([](auto containerType) {
                     return containerType.getElementType();
                   })
                   .Case([&](LLVMStructType structType) -> Type {
                     int64_t memberIndex = rawConstantIndices.back();
                     if (memberIndex >= 0 && static_cast<size_t>(memberIndex) <
                                                 structType.getBody().size())
                       return structType.getBody()[memberIndex];
                     return nullptr;
                   })
                   .Default(Type(nullptr));
  }
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Type elementType, Value basePtr, ArrayRef<GEPArg> indices,
                  GEPNoWrapFlags noWrapFlags,
                  ArrayRef<NamedAttribute> attributes) {
  SmallVector<int32_t> rawConstantIndices;
  SmallVector<Value> dynamicIndices;
  destructureIndices(elementType, indices, rawConstantIndices, dynamicIndices);

  result.addTypes(resultType);
  result.addAttributes(attributes);
  result.getOrAddProperties<Properties>().rawConstantIndices =
      builder.getDenseI32ArrayAttr(rawConstantIndices);
  result.getOrAddProperties<Properties>().noWrapFlags = noWrapFlags;
  result.getOrAddProperties<Properties>().elem_type =
      TypeAttr::get(elementType);
  result.addOperands(basePtr);
  result.addOperands(dynamicIndices);
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Type elementType, Value basePtr, ValueRange indices,
                  GEPNoWrapFlags noWrapFlags,
                  ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultType, elementType, basePtr,
        SmallVector<GEPArg>(indices), noWrapFlags, attributes);
}

static ParseResult
parseGEPIndices(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
                DenseI32ArrayAttr &rawConstantIndices) {
  SmallVector<int32_t> constantIndices;

  auto idxParser = [&]() -> ParseResult {
    int32_t constantIndex;
    OptionalParseResult parsedInteger =
        parser.parseOptionalInteger(constantIndex);
    if (parsedInteger.has_value()) {
      if (failed(parsedInteger.value()))
        return failure();
      constantIndices.push_back(constantIndex);
      return success();
    }

    constantIndices.push_back(LLVM::GEPOp::kDynamicIndex);
    return parser.parseOperand(indices.emplace_back());
  };
  if (parser.parseCommaSeparatedList(idxParser))
    return failure();

  rawConstantIndices =
      DenseI32ArrayAttr::get(parser.getContext(), constantIndices);
  return success();
}

static void printGEPIndices(OpAsmPrinter &printer, LLVM::GEPOp gepOp,
                            OperandRange indices,
                            DenseI32ArrayAttr rawConstantIndices) {
  llvm::interleaveComma(
      GEPIndicesAdaptor<OperandRange>(rawConstantIndices, indices), printer,
      [&](PointerUnion<IntegerAttr, Value> cst) {
        if (Value val = llvm::dyn_cast_if_present<Value>(cst))
          printer.printOperand(val);
        else
          printer << cast<IntegerAttr>(cst).getInt();
      });
}

/// For the given `indices`, check if they comply with `baseGEPType`,
/// especially check against LLVMStructTypes nested within.
static LogicalResult
verifyStructIndices(Type baseGEPType, unsigned indexPos,
                    GEPIndicesAdaptor<ValueRange> indices,
                    function_ref<InFlightDiagnostic()> emitOpError) {
  if (indexPos >= indices.size())
    // Stop searching
    return success();

  return TypeSwitch<Type, LogicalResult>(baseGEPType)
      .Case<LLVMStructType>([&](LLVMStructType structType) -> LogicalResult {
        auto attr = dyn_cast<IntegerAttr>(indices[indexPos]);
        if (!attr)
          return emitOpError() << "expected index " << indexPos
                               << " indexing a struct to be constant";

        int32_t gepIndex = attr.getInt();
        ArrayRef<Type> elementTypes = structType.getBody();
        if (gepIndex < 0 ||
            static_cast<size_t>(gepIndex) >= elementTypes.size())
          return emitOpError() << "index " << indexPos
                               << " indexing a struct is out of bounds";

        // Instead of recursively going into every children types, we only
        // dive into the one indexed by gepIndex.
        return verifyStructIndices(elementTypes[gepIndex], indexPos + 1,
                                   indices, emitOpError);
      })
      .Case<VectorType, LLVMArrayType>(
          [&](auto containerType) -> LogicalResult {
            return verifyStructIndices(containerType.getElementType(),
                                       indexPos + 1, indices, emitOpError);
          })
      .Default([&](auto otherType) -> LogicalResult {
        return emitOpError()
               << "type " << otherType << " cannot be indexed (index #"
               << indexPos << ")";
      });
}

/// Driver function around `verifyStructIndices`.
static LogicalResult
verifyStructIndices(Type baseGEPType, GEPIndicesAdaptor<ValueRange> indices,
                    function_ref<InFlightDiagnostic()> emitOpError) {
  return verifyStructIndices(baseGEPType, /*indexPos=*/1, indices, emitOpError);
}

LogicalResult LLVM::GEPOp::verify() {
  if (static_cast<size_t>(
          llvm::count(getRawConstantIndices(), kDynamicIndex)) !=
      getDynamicIndices().size())
    return emitOpError("expected as many dynamic indices as specified in '")
           << getRawConstantIndicesAttrName().getValue() << "'";

  if (getNoWrapFlags() == GEPNoWrapFlags::inboundsFlag)
    return emitOpError("'inbounds_flag' cannot be used directly.");

  return verifyStructIndices(getElemType(), getIndices(),
                             [&] { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAddrMutable());
  // Volatile operations can have target-specific read-write effects on
  // memory besides the one referred to by the pointer operand.
  // Similarly, atomic operations that are monotonic or stricter cause
  // synchronization that from a language point-of-view, are arbitrary
  // read-writes into memory.
  if (getVolatile_() || (getOrdering() != AtomicOrdering::not_atomic &&
                         getOrdering() != AtomicOrdering::unordered)) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

/// Returns true if the given type is supported by atomic operations. All
/// integer, float, and pointer types with a power-of-two bitsize and a minimal
/// size of 8 bits are supported.
static bool isTypeCompatibleWithAtomicOp(Type type,
                                         const DataLayout &dataLayout) {
  if (!isa<IntegerType, LLVMPointerType>(type))
    if (!isCompatibleFloatingPointType(type))
      return false;

  llvm::TypeSize bitWidth = dataLayout.getTypeSizeInBits(type);
  if (bitWidth.isScalable())
    return false;
  // Needs to be at least 8 bits and a power of two.
  return bitWidth >= 8 && (bitWidth & (bitWidth - 1)) == 0;
}

/// Verifies the attributes and the type of atomic memory access operations.
template <typename OpTy>
LogicalResult verifyAtomicMemOp(OpTy memOp, Type valueType,
                                ArrayRef<AtomicOrdering> unsupportedOrderings) {
  if (memOp.getOrdering() != AtomicOrdering::not_atomic) {
    DataLayout dataLayout = DataLayout::closest(memOp);
    if (!isTypeCompatibleWithAtomicOp(valueType, dataLayout))
      return memOp.emitOpError("unsupported type ")
             << valueType << " for atomic access";
    if (llvm::is_contained(unsupportedOrderings, memOp.getOrdering()))
      return memOp.emitOpError("unsupported ordering '")
             << stringifyAtomicOrdering(memOp.getOrdering()) << "'";
    if (!memOp.getAlignment())
      return memOp.emitOpError("expected alignment for atomic access");
    return success();
  }
  if (memOp.getSyncscope())
    return memOp.emitOpError(
        "expected syncscope to be null for non-atomic access");
  return success();
}

LogicalResult LoadOp::verify() {
  Type valueType = getResult().getType();
  return verifyAtomicMemOp(*this, valueType,
                           {AtomicOrdering::release, AtomicOrdering::acq_rel});
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Type type,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal, bool isInvariant, bool isInvariantGroup,
                   AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, type, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, isInvariant, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope),
        /*dereferenceable=*/nullptr,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr,
        /*tbaa=*/nullptr);
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getAddrMutable());
  // Volatile operations can have target-specific read-write effects on
  // memory besides the one referred to by the pointer operand.
  // Similarly, atomic operations that are monotonic or stricter cause
  // synchronization that from a language point-of-view, are arbitrary
  // read-writes into memory.
  if (getVolatile_() || (getOrdering() != AtomicOrdering::not_atomic &&
                         getOrdering() != AtomicOrdering::unordered)) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

LogicalResult StoreOp::verify() {
  Type valueType = getValue().getType();
  return verifyAtomicMemOp(*this, valueType,
                           {AtomicOrdering::acquire, AtomicOrdering::acq_rel});
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal, bool isInvariantGroup,
                    AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, value, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope),
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Gets the MLIR Op-like result types of a LLVMFunctionType.
static SmallVector<Type, 1> getCallOpResultTypes(LLVMFunctionType calleeType) {
  SmallVector<Type, 1> results;
  Type resultType = calleeType.getReturnType();
  if (!isa<LLVM::LLVMVoidType>(resultType))
    results.push_back(resultType);
  return results;
}

/// Gets the variadic callee type for a LLVMFunctionType.
static TypeAttr getCallOpVarCalleeType(LLVMFunctionType calleeType) {
  return calleeType.isVarArg() ? TypeAttr::get(calleeType) : nullptr;
}

/// Constructs a LLVMFunctionType from MLIR `results` and `args`.
static LLVMFunctionType getLLVMFuncType(MLIRContext *context, TypeRange results,
                                        ValueRange args) {
  Type resultType;
  if (results.empty())
    resultType = LLVMVoidType::get(context);
  else
    resultType = results.front();
  return LLVMFunctionType::get(resultType, llvm::to_vector(args.getTypes()),
                               /*isVarArg=*/false);
}

void CallOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                   StringRef callee, ValueRange args) {
  build(builder, state, results, builder.getStringAttr(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                   StringAttr callee, ValueRange args) {
  build(builder, state, results, SymbolRefAttr::get(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state, TypeRange results,
                   FlatSymbolRefAttr callee, ValueRange args) {
  assert(callee && "expected non-null callee in direct call builder");
  build(builder, state, results,
        /*var_callee_type=*/nullptr, callee, args, /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr, /*TailCallKind=*/nullptr,
        /*memory_effects=*/nullptr,
        /*convergent=*/nullptr, /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
        /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, StringRef callee,
                   ValueRange args) {
  build(builder, state, calleeType, builder.getStringAttr(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, StringAttr callee,
                   ValueRange args) {
  build(builder, state, calleeType, SymbolRefAttr::get(callee), args);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, FlatSymbolRefAttr callee,
                   ValueRange args) {
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), callee, args,
        /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr,
        /*TailCallKind=*/nullptr, /*memory_effects=*/nullptr,
        /*convergent=*/nullptr,
        /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   LLVMFunctionType calleeType, ValueRange args) {
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType),
        /*callee=*/nullptr, args,
        /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr, /*TailCallKind=*/nullptr, /*memory_effects=*/nullptr,
        /*convergent=*/nullptr, /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
        /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

void CallOp::build(OpBuilder &builder, OperationState &state, LLVMFuncOp func,
                   ValueRange args) {
  auto calleeType = func.getFunctionType();
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), SymbolRefAttr::get(func), args,
        /*fastmathFlags=*/nullptr,
        /*CConv=*/nullptr, /*TailCallKind=*/nullptr, /*memory_effects=*/nullptr,
        /*convergent=*/nullptr, /*no_unwind=*/nullptr, /*will_return=*/nullptr,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{},
        /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr,
        /*no_inline=*/nullptr, /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr);
}

CallInterfaceCallable CallOp::getCallableForCallee() {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr())
    return calleeAttr;
  // Indirect call, callee Value is the first operand.
  return getOperand(0);
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = cast<SymbolRefAttr>(callee);
    return setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, cast<Value>(callee));
}

Operation::operand_range CallOp::getArgOperands() {
  return getCalleeOperands().drop_front(getCallee().has_value() ? 0 : 1);
}

MutableOperandRange CallOp::getArgOperandsMutable() {
  return MutableOperandRange(*this, getCallee().has_value() ? 0 : 1,
                             getCalleeOperands().size());
}

/// Verify that an inlinable callsite of a debug-info-bearing function in a
/// debug-info-bearing function has a debug location attached to it. This
/// mirrors an LLVM IR verifier.
static LogicalResult verifyCallOpDebugInfo(CallOp callOp, LLVMFuncOp callee) {
  if (callee.isExternal())
    return success();
  auto parentFunc = callOp->getParentOfType<FunctionOpInterface>();
  if (!parentFunc)
    return success();

  auto hasSubprogram = [](Operation *op) {
    return op->getLoc()
               ->findInstanceOf<FusedLocWith<LLVM::DISubprogramAttr>>() !=
           nullptr;
  };
  if (!hasSubprogram(parentFunc) || !hasSubprogram(callee))
    return success();
  bool containsLoc = !isa<UnknownLoc>(callOp->getLoc());
  if (!containsLoc)
    return callOp.emitError()
           << "inlinable function call in a function with a DISubprogram "
              "location must have a debug location";
  return success();
}

/// Verify that the parameter and return types of the variadic callee type match
/// the `callOp` argument and result types.
template <typename OpTy>
LogicalResult verifyCallOpVarCalleeType(OpTy callOp) {
  std::optional<LLVMFunctionType> varCalleeType = callOp.getVarCalleeType();
  if (!varCalleeType)
    return success();

  // Verify the variadic callee type is a variadic function type.
  if (!varCalleeType->isVarArg())
    return callOp.emitOpError(
        "expected var_callee_type to be a variadic function type");

  // Verify the variadic callee type has at most as many parameters as the call
  // has argument operands.
  if (varCalleeType->getNumParams() > callOp.getArgOperands().size())
    return callOp.emitOpError("expected var_callee_type to have at most ")
           << callOp.getArgOperands().size() << " parameters";

  // Verify the variadic callee type matches the call argument types.
  for (auto [paramType, operand] :
       llvm::zip(varCalleeType->getParams(), callOp.getArgOperands()))
    if (paramType != operand.getType())
      return callOp.emitOpError()
             << "var_callee_type parameter type mismatch: " << paramType
             << " != " << operand.getType();

  // Verify the variadic callee type matches the call result type.
  if (!callOp.getNumResults()) {
    if (!isa<LLVMVoidType>(varCalleeType->getReturnType()))
      return callOp.emitOpError("expected var_callee_type to return void");
  } else {
    if (callOp.getResult().getType() != varCalleeType->getReturnType())
      return callOp.emitOpError("var_callee_type return type mismatch: ")
             << varCalleeType->getReturnType()
             << " != " << callOp.getResult().getType();
  }
  return success();
}

template <typename OpType>
static LogicalResult verifyOperandBundles(OpType &op) {
  OperandRangeRange opBundleOperands = op.getOpBundleOperands();
  std::optional<ArrayAttr> opBundleTags = op.getOpBundleTags();

  auto isStringAttr = [](Attribute tagAttr) {
    return isa<StringAttr>(tagAttr);
  };
  if (opBundleTags && !llvm::all_of(*opBundleTags, isStringAttr))
    return op.emitError("operand bundle tag must be a StringAttr");

  size_t numOpBundles = opBundleOperands.size();
  size_t numOpBundleTags = opBundleTags ? opBundleTags->size() : 0;
  if (numOpBundles != numOpBundleTags)
    return op.emitError("expected ")
           << numOpBundles << " operand bundle tags, but actually got "
           << numOpBundleTags;

  return success();
}

LogicalResult CallOp::verify() { return verifyOperandBundles(*this); }

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyCallOpVarCalleeType(*this)))
    return failure();

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
    auto ptrType = llvm::dyn_cast<LLVMPointerType>(getOperand(0).getType());
    if (!ptrType)
      return emitOpError("indirect call expects a pointer as callee: ")
             << getOperand(0).getType();

    return success();
  } else {
    Operation *callee =
        symbolTable.lookupNearestSymbolFrom(*this, calleeName.getAttr());
    if (!callee)
      return emitOpError()
             << "'" << calleeName.getValue()
             << "' does not reference a symbol in the current scope";
    if (auto fn = dyn_cast<LLVMFuncOp>(callee)) {
      if (failed(verifyCallOpDebugInfo(*this, fn)))
        return failure();
      fnType = fn.getFunctionType();
    } else if (auto ifunc = dyn_cast<IFuncOp>(callee)) {
      fnType = ifunc.getIFuncType();
    } else {
      return emitOpError()
             << "'" << calleeName.getValue()
             << "' does not reference a valid LLVM function or IFunc";
    }
  }

  LLVMFunctionType funcType = llvm::dyn_cast<LLVMFunctionType>(fnType);
  if (!funcType)
    return emitOpError("callee does not have a functional type: ") << fnType;

  if (funcType.isVarArg() && !getVarCalleeType())
    return emitOpError() << "missing var_callee_type attribute for vararg call";

  // Verify that the operand and result types match the callee.

  if (!funcType.isVarArg() &&
      funcType.getNumParams() != (getCalleeOperands().size() - isIndirect))
    return emitOpError() << "incorrect number of operands ("
                         << (getCalleeOperands().size() - isIndirect)
                         << ") for callee (expecting: "
                         << funcType.getNumParams() << ")";

  if (funcType.getNumParams() > (getCalleeOperands().size() - isIndirect))
    return emitOpError() << "incorrect number of operands ("
                         << (getCalleeOperands().size() - isIndirect)
                         << ") for varargs callee (expecting at least: "
                         << funcType.getNumParams() << ")";

  for (unsigned i = 0, e = funcType.getNumParams(); i != e; ++i)
    if (getOperand(i + isIndirect).getType() != funcType.getParamType(i))
      return emitOpError() << "operand type mismatch for operand " << i << ": "
                           << getOperand(i + isIndirect).getType()
                           << " != " << funcType.getParamType(i);

  if (getNumResults() == 0 &&
      !llvm::isa<LLVM::LLVMVoidType>(funcType.getReturnType()))
    return emitOpError() << "expected function call to produce a value";

  if (getNumResults() != 0 &&
      llvm::isa<LLVM::LLVMVoidType>(funcType.getReturnType()))
    return emitOpError()
           << "calling function with void result must not produce values";

  if (getNumResults() > 1)
    return emitOpError()
           << "expected LLVM function call to produce 0 or 1 result";

  if (getNumResults() && getResult().getType() != funcType.getReturnType())
    return emitOpError() << "result type mismatch: " << getResult().getType()
                         << " != " << funcType.getReturnType();

  return success();
}

void CallOp::print(OpAsmPrinter &p) {
  auto callee = getCallee();
  bool isDirect = callee.has_value();

  p << ' ';

  // Print calling convention.
  if (getCConv() != LLVM::CConv::C)
    p << stringifyCConv(getCConv()) << ' ';

  if (getTailCallKind() != LLVM::TailCallKind::None)
    p << tailcallkind::stringifyTailCallKind(getTailCallKind()) << ' ';

  // Print the direct callee if present as a function attribute, or an indirect
  // callee (first operand) otherwise.
  if (isDirect)
    p.printSymbolName(callee.value());
  else
    p << getOperand(0);

  auto args = getCalleeOperands().drop_front(isDirect ? 0 : 1);
  p << '(' << args << ')';

  // Print the variadic callee type if the call is variadic.
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    p << " vararg(" << *varCalleeType << ")";

  if (!getOpBundleOperands().empty()) {
    p << " ";
    printOpBundles(p, *this, getOpBundleOperands(),
                   getOpBundleOperands().getTypes(), getOpBundleTags());
  }

  p.printOptionalAttrDict(processFMFAttr((*this)->getAttrs()),
                          {getCalleeAttrName(), getTailCallKindAttrName(),
                           getVarCalleeTypeAttrName(), getCConvAttrName(),
                           getOperandSegmentSizesAttrName(),
                           getOpBundleSizesAttrName(),
                           getOpBundleTagsAttrName(), getArgAttrsAttrName(),
                           getResAttrsAttrName()});

  p << " : ";
  if (!isDirect)
    p << getOperand(0).getType() << ", ";

  // Reconstruct the MLIR function type from operand and result types.
  call_interface_impl::printFunctionSignature(
      p, args.getTypes(), getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

/// Parses the type of a call operation and resolves the operands if the parsing
/// succeeds. Returns failure otherwise.
static ParseResult parseCallTypeAndResolveOperands(
    OpAsmParser &parser, OperationState &result, bool isDirect,
    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    SmallVectorImpl<DictionaryAttr> &argAttrs,
    SmallVectorImpl<DictionaryAttr> &resultAttrs) {
  SMLoc trailingTypesLoc = parser.getCurrentLocation();
  SmallVector<Type> types;
  if (parser.parseColon())
    return failure();
  if (!isDirect) {
    types.emplace_back();
    if (parser.parseType(types.back()))
      return failure();
    if (parser.parseOptionalComma())
      return parser.emitError(
          trailingTypesLoc, "expected indirect call to have 2 trailing types");
  }
  SmallVector<Type> argTypes;
  SmallVector<Type> resTypes;
  if (call_interface_impl::parseFunctionSignature(parser, argTypes, argAttrs,
                                                  resTypes, resultAttrs)) {
    if (isDirect)
      return parser.emitError(trailingTypesLoc,
                              "expected direct call to have 1 trailing types");
    return parser.emitError(trailingTypesLoc,
                            "expected trailing function type");
  }

  if (resTypes.size() > 1)
    return parser.emitError(trailingTypesLoc,
                            "expected function with 0 or 1 result");
  if (resTypes.size() == 1 && llvm::isa<LLVM::LLVMVoidType>(resTypes[0]))
    return parser.emitError(trailingTypesLoc,
                            "expected a non-void result type");

  // The head element of the types list matches the callee type for
  // indirect calls, while the types list is emtpy for direct calls.
  // Append the function input types to resolve the call operation
  // operands.
  llvm::append_range(types, argTypes);
  if (parser.resolveOperands(operands, types, parser.getNameLoc(),
                             result.operands))
    return failure();
  if (resTypes.size() != 0)
    result.addTypes(resTypes);

  return success();
}

/// Parses an optional function pointer operand before the call argument list
/// for indirect calls, or stops parsing at the function identifier otherwise.
static ParseResult parseOptionalCallFuncPtr(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands) {
  OpAsmParser::UnresolvedOperand funcPtrOperand;
  OptionalParseResult parseResult = parser.parseOptionalOperand(funcPtrOperand);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return *parseResult;
    operands.push_back(funcPtrOperand);
  }
  return success();
}

static ParseResult resolveOpBundleOperands(
    OpAsmParser &parser, SMLoc loc, OperationState &state,
    ArrayRef<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands,
    ArrayRef<SmallVector<Type>> opBundleOperandTypes,
    StringAttr opBundleSizesAttrName) {
  unsigned opBundleIndex = 0;
  for (const auto &[operands, types] :
       llvm::zip_equal(opBundleOperands, opBundleOperandTypes)) {
    if (operands.size() != types.size())
      return parser.emitError(loc, "expected ")
             << operands.size()
             << " types for operand bundle operands for operand bundle #"
             << opBundleIndex << ", but actually got " << types.size();
    if (parser.resolveOperands(operands, types, loc, state.operands))
      return failure();
  }

  SmallVector<int32_t> opBundleSizes;
  opBundleSizes.reserve(opBundleOperands.size());
  for (const auto &operands : opBundleOperands)
    opBundleSizes.push_back(operands.size());

  state.addAttribute(
      opBundleSizesAttrName,
      DenseI32ArrayAttr::get(parser.getContext(), opBundleSizes));

  return success();
}

// <operation> ::= `llvm.call` (cconv)? (tailcallkind)? (function-id | ssa-use)
//                             `(` ssa-use-list `)`
//                             ( `vararg(` var-callee-type `)` )?
//                             ( `[` op-bundles-list `]` )?
//                             attribute-dict? `:` (type `,`)? function-type
ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
  SymbolRefAttr funcAttr;
  TypeAttr varCalleeType;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands;
  SmallVector<SmallVector<Type>> opBundleOperandTypes;
  ArrayAttr opBundleTags;

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(),
                     parseOptionalLLVMKeyword<CConv>(parser, LLVM::CConv::C)));

  result.addAttribute(
      getTailCallKindAttrName(result.name),
      TailCallKindAttr::get(parser.getContext(),
                            parseOptionalLLVMKeyword<TailCallKind>(
                                parser, LLVM::TailCallKind::None)));

  // Parse a function pointer for indirect calls.
  if (parseOptionalCallFuncPtr(parser, operands))
    return failure();
  bool isDirect = operands.empty();

  // Parse a function identifier for direct calls.
  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", result.attributes))
      return failure();

  // Parse the function arguments.
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren))
    return failure();

  bool isVarArg = parser.parseOptionalKeyword("vararg").succeeded();
  if (isVarArg) {
    StringAttr varCalleeTypeAttrName =
        CallOp::getVarCalleeTypeAttrName(result.name);
    if (parser.parseLParen().failed() ||
        parser
            .parseAttribute(varCalleeType, varCalleeTypeAttrName,
                            result.attributes)
            .failed() ||
        parser.parseRParen().failed())
      return failure();
  }

  SMLoc opBundlesLoc = parser.getCurrentLocation();
  if (std::optional<ParseResult> result = parseOpBundles(
          parser, opBundleOperands, opBundleOperandTypes, opBundleTags);
      result && failed(*result))
    return failure();
  if (opBundleTags && !opBundleTags.empty())
    result.addAttribute(CallOp::getOpBundleTagsAttrName(result.name).getValue(),
                        opBundleTags);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the trailing type list and resolve the operands.
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parseCallTypeAndResolveOperands(parser, result, isDirect, operands,
                                      argAttrs, resultAttrs))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  if (resolveOpBundleOperands(parser, opBundlesLoc, result, opBundleOperands,
                              opBundleOperandTypes,
                              getOpBundleSizesAttrName(result.name)))
    return failure();

  int32_t numOpBundleOperands = 0;
  for (const auto &operands : opBundleOperands)
    numOpBundleOperands += operands.size();

  result.addAttribute(
      CallOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(operands.size()), numOpBundleOperands}));
  return success();
}

LLVMFunctionType CallOp::getCalleeFunctionType() {
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    return *varCalleeType;
  return getLLVMFuncType(getContext(), getResultTypes(), getArgOperands());
}

///===---------------------------------------------------------------------===//
/// LLVM::InvokeOp
///===---------------------------------------------------------------------===//

void InvokeOp::build(OpBuilder &builder, OperationState &state, LLVMFuncOp func,
                     ValueRange ops, Block *normal, ValueRange normalOps,
                     Block *unwind, ValueRange unwindOps) {
  auto calleeType = func.getFunctionType();
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), SymbolRefAttr::get(func), ops,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr, normalOps, unwindOps,
        nullptr, nullptr, {}, {}, normal, unwind);
}

void InvokeOp::build(OpBuilder &builder, OperationState &state, TypeRange tys,
                     FlatSymbolRefAttr callee, ValueRange ops, Block *normal,
                     ValueRange normalOps, Block *unwind,
                     ValueRange unwindOps) {
  build(builder, state, tys,
        /*var_callee_type=*/nullptr, callee, ops, /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr, normalOps, unwindOps, nullptr, nullptr, {}, {},
        normal, unwind);
}

void InvokeOp::build(OpBuilder &builder, OperationState &state,
                     LLVMFunctionType calleeType, FlatSymbolRefAttr callee,
                     ValueRange ops, Block *normal, ValueRange normalOps,
                     Block *unwind, ValueRange unwindOps) {
  build(builder, state, getCallOpResultTypes(calleeType),
        getCallOpVarCalleeType(calleeType), callee, ops,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr, normalOps, unwindOps,
        nullptr, nullptr, {}, {}, normal, unwind);
}

SuccessorOperands InvokeOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getNormalDestOperandsMutable()
                                      : getUnwindDestOperandsMutable());
}

CallInterfaceCallable InvokeOp::getCallableForCallee() {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr())
    return calleeAttr;
  // Indirect call, callee Value is the first operand.
  return getOperand(0);
}

void InvokeOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  // Direct call.
  if (FlatSymbolRefAttr calleeAttr = getCalleeAttr()) {
    auto symRef = cast<SymbolRefAttr>(callee);
    return setCalleeAttr(cast<FlatSymbolRefAttr>(symRef));
  }
  // Indirect call, callee Value is the first operand.
  return setOperand(0, cast<Value>(callee));
}

Operation::operand_range InvokeOp::getArgOperands() {
  return getCalleeOperands().drop_front(getCallee().has_value() ? 0 : 1);
}

MutableOperandRange InvokeOp::getArgOperandsMutable() {
  return MutableOperandRange(*this, getCallee().has_value() ? 0 : 1,
                             getCalleeOperands().size());
}

LogicalResult InvokeOp::verify() {
  if (failed(verifyCallOpVarCalleeType(*this)))
    return failure();

  Block *unwindDest = getUnwindDest();
  if (unwindDest->empty())
    return emitError("must have at least one operation in unwind destination");

  // In unwind destination, first operation must be LandingpadOp
  if (!isa<LandingpadOp>(unwindDest->front()))
    return emitError("first operation in unwind destination should be a "
                     "llvm.landingpad operation");

  if (failed(verifyOperandBundles(*this)))
    return failure();

  return success();
}

void InvokeOp::print(OpAsmPrinter &p) {
  auto callee = getCallee();
  bool isDirect = callee.has_value();

  p << ' ';

  // Print calling convention.
  if (getCConv() != LLVM::CConv::C)
    p << stringifyCConv(getCConv()) << ' ';

  // Either function name or pointer
  if (isDirect)
    p.printSymbolName(callee.value());
  else
    p << getOperand(0);

  p << '(' << getCalleeOperands().drop_front(isDirect ? 0 : 1) << ')';
  p << " to ";
  p.printSuccessorAndUseList(getNormalDest(), getNormalDestOperands());
  p << " unwind ";
  p.printSuccessorAndUseList(getUnwindDest(), getUnwindDestOperands());

  // Print the variadic callee type if the invoke is variadic.
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    p << " vararg(" << *varCalleeType << ")";

  if (!getOpBundleOperands().empty()) {
    p << " ";
    printOpBundles(p, *this, getOpBundleOperands(),
                   getOpBundleOperands().getTypes(), getOpBundleTags());
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getCalleeAttrName(), getOperandSegmentSizeAttr(),
                           getCConvAttrName(), getVarCalleeTypeAttrName(),
                           getOpBundleSizesAttrName(),
                           getOpBundleTagsAttrName(), getArgAttrsAttrName(),
                           getResAttrsAttrName()});

  p << " : ";
  if (!isDirect)
    p << getOperand(0).getType() << ", ";
  call_interface_impl::printFunctionSignature(
      p, getCalleeOperands().drop_front(isDirect ? 0 : 1).getTypes(),
      getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

// <operation> ::= `llvm.invoke` (cconv)? (function-id | ssa-use)
//                  `(` ssa-use-list `)`
//                  `to` bb-id (`[` ssa-use-and-type-list `]`)?
//                  `unwind` bb-id (`[` ssa-use-and-type-list `]`)?
//                  ( `vararg(` var-callee-type `)` )?
//                  ( `[` op-bundles-list `]` )?
//                  attribute-dict? `:` (type `,`)?
//                  function-type-with-argument-attributes
ParseResult InvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SymbolRefAttr funcAttr;
  TypeAttr varCalleeType;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands;
  SmallVector<SmallVector<Type>> opBundleOperandTypes;
  ArrayAttr opBundleTags;
  Block *normalDest, *unwindDest;
  SmallVector<Value, 4> normalOperands, unwindOperands;
  Builder &builder = parser.getBuilder();

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(),
                     parseOptionalLLVMKeyword<CConv>(parser, LLVM::CConv::C)));

  // Parse a function pointer for indirect calls.
  if (parseOptionalCallFuncPtr(parser, operands))
    return failure();
  bool isDirect = operands.empty();

  // Parse a function identifier for direct calls.
  if (isDirect && parser.parseAttribute(funcAttr, "callee", result.attributes))
    return failure();

  // Parse the function arguments.
  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("to") ||
      parser.parseSuccessorAndUseList(normalDest, normalOperands) ||
      parser.parseKeyword("unwind") ||
      parser.parseSuccessorAndUseList(unwindDest, unwindOperands))
    return failure();

  bool isVarArg = parser.parseOptionalKeyword("vararg").succeeded();
  if (isVarArg) {
    StringAttr varCalleeTypeAttrName =
        InvokeOp::getVarCalleeTypeAttrName(result.name);
    if (parser.parseLParen().failed() ||
        parser
            .parseAttribute(varCalleeType, varCalleeTypeAttrName,
                            result.attributes)
            .failed() ||
        parser.parseRParen().failed())
      return failure();
  }

  SMLoc opBundlesLoc = parser.getCurrentLocation();
  if (std::optional<ParseResult> result = parseOpBundles(
          parser, opBundleOperands, opBundleOperandTypes, opBundleTags);
      result && failed(*result))
    return failure();
  if (opBundleTags && !opBundleTags.empty())
    result.addAttribute(
        InvokeOp::getOpBundleTagsAttrName(result.name).getValue(),
        opBundleTags);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the trailing type list and resolve the function operands.
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parseCallTypeAndResolveOperands(parser, result, isDirect, operands,
                                      argAttrs, resultAttrs))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  if (resolveOpBundleOperands(parser, opBundlesLoc, result, opBundleOperands,
                              opBundleOperandTypes,
                              getOpBundleSizesAttrName(result.name)))
    return failure();

  result.addSuccessors({normalDest, unwindDest});
  result.addOperands(normalOperands);
  result.addOperands(unwindOperands);

  int32_t numOpBundleOperands = 0;
  for (const auto &operands : opBundleOperands)
    numOpBundleOperands += operands.size();

  result.addAttribute(
      InvokeOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(operands.size()),
                                    static_cast<int32_t>(normalOperands.size()),
                                    static_cast<int32_t>(unwindOperands.size()),
                                    numOpBundleOperands}));
  return success();
}

LLVMFunctionType InvokeOp::getCalleeFunctionType() {
  if (std::optional<LLVMFunctionType> varCalleeType = getVarCalleeType())
    return *varCalleeType;
  return getLLVMFuncType(getContext(), getResultTypes(), getArgOperands());
}

///===----------------------------------------------------------------------===//
/// Verifying/Printing/Parsing for LLVM::LandingpadOp.
///===----------------------------------------------------------------------===//

LogicalResult LandingpadOp::verify() {
  Value value;
  if (LLVMFuncOp func = (*this)->getParentOfType<LLVMFuncOp>()) {
    if (!func.getPersonality())
      return emitError(
          "llvm.landingpad needs to be in a function with a personality");
  }

  // Consistency of llvm.landingpad result types is checked in
  // LLVMFuncOp::verify().

  if (!getCleanup() && getOperands().empty())
    return emitError("landingpad instruction expects at least one clause or "
                     "cleanup attribute");

  for (unsigned idx = 0, ie = getNumOperands(); idx < ie; idx++) {
    value = getOperand(idx);
    bool isFilter = llvm::isa<LLVMArrayType>(value.getType());
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
      // ZeroOp and AddressOfOp allowed
      if (value.getDefiningOp<ZeroOp>())
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
    bool isArrayTy = llvm::isa<LLVMArrayType>(value.getType());
    p << '(' << (isArrayTy ? "filter " : "catch ") << value << " : "
      << value.getType() << ") ";
  }

  p.printOptionalAttrDict((*this)->getAttrs(), {"cleanup"});

  p << ": " << getType();
}

// <operation> ::= `llvm.landingpad` `cleanup`?
//                 ((`catch` | `filter`) operand-type ssa-use)* attribute-dict?
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
// ExtractValueOp
//===----------------------------------------------------------------------===//

/// Extract the type at `position` in the LLVM IR aggregate type
/// `containerType`. Each element of `position` is an index into a nested
/// aggregate type. Return the resulting type or emit an error.
static Type getInsertExtractValueElementType(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type containerType,
    ArrayRef<int64_t> position) {
  Type llvmType = containerType;
  if (!isCompatibleType(containerType)) {
    emitError("expected LLVM IR Dialect type, got ") << containerType;
    return {};
  }

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // structures.  Check the position index before accessing, it is supposed to
  // be in bounds.
  for (int64_t idx : position) {
    if (auto arrayType = llvm::dyn_cast<LLVMArrayType>(llvmType)) {
      if (idx < 0 || static_cast<unsigned>(idx) >= arrayType.getNumElements()) {
        emitError("position out of bounds: ") << idx;
        return {};
      }
      llvmType = arrayType.getElementType();
    } else if (auto structType = llvm::dyn_cast<LLVMStructType>(llvmType)) {
      if (idx < 0 ||
          static_cast<unsigned>(idx) >= structType.getBody().size()) {
        emitError("position out of bounds: ") << idx;
        return {};
      }
      llvmType = structType.getBody()[idx];
    } else {
      emitError("expected LLVM IR structure/array type, got: ") << llvmType;
      return {};
    }
  }
  return llvmType;
}

/// Extract the type at `position` in the wrapped LLVM IR aggregate type
/// `containerType`.
static Type getInsertExtractValueElementType(Type llvmType,
                                             ArrayRef<int64_t> position) {
  for (int64_t idx : position) {
    if (auto structType = llvm::dyn_cast<LLVMStructType>(llvmType))
      llvmType = structType.getBody()[idx];
    else
      llvmType = llvm::cast<LLVMArrayType>(llvmType).getElementType();
  }
  return llvmType;
}

OpFoldResult LLVM::ExtractValueOp::fold(FoldAdaptor adaptor) {
  if (auto extractValueOp = getContainer().getDefiningOp<ExtractValueOp>()) {
    SmallVector<int64_t, 4> newPos(extractValueOp.getPosition());
    newPos.append(getPosition().begin(), getPosition().end());
    setPosition(newPos);
    getContainerMutable().set(extractValueOp.getContainer());
    return getResult();
  }

  {
    DenseElementsAttr constval;
    matchPattern(getContainer(), m_Constant(&constval));
    if (constval && constval.getElementType() == getType()) {
      if (isa<SplatElementsAttr>(constval))
        return constval.getSplatValue<Attribute>();
      if (getPosition().size() == 1)
        return constval.getValues<Attribute>()[getPosition()[0]];
    }
  }

  auto insertValueOp = getContainer().getDefiningOp<InsertValueOp>();
  OpFoldResult result = {};
  ArrayRef<int64_t> extractPos = getPosition();
  bool switchedToInsertedValue = false;
  while (insertValueOp) {
    ArrayRef<int64_t> insertPos = insertValueOp.getPosition();
    auto extractPosSize = extractPos.size();
    auto insertPosSize = insertPos.size();

    // Case 1: Exact match of positions.
    if (extractPos == insertPos)
      return insertValueOp.getValue();

    // Case 2: Insert position is a prefix of extract position. Continue
    // traversal with the inserted value. Example:
    // ```
    // %0 = llvm.insertvalue %arg1, %undef[0] : !llvm.struct<(i32, i32, i32)>
    // %1 = llvm.insertvalue %arg2, %0[1] : !llvm.struct<(i32, i32, i32)>
    // %2 = llvm.insertvalue %arg3, %1[2] : !llvm.struct<(i32, i32, i32)>
    // %3 = llvm.insertvalue %2, %foo[0]
    //     : !llvm.struct<(struct<(i32, i32, i32)>, i64)>
    // %4 = llvm.extractvalue %3[0, 0]
    //     : !llvm.struct<(struct<(i32, i32, i32)>, i64)>
    // ```
    // In the above example, %4 is folded to %arg1.
    if (extractPosSize > insertPosSize &&
        extractPos.take_front(insertPosSize) == insertPos) {
      insertValueOp = insertValueOp.getValue().getDefiningOp<InsertValueOp>();
      extractPos = extractPos.drop_front(insertPosSize);
      switchedToInsertedValue = true;
      continue;
    }

    // Case 3: Try to continue the traversal with the container value.
    unsigned min = std::min(extractPosSize, insertPosSize);

    // If one is fully prefix of the other, stop propagating back as it will
    // miss dependencies. For instance, %3 should not fold to %f0 in the
    // following example:
    // ```
    //   %1 = llvm.insertvalue %f0, %0[0, 0] :
    //     !llvm.array<4 x !llvm.array<4 x f32>>
    //   %2 = llvm.insertvalue %arr, %1[0] :
    //     !llvm.array<4 x !llvm.array<4 x f32>>
    //   %3 = llvm.extractvalue %2[0, 0] : !llvm.array<4 x !llvm.array<4 x f32>>
    // ```
    if (extractPos.take_front(min) == insertPos.take_front(min))
      return result;
    // If neither a prefix, nor the exact position, we can extract out of the
    // value being inserted into. Moreover, we can try again if that operand
    // is itself an insertvalue expression.
    if (!switchedToInsertedValue) {
      // Do not swap out the container operand if we decided earlier to
      // continue the traversal with the inserted value (Case 2).
      getContainerMutable().assign(insertValueOp.getContainer());
      result = getResult();
    }
    insertValueOp = insertValueOp.getContainer().getDefiningOp<InsertValueOp>();
  }
  return result;
}

LogicalResult ExtractValueOp::verify() {
  auto emitError = [this](StringRef msg) { return emitOpError(msg); };
  Type valueType = getInsertExtractValueElementType(
      emitError, getContainer().getType(), getPosition());
  if (!valueType)
    return failure();

  if (getRes().getType() != valueType)
    return emitOpError() << "Type mismatch: extracting from "
                         << getContainer().getType() << " should produce "
                         << valueType << " but this op returns "
                         << getRes().getType();
  return success();
}

void ExtractValueOp::build(OpBuilder &builder, OperationState &state,
                           Value container, ArrayRef<int64_t> position) {
  build(builder, state,
        getInsertExtractValueElementType(container.getType(), position),
        container, builder.getAttr<DenseI64ArrayAttr>(position));
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

/// Infer the value type from the container type and position.
static ParseResult
parseInsertExtractValueElementType(AsmParser &parser, Type &valueType,
                                   Type containerType,
                                   DenseI64ArrayAttr position) {
  valueType = getInsertExtractValueElementType(
      [&](StringRef msg) {
        return parser.emitError(parser.getCurrentLocation(), msg);
      },
      containerType, position.asArrayRef());
  return success(!!valueType);
}

/// Nothing to print for an inferred type.
static void printInsertExtractValueElementType(AsmPrinter &printer,
                                               Operation *op, Type valueType,
                                               Type containerType,
                                               DenseI64ArrayAttr position) {}

LogicalResult InsertValueOp::verify() {
  auto emitError = [this](StringRef msg) { return emitOpError(msg); };
  Type valueType = getInsertExtractValueElementType(
      emitError, getContainer().getType(), getPosition());
  if (!valueType)
    return failure();

  if (getValue().getType() != valueType)
    return emitOpError() << "Type mismatch: cannot insert "
                         << getValue().getType() << " into "
                         << getContainer().getType();

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto parent = (*this)->getParentOfType<LLVMFuncOp>();
  if (!parent)
    return success();

  Type expectedType = parent.getFunctionType().getReturnType();
  if (llvm::isa<LLVMVoidType>(expectedType)) {
    if (!getArg())
      return success();
    InFlightDiagnostic diag = emitOpError("expected no operands");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  if (!getArg()) {
    if (llvm::isa<LLVMVoidType>(expectedType))
      return success();
    InFlightDiagnostic diag = emitOpError("expected 1 operand");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  if (expectedType != getArg().getType()) {
    InFlightDiagnostic diag = emitOpError("mismatching result types");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::AddressOfOp.
//===----------------------------------------------------------------------===//

GlobalOp AddressOfOp::getGlobal(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<GlobalOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

LLVMFuncOp AddressOfOp::getFunction(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<LLVMFuncOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

AliasOp AddressOfOp::getAlias(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<AliasOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

IFuncOp AddressOfOp::getIFunc(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<IFuncOp>(
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr()));
}

LogicalResult
AddressOfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol =
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr());

  auto global = dyn_cast_or_null<GlobalOp>(symbol);
  auto function = dyn_cast_or_null<LLVMFuncOp>(symbol);
  auto alias = dyn_cast_or_null<AliasOp>(symbol);
  auto ifunc = dyn_cast_or_null<IFuncOp>(symbol);

  if (!global && !function && !alias && !ifunc)
    return emitOpError("must reference a global defined by 'llvm.mlir.global', "
                       "'llvm.mlir.alias' or 'llvm.func' or 'llvm.mlir.ifunc'");

  LLVMPointerType type = getType();
  if ((global && global.getAddrSpace() != type.getAddressSpace()) ||
      (alias && alias.getAddrSpace() != type.getAddressSpace()))
    return emitOpError("pointer address space must match address space of the "
                       "referenced global or alias");

  return success();
}

// AddressOfOp constant-folds to the global symbol name.
OpFoldResult LLVM::AddressOfOp::fold(FoldAdaptor) {
  return getGlobalNameAttr();
}

//===----------------------------------------------------------------------===//
// LLVM::DSOLocalEquivalentOp
//===----------------------------------------------------------------------===//

LLVMFuncOp
DSOLocalEquivalentOp::getFunction(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<LLVMFuncOp>(symbolTable.lookupSymbolIn(
      parentLLVMModule(*this), getFunctionNameAttr()));
}

AliasOp DSOLocalEquivalentOp::getAlias(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<AliasOp>(symbolTable.lookupSymbolIn(
      parentLLVMModule(*this), getFunctionNameAttr()));
}

LogicalResult
DSOLocalEquivalentOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol = symbolTable.lookupSymbolIn(parentLLVMModule(*this),
                                                 getFunctionNameAttr());
  auto function = dyn_cast_or_null<LLVMFuncOp>(symbol);
  auto alias = dyn_cast_or_null<AliasOp>(symbol);

  if (!function && !alias)
    return emitOpError(
        "must reference a global defined by 'llvm.func' or 'llvm.mlir.alias'");

  if (alias) {
    if (alias.getInitializer()
            .walk([&](AddressOfOp addrOp) {
              if (addrOp.getGlobal(symbolTable))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted())
      return emitOpError("must reference an alias to a function");
  }

  if ((function && function.getLinkage() == LLVM::Linkage::ExternWeak) ||
      (alias && alias.getLinkage() == LLVM::Linkage::ExternWeak))
    return emitOpError(
        "target function with 'extern_weak' linkage not allowed");

  return success();
}

/// Fold a dso_local_equivalent operation to a dedicated dso_local_equivalent
/// attribute.
OpFoldResult DSOLocalEquivalentOp::fold(FoldAdaptor) {
  return DSOLocalEquivalentAttr::get(getContext(), getFunctionNameAttr());
}

//===----------------------------------------------------------------------===//
// Verifier for LLVM::ComdatOp.
//===----------------------------------------------------------------------===//

void ComdatOp::build(OpBuilder &builder, OperationState &result,
                     StringRef symName) {
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(symName));
  Region *body = result.addRegion();
  body->emplaceBlock();
}

LogicalResult ComdatOp::verifyRegions() {
  Region &body = getBody();
  for (Operation &op : body.getOps())
    if (!isa<ComdatSelectorOp>(op))
      return op.emitError(
          "only comdat selector symbols can appear in a comdat region");

  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::GlobalOp.
//===----------------------------------------------------------------------===//

void GlobalOp::build(OpBuilder &builder, OperationState &result, Type type,
                     bool isConstant, Linkage linkage, StringRef name,
                     Attribute value, uint64_t alignment, unsigned addrSpace,
                     bool dsoLocal, bool threadLocal, SymbolRefAttr comdat,
                     ArrayRef<NamedAttribute> attrs,
                     ArrayRef<Attribute> dbgExprs) {
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
  if (comdat)
    result.addAttribute(getComdatAttrName(result.name), comdat);

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

  if (!dbgExprs.empty())
    result.addAttribute(getDbgExprsAttrName(result.name),
                        ArrayAttr::get(builder.getContext(), dbgExprs));

  result.addRegion();
}

template <typename OpType>
static void printCommonGlobalAndAlias(OpAsmPrinter &p, OpType op) {
  p << ' ' << stringifyLinkage(op.getLinkage()) << ' ';
  StringRef visibility = stringifyVisibility(op.getVisibility_());
  if (!visibility.empty())
    p << visibility << ' ';
  if (op.getThreadLocal_())
    p << "thread_local ";
  if (auto unnamedAddr = op.getUnnamedAddr()) {
    StringRef str = stringifyUnnamedAddr(*unnamedAddr);
    if (!str.empty())
      p << str << ' ';
  }
}

void GlobalOp::print(OpAsmPrinter &p) {
  printCommonGlobalAndAlias<GlobalOp>(p, *this);
  if (getConstant())
    p << "constant ";
  p.printSymbolName(getSymName());
  p << '(';
  if (auto value = getValueOrNull())
    p.printAttribute(value);
  p << ')';
  if (auto comdat = getComdat())
    p << " comdat(" << *comdat << ')';

  // Note that the alignment attribute is printed using the
  // default syntax here, even though it is an inherent attribute
  // (as defined in https://mlir.llvm.org/docs/LangRef/#attributes)
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {SymbolTable::getSymbolAttrName(),
                           getGlobalTypeAttrName(), getConstantAttrName(),
                           getValueAttrName(), getLinkageAttrName(),
                           getUnnamedAddrAttrName(), getThreadLocal_AttrName(),
                           getVisibility_AttrName(), getComdatAttrName()});

  // Print the trailing type unless it's a string global.
  if (llvm::dyn_cast_or_null<StringAttr>(getValueOrNull()))
    return;
  p << " : " << getType();

  Region &initializer = getInitializerRegion();
  if (!initializer.empty()) {
    p << ' ';
    p.printRegion(initializer, /*printEntryBlockArgs=*/false);
  }
}

static LogicalResult verifyComdat(Operation *op,
                                  std::optional<SymbolRefAttr> attr) {
  if (!attr)
    return success();

  auto *comdatSelector = SymbolTable::lookupNearestSymbolFrom(op, *attr);
  if (!isa_and_nonnull<ComdatSelectorOp>(comdatSelector))
    return op->emitError() << "expected comdat symbol";

  return success();
}

static LogicalResult verifyBlockTags(LLVMFuncOp funcOp) {
  llvm::DenseSet<BlockTagAttr> blockTags;
  // Note that presence of `BlockTagOp`s currently can't prevent an unrecheable
  // block to be removed by canonicalizer's region simplify pass, which needs to
  // be dialect aware to allow extra constraints to be described.
  WalkResult res = funcOp.walk([&](BlockTagOp blockTagOp) {
    if (blockTags.contains(blockTagOp.getTag())) {
      blockTagOp.emitError()
          << "duplicate block tag '" << blockTagOp.getTag().getId()
          << "' in the same function: ";
      return WalkResult::interrupt();
    }
    blockTags.insert(blockTagOp.getTag());
    return WalkResult::advance();
  });

  return failure(res.wasInterrupted());
}

/// Parse common attributes that might show up in the same order in both
/// GlobalOp and AliasOp.
template <typename OpType>
static ParseResult parseCommonGlobalAndAlias(OpAsmParser &parser,
                                             OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  // Parse optional linkage, default to External.
  result.addAttribute(
      OpType::getLinkageAttrName(result.name),
      LLVM::LinkageAttr::get(ctx, parseOptionalLLVMKeyword<Linkage>(
                                      parser, LLVM::Linkage::External)));

  // Parse optional visibility, default to Default.
  result.addAttribute(OpType::getVisibility_AttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<LLVM::Visibility, int64_t>(
                              parser, LLVM::Visibility::Default)));

  if (succeeded(parser.parseOptionalKeyword("thread_local")))
    result.addAttribute(OpType::getThreadLocal_AttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  // Parse optional UnnamedAddr, default to None.
  result.addAttribute(OpType::getUnnamedAddrAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<UnnamedAddr, int64_t>(
                              parser, LLVM::UnnamedAddr::None)));

  return success();
}

// operation ::= `llvm.mlir.global` linkage? visibility?
//               (`unnamed_addr` | `local_unnamed_addr`)?
//               `thread_local`? `constant`? `@` identifier
//               `(` attribute? `)` (`comdat(` symbol-ref-id `)`)?
//               attribute-list? (`:` type)? region?
//
// The type can be omitted for string attributes, in which case it will be
// inferred from the value of the string as [strlen(value) x i8].
ParseResult GlobalOp::parse(OpAsmParser &parser, OperationState &result) {
  // Call into common parsing between GlobalOp and AliasOp.
  if (parseCommonGlobalAndAlias<GlobalOp>(parser, result).failed())
    return failure();

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

  if (succeeded(parser.parseOptionalKeyword("comdat"))) {
    SymbolRefAttr comdat;
    if (parser.parseLParen() || parser.parseAttribute(comdat) ||
        parser.parseRParen())
      return failure();

    result.addAttribute(getComdatAttrName(result.name), comdat);
  }

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  Region &initRegion = *result.addRegion();
  if (types.empty()) {
    if (auto strAttr = llvm::dyn_cast_or_null<StringAttr>(value)) {
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
    if (parseResult.has_value() && failed(*parseResult))
      return failure();
  }

  result.addAttribute(getGlobalTypeAttrName(result.name),
                      TypeAttr::get(types[0]));
  return success();
}

static bool isZeroAttribute(Attribute value) {
  if (auto intValue = llvm::dyn_cast<IntegerAttr>(value))
    return intValue.getValue().isZero();
  if (auto fpValue = llvm::dyn_cast<FloatAttr>(value))
    return fpValue.getValue().isZero();
  if (auto splatValue = llvm::dyn_cast<SplatElementsAttr>(value))
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = llvm::dyn_cast<ElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value))
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

LogicalResult GlobalOp::verify() {
  bool validType = isCompatibleOuterType(getType())
                       ? !llvm::isa<LLVMVoidType, LLVMTokenType,
                                    LLVMMetadataType, LLVMLabelType>(getType())
                       : llvm::isa<PointerElementTypeInterface>(getType());
  if (!validType)
    return emitOpError(
        "expects type to be a valid element type for an LLVM global");
  if ((*this)->getParentOp() && !satisfiesLLVMModule((*this)->getParentOp()))
    return emitOpError("must appear at the module level");

  if (auto strAttr = llvm::dyn_cast_or_null<StringAttr>(getValueOrNull())) {
    auto type = llvm::dyn_cast<LLVMArrayType>(getType());
    IntegerType elementType =
        type ? llvm::dyn_cast<IntegerType>(type.getElementType()) : nullptr;
    if (!elementType || elementType.getWidth() != 8 ||
        type.getNumElements() != strAttr.getValue().size())
      return emitOpError(
          "requires an i8 array type of the length equal to that of the string "
          "attribute");
  }

  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getType())) {
    if (!targetExtType.hasProperty(LLVMTargetExtType::CanBeGlobal))
      return emitOpError()
             << "this target extension type cannot be used in a global";

    if (Attribute value = getValueOrNull())
      return emitOpError() << "global with target extension type can only be "
                              "initialized with zero-initializer";
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
    if (!llvm::isa<LLVMArrayType>(getType())) {
      return emitOpError() << "expected array type for '"
                           << stringifyLinkage(Linkage::Appending)
                           << "' linkage";
    }
  }

  if (failed(verifyComdat(*this, getComdat())))
    return failure();

  std::optional<uint64_t> alignAttr = getAlignment();
  if (alignAttr.has_value()) {
    uint64_t value = alignAttr.value();
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

LogicalResult checkGlobalXtorData(Operation *op, ArrayAttr data) {
  if (data.empty())
    return success();

  if (llvm::all_of(data.getAsRange<Attribute>(), [](Attribute v) {
        return isa<FlatSymbolRefAttr, ZeroAttr>(v);
      }))
    return success();
  return op->emitError("data element must be symbol or #llvm.zero");
}

LogicalResult
GlobalCtorsOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute ctor : getCtors()) {
    if (failed(verifySymbolAttrUse(llvm::cast<FlatSymbolRefAttr>(ctor), *this,
                                   symbolTable)))
      return failure();
  }
  return success();
}

LogicalResult GlobalCtorsOp::verify() {
  if (checkGlobalXtorData(*this, getData()).failed())
    return failure();

  if (getCtors().size() == getPriorities().size() &&
      getCtors().size() == getData().size())
    return success();
  return emitError(
      "ctors, priorities, and data must have the same number of elements");
}

//===----------------------------------------------------------------------===//
// LLVM::GlobalDtorsOp
//===----------------------------------------------------------------------===//

LogicalResult
GlobalDtorsOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  for (Attribute dtor : getDtors()) {
    if (failed(verifySymbolAttrUse(llvm::cast<FlatSymbolRefAttr>(dtor), *this,
                                   symbolTable)))
      return failure();
  }
  return success();
}

LogicalResult GlobalDtorsOp::verify() {
  if (checkGlobalXtorData(*this, getData()).failed())
    return failure();

  if (getDtors().size() == getPriorities().size() &&
      getDtors().size() == getData().size())
    return success();
  return emitError(
      "dtors, priorities, and data must have the same number of elements");
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::AliasOp.
//===----------------------------------------------------------------------===//

void AliasOp::build(OpBuilder &builder, OperationState &result, Type type,
                    Linkage linkage, StringRef name, bool dsoLocal,
                    bool threadLocal, ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(getSymNameAttrName(result.name),
                      builder.getStringAttr(name));
  result.addAttribute(getAliasTypeAttrName(result.name), TypeAttr::get(type));
  if (dsoLocal)
    result.addAttribute(getDsoLocalAttrName(result.name),
                        builder.getUnitAttr());
  if (threadLocal)
    result.addAttribute(getThreadLocal_AttrName(result.name),
                        builder.getUnitAttr());

  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(builder.getContext(), linkage));
  result.attributes.append(attrs.begin(), attrs.end());

  result.addRegion();
}

void AliasOp::print(OpAsmPrinter &p) {
  printCommonGlobalAndAlias<AliasOp>(p, *this);

  p.printSymbolName(getSymName());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {SymbolTable::getSymbolAttrName(),
                           getAliasTypeAttrName(), getLinkageAttrName(),
                           getUnnamedAddrAttrName(), getThreadLocal_AttrName(),
                           getVisibility_AttrName()});

  // Print the trailing type.
  p << " : " << getType() << ' ';
  // Print the initializer region.
  p.printRegion(getInitializerRegion(), /*printEntryBlockArgs=*/false);
}

// operation ::= `llvm.mlir.alias` linkage? visibility?
//               (`unnamed_addr` | `local_unnamed_addr`)?
//               `thread_local`? `@` identifier
//               `(` attribute? `)`
//               attribute-list? `:` type region
//
ParseResult AliasOp::parse(OpAsmParser &parser, OperationState &result) {
  // Call into common parsing between GlobalOp and AliasOp.
  if (parseCommonGlobalAndAlias<AliasOp>(parser, result).failed())
    return failure();

  StringAttr name;
  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  Region &initRegion = *result.addRegion();
  if (parser.parseRegion(initRegion).failed())
    return failure();

  result.addAttribute(getAliasTypeAttrName(result.name),
                      TypeAttr::get(types[0]));
  return success();
}

LogicalResult AliasOp::verify() {
  bool validType = isCompatibleOuterType(getType())
                       ? !llvm::isa<LLVMVoidType, LLVMTokenType,
                                    LLVMMetadataType, LLVMLabelType>(getType())
                       : llvm::isa<PointerElementTypeInterface>(getType());
  if (!validType)
    return emitOpError(
        "expects type to be a valid element type for an LLVM global alias");

  // This matches LLVM IR verification logic, see llvm/lib/IR/Verifier.cpp
  switch (getLinkage()) {
  case Linkage::External:
  case Linkage::Internal:
  case Linkage::Private:
  case Linkage::Weak:
  case Linkage::WeakODR:
  case Linkage::Linkonce:
  case Linkage::LinkonceODR:
  case Linkage::AvailableExternally:
    break;
  default:
    return emitOpError()
           << "'" << stringifyLinkage(getLinkage())
           << "' linkage not supported in aliases, available options: private, "
              "internal, linkonce, weak, linkonce_odr, weak_odr, external or "
              "available_externally";
  }

  return success();
}

LogicalResult AliasOp::verifyRegions() {
  Block &b = getInitializerBlock();
  auto ret = cast<ReturnOp>(b.getTerminator());
  if (ret.getNumOperands() == 0 ||
      !isa<LLVM::LLVMPointerType>(ret.getOperand(0).getType()))
    return emitOpError("initializer region must always return a pointer");

  for (Operation &op : b) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface || !iface.hasNoEffect())
      return op.emitError()
             << "ops with side effects are not allowed in alias initializers";
  }

  return success();
}

unsigned AliasOp::getAddrSpace() {
  Block &initializer = getInitializerBlock();
  auto ret = cast<ReturnOp>(initializer.getTerminator());
  auto ptrTy = cast<LLVMPointerType>(ret.getOperand(0).getType());
  return ptrTy.getAddressSpace();
}

//===----------------------------------------------------------------------===//
// IFuncOp
//===----------------------------------------------------------------------===//

void IFuncOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                    Type iFuncType, StringRef resolverName, Type resolverType,
                    Linkage linkage, LLVM::Visibility visibility) {
  return build(builder, result, name, iFuncType, resolverName, resolverType,
               linkage, /*dso_local=*/false, /*address_space=*/0,
               UnnamedAddr::None, visibility);
}

LogicalResult IFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol =
      symbolTable.lookupSymbolIn(parentLLVMModule(*this), getResolverAttr());
  // This matches LLVM IR verification logic, see llvm/lib/IR/Verifier.cpp
  auto resolver = dyn_cast<LLVMFuncOp>(symbol);
  auto alias = dyn_cast<AliasOp>(symbol);
  while (alias) {
    Block &initBlock = alias.getInitializerBlock();
    auto returnOp = cast<ReturnOp>(initBlock.getTerminator());
    auto addrOp = returnOp.getArg().getDefiningOp<AddressOfOp>();
    // FIXME: This is a best effort solution. The AliasOp body might be more
    // complex and in that case we bail out with success. To completely match
    // the LLVM IR logic it would be necessary to implement proper alias and
    // cast stripping.
    if (!addrOp)
      return success();
    resolver = addrOp.getFunction(symbolTable);
    alias = addrOp.getAlias(symbolTable);
  }
  if (!resolver)
    return emitOpError("must have a function resolver");
  Linkage linkage = resolver.getLinkage();
  if (resolver.isExternal() || linkage == Linkage::AvailableExternally)
    return emitOpError("resolver must be a definition");
  if (!isa<LLVMPointerType>(resolver.getFunctionType().getReturnType()))
    return emitOpError("resolver must return a pointer");
  auto resolverPtr = dyn_cast<LLVMPointerType>(getResolverType());
  if (!resolverPtr || resolverPtr.getAddressSpace() != getAddressSpace())
    return emitOpError("resolver has incorrect type");
  return success();
}

LogicalResult IFuncOp::verify() {
  switch (getLinkage()) {
  case Linkage::External:
  case Linkage::Internal:
  case Linkage::Private:
  case Linkage::Weak:
  case Linkage::WeakODR:
  case Linkage::Linkonce:
  case Linkage::LinkonceODR:
    break;
  default:
    return emitOpError() << "'" << stringifyLinkage(getLinkage())
                         << "' linkage not supported in ifuncs, available "
                            "options: private, internal, linkonce, weak, "
                            "linkonce_odr, weak_odr, or external linkage";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleVectorOp
//===----------------------------------------------------------------------===//

void ShuffleVectorOp::build(OpBuilder &builder, OperationState &state, Value v1,
                            Value v2, DenseI32ArrayAttr mask,
                            ArrayRef<NamedAttribute> attrs) {
  auto containerType = v1.getType();
  auto vType = LLVM::getVectorType(
      cast<VectorType>(containerType).getElementType(), mask.size(),
      LLVM::isScalableVectorType(containerType));
  build(builder, state, vType, v1, v2, mask);
  state.addAttributes(attrs);
}

void ShuffleVectorOp::build(OpBuilder &builder, OperationState &state, Value v1,
                            Value v2, ArrayRef<int32_t> mask) {
  build(builder, state, v1, v2, builder.getDenseI32ArrayAttr(mask));
}

/// Build the result type of a shuffle vector operation.
static ParseResult parseShuffleType(AsmParser &parser, Type v1Type,
                                    Type &resType, DenseI32ArrayAttr mask) {
  if (!LLVM::isCompatibleVectorType(v1Type))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected an LLVM compatible vector type");
  resType =
      LLVM::getVectorType(cast<VectorType>(v1Type).getElementType(),
                          mask.size(), LLVM::isScalableVectorType(v1Type));
  return success();
}

/// Nothing to do when the result type is inferred.
static void printShuffleType(AsmPrinter &printer, Operation *op, Type v1Type,
                             Type resType, DenseI32ArrayAttr mask) {}

LogicalResult ShuffleVectorOp::verify() {
  if (LLVM::isScalableVectorType(getV1().getType()) &&
      llvm::any_of(getMask(), [](int32_t v) { return v != 0; }))
    return emitOpError("expected a splat operation for scalable vectors");
  return success();
}

//===----------------------------------------------------------------------===//
// Implementations for LLVM::LLVMFuncOp.
//===----------------------------------------------------------------------===//

// Add the entry block to the function.
Block *LLVMFuncOp::addEntryBlock(OpBuilder &builder) {
  assert(empty() && "function already has an entry block");
  OpBuilder::InsertionGuard g(builder);
  Block *entry = builder.createBlock(&getBody());

  // FIXME: Allow passing in proper locations for the entry arguments.
  LLVMFunctionType type = getFunctionType();
  for (unsigned i = 0, e = type.getNumParams(); i < e; ++i)
    entry->addArgument(type.getParamType(i), getLoc());
  return entry;
}

void LLVMFuncOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, Type type, LLVM::Linkage linkage,
                       bool dsoLocal, CConv cconv, SymbolRefAttr comdat,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<DictionaryAttr> argAttrs,
                       std::optional<uint64_t> functionEntryCount) {
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
    result.addAttribute(getDsoLocalAttrName(result.name),
                        builder.getUnitAttr());
  if (comdat)
    result.addAttribute(getComdatAttrName(result.name), comdat);
  if (functionEntryCount)
    result.addAttribute(getFunctionEntryCountAttrName(result.name),
                        builder.getI64IntegerAttr(functionEntryCount.value()));
  if (argAttrs.empty())
    return;

  assert(llvm::cast<LLVMFunctionType>(type).getNumParams() == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  call_interface_impl::addArgAndResultAttrs(
      builder, result, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
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
//                (`comdat(` symbol-ref-id `)`)?
//                function-attributes?
//                function-body
//
ParseResult LLVMFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  // Default to external linkage if no keyword is provided.
  result.addAttribute(getLinkageAttrName(result.name),
                      LinkageAttr::get(parser.getContext(),
                                       parseOptionalLLVMKeyword<Linkage>(
                                           parser, LLVM::Linkage::External)));

  // Parse optional visibility, default to Default.
  result.addAttribute(getVisibility_AttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<LLVM::Visibility, int64_t>(
                              parser, LLVM::Visibility::Default)));

  // Parse optional UnnamedAddr, default to None.
  result.addAttribute(getUnnamedAddrAttrName(result.name),
                      parser.getBuilder().getI64IntegerAttr(
                          parseOptionalLLVMKeyword<UnnamedAddr, int64_t>(
                              parser, LLVM::UnnamedAddr::None)));

  // Default to C Calling Convention if no keyword is provided.
  result.addAttribute(
      getCConvAttrName(result.name),
      CConvAttr::get(parser.getContext(),
                     parseOptionalLLVMKeyword<CConv>(parser, LLVM::CConv::C)));

  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  auto signatureLocation = parser.getCurrentLocation();
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      function_interface_impl::parseFunctionSignatureWithArguments(
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
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  if (succeeded(parser.parseOptionalKeyword("vscale_range"))) {
    int64_t minRange, maxRange;
    if (parser.parseLParen() || parser.parseInteger(minRange) ||
        parser.parseComma() || parser.parseInteger(maxRange) ||
        parser.parseRParen())
      return failure();
    auto intTy = IntegerType::get(parser.getContext(), 32);
    result.addAttribute(
        getVscaleRangeAttrName(result.name),
        LLVM::VScaleRangeAttr::get(parser.getContext(),
                                   IntegerAttr::get(intTy, minRange),
                                   IntegerAttr::get(intTy, maxRange)));
  }
  // Parse the optional comdat selector.
  if (succeeded(parser.parseOptionalKeyword("comdat"))) {
    SymbolRefAttr comdat;
    if (parser.parseLParen() || parser.parseAttribute(comdat) ||
        parser.parseRParen())
      return failure();

    result.addAttribute(getComdatAttrName(result.name), comdat);
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, entryArgs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  auto *body = result.addRegion();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs);
  return failure(parseResult.has_value() && failed(*parseResult));
}

// Print the LLVMFuncOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back. Skips
// the external linkage since it is the default value.
void LLVMFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getLinkage() != LLVM::Linkage::External)
    p << stringifyLinkage(getLinkage()) << ' ';
  StringRef visibility = stringifyVisibility(getVisibility_());
  if (!visibility.empty())
    p << visibility << ' ';
  if (auto unnamedAddr = getUnnamedAddr()) {
    StringRef str = stringifyUnnamedAddr(*unnamedAddr);
    if (!str.empty())
      p << str << ' ';
  }
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
  if (!llvm::isa<LLVMVoidType>(returnType))
    resTypes.push_back(returnType);

  function_interface_impl::printFunctionSignature(p, *this, argTypes,
                                                  isVarArg(), resTypes);

  // Print vscale range if present
  if (std::optional<VScaleRangeAttr> vscale = getVscaleRange())
    p << " vscale_range(" << vscale->getMinRange().getInt() << ", "
      << vscale->getMaxRange().getInt() << ')';

  // Print the optional comdat selector.
  if (auto comdat = getComdat())
    p << " comdat(" << *comdat << ')';

  function_interface_impl::printFunctionAttributes(
      p, *this,
      {getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName(),
       getLinkageAttrName(), getCConvAttrName(), getVisibility_AttrName(),
       getComdatAttrName(), getUnnamedAddrAttrName(),
       getVscaleRangeAttrName()});

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

  if (failed(verifyComdat(*this, getComdat())))
    return failure();

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

  // In LLVM IR, these attributes are composed by convention, not by design.
  if (isNoInline() && isAlwaysInline())
    return emitError("no_inline and always_inline attributes are incompatible");

  if (isOptimizeNone() && !isNoInline())
    return emitOpError("with optimize_none must also be no_inline");

  Type landingpadResultTy;
  StringRef diagnosticMessage;
  bool isLandingpadTypeConsistent =
      !walk([&](Operation *op) {
         const auto checkType = [&](Type type, StringRef errorMessage) {
           if (!landingpadResultTy) {
             landingpadResultTy = type;
             return WalkResult::advance();
           }
           if (landingpadResultTy != type) {
             diagnosticMessage = errorMessage;
             return WalkResult::interrupt();
           }
           return WalkResult::advance();
         };
         return TypeSwitch<Operation *, WalkResult>(op)
             .Case<LandingpadOp>([&](auto landingpad) {
               constexpr StringLiteral errorMessage =
                   "'llvm.landingpad' should have a consistent result type "
                   "inside a function";
               return checkType(landingpad.getType(), errorMessage);
             })
             .Case<ResumeOp>([&](auto resume) {
               constexpr StringLiteral errorMessage =
                   "'llvm.resume' should have a consistent input type inside a "
                   "function";
               return checkType(resume.getValue().getType(), errorMessage);
             })
             .Default([](auto) { return WalkResult::skip(); });
       }).wasInterrupted();
  if (!isLandingpadTypeConsistent) {
    assert(!diagnosticMessage.empty() &&
           "Expecting a non-empty diagnostic message");
    return emitError(diagnosticMessage);
  }

  if (failed(verifyBlockTags(*this)))
    return failure();

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

Region *LLVMFuncOp::getCallableRegion() {
  if (isExternal())
    return nullptr;
  return &getBody();
}

//===----------------------------------------------------------------------===//
// UndefOp.
//===----------------------------------------------------------------------===//

/// Fold an undef operation to a dedicated undef attribute.
OpFoldResult LLVM::UndefOp::fold(FoldAdaptor) {
  return LLVM::UndefAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// PoisonOp.
//===----------------------------------------------------------------------===//

/// Fold a poison operation to a dedicated poison attribute.
OpFoldResult LLVM::PoisonOp::fold(FoldAdaptor) {
  return LLVM::PoisonAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// ZeroOp.
//===----------------------------------------------------------------------===//

LogicalResult LLVM::ZeroOp::verify() {
  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getType()))
    if (!targetExtType.hasProperty(LLVM::LLVMTargetExtType::HasZeroInit))
      return emitOpError()
             << "target extension type does not support zero-initializer";

  return success();
}

/// Fold a zero operation to a builtin zero attribute when possible and fall
/// back to a dedicated zero attribute.
OpFoldResult LLVM::ZeroOp::fold(FoldAdaptor) {
  OpFoldResult result = Builder(getContext()).getZeroAttr(getType());
  if (result)
    return result;
  return LLVM::ZeroAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// ConstantOp.
//===----------------------------------------------------------------------===//

/// Compute the total number of elements in the given type, also taking into
/// account nested types. Supported types are `VectorType` and `LLVMArrayType`.
/// Everything else is treated as a scalar.
static int64_t getNumElements(Type t) {
  if (auto vecType = dyn_cast<VectorType>(t)) {
    assert(!vecType.isScalable() &&
           "number of elements of a scalable vector type is unknown");
    return vecType.getNumElements() * getNumElements(vecType.getElementType());
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t))
    return arrayType.getNumElements() *
           getNumElements(arrayType.getElementType());
  return 1;
}

/// Determine the element type of `type`. Supported types are `VectorType`,
/// `TensorType`, and `LLVMArrayType`. Everything else is treated as a scalar.
static Type getElementType(Type type) {
  while (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    type = arrayType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  if (auto tenType = dyn_cast<TensorType>(type))
    return tenType.getElementType();
  return type;
}

/// Check if the given type is a scalable vector type or a vector/array type
/// that contains a nested scalable vector type.
static bool hasScalableVectorType(Type t) {
  if (auto vecType = dyn_cast<VectorType>(t)) {
    if (vecType.isScalable())
      return true;
    return hasScalableVectorType(vecType.getElementType());
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(t))
    return hasScalableVectorType(arrayType.getElementType());
  return false;
}

/// Verifies the constant array represented by `arrayAttr` matches the provided
/// `arrayType`.
static LogicalResult verifyStructArrayConstant(LLVM::ConstantOp op,
                                               LLVM::LLVMArrayType arrayType,
                                               ArrayAttr arrayAttr, int dim) {
  if (arrayType.getNumElements() != arrayAttr.size())
    return op.emitOpError()
           << "array attribute size does not match array type size in "
              "dimension "
           << dim << ": " << arrayAttr.size() << " vs. "
           << arrayType.getNumElements();

  llvm::DenseSet<Attribute> elementsVerified;

  // Recursively verify sub-dimensions for multidimensional arrays.
  if (auto subArrayType =
          dyn_cast<LLVM::LLVMArrayType>(arrayType.getElementType())) {
    for (auto [idx, elementAttr] : llvm::enumerate(arrayAttr))
      if (elementsVerified.insert(elementAttr).second) {
        if (isa<LLVM::ZeroAttr, LLVM::UndefAttr>(elementAttr))
          continue;
        auto subArrayAttr = dyn_cast<ArrayAttr>(elementAttr);
        if (!subArrayAttr)
          return op.emitOpError()
                 << "nested attribute for sub-array in dimension " << dim
                 << " at index " << idx
                 << " must be a zero, or undef, or array attribute";
        if (failed(verifyStructArrayConstant(op, subArrayType, subArrayAttr,
                                             dim + 1)))
          return failure();
      }
    return success();
  }

  // Forbid usages of ArrayAttr for simple array types that should use
  // DenseElementsAttr instead. Note that there would be a use case for such
  // array types when one element value is obtained via a ptr-to-int conversion
  // from a symbol and cannot be represented in a DenseElementsAttr, but no MLIR
  // user needs this so far, and it seems better to avoid people misusing the
  // ArrayAttr for simple types.
  auto structType = dyn_cast<LLVM::LLVMStructType>(arrayType.getElementType());
  if (!structType)
    return op.emitOpError() << "for array with an array attribute must have a "
                               "struct element type";

  // Shallow verification that leaf attributes are appropriate as struct initial
  // value.
  size_t numStructElements = structType.getBody().size();
  for (auto [idx, elementAttr] : llvm::enumerate(arrayAttr)) {
    if (elementsVerified.insert(elementAttr).second) {
      if (isa<LLVM::ZeroAttr, LLVM::UndefAttr>(elementAttr))
        continue;
      auto subArrayAttr = dyn_cast<ArrayAttr>(elementAttr);
      if (!subArrayAttr)
        return op.emitOpError()
               << "nested attribute for struct element at index " << idx
               << " must be a zero, or undef, or array attribute";
      if (subArrayAttr.size() != numStructElements)
        return op.emitOpError()
               << "nested array attribute size for struct element at index "
               << idx << " must match struct size: " << subArrayAttr.size()
               << " vs. " << numStructElements;
    }
  }

  return success();
}

LogicalResult LLVM::ConstantOp::verify() {
  if (StringAttr sAttr = llvm::dyn_cast<StringAttr>(getValue())) {
    auto arrayType = llvm::dyn_cast<LLVMArrayType>(getType());
    if (!arrayType || arrayType.getNumElements() != sAttr.getValue().size() ||
        !arrayType.getElementType().isInteger(8)) {
      return emitOpError() << "expected array type of "
                           << sAttr.getValue().size()
                           << " i8 elements for the string constant";
    }
    return success();
  }
  if (auto structType = dyn_cast<LLVMStructType>(getType())) {
    auto arrayAttr = dyn_cast<ArrayAttr>(getValue());
    if (!arrayAttr)
      return emitOpError() << "expected array attribute for struct type";

    ArrayRef<Type> elementTypes = structType.getBody();
    if (arrayAttr.size() != elementTypes.size()) {
      return emitOpError() << "expected array attribute of size "
                           << elementTypes.size();
    }
    for (auto [i, attr, type] : llvm::enumerate(arrayAttr, elementTypes)) {
      if (!type.isSignlessIntOrIndexOrFloat()) {
        return emitOpError() << "expected struct element types to be floating "
                                "point type or integer type";
      }
      if (!isa<FloatAttr, IntegerAttr>(attr)) {
        return emitOpError() << "expected element of array attribute to be "
                                "floating point or integer";
      }
      if (cast<TypedAttr>(attr).getType() != type)
        return emitOpError()
               << "struct element at index " << i << " is of wrong type";
    }

    return success();
  }
  if (auto targetExtType = dyn_cast<LLVMTargetExtType>(getType()))
    return emitOpError() << "does not support target extension type.";

  // Check that an attribute whose element type has floating point semantics
  // `attributeFloatSemantics` is compatible with a type whose element type
  // is `constantElementType`.
  //
  // Requirement is that either
  // 1) They have identical floating point types.
  // 2) `constantElementType` is an integer type of the same width as the float
  //     attribute. This is to support builtin MLIR float types without LLVM
  //     equivalents, see comments in getLLVMConstant for more details.
  auto verifyFloatSemantics =
      [this](const llvm::fltSemantics &attributeFloatSemantics,
             Type constantElementType) -> LogicalResult {
    if (auto floatType = dyn_cast<FloatType>(constantElementType)) {
      if (&floatType.getFloatSemantics() != &attributeFloatSemantics) {
        return emitOpError()
               << "attribute and type have different float semantics";
      }
      return success();
    }
    unsigned floatWidth = APFloat::getSizeInBits(attributeFloatSemantics);
    if (isa<IntegerType>(constantElementType)) {
      if (!constantElementType.isInteger(floatWidth))
        return emitOpError() << "expected integer type of width " << floatWidth;

      return success();
    }
    return success();
  };

  // Verification of IntegerAttr, FloatAttr, ElementsAttr, ArrayAttr.
  if (isa<IntegerAttr>(getValue())) {
    if (!llvm::isa<IntegerType>(getType()))
      return emitOpError() << "expected integer type";
  } else if (auto floatAttr = dyn_cast<FloatAttr>(getValue())) {
    return verifyFloatSemantics(floatAttr.getValue().getSemantics(), getType());
  } else if (auto elementsAttr = dyn_cast<ElementsAttr>(getValue())) {
    if (hasScalableVectorType(getType())) {
      // The exact number of elements of a scalable vector is unknown, so we
      // allow only splat attributes.
      auto splatElementsAttr = dyn_cast<SplatElementsAttr>(getValue());
      if (!splatElementsAttr)
        return emitOpError()
               << "scalable vector type requires a splat attribute";
      return success();
    }
    if (!isa<VectorType, LLVM::LLVMArrayType>(getType()))
      return emitOpError() << "expected vector or array type";

    // The number of elements of the attribute and the type must match.
    int64_t attrNumElements = elementsAttr.getNumElements();
    if (getNumElements(getType()) != attrNumElements) {
      return emitOpError()
             << "type and attribute have a different number of elements: "
             << getNumElements(getType()) << " vs. " << attrNumElements;
    }

    Type attrElmType = getElementType(elementsAttr.getType());
    Type resultElmType = getElementType(getType());
    if (auto floatType = dyn_cast<FloatType>(attrElmType))
      return verifyFloatSemantics(floatType.getFloatSemantics(), resultElmType);

    if (isa<IntegerType>(attrElmType) && !isa<IntegerType>(resultElmType)) {
      return emitOpError(
          "expected integer element type for integer elements attribute");
    }
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(getValue())) {

    // The case where the constant is LLVMStructType has already been handled.
    auto arrayType = dyn_cast<LLVM::LLVMArrayType>(getType());
    if (!arrayType)
      return emitOpError()
             << "expected array or struct type for array attribute";

    // When the attribute is an ArrayAttr, check that its nesting matches the
    // corresponding ArrayType or VectorType nesting.
    return verifyStructArrayConstant(*this, arrayType, arrayAttr, /*dim=*/0);
  } else {
    return emitOpError()
           << "only supports integer, float, string or elements attributes";
  }

  return success();
}

bool LLVM::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // The value's type must be the same as the provided type.
  auto typedAttr = dyn_cast<TypedAttr>(value);
  if (!typedAttr || typedAttr.getType() != type || !isCompatibleType(type))
    return false;
  // The value's type must be an LLVM compatible type.
  if (!isCompatibleType(type))
    return false;
  // TODO: Add support for additional attributes kinds once needed.
  return isa<IntegerAttr, FloatAttr, ElementsAttr>(value);
}

ConstantOp LLVM::ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
  if (isBuildableWith(value, type))
    return LLVM::ConstantOp::create(builder, loc, cast<TypedAttr>(value));
  return nullptr;
}

// Constant op constant-folds to its value.
OpFoldResult LLVM::ConstantOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

void AtomicRMWOp::build(OpBuilder &builder, OperationState &state,
                        AtomicBinOp binOp, Value ptr, Value val,
                        AtomicOrdering ordering, StringRef syncscope,
                        unsigned alignment, bool isVolatile) {
  build(builder, state, val.getType(), binOp, ptr, val, ordering,
        !syncscope.empty() ? builder.getStringAttr(syncscope) : nullptr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
}

LogicalResult AtomicRMWOp::verify() {
  auto valType = getVal().getType();
  if (getBinOp() == AtomicBinOp::fadd || getBinOp() == AtomicBinOp::fsub ||
      getBinOp() == AtomicBinOp::fmin || getBinOp() == AtomicBinOp::fmax ||
      getBinOp() == AtomicBinOp::fminimum ||
      getBinOp() == AtomicBinOp::fmaximum) {
    if (isCompatibleVectorType(valType)) {
      if (isScalableVectorType(valType))
        return emitOpError("expected LLVM IR fixed vector type");
      Type elemType = llvm::cast<VectorType>(valType).getElementType();
      if (!isCompatibleFloatingPointType(elemType))
        return emitOpError(
            "expected LLVM IR floating point type for vector element");
    } else if (!isCompatibleFloatingPointType(valType)) {
      return emitOpError("expected LLVM IR floating point type");
    }
  } else if (getBinOp() == AtomicBinOp::xchg) {
    DataLayout dataLayout = DataLayout::closest(*this);
    if (!isTypeCompatibleWithAtomicOp(valType, dataLayout))
      return emitOpError("unexpected LLVM IR type for 'xchg' bin_op");
  } else {
    auto intType = llvm::dyn_cast<IntegerType>(valType);
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
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

/// Returns an LLVM struct type that contains a value type and a boolean type.
static LLVMStructType getValAndBoolStructType(Type valType) {
  auto boolType = IntegerType::get(valType.getContext(), 1);
  return LLVMStructType::getLiteral(valType.getContext(), {valType, boolType});
}

void AtomicCmpXchgOp::build(OpBuilder &builder, OperationState &state,
                            Value ptr, Value cmp, Value val,
                            AtomicOrdering successOrdering,
                            AtomicOrdering failureOrdering, StringRef syncscope,
                            unsigned alignment, bool isWeak, bool isVolatile) {
  build(builder, state, getValAndBoolStructType(val.getType()), ptr, cmp, val,
        successOrdering, failureOrdering,
        !syncscope.empty() ? builder.getStringAttr(syncscope) : nullptr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isWeak,
        isVolatile, /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
}

LogicalResult AtomicCmpXchgOp::verify() {
  auto ptrType = llvm::cast<LLVM::LLVMPointerType>(getPtr().getType());
  if (!ptrType)
    return emitOpError("expected LLVM IR pointer type for operand #0");
  auto valType = getVal().getType();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!isTypeCompatibleWithAtomicOp(valType, dataLayout))
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
// FenceOp
//===----------------------------------------------------------------------===//

void FenceOp::build(OpBuilder &builder, OperationState &state,
                    AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
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
// Verifier for extension ops
//===----------------------------------------------------------------------===//

/// Verifies that the given extension operation operates on consistent scalars
/// or vectors, and that the target width is larger than the input width.
template <class ExtOp>
static LogicalResult verifyExtOp(ExtOp op) {
  IntegerType inputType, outputType;
  if (isCompatibleVectorType(op.getArg().getType())) {
    if (!isCompatibleVectorType(op.getResult().getType()))
      return op.emitError(
          "input type is a vector but output type is an integer");
    if (getVectorNumElements(op.getArg().getType()) !=
        getVectorNumElements(op.getResult().getType()))
      return op.emitError("input and output vectors are of incompatible shape");
    // Because this is a CastOp, the element of vectors is guaranteed to be an
    // integer.
    inputType = cast<IntegerType>(
        cast<VectorType>(op.getArg().getType()).getElementType());
    outputType = cast<IntegerType>(
        cast<VectorType>(op.getResult().getType()).getElementType());
  } else {
    // Because this is a CastOp and arg is not a vector, arg is guaranteed to be
    // an integer.
    inputType = cast<IntegerType>(op.getArg().getType());
    outputType = dyn_cast<IntegerType>(op.getResult().getType());
    if (!outputType)
      return op.emitError(
          "input type is an integer but output type is a vector");
  }

  if (outputType.getWidth() <= inputType.getWidth())
    return op.emitError("integer width of the output type is smaller or "
                        "equal to the integer width of the input type");
  return success();
}

//===----------------------------------------------------------------------===//
// ZExtOp
//===----------------------------------------------------------------------===//

LogicalResult ZExtOp::verify() { return verifyExtOp<ZExtOp>(*this); }

OpFoldResult LLVM::ZExtOp::fold(FoldAdaptor adaptor) {
  auto arg = dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg)
    return {};

  size_t targetSize = cast<IntegerType>(getType()).getWidth();
  return IntegerAttr::get(getType(), arg.getValue().zext(targetSize));
}

//===----------------------------------------------------------------------===//
// SExtOp
//===----------------------------------------------------------------------===//

LogicalResult SExtOp::verify() { return verifyExtOp<SExtOp>(*this); }

//===----------------------------------------------------------------------===//
// Folder and verifier for LLVM::BitcastOp
//===----------------------------------------------------------------------===//

/// Folds a cast op that can be chained.
template <typename T>
static OpFoldResult foldChainableCast(T castOp,
                                      typename T::FoldAdaptor adaptor) {
  // cast(x : T0, T0) -> x
  if (castOp.getArg().getType() == castOp.getType())
    return castOp.getArg();
  if (auto prev = castOp.getArg().template getDefiningOp<T>()) {
    // cast(cast(x : T0, T1), T0) -> x
    if (prev.getArg().getType() == castOp.getType())
      return prev.getArg();
    // cast(cast(x : T0, T1), T2) -> cast(x: T0, T2)
    castOp.getArgMutable().set(prev.getArg());
    return Value{castOp};
  }
  return {};
}

OpFoldResult LLVM::BitcastOp::fold(FoldAdaptor adaptor) {
  return foldChainableCast(*this, adaptor);
}

LogicalResult LLVM::BitcastOp::verify() {
  auto resultType = llvm::dyn_cast<LLVMPointerType>(
      extractVectorElementType(getResult().getType()));
  auto sourceType = llvm::dyn_cast<LLVMPointerType>(
      extractVectorElementType(getArg().getType()));

  // If one of the types is a pointer (or vector of pointers), then
  // both source and result type have to be pointers.
  if (static_cast<bool>(resultType) != static_cast<bool>(sourceType))
    return emitOpError("can only cast pointers from and to pointers");

  if (!resultType)
    return success();

  auto isVector = llvm::IsaPred<VectorType>;

  // Due to bitcast requiring both operands to be of the same size, it is not
  // possible for only one of the two to be a pointer of vectors.
  if (isVector(getResult().getType()) && !isVector(getArg().getType()))
    return emitOpError("cannot cast pointer to vector of pointers");

  if (!isVector(getResult().getType()) && isVector(getArg().getType()))
    return emitOpError("cannot cast vector of pointers to pointer");

  // Bitcast cannot cast between pointers of different address spaces.
  // 'llvm.addrspacecast' must be used for this purpose instead.
  if (resultType.getAddressSpace() != sourceType.getAddressSpace())
    return emitOpError("cannot cast pointers of different address spaces, "
                       "use 'llvm.addrspacecast' instead");

  return success();
}

//===----------------------------------------------------------------------===//
// Folder for LLVM::AddrSpaceCastOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::AddrSpaceCastOp::fold(FoldAdaptor adaptor) {
  return foldChainableCast(*this, adaptor);
}

Value LLVM::AddrSpaceCastOp::getViewSource() { return getArg(); }

//===----------------------------------------------------------------------===//
// Folder for LLVM::GEPOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::GEPOp::fold(FoldAdaptor adaptor) {
  GEPIndicesAdaptor<ArrayRef<Attribute>> indices(getRawConstantIndicesAttr(),
                                                 adaptor.getDynamicIndices());

  // gep %x:T, 0 -> %x
  if (getBase().getType() == getType() && indices.size() == 1)
    if (auto integer = llvm::dyn_cast_or_null<IntegerAttr>(indices[0]))
      if (integer.getValue().isZero())
        return getBase();

  // Canonicalize any dynamic indices of constant value to constant indices.
  bool changed = false;
  SmallVector<GEPArg> gepArgs;
  for (auto iter : llvm::enumerate(indices)) {
    auto integer = llvm::dyn_cast_or_null<IntegerAttr>(iter.value());
    // Constant indices can only be int32_t, so if integer does not fit we
    // are forced to keep it dynamic, despite being a constant.
    if (!indices.isDynamicIndex(iter.index()) || !integer ||
        !integer.getValue().isSignedIntN(kGEPConstantBitWidth)) {

      PointerUnion<IntegerAttr, Value> existing = getIndices()[iter.index()];
      if (Value val = llvm::dyn_cast_if_present<Value>(existing))
        gepArgs.emplace_back(val);
      else
        gepArgs.emplace_back(cast<IntegerAttr>(existing).getInt());

      continue;
    }

    changed = true;
    gepArgs.emplace_back(integer.getInt());
  }
  if (changed) {
    SmallVector<int32_t> rawConstantIndices;
    SmallVector<Value> dynamicIndices;
    destructureIndices(getElemType(), gepArgs, rawConstantIndices,
                       dynamicIndices);

    getDynamicIndicesMutable().assign(dynamicIndices);
    setRawConstantIndices(rawConstantIndices);
    return Value{*this};
  }

  return {};
}

Value LLVM::GEPOp::getViewSource() { return getBase(); }

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::ShlOp::fold(FoldAdaptor adaptor) {
  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!rhs)
    return {};

  if (rhs.getValue().getZExtValue() >=
      getLhs().getType().getIntOrFloatBitWidth())
    return {}; // TODO: Fold into poison.

  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  if (!lhs)
    return {};

  return IntegerAttr::get(getType(), lhs.getValue().shl(rhs.getValue()));
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::OrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  if (!lhs)
    return {};

  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!rhs)
    return {};

  return IntegerAttr::get(getType(), lhs.getValue() | rhs.getValue());
}

//===----------------------------------------------------------------------===//
// CallIntrinsicOp
//===----------------------------------------------------------------------===//

LogicalResult CallIntrinsicOp::verify() {
  if (!getIntrin().starts_with("llvm."))
    return emitOpError() << "intrinsic name must start with 'llvm.'";
  if (failed(verifyOperandBundles(*this)))
    return failure();
  return success();
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::StringAttr intrin, mlir::ValueRange args) {
  build(builder, state, /*resultTypes=*/TypeRange{}, intrin, args,
        FastmathFlagsAttr{},
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::StringAttr intrin, mlir::ValueRange args,
                            mlir::LLVM::FastmathFlagsAttr fastMathFlags) {
  build(builder, state, /*resultTypes=*/TypeRange{}, intrin, args,
        fastMathFlags,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::Type resultType, mlir::StringAttr intrin,
                            mlir::ValueRange args) {
  build(builder, state, {resultType}, intrin, args, FastmathFlagsAttr{},
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

void CallIntrinsicOp::build(OpBuilder &builder, OperationState &state,
                            mlir::TypeRange resultTypes,
                            mlir::StringAttr intrin, mlir::ValueRange args,
                            mlir::LLVM::FastmathFlagsAttr fastMathFlags) {
  build(builder, state, resultTypes, intrin, args, fastMathFlags,
        /*op_bundle_operands=*/{}, /*op_bundle_tags=*/{}, /*arg_attrs=*/{},
        /*res_attrs=*/{});
}

ParseResult CallIntrinsicOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  StringAttr intrinAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>> opBundleOperands;
  SmallVector<SmallVector<Type>> opBundleOperandTypes;
  ArrayAttr opBundleTags;

  // Parse intrinsic name.
  if (parser.parseCustomAttributeWithFallback(
          intrinAttr, parser.getBuilder().getType<NoneType>()))
    return failure();
  result.addAttribute(CallIntrinsicOp::getIntrinAttrName(result.name),
                      intrinAttr);

  if (parser.parseLParen())
    return failure();

  // Parse the function arguments.
  if (parser.parseOperandList(operands))
    return mlir::failure();

  if (parser.parseRParen())
    return mlir::failure();

  // Handle bundles.
  SMLoc opBundlesLoc = parser.getCurrentLocation();
  if (std::optional<ParseResult> result = parseOpBundles(
          parser, opBundleOperands, opBundleOperandTypes, opBundleTags);
      result && failed(*result))
    return failure();
  if (opBundleTags && !opBundleTags.empty())
    result.addAttribute(
        CallIntrinsicOp::getOpBundleTagsAttrName(result.name).getValue(),
        opBundleTags);

  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  if (parseCallTypeAndResolveOperands(parser, result, /*isDirect=*/true,
                                      operands, argAttrs, resultAttrs))
    return failure();
  call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, argAttrs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  if (resolveOpBundleOperands(parser, opBundlesLoc, result, opBundleOperands,
                              opBundleOperandTypes,
                              getOpBundleSizesAttrName(result.name)))
    return failure();

  int32_t numOpBundleOperands = 0;
  for (const auto &operands : opBundleOperands)
    numOpBundleOperands += operands.size();

  result.addAttribute(
      CallIntrinsicOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(operands.size()), numOpBundleOperands}));

  return mlir::success();
}

void CallIntrinsicOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getIntrinAttr());

  OperandRange args = getArgs();
  p << "(" << args << ")";

  // Operand bundles.
  if (!getOpBundleOperands().empty()) {
    p << ' ';
    printOpBundles(p, *this, getOpBundleOperands(),
                   getOpBundleOperands().getTypes(), getOpBundleTagsAttr());
  }

  p.printOptionalAttrDict(processFMFAttr((*this)->getAttrs()),
                          {getOperandSegmentSizesAttrName(),
                           getOpBundleSizesAttrName(), getIntrinAttrName(),
                           getOpBundleTagsAttrName(), getArgAttrsAttrName(),
                           getResAttrsAttrName()});

  p << " : ";

  // Reconstruct the MLIR function type from operand and result types.
  call_interface_impl::printFunctionSignature(
      p, args.getTypes(), getArgAttrsAttr(),
      /*isVariadic=*/false, getResultTypes(), getResAttrsAttr());
}

//===----------------------------------------------------------------------===//
// LinkerOptionsOp
//===----------------------------------------------------------------------===//

LogicalResult LinkerOptionsOp::verify() {
  if (mlir::Operation *parentOp = (*this)->getParentOp();
      parentOp && !satisfiesLLVMModule(parentOp))
    return emitOpError("must appear at the module level");
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleFlagsOp
//===----------------------------------------------------------------------===//

LogicalResult ModuleFlagsOp::verify() {
  if (Operation *parentOp = (*this)->getParentOp();
      parentOp && !satisfiesLLVMModule(parentOp))
    return emitOpError("must appear at the module level");
  for (Attribute flag : getFlags())
    if (!isa<ModuleFlagAttr>(flag))
      return emitOpError("expected a module flag attribute");
  return success();
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

void InlineAsmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getHasSideEffects()) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

//===----------------------------------------------------------------------===//
// BlockAddressOp
//===----------------------------------------------------------------------===//

LogicalResult
BlockAddressOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *symbol = symbolTable.lookupSymbolIn(parentLLVMModule(*this),
                                                 getBlockAddr().getFunction());
  auto function = dyn_cast_or_null<LLVMFuncOp>(symbol);

  if (!function)
    return emitOpError("must reference a function defined by 'llvm.func'");

  return success();
}

LLVMFuncOp BlockAddressOp::getFunction(SymbolTableCollection &symbolTable) {
  return dyn_cast_or_null<LLVMFuncOp>(symbolTable.lookupSymbolIn(
      parentLLVMModule(*this), getBlockAddr().getFunction()));
}

BlockTagOp BlockAddressOp::getBlockTagOp() {
  auto funcOp = dyn_cast<LLVMFuncOp>(mlir::SymbolTable::lookupNearestSymbolFrom(
      parentLLVMModule(*this), getBlockAddr().getFunction()));
  if (!funcOp)
    return nullptr;

  BlockTagOp blockTagOp = nullptr;
  funcOp.walk([&](LLVM::BlockTagOp labelOp) {
    if (labelOp.getTag() == getBlockAddr().getTag()) {
      blockTagOp = labelOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return blockTagOp;
}

LogicalResult BlockAddressOp::verify() {
  if (!getBlockTagOp())
    return emitOpError(
        "expects an existing block label target in the referenced function");

  return success();
}

/// Fold a blockaddress operation to a dedicated blockaddress
/// attribute.
OpFoldResult BlockAddressOp::fold(FoldAdaptor) { return getBlockAddr(); }

//===----------------------------------------------------------------------===//
// LLVM::IndirectBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands IndirectBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(getSuccOperandsMutable()[index]);
}

void IndirectBrOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         Value addr, ArrayRef<ValueRange> succOperands,
                         BlockRange successors) {
  odsState.addOperands(addr);
  for (ValueRange range : succOperands)
    odsState.addOperands(range);
  SmallVector<int32_t> rangeSegments;
  for (ValueRange range : succOperands)
    rangeSegments.push_back(range.size());
  odsState.getOrAddProperties<Properties>().indbr_operand_segments =
      odsBuilder.getDenseI32ArrayAttr(rangeSegments);
  odsState.addSuccessors(successors);
}

static ParseResult parseIndirectBrOpSucessors(
    OpAsmParser &parser, Type &flagType,
    SmallVectorImpl<Block *> &succOperandBlocks,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &succOperands,
    SmallVectorImpl<SmallVector<Type>> &succOperandsTypes) {
  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square,
          [&]() {
            Block *destination = nullptr;
            SmallVector<OpAsmParser::UnresolvedOperand> operands;
            SmallVector<Type> operandTypes;

            if (parser.parseSuccessor(destination).failed())
              return failure();

            if (succeeded(parser.parseOptionalLParen())) {
              if (failed(parser.parseOperandList(
                      operands, OpAsmParser::Delimiter::None)) ||
                  failed(parser.parseColonTypeList(operandTypes)) ||
                  failed(parser.parseRParen()))
                return failure();
            }
            succOperandBlocks.push_back(destination);
            succOperands.emplace_back(operands);
            succOperandsTypes.emplace_back(operandTypes);
            return success();
          },
          "successor blocks")))
    return failure();
  return success();
}

static void
printIndirectBrOpSucessors(OpAsmPrinter &p, IndirectBrOp op, Type flagType,
                           SuccessorRange succs, OperandRangeRange succOperands,
                           const TypeRangeRange &succOperandsTypes) {
  p << "[";
  llvm::interleave(
      llvm::zip(succs, succOperands),
      [&](auto i) {
        p.printNewline();
        p.printSuccessorAndUseList(std::get<0>(i), std::get<1>(i));
      },
      [&] { p << ','; });
  if (!succOperands.empty())
    p.printNewline();
  p << "]";
}

//===----------------------------------------------------------------------===//
// AssumeOp (intrinsic)
//===----------------------------------------------------------------------===//

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           mlir::Value cond) {
  return build(builder, state, cond, /*op_bundle_operands=*/{},
               /*op_bundle_tags=*/ArrayAttr{});
}

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           Value cond, llvm::StringRef tag, ValueRange args) {
  return build(builder, state, cond, ArrayRef<ValueRange>(args),
               builder.getStrArrayAttr(tag));
}

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           Value cond, AssumeAlignTag, Value ptr, Value align) {
  return build(builder, state, cond, "align", ValueRange{ptr, align});
}

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           Value cond, AssumeSeparateStorageTag, Value ptr1,
                           Value ptr2) {
  return build(builder, state, cond, "separate_storage",
               ValueRange{ptr1, ptr2});
}

LogicalResult LLVM::AssumeOp::verify() { return verifyOperandBundles(*this); }

//===----------------------------------------------------------------------===//
// masked_gather (intrinsic)
//===----------------------------------------------------------------------===//

LogicalResult LLVM::masked_gather::verify() {
  auto ptrsVectorType = getPtrs().getType();
  Type expectedPtrsVectorType =
      LLVM::getVectorType(extractVectorElementType(ptrsVectorType),
                          LLVM::getVectorNumElements(getRes().getType()));
  // Vector of pointers type should match result vector type, other than the
  // element type.
  if (ptrsVectorType != expectedPtrsVectorType)
    return emitOpError("expected operand #1 type to be ")
           << expectedPtrsVectorType;
  return success();
}

//===----------------------------------------------------------------------===//
// masked_scatter (intrinsic)
//===----------------------------------------------------------------------===//

LogicalResult LLVM::masked_scatter::verify() {
  auto ptrsVectorType = getPtrs().getType();
  Type expectedPtrsVectorType =
      LLVM::getVectorType(extractVectorElementType(ptrsVectorType),
                          LLVM::getVectorNumElements(getValue().getType()));
  // Vector of pointers type should match value vector type, other than the
  // element type.
  if (ptrsVectorType != expectedPtrsVectorType)
    return emitOpError("expected operand #2 type to be ")
           << expectedPtrsVectorType;
  return success();
}

//===----------------------------------------------------------------------===//
// masked_expandload (intrinsic)
//===----------------------------------------------------------------------===//

void LLVM::masked_expandload::build(OpBuilder &builder, OperationState &state,
                                    mlir::TypeRange resTys, Value ptr,
                                    Value mask, Value passthru,
                                    uint64_t align) {
  ArrayAttr argAttrs = getLLVMAlignParamForCompressExpand(builder, true, align);
  build(builder, state, resTys, ptr, mask, passthru, /*arg_attrs=*/argAttrs,
        /*res_attrs=*/nullptr);
}

//===----------------------------------------------------------------------===//
// masked_compressstore (intrinsic)
//===----------------------------------------------------------------------===//

void LLVM::masked_compressstore::build(OpBuilder &builder,
                                       OperationState &state, Value value,
                                       Value ptr, Value mask, uint64_t align) {
  ArrayAttr argAttrs =
      getLLVMAlignParamForCompressExpand(builder, false, align);
  build(builder, state, value, ptr, mask, /*arg_attrs=*/argAttrs,
        /*res_attrs=*/nullptr);
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

LogicalResult InlineAsmOp::verify() {
  if (!getTailCallKindAttr())
    return success();

  if (getTailCallKindAttr().getTailCallKind() == TailCallKind::MustTail)
    return emitOpError(
        "tail call kind 'musttail' is not supported by this operation");

  return success();
}

//===----------------------------------------------------------------------===//
// LLVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void LLVMDialect::initialize() {
  registerAttributes();

  // clang-format off
  addTypes<LLVMVoidType,
           LLVMTokenType,
           LLVMLabelType,
           LLVMMetadataType>();
  // clang-format on
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"

      ,
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.cpp.inc"

      >();

  // Support unknown operations because not all LLVM operations are registered.
  allowUnknownOperations();
  declarePromisedInterface<DialectInlinerInterface, LLVMDialect>();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.cpp.inc"

LogicalResult LLVMDialect::verifyDataLayoutString(
    StringRef descr, llvm::function_ref<void(const Twine &)> reportError) {
  llvm::Expected<llvm::DataLayout> maybeDataLayout =
      llvm::DataLayout::parse(descr);
  if (maybeDataLayout)
    return success();

  std::string message;
  llvm::raw_string_ostream messageStream(message);
  llvm::logAllUnhandledErrors(maybeDataLayout.takeError(), messageStream);
  reportError("invalid data layout descriptor: " + message);
  return failure();
}

/// Verify LLVM dialect attributes.
LogicalResult LLVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // If the data layout attribute is present, it must use the LLVM data layout
  // syntax. Try parsing it and report errors in case of failure. Users of this
  // attribute may assume it is well-formed and can pass it to the (asserting)
  // llvm::DataLayout constructor.
  if (attr.getName() != LLVM::LLVMDialect::getDataLayoutAttrName())
    return success();
  if (auto stringAttr = llvm::dyn_cast<StringAttr>(attr.getValue()))
    return verifyDataLayoutString(
        stringAttr.getValue(),
        [op](const Twine &message) { op->emitOpError() << message.str(); });

  return op->emitOpError() << "expected '"
                           << LLVM::LLVMDialect::getDataLayoutAttrName()
                           << "' to be a string attributes";
}

LogicalResult LLVMDialect::verifyParameterAttribute(Operation *op,
                                                    Type paramType,
                                                    NamedAttribute paramAttr) {
  // LLVM attribute may be attached to a result of operation that has not been
  // converted to LLVM dialect yet, so the result may have a type with unknown
  // representation in LLVM dialect type space. In this case we cannot verify
  // whether the attribute may be
  bool verifyValueType = isCompatibleType(paramType);
  StringAttr name = paramAttr.getName();

  auto checkUnitAttrType = [&]() -> LogicalResult {
    if (!llvm::isa<UnitAttr>(paramAttr.getValue()))
      return op->emitError() << name << " should be a unit attribute";
    return success();
  };
  auto checkTypeAttrType = [&]() -> LogicalResult {
    if (!llvm::isa<TypeAttr>(paramAttr.getValue()))
      return op->emitError() << name << " should be a type attribute";
    return success();
  };
  auto checkIntegerAttrType = [&]() -> LogicalResult {
    if (!llvm::isa<IntegerAttr>(paramAttr.getValue()))
      return op->emitError() << name << " should be an integer attribute";
    return success();
  };
  auto checkPointerType = [&]() -> LogicalResult {
    if (!llvm::isa<LLVMPointerType>(paramType))
      return op->emitError()
             << name << " attribute attached to non-pointer LLVM type";
    return success();
  };
  auto checkIntegerType = [&]() -> LogicalResult {
    if (!llvm::isa<IntegerType>(paramType))
      return op->emitError()
             << name << " attribute attached to non-integer LLVM type";
    return success();
  };
  auto checkPointerTypeMatches = [&]() -> LogicalResult {
    if (failed(checkPointerType()))
      return failure();

    return success();
  };

  // Check a unit attribute that is attached to a pointer value.
  if (name == LLVMDialect::getNoAliasAttrName() ||
      name == LLVMDialect::getReadonlyAttrName() ||
      name == LLVMDialect::getReadnoneAttrName() ||
      name == LLVMDialect::getWriteOnlyAttrName() ||
      name == LLVMDialect::getNestAttrName() ||
      name == LLVMDialect::getNoCaptureAttrName() ||
      name == LLVMDialect::getNoFreeAttrName() ||
      name == LLVMDialect::getNonNullAttrName()) {
    if (failed(checkUnitAttrType()))
      return failure();
    if (verifyValueType && failed(checkPointerType()))
      return failure();
    return success();
  }

  // Check a type attribute that is attached to a pointer value.
  if (name == LLVMDialect::getStructRetAttrName() ||
      name == LLVMDialect::getByValAttrName() ||
      name == LLVMDialect::getByRefAttrName() ||
      name == LLVMDialect::getElementTypeAttrName() ||
      name == LLVMDialect::getInAllocaAttrName() ||
      name == LLVMDialect::getPreallocatedAttrName()) {
    if (failed(checkTypeAttrType()))
      return failure();
    if (verifyValueType && failed(checkPointerTypeMatches()))
      return failure();
    return success();
  }

  // Check a unit attribute that is attached to an integer value.
  if (name == LLVMDialect::getSExtAttrName() ||
      name == LLVMDialect::getZExtAttrName()) {
    if (failed(checkUnitAttrType()))
      return failure();
    if (verifyValueType && failed(checkIntegerType()))
      return failure();
    return success();
  }

  // Check an integer attribute that is attached to a pointer value.
  if (name == LLVMDialect::getAlignAttrName() ||
      name == LLVMDialect::getDereferenceableAttrName() ||
      name == LLVMDialect::getDereferenceableOrNullAttrName()) {
    if (failed(checkIntegerAttrType()))
      return failure();
    if (verifyValueType && failed(checkPointerType()))
      return failure();
    return success();
  }

  // Check an integer attribute that is attached to a pointer value.
  if (name == LLVMDialect::getStackAlignmentAttrName()) {
    if (failed(checkIntegerAttrType()))
      return failure();
    return success();
  }

  // Check a unit attribute that can be attached to arbitrary types.
  if (name == LLVMDialect::getNoUndefAttrName() ||
      name == LLVMDialect::getInRegAttrName() ||
      name == LLVMDialect::getReturnedAttrName())
    return checkUnitAttrType();

  return success();
}

/// Verify LLVMIR function argument attributes.
LogicalResult LLVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIdx,
                                                    unsigned argIdx,
                                                    NamedAttribute argAttr) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();
  Type argType = funcOp.getArgumentTypes()[argIdx];

  return verifyParameterAttribute(op, argType, argAttr);
}

LogicalResult LLVMDialect::verifyRegionResultAttribute(Operation *op,
                                                       unsigned regionIdx,
                                                       unsigned resIdx,
                                                       NamedAttribute resAttr) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();
  Type resType = funcOp.getResultTypes()[resIdx];

  // Check to see if this function has a void return with a result attribute
  // to it. It isn't clear what semantics we would assign to that.
  if (llvm::isa<LLVMVoidType>(resType))
    return op->emitError() << "cannot attach result attributes to functions "
                              "with a void return";

  // Check to see if this attribute is allowed as a result attribute. Only
  // explicitly forbidden LLVM attributes will cause an error.
  auto name = resAttr.getName();
  if (name == LLVMDialect::getAllocAlignAttrName() ||
      name == LLVMDialect::getAllocatedPointerAttrName() ||
      name == LLVMDialect::getByValAttrName() ||
      name == LLVMDialect::getByRefAttrName() ||
      name == LLVMDialect::getInAllocaAttrName() ||
      name == LLVMDialect::getNestAttrName() ||
      name == LLVMDialect::getNoCaptureAttrName() ||
      name == LLVMDialect::getNoFreeAttrName() ||
      name == LLVMDialect::getPreallocatedAttrName() ||
      name == LLVMDialect::getReadnoneAttrName() ||
      name == LLVMDialect::getReadonlyAttrName() ||
      name == LLVMDialect::getReturnedAttrName() ||
      name == LLVMDialect::getStackAlignmentAttrName() ||
      name == LLVMDialect::getStructRetAttrName() ||
      name == LLVMDialect::getWriteOnlyAttrName())
    return op->emitError() << name << " is not a valid result attribute";
  return verifyParameterAttribute(op, resType, resAttr);
}

Operation *LLVMDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // If this was folded from an operation other than llvm.mlir.constant, it
  // should be materialized as such. Note that an llvm.mlir.zero may fold into
  // a builtin zero attribute and thus will materialize as a llvm.mlir.constant.
  if (auto symbol = dyn_cast<FlatSymbolRefAttr>(value))
    if (isa<LLVM::LLVMPointerType>(type))
      return LLVM::AddressOfOp::create(builder, loc, type, symbol);
  if (isa<LLVM::UndefAttr>(value))
    return LLVM::UndefOp::create(builder, loc, type);
  if (isa<LLVM::PoisonAttr>(value))
    return LLVM::PoisonOp::create(builder, loc, type);
  if (isa<LLVM::ZeroAttr>(value))
    return LLVM::ZeroOp::create(builder, loc, type);
  // Otherwise try materializing it as a regular llvm.mlir.constant op.
  return LLVM::ConstantOp::materialize(builder, value, type, loc);
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
  auto global = LLVM::GlobalOp::create(
      moduleBuilder, loc, type, /*isConstant=*/true, linkage, name,
      builder.getStringAttr(value), /*alignment=*/0);

  LLVMPointerType ptrType = LLVMPointerType::get(ctx);
  // Get the pointer to the first character in the global string.
  Value globalPtr =
      LLVM::AddressOfOp::create(builder, loc, ptrType, global.getSymNameAttr());
  return LLVM::GEPOp::create(builder, loc, ptrType, type, globalPtr,
                             ArrayRef<GEPArg>{0, 0});
}

bool mlir::LLVM::satisfiesLLVMModule(Operation *op) {
  return op->hasTrait<OpTrait::SymbolTable>() &&
         op->hasTrait<OpTrait::IsIsolatedFromAbove>();
}

Operation *mlir::LLVM::parentLLVMModule(Operation *op) {
  Operation *module = op->getParentOp();
  while (module && !satisfiesLLVMModule(module))
    module = module->getParentOp();
  assert(module && "unexpected operation outside of a module");
  return module;
}
