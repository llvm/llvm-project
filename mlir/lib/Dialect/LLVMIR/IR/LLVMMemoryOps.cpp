//===- LLVMMemoryOps.cpp - LLVM dialect memory ops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Memory` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMArithOps.h"
#include "mlir/Dialect/LLVMIR/LLVMCastOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDebugIntrinsicOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"

#include <optional>

#define DEBUG_TYPE "sroa"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Attribute Helpers
//===----------------------------------------------------------------------===//

static constexpr const char kElemTypeAttrName[] = "elem_type";

//===----------------------------------------------------------------------===//
// Printing, parsing, verification and canonicalization for LLVM::AllocaOp.
//===----------------------------------------------------------------------===//

void AllocaOp::print(OpAsmPrinter &p) {
  auto funcTy =
      FunctionType::get(getContext(), {getArraySize().getType()}, {getType()});

  if (getInalloca())
    p << " inalloca";

  p << ' ' << getArraySize() << " x " << getElemType();
  NamedAttrList attrs((*this)->getDiscardableAttrDictionary().getValue());
  if (getAlignment() && *getAlignment() != 0)
    attrs.append(getAlignmentAttrName(), getAlignmentAttr());
  p.printOptionalAttrDict(attrs);
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

LogicalResult AllocaOp::canonicalize(AllocaOp op, PatternRewriter &rewriter) {
  // Convert `alloca Ty, C` to the canonical `alloca [C x Ty], 1` form.
  APInt numElements;
  if (!matchPattern(op.getArraySize(), m_ConstantInt(&numElements)) ||
      numElements.isOne() || numElements.getActiveBits() > 64)
    return failure();

  auto arrayType =
      LLVMArrayType::get(op.getElemType(), numElements.getZExtValue());
  Value one = ConstantOp::create(rewriter, op.getLoc(), rewriter.getI32Type(),
                                 /*value=*/1);
  auto newAlloca =
      AllocaOp::create(rewriter, op.getLoc(), op.getType(), one,
                       op.getAlignmentAttr(), arrayType, op.getInalloca());
  newAlloca->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  rewriter.replaceOp(op, newAlloca);
  return success();
}

//===----------------------------------------------------------------------===//
// Code for LLVM::GEPOp.
//===----------------------------------------------------------------------===//

GEPIndicesAdaptor<ValueRange> GEPOp::getIndices() {
  return GEPIndicesAdaptor<ValueRange>(getRawConstantIndicesAttr(),
                                       getDynamicIndices());
}

/// Returns the elemental type of any LLVM-compatible vector type or self.
Type mlir::LLVM::extractVectorElementType(Type type) {
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
                   .Default(nullptr);
  }
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Type elementType, Value basePtr, ArrayRef<GEPArg> indices,
                  GEPNoWrapFlags noWrapFlags, ConstantRangeAttr inrange,
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
  result.getOrAddProperties<Properties>().inrange = inrange;
  result.addOperands(basePtr);
  result.addOperands(dynamicIndices);
}

void GEPOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  Type elementType, Value basePtr, ValueRange indices,
                  GEPNoWrapFlags noWrapFlags, ConstantRangeAttr inrange,
                  ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultType, elementType, basePtr,
        SmallVector<GEPArg>(indices), noWrapFlags, inrange, attributes);
}

ParseResult mlir::LLVM::parseGEPIndices(
    OpAsmParser &parser,
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

void mlir::LLVM::printGEPIndices(OpAsmPrinter &printer, LLVM::GEPOp gepOp,
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
      .Case([&](LLVMStructType structType) -> LogicalResult {
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

  // LLVM stores `inrange` only on GetElementPtrConstantExpr, whose operands
  // are already Constants. This dialect has one GEP operation, so proving the
  // base and indices are constant expressions means walking SSA. A local check
  // rejects valid cases such as a nested constant GEP base or a `ptrtoint`
  // index.
  if (auto inrange = getInrangeAttr()) {
    auto pointerType =
        cast<LLVMPointerType>(extractVectorElementType(getBase().getType()));
    DataLayout dataLayout = DataLayout::closest(*this);
    std::optional<uint64_t> indexWidth =
        dataLayout.getTypeIndexBitwidth(pointerType);
    assert(indexWidth && "pointers always return an index bitwidth");
    if (inrange.getLower().getBitWidth() != *indexWidth)
      return emitOpError("'inrange' bitwidth ")
             << inrange.getLower().getBitWidth()
             << " must match the pointer index bitwidth (" << *indexWidth
             << ") specified in the datalayout";
    if (inrange.getLower().sge(inrange.getUpper()))
      return emitOpError("expected 'inrange' end to be larger than start");
  }

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
bool LLVM::isTypeCompatibleWithAtomicOp(Type type,
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
static LogicalResult
verifyAtomicMemOp(OpTy memOp, Type valueType,
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
                   bool isNoundef, AtomicOrdering ordering,
                   StringRef syncscope) {
  build(builder, state, type, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, isInvariant, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope),
        /*dereferenceable=*/nullptr, isNoundef, /*range=*/nullptr,
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
// AtomicRMWOp
//===----------------------------------------------------------------------===//

void AtomicRMWOp::build(OpBuilder &builder, OperationState &state,
                        AtomicBinOp binOp, Value ptr, Value val,
                        AtomicOrdering ordering, StringRef syncscope,
                        unsigned alignment, bool isVolatile,
                        bool ignoreDenormalMode) {
  build(builder, state, val.getType(), binOp, ptr, val, ordering,
        !syncscope.empty() ? builder.getStringAttr(syncscope) : nullptr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        ignoreDenormalMode,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
}

LogicalResult AtomicRMWOp::verify() {
  auto valType = getVal().getType();
  if (isFloatingPointOperation()) {
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

  if (getIgnoreDenormalMode() && !isFloatingPointOperation())
    return emitOpError(
        "expected floating-point operation with 'ignore_denormal_mode'");

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
LLVMStructType mlir::LLVM::getValAndBoolStructType(Type valType) {
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
// Folder for LLVM::GEPOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::GEPOp::fold(FoldAdaptor adaptor) {
  GEPIndicesAdaptor<ArrayRef<Attribute>> indices(getRawConstantIndicesAttr(),
                                                 adaptor.getDynamicIndices());

  // Avoid losing inrange information.
  if (getInrangeAttr())
    return {};

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
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> LLVM::AllocaOp::getPromotableSlots() {
  if (!getOperation()->getBlock()->isEntryBlock())
    return {};

  return {MemorySlot{getResult(), getElemType()}};
}

Value LLVM::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                      OpBuilder &builder) {
  return LLVM::UndefOp::create(builder, getLoc(), slot.valueType);
}

void LLVM::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                         BlockArgument argument,
                                         OpBuilder &builder) {
  for (Operation *user : getOperation()->getUsers())
    if (auto declareOp = llvm::dyn_cast<LLVM::DbgDeclareOp>(user))
      LLVM::DbgValueOp::create(builder, declareOp.getLoc(), argument,
                               declareOp.getVarInfo(),
                               declareOp.getLocationExpr());
}

std::optional<PromotableAllocationOpInterface>
LLVM::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                        Value defaultValue,
                                        OpBuilder &builder) {
  if (defaultValue && defaultValue.use_empty())
    defaultValue.getDefiningOp()->erase();
  this->erase();
  return std::nullopt;
}

SmallVector<DestructurableMemorySlot> LLVM::AllocaOp::getDestructurableSlots() {
  if (!mlir::matchPattern(getArraySize(), m_One()))
    return {};

  auto destructurable = dyn_cast<DestructurableTypeInterface>(getElemType());
  if (!destructurable)
    return {};

  std::optional<DenseMap<Attribute, Type>> destructuredType =
      destructurable.getSubelementIndexMap();
  if (!destructuredType)
    return {};

  return {DestructurableMemorySlot{{getResult(), getElemType()},
                                   *destructuredType}};
}

DenseMap<Attribute, MemorySlot> LLVM::AllocaOp::destructure(
    const DestructurableMemorySlot &slot,
    const SmallPtrSetImpl<Attribute> &usedIndices, OpBuilder &builder,
    SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators) {
  assert(slot.ptr == getResult());
  builder.setInsertionPointAfter(*this);

  auto destructurableType = cast<DestructurableTypeInterface>(getElemType());
  DenseMap<Attribute, MemorySlot> slotMap;
  // Iterate subelements in their original type order to produce allocas in a
  // deterministic, readable order (matching appearance in the source type).
  Type i32 = IntegerType::get(getContext(), 32);
  for (size_t i = 0; i < slot.subelementTypes.size(); i++) {
    Attribute index = IntegerAttr::get(i32, i);
    if (!usedIndices.contains(index))
      continue;
    Type elemType = destructurableType.getTypeAtIndex(index);
    assert(elemType && "used index must exist");
    auto subAlloca = LLVM::AllocaOp::create(
        builder, getLoc(), LLVM::LLVMPointerType::get(getContext()), elemType,
        getArraySize());
    newAllocators.push_back(subAlloca);
    slotMap.try_emplace<MemorySlot>(index, {subAlloca.getResult(), elemType});
  }

  return slotMap;
}

std::optional<DestructurableAllocationOpInterface>
LLVM::AllocaOp::handleDestructuringComplete(
    const DestructurableMemorySlot &slot, OpBuilder &builder) {
  assert(slot.ptr == getResult());
  this->erase();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp/StoreOp
//===----------------------------------------------------------------------===//

bool LLVM::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool LLVM::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value LLVM::LoadOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                              Value reachingDef, const DataLayout &dataLayout) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool LLVM::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool LLVM::StoreOp::storesTo(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

/// Checks if `type` can be used in any kind of conversion sequences.
static bool isSupportedTypeForConversion(Type type) {
  // Aggregate types are not bitcastable.
  if (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(type))
    return false;

  if (auto vectorType = dyn_cast<VectorType>(type)) {
    // Vectors of pointers cannot be casted.
    if (isa<LLVM::LLVMPointerType>(vectorType.getElementType()))
      return false;
    // Scalable types are not supported.
    return !vectorType.isScalable();
  }
  return true;
}

/// Checks that `rhs` can be converted to `lhs` by a sequence of casts and
/// truncations. Checks for narrowing or widening conversion compatibility
/// depending on `narrowingConversion`.
static bool areConversionCompatible(const DataLayout &layout, Type targetType,
                                    Type srcType, bool narrowingConversion) {
  if (targetType == srcType)
    return true;

  if (!isSupportedTypeForConversion(targetType) ||
      !isSupportedTypeForConversion(srcType))
    return false;

  uint64_t targetSize = layout.getTypeSize(targetType);
  uint64_t srcSize = layout.getTypeSize(srcType);

  // Pointer casts will only be sane when the bitsize of both pointer types is
  // the same.
  if (isa<LLVM::LLVMPointerType>(targetType) &&
      isa<LLVM::LLVMPointerType>(srcType))
    return targetSize == srcSize;

  if (narrowingConversion)
    return targetSize <= srcSize;
  return targetSize >= srcSize;
}

/// Checks if `dataLayout` describes a little endian layout.
static bool isBigEndian(const DataLayout &dataLayout) {
  auto endiannessStr = dyn_cast_or_null<StringAttr>(dataLayout.getEndianness());
  return endiannessStr && endiannessStr == "big";
}

/// Converts a value to an integer type of the same size.
/// Assumes that the type can be converted.
static Value castToSameSizedInt(OpBuilder &builder, Location loc, Value val,
                                const DataLayout &dataLayout) {
  Type type = val.getType();
  assert(isSupportedTypeForConversion(type) &&
         "expected value to have a convertible type");

  if (isa<IntegerType>(type))
    return val;

  uint64_t typeBitSize = dataLayout.getTypeSizeInBits(type);
  IntegerType valueSizeInteger = builder.getIntegerType(typeBitSize);

  if (isa<LLVM::LLVMPointerType>(type))
    return builder.createOrFold<LLVM::PtrToIntOp>(loc, valueSizeInteger, val);
  return builder.createOrFold<LLVM::BitcastOp>(loc, valueSizeInteger, val);
}

/// Converts a value with an integer type to `targetType`.
static Value castIntValueToSameSizedType(OpBuilder &builder, Location loc,
                                         Value val, Type targetType) {
  assert(isa<IntegerType>(val.getType()) &&
         "expected value to have an integer type");
  assert(isSupportedTypeForConversion(targetType) &&
         "expected the target type to be supported for conversions");
  if (val.getType() == targetType)
    return val;
  if (isa<LLVM::LLVMPointerType>(targetType))
    return builder.createOrFold<LLVM::IntToPtrOp>(loc, targetType, val);
  return builder.createOrFold<LLVM::BitcastOp>(loc, targetType, val);
}

/// Constructs operations that convert `srcValue` into a new value of type
/// `targetType`. Assumes the types have the same bitsize.
static Value castSameSizedTypes(OpBuilder &builder, Location loc,
                                Value srcValue, Type targetType,
                                const DataLayout &dataLayout) {
  Type srcType = srcValue.getType();
  assert(areConversionCompatible(dataLayout, targetType, srcType,
                                 /*narrowingConversion=*/true) &&
         "expected that the compatibility was checked before");

  // Nothing has to be done if the types are already the same.
  if (srcType == targetType)
    return srcValue;

  // In the special case of casting one pointer to another, we want to generate
  // an address space cast. Bitcasts of pointers are not allowed and using
  // pointer to integer conversions are not equivalent due to the loss of
  // provenance.
  if (isa<LLVM::LLVMPointerType>(targetType) &&
      isa<LLVM::LLVMPointerType>(srcType))
    return builder.createOrFold<LLVM::AddrSpaceCastOp>(loc, targetType,
                                                       srcValue);

  // For all other castable types, casting through integers is necessary.
  Value replacement = castToSameSizedInt(builder, loc, srcValue, dataLayout);
  return castIntValueToSameSizedType(builder, loc, replacement, targetType);
}

/// Constructs operations that convert `srcValue` into a new value of type
/// `targetType`. Performs bit-level extraction if the source type is larger
/// than the target type. Assumes that this conversion is possible.
static Value createExtractAndCast(OpBuilder &builder, Location loc,
                                  Value srcValue, Type targetType,
                                  const DataLayout &dataLayout) {
  // Get the types of the source and target values.
  Type srcType = srcValue.getType();
  assert(areConversionCompatible(dataLayout, targetType, srcType,
                                 /*narrowingConversion=*/true) &&
         "expected that the compatibility was checked before");

  // Nothing has to be done if the types are already the same. This also
  // avoids querying the bit size of scalable vector types below.
  if (srcType == targetType)
    return srcValue;

  uint64_t srcTypeSize = dataLayout.getTypeSizeInBits(srcType);
  uint64_t targetTypeSize = dataLayout.getTypeSizeInBits(targetType);
  if (srcTypeSize == targetTypeSize)
    return castSameSizedTypes(builder, loc, srcValue, targetType, dataLayout);

  // First, cast the value to a same-sized integer type.
  Value replacement = castToSameSizedInt(builder, loc, srcValue, dataLayout);

  // Truncate the integer if the size of the target is less than the value.
  if (isBigEndian(dataLayout)) {
    uint64_t shiftAmount = srcTypeSize - targetTypeSize;
    auto shiftConstant = LLVM::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(srcType, shiftAmount));
    replacement =
        builder.createOrFold<LLVM::LShrOp>(loc, srcValue, shiftConstant);
  }

  replacement = LLVM::TruncOp::create(
      builder, loc, builder.getIntegerType(targetTypeSize), replacement);

  // Now cast the integer to the actual target type if required.
  return castIntValueToSameSizedType(builder, loc, replacement, targetType);
}

/// Constructs operations that insert the bits of `srcValue` into the
/// "beginning" of `reachingDef` (beginning is endianness dependent).
/// Assumes that this conversion is possible.
static Value createInsertAndCast(OpBuilder &builder, Location loc,
                                 Value srcValue, Value reachingDef,
                                 const DataLayout &dataLayout) {

  assert(areConversionCompatible(dataLayout, reachingDef.getType(),
                                 srcValue.getType(),
                                 /*narrowingConversion=*/false) &&
         "expected that the compatibility was checked before");

  // Nothing has to be done if the types are already the same. This also
  // avoids querying the bit size of scalable vector types below.
  if (srcValue.getType() == reachingDef.getType())
    return srcValue;

  uint64_t valueTypeSize = dataLayout.getTypeSizeInBits(srcValue.getType());
  uint64_t slotTypeSize = dataLayout.getTypeSizeInBits(reachingDef.getType());
  if (slotTypeSize == valueTypeSize)
    return castSameSizedTypes(builder, loc, srcValue, reachingDef.getType(),
                              dataLayout);

  // In the case where the store only overwrites parts of the memory,
  // bit fiddling is required to construct the new value.

  // First convert both values to integers of the same size.
  Value defAsInt = castToSameSizedInt(builder, loc, reachingDef, dataLayout);
  Value valueAsInt = castToSameSizedInt(builder, loc, srcValue, dataLayout);
  // Extend the value to the size of the reaching definition.
  valueAsInt =
      builder.createOrFold<LLVM::ZExtOp>(loc, defAsInt.getType(), valueAsInt);
  uint64_t sizeDifference = slotTypeSize - valueTypeSize;
  if (isBigEndian(dataLayout)) {
    // On big endian systems, a store to the base pointer overwrites the most
    // significant bits. To accomodate for this, the stored value needs to be
    // shifted into the according position.
    Value bigEndianShift = LLVM::ConstantOp::create(
        builder, loc,
        builder.getIntegerAttr(defAsInt.getType(), sizeDifference));
    valueAsInt =
        builder.createOrFold<LLVM::ShlOp>(loc, valueAsInt, bigEndianShift);
  }

  // Construct the mask that is used to erase the bits that are overwritten by
  // the store.
  APInt maskValue;
  if (isBigEndian(dataLayout)) {
    // Build a mask that has the most significant bits set to zero.
    // Note: This is the same as 2^sizeDifference - 1
    maskValue = APInt::getAllOnes(sizeDifference).zext(slotTypeSize);
  } else {
    // Build a mask that has the least significant bits set to zero.
    // Note: This is the same as -(2^valueTypeSize)
    maskValue = APInt::getAllOnes(valueTypeSize).zext(slotTypeSize);
    maskValue.flipAllBits();
  }

  // Mask out the affected bits ...
  Value mask = LLVM::ConstantOp::create(
      builder, loc, builder.getIntegerAttr(defAsInt.getType(), maskValue));
  Value masked = builder.createOrFold<LLVM::AndOp>(loc, defAsInt, mask);

  // ... and combine the result with the new value.
  Value combined = builder.createOrFold<LLVM::OrOp>(loc, masked, valueAsInt);

  return castIntValueToSameSizedType(builder, loc, combined,
                                     reachingDef.getType());
}

Value LLVM::StoreOp::getStored(const MemorySlot &slot, OpBuilder &builder,
                               Value reachingDef,
                               const DataLayout &dataLayout) {
  assert(reachingDef && reachingDef.getType() == slot.valueType &&
         "expected the reaching definition's type to match the slot's type");
  return createInsertAndCast(builder, getLoc(), getValue(), reachingDef,
                             dataLayout);
}

bool LLVM::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it is not volatile).
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         areConversionCompatible(dataLayout, getResult().getType(),
                                 slot.valueType,
                                 /*narrowingConversion=*/true) &&
         !getVolatile_();
}

DeletionKind LLVM::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  Value newResult = createExtractAndCast(builder, getLoc(), reachingDefinition,
                                         getResult().getType(), dataLayout);
  getResult().replaceAllUsesWith(newResult);
  return DeletionKind::Delete;
}

bool LLVM::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr &&
         areConversionCompatible(dataLayout, slot.valueType,
                                 getValue().getType(),
                                 /*narrowingConversion=*/false) &&
         !getVolatile_();
}

DeletionKind LLVM::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    OpBuilder &builder, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}

/// Checks if `slot` can be accessed through the provided access type.
static bool isValidAccessType(const MemorySlot &slot, Type accessType,
                              const DataLayout &dataLayout) {
  return dataLayout.getTypeSize(accessType) <=
         dataLayout.getTypeSize(slot.valueType);
}

LogicalResult LLVM::LoadOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(getAddr() != slot.ptr ||
                 isValidAccessType(slot, getType(), dataLayout));
}

LogicalResult LLVM::StoreOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success(getAddr() != slot.ptr ||
                 isValidAccessType(slot, getValue().getType(), dataLayout));
}

/// Returns the subslot's type at the requested index.
static Type getTypeAtIndex(const DestructurableMemorySlot &slot,
                           Attribute index) {
  auto subelementIndexMap =
      cast<DestructurableTypeInterface>(slot.valueType).getSubelementIndexMap();
  if (!subelementIndexMap)
    return {};
  assert(!subelementIndexMap->empty());

  // Note: Returns a null-type when no entry was found.
  return subelementIndexMap->lookup(index);
}

bool LLVM::LoadOp::canRewire(const DestructurableMemorySlot &slot,
                             SmallPtrSetImpl<Attribute> &usedIndices,
                             SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                             const DataLayout &dataLayout) {
  if (getVolatile_())
    return false;

  // A load always accesses the first element of the destructured slot.
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  Type subslotType = getTypeAtIndex(slot, index);
  if (!subslotType)
    return false;

  // The access can only be replaced when the subslot is read within its bounds.
  if (dataLayout.getTypeSize(getType()) > dataLayout.getTypeSize(subslotType))
    return false;

  usedIndices.insert(index);
  return true;
}

DeletionKind LLVM::LoadOp::rewire(const DestructurableMemorySlot &slot,
                                  DenseMap<Attribute, MemorySlot> &subslots,
                                  OpBuilder &builder,
                                  const DataLayout &dataLayout) {
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  auto it = subslots.find(index);
  assert(it != subslots.end());

  getAddrMutable().set(it->getSecond().ptr);
  return DeletionKind::Keep;
}

bool LLVM::StoreOp::canRewire(const DestructurableMemorySlot &slot,
                              SmallPtrSetImpl<Attribute> &usedIndices,
                              SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                              const DataLayout &dataLayout) {
  if (getVolatile_())
    return false;

  // Storing the pointer to memory cannot be dealt with.
  if (getValue() == slot.ptr)
    return false;

  // A store always accesses the first element of the destructured slot.
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  Type subslotType = getTypeAtIndex(slot, index);
  if (!subslotType)
    return false;

  // The access can only be replaced when the subslot is read within its bounds.
  if (dataLayout.getTypeSize(getValue().getType()) >
      dataLayout.getTypeSize(subslotType))
    return false;

  usedIndices.insert(index);
  return true;
}

DeletionKind LLVM::StoreOp::rewire(const DestructurableMemorySlot &slot,
                                   DenseMap<Attribute, MemorySlot> &subslots,
                                   OpBuilder &builder,
                                   const DataLayout &dataLayout) {
  auto index = IntegerAttr::get(IntegerType::get(getContext(), 32), 0);
  auto it = subslots.find(index);
  assert(it != subslots.end());

  getAddrMutable().set(it->getSecond().ptr);
  return DeletionKind::Keep;
}

//===----------------------------------------------------------------------===//
// Interfaces for discardable OPs
//===----------------------------------------------------------------------===//

/// Conditions the deletion of the operation to the removal of all its uses.
bool mlir::LLVM::forwardToUsers(Operation *op,
                                SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

//===----------------------------------------------------------------------===//
// Interfaces for GEPOp
//===----------------------------------------------------------------------===//

static bool hasAllZeroIndices(LLVM::GEPOp gepOp) {
  return llvm::all_of(gepOp.getIndices(), [](auto index) {
    auto indexAttr = llvm::dyn_cast_if_present<IntegerAttr>(index);
    return indexAttr && indexAttr.getValue() == 0;
  });
}

bool LLVM::GEPOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  // GEP can be removed as long as it is a no-op and its users can be removed.
  // `inrange` is only valid on constant GEP expressions, so an inrange GEP on
  // an alloca is illegal and we bail out.
  if (getInrangeAttr() || !hasAllZeroIndices(*this))
    return false;
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::GEPOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

/// Returns the amount of bytes the provided GEP elements will offset the
/// pointer by. Returns nullopt if no constant offset could be computed.
static std::optional<uint64_t> gepToByteOffset(const DataLayout &dataLayout,
                                               LLVM::GEPOp gep) {
  // Collects all indices.
  SmallVector<uint64_t> indices;
  for (auto index : gep.getIndices()) {
    auto constIndex = dyn_cast<IntegerAttr>(index);
    if (!constIndex)
      return {};
    int64_t gepIndex = constIndex.getInt();
    // Negative indices are not supported.
    if (gepIndex < 0)
      return {};
    indices.push_back(gepIndex);
  }

  Type currentType = gep.getElemType();
  uint64_t offset = indices[0] * dataLayout.getTypeSize(currentType);

  for (uint64_t index : llvm::drop_begin(indices)) {
    bool shouldCancel =
        TypeSwitch<Type, bool>(currentType)
            .Case([&](LLVM::LLVMArrayType arrayType) {
              offset +=
                  index * dataLayout.getTypeSize(arrayType.getElementType());
              currentType = arrayType.getElementType();
              return false;
            })
            .Case([&](LLVM::LLVMStructType structType) {
              offset += structType.getElementOffset(dataLayout, index);
              currentType = structType.getBody()[index];
              return false;
            })
            .Default([&](Type type) {
              LDBG() << "[sroa] Unsupported type for offset computations"
                     << type;
              return true;
            });

    if (shouldCancel)
      return std::nullopt;
  }

  return offset;
}

namespace {
/// A struct that stores both the index into the aggregate type of the slot as
/// well as the corresponding byte offset in memory.
struct SubslotAccessInfo {
  /// The parent slot's index that the access falls into.
  uint32_t index;
  /// The offset into the subslot of the access.
  uint64_t subslotOffset;
};
} // namespace

/// Computes subslot access information for an access into `slot` with the given
/// offset.
/// Returns nullopt when the offset is out-of-bounds or when the access is into
/// the padding of `slot`.
static std::optional<SubslotAccessInfo>
getSubslotAccessInfo(const DestructurableMemorySlot &slot,
                     const DataLayout &dataLayout, LLVM::GEPOp gep) {
  std::optional<uint64_t> offset = gepToByteOffset(dataLayout, gep);
  if (!offset)
    return {};

  // Helper to check that a constant index is in the bounds of the GEP index
  // representation. LLVM dialects's GEP arguments have a limited bitwidth, thus
  // this additional check is necessary.
  auto isOutOfBoundsGEPIndex = [](uint64_t index) {
    return index >= (1 << LLVM::kGEPConstantBitWidth);
  };

  Type type = slot.valueType;
  if (*offset >= dataLayout.getTypeSize(type))
    return {};
  return TypeSwitch<Type, std::optional<SubslotAccessInfo>>(type)
      .Case([&](LLVM::LLVMArrayType arrayType)
                -> std::optional<SubslotAccessInfo> {
        // Find which element of the array contains the offset.
        uint64_t elemSize = dataLayout.getTypeSize(arrayType.getElementType());
        uint64_t index = *offset / elemSize;
        if (isOutOfBoundsGEPIndex(index))
          return {};
        return SubslotAccessInfo{static_cast<uint32_t>(index),
                                 *offset - (index * elemSize)};
      })
      .Case([&](LLVM::LLVMStructType structType)
                -> std::optional<SubslotAccessInfo> {
        // Walk over the elements of the struct to find in which of
        // them the offset is.
        for (auto [index, elem] : llvm::enumerate(structType.getBody())) {
          uint64_t elemOffset = structType.getElementOffset(dataLayout, index);
          // If the offset is in padding, cancel the rewrite.
          if (*offset < elemOffset)
            return {};

          if (*offset < elemOffset + dataLayout.getTypeSize(elem)) {
            if (isOutOfBoundsGEPIndex(index))
              return {};
            // The offset is within this element, stop iterating the
            // struct and return the index.
            return SubslotAccessInfo{static_cast<uint32_t>(index),
                                     *offset - elemOffset};
          }
        }

        return {};
      });
}

/// Constructs a byte array type of the given size.
static LLVM::LLVMArrayType getByteArrayType(MLIRContext *context,
                                            unsigned size) {
  auto byteType = IntegerType::get(context, 8);
  return LLVM::LLVMArrayType::get(context, byteType, size);
}

LogicalResult LLVM::GEPOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  if (getBase() != slot.ptr)
    return success();
  std::optional<uint64_t> gepOffset = gepToByteOffset(dataLayout, *this);
  if (!gepOffset)
    return failure();
  uint64_t slotSize = dataLayout.getTypeSize(slot.valueType);
  // Check that the access is strictly inside the slot.
  if (*gepOffset >= slotSize)
    return failure();
  // Every access that remains in bounds of the remaining slot is considered
  // legal.
  mustBeSafelyUsed.emplace_back<MemorySlot>(
      {getRes(), getByteArrayType(getContext(), slotSize - *gepOffset)});
  return success();
}

bool LLVM::GEPOp::canRewire(const DestructurableMemorySlot &slot,
                            SmallPtrSetImpl<Attribute> &usedIndices,
                            SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                            const DataLayout &dataLayout) {
  if (!isa<LLVM::LLVMPointerType>(getBase().getType()))
    return false;

  if (getBase() != slot.ptr)
    return false;
  // `inrange` is only valid on constant GEP expressions, so an inrange GEP on
  // an alloca is illegal and SROA bails out.
  if (getInrangeAttr())
    return false;
  std::optional<SubslotAccessInfo> accessInfo =
      getSubslotAccessInfo(slot, dataLayout, *this);
  if (!accessInfo)
    return false;
  auto indexAttr =
      IntegerAttr::get(IntegerType::get(getContext(), 32), accessInfo->index);
  assert(slot.subelementTypes.contains(indexAttr));
  usedIndices.insert(indexAttr);

  // The remainder of the subslot should be accesses in-bounds. Thus, we create
  // a dummy slot with the size of the remainder.
  Type subslotType = slot.subelementTypes.lookup(indexAttr);
  uint64_t slotSize = dataLayout.getTypeSize(subslotType);
  LLVM::LLVMArrayType remainingSlotType =
      getByteArrayType(getContext(), slotSize - accessInfo->subslotOffset);
  mustBeSafelyUsed.emplace_back<MemorySlot>({getRes(), remainingSlotType});

  return true;
}

DeletionKind LLVM::GEPOp::rewire(const DestructurableMemorySlot &slot,
                                 DenseMap<Attribute, MemorySlot> &subslots,
                                 OpBuilder &builder,
                                 const DataLayout &dataLayout) {
  std::optional<SubslotAccessInfo> accessInfo =
      getSubslotAccessInfo(slot, dataLayout, *this);
  assert(accessInfo && "expected access info to be checked before");
  auto indexAttr =
      IntegerAttr::get(IntegerType::get(getContext(), 32), accessInfo->index);
  const MemorySlot &newSlot = subslots.at(indexAttr);

  auto byteType = IntegerType::get(builder.getContext(), 8);
  auto newPtr = builder.createOrFold<LLVM::GEPOp>(
      getLoc(), getResult().getType(), byteType, newSlot.ptr,
      ArrayRef<GEPArg>(accessInfo->subslotOffset), getNoWrapFlags());
  getResult().replaceAllUsesWith(newPtr);
  return DeletionKind::Delete;
}

SmallVector<Value> mlir::LLVM::AtomicCmpXchgOp::getAccessedOperands() {
  return {getPtr()};
}

SmallVector<Value> mlir::LLVM::AtomicRMWOp::getAccessedOperands() {
  return {getPtr()};
}

SmallVector<Value> mlir::LLVM::LoadOp::getAccessedOperands() {
  return {getAddr()};
}

SmallVector<Value> mlir::LLVM::StoreOp::getAccessedOperands() {
  return {getAddr()};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.cpp.inc"
