//===- LLVMCastOps.cpp - LLVM dialect conversion ops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Cast` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMCastOps.h"

#include "IR/LLVMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

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
  Type srcElemType = extractVectorElementType(getArg().getType());
  Type dstElemType = extractVectorElementType(getResult().getType());

  // TODO: 'bitcast' requires result and operand type to be identical in size.
  // Byte types may be cast from/to any type pointer constraints.
  if (isa<LLVMByteType>(srcElemType) || isa<LLVMByteType>(dstElemType))
    return success();

  auto resultType = llvm::dyn_cast<LLVMPointerType>(dstElemType);
  auto sourceType = llvm::dyn_cast<LLVMPointerType>(srcElemType);

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

LogicalResult LLVM::PtrToAddrOp::verify() {
  auto pointerType =
      cast<LLVM::LLVMPointerType>(extractVectorElementType(getArg().getType()));
  auto integerType = cast<IntegerType>(extractVectorElementType(getType()));

  auto dataLayout = DataLayout::closest(*this);
  std::optional<unsigned> width = dataLayout.getTypeIndexBitwidth(pointerType);
  assert(width && "pointers always return an index bitwidth");
  if (width != integerType.getWidth())
    return emitOpError("bit-width of integer result type ")
           << integerType << " must match the pointer bitwidth (" << *width
           << ") specified in the datalayout";

  return success();
}

//===----------------------------------------------------------------------===//
// Folder for LLVM::AddrSpaceCastOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::AddrSpaceCastOp::fold(FoldAdaptor adaptor) {
  return foldChainableCast(*this, adaptor);
}

Value LLVM::AddrSpaceCastOp::getViewSource() { return getArg(); }

bool LLVM::BitcastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::BitcastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::AddrSpaceCastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind LLVM::AddrSpaceCastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMCastOps.cpp.inc"
