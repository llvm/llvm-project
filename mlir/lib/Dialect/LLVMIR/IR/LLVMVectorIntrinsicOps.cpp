//===- LLVMVectorIntrinsicOps.cpp - LLVM dialect vector intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `VectorIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMVectorIntrinsicOps.h"

#include "IR/LLVMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::LLVM;

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

SmallVector<Value> mlir::LLVM::MaskedLoadOp::getAccessedOperands() {
  return {getData()};
}

SmallVector<Value> mlir::LLVM::MaskedStoreOp::getAccessedOperands() {
  return {getData()};
}

SmallVector<Value> mlir::LLVM::masked_gather::getAccessedOperands() {
  return {getPtrs()};
}

SmallVector<Value> mlir::LLVM::masked_scatter::getAccessedOperands() {
  return {getPtrs()};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMVectorIntrinsicOps.cpp.inc"
