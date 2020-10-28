//===- StandardTypes.cpp - C Interface to MLIR Standard Types -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/StandardTypes.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;

/* ========================================================================== */
/* Integer types.                                                             */
/* ========================================================================== */

int mlirTypeIsAInteger(MlirType type) {
  return unwrap(type).isa<IntegerType>();
}

MlirType mlirIntegerTypeGet(MlirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(bitwidth, unwrap(ctx)));
}

MlirType mlirIntegerTypeSignedGet(MlirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(bitwidth, IntegerType::Signed, unwrap(ctx)));
}

MlirType mlirIntegerTypeUnsignedGet(MlirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(bitwidth, IntegerType::Unsigned, unwrap(ctx)));
}

unsigned mlirIntegerTypeGetWidth(MlirType type) {
  return unwrap(type).cast<IntegerType>().getWidth();
}

int mlirIntegerTypeIsSignless(MlirType type) {
  return unwrap(type).cast<IntegerType>().isSignless();
}

int mlirIntegerTypeIsSigned(MlirType type) {
  return unwrap(type).cast<IntegerType>().isSigned();
}

int mlirIntegerTypeIsUnsigned(MlirType type) {
  return unwrap(type).cast<IntegerType>().isUnsigned();
}

/* ========================================================================== */
/* Index type.                                                                */
/* ========================================================================== */

int mlirTypeIsAIndex(MlirType type) { return unwrap(type).isa<IndexType>(); }

MlirType mlirIndexTypeGet(MlirContext ctx) {
  return wrap(IndexType::get(unwrap(ctx)));
}

/* ========================================================================== */
/* Floating-point types.                                                      */
/* ========================================================================== */

int mlirTypeIsABF16(MlirType type) { return unwrap(type).isBF16(); }

MlirType mlirBF16TypeGet(MlirContext ctx) {
  return wrap(FloatType::getBF16(unwrap(ctx)));
}

int mlirTypeIsAF16(MlirType type) { return unwrap(type).isF16(); }

MlirType mlirF16TypeGet(MlirContext ctx) {
  return wrap(FloatType::getF16(unwrap(ctx)));
}

int mlirTypeIsAF32(MlirType type) { return unwrap(type).isF32(); }

MlirType mlirF32TypeGet(MlirContext ctx) {
  return wrap(FloatType::getF32(unwrap(ctx)));
}

int mlirTypeIsAF64(MlirType type) { return unwrap(type).isF64(); }

MlirType mlirF64TypeGet(MlirContext ctx) {
  return wrap(FloatType::getF64(unwrap(ctx)));
}

/* ========================================================================== */
/* None type.                                                                 */
/* ========================================================================== */

int mlirTypeIsANone(MlirType type) { return unwrap(type).isa<NoneType>(); }

MlirType mlirNoneTypeGet(MlirContext ctx) {
  return wrap(NoneType::get(unwrap(ctx)));
}

/* ========================================================================== */
/* Complex type.                                                              */
/* ========================================================================== */

int mlirTypeIsAComplex(MlirType type) {
  return unwrap(type).isa<ComplexType>();
}

MlirType mlirComplexTypeGet(MlirType elementType) {
  return wrap(ComplexType::get(unwrap(elementType)));
}

MlirType mlirComplexTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<ComplexType>().getElementType());
}

/* ========================================================================== */
/* Shaped type.                                                               */
/* ========================================================================== */

int mlirTypeIsAShaped(MlirType type) { return unwrap(type).isa<ShapedType>(); }

MlirType mlirShapedTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<ShapedType>().getElementType());
}

int mlirShapedTypeHasRank(MlirType type) {
  return unwrap(type).cast<ShapedType>().hasRank();
}

int64_t mlirShapedTypeGetRank(MlirType type) {
  return unwrap(type).cast<ShapedType>().getRank();
}

int mlirShapedTypeHasStaticShape(MlirType type) {
  return unwrap(type).cast<ShapedType>().hasStaticShape();
}

int mlirShapedTypeIsDynamicDim(MlirType type, intptr_t dim) {
  return unwrap(type).cast<ShapedType>().isDynamicDim(
      static_cast<unsigned>(dim));
}

int64_t mlirShapedTypeGetDimSize(MlirType type, intptr_t dim) {
  return unwrap(type).cast<ShapedType>().getDimSize(static_cast<unsigned>(dim));
}

int mlirShapedTypeIsDynamicSize(int64_t size) {
  return ShapedType::isDynamic(size);
}

int mlirShapedTypeIsDynamicStrideOrOffset(int64_t val) {
  return ShapedType::isDynamicStrideOrOffset(val);
}

/* ========================================================================== */
/* Vector type.                                                               */
/* ========================================================================== */

int mlirTypeIsAVector(MlirType type) { return unwrap(type).isa<VectorType>(); }

MlirType mlirVectorTypeGet(intptr_t rank, int64_t *shape,
                           MlirType elementType) {
  return wrap(
      VectorType::get(llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
                      unwrap(elementType)));
}

/* ========================================================================== */
/* Ranked / Unranked tensor type.                                             */
/* ========================================================================== */

int mlirTypeIsATensor(MlirType type) { return unwrap(type).isa<TensorType>(); }

int mlirTypeIsARankedTensor(MlirType type) {
  return unwrap(type).isa<RankedTensorType>();
}

int mlirTypeIsAUnrankedTensor(MlirType type) {
  return unwrap(type).isa<UnrankedTensorType>();
}

MlirType mlirRankedTensorTypeGet(intptr_t rank, int64_t *shape,
                                 MlirType elementType) {
  return wrap(RankedTensorType::get(
      llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType)));
}

MlirType mlirUnrankedTensorTypeGet(MlirType elementType) {
  return wrap(UnrankedTensorType::get(unwrap(elementType)));
}

/* ========================================================================== */
/* Ranked / Unranked MemRef type.                                             */
/* ========================================================================== */

int mlirTypeIsAMemRef(MlirType type) { return unwrap(type).isa<MemRefType>(); }

MlirType mlirMemRefTypeGet(MlirType elementType, intptr_t rank, int64_t *shape,
                           intptr_t numMaps, MlirAffineMap *affineMaps,
                           unsigned memorySpace) {
  SmallVector<AffineMap, 1> maps;
  (void)unwrapList(numMaps, affineMaps, maps);
  return wrap(
      MemRefType::get(llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
                      unwrap(elementType), maps, memorySpace));
}

MlirType mlirMemRefTypeContiguousGet(MlirType elementType, intptr_t rank,
                                     int64_t *shape, unsigned memorySpace) {
  return wrap(
      MemRefType::get(llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
                      unwrap(elementType), llvm::None, memorySpace));
}

intptr_t mlirMemRefTypeGetNumAffineMaps(MlirType type) {
  return static_cast<intptr_t>(
      unwrap(type).cast<MemRefType>().getAffineMaps().size());
}

MlirAffineMap mlirMemRefTypeGetAffineMap(MlirType type, intptr_t pos) {
  return wrap(unwrap(type).cast<MemRefType>().getAffineMaps()[pos]);
}

unsigned mlirMemRefTypeGetMemorySpace(MlirType type) {
  return unwrap(type).cast<MemRefType>().getMemorySpace();
}

int mlirTypeIsAUnrankedMemRef(MlirType type) {
  return unwrap(type).isa<UnrankedMemRefType>();
}

MlirType mlirUnrankedMemRefTypeGet(MlirType elementType, unsigned memorySpace) {
  return wrap(UnrankedMemRefType::get(unwrap(elementType), memorySpace));
}

unsigned mlirUnrankedMemrefGetMemorySpace(MlirType type) {
  return unwrap(type).cast<UnrankedMemRefType>().getMemorySpace();
}

/* ========================================================================== */
/* Tuple type.                                                                */
/* ========================================================================== */

int mlirTypeIsATuple(MlirType type) { return unwrap(type).isa<TupleType>(); }

MlirType mlirTupleTypeGet(MlirContext ctx, intptr_t numElements,
                          MlirType *elements) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typeRef = unwrapList(numElements, elements, types);
  return wrap(TupleType::get(typeRef, unwrap(ctx)));
}

intptr_t mlirTupleTypeGetNumTypes(MlirType type) {
  return unwrap(type).cast<TupleType>().size();
}

MlirType mlirTupleTypeGetType(MlirType type, intptr_t pos) {
  return wrap(unwrap(type).cast<TupleType>().getType(static_cast<size_t>(pos)));
}
