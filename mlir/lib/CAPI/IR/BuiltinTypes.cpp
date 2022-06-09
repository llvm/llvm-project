//===- BuiltinTypes.cpp - C Interface to MLIR Builtin Types ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Integer types.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAInteger(MlirType type) {
  return unwrap(type).isa<IntegerType>();
}

MlirType mlirIntegerTypeGet(MlirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(unwrap(ctx), bitwidth));
}

MlirType mlirIntegerTypeSignedGet(MlirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(unwrap(ctx), bitwidth, IntegerType::Signed));
}

MlirType mlirIntegerTypeUnsignedGet(MlirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(unwrap(ctx), bitwidth, IntegerType::Unsigned));
}

unsigned mlirIntegerTypeGetWidth(MlirType type) {
  return unwrap(type).cast<IntegerType>().getWidth();
}

bool mlirIntegerTypeIsSignless(MlirType type) {
  return unwrap(type).cast<IntegerType>().isSignless();
}

bool mlirIntegerTypeIsSigned(MlirType type) {
  return unwrap(type).cast<IntegerType>().isSigned();
}

bool mlirIntegerTypeIsUnsigned(MlirType type) {
  return unwrap(type).cast<IntegerType>().isUnsigned();
}

//===----------------------------------------------------------------------===//
// Index type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAIndex(MlirType type) { return unwrap(type).isa<IndexType>(); }

MlirType mlirIndexTypeGet(MlirContext ctx) {
  return wrap(IndexType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//

bool mlirTypeIsABF16(MlirType type) { return unwrap(type).isBF16(); }

MlirType mlirBF16TypeGet(MlirContext ctx) {
  return wrap(FloatType::getBF16(unwrap(ctx)));
}

bool mlirTypeIsAF16(MlirType type) { return unwrap(type).isF16(); }

MlirType mlirF16TypeGet(MlirContext ctx) {
  return wrap(FloatType::getF16(unwrap(ctx)));
}

bool mlirTypeIsAF32(MlirType type) { return unwrap(type).isF32(); }

MlirType mlirF32TypeGet(MlirContext ctx) {
  return wrap(FloatType::getF32(unwrap(ctx)));
}

bool mlirTypeIsAF64(MlirType type) { return unwrap(type).isF64(); }

MlirType mlirF64TypeGet(MlirContext ctx) {
  return wrap(FloatType::getF64(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// None type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsANone(MlirType type) { return unwrap(type).isa<NoneType>(); }

MlirType mlirNoneTypeGet(MlirContext ctx) {
  return wrap(NoneType::get(unwrap(ctx)));
}

//===----------------------------------------------------------------------===//
// Complex type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAComplex(MlirType type) {
  return unwrap(type).isa<ComplexType>();
}

MlirType mlirComplexTypeGet(MlirType elementType) {
  return wrap(ComplexType::get(unwrap(elementType)));
}

MlirType mlirComplexTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<ComplexType>().getElementType());
}

//===----------------------------------------------------------------------===//
// Shaped type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAShaped(MlirType type) { return unwrap(type).isa<ShapedType>(); }

MlirType mlirShapedTypeGetElementType(MlirType type) {
  return wrap(unwrap(type).cast<ShapedType>().getElementType());
}

bool mlirShapedTypeHasRank(MlirType type) {
  return unwrap(type).cast<ShapedType>().hasRank();
}

int64_t mlirShapedTypeGetRank(MlirType type) {
  return unwrap(type).cast<ShapedType>().getRank();
}

bool mlirShapedTypeHasStaticShape(MlirType type) {
  return unwrap(type).cast<ShapedType>().hasStaticShape();
}

bool mlirShapedTypeIsDynamicDim(MlirType type, intptr_t dim) {
  return unwrap(type).cast<ShapedType>().isDynamicDim(
      static_cast<unsigned>(dim));
}

int64_t mlirShapedTypeGetDimSize(MlirType type, intptr_t dim) {
  return unwrap(type).cast<ShapedType>().getDimSize(static_cast<unsigned>(dim));
}

bool mlirShapedTypeIsDynamicSize(int64_t size) {
  return ShapedType::isDynamic(size);
}

bool mlirShapedTypeIsDynamicStrideOrOffset(int64_t val) {
  return ShapedType::isDynamicStrideOrOffset(val);
}

//===----------------------------------------------------------------------===//
// Vector type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAVector(MlirType type) { return unwrap(type).isa<VectorType>(); }

MlirType mlirVectorTypeGet(intptr_t rank, const int64_t *shape,
                           MlirType elementType) {
  return wrap(
      VectorType::get(llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
                      unwrap(elementType)));
}

MlirType mlirVectorTypeGetChecked(MlirLocation loc, intptr_t rank,
                                  const int64_t *shape, MlirType elementType) {
  return wrap(VectorType::getChecked(
      unwrap(loc), llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType)));
}

//===----------------------------------------------------------------------===//
// Ranked / Unranked tensor type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsATensor(MlirType type) { return unwrap(type).isa<TensorType>(); }

bool mlirTypeIsARankedTensor(MlirType type) {
  return unwrap(type).isa<RankedTensorType>();
}

bool mlirTypeIsAUnrankedTensor(MlirType type) {
  return unwrap(type).isa<UnrankedTensorType>();
}

MlirType mlirRankedTensorTypeGet(intptr_t rank, const int64_t *shape,
                                 MlirType elementType, MlirAttribute encoding) {
  return wrap(RankedTensorType::get(
      llvm::makeArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType),
      unwrap(encoding)));
}

MlirType mlirRankedTensorTypeGetChecked(MlirLocation loc, intptr_t rank,
                                        const int64_t *shape,
                                        MlirType elementType,
                                        MlirAttribute encoding) {
  return wrap(RankedTensorType::getChecked(
      unwrap(loc), llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType), unwrap(encoding)));
}

MlirAttribute mlirRankedTensorTypeGetEncoding(MlirType type) {
  return wrap(unwrap(type).cast<RankedTensorType>().getEncoding());
}

MlirType mlirUnrankedTensorTypeGet(MlirType elementType) {
  return wrap(UnrankedTensorType::get(unwrap(elementType)));
}

MlirType mlirUnrankedTensorTypeGetChecked(MlirLocation loc,
                                          MlirType elementType) {
  return wrap(UnrankedTensorType::getChecked(unwrap(loc), unwrap(elementType)));
}

//===----------------------------------------------------------------------===//
// Ranked / Unranked MemRef type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAMemRef(MlirType type) { return unwrap(type).isa<MemRefType>(); }

MlirType mlirMemRefTypeGet(MlirType elementType, intptr_t rank,
                           const int64_t *shape, MlirAttribute layout,
                           MlirAttribute memorySpace) {
  return wrap(MemRefType::get(
      llvm::makeArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType),
      mlirAttributeIsNull(layout)
          ? MemRefLayoutAttrInterface()
          : unwrap(layout).cast<MemRefLayoutAttrInterface>(),
      unwrap(memorySpace)));
}

MlirType mlirMemRefTypeGetChecked(MlirLocation loc, MlirType elementType,
                                  intptr_t rank, const int64_t *shape,
                                  MlirAttribute layout,
                                  MlirAttribute memorySpace) {
  return wrap(MemRefType::getChecked(
      unwrap(loc), llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType),
      mlirAttributeIsNull(layout)
          ? MemRefLayoutAttrInterface()
          : unwrap(layout).cast<MemRefLayoutAttrInterface>(),
      unwrap(memorySpace)));
}

MlirType mlirMemRefTypeContiguousGet(MlirType elementType, intptr_t rank,
                                     const int64_t *shape,
                                     MlirAttribute memorySpace) {
  return wrap(MemRefType::get(
      llvm::makeArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType),
      MemRefLayoutAttrInterface(), unwrap(memorySpace)));
}

MlirType mlirMemRefTypeContiguousGetChecked(MlirLocation loc,
                                            MlirType elementType, intptr_t rank,
                                            const int64_t *shape,
                                            MlirAttribute memorySpace) {
  return wrap(MemRefType::getChecked(
      unwrap(loc), llvm::makeArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType), MemRefLayoutAttrInterface(), unwrap(memorySpace)));
}

MlirAttribute mlirMemRefTypeGetLayout(MlirType type) {
  return wrap(unwrap(type).cast<MemRefType>().getLayout());
}

MlirAffineMap mlirMemRefTypeGetAffineMap(MlirType type) {
  return wrap(unwrap(type).cast<MemRefType>().getLayout().getAffineMap());
}

MlirAttribute mlirMemRefTypeGetMemorySpace(MlirType type) {
  return wrap(unwrap(type).cast<MemRefType>().getMemorySpace());
}

bool mlirTypeIsAUnrankedMemRef(MlirType type) {
  return unwrap(type).isa<UnrankedMemRefType>();
}

MlirType mlirUnrankedMemRefTypeGet(MlirType elementType,
                                   MlirAttribute memorySpace) {
  return wrap(
      UnrankedMemRefType::get(unwrap(elementType), unwrap(memorySpace)));
}

MlirType mlirUnrankedMemRefTypeGetChecked(MlirLocation loc,
                                          MlirType elementType,
                                          MlirAttribute memorySpace) {
  return wrap(UnrankedMemRefType::getChecked(unwrap(loc), unwrap(elementType),
                                             unwrap(memorySpace)));
}

MlirAttribute mlirUnrankedMemrefGetMemorySpace(MlirType type) {
  return wrap(unwrap(type).cast<UnrankedMemRefType>().getMemorySpace());
}

//===----------------------------------------------------------------------===//
// Tuple type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsATuple(MlirType type) { return unwrap(type).isa<TupleType>(); }

MlirType mlirTupleTypeGet(MlirContext ctx, intptr_t numElements,
                          MlirType const *elements) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typeRef = unwrapList(numElements, elements, types);
  return wrap(TupleType::get(unwrap(ctx), typeRef));
}

intptr_t mlirTupleTypeGetNumTypes(MlirType type) {
  return unwrap(type).cast<TupleType>().size();
}

MlirType mlirTupleTypeGetType(MlirType type, intptr_t pos) {
  return wrap(unwrap(type).cast<TupleType>().getType(static_cast<size_t>(pos)));
}

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFunction(MlirType type) {
  return unwrap(type).isa<FunctionType>();
}

MlirType mlirFunctionTypeGet(MlirContext ctx, intptr_t numInputs,
                             MlirType const *inputs, intptr_t numResults,
                             MlirType const *results) {
  SmallVector<Type, 4> inputsList;
  SmallVector<Type, 4> resultsList;
  (void)unwrapList(numInputs, inputs, inputsList);
  (void)unwrapList(numResults, results, resultsList);
  return wrap(FunctionType::get(unwrap(ctx), inputsList, resultsList));
}

intptr_t mlirFunctionTypeGetNumInputs(MlirType type) {
  return unwrap(type).cast<FunctionType>().getNumInputs();
}

intptr_t mlirFunctionTypeGetNumResults(MlirType type) {
  return unwrap(type).cast<FunctionType>().getNumResults();
}

MlirType mlirFunctionTypeGetInput(MlirType type, intptr_t pos) {
  assert(pos >= 0 && "pos in array must be positive");
  return wrap(
      unwrap(type).cast<FunctionType>().getInput(static_cast<unsigned>(pos)));
}

MlirType mlirFunctionTypeGetResult(MlirType type, intptr_t pos) {
  assert(pos >= 0 && "pos in array must be positive");
  return wrap(
      unwrap(type).cast<FunctionType>().getResult(static_cast<unsigned>(pos)));
}

//===----------------------------------------------------------------------===//
// Opaque type.
//===----------------------------------------------------------------------===//

bool mlirTypeIsAOpaque(MlirType type) { return unwrap(type).isa<OpaqueType>(); }

MlirType mlirOpaqueTypeGet(MlirContext ctx, MlirStringRef dialectNamespace,
                           MlirStringRef typeData) {
  return wrap(
      OpaqueType::get(StringAttr::get(unwrap(ctx), unwrap(dialectNamespace)),
                      unwrap(typeData)));
}

MlirStringRef mlirOpaqueTypeGetDialectNamespace(MlirType type) {
  return wrap(unwrap(type).cast<OpaqueType>().getDialectNamespace().strref());
}

MlirStringRef mlirOpaqueTypeGetData(MlirType type) {
  return wrap(unwrap(type).cast<OpaqueType>().getTypeData());
}
