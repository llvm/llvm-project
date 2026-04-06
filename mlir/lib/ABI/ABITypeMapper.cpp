//===- ABITypeMapper.cpp - Map MLIR types to ABI types --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ABI/ABITypeMapper.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Alignment.h"

using namespace mlir;
using namespace mlir::abi;

ABITypeMapper::ABITypeMapper(const DataLayout &dl)
    : DL(dl), Builder(Allocator) {}

const llvm::abi::Type *ABITypeMapper::map(mlir::Type type) {
  if (auto intTy = dyn_cast<mlir::IntegerType>(type))
    return mapIntegerType(intTy);

  if (auto floatTy = dyn_cast<mlir::FloatType>(type))
    return mapFloatType(floatTy);

  if (auto indexTy = dyn_cast<mlir::IndexType>(type))
    return mapIndexType(indexTy);

  if (auto vecTy = dyn_cast<mlir::VectorType>(type))
    return mapVectorType(vecTy);

  if (auto memRefTy = dyn_cast<mlir::MemRefType>(type))
    return mapMemRefType(memRefTy);

  if (auto noneTy = dyn_cast<mlir::NoneType>(type))
    return mapNoneType(noneTy);

  // For dialect-specific types, fall back to DataLayout queries.
  // The type must implement DataLayoutTypeInterface for this to work.
  llvm::TypeSize sizeInBits = DL.getTypeSizeInBits(type);
  uint64_t abiAlign = DL.getTypeABIAlignment(type);
  return Builder.getIntegerType(sizeInBits.getFixedValue(),
                                llvm::Align(abiAlign),
                                /*Signed=*/false);
}

const llvm::abi::Type *ABITypeMapper::mapIntegerType(mlir::IntegerType type) {
  uint64_t width = type.getWidth();
  uint64_t abiAlign = DL.getTypeABIAlignment(type);
  // MLIR signless integers are treated as signed for ABI purposes.
  // Most C/C++ integer types are signless in MLIR but behave as
  // signed for ABI classification (sign extension, etc.).
  bool isSigned = type.isSigned() || type.isSignless();
  return Builder.getIntegerType(width, llvm::Align(abiAlign), isSigned);
}

const llvm::abi::Type *ABITypeMapper::mapFloatType(mlir::FloatType type) {
  uint64_t abiAlign = DL.getTypeABIAlignment(type);
  const llvm::fltSemantics &semantics = type.getFloatSemantics();
  return Builder.getFloatType(semantics, llvm::Align(abiAlign));
}

const llvm::abi::Type *ABITypeMapper::mapIndexType(mlir::IndexType type) {
  llvm::TypeSize sizeInBits = DL.getTypeSizeInBits(type);
  uint64_t abiAlign = DL.getTypeABIAlignment(type);
  return Builder.getIntegerType(sizeInBits.getFixedValue(),
                                llvm::Align(abiAlign),
                                /*Signed=*/false);
}

const llvm::abi::Type *ABITypeMapper::mapVectorType(mlir::VectorType type) {
  const llvm::abi::Type *elementTy = map(type.getElementType());
  if (!elementTy)
    return nullptr;

  auto shape = type.getShape();
  // MLIR VectorType is always fixed-length and can be multi-dimensional.
  // Flatten to a single dimension for ABI purposes.
  uint64_t totalElements = 1;
  for (int64_t dim : shape)
    totalElements *= dim;

  llvm::ElementCount ec = llvm::ElementCount::getFixed(totalElements);
  uint64_t abiAlign = DL.getTypeABIAlignment(type);
  return Builder.getVectorType(elementTy, ec, llvm::Align(abiAlign));
}

const llvm::abi::Type *ABITypeMapper::mapMemRefType(mlir::MemRefType type) {
  // MemRef is pointer-like for ABI purposes.
  llvm::TypeSize sizeInBits = DL.getTypeSizeInBits(type);
  uint64_t abiAlign = DL.getTypeABIAlignment(type);
  unsigned addrSpace = 0;
  if (auto as = type.getMemorySpace())
    if (auto intAttr = dyn_cast<IntegerAttr>(as))
      addrSpace = intAttr.getInt();
  return Builder.getPointerType(sizeInBits.getFixedValue(),
                                llvm::Align(abiAlign), addrSpace);
}

const llvm::abi::Type *ABITypeMapper::mapNoneType(mlir::NoneType type) {
  return Builder.getVoidType();
}
