//===- OpenACCUtilsType.cpp - OpenACC Type Utilities ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace acc {

static std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignmentHelper(Type ty, ModuleOp module, const DataLayout &dl,
                              OpenACCSupport *support) {
  if (support)
    return support->getTypeSizeAndAlignment(ty, module);
  return getTypeSizeAndAlignment(ty, module, dl);
}

std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignment(Type ty, ModuleOp module, const DataLayout &dl,
                        OpenACCSupport *support) {
  if (ty.isIntOrIndexOrFloat() ||
      isa<ComplexType, VectorType, DataLayoutTypeInterface>(ty))
    return TypeSizeAndAlignment{
        dl.getTypeSize(ty),
        llvm::TypeSize::getFixed(dl.getTypeABIAlignment(ty))};

  // Product of element size and static dimensions; no inter-element padding or
  // array alignment rules are applied. This is acceptable as per API
  // documentation.
  if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
    if (!memrefTy.hasStaticShape())
      return std::nullopt;
    auto elemSizeAndAlignment = getTypeSizeAndAlignmentHelper(
        memrefTy.getElementType(), module, dl, support);
    if (!elemSizeAndAlignment)
      return std::nullopt;
    int64_t totalSize = elemSizeAndAlignment->first.getFixedValue();
    int64_t alignment = elemSizeAndAlignment->second.getFixedValue();
    for (int64_t dim : memrefTy.getShape())
      totalSize *= dim;
    return TypeSizeAndAlignment{llvm::TypeSize::getFixed(totalSize),
                                llvm::TypeSize::getFixed(alignment)};
  }

  // Sum of member sizes with no padding between members or tuple alignment
  // rules applied. This is acceptable as per API documentation.
  if (auto tupleTy = dyn_cast<TupleType>(ty)) {
    if (tupleTy.size() == 0)
      return std::nullopt;
    auto sizeAndAlignment =
        getTypeSizeAndAlignmentHelper(tupleTy.getType(0), module, dl, support);
    if (!sizeAndAlignment)
      return std::nullopt;
    llvm::TypeSize size = sizeAndAlignment->first;
    for (unsigned i = 1, e = tupleTy.size(); i < e; ++i) {
      auto next = getTypeSizeAndAlignmentHelper(tupleTy.getType(i), module, dl,
                                                support);
      if (!next)
        return std::nullopt;
      size += next->first;
    }
    return TypeSizeAndAlignment{size, sizeAndAlignment->second};
  }

  if (isa<FunctionType>(ty))
    return getTypeSizeAndAlignmentHelper(
        LLVM::LLVMPointerType::get(ty.getContext()), module, dl, support);

  return std::nullopt;
}

std::optional<TypeSizeAndAlignment>
getTypeSizeAndAlignment(Type ty, ModuleOp module, OpenACCSupport *support) {
  std::optional<DataLayout> dl = getDataLayout(module);
  if (!dl)
    return std::nullopt;
  return getTypeSizeAndAlignment(ty, module, *dl, support);
}

} // namespace acc
} // namespace mlir
