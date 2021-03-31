//===- Intrinsics.h - MLIR EDSC Intrinsics for MemRefOps --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MEMREF_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_MEMREF_EDSC_INTRINSICS_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using memref_alloc = ValueBuilder<memref::AllocOp>;
using memref_alloca = ValueBuilder<memref::AllocaOp>;
using memref_cast = ValueBuilder<memref::CastOp>;
using memref_dealloc = OperationBuilder<memref::DeallocOp>;
using memref_dim = ValueBuilder<memref::DimOp>;
using memref_load = ValueBuilder<memref::LoadOp>;
using memref_store = OperationBuilder<memref::StoreOp>;
using memref_sub_view = ValueBuilder<memref::SubViewOp>;
using memref_tensor_load = ValueBuilder<memref::TensorLoadOp>;
using memref_tensor_store = OperationBuilder<memref::TensorStoreOp>;
using memref_view = ValueBuilder<memref::ViewOp>;

/// Provide an index notation around memref_load and memref_store.
using MemRefIndexedValue =
    TemplatedIndexedValue<intrinsics::memref_load, intrinsics::memref_store>;
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

static inline ::llvm::SmallVector<mlir::Value, 8>
getMemRefSizes(mlir::Value memRef) {
  using namespace mlir;
  using namespace mlir::edsc;
  using namespace mlir::edsc::intrinsics;
  mlir::MemRefType memRefType = memRef.getType().cast<mlir::MemRefType>();
  assert(isStrided(memRefType) && "Expected strided MemRef type");

  SmallVector<mlir::Value, 8> res;
  res.reserve(memRefType.getShape().size());
  const auto &shape = memRefType.getShape();
  for (unsigned idx = 0, n = shape.size(); idx < n; ++idx) {
    if (shape[idx] == -1)
      res.push_back(memref_dim(memRef, idx));
    else
      res.push_back(std_constant_index(shape[idx]));
  }
  return res;
}

namespace mlir {
namespace edsc {

/// A MemRefBoundsCapture represents the information required to step through a
/// MemRef. It has placeholders for non-contiguous tensors that fit within the
/// Fortran subarray model.
/// At the moment it can only capture a MemRef with an identity layout map.
// TODO: Support MemRefs with layoutMaps.
class MemRefBoundsCapture : public BoundsCapture {
public:
  explicit MemRefBoundsCapture(Value v) {
    auto memrefSizeValues = getMemRefSizes(v);
    for (auto s : memrefSizeValues) {
      lbs.push_back(intrinsics::std_constant_index(0));
      ubs.push_back(s);
      steps.push_back(1);
    }
  }

  unsigned fastestVarying() const { return rank() - 1; }

private:
  Value base;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_EDSC_INTRINSICS_H_
