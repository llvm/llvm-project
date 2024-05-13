//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include <optional>

using namespace mlir;
using namespace mlir::memref;

#include "mlir/Dialect/MemRef/IR/MemRefOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MemRefDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct MemRefInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::memref::MemRefDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
      >();
  addInterfaces<MemRefInlinerInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, MemRefDialect>();
  declarePromisedInterfaces<bufferization::AllocationOpInterface, AllocOp,
                            AllocaOp, ReallocOp>();
  declarePromisedInterfaces<RuntimeVerifiableOpInterface, CastOp, ExpandShapeOp,
                            LoadOp, ReinterpretCastOp, StoreOp, SubViewOp>();
  declarePromisedInterfaces<ValueBoundsOpInterface, AllocOp, AllocaOp, CastOp,
                            DimOp, GetGlobalOp, RankOp, SubViewOp>();
  declarePromisedInterface<DestructurableTypeInterface, MemRefType>();
}

/// Finds the unique dealloc operation (if one exists) for `allocValue`.
std::optional<Operation *> mlir::memref::findDealloc(Value allocValue) {
  Operation *dealloc = nullptr;
  for (Operation *user : allocValue.getUsers()) {
    if (!hasEffect<MemoryEffects::Free>(user, allocValue))
      continue;
    // If we found a realloc instead of dealloc, return std::nullopt.
    if (isa<memref::ReallocOp>(user))
      return std::nullopt;
    // If we found > 1 dealloc, return std::nullopt.
    if (dealloc)
      return std::nullopt;
    dealloc = user;
  }
  return dealloc;
}
