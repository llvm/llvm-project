#include "mlir/Dialect/AMDGPU/Transforms/Utils.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::amdgpu;

std::optional<Operation::operand_range> amdgpu::getIndices(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndices();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndices();
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  return std::nullopt;
}

void amdgpu::setIndices(Operation *op, ArrayRef<Value> indices) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getIndicesMutable().assign(indices);
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getIndicesMutable().assign(indices);
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndicesMutable().assign(indices);
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndicesMutable().assign(indices);
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndicesMutable().assign(indices);
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndicesMutable().assign(indices);
}
