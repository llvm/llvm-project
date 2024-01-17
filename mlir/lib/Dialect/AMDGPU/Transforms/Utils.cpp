#include "mlir/Dialect/AMDGPU/Transforms/Utils.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::amdgpu;

// Define an interface for operations with indices
class IndicesInterface {
public:
  virtual std::optional<Operation::operand_range> getIndices() = 0;
  virtual void setIndices(ArrayRef<Value> indices) = 0;
  virtual ~IndicesInterface() = default;
};

// Implement a generic class that uses IndicesInterface
class OperationWithIndices : public IndicesInterface {
private:
  Operation *op;
  template <typename OpType>
  static std::optional<Operation::operand_range> getIndicesImpl(Operation *op) {
    if (auto specificOp = dyn_cast<OpType>(op))
      return specificOp.getIndices();
    return std::nullopt;
  }

  template <typename OpType>
  static void setIndicesImpl(Operation *op, ArrayRef<Value> indices) {
    if (auto specificOp = dyn_cast<OpType>(op))
      specificOp.getIndicesMutable().assign(indices);
  }

public:
  OperationWithIndices(Operation *op) : op(op) {}

  std::optional<Operation::operand_range> getIndices() override {
    auto result = getIndicesImpl<memref::LoadOp>(op);
    if (!result)
      result = getIndicesImpl<memref::StoreOp>(op);
    if (!result)
      result = getIndicesImpl<vector::LoadOp>(op);
    if (!result)
      result = getIndicesImpl<vector::StoreOp>(op);
    if (!result)
      result = getIndicesImpl<vector::TransferReadOp>(op);
    if (!result)
      result = getIndicesImpl<vector::TransferWriteOp>(op);

    return result;
  }

  void setIndices(ArrayRef<Value> indices) override {
    setIndicesImpl<memref::LoadOp>(op, indices);
    setIndicesImpl<memref::StoreOp>(op, indices);
    setIndicesImpl<vector::LoadOp>(op, indices);
    setIndicesImpl<vector::StoreOp>(op, indices);
    setIndicesImpl<vector::TransferReadOp>(op, indices);
    setIndicesImpl<vector::TransferWriteOp>(op, indices);
  }
};

std::optional<Operation::operand_range> amdgpu::getIndices(Operation *op) {
  OperationWithIndices operationWithIndices(op);
  return operationWithIndices.getIndices();
}

void amdgpu::setIndices(Operation *op, ArrayRef<Value> indices) {
  OperationWithIndices operationWithIndices(op);
  operationWithIndices.setIndices(indices);
}