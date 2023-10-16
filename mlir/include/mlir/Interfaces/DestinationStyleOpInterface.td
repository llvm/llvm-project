//===- DestinationStyleOpInterface.td ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DESTINATIONSTYLEOPINTERFACE
#define MLIR_DESTINATIONSTYLEOPINTERFACE

include "mlir/IR/OpBase.td"

def DestinationStyleOpInterface : OpInterface<"DestinationStyleOpInterface"> {
  let description = [{
    Ops that are in destination style have designated "init" operands, which act
    as initial tensor values for the results of the operation or the init
    buffers to which the results of the op will be written.

    Init operands must be ranked tensors or ranked memrefs. Input operands can
    have any type. All non-init operands are DPS inputs.

    The init operands of this op are specified by the OpOperands that
    the `getDpsInitsMutable` interface methods returns. The init operands must
    be a consecutive range of operands.

    If the op has "tensor semantics", then the input operands are either ranked
    tensors or other non-tensor/memref types ("scalars"). The init operands are
    ranked tensors and every tensor init is tied to a corresponding tensor
    OpResult in a 1-to-1 fashion. The i-th init tensor is tied to the i-th
    OpResult. The op may not have any additional OpResults. Init operands and
    their tied OpResults have the same type. Dynamic dimension sizes also match
    at runtime.

    If the op has "buffer semantics", then the input operands are either ranked
    memrefs or other non-tensor/memref types ("scalar" types). Furthermore, the
    init operands are ranked memrefs and the op has no results.

    Destination-passing style abstraction makes certain transformations easier.
    For example, tiling implementation can extract/insert slices from/into the
    destination of an op and use the resulting shaped value as an iter_arg in
    the surrounding loop structure. As another example, bufferization does not
    have to allocate new buffers for destinations (in case of in-place
    bufferization) and can directly reuse the existing destination buffer.

    Example of a destination style op: `%r = tensor.insert_slice %t into %d`,
    where `%t` is the single input and `%d` is the single init. `%d` is tied
    to `%r`.

    Example of an op that is not in destination style: `%r = tensor.pad %t`.
    This op is not in destination style because `%r` and `%t` have different
    shape.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the consecutive range of init operands.",
      /*retTy=*/"::llvm::MutableArrayRef<::mlir::OpOperand>",
      /*methodName=*/"getDpsInitsMutable",
      /*args=*/(ins)
    >,
  ];

  let extraSharedClassDeclaration = [{
    ::mlir::OperandRange getDpsInits() {
      auto initsMutable = $_op.getDpsInitsMutable();
      if (initsMutable.empty())
        return ::mlir::OperandRange($_op->operand_end(), $_op->operand_end());
      unsigned firstOperandIndex = initsMutable.begin()->getOperandNumber();
      return OperandRange(
          $_op->operand_begin() + firstOperandIndex,
          $_op->operand_begin() + firstOperandIndex + initsMutable.size());
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInits() { return $_op.getDpsInits().size(); }

    /// Return the `i`-th DPS init.
    ::mlir::OpOperand *getDpsInitOperand(int64_t i) {
      return &$_op.getDpsInitsMutable()[i];
    }

    /// Set the `i`-th DPS init.
    void setDpsInitOperand(int64_t i, Value value) {
      assert(i >= 0 && i < $_op.getNumDpsInits() && "invalid index");
      $_op->setOperand($_op.getDpsInits().getBeginOperandIndex() + i, value);
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInputs() {
      return $_op->getNumOperands() - $_op.getNumDpsInits();
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::OpOperand *> getDpsInputOperands() {
      ::llvm::SmallVector<::mlir::OpOperand *> result;
      int64_t numOperands = $_op->getNumOperands();
      ::mlir::OperandRange range = $_op.getDpsInits();
      if (range.empty()) {
        result.reserve(numOperands);
        for (int64_t i = 0; i < numOperands; ++i)
          result.push_back(&$_op->getOpOperand(i));
        return result;
      }
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      result.reserve(numOperands - numInits);
      for (int64_t i = 0; i < firstInitPos; ++i)
        result.push_back(&$_op->getOpOperand(i));
      for (int64_t i = firstInitPos + numInits; i < numOperands; ++i)
        result.push_back(&$_op->getOpOperand(i));
      return result;
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::Value> getDpsInputs() {
      return ::llvm::to_vector(::llvm::map_range(
          $_op.getDpsInputOperands(), [](OpOperand *o) { return o->get(); }));
    }

    /// Return the `i`-th DPS input operand.
    ::mlir::OpOperand *getDpsInputOperand(int64_t i) {
      ::mlir::OperandRange range = $_op.getDpsInits();
      if (range.empty())
        return &$_op->getOpOperand(i);
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      assert(i >= 0 && i < $_op->getNumOperands() - numInits
             && "invalid index");
      return &$_op->getOpOperand(
          i < firstInitPos ? i : i + firstInitPos + numInits);
    }

    /// Return "true" if `opOperand` is an "input".
    bool isDpsInput(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == $_op && "invalid operand");
      return !$_op.isDpsInit(opOperand);
    }

    /// Return "true" if `opOperand` is an "init".
    bool isDpsInit(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == $_op && "invalid operand");
      ::mlir::OperandRange range = $_op.getDpsInits();
      if (range.empty())
        return false;
      auto operandNumber = opOperand->getOperandNumber();
      return operandNumber >= range.getBeginOperandIndex()
          && operandNumber < range.getBeginOperandIndex() + range.size();
    }

    /// Return "true" if `opOperand` is a scalar value. A sclar is defined as
    /// neither a MemRef nor a tensor value.
    bool isScalar(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == $_op && "invalid operand");
      return !::llvm::isa<MemRefType, TensorType>(opOperand->get().getType());
    }

    /// Return the OpResult that is tied to the given OpOperand.
    ::mlir::OpResult getTiedOpResult(::mlir::OpOperand *opOperand) {
        assert(opOperand->getOwner() == $_op && "invalid operand");
        ::mlir::OperandRange range = $_op.getDpsInits();
        assert(!range.empty() && "op has no inits");
        int64_t resultIndex =
            opOperand->getOperandNumber() - range.getBeginOperandIndex();
        assert(resultIndex >= 0 &&
               resultIndex < $_op->getNumResults());
        return $_op->getResult(resultIndex);
    }

    /// Return the OpOperand that is tied to the given OpResult.
    ::mlir::OpOperand *getTiedOpOperand(::mlir::OpResult opResult) {
      assert(opResult.getDefiningOp() == $_op && "invalid opresult");
      return $_op.getDpsInitOperand(opResult.getResultNumber());
    }

    /// Return whether the op has buffer semantics. That is the case if the op
    /// has no ranked tensor operands and at least one memref operand.
    bool hasBufferSemantics() {
      // No tensors.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::RankedTensorType>(v.getType());
      };
      if (::llvm::any_of($_op->getOperands(), isTensor))
        return false;
      // At least one memref.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::MemRefType>(v.getType());
      };
      return llvm::any_of($_op->getOperands(), isMemref);
    }

    /// Return whether the op has tensor semantics. That is the case if the op
    /// has no memref operands and at least one ranked tensor operand.
    bool hasTensorSemantics() {
      // No memrefs.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::MemRefType>(v.getType());
      };
      if (::llvm::any_of($_op->getOperands(), isMemref))
        return false;
      // At least one tensor.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::RankedTensorType>(v.getType());
      };
      return llvm::any_of($_op->getOperands(), isTensor);
    }
  }];

  let verify = [{ return detail::verifyDestinationStyleOpInterface($_op); }];
  let verifyWithRegions = 1;
}


#endif // MLIR_DESTINATIONSTYLEOPINTERFACE
