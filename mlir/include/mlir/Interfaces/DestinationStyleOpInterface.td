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
    Ops that are in destination style have designated init operands, which act
    as initial tensor values for the results of the operation or the init
    buffers to which the results of the op will be written.

    Init operands must be ranked tensors or ranked memrefs. Input operands can
    have any type. All non-init operands are DPS inputs.

    It is assumed that the init operands of the op are the operands at
    position [start, end). The positions are defined by getDpsInitsPositionRange
    method.

    If the op has "tensor semantics", then the input operands are either scalars
    or ranked tensors. The init operands are ranked tensors and every tensor
    init is tied to a corresponding tensor OpResult in a 1-to-1 fashion.
    The i-th init tensor is tied to the i-th OpResult. The op may not have any
    additional OpResults. Init operands and their tied OpResults have the same
    type.

    If the op has "buffer semantics", then the input operands are either ranked
    memrefs or other non-tensor types, e.g. scalar types. Furthermore, the
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

    Each op that wants to implement DestinationStyleOpInterface needs to define
    the getDpsInitsPositionRange() method.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    // This method has to be defined for every DPS op.
    InterfaceMethod<
      /*desc=*/"Return start and end indices of the init operands range.",
      /*retTy=*/"std::pair<int64_t, int64_t>",
      /*methodName=*/"getDpsInitsPositionRange",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    //===------------------------------------------------------------------===//
    // Operands handling.
    //===------------------------------------------------------------------===//
    // The operand list is assumed to start with the input operands and end
    // with the init operands. Therefore, all methods to access the inputs
    // and inits can be expressed if the number of init operands is know.
    InterfaceMethod<
      /*desc=*/"Return the number of inits.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumDpsInits",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto [start, end] = $_op.getDpsInitsPositionRange();
        return end - start;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the init operands.",
      /*retTy=*/"::mlir::OpOperandVector",
      /*methodName=*/"getDpsInitOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto [start, end] = $_op.getDpsInitsPositionRange();

        ::mlir::OpOperandVector result;
        result.reserve(end - start);
        for (int i = start; i < end; ++i)
          result.push_back(&$_op->getOpOperand(i));
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the `i`-th init operand.",
      /*retTy=*/"::mlir::OpOperand *",
      /*methodName=*/"getDpsInitOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < $_op.getNumDpsInits());
        auto [start, end] = $_op.getDpsInitsPositionRange();
        return &$_op->getOpOperand(start + i);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Set the `i`-th init operand.",
      /*retTy=*/"void",
      /*methodName=*/"setDpsInitOperand",
      /*args=*/(ins "int64_t":$i, "::mlir::Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < $_op.getNumDpsInits());
        auto [start, end] = $_op.getDpsInitsPositionRange();
        $_op->setOperand(start + i, value);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the number of inputs.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumDpsInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getNumOperands() - $_op.getNumDpsInits();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the input operands.",
      /*retTy=*/"::mlir::OpOperandVector",
      /*methodName=*/"getDpsInputOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto [start, end] = $_op.getDpsInitsPositionRange();
        int64_t numInits = end - start;
        int64_t numOperands = $_op.getNumOperands();

        ::mlir::OpOperandVector result;
        result.reserve(numOperands - numInits);
        for (int i = 0; i < start; ++i)
          result.push_back(&$_op->getOpOperand(i));
        for (int i = end; i < numOperands; ++i)
          result.push_back(&$_op->getOpOperand(end + i));

        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return the `i`-th input operand.  }],
      /*retTy=*/"::mlir::OpOperand *",
      /*methodName=*/"getDpsInputOperand",
      /*args=*/(ins "int64_t":$i),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(i >= 0 && i < getNumDpsInputs());
        auto [start, end] = $_op.getDpsInitsPositionRange();
        return &$_op->getOpOperand(i < start ? i : i + end - start) ;
      }]
    >,
    //===------------------------------------------------------------------===//
    // Input and DpsInit arguments handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/"Return true if `opOperand` is an input.",
      /*retTy=*/"bool",
      /*methodName=*/"isDpsInput",
      /*args=*/(ins "::mlir::OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto [start, end] = $_op.getDpsInitsPositionRange();
        auto operandNumber = opOperand->getOperandNumber();
        return operandNumber < start || operandNumber >= end;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return true if `opOperand` is an init.",
      /*retTy=*/"bool",
      /*methodName=*/"isDpsInit",
      /*args=*/(ins "::mlir::OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto [start, end] = $_op.getDpsInitsPositionRange();
        auto operandNumber = opOperand->getOperandNumber();
        return operandNumber >= start && operandNumber < end;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return true if the `opOperand` is a scalar value.",
      /*retTy=*/"bool",
      /*methodName=*/"isScalar",
      /*args=*/(ins "::mlir::OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == $_op.getOperation());
        return !::llvm::isa<ShapedType>(opOperand->get().getType());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the OpResult that is tied to the given OpOperand.",
      /*retTy=*/"::mlir::OpResult",
      /*methodName=*/"getTiedOpResult",
      /*args=*/(ins "::mlir::OpOperand *":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == $_op.getOperation());

        auto [start, end] = $_op.getDpsInitsPositionRange();
        int64_t resultIndex = opOperand->getOperandNumber() - start;
        assert(resultIndex >= 0 &&
               resultIndex < $_op->getNumResults() );
        return $_op->getResult(resultIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the OpOperand that is tied to the given OpResult.",
      /*retTy=*/"::mlir::OpOperand *",
      /*methodName=*/"getTiedOpOperand",
      /*args=*/(ins "::mlir::OpResult":$opResult),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opResult.getDefiningOp() == $_op.getOperation());
        return $_op.getDpsInitOperand(opResult.getResultNumber());
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/"Return whether the op has only ranked MemRef input/inits.",
      /*retTy=*/"bool",
      /*methodName=*/"hasBufferSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op->getNumResults() == 0 &&
          ::llvm::all_of($_op->getOpOperands(),
            [&](::mlir::OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     ::llvm::isa<::mlir::MemRefType>(opOperand.get().getType());
            });
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return whether the op has only ranked tensor inputs/inits.",
      /*retTy=*/"bool",
      /*methodName=*/"hasTensorSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::all_of($_op->getOpOperands(),
          [&](::mlir::OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   ::llvm::isa<::mlir::RankedTensorType>(opOperand.get().getType());
          });
      }]
    >
  ];

  let verify = [{ return detail::verifyDestinationStyleOpInterface($_op); }];
  let verifyWithRegions = 1;
}


#endif // MLIR_DESTINATIONSTYLEOPINTERFACE
