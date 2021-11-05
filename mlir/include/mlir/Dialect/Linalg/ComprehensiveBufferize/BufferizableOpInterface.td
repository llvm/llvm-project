//===-- BufferizableOpInterface.td - Compreh. Bufferize ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFERIZABLE_OP_INTERFACE
#define BUFFERIZABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def BufferizableOpInterface : OpInterface<"BufferizableOpInterface"> {
  let description = [{
    An op interface for Comprehensive Bufferization. Ops that implement this
    interface can be bufferized using Comprehensive Bufferization.
  }];
  let cppNamespace = "::mlir::linalg";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand bufferizes to a memory read. This
          method will never be called on OpOperands that do not have a tensor
          type.

          Note: It is always safe to consider an OpOperand as a memory read,
          even if it does actually not read; however, this can introduce
          unnecessary out-of-place bufferization decisions. The analysis of
          Comprehensive Bufferize considers OpOperands of unknown ops (that do
          not implement this interface) as reading OpOperands.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToMemoryRead",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("bufferizesToMemoryRead not implemented");
         }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand bufferizes to a memory write.
          This method will never be called on OpOperands that do not have a
          tensor type.

          Note: It is always safe to consider an OpOperand as a memory write,
          even if it does actually not write; however, this can introduce
          unnecessary out-of-place bufferization decisions. The analysis of
          Comprehensive Bufferize considers OpOperands of unknown ops (that do
          not implement this interface) as writing OpOperands.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToMemoryWrite",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("bufferizesToMemoryWrite not implemented");
         }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand creates an alias but does neither
          read nor write. This implies that `bufferizesToMemoryRead` and
          `bufferizesToMemoryWrite` must return `false`. This method will never
          be called on OpOperands that do not have a tensor type.

          Examples of such ops are `tensor.extract_slice` and `tensor.cast`.

          Note: This method is not meant to be reimplemented.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToAliasOnly",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        // TODO: This should be in methodBody instead of defaultImplementation.
        // Due to a bug in TableGen codegen, this does not compile.
        /*defaultImplementation=*/[{
          auto bufferizableOp =
              cast<BufferizableOpInterface>($_op.getOperation());
          return !bufferizableOp.bufferizesToMemoryRead(opOperand)
              && !bufferizableOp.bufferizesToMemoryWrite(opOperand)
              && static_cast<bool>(
                  bufferizableOp.getAliasingOpResult(opOperand));
         }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the OpResult that aliases with a given OpOperand when
          bufferized in-place. This method will never be called on OpOperands
          that do not have a tensor type.
        }],
        /*retType=*/"OpResult",
        /*methodName=*/"getAliasingOpResult",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("getAliasingOpResult not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the OpOperands that alias with a given OpResult when
          bufferized in-place. This method will never be called on OpResults
          that do not have a tensor type.

          Note: This method can return multiple OpOperands, indicating that the
          given OpResult may at runtime alias with any of the OpOperands. This
          is useful for branches and for ops such as `std.select`.
        }],
        /*retType=*/"SmallVector<OpOperand *>",
        /*methodName=*/"getAliasingOpOperand",
        /*args=*/(ins "OpResult":$opResult),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpResults.
          llvm_unreachable("getInplaceableOpResult not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the buffer relation between the given OpOperand and its
          aliasing OpResult when bufferized in-place. Most OpOperands have an
          "equivalence" relation.

          TODO: Support other relations such as "OpOperand is included in
          OpResult".
        }],
        /*retType=*/"BufferRelation",
        /*methodName=*/"bufferRelation",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("bufferRelation not implemented");
        }]
      >,
      // TODO: Simplify method signature: Pass an OpBuilder and a
      // BufferizationState object.
      InterfaceMethod<
        /*desc=*/[{
          Bufferize this op, i.e., rewrite it into a memref-based equivalent.
          `bvm` maps tensor values to memref values and this method should map
          tensor results to memref results after creating/modifying ops.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"bufferize",
        /*args=*/(ins "OpBuilder &":$b,
                      "BlockAndValueMapping &":$bvm,
                      "BufferizationAliasInfo &":$aliasInfo,
                      "AllocationCallbacks &":$allocationFn),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("bufferize not implemented");
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand can be written to in-place. This
          is the case for most ops, but some ops such as ConstantOp may
          bufferize to non-writable (read-only) memory locations. This method
          will never be called on OpResults that do not have a tensor type.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isWritable",
        /*args=*/(ins "OpResult":$opResult),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return true;
        }]
      >
  ];
}

#endif  // BUFFERIZABLE_OP_INTERFACE
