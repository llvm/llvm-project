//===-- BufferizableOpInterface.td - Compreh. Bufferize ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACE
#define MLIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACE

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
      // TODO: Simplify this interface by removing `bufferizesToAliasOnly` and
      // `getInplaceableOpResult`. Instead, always use `getAliasingOpResult`. If
      // `getAliasingOpResult` returns a non-null value, we know that an alias
      // is created. If `bufferizesToMemoryRead` and `bufferizesToMemoryWrite`
      // return `false`, we know that the operands "bufferizes to alias only".
      InterfaceMethod<
        /*desc=*/[{
          Return `true` if the given OpOperand creates an alias but does neither
          read nor write. This implies that `bufferizesToMemoryRead` and
          `bufferizesToMemoryWrite` must return `false`. This method will never
          be called on OpOperands that do not have a tensor type.

          Examples of such ops are `tensor.extract_slice` and `tensor.cast`.
        }],
        /*retType=*/"bool",
        /*methodName=*/"bufferizesToAliasOnly",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // For better debugging, run `bufferizesToMemoryWrite`, which fires an
          // assertion when called on an op that should not have tensor
          // OpOperands.
          (void) cast<BufferizableOpInterface>($_op.getOperation())
              .bufferizesToMemoryWrite(opOperand);
          // Return `false` by default, as most ops are not "alias only".
          return false;
         }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the OpResult that can bufferize in-place with a given
          OpOperand. Return a null value if the OpOperand cannot bufferize
          in-place. This method will never be called on OpOperands that do not
          have a tensor type.
        }],
        /*retType=*/"OpResult",
        /*methodName=*/"getInplaceableOpResult",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          // Does not have to be implemented for ops without tensor OpOperands.
          llvm_unreachable("getInplaceableOpResult not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the OpResult that aliases with a given OpOperand when
          bufferized in-place. This is a superset of `getInplaceableOpResult`.
          This method will never be called on OpOperands that do not have a
          tensor type.
        }],
        /*retType=*/"OpResult",
        /*methodName=*/"getAliasingOpResult",
        /*args=*/(ins "OpOperand &":$opOperand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return cast<BufferizableOpInterface>($_op.getOperation())
              .getInplaceableOpResult(opOperand);
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
  ];
}

#endif  // MLIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACE
