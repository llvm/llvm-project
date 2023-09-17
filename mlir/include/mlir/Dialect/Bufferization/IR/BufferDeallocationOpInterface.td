//===-- BufferDeallocationOpInterface.td -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFER_DEALLOCATION_OP_INTERFACE
#define BUFFER_DEALLOCATION_OP_INTERFACE

include "mlir/IR/OpBase.td"

def BufferDeallocationOpInterface :
    OpInterface<"BufferDeallocationOpInterface"> {
  let description = [{
    An op interface for Buffer Deallocation. Ops that implement this interface
    can provide custom logic for computing the ownership of OpResults, modify
    the operation to properly pass the ownership values around, and insert
    `bufferization.dealloc` operations when necessary.
  }];
  let cppNamespace = "::mlir::bufferization";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          This method takes the current deallocation state and transformation
          options and updates the deallocation state as necessary for the
          operation implementing this interface. It may also insert
          `bufferization.dealloc` operations and rebuild itself with different
          result types. For operations implementing this interface all other
          interface handlers (e.g., default handlers for interfaces like
          RegionBranchOpInterface, CallOpInterface, etc.) are skipped by the
          deallocation pass. On success, either the current operation or one of
          the newly inserted operations is returned from which on the driver
          should continue the processing. On failure, the deallocation pass
          will terminate. It is recommended to emit a useful error message in
          that case.
        }],
        /*retType=*/"FailureOr<Operation *>",
        /*methodName=*/"process",
        /*args=*/(ins "DeallocationState &":$state,
                      "const DeallocationOptions &":$options)>,
      InterfaceMethod<
        /*desc=*/[{
          This method allows the implementing operation to specify custom logic
          to materialize an ownership indicator value for the given MemRef typed
          value it defines (including block arguments of nested regions). Since
          the operation itself has more information about its semantics the
          materialized IR can be more efficient compared to the default
          implementation and avoid cloning MemRefs and/or doing alias checking
          at runtime.
          Note that the same logic could also be implemented in the 'process'
          method above, however, the IR is always materialized then. If
          it's desirable to only materialize the IR to compute an updated
          ownership indicator when needed, it should be implemented using this
          method (which is especially important if operations are created that
          cannot be easily canonicalized away anymore).
        }],
        /*retType=*/"std::pair<Value, Value>",
        /*methodName=*/"materializeUniqueOwnershipForMemref",
        /*args=*/(ins "DeallocationState &":$state,
                      "const DeallocationOptions &":$options,
                      "OpBuilder &":$builder,
                      "Value":$memref),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
          return state.getMemrefWithUniqueOwnership(
            builder, memref, memref.getParentBlock());
        }]>,
  ];
}

#endif  // BUFFER_DEALLOCATION_OP_INTERFACE
