//===-- BufferViewFlowOpInterface.td - Buffer View Flow ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BUFFER_VIEW_FLOW_OP_INTERFACE
#define BUFFER_VIEW_FLOW_OP_INTERFACE

include "mlir/IR/OpBase.td"

def BufferViewFlowOpInterface :
    OpInterface<"BufferViewFlowOpInterface"> {
  let description = [{
    An op interface for the buffer view flow analysis. This interface describes
    buffer dependencies between operands and op results/region entry block
    arguments.
  }];
  let cppNamespace = "::mlir::bufferization";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Populate buffer dependencies between operands and op results/region
          entry block arguments.

          Implementations should register dependencies between an operand ("X")
          and an op result/region entry block argument ("Y") if Y may depend
          on X. Y depends on X if Y and X are the same buffer or if Y is a
          subview of X.

          Example:
          ```
          %r = arith.select %c, %m1, %m2 : memref<5xf32>
          ```
          In the above example, %0 may depend on %m1 or %m2 and a correct
          interface implementation should call:
          - "registerDependenciesFn(%m1, %r)".
          - "registerDependenciesFn(%m2, %r)"
        }],
        /*retType=*/"void",
        /*methodName=*/"populateDependencies",
        /*args=*/(ins
            "::mlir::bufferization::RegisterDependenciesFn"
                :$registerDependenciesFn)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return "true" if the given value is a terminal buffer. A buffer value
          is "terminal" if it cannot be traced back any further in the buffer
          view flow analysis. E.g., because the value is a newly allocated
          buffer or because there is not enough information available.

          Implementations can assume that the given SSA value is an OpResult of
          this operation or a region entry block argument of this operation.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isTerminalBuffer",
        /*args=*/(ins "Value":$value),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return false;"
      >,
  ];
}

#endif  // BUFFER_VIEW_FLOW_OP_INTERFACE
