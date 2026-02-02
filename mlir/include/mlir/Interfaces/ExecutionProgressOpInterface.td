//===- ExecutionProgressOpInterface.td - Interface Decl. -*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the ExecutionProgressOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_EXECUTION_PROGRESS_OP_INTERFACE
#define MLIR_INTERFACES_EXECUTION_PROGRESS_OP_INTERFACE

include "mlir/IR/OpBase.td"

def ExecutionProgressOpInterface : OpInterface<"ExecutionProgressOpInterface"> {
  let description = [{
    This interface models execution progress properties of operations.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Operations that "must progress" are required to return normally (control
        flow reaches the next operation) or interact with the environment in an
        observable way (e.g., volatile memory access, I/O, synchronization or
        program termination). If a "must progress" op executes indefinitely
        without any observable interaction, it may be erased.
        
        See LLVM "llvm.loop.mustprogress" / "mustprogress" function attribute
        for more details.

        Operations must progress by default.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"mustProgress",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return true;
      }]
    >
  ];
}

#endif // MLIR_INTERFACES_EXECUTION_PROGRESS_OP_INTERFACE
