//===-- AllocationOpInterface.td - Allocation op interface -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface with allocation-related methods. It is used by the
// buffer deallocation pass.
//
//===----------------------------------------------------------------------===//

#ifndef ALLOCATION_OP_INTERFACE
#define ALLOCATION_OP_INTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AllocationOpInterface
//===----------------------------------------------------------------------===//

def AllocationOpInterface : OpInterface<"AllocationOpInterface"> {
  let description = [{
    This interface provides general allocation-related methods that are
    designed for allocation operations. For example, it offers the ability to
    construct associated deallocation and clone operations that are compatible
    with the current allocation operation.
  }];
  let cppNamespace = "::mlir::bufferization";

  let methods = [
    StaticInterfaceMethod<[{
        Builds a deallocation operation using the provided builder and the
        current allocation value (which refers to the current Op implementing
        this interface). The allocation value is a result of the current
        operation implementing this interface. If there is no compatible
        deallocation operation, this method can return ::llvm::None.
      }],
      "::mlir::Optional<::mlir::Operation*>", "buildDealloc",
      (ins "::mlir::OpBuilder&":$builder, "::mlir::Value":$alloc), [{}],
      /*defaultImplementation=*/[{ return std::nullopt; }]
    >,
    StaticInterfaceMethod<[{
        Builds a clone operation using the provided builder and the current
        allocation value (which refers to the current Op implementing this
        interface). The allocation value is a result of the current operation
        implementing this interface. If there is no compatible clone operation,
        this method can return ::llvm::None.
      }],
      "::mlir::Optional<::mlir::Value>", "buildClone",
      (ins "::mlir::OpBuilder&":$builder, "::mlir::Value":$alloc), [{}],
      /*defaultImplementation=*/[{ return std::nullopt; }]
    >
  ];
}

#endif  // ALLOCATION_OP_INTERFACE
