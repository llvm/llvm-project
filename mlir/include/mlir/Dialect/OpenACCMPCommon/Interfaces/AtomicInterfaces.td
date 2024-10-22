//===- DirectiveAtomicInterfaces.td - atomic interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operation interface for atomic operations used in OpenACC and
// OpenMP.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_MP_COMMON_INTERFACES_ATOMICINTERFACES
#define OPENACC_MP_COMMON_INTERFACES_ATOMICINTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def AtomicReadOpInterface : OpInterface<"AtomicReadOpInterface"> {
  let description = [{
    This interface is used for OpenACC/OpenMP dialect operation that performs an
    atomic read.

    The interface terminology uses `x` and `v` like the directive
    specifications:
      `v = x;`
    `x` is the address from where the value is atomically read.
    `v` is the address where the value is stored after reading.
  }];
  let cppNamespace = "::mlir::accomp";

  let methods = [
    InterfaceMethod<[{
        Common verifier for operation that implements atomic read interface.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyCommon",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.getX() == $_op.getV()) {
          return $_op.emitError(
            "read and write must not be to the same location for atomic reads");
        }
        return mlir::success();
      }]
    >,
    InterfaceMethod<[{
        Obtains `x` which is the address from where the value is atomically
        read.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getX",
      /*args=*/(ins)
    >,
    InterfaceMethod<[{
        Obtains `v` which is the address where the value is stored after
        reading.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getV",
      /*args=*/(ins)
    >,
  ];
}

def AtomicWriteOpInterface : OpInterface<"AtomicWriteOpInterface"> {
  let description = [{
    This interface is used for OpenACC/OpenMP dialect operation that performs an
    atomic write.

    The interface terminology uses `x` and `expr` like the directive
    specifications:
      `x = expr;`
    `x` is the address to where the `expr` is atomically written.
  }];
  let cppNamespace = "::mlir::accomp";

  let methods = [
    InterfaceMethod<[{
        Common verifier for operation that implements atomic write interface.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyCommon",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::Type elementType = $_op.getX().getType().getElementType();
        if (elementType && elementType != $_op.getExpr().getType())
          return $_op.emitError("address must dereference to value type");
        return mlir::success();
      }]
    >,
    InterfaceMethod<[{
        Obtains `x` which is the address to which the value is atomically
        written to.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getX",
      /*args=*/(ins)
    >,
    InterfaceMethod<[{
        Obtains `expr` which corresponds to the expression whose value is
        written to `x`.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getExpr",
      /*args=*/(ins)
    >,
  ];
}

def AtomicUpdateOpInterface : OpInterface<"AtomicUpdateOpInterface"> {
  let description = [{
    This interface is used for OpenACC/OpenMP dialect operation that performs an
    atomic update.

    The interface terminology uses `x` to specify the address where a value
    is atomically written/read.

    Since atomic update expression comes in many forms, this interface requires
    that the operation uses a region with a single argument to capture the
    expression.

    The region describes how to update the value of `x`. It takes the value at
    `x` as an input and must yield the updated value. Only the update to `x` is
    atomic. Generally the region must have only one instruction, but can
    potentially have more than one instructions too. The update is semantically
    similar to a compare-exchange loop based atomic update.
  }];
  let cppNamespace = "::mlir::accomp";

  let methods = [
    InterfaceMethod<[{
        Obtains `x` which is the address to which the value is atomically
        written to / read from.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getX",
      /*args=*/(ins)
    >,
    InterfaceMethod<[{
        Returns the first operation in atomic update region.
      }],
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"getFirstOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return &($_op.getRegion().front().getOperations().front());
      }]
    >,
    InterfaceMethod<[{
        Returns true if the new value is same as old value and the operation is
        a no-op, false otherwise.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isNoOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // The atomic update is a no-op if the terminator is the first and only
        // operation in its region.
        mlir::Operation* terminator =
          llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>($_op.getFirstOp());
        return terminator && terminator->getOperands().front() ==
          $_op.getRegion().front().getArgument(0);
      }]
    >,
    InterfaceMethod<[{
        Returns the new value if the operation is equivalent to just a write
        operation. Otherwise, returns nullptr.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getWriteOpVal",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::Operation* terminator =
          llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>($_op.getFirstOp());
        if (terminator && terminator->getOperands().front() !=
          $_op.getRegion().front().getArgument(0)) {
          return terminator->getOperands().front();
        }
        return nullptr;
      }]
    >,
    InterfaceMethod<[{
        Common verifier for operation that implements atomic update interface.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyCommon",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.getRegion().getNumArguments() != 1)
          return $_op.emitError("the region must accept exactly one argument");

        Type elementType = $_op.getX().getType().getElementType();
        if (elementType && elementType != $_op.getRegion().getArgument(0).getType()) {
          return $_op.emitError("the type of the operand must be a pointer type whose "
                          "element type is the same as that of the region argument");
        }

        return mlir::success();
      }]
    >,
    InterfaceMethod<[{
        Common verifier of the required region for operation that implements
        atomic update interface.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyRegionsCommon",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::Operation *terminator = $_op.getRegion().front().getTerminator();

        if (terminator->getOperands().size() != 1)
          return $_op.emitError("only updated value must be returned");

        if (terminator->getOperands().front().getType() !=
            $_op.getRegion().getArgument(0).getType())
          return $_op.emitError("input and yielded value must have the same type");

        return mlir::success();
      }]
    >,
  ];
}

def AtomicCaptureOpInterface : OpInterface<"AtomicCaptureOpInterface"> {
  let description = [{
    This interface is used for OpenACC/OpenMP dialect operation that performs an
    atomic capture.

    This interface requires a single region with two operations that each
    implement one of the atomic interfaces. It can be found in one of these
    forms:
      `{ atomic.update, atomic.read }`
      `{ atomic.read, atomic.update }`
      `{ atomic.read, atomic.write }`
  }];
  let cppNamespace = "::mlir::accomp";

  let methods = [
    InterfaceMethod<[{
        Returns the first operation in atomic capture region.
      }],
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"getFirstOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return &($_op.getRegion().front().getOperations().front());
      }]
    >,
    InterfaceMethod<[{
        Returns the second operation in atomic capture region.
      }],
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"getSecondOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto &ops = $_op.getRegion().front().getOperations();
        return ops.getNextNode(ops.front());
      }]
    >,
      InterfaceMethod<[{
        Common verifier of the required region for operation that implements
        atomic capture interface.
      }],
      /*retTy=*/"::llvm::LogicalResult",
      /*methodName=*/"verifyRegionsCommon",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Block::OpListType &ops = $_op.getRegion().front().getOperations();
        if (ops.size() != 3)
          return $_op.emitError()
                << "expected three operations in atomic.capture region (one "
                    "terminator, and two atomic ops)";
        auto &firstOp = ops.front();
        auto &secondOp = *ops.getNextNode(firstOp);
        auto firstReadStmt = dyn_cast<AtomicReadOpInterface>(firstOp);
        auto firstUpdateStmt = dyn_cast<AtomicUpdateOpInterface>(firstOp);
        auto secondReadStmt = dyn_cast<AtomicReadOpInterface>(secondOp);
        auto secondUpdateStmt = dyn_cast<AtomicUpdateOpInterface>(secondOp);
        auto secondWriteStmt = dyn_cast<AtomicWriteOpInterface>(secondOp);

        if (!((firstUpdateStmt && secondReadStmt) ||
              (firstReadStmt && secondUpdateStmt) ||
              (firstReadStmt && secondWriteStmt)))
          return ops.front().emitError()
                << "invalid sequence of operations in the capture region";
        if (firstUpdateStmt && secondReadStmt &&
            firstUpdateStmt.getX() != secondReadStmt.getX())
          return firstUpdateStmt.emitError()
                << "updated variable in atomic.update must be captured in "
                    "second operation";
        if (firstReadStmt && secondUpdateStmt &&
            firstReadStmt.getX() != secondUpdateStmt.getX())
          return firstReadStmt.emitError()
                << "captured variable in atomic.read must be updated in second "
                    "operation";
        if (firstReadStmt && secondWriteStmt &&
            firstReadStmt.getX() != secondWriteStmt.getX())
          return firstReadStmt.emitError()
                << "captured variable in atomic.read must be updated in "
                    "second operation";

        return mlir::success();
      }]
    >,
  ];
}

#endif // OPENACC_MP_COMMON_INTERFACES_ATOMICINTERFACES
