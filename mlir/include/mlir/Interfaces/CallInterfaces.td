//===- CallInterfaces.td - Call Interfaces for ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// related to call-like and callable operations. Each of which are defined along
// with the respective interface below.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CALLINTERFACES
#define MLIR_INTERFACES_CALLINTERFACES

include "mlir/IR/OpBase.td"

// `CallInterfaceCallable`: This is a type used to represent a single callable
// region. A callable is either a symbol, or an SSA value, that is referenced by
// a call-like operation. This represents the destination of the call.

/// Interface for call-like operations.
def CallOpInterface : OpInterface<"CallOpInterface"> {
  let description = [{
    A call-like operation is one that transfers control from one sub-routine to
    another. These operations may be traditional direct calls `call @foo`, or
    indirect calls to other operations `call_indirect %foo`. An operation that
    uses this interface, must *not* also provide the `CallableOpInterface`.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the callee of this call-like operation. A `callee` is either a
        reference to a symbol, via SymbolRefAttr, or a reference to a defined
        SSA value. If the reference is an SSA value, the SSA value corresponds
        to a region of a lambda-like operation.
      }],
      "::mlir::CallInterfaceCallable", "getCallableForCallee"
    >,
    InterfaceMethod<[{
        Returns the operands within this call that are used as arguments to the
        callee.
      }],
      "::mlir::Operation::operand_range", "getArgOperands"
    >,
  ];

  let extraClassDeclaration = [{
    /// Resolve the callable operation for given callee to a
    /// CallableOpInterface, or nullptr if a valid callable was not resolved.
    /// `symbolTable` is an optional parameter that will allow for using a
    /// cached symbol table for symbol lookups instead of performing an O(N)
    /// scan.
    Operation *resolveCallable(SymbolTableCollection *symbolTable = nullptr);
  }];
}

/// Interface for callable operations.
def CallableOpInterface : OpInterface<"CallableOpInterface"> {
  let description = [{
    A callable operation is one who represents a potential sub-routine, and may
    be a target for a call-like operation (those providing the CallOpInterface
    above). These operations may be traditional functional operation
    `func @foo(...)`, as well as function producing operations
    `%foo = dialect.create_function(...)`. These operations may only contain a
    single region, or subroutine.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the region on the current operation that is callable. This may
        return null in the case of an external callable object, e.g. an external
        function.
      }],
      "::mlir::Region *", "getCallableRegion"
    >,
    InterfaceMethod<[{
        Returns the results types that the callable region produces when
        executed.
      }],
      "::llvm::ArrayRef<::mlir::Type>", "getCallableResults"
    >,
  ];
}

#endif // MLIR_INTERFACES_CALLINTERFACES
