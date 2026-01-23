//===-- OpenACCOpsInterfaces.td - OpenACC type interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_OPS_INTERFACES
#define OPENACC_OPS_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ComputeRegionOpInterface : OpInterface<"ComputeRegionOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for compute and loop construct operations.
  }];

  let methods = [
    InterfaceMethod<"Get alloca block", "::mlir::Block*", "getAllocaBlock",
      (ins), [{
        return &$_op.getRegion().front();
      }]>,
  ];
}

def PartialEntityAccessOpInterface : OpInterface<"PartialEntityAccessOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations that access a partial entity such as
    field or array element access.
  }];

  let methods = [
    InterfaceMethod<"Get the base entity being accessed", "::mlir::Value",
      "getBaseEntity", (ins)>,
    InterfaceMethod<"Check if this is a complete view of the entity", "bool",
      "isCompleteView", (ins), [{
        return false;
      }]>,
  ];
}

def AddressOfGlobalOpInterface : OpInterface<"AddressOfGlobalOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations that compute the address of a global variable
    or symbol.
  }];

  let methods = [
    InterfaceMethod<"Get the symbol reference to the global", "::mlir::SymbolRefAttr",
      "getSymbol", (ins)>,
  ];
}

def GlobalVariableOpInterface : OpInterface<"GlobalVariableOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations that define global variables. This interface
    provides a uniform way to query properties of global variables across
    different dialects.
  }];

  let methods = [
    InterfaceMethod<"Check if the global variable is constant", "bool",
      "isConstant", (ins), [{
        return false;
      }]>,
    InterfaceMethod<"Get the initialization region (returns nullptr if none)",
      "::mlir::Region*", "getInitRegion", (ins)>,
    InterfaceMethod<"Check if the global variable is device data",
      "bool", "isDeviceData", (ins), [{
        return false;
      }]>,
  ];
}

def IndirectGlobalAccessOpInterface : OpInterface<"IndirectGlobalAccessOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations that indirectly access global symbols.
    This interface provides a way to query which global symbols are referenced
    by an operation, which is useful for tracking dependencies and performing
    analysis on global variable usage.

    The symbolTable parameter is optional. If null, implementations will look up
    their own symbol table. This allows callers to pass a pre-existing symbol
    table for efficiency when querying multiple operations.
  }];

  let methods = [
    InterfaceMethod<"Get the symbols referenced by this operation",
      "void",
      "getReferencedSymbols",
      (ins "::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>&":$symbols,
           "::mlir::SymbolTable *":$symbolTable)>,
  ];
}

def OutlineRematerializationOpInterface : OpInterface<"OutlineRematerializationOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations that are candidates for rematerialization
    during outlining. These operations produce synthetic types or values
    that cannot be passed as arguments to outlined regions and must be
    rematerialized inside the region instead.

    Operations implementing this interface are expected to be memory effect
    free. Their results are typically used for type construction or providing
    metadata (such as shape information for arrays).
  }];

  let verify = [{
    if (!::mlir::isMemoryEffectFree($_op))
      return $_op->emitOpError("must be memory effect free");
    return ::mlir::success();
  }];
}

def OffloadRegionOpInterface : OpInterface<"OffloadRegionOpInterface"> {
  let cppNamespace = "::mlir::acc";

  let description = [{
    An interface for operations whose regions are targets for offloading
    and outlining. Operations implementing this interface indicate that
    their regions will be extracted and compiled separately (e.g., as
    device kernels or outlined functions).
  }];
}

#endif // OPENACC_OPS_INTERFACES
