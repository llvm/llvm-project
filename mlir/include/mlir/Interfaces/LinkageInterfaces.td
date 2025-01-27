//===-- LinkageInterfaces.td - Interfaces for Linkage ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for ops that interact with linkage.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LINKAGEINTERFACES
#define MLIR_INTERFACES_LINKAGEINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LinkableModuleOpInterface
//===----------------------------------------------------------------------===//

// Module-like ops that can be linked into other modules.
def LinkableModuleOpInterface : OpInterface<"LinkableModuleOpInterface"> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";

  let extraClassDeclaration = [{
    ::mlir::link::ComdatSymbolTable getComdatSymbolTable();
  }];
}

def GlobalValueLinkageOpInterface : OpInterface<"GlobalValueLinkageOpInterface"> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the linkage of the operation",
      /*returnType=*/  "::mlir::link::Linkage",
      /*methodName=*/  "getLinkage",
      /*args=*/        (ins)
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has external linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasExternalLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isExternalLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has available_externally linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasAvailableExternallyLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isAvailableExternallyLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has linkonce linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasLinkOnceLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isLinkOnceLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has linkonce any linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasLinkOnceAnyLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isLinkOnceAnyLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has linkonce ODR linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasLinkOnceODRLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isLinkOnceODRLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has weak linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasWeakLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isWeakLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has weak any linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasWeakAnyLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isWeakAnyLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has weak ODR linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasWeakODRLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isWeakODRLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has appending linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasAppendingLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isAppendingLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has internal linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasInternalLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isInternalLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has private linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasPrivateLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isPrivateLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has local linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasLocalLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isLocalLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has external weak linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasExternalWeakLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isExternalWeakLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/       "Returns true if the operation has common linkage",
      /*returnType=*/ "bool",
      /*methodName=*/ "hasCommonLinkage",
      /*args=*/       (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/[{
        return ::mlir::link::isCommonLinkage($_op.getLinkage());
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Returns true if the operation has a comdat",
      /*returnType=*/  "bool",
      /*methodName=*/  "hasComdat",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImplementation=*/[{
        return $_op.getComdatSelectionKind().has_value();
      }]
    >,
    InterfaceMethod<
      /*desc=*/        "Returns comdat selection kind of the operation",
      /*returnType=*/  "std::optional<::mlir::link::ComdatSelectionKind>",
      /*methodName=*/  "getComdatSelectionKind",
      /*args=*/        (ins)
    >
  ];
}

def GlobalAliasLinkageOpInterface
  : OpInterface<"GlobalAliasLinkageOpInterface",
                [GlobalValueLinkageOpInterface]> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";
}

def GlobalObjectLinkageOpInterface
  : OpInterface<"GlobalObjectLinkageOpInterface",
                [GlobalValueLinkageOpInterface]> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";
}

def FunctionLinkageOpInterface
  : OpInterface<"FunctionLinkageOpInterface",
                [GlobalObjectLinkageOpInterface]> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";
}

def GlobalFuncLinkageOpInterface
  : OpInterface<"GlobalFuncLinkageOpInterface",
                [GlobalObjectLinkageOpInterface]> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";
}

def GlobalVariableLinkageOpInterface
  : OpInterface<"GlobalVariableLinkageOpInterface",
                [GlobalObjectLinkageOpInterface]> {
  let description = [{ WIP }];
  let cppNamespace = "::mlir";
}

#endif // MLIR_INTERFACES_LINKAGEINTERFACES
