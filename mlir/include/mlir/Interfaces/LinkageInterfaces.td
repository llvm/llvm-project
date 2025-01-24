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
