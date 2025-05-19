//===-- NVVMRequiresSMTraits.td - NVVM Requires SM Traits --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines traits for the NVVM Dialect in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef NVVM_REQUIRES_SM_TRAITS
#define NVVM_REQUIRES_SM_TRAITS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// Interface for NVVM Ops with the NVVMRequiresSM parametric trait
def RequiresSMInterface: OpInterface<"RequiresSMInterface"> {
  let cppNamespace = "::mlir::NVVM";
  let methods = [
    InterfaceMethod<
      "Get the SM version required by the op from the trait", 
      "const mlir::NVVM::NVVMCheckSMVersion", "getRequiredMinSMVersion"
    >
  ];
}

class NVVMRequiresSM<int minVersion> :
  ParamNativeOpTrait<"NVVMRequiresSM", !cast<string>(minVersion)>;

class StrJoin<string sep, list<string> str_list> {
  string ret = !foldl("", str_list, a, b,
               !if(!eq(a, ""), b, !if(!eq(b, ""), a, !strconcat(a, sep, b))));
}

class NVVMRequiresSMa<list<int> smVersions> :
  ParamNativeOpTrait<"NVVMRequiresSMa",
                      StrJoin<",", !foreach(vers, smVersions,
                                            !cast<string>(vers))>.ret>;

#endif //NVVM_REQUIRES_SM_TRAITS
