//===-- NVVMTraits.td - NVVM Traits ------------------------*- tablegen -*-===//
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

#ifndef NVVM_TRAITS
#define NVVM_TRAITS

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

class NVVMRequiresSM<int Version, string ArchAccelerated = "false"> :
  ParamNativeOpTrait<"NVVMRequiresSM",
                    !cast<string>(Version) # "," # ArchAccelerated>;

#endif //NVVM_TRAITS
