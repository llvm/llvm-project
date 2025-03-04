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

class NVVMRequiresSM<int Version, string ArchAccelerated = "false"> :
  ParamNativeOpTrait<"NVVMRequiresSM",
                    !cast<string>(Version) # "," # ArchAccelerated>;

#endif //NVVM_TRAITS
