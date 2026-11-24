//===- AllExtensions.cpp - Standard OpenACC external interfaces -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Extensions/AllExtensions.h"
#include "mlir/Dialect/OpenACC/Extensions/GPUInterfaces.h"
#include "mlir/Dialect/OpenACC/Extensions/LLVMInterfaces.h"
#include "mlir/Dialect/OpenACC/Extensions/MemRefInterfaces.h"

void mlir::acc::registerAllExtensions(DialectRegistry &registry) {
  registerGPUInterfaces(registry);
  registerLLVMInterfaces(registry);
  registerMemRefInterfaces(registry);
}
