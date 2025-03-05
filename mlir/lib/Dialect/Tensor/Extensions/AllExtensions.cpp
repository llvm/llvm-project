//===- AllExtensions.cpp - All Tensor Dialect Extensions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "mlir/Dialect/Tensor/Extensions/MeshShardingExtensions.h"

using namespace mlir;

void mlir::tensor::registerAllExtensions(DialectRegistry &registry) {
  registerShardingInterfaceExternalModels(registry);
}