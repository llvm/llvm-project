//===- AllExtensions.cpp - All Linalg Dialect Extensions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Extensions/AllExtensions.h"
#include "mlir/Dialect/Linalg/Extensions/ShardingInterfaceImpl.h"

using namespace mlir;

void mlir::linalg::registerAllExtensions(DialectRegistry &registry) {
  registerShardingInterfaceExternalModels(registry);
}
