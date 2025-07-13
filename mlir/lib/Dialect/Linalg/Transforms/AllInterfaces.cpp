//===- AllInterfaces.cpp - ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"

#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/MeshShardingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

void mlir::linalg::registerAllDialectInterfaceImplementations(
    DialectRegistry &registry) {
  registerBufferizableOpInterfaceExternalModels(registry);
  registerMeshShardingInterfaceExternalModels(registry);
  registerSubsetOpInterfaceExternalModels(registry);
  registerTilingInterfaceExternalModels(registry);
  registerValueBoundsOpInterfaceExternalModels(registry);
}
