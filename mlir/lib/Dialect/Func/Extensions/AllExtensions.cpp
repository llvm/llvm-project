//===- AllExtensions.cpp - All Func Dialect Extensions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"

using namespace mlir;

void mlir::func::registerAllExtensions(DialectRegistry &registry) {
  registerInlinerExtension(registry);
}
