//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinLinkerInterface.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;

struct BuiltinLinkerInterface : public link::LinkerInterface {
  using LinkerInterface::LinkerInterface;
};

void mlir::builtin::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<BuiltinLinkerInterface>();
  });
}
