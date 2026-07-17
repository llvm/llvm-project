//===- SMTExtension.cpp - SMT extension for the Transform dialect ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/SMTExtension/SMTExtension.h"
#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class SMTExtension : public transform::TransformDialectExtension<SMTExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SMTExtension)

  SMTExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::transform::registerSMTExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<SMTExtension>();
}
