//===- IRDLExtension.cpp - IRDL extension for the Transform dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {
class IRDLExtension
    : public transform::TransformDialectExtension<IRDLExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IRDLExtension)

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.cpp.inc"
        >();

    declareDependentDialect<irdl::IRDLDialect>();
  }
};
} // namespace

void mlir::transform::registerIRDLExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<IRDLExtension>();
}
