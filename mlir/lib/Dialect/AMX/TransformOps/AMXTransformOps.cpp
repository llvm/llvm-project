//===- AMXTransformOps.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/TransformOps/AMXTransformOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

using namespace mlir;
using namespace mlir::amx;
using namespace mlir::transform;

void mlir::transform::ApplyVectorContractToPackedTypeTiledDotProductPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  amx::populateVectorContractToPackedTypeTiledDotProductPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AMXTransformDialectExtension
    : public transform::TransformDialectExtension<
          AMXTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMXTransformDialectExtension)

  AMXTransformDialectExtension() {
    declareGeneratedDialect<amx::AMXDialect>();
    declareGeneratedDialect<LLVM::LLVMDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/AMX/TransformOps/AMXTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/AMX/TransformOps/AMXTransformOps.cpp.inc"

void mlir::amx::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<AMXTransformDialectExtension>();
}
