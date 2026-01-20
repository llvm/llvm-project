//===- X86VectorTransformOps.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/TransformOps/X86VectorTransformOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

using namespace mlir;
using namespace mlir::x86vector;
using namespace mlir::transform;

void mlir::transform::ApplyVectorContractToFMAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86vector::populateVectorContractToFMAPatterns(patterns);
}

void mlir::transform::ApplyVectorContractToPackedTypeDotProductPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  x86vector::populateVectorContractToPackedTypeDotProductPatterns(patterns);
}

void mlir::transform::ApplyVectorContractBF16ToFMAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86vector::populateVectorContractBF16ToFMAPatterns(patterns);
}

void mlir::transform::ApplySinkVectorProducerOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86vector::populateSinkVectorProducerOpsPatterns(patterns);
}

void mlir::transform::ApplyShuffleVectorFMAOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  x86vector::populateShuffleVectorFMAOpsPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class X86VectorTransformDialectExtension
    : public transform::TransformDialectExtension<
          X86VectorTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      X86VectorTransformDialectExtension)

  X86VectorTransformDialectExtension() {
    declareGeneratedDialect<x86vector::X86VectorDialect>();
    declareGeneratedDialect<LLVM::LLVMDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/X86Vector/TransformOps/X86VectorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/X86Vector/TransformOps/X86VectorTransformOps.cpp.inc"

void mlir::x86vector::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<X86VectorTransformDialectExtension>();
}
