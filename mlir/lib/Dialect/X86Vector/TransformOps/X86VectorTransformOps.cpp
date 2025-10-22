//===- X86VectorTransformOps.cpp - Implementation of Vector transform ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/Transforms.h"

#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

using namespace mlir;
using namespace mlir::x86vector;
using namespace mlir::transform;



void mlir::transform::ApplyVectorContractNanokernelLoweringPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  vector::populateVectorTransferLoweringPatterns(patterns);//,
                                                 //getVectorSize());
}


#define GET_OP_CLASSES
#include "mlir/Dialect/X86Vector/TransformOps/X86VectorTransformOps.cpp.inc"

void mlir::x86vector::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<X86VectorTransformDialectExtension>();
}

