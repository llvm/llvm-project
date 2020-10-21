//===- Bufferize.cpp - scf bufferize pass ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Bufferize.h"
#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
struct SCFBufferizePass : public SCFBufferizeBase<SCFBufferizePass> {
  void runOnFunction() override {
    auto func = getOperation();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    ConversionTarget target(*context);

    populateBufferizeMaterializationLegality(target);
    populateSCFStructuralTypeConversionsAndLegality(context, typeConverter,
                                                    patterns, target);
    if (failed(applyPartialConversion(func, target, patterns)))
      return signalPassFailure();
  };
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createSCFBufferizePass() {
  return std::make_unique<SCFBufferizePass>();
}
