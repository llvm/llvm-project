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

    // TODO: Move this to BufferizeTypeConverter's constructor.
    //
    // This doesn't currently play well with "finalizing" bufferizations (ones
    // that expect all materializations to be gone). In particular, there seems
    // to at least be a double-free in the dialect conversion framework
    // when this materialization gets inserted and then folded away because
    // it is marked as illegal.
    typeConverter.addArgumentMaterialization(
        [](OpBuilder &builder, RankedTensorType type, ValueRange inputs,
           Location loc) -> Value {
          assert(inputs.size() == 1);
          assert(inputs[0].getType().isa<BaseMemRefType>());
          return builder.create<TensorLoadOp>(loc, type, inputs[0]);
        });

    populateBufferizeMaterializationLegality(target);
    populateSCFStructuralTypeConversionsAndLegality(context, typeConverter,
                                                    patterns, target);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  };
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createSCFBufferizePass() {
  return std::make_unique<SCFBufferizePass>();
}
