//===- ACCEmitRemarksPrivate.cpp - Emit OpenACC privatization remarks ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass emits remarks describing the private and firstprivate variables
// associated with OpenACC compute and loop constructs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCEMITREMARKSPRIVATE
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-emit-remarks-private"

using namespace mlir;

namespace {

template <typename OpTy>
static void reportPrivatization(Operation *accOp, ValueRange operands,
                                acc::OpenACCSupport &accSupport,
                                StringRef clause) {
  SmallVector<std::string> implicitNames;
  SmallVector<std::string> explicitNames;
  for (Value operand : operands) {
    auto op = cast<OpTy>(operand.getDefiningOp());
    std::string varName = accSupport.getVariableName(op.getAccVar());
    (op.getImplicit() ? implicitNames : explicitNames)
        .push_back(varName.empty() ? "<unknown>" : varName);
  }

  if (!implicitNames.empty())
    accSupport.emitRemark(
        accOp,
        [clause, names = std::move(implicitNames)]() {
          return (Twine("Generating implicit ") + clause + "(" +
                  llvm::join(names, ",") + ")")
              .str();
        },
        DEBUG_TYPE);

  if (!explicitNames.empty())
    accSupport.emitRemark(
        accOp,
        [clause, names = std::move(explicitNames)]() {
          return (Twine("Generating ") + clause + "(" + llvm::join(names, ",") +
                  ")")
              .str();
        },
        DEBUG_TYPE);
}

template <typename OpTy>
static void emitRemarksForACCOp(OpTy accOp, acc::OpenACCSupport &accSupport) {
  reportPrivatization<acc::FirstprivateOp>(
      accOp, accOp.getFirstprivateOperands(), accSupport, "firstprivate");
  reportPrivatization<acc::PrivateOp>(accOp, accOp.getPrivateOperands(),
                                      accSupport, "private");
}

class ACCEmitRemarksPrivate
    : public acc::impl::ACCEmitRemarksPrivateBase<ACCEmitRemarksPrivate> {
public:
  using ACCEmitRemarksPrivateBase<
      ACCEmitRemarksPrivate>::ACCEmitRemarksPrivateBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto cachedAnalysis = getCachedParentAnalysis<acc::OpenACCSupport>();
    acc::OpenACCSupport &accSupport = cachedAnalysis
                                          ? cachedAnalysis->get()
                                          : getAnalysis<acc::OpenACCSupport>();

    func.walk([&](Operation *op) {
      TypeSwitch<Operation *>(op).Case<ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS>(
          [&](auto constructOp) {
            emitRemarksForACCOp(constructOp, accSupport);
          });
    });
  }
};

} // namespace
