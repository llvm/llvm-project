//===- TestLinalgCodegenStrategy.cpp - Test Linalg codegen strategy -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the Linalg codegen strategy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgCodegenStrategy
    : public PassWrapper<TestLinalgCodegenStrategy, FunctionPass> {
  StringRef getArgument() const final { return "test-linalg-codegen-strategy"; }
  StringRef getDescription() const final {
    return "Test Linalg Codegen Strategy.";
  }
  TestLinalgCodegenStrategy() = default;
  TestLinalgCodegenStrategy(const TestLinalgCodegenStrategy &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    StandardOpsDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnFunction() override;

  void runStrategy(LinalgTilingOptions tilingOptions,
                   LinalgTilingOptions registerTilingOptions,
                   LinalgPaddingOptions paddingOptions,
                   vector::VectorContractLowering vectorContractLowering,
                   vector::VectorTransferSplit vectorTransferSplit);

  ListOption<int64_t> tileSizes{*this, "tile-sizes",
                                llvm::cl::MiscFlags::CommaSeparated,
                                llvm::cl::desc("Specifies the tile sizes.")};
  ListOption<unsigned> tileInterchange{
      *this, "tile-interchange", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the tile interchange.")};

  Option<bool> promote{
      *this, "promote",
      llvm::cl::desc("Promote the tile into a small aligned memory buffer."),
      llvm::cl::init(false)};
  Option<bool> promoteFullTile{
      *this, "promote-full-tile-pad",
      llvm::cl::desc("Pad the small aligned memory buffer to the tile sizes."),
      llvm::cl::init(false)};
  ListOption<int64_t> registerTileSizes{
      *this, "register-tile-sizes", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc(
          "Specifies the size of the register tile that will be used "
          " to vectorize")};
  Option<bool> registerPromote{
      *this, "register-promote",
      llvm::cl::desc(
          "Promote the register tile into a small aligned memory buffer."),
      llvm::cl::init(false)};
  Option<bool> registerPromoteFullTile{
      *this, "register-promote-full-tile-pad",
      llvm::cl::desc("Pad the small aligned memory buffer to the tile sizes."),
      llvm::cl::init(false)};
  Option<bool> pad{*this, "pad", llvm::cl::desc("Pad the operands."),
                   llvm::cl::init(false)};
  ListOption<int64_t> packPaddings{
      *this, "pack-paddings",
      llvm::cl::desc("Operand packing flags when test-pad-pattern"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
  ListOption<int64_t> hoistPaddings{
      *this, "hoist-paddings",
      llvm::cl::desc("Operand hoisting depths when test-pad-pattern"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
  Option<bool> generalize{*this, "generalize",
                          llvm::cl::desc("Generalize named operations."),
                          llvm::cl::init(false)};
  ListOption<int64_t> iteratorInterchange{
      *this, "iterator-interchange", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the iterator interchange.")};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Rewrite the linalg op as a vector operation."),
      llvm::cl::init(false)};
  Option<std::string> splitVectorTransfersTo{
      *this, "split-transfers",
      llvm::cl::desc(
          "Split vector transfers between slow (masked) and fast "
          "(unmasked) variants. Possible options are:\n"
          "\tnone: keep unsplit vector.transfer and pay the full price\n"
          "\tlinalg-copy: use linalg.fill + linalg.copy for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfer for"
          " the slow path\n"),
      llvm::cl::init("none")};
  Option<std::string> vectorizeContractionTo{
      *this, "vectorize-contraction-to",
      llvm::cl::desc("the type of vector op to use for linalg contractions"),
      llvm::cl::init("outerproduct")};
  Option<bool> unrollVectorTransfers{
      *this, "unroll-vector-transfers",
      llvm::cl::desc("Enable full unrolling of vector.transfer operations"),
      llvm::cl::init(false)};
  Option<std::string> anchorOpName{
      *this, "anchor-op",
      llvm::cl::desc(
          "Which single linalg op is the anchor for the codegen strategy to "
          "latch on:\n"
          "\tlinalg.matmul: anchor on linalg.matmul\n"
          "\tlinalg.matmul_column_major: anchor on linalg.matmul_column_major\n"
          "\tlinalg.copy: anchor on linalg.copy\n"
          "\tlinalg.fill: anchor on linalg.fill\n"),
      llvm::cl::init("")};
  Option<std::string> anchorFuncOpName{
      *this, "anchor-func",
      llvm::cl::desc(
          "Which single func op is the anchor for the codegen strategy to "
          "latch on."),
      llvm::cl::init("")};
};

// For now, just assume it is the zero of type.
// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get());
  return b.create<arith::ConstantOp>(op.getOwner()->getLoc(), t,
                                     b.getZeroAttr(t));
}

void TestLinalgCodegenStrategy::runStrategy(
    LinalgTilingOptions tilingOptions,
    LinalgTilingOptions registerTilingOptions,
    LinalgPaddingOptions paddingOptions,
    vector::VectorContractLowering vectorContractLowering,
    vector::VectorTransferSplit vectorTransferSplit) {
  assert(!anchorOpName.empty());
  CodegenStrategy strategy;
  StringRef genericOpName = GenericOp::getOperationName();
  strategy.tileIf(!tileSizes.empty(), anchorOpName, tilingOptions)
      .promoteIf(promote, anchorOpName,
                 LinalgPromotionOptions()
                     .setAlignment(16)
                     .setUseFullTileBuffersByDefault(promoteFullTile))
      .tileIf(!registerTileSizes.empty(), anchorOpName, registerTilingOptions)
      .promoteIf(registerPromote, anchorOpName,
                 LinalgPromotionOptions()
                     .setAlignment(16)
                     .setUseFullTileBuffersByDefault(registerPromoteFullTile))
      .padIf(pad, anchorOpName, paddingOptions)
      .generalizeIf(generalize, anchorOpName)
      .interchangeIf(!iteratorInterchange.empty(), iteratorInterchange)
      .vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName)
      .vectorLowering(
          LinalgVectorLoweringOptions()
              .setVectorTransformsOptions(
                  vector::VectorTransformsOptions()
                      .setVectorTransformsOptions(vectorContractLowering)
                      .setVectorTransferSplit(vectorTransferSplit))
              .setVectorTransferToSCFOptions(
                  VectorTransferToSCFOptions().enableFullUnroll(
                      unrollVectorTransfers))
              .enableTransferPartialRewrite()
              .enableContractionLowering()
              .enableTransferToSCFConversion());

  // Created a nested OpPassManager and run.
  FuncOp funcOp = getFunction();
  OpPassManager dynamicPM("builtin.func");
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}
} // end anonymous namespace

/// Apply transformations specified as patterns.
void TestLinalgCodegenStrategy::runOnFunction() {
  if (!anchorFuncOpName.empty() && anchorFuncOpName != getFunction().getName())
    return;

  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty())
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (!tileInterchange.empty())
    tilingOptions = tilingOptions.setInterchange(tileInterchange);

  LinalgTilingOptions registerTilingOptions;
  if (!registerTileSizes.empty())
    registerTilingOptions =
        registerTilingOptions.setTileSizes(registerTileSizes);

  LinalgPaddingOptions paddingOptions;
  auto packFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < packPaddings.size()
               ? packPaddings[opOperand.getOperandNumber()]
               : false;
  };
  auto hoistingFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < hoistPaddings.size()
               ? hoistPaddings[opOperand.getOperandNumber()]
               : 0;
  };
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);

  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          vectorizeContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  runStrategy(tilingOptions, registerTilingOptions, paddingOptions,
              vectorContractLowering, vectorTransferSplit);
}

namespace mlir {
namespace test {
void registerTestLinalgCodegenStrategy() {
  PassRegistration<TestLinalgCodegenStrategy>();
}
} // namespace test
} // namespace mlir
