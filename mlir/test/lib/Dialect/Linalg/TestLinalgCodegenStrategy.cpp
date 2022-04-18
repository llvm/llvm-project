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

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TestLinalgCodegenStrategy
    : public PassWrapper<TestLinalgCodegenStrategy,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLinalgCodegenStrategy)

  StringRef getArgument() const final { return "test-linalg-codegen-strategy"; }
  StringRef getDescription() const final {
    return "Test Linalg Codegen Strategy.";
  }
  TestLinalgCodegenStrategy() = default;
  TestLinalgCodegenStrategy(const TestLinalgCodegenStrategy &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnOperation() override;

  void runStrategy(const LinalgTilingAndFusionOptions &tilingAndFusionOptions,
                   const LinalgTilingOptions &tilingOptions,
                   const LinalgTilingOptions &registerTilingOptions,
                   LinalgPaddingOptions paddingOptions,
                   vector::VectorContractLowering vectorContractLowering,
                   vector::VectorTransferSplit vectorTransferSplit);

  Option<bool> fuse{
      *this, "fuse",
      llvm::cl::desc("Fuse the producers after tiling the root op."),
      llvm::cl::init(false)};
  ListOption<int64_t> tileSizes{*this, "tile-sizes",
                                llvm::cl::desc("Specifies the tile sizes.")};
  ListOption<int64_t> tileInterchange{
      *this, "tile-interchange",
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
      *this, "register-tile-sizes",
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
  ListOption<std::string> paddingValues{
      *this, "padding-values",
      llvm::cl::desc("Operand padding values parsed by the attribute parser."),
      llvm::cl::ZeroOrMore};
  ListOption<int64_t> paddingDimensions{
      *this, "padding-dimensions",
      llvm::cl::desc("Operation iterator dimensions to pad."),
      llvm::cl::ZeroOrMore};
  ListOption<int64_t> packPaddings{*this, "pack-paddings",
                                   llvm::cl::desc("Operand packing flags."),
                                   llvm::cl::ZeroOrMore};
  ListOption<int64_t> hoistPaddings{*this, "hoist-paddings",
                                    llvm::cl::desc("Operand hoisting depths."),
                                    llvm::cl::ZeroOrMore};
  ListOption<SmallVector<int64_t>> transposePaddings{
      *this, "transpose-paddings",
      llvm::cl::desc(
          "Transpose paddings. Specify a operand dimension interchange "
          "using the following format:\n"
          "-transpose-paddings=[1,0,2],[0,1],[0,1]\n"
          "It defines the interchange [1, 0, 2] for operand one and "
          "the interchange [0, 1] (no transpose) for the remaining operands."
          "All interchange vectors have to be permuations matching the "
          "operand rank."),
      llvm::cl::ZeroOrMore};
  Option<bool> generalize{*this, "generalize",
                          llvm::cl::desc("Generalize named operations."),
                          llvm::cl::init(false)};
  ListOption<int64_t> iteratorInterchange{
      *this, "iterator-interchange",
      llvm::cl::desc("Specifies the iterator interchange.")};
  Option<bool> decompose{
      *this, "decompose",
      llvm::cl::desc("Decompose convolutions to lower dimensional ones."),
      llvm::cl::init(false)};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Rewrite the linalg op as a vector operation."),
      llvm::cl::init(false)};
  Option<bool> vectorizePadding{
      *this, "vectorize-padding",
      llvm::cl::desc("Rewrite pad tensor ops as vector operations."),
      llvm::cl::init(false)};
  Option<std::string> splitVectorTransfersTo{
      *this, "split-transfers",
      llvm::cl::desc(
          "Split vector transfers between slow (masked) and fast "
          "(unmasked) variants. Possible options are:\n"
          "\tnone: keep unsplit vector.transfer and pay the full price\n"
          "\tmemref.copy: use linalg.fill + memref.copy for the slow path\n"
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
  Option<bool> runEnablePass{
      *this, "run-enable-pass",
      llvm::cl::desc("Run the enable pass between transformations"),
      llvm::cl::init(true)};
  Option<std::string> anchorOpName{
      *this, "anchor-op",
      llvm::cl::desc(
          "Which single linalg op is the anchor for the codegen strategy to "
          "latch on:\n"
          "\tlinalg.matmul: anchor on linalg.matmul\n"
          "\tlinalg.matmul_column_major: anchor on linalg.matmul_column_major\n"
          "\tmemref.copy: anchor on memref.copy\n"
          "\tlinalg.fill: anchor on linalg.fill\n"),
      llvm::cl::init("")};
  Option<std::string> anchorFuncOpName{
      *this, "anchor-func",
      llvm::cl::desc(
          "Which single func op is the anchor for the codegen strategy to "
          "latch on."),
      llvm::cl::init("")};
};

void TestLinalgCodegenStrategy::runStrategy(
    const LinalgTilingAndFusionOptions &tilingAndFusionOptions,
    const LinalgTilingOptions &tilingOptions,
    const LinalgTilingOptions &registerTilingOptions,
    LinalgPaddingOptions paddingOptions,
    vector::VectorContractLowering vectorContractLowering,
    vector::VectorTransferSplit vectorTransferSplit) {
  std::string anchorOpNameOrWildcard = fuse ? "" : anchorOpName.getValue();
  CodegenStrategy strategy;
  strategy
      .tileAndFuseIf(fuse && !tileSizes.empty(), anchorOpName,
                     tilingAndFusionOptions)
      .tileIf(!fuse && !tileSizes.empty(), anchorOpName, tilingOptions)
      .promoteIf(!fuse && promote, anchorOpName,
                 LinalgPromotionOptions()
                     .setAlignment(16)
                     .setUseFullTileBuffersByDefault(promoteFullTile))
      .tileIf(!fuse && !registerTileSizes.empty(), anchorOpName,
              registerTilingOptions)
      .promoteIf(!fuse && registerPromote, anchorOpName,
                 LinalgPromotionOptions()
                     .setAlignment(16)
                     .setUseFullTileBuffersByDefault(registerPromoteFullTile))
      .padIf(pad, anchorOpNameOrWildcard, std::move(paddingOptions))
      .decomposeIf(decompose)
      .generalizeIf(generalize, anchorOpNameOrWildcard)
      .interchangeIf(!iteratorInterchange.empty(), iteratorInterchange)
      .vectorizeIf(vectorize, anchorOpNameOrWildcard, nullptr, vectorizePadding)
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
  func::FuncOp funcOp = getOperation();
  OpPassManager dynamicPM("func.func");
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext(), runEnablePass);
  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}
} // namespace

/// Apply transformations specified as patterns.
void TestLinalgCodegenStrategy::runOnOperation() {
  if (!anchorFuncOpName.empty() && anchorFuncOpName != getOperation().getName())
    return;

  LinalgTilingAndFusionOptions tilingAndFusionOptions;
  tilingAndFusionOptions.tileSizes = {tileSizes.begin(), tileSizes.end()};
  tilingAndFusionOptions.tileInterchange = {tileInterchange.begin(),
                                            tileInterchange.end()};

  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty())
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (!tileInterchange.empty())
    tilingOptions = tilingOptions.setInterchange(
        SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));

  LinalgTilingOptions registerTilingOptions;
  if (!registerTileSizes.empty())
    registerTilingOptions =
        registerTilingOptions.setTileSizes(registerTileSizes);

  // Parse the padding values.
  SmallVector<Attribute> paddingValueAttributes;
  for (const std::string &paddingValue : paddingValues) {
    paddingValueAttributes.push_back(
        parseAttribute(paddingValue, &getContext()));
  }

  // Parse the transpose vectors.
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValueAttributes);
  paddingOptions.setPaddingDimensions(
      SmallVector<int64_t>{paddingDimensions.begin(), paddingDimensions.end()});
  paddingOptions.setPackPaddings(
      SmallVector<bool>{packPaddings.begin(), packPaddings.end()});
  paddingOptions.setHoistPaddings(
      SmallVector<int64_t>{hoistPaddings.begin(), hoistPaddings.end()});
  paddingOptions.setTransposePaddings(transposePaddings);

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
          .Case("memref-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  runStrategy(tilingAndFusionOptions, tilingOptions, registerTilingOptions,
              paddingOptions, vectorContractLowering, vectorTransferSplit);
}

namespace mlir {
namespace test {
void registerTestLinalgCodegenStrategy() {
  PassRegistration<TestLinalgCodegenStrategy>();
}
} // namespace test
} // namespace mlir
