//===- VectorizerTestPass.cpp - VectorizerTestPass Pass Impl --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple testing pass for vectorization functionality.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-super-vectorizer-test"

using namespace mlir;
using namespace mlir::affine;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

namespace {
struct VectorizerTestPass
    : public PassWrapper<VectorizerTestPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizerTestPass)

  static constexpr auto kTestAffineMapOpName = "test_affine_map";
  static constexpr auto kTestAffineMapAttrName = "affine_map";
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  StringRef getArgument() const final { return "affine-super-vectorizer-test"; }
  StringRef getDescription() const final {
    return "Tests vectorizer standalone functionality.";
  }

  VectorizerTestPass() = default;
  VectorizerTestPass(const VectorizerTestPass &pass) : PassWrapper(pass){};

  ListOption<int> clTestVectorShapeRatio{
      *this, "vector-shape-ratio",
      llvm::cl::desc("Specify the HW vector size for vectorization")};
  Option<bool> clTestForwardSlicingAnalysis{
      *this, "forward-slicing",
      llvm::cl::desc(
          "Enable testing forward static slicing and topological sort "
          "functionalities")};
  Option<bool> clTestBackwardSlicingAnalysis{
      *this, "backward-slicing",
      llvm::cl::desc("Enable testing backward static slicing and "
                     "topological sort functionalities")};
  Option<bool> clTestSlicingAnalysis{
      *this, "slicing",
      llvm::cl::desc("Enable testing static slicing and topological sort "
                     "functionalities")};
  Option<bool> clTestComposeMaps{
      *this, "compose-maps",
      llvm::cl::desc("Enable testing the composition of AffineMap where each "
                     "AffineMap in the composition is specified as the "
                     "affine_map attribute "
                     "in a constant op.")};
  Option<bool> clTestVecAffineLoopNest{
      *this, "vectorize-affine-loop-nest",
      llvm::cl::desc(
          "Enable testing for the 'vectorizeAffineLoopNest' utility by "
          "vectorizing the outermost loops found")};

  void runOnOperation() override;
  void testVectorShapeRatio(llvm::raw_ostream &outs);
  void testForwardSlicing(llvm::raw_ostream &outs);
  void testBackwardSlicing(llvm::raw_ostream &outs);
  void testSlicing(llvm::raw_ostream &outs);
  void testComposeMaps(llvm::raw_ostream &outs);

  /// Test for 'vectorizeAffineLoopNest' utility.
  void testVecAffineLoopNest(llvm::raw_ostream &outs);
};

} // namespace

void VectorizerTestPass::testVectorShapeRatio(llvm::raw_ostream &outs) {
  auto f = getOperation();
  using affine::matcher::Op;
  SmallVector<int64_t, 8> shape(clTestVectorShapeRatio.begin(),
                                clTestVectorShapeRatio.end());
  auto subVectorType = VectorType::get(shape, Float32Type::get(f.getContext()));
  // Only filter operations that operate on a strict super-vector and have one
  // return. This makes testing easier.
  auto filter = [&](Operation &op) {
    assert(subVectorType.getElementType().isF32() &&
           "Only f32 supported for now");
    if (!mlir::matcher::operatesOnSuperVectorsOf(op, subVectorType)) {
      return false;
    }
    if (op.getNumResults() != 1) {
      return false;
    }
    return true;
  };
  auto pat = Op(filter);
  SmallVector<NestedMatch, 8> matches;
  pat.match(f, &matches);
  for (auto m : matches) {
    auto *opInst = m.getMatchedOperation();
    // This is a unit test that only checks and prints shape ratio.
    // As a consequence we write only Ops with a single return type for the
    // purpose of this test. If we need to test more intricate behavior in the
    // future we can always extend.
    auto superVectorType = cast<VectorType>(opInst->getResult(0).getType());
    auto ratio =
        computeShapeRatio(superVectorType.getShape(), subVectorType.getShape());
    if (!ratio) {
      opInst->emitRemark("NOT MATCHED");
    } else {
      outs << "\nmatched: " << *opInst << " with shape ratio: ";
      llvm::interleaveComma(MutableArrayRef<int64_t>(*ratio), outs);
    }
  }
}

static NestedPattern patternTestSlicingOps() {
  using affine::matcher::Op;
  // Match all operations with the kTestSlicingOpName name.
  auto filter = [](Operation &op) {
    // Just use a custom op name for this test, it makes life easier.
    return op.getName().getStringRef() == "slicing-test-op";
  };
  return Op(filter);
}

void VectorizerTestPass::testBackwardSlicing(llvm::raw_ostream &outs) {
  auto f = getOperation();
  outs << "\n" << f.getName();

  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Operation *> backwardSlice;
    LogicalResult result =
        getBackwardSlice(m.getMatchedOperation(), &backwardSlice);
    assert(result.succeeded() && "expected a backward slice");
    outs << "\nmatched: " << *m.getMatchedOperation()
         << " backward static slice: ";
    for (auto *op : backwardSlice)
      outs << "\n" << *op;
  }
}

void VectorizerTestPass::testForwardSlicing(llvm::raw_ostream &outs) {
  auto f = getOperation();
  outs << "\n" << f.getName();

  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(m.getMatchedOperation(), &forwardSlice);
    outs << "\nmatched: " << *m.getMatchedOperation()
         << " forward static slice: ";
    for (auto *op : forwardSlice)
      outs << "\n" << *op;
  }
}

void VectorizerTestPass::testSlicing(llvm::raw_ostream &outs) {
  auto f = getOperation();
  outs << "\n" << f.getName();

  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Operation *> staticSlice = getSlice(m.getMatchedOperation());
    outs << "\nmatched: " << *m.getMatchedOperation() << " static slice: ";
    for (auto *op : staticSlice)
      outs << "\n" << *op;
  }
}

static bool customOpWithAffineMapAttribute(Operation &op) {
  return op.getName().getStringRef() ==
         VectorizerTestPass::kTestAffineMapOpName;
}

void VectorizerTestPass::testComposeMaps(llvm::raw_ostream &outs) {
  auto f = getOperation();

  using affine::matcher::Op;
  auto pattern = Op(customOpWithAffineMapAttribute);
  SmallVector<NestedMatch, 8> matches;
  pattern.match(f, &matches);
  SmallVector<AffineMap, 4> maps;
  maps.reserve(matches.size());
  for (auto m : llvm::reverse(matches)) {
    auto *opInst = m.getMatchedOperation();
    auto map =
        cast<AffineMapAttr>(opInst->getDiscardableAttr(
                                VectorizerTestPass::kTestAffineMapAttrName))
            .getValue();
    maps.push_back(map);
  }
  if (maps.empty())
    // Nothing to compose
    return;
  AffineMap res;
  for (auto m : maps) {
    res = res ? res.compose(m) : m;
  }
  simplifyAffineMap(res).print(outs << "\nComposed map: ");
}

/// Test for 'vectorizeAffineLoopNest' utility.
void VectorizerTestPass::testVecAffineLoopNest(llvm::raw_ostream &outs) {
  std::vector<SmallVector<AffineForOp, 2>> loops;
  gatherLoops(getOperation(), loops);

  // Expected only one loop nest.
  if (loops.empty() || loops[0].size() != 1)
    return;

  // We vectorize the outermost loop found with VF=4.
  AffineForOp outermostLoop = loops[0][0];
  VectorizationStrategy strategy;
  strategy.vectorSizes.push_back(4 /*vectorization factor*/);
  strategy.loopToVectorDim[outermostLoop] = 0;

  SmallVector<LoopReduction, 2> reductions;
  if (!isLoopParallel(outermostLoop, &reductions)) {
    outs << "Outermost loop cannot be parallel\n";
    return;
  }
  std::vector<SmallVector<AffineForOp, 2>> loopsToVectorize;
  loopsToVectorize.push_back({outermostLoop});
  (void)vectorizeAffineLoopNest(loopsToVectorize, strategy);
}

void VectorizerTestPass::runOnOperation() {
  // Only support single block functions at this point.
  func::FuncOp f = getOperation();
  if (!llvm::hasSingleElement(f))
    return;

  std::string str;
  llvm::raw_string_ostream outs(str);

  { // Tests that expect a NestedPatternContext to be allocated externally.
    NestedPatternContext mlContext;

    if (!clTestVectorShapeRatio.empty())
      testVectorShapeRatio(outs);

    if (clTestForwardSlicingAnalysis)
      testForwardSlicing(outs);

    if (clTestBackwardSlicingAnalysis)
      testBackwardSlicing(outs);

    if (clTestSlicingAnalysis)
      testSlicing(outs);

    if (clTestComposeMaps)
      testComposeMaps(outs);
  }

  if (clTestVecAffineLoopNest)
    testVecAffineLoopNest(outs);

  if (!outs.str().empty()) {
    emitRemark(UnknownLoc::get(&getContext()), outs.str());
  }
}

namespace mlir {
void registerVectorizerTestPass() { PassRegistration<VectorizerTestPass>(); }
} // namespace mlir
