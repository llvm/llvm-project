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

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-super-vectorizer-test"

using namespace mlir;

using llvm::SetVector;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::list<int> clTestVectorShapeRatio(
    "vector-shape-ratio",
    llvm::cl::desc("Specify the HW vector size for vectorization"),
    llvm::cl::ZeroOrMore, llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestForwardSlicingAnalysis(
    "forward-slicing",
    llvm::cl::desc("Enable testing forward static slicing and topological sort "
                   "functionalities"),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestBackwardSlicingAnalysis(
    "backward-slicing",
    llvm::cl::desc("Enable testing backward static slicing and "
                   "topological sort functionalities"),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestSlicingAnalysis(
    "slicing",
    llvm::cl::desc("Enable testing static slicing and topological sort "
                   "functionalities"),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestComposeMaps(
    "compose-maps",
    llvm::cl::desc(
        "Enable testing the composition of AffineMap where each "
        "AffineMap in the composition is specified as the affine_map attribute "
        "in a constant op."),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestNormalizeMaps(
    "normalize-maps",
    llvm::cl::desc(
        "Enable testing the normalization of AffineAffineApplyOp "
        "where each AffineAffineApplyOp in the composition is a single output "
        "operation."),
    llvm::cl::cat(clOptionsCategory));
static llvm::cl::opt<bool> clTestVecAffineLoopNest(
    "vectorize-affine-loop-nest",
    llvm::cl::desc(
        "Enable testing for the 'vectorizeAffineLoopNest' utility by "
        "vectorizing the outermost loops found"),
    llvm::cl::cat(clOptionsCategory));

namespace {
struct VectorizerTestPass
    : public PassWrapper<VectorizerTestPass, FunctionPass> {
  static constexpr auto kTestAffineMapOpName = "test_affine_map";
  static constexpr auto kTestAffineMapAttrName = "affine_map";
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnFunction() override;
  void testVectorShapeRatio(llvm::raw_ostream &outs);
  void testForwardSlicing(llvm::raw_ostream &outs);
  void testBackwardSlicing(llvm::raw_ostream &outs);
  void testSlicing(llvm::raw_ostream &outs);
  void testComposeMaps(llvm::raw_ostream &outs);
  void testNormalizeMaps();

  /// Test for 'vectorizeAffineLoopNest' utility.
  void testVecAffineLoopNest();
};

} // end anonymous namespace

void VectorizerTestPass::testVectorShapeRatio(llvm::raw_ostream &outs) {
  auto f = getFunction();
  using matcher::Op;
  SmallVector<int64_t, 8> shape(clTestVectorShapeRatio.begin(),
                                clTestVectorShapeRatio.end());
  auto subVectorType =
      VectorType::get(shape, FloatType::getF32(f.getContext()));
  // Only filter operations that operate on a strict super-vector and have one
  // return. This makes testing easier.
  auto filter = [&](Operation &op) {
    assert(subVectorType.getElementType().isF32() &&
           "Only f32 supported for now");
    if (!matcher::operatesOnSuperVectorsOf(op, subVectorType)) {
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
    auto superVectorType = opInst->getResult(0).getType().cast<VectorType>();
    auto ratio = shapeRatio(superVectorType, subVectorType);
    if (!ratio.hasValue()) {
      opInst->emitRemark("NOT MATCHED");
    } else {
      outs << "\nmatched: " << *opInst << " with shape ratio: ";
      llvm::interleaveComma(MutableArrayRef<int64_t>(*ratio), outs);
    }
  }
}

static NestedPattern patternTestSlicingOps() {
  using matcher::Op;
  // Match all operations with the kTestSlicingOpName name.
  auto filter = [](Operation &op) {
    // Just use a custom op name for this test, it makes life easier.
    return op.getName().getStringRef() == "slicing-test-op";
  };
  return Op(filter);
}

void VectorizerTestPass::testBackwardSlicing(llvm::raw_ostream &outs) {
  auto f = getFunction();
  outs << "\n" << f.getName();

  SmallVector<NestedMatch, 8> matches;
  patternTestSlicingOps().match(f, &matches);
  for (auto m : matches) {
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(m.getMatchedOperation(), &backwardSlice);
    outs << "\nmatched: " << *m.getMatchedOperation()
         << " backward static slice: ";
    for (auto *op : backwardSlice)
      outs << "\n" << *op;
  }
}

void VectorizerTestPass::testForwardSlicing(llvm::raw_ostream &outs) {
  auto f = getFunction();
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
  auto f = getFunction();
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
  auto f = getFunction();

  using matcher::Op;
  auto pattern = Op(customOpWithAffineMapAttribute);
  SmallVector<NestedMatch, 8> matches;
  pattern.match(f, &matches);
  SmallVector<AffineMap, 4> maps;
  maps.reserve(matches.size());
  for (auto m : llvm::reverse(matches)) {
    auto *opInst = m.getMatchedOperation();
    auto map = opInst->getAttr(VectorizerTestPass::kTestAffineMapAttrName)
                   .cast<AffineMapAttr>()
                   .getValue();
    maps.push_back(map);
  }
  AffineMap res;
  for (auto m : maps) {
    res = res ? res.compose(m) : m;
  }
  simplifyAffineMap(res).print(outs << "\nComposed map: ");
}

static bool affineApplyOp(Operation &op) { return isa<AffineApplyOp>(op); }

static bool singleResultAffineApplyOpWithoutUses(Operation &op) {
  auto app = dyn_cast<AffineApplyOp>(op);
  return app && app.use_empty();
}

void VectorizerTestPass::testNormalizeMaps() {
  using matcher::Op;

  auto f = getFunction();

  // Save matched AffineApplyOp that all need to be erased in the end.
  auto pattern = Op(affineApplyOp);
  SmallVector<NestedMatch, 8> toErase;
  pattern.match(f, &toErase);
  {
    // Compose maps.
    auto pattern = Op(singleResultAffineApplyOpWithoutUses);
    SmallVector<NestedMatch, 8> matches;
    pattern.match(f, &matches);
    for (auto m : matches) {
      auto app = cast<AffineApplyOp>(m.getMatchedOperation());
      OpBuilder b(m.getMatchedOperation());
      SmallVector<Value, 8> operands(app.getOperands());
      makeComposedAffineApply(b, app.getLoc(), app.getAffineMap(), operands);
    }
  }
  // We should now be able to erase everything in reverse order in this test.
  for (auto m : llvm::reverse(toErase)) {
    m.getMatchedOperation()->erase();
  }
}

/// Test for 'vectorizeAffineLoopNest' utility.
void VectorizerTestPass::testVecAffineLoopNest() {
  std::vector<SmallVector<AffineForOp, 2>> loops;
  gatherLoops(getFunction(), loops);

  // Expected only one loop nest.
  if (loops.empty() || loops[0].size() != 1)
    return;

  // We vectorize the outermost loop found with VF=4.
  AffineForOp outermostLoop = loops[0][0];
  VectorizationStrategy strategy;
  strategy.vectorSizes.push_back(4 /*vectorization factor*/);
  strategy.loopToVectorDim[outermostLoop] = 0;
  std::vector<SmallVector<AffineForOp, 2>> loopsToVectorize;
  loopsToVectorize.push_back({outermostLoop});
  vectorizeAffineLoopNest(loopsToVectorize, strategy);
}

void VectorizerTestPass::runOnFunction() {
  // Only support single block functions at this point.
  FuncOp f = getFunction();
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

    if (clTestNormalizeMaps)
      testNormalizeMaps();
  }

  if (clTestVecAffineLoopNest)
    testVecAffineLoopNest();

  if (!outs.str().empty()) {
    emitRemark(UnknownLoc::get(&getContext()), outs.str());
  }
}

namespace mlir {
void registerVectorizerTestPass() {
  PassRegistration<VectorizerTestPass> pass(
      "affine-super-vectorizer-test",
      "Tests vectorizer standalone functionality.");
}
} // namespace mlir
