//====----- OutlineShapeComputation.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/Analysis/ShapeMappingAnalysis.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <unordered_set>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_OUTLINESHAPECOMPUTATION
#include "mlir/Dialect/Shape/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "outline-shape-computation"

using namespace mlir;

namespace {

// A Value is an input of the cluster if it is an operand of an operation in the
// cluster and its defining operation is not in the cluster.
SmallVector<Value, 4>
getInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster) {
  SmallVector<Value, 4> inputs;
  llvm::SmallDenseSet<Value> inputSet;
  llvm::SmallDenseSet<Operation *> opSet;
  for (Operation *op : cluster) {
    bool inserted = opSet.insert(op).second;
    (void)inserted;
    assert(inserted && "cluster contains duplicate operations");
  }

  for (Operation *op : cluster) {
    for (Value operand : op->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (opSet.find(operandOp) != opSet.end()) {
        // Skip if defining op is in the cluster.
        continue;
      }
      if (inputSet.insert(operand).second)
        inputs.push_back(operand);
    }
  }
  return inputs;
}

// Create a shape.func representing the shape computation for `shape`.
std::pair<shape::FuncOp, SmallVector<Value>>
createFuncFromCluster(OpBuilder &b, const SmallVector<Operation *, 8> &cluster,
                      Value shape, StringRef fnName, Location loc) {
  SmallVector<Value, 4> inputs = getInputsOfCluster(cluster);
  auto fnType =
      cluster.empty()
          ? b.getFunctionType(shape.getType(), shape.getType())
          : b.getFunctionType(ValueRange(inputs).getTypes(), shape.getType());
  shape::FuncOp fnOp = b.create<shape::FuncOp>(loc, fnName, fnType);
  Block *block = fnOp.addEntryBlock();
  b.setInsertionPoint(block, block->end());
  BlockAndValueMapping bvm;
  if (cluster.empty()) {
    bvm.map(shape, fnOp.getArgument(0));
  } else {
    for (auto inputAndArg : llvm::zip(inputs, fnOp.getArguments()))
      bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
  }

  for (Operation *op : cluster)
    b.clone(*op, bvm);
  llvm::SmallVector<Value, 4> fnReturns;
  fnReturns.push_back(bvm.lookupOrDefault(shape));

  b.create<shape::ReturnOp>(loc, fnReturns);
  fnOp.setPrivate();
  return std::make_pair(fnOp, inputs);
}

// The operations in the cluster might be unsorted, which could be inconvenient
// when creating shape.func op.
DenseMap<Value, SmallVector<Operation *, 8>>
getOrderedClusters(const DenseMap<Value, DenseSet<Operation *>> &clusters,
                   func::FuncOp funcOp) {
  // Compute all clusters that each operation is in
  DenseMap<Operation *, SmallVector<Value>> op2Shapes;
  for (const auto &it : clusters) {
    Value shape = it.first;
    const DenseSet<Operation *> &cluster = it.second;
    for (Operation *cOp : cluster)
      op2Shapes[cOp].push_back(shape);
  }

  // Iterate through all operations in order. Get all the clusters `cOp` belongs
  // to and construct the new ordered cluster as it traverses.
  DenseMap<Value, SmallVector<Operation *, 8>> orderedClusters;
  funcOp.walk([&](Operation *op) {
    auto it = op2Shapes.find(op);
    if (it != op2Shapes.end()) {
      Operation *cOp = it->first;
      for (Value shape : it->second)
        orderedClusters[shape].push_back(cOp);
    }
  });

  return orderedClusters;
}

void constructShapeFunc(
    const std::vector<shape::WithOp> &allWithOps, MLIRContext *context,
    DenseMap<Value, SmallVector<Operation *, 8>> &clusters,
    SymbolTable &symbolTable,
    DenseMap<Value, shape::ShapeMappingValue> &dynShape2ShapeFunc,
    func::FuncOp funcOp, shape::ShapeMappingAnalysis &shapeMappingAnalysis) {
  std::string shapeCalculationNamePrefix = "shape_cal_";
  int shapeCalculationNameIdx = 0;
  OpBuilder builder(context);

  // Construct a shape function
  for (shape::WithOp withOp : allWithOps) {
    Value value = withOp.getOperand();
    Value shape = withOp.getShape();
    RankedTensorType rankedType = value.getType().dyn_cast<RankedTensorType>();
    if (rankedType == nullptr)
      continue;

    const SmallVector<Operation *, 8> &cluster = clusters[shape];
    shape::ShapeMappingValue shapeMappingValue;
    auto it = dynShape2ShapeFunc.find(shape);
    if (it == dynShape2ShapeFunc.end()) {
      std::string name = shapeCalculationNamePrefix +
                         std::to_string(shapeCalculationNameIdx++);
      Location loc = value.getLoc();
      builder.setInsertionPointAfter(funcOp);
      auto pair = createFuncFromCluster(builder, cluster, shape, name, loc);
      const SmallVector<Value> &inputs = pair.second;
      shape::FuncOp shapeFuncOp = pair.first;
      StringAttr insertedName = symbolTable.insert(shapeFuncOp);
      auto symbol = FlatSymbolRefAttr::get(context, insertedName);

      shapeMappingValue.funcSymbol = symbol;
      shapeMappingValue.inputs = inputs;
    } else {
      shapeMappingValue = it->second;
    }
    dynShape2ShapeFunc[shape] = shapeMappingValue;
    shapeMappingAnalysis.shapeMapping.insert(
        std::make_pair(value, shapeMappingValue));
  }
}

struct OutlineShapeComputationPass
    : public impl::OutlineShapeComputationBase<OutlineShapeComputationPass> {

  void runOnOperation() override;

private:
  bool calOnlyUsedByWithShapesRecursively(Operation *op, Value prevOutput);

  void getClusterFromValue(Value shape,
                           DenseMap<Value, DenseSet<Operation *>> &clusters);

  DenseMap<Value, SmallVector<Operation *, 8>>
  constructClustersForEachShape(const std::vector<shape::WithOp> &allWithOps,
                                func::FuncOp funcOp);

  DenseSet<Operation *> onlyUsedByWithShapes;
};

class TensorDimOpRewriter : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto shapeOf =
        rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.getSource());
    rewriter.replaceOpWithNewOp<shape::GetExtentOp>(op, op.getType(), shapeOf,
                                                    op.getIndex());
    return success();
  }
};

void OutlineShapeComputationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SymbolTable symbolTable(moduleOp);
  DenseMap<Value, shape::ShapeMappingValue> dynShape2ShapeFunc;
  auto &shapeMappingAnalysis = getAnalysis<shape::ShapeMappingAnalysis>();
  // TODO: This is as we populate this analysis during a pass that mutates. This
  // pass currently requires 1 single module being compiled.
  shapeMappingAnalysis.shapeMapping.clear();
  markAnalysesPreserved<shape::ShapeMappingAnalysis>();

  moduleOp.walk([&](func::FuncOp funcOp) {
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet prevPatterns(context);
    prevPatterns.insert<TensorDimOpRewriter>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(prevPatterns))))
      return signalPassFailure();

    // initialize class member `onlyUsedByWithShapes`
    onlyUsedByWithShapes.clear();
    funcOp.walk([&](Operation *op) {
      calOnlyUsedByWithShapesRecursively(op, /*prevOutput=*/nullptr);
    });
    LLVM_DEBUG({
      llvm::dbgs() << "onlyUsedByWithShapes table: \n";
      for (auto it : onlyUsedByWithShapes)
        llvm::dbgs() << *it << "\n";
    });

    // collect all the shape.with_shape ops.
    std::vector<shape::WithOp> allWithOps;
    funcOp.walk([&](shape::WithOp withOp) { allWithOps.push_back(withOp); });

    DenseMap<Value, SmallVector<Operation *, 8>> clusters =
        constructClustersForEachShape(allWithOps, funcOp);
    constructShapeFunc(allWithOps, context, clusters, symbolTable,
                       dynShape2ShapeFunc, funcOp, shapeMappingAnalysis);

    for (shape::WithOp withOp : allWithOps) {
      Value value = withOp.getOperand();
      for (Operation *user : withOp.getResult().getUsers()) {
        if (Value valueOf = llvm::dyn_cast<shape::ValueOfOp>(user))
          valueOf.replaceAllUsesExcept(value, withOp);
      }
    }

    // Apply patterns, note this also performs DCE.
    if (failed(applyPatternsAndFoldGreedily(funcOp, {})))
      return signalPassFailure();
  });
}

DenseMap<Value, SmallVector<Operation *, 8>>
OutlineShapeComputationPass::constructClustersForEachShape(
    const std::vector<shape::WithOp> &allWithOps, func::FuncOp funcOp) {
  DenseMap<Value, DenseSet<Operation *>> clusters;
  for (shape::WithOp withOp : allWithOps) {
    Value shape = withOp.getShape();
    if (clusters.count(shape) == 0)
      getClusterFromValue(shape, clusters);
  }
  return getOrderedClusters(clusters, funcOp);
}

// The output of a cluster is the `shape`, and the inputs are the outputs of
// operations who are not in `onlyUsedByWithShapes`
void OutlineShapeComputationPass::getClusterFromValue(
    Value shape, DenseMap<Value, DenseSet<Operation *>> &clusters) {
  DenseSet<Operation *> cluster;

  DenseSet<Operation *> visited;
  std::queue<Operation *> queue;

  // defOp == nullptr means shape is the argument of the func op
  if (Operation *defOp = shape.getDefiningOp()) {
    visited.insert(defOp);
    queue.push(defOp);
  }
  while (!queue.empty()) {
    Operation *op = queue.front();
    queue.pop();
    if (onlyUsedByWithShapes.contains(op)) {
      cluster.insert(op);
      for (Value inp : op->getOperands()) {
        Operation *inpDefOp = inp.getDefiningOp();
        if (nullptr != inpDefOp && !visited.contains(inpDefOp)) {
          visited.insert(inpDefOp);
          queue.push(inpDefOp);
        }
      }
    }
  }

  clusters[shape] = std::move(cluster);
}

// Returns whether `op` is a shape.with_shape, or all the users' of `op`
// eventually point to the shape operand of shape.with_shape ops
bool OutlineShapeComputationPass::calOnlyUsedByWithShapesRecursively(
    Operation *op, Value prevOutput) {
  if (onlyUsedByWithShapes.contains(op))
    return true;

  if (auto withOp = llvm::dyn_cast<shape::WithOp>(op))
    return withOp.getShape() == prevOutput;

  if (op->use_empty())
    return false;

  for (Value oup : op->getResults())
    for (Operation *user : oup.getUsers())
      if (!calOnlyUsedByWithShapesRecursively(user, oup))
        return false;

  onlyUsedByWithShapes.insert(op);
  return true;
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createOutlineShapeComputationPass() {
  return std::make_unique<OutlineShapeComputationPass>();
}
