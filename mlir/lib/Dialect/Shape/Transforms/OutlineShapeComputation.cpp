//====----- OutlineShapeComputation.cpp -------------------------*- C++-*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <vector>

#define DEBUG_TYPE "outline-shape-computation"

using namespace mlir;

namespace {

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
        // skip if defining op is in the cluster
        continue;
      }
      if (inputSet.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
  }
  return inputs;
}

std::pair<shape::FuncOp, SmallVector<Value>>
createFuncFromCluster(OpBuilder &b, const SmallVector<Operation *, 8> &cluster,
                      Value dimSize, StringRef fnName) {
  SmallVector<Value, 4> inputs = getInputsOfCluster(cluster);
  SmallVector<Type, 4> output_types;
  output_types.push_back(dimSize.getType());

  SmallVector<Type, 4> input_types;
  input_types.reserve(inputs.size());
  for (Value v : inputs) {
    input_types.push_back(v.getType());
  }
  auto fnType = b.getFunctionType(input_types, output_types);
  b.setInsertionPointAfter(cluster[0]->getParentOp());
  shape::FuncOp fnOp =
      b.create<shape::FuncOp>(UnknownLoc::get(b.getContext()), fnName, fnType);
  Block *block = fnOp.addEntryBlock();
  b.setInsertionPoint(block, block->end());
  BlockAndValueMapping bvm;
  for (auto inputAndArg : llvm::zip(inputs, fnOp.getArguments())) {
    bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
  }
  for (Operation *op : cluster) {
    b.clone(*op, bvm);
  }
  llvm::SmallVector<Value, 4> fnReturns;
  fnReturns.push_back(bvm.lookupOrDefault(dimSize));

  b.create<shape::ReturnOp>(UnknownLoc::get(b.getContext()), fnReturns);
  fnOp.setPrivate();
  return std::make_pair(fnOp, inputs);
}

DenseMap<Value, SmallVector<Operation *, 8>>
getOrderedClusters(const DenseMap<Value, DenseSet<Operation *>> &clusters,
                   ModuleOp &moduleOp) {
  DenseMap<Operation *, SmallVector<Value>> op2Shapes;
  for (auto it : clusters) {
    Value shape = it.first;
    const DenseSet<Operation *> &cluster = it.second;
    for (Operation *cOp : cluster) {
      op2Shapes[cOp].push_back(shape);
    }
  }

  DenseMap<Value, SmallVector<Operation *, 8>> orderedClusters;
  moduleOp.walk([&](Operation *op) {
    auto it = op2Shapes.find(op);
    if (it != op2Shapes.end()) {
      Operation *cOp = it->first;
      for (Value shape : it->second) {
        orderedClusters[shape].push_back(cOp);
      }
    }
  });

  return orderedClusters;
}

bool isDimOpNotFromFuncArg(Value dimSize) {
  tensor::DimOp dimOp = dimSize.getDefiningOp<tensor::DimOp>();
  assert(dimOp != nullptr &&
         "input of the cluster is supposed to be a tensor.dim op");
  Value source = dimOp.getSource();
  return source.getDefiningOp();
}

std::pair<unsigned, unsigned> getArgAndDimPos(Value dimSize) {
  tensor::DimOp dimOp = dimSize.getDefiningOp<tensor::DimOp>();
  assert(dimOp != nullptr &&
         "input of the cluster is supposed to be a tensor.dim op");
  Value source = dimOp.getSource();
  func::FuncOp funcOp = dimOp->getParentOfType<func::FuncOp>();

  unsigned argPos = 0;
  APInt dimPos;

  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    if (funcOp.getArgument(i) == source) {
      argPos = i;
      break;
    }
  }

  if (!matchPattern(dimOp.getIndex(), m_ConstantInt(&dimPos))) {
    llvm_unreachable("The index of tensor.dim is not constant-like");
  }

  return {argPos, dimPos.getZExtValue()};
}

struct OutlineShapeComputationPass
    : public OutlineShapeComputationBase<OutlineShapeComputationPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    init(moduleOp);
    SymbolTable symbolTable(moduleOp);

    MLIRContext *context = moduleOp.getContext();
    std::vector<shape::WithOp> allWithOps;

    DenseMap<Value, DenseSet<Operation *>> clusters;
    moduleOp.walk([&](shape::WithOp withOp) { allWithOps.push_back(withOp); });
    std::string dynamicSourceNamePrefix = "s";
    int dynamicSourceNameIdx = 0;
    std::string shapeCalculationNamePrefix = "shape_cal_";
    int shapeCalculationNameIdx = 0;
    OpBuilder builder(context);
    DenseMap<Value, FlatSymbolRefAttr> dynDimSize2Symbol;

    auto getOrConstructSymbolFromDimSize = [&](Value dimSize) {
      auto symbolIt = dynDimSize2Symbol.find(dimSize);
      if (symbolIt == dynDimSize2Symbol.end()) {
        std::string name;
        if (isDimOpNotFromFuncArg(dimSize)) {
          name =
              dynamicSourceNamePrefix + std::to_string(dynamicSourceNameIdx++);
        } else {
          auto posPair = getArgAndDimPos(dimSize);
          name = "arg_" + std::to_string(posPair.first) + "_dim_" +
                 std::to_string(posPair.second);
        }
        auto symbol = FlatSymbolRefAttr::get(context, name);
        dynDimSize2Symbol[dimSize] = symbol;
        return symbol;
      } else {
        return symbolIt->second;
      }
    };

    for (shape::WithOp withOp : allWithOps) {
      Value shape = withOp.getShape();
      // Get the operations cluster for each dimension size
      // FIXME: handle the case that shape is not constructed from
      // shape.from_extents
      if (auto fromExtentsOp = shape.getDefiningOp<shape::FromExtentsOp>()) {
        for (Value dimSize : fromExtentsOp.getExtents()) {
          if (clusters.count(dimSize) == 0)
            getClusterFromValue(dimSize, clusters);
        }
      }
    }
    DenseMap<Value, SmallVector<Operation *, 8>> orderedClusters =
        getOrderedClusters(clusters, moduleOp);

    // Construct a shape function or a symbol for each cluster
    for (shape::WithOp withOp : allWithOps) {
      Value value = withOp.getOperand();
      SmallVector<FlatSymbolRefAttr> symbols;
      Value shape = withOp.getShape();
      RankedTensorType rankedType =
          value.getType().dyn_cast<RankedTensorType>();
      if (rankedType == nullptr) {
        continue;
      }

      // FIXME: handle the case that shape is not constructed from
      // shape.from_extents
      if (auto fromExtentsOp = shape.getDefiningOp<shape::FromExtentsOp>()) {
        for (auto it : llvm::enumerate(fromExtentsOp.getExtents())) {
          size_t idx = it.index();
          if (!rankedType.isDynamicDim(idx))
            continue;
          Value dimSize = it.value();
          const SmallVector<Operation *, 8> &cluster = orderedClusters[dimSize];
          // The cluster is empty if the dimension an internal dynamic dimension
          // source or a function argument
          if (cluster.empty()) {
            FlatSymbolRefAttr symbol = getOrConstructSymbolFromDimSize(dimSize);
            symbols.push_back(symbol);
            LLVM_DEBUG(llvm::dbgs() << "Symbol for " << dimSize << "\n"
                                    << symbol << "\n");
          } else {
            std::string name = shapeCalculationNamePrefix +
                               std::to_string(shapeCalculationNameIdx++);
            auto pair = createFuncFromCluster(builder, cluster, dimSize, name);
            shape::FuncOp shapeFuncOp = pair.first;
            StringAttr insertedName = symbolTable.insert(shapeFuncOp);
            auto symbol = FlatSymbolRefAttr::get(context, insertedName);
            const SmallVector<Value> &inputs = pair.second;
            symbols.push_back(symbol);
            for (Value inp : inputs) {
              FlatSymbolRefAttr argSymbol =
                  getOrConstructSymbolFromDimSize(inp);
              symbols.push_back(argSymbol);
            }
            LLVM_DEBUG(llvm::dbgs() << "Symbol for " << dimSize << "\n";
                       for (auto symbol
                            : symbols) llvm::dbgs()
                       << symbol << "\n";);
          }
        }
      }

      auto shapeInfo = shape::ExtShapeInfoAttr::get(context, symbols);
      value.setType(RankedTensorType::get(
          rankedType.getShape(), rankedType.getElementType(), shapeInfo));
    }

    // FIXME: fix nested call
    moduleOp.walk([&](func::FuncOp funcOp) {
      funcOp.setType(
          FunctionType::get(context, funcOp.front().getArgumentTypes(),
                            funcOp.front().getTerminator()->getOperandTypes()));
    });

    // FIXME: handle the users of shape.with_shape
    for (shape::WithOp withOp : allWithOps) {
      withOp->erase();
    }

    // dce
    if (failed(applyPatternsAndFoldGreedily(moduleOp, {}))) {
      return signalPassFailure();
    }
  }

private:
  void init(ModuleOp op);
  bool calOnlyUsedByWithShapesRecursively(Operation *op);
  void getClusterFromValue(Value shape,
                           DenseMap<Value, DenseSet<Operation *>> &clusters);
  DenseMap<Operation *, bool> onlyUsedByWithShapes_;
};

void OutlineShapeComputationPass::getClusterFromValue(
    Value shape, DenseMap<Value, DenseSet<Operation *>> &clusters) {
  DenseSet<Operation *> cluster;

  Operation *defOp = shape.getDefiningOp();
  if (nullptr == defOp) {
    return;
  }

  DenseSet<Operation *> visited;
  std::queue<Operation *> queue;
  visited.insert(defOp);
  queue.push(defOp);
  while (!queue.empty()) {
    Operation *op = queue.front();
    queue.pop();
    if (op->getNumOperands() == 0) {
      cluster.insert(op);
    } else if (llvm::isa<tensor::DimOp>(op) &&
               !onlyUsedByWithShapes_.count(
                   op->getOperand(0).getDefiningOp())) {
      continue;
    } else {
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

bool OutlineShapeComputationPass::calOnlyUsedByWithShapesRecursively(
    Operation *op) {
  auto it = onlyUsedByWithShapes_.find(op);
  if (it != onlyUsedByWithShapes_.end())
    return it->second;

  if (llvm::isa<shape::WithOp>(op)) {
    onlyUsedByWithShapes_[op] = true;
    return true;
  }

  if (op->use_empty()) {
    onlyUsedByWithShapes_[op] = false;
    return false;
  }

  bool allUsers = true;
  for (Operation *op : op->getUsers()) {
    allUsers |= calOnlyUsedByWithShapesRecursively(op);
  }

  onlyUsedByWithShapes_[op] = allUsers;
  return allUsers;
}

void OutlineShapeComputationPass::init(ModuleOp moduleOp) {
  moduleOp.walk([&](Operation *op) { calOnlyUsedByWithShapesRecursively(op); });
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createOutlineShapeComputationPass() {
  return std::make_unique<OutlineShapeComputationPass>();
}
