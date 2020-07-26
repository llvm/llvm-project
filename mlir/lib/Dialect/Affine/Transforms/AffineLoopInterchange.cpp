//===- affineLoopInterchange.cpp - Code to perform loop interchange-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
#include <unordered_map>

#define cache_line_size 64 // 512 bits of cache_line_size

using namespace mlir;
llvm::DenseMap<std::pair<std::pair<Operation *, Operation *>, unsigned>, int>
    hasDependences;

namespace {
struct LoopInterchange : public AffineLoopInterchangeBase<LoopInterchange> {
  void runOnFunction() override;
  void handleNestedLoops(Operation &func);
  // bool testRectangularForloopNest(AffineForOp forOp);
  void getAccessMatrix(AffineForOp op, std::vector<AffineForOp> &foropvector,
                       llvm::DenseMap<Value, int64_t> &index_values);
  void buildRefGroups(AffineForOp ForOp,
                      std::vector<std::set<Operation *>> *refGroups,
                      unsigned max_depth, unsigned innermostlooplevel);
  int64_t getLoopCost(std::vector<AffineForOp> &foropvector, int indexofforloop,
                      std::vector<Operation *> &references);
  void getDependenceMatrix(AffineForOp forOp, unsigned maxLoopDepth,
                           std::vector<std::vector<int64_t>> *depCompsVec);
  void getLoopCosts(std::vector<AffineForOp> &forops);

private:
  llvm::DenseMap<Operation *, std::vector<std::vector<int64_t>>>
      LoopAccessMatrix;
  llvm::DenseMap<Operation *, std::vector<int64_t>> const_vector;
  // used to detect the presence of mod/ceildiv/floordiv operations by the
  // getAccessMatrix() method. if any invalid operations are found, this flag is
  // set and the LoopAccessMatrix and const_vector are cleared. any subsequent
  // operations on the loop nest must first check this flag first.
  
  llvm::DenseMap<Operation *, unsigned> elementSize;
  std::vector<unsigned> iteration_counts;
  std::vector<int> loop_dependence_vector;
  std::unordered_map<AffineForOp *, long double> loopcost;
  double r;
  std::vector<std::set<Operation *>> groups;
};
} // end anonymous namespace

SmallVector<Operation *, 4> loadsAndStores;

bool hasIfStatement(AffineForOp forOp) {
  int iffound = 0;
  forOp.walk([&](Operation *op) {
    if (isa<AffineIfOp>(op))
      iffound = 1;
  });
  return iffound;
}

// test for cases where the lower bound / upper bound has some constant variable
// i<=x rather than a simple contant value
bool testRectangularForloopNest(std::vector<AffineForOp> loopnest) {
  for (auto a : loopnest) {
    if (!a.hasConstantUpperBound() || !a.hasConstantLowerBound())
      return false;
  }
  return true;
}

/// need to create some ID mechanism for all the operations (load/store)
/// denseMap<unsigned, Operation> Store the accessMatrix info as a map
/// <Operation*, vector<vector<int>> and map<Operation*, vector<int> constants>


void LoopInterchange::getAccessMatrix(
    AffineForOp op, std::vector<AffineForOp>& foropvector,
    llvm::DenseMap<Value, int64_t>& index_values) {
    // for the mod / ceildiv / floordiv operations, consider them similar to
    // access to dimensions.
    this->LoopAccessMatrix.clear();
    this->const_vector.clear();
    SmallVector<Operation*, 8> loadAndStoreOpInsts;
    unsigned opcount = 0;
    op.getOperation()->walk([&](Operation* opInst) {
        // operation_depth[opcount++] = opInst;
        if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst)) {
            loadAndStoreOpInsts.push_back(opInst);
        }
        });
    opcount = 0;

    unsigned numOps = loadAndStoreOpInsts.size();
    for (unsigned i = 0; (i < numOps); ++i) {
        auto* srcOpInst = loadAndStoreOpInsts[i];
        MemRefAccess srcAccess(srcOpInst);
        AffineMap map;
        if (auto loadOp = dyn_cast<AffineLoadOp>(srcAccess.opInst))
            map = loadOp.getAffineMap();
        else if (auto storeOp = dyn_cast<AffineStoreOp>(srcAccess.opInst))
            map = storeOp.getAffineMap();
        SmallVector<Value, 8> operands(srcAccess.indices.begin(),
            srcAccess.indices.end());

        fullyComposeAffineMapAndOperands(&map, &operands);
        map = simplifyAffineMap(map);
        canonicalizeMapAndOperands(&map, &operands);
        auto a = map.getResults(); // ArrayRef<mlir::AffineExpr>
        // no. of values in a = NO. of dimensions of array
        uint64_t n_dim = map.getNumDims();
        uint64_t n_symbols = map.getNumSymbols();

        // ROW: DIMENSIONS of the array
        // COL: (NO. OF DIMS + NO. OF SYMBOLS)
        std::vector<std::vector<int64_t>> accessMatrix;
        unsigned I_matrix[n_dim + n_symbols];
        std::vector<int64_t> const_vector(a.size());
        // accesssing the affine operations.
        // all for a single instruction

        for (auto l = 0; l < a.size(); l++) {
            // a has a size equal to the dimension of the array being accessed.
            AffineExpr b = a[l]; // expression in the l-th dimension of the array

            std::vector<int64_t> Row(std::max(foropvector.size(), n_dim + n_symbols), 0);

            // check if b is not a constant expr.
            // If it is constant expr like A[5], then no need to walk it. 5 = [0 0
            // 0][i j k]' + [5] Instead simply push the value in const_vector and
            // leave the Row matrix to be empty full of zeroes.

            if (b.getKind() == AffineExprKind::Constant) {
                auto constant = b.cast<AffineConstantExpr>();
                const_vector[l] = (constant.getValue());
            }
            else {
                // constructing one row of the access matrix

                b.walk([&](AffineExpr expr) {
                    // all these expr are sub-level expressions inside the b (top-level)
                    // b = expr op expr op expr ...

                    switch (expr.getKind()) {

                        // A[ a + b] ----------> a and b themselves can be expressions.
                        // but if there is a constant involved, it will always be on the
                        // right. Then, A[a + constant] = [1 0 0 .. ][a b c ...]' +
                        // [constant]

                    case AffineExprKind::Add: {
                        bool insertF = true;
                        AffineBinaryOpExpr op = expr.cast<AffineBinaryOpExpr>();
                        auto lhs = op.getLHS();
                        auto rhs = op.getRHS();
                        unsigned lhs_position = 0;
                        unsigned rhs_position = 0;

                        auto lhskind = lhs.getKind();

                        if (lhskind == AffineExprKind::DimId) {
                            auto dim = lhs.cast<AffineDimExpr>();
                            lhs_position = index_values[operands[dim.getPosition()]];
                        }
                        else if (lhskind == AffineExprKind::SymbolId) {
                            auto symbol = lhs.cast<AffineSymbolExpr>();
                            lhs_position = n_dim + symbol.getPosition();
                        }

                        // there cannot be a constant in the left. However since it was
                        // already written..

                        else if (lhskind == AffineExprKind::Constant) {
                            int cons = rhs.cast<AffineConstantExpr>().getValue();
                            const_vector.push_back(cons);
                            insertF = false;
                        }

                        // only if there is a 0 multiple for this variable in the access
                        // matrix, mark a 1. A[a + b] = [1 ...][a ... ]' however, if there
                        // is a non-zero multiple present, it means that the variable has
                        // some multiple x associated with some sub-expr. A[ c*a + ...] =
                        // [c ...][a ..]' in that case, no need to reset the value/ add
                        // the value on the access matrix

                        if (Row[lhs_position] == 0 && insertF)
                            Row[lhs_position] = 1;

                        int64_t cons = 0;
                        auto rhskind = rhs.getKind();
                        insertF = true; // to flag whether or not to modify the Row
                                        // or not. Please note that the Row will not
                                        // be modified in case of a constant.
                        if (rhskind == AffineExprKind::DimId) {
                            auto dim = rhs.cast<AffineDimExpr>();
                            rhs_position = dim.getPosition();
                            rhs_position = index_values[operands[rhs_position]];
                        }
                        else if (rhskind == AffineExprKind::SymbolId) {
                            auto symbol = rhs.cast<AffineSymbolExpr>();
                            rhs_position = symbol.getPosition();
                        }
                        else if (rhskind == AffineExprKind::Constant) {
                            cons = rhs.cast<AffineConstantExpr>().getValue();
                            // if the rhs is a constant, then simply add this constant to
                            // this dimension.
                            const_vector[l] += cons;
                            insertF = false;
                        }

                        if (Row[rhs_position] == 0 && insertF)
                            Row[rhs_position] = 1;

                        break;
                    }

                                            // A[5 * j] = [0 5 0][i j k]' + [0]
                                            // The constant multiple is always on right.

                    case AffineExprKind::Mul: {
                        AffineBinaryOpExpr op = expr.cast<AffineBinaryOpExpr>();
                        auto lhs = op.getLHS();
                        auto rhs = op.getRHS();
                        unsigned position = 0;
                        switch (lhs.getKind()) {
                        case AffineExprKind::DimId: {
                            auto dim = lhs.cast<AffineDimExpr>();
                            position = index_values[operands[dim.getPosition()]];
                            break;
                        }
                        case AffineExprKind::SymbolId: {
                            auto symbol = lhs.cast<AffineSymbolExpr>();
                            position = n_dim + symbol.getPosition();
                            break;
                        }
                        }
                        switch (rhs.getKind()) {
                        case AffineExprKind::Constant: {
                            auto constant = rhs.cast<AffineConstantExpr>();
                            Row[position] = constant.getValue();
                            break;
                        }
                        }
                        break;
                    }

                                            // for DimID such as A[dim] = [0 1 0][j dim k]' + [0]

                    case AffineExprKind::DimId: {
                        auto dim = expr.cast<AffineDimExpr>();
                        Row[index_values[operands[dim.getPosition()]]] = 1;
                        const_vector[l] += 0;
                        break;
                    }

                                              // for symbolID such as A[l] = [0 0 1][j k l]' + [0]

                    case AffineExprKind::SymbolId: {
                        auto symbol = expr.cast<AffineDimExpr>();
                        Row[index_values[operands[symbol.getPosition()]]] = 1;
                        const_vector[l] += 0;
                        break;
                    }

                    case AffineExprKind::CeilDiv: // A[i op 5] ========> op =
                                                  // [ceildiv,floordiv, mod]
                    case AffineExprKind::FloorDiv:
                    case AffineExprKind::Mod: {
                        auto dim = expr.cast<AffineBinaryOpExpr>();
                        auto lhs = dim.getLHS();
                        // rhs is always contant or symbolic
                        auto rhs = dim.getRHS();
                        unsigned position = 0;
                        switch (lhs.getKind()) {
                        case AffineExprKind::DimId: {
                            auto dim = lhs.cast<AffineDimExpr>();
                            Row[index_values[operands[dim.getPosition()]]] = 1;
                            break;
                        }
                        case AffineExprKind::SymbolId: {
                            auto symbol = lhs.cast<AffineSymbolExpr>();
                            Row[n_dim + symbol.getPosition()] = 1;
                            break;
                        }
                        }
                        switch (rhs.getKind()) {
                        case AffineExprKind::Constant: {
                            auto constant = rhs.cast<AffineConstantExpr>();
                            break;
                        }
                        case AffineExprKind::SymbolId: {
                            auto symbol = rhs.cast<AffineSymbolExpr>();
                            Row[n_dim + symbol.getPosition()] = 1;
                            break;
                        }
                        }
                    }
                    }

                    });
            }

            // inserting a row into AccessMatrix
            // Row may be allowed to be only zeroes for cases such as A[5]

            accessMatrix.push_back(Row);
        }

        this->LoopAccessMatrix[srcOpInst] = accessMatrix;
        this->const_vector[srcOpInst] = const_vector;
        const_vector.clear();
        accessMatrix.clear();
    }
}

void handleSiblingNestingLoops(AffineForOp parentOp, AffineForOp forOpA,
                               AffineForOp forOpB) {
  OpBuilder builder(parentOp.getOperation()->getBlock(),
                    std::next(Block::iterator(parentOp)));
  auto copyParentForOp = cast<AffineForOp>(builder.clone(*parentOp));
  // remove forOpA from the original loopnest.

  int pos = 0, pos2 = 0;
  int parentpos = 0;
  int i = 0;
  bool update = true;
  parentOp.getOperation()->walk([&](AffineForOp op) {
    i++;
    if (op.getOperation() == forOpA.getOperation()) {
      pos = i;
    }
    if (parentOp.getOperation() == op.getOperation()) {
      parentpos = i;
    }
  });
  copyParentForOp.getOperation()->walk([&](AffineForOp opp) {
    pos2++;
    if ((pos2 != pos) && (pos2 != parentpos)) {
      opp.getOperation()->erase();
    }
  });
  forOpA.getOperation()->erase();
  // forOpB.getOperation()->erase();
}

void LoopInterchange::handleNestedLoops(Operation &func) {  // converts the imperfectly nested loop nest to a perfectly nested loop nest by loop splitting.
  std::vector<AffineForOp> foropvector;
  std::vector<Operation *> opvector;
  llvm::DenseMap<Operation *, std::vector<AffineForOp>> fortree;
  llvm::DenseMap<Operation *, AffineForOp> op_forop;

  func.walk([&](AffineForOp op) {
    foropvector.push_back(op);
    if (op.getParentOp()->getName().getStringRef() == "affine.for")
      fortree[op.getOperation()->getParentOp()].push_back(op);
    op_forop[op.getOperation()] = op;
  });

  for (auto a : fortree) {
    for (auto b = a.second.size() - 1; b > 0; b--) {
      handleSiblingNestingLoops(op_forop[a.first], a.second[b],
                                a.second[b - 1]);
    }
  }
  return;
}

void getValidPermutations(
    int n, AffineForOp forop,
    std::vector<std::vector<unsigned>> &validPermutations) {
  std::vector<unsigned> permutation;
  for (unsigned i = 0; i < n; i++)
    permutation.push_back(i);
  validPermutations.push_back(permutation);
  SmallVector<AffineForOp, 4> perfectloopnest;
  getPerfectlyNestedLoops(perfectloopnest, forop);
  while (std::next_permutation(permutation.begin(), permutation.end())) {
    if (isValidLoopInterchangePermutation(
            ArrayRef<AffineForOp>(perfectloopnest),
            ArrayRef<unsigned>(permutation)))
      validPermutations.push_back(permutation);
  }
}

#define STAR -99999999999999999
void LoopInterchange::getDependenceMatrix(
    AffineForOp forOp, unsigned maxLoopDepth,
    std::vector<std::vector<int64_t>> *depCompsVec) {
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  forOp.getOperation()->walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst))
      loadAndStoreOpInsts.push_back(opInst);
  });
  // hasDependences.clear();
  bool hasDependencepair = false;
  unsigned numOps = loadAndStoreOpInsts.size();
  for (unsigned i = 0; i < numOps; ++i) {
    auto *srcOpInst = loadAndStoreOpInsts[i];
    for (unsigned j = 0; j < numOps; ++j) {
      auto *dstOpInst = loadAndStoreOpInsts[j];
      hasDependencepair = false;
      for (unsigned d = 1; d <= maxLoopDepth + 1; ++d) {
        MemRefAccess srcAccess(srcOpInst);
        MemRefAccess dstAccess(dstOpInst);
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints, &depComps);

        if (mlir::hasDependence(result)) {
          hasDependences[std::make_pair(std::make_pair(srcOpInst, dstOpInst),
                                        d)] = 1;
          std::vector<int64_t> components;
          for (auto a : depComps) {
            if (a.lb.getValue() * a.ub.getValue() >= 0) {
              components.push_back((a.lb.getValue() + a.ub.getValue()) / 2);
            } else {
              if (a.lb.getValue() < 0 && a.ub.getValue() > 0)
                components.push_back(a.lb.getValue());
            }
          }
          depCompsVec->push_back(components);
          break;
        }
      }
    }
  }
  if (depCompsVec->size() == 0) {
    // create a zero vector
    std::vector<int64_t> depc(maxLoopDepth);
    depCompsVec->push_back(depc);
  }
  // set the dependence vector
  std::vector<int> depVec((*depCompsVec)[0].size());
  for (int i = 0; i < ((*depCompsVec)[0].size()); i++) {
    for (int j = 0; j < depCompsVec->size(); j++) {
      if (((*depCompsVec)[j][i]) != 0)
        depVec[i] = 1;
    }
  }
  this->loop_dependence_vector = depVec;
}

int64_t getPermutationParallelismCost(std::vector<unsigned> permutation,
                                      std::vector<int> loop_dependence_vector,
                                      std::vector<unsigned> iteration_vector) {
  unsigned cost = 0;
  for (int i = 0; i < permutation.size(); i++) {
    if (loop_dependence_vector[permutation[i]]) {
      int c = 1;
      for (int j = i + 1; j < permutation.size(); j++) {
        c *= iteration_vector[permutation[j]];
      }
      cost += c;
    }
  }
  return cost;
}

void LoopInterchange::buildRefGroups(
    AffineForOp ForOp, std::vector<std::set<Operation *>> *refGroups,
    unsigned max_depth, unsigned innermostlooplevel) {
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  ForOp.getOperation()->walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      MemRefType memreft;
      if (isa<AffineLoadOp>(op)) {
        AffineLoadOp loadOp = dyn_cast<AffineLoadOp>(*op);
        memreft = loadOp.getMemRefType();
      } else {
        AffineStoreOp storeOp = dyn_cast<AffineStoreOp>(*op);
        memreft = storeOp.getMemRefType();
      }

      // get size of data
      auto elementType = memreft.getElementType();

      unsigned sizeInBits = 0;
      if (elementType.isIntOrFloat()) {
        sizeInBits = elementType.getIntOrFloatBitWidth();
      }

      unsigned elementsize = llvm::divideCeil(sizeInBits, 8);
      this->elementSize[op] = elementsize;
      loadAndStoreOpInsts.push_back(op);
    }
  });
  // now test for dependences
  std::vector<std::set<Operation *>> groups(loadAndStoreOpInsts.size());
  std::unordered_map<Operation *, int> getlocation;
  std::unordered_map<Operation *, int> getlocation2;
  for (unsigned i = 0; i < loadAndStoreOpInsts.size(); i++) {
    getlocation[loadAndStoreOpInsts[i]] = i;
    getlocation2[loadAndStoreOpInsts[i]] = i;
    groups[i].insert(loadAndStoreOpInsts[i]);
  }
  // case 1
  unsigned numOps = loadAndStoreOpInsts.size();

  bool dependencefound = false;
  // for (unsigned d = 1; d <= max_depth+1; d++) {
  for (unsigned i = 0; i < numOps; ++i) {
    auto *srcOpInst = loadAndStoreOpInsts[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = i + 1; j < numOps; ++j) {
      auto *dstOpInst = loadAndStoreOpInsts[j];
      MemRefAccess dstAccess(dstOpInst);
      if (srcOpInst != dstOpInst) {
        if (srcAccess.memref == dstAccess.memref) {
          // get the access matrices of both the source and destination
          // operations.
          std::vector<int64_t> src_f = this->const_vector[srcOpInst];
          std::vector<int64_t> dest_f = this->const_vector[dstOpInst];

          bool identicalF_values =
              true; // all the values should be similar except the last value
          for (int i = 0; i < src_f.size() - 1; i++) {
            if ((src_f[i] != dest_f[i])) {
              identicalF_values = false;
              break;
            }
          }

          std::vector<std::vector<int64_t>> lams =
              this->LoopAccessMatrix[srcOpInst];
          std::vector<std::vector<int64_t>> lamd =
              this->LoopAccessMatrix[dstOpInst];

          if (this->LoopAccessMatrix[srcOpInst] ==
                  this->LoopAccessMatrix[dstOpInst] &&
              identicalF_values) {
            // test for the last dimension access. The difference should be
            // constant. create the MemRefType object
            unsigned elementSize = this->elementSize[srcOpInst];

            if (abs(src_f[src_f.size() - 1] - dest_f[dest_f.size() - 1]) <=
                cache_line_size / elementSize) {
              groups[getlocation[srcOpInst]].insert(
                  groups[getlocation[dstOpInst]].begin(),
                  groups[getlocation[dstOpInst]].end());
              groups.erase(groups.begin() + getlocation[dstOpInst]);
              getlocation[dstOpInst] = getlocation[srcOpInst];
            }
          }
        } else {
          for (unsigned d = 1; d <= max_depth + 1; d++) {
            FlatAffineConstraints dependenceConstraints;
            SmallVector<DependenceComponent, 2> depComps;
            DependenceResult result = checkMemrefAccessDependence(
                srcAccess, dstAccess, d, &dependenceConstraints, &depComps);
            if (mlir::hasDependence(result)) {
              if (d == max_depth + 1) {
                // there is a loop-independent dependence
                groups[getlocation[srcOpInst]].insert(
                    groups[getlocation[dstOpInst]].begin(),
                    groups[getlocation[dstOpInst]].end());
                groups.erase(groups.begin() + getlocation[dstOpInst]);
                getlocation[dstOpInst] = getlocation[srcOpInst];
              } else if (!dependencefound) {
                // search for all the dependence values other than this level
                bool gooddependency = true;
                for (unsigned i = 0; i < max_depth; i++) {
                  if ((depComps[i].lb.getValue() != 0) &&
                      (i != innermostlooplevel)) {
                    gooddependency = false;
                    break;
                  }
                }
                if (gooddependency &&
                    abs(depComps[innermostlooplevel].lb.getValue()) <= 2) {
                  groups[getlocation[srcOpInst]].insert(
                      groups[getlocation[dstOpInst]].begin(),
                      groups[getlocation[dstOpInst]].end());
                  groups.erase(groups.begin() + getlocation[dstOpInst]);
                  getlocation[dstOpInst] = getlocation[srcOpInst];
                }
                dependencefound = true;
              }
            }
          }
        }
      }
    }
  }
  //}
  *refGroups = groups;
  this->groups = groups;
}

void LoopInterchange::getLoopCosts(std::vector<AffineForOp> &foropvector) {
  long double totaliterations = 1;
  for (int i = 0; i < foropvector.size(); i++)
    totaliterations *= this->iteration_counts[i];
  for (int innerloop = 0; innerloop < foropvector.size(); innerloop++) {
    AffineForOp forop = foropvector[innerloop];
    float step = forop.getStep();
    float trip =
        (forop.getConstantUpperBound() - forop.getConstantLowerBound() + step) /
        step;
    long double loopcost = 0;

    for (int i = 0; i < this->groups.size(); i++) {
      Operation *op = *this->groups[i].begin();
      float stride =
          step * this->LoopAccessMatrix[op][this->LoopAccessMatrix[op].size() -
                                            1][innerloop];
      long double refcost = trip;
      int test = 1;
      for (int j = 0; j < this->LoopAccessMatrix[op].size(); j++) {
        if (this->LoopAccessMatrix[op][this->LoopAccessMatrix[op].size() - 1]
                                  [innerloop] != 0)
          test = 0;
      }
      if (test) {
        refcost = 1;
      } else if (stride < 8) {
        test = 1;
        for (int j = 0; j < this->LoopAccessMatrix[op].size() - 1; j++) {
          if (this->LoopAccessMatrix[op][this->LoopAccessMatrix[op].size() - 1]
                                    [innerloop] != 0)
            test = 0;
        }
        if (test) {
          refcost = (trip * stride) / 8;
        }
      }
      loopcost += refcost / this->groups[i].size();
    }
    this->loopcost[&foropvector[innerloop]] = loopcost * totaliterations / trip;
  }

  double e = 0;
  for (int i = 0; i < this->loopcost.size(); i++) {
    for (int j = 0; j < this->loopcost.size(); j++) {
      if (this->loopcost[&foropvector[i]] / this->loopcost[&foropvector[j]] > e)
        e = this->loopcost[&foropvector[i]] / this->loopcost[&foropvector[j]];
    }
  }
  this->r = e;
}

long double getPermutationCostLocality(
    std::vector<unsigned> permutation, std::vector<AffineForOp> &foropvector,
    std::unordered_map<AffineForOp *, long double> loopcost, double fr) {
  long double cost = 0;
  for (int i = 0; i < loopcost.size(); i++) {
    cost += pow(fr, /*foropvector.size()-1-*/ loopcost.size() - 1 -
                        permutation[i]) *
            loopcost[&foropvector[i]];
  }
  return cost;
}

// TODO: test the getAccessMatrix again for all the possible cases -- take help
// from test file.
void LoopInterchange::runOnFunction() {
  //handleNestedLoops();

  std::vector<AffineForOp> foropvector;
  std::vector<Value> forindexvector;
  std::vector<std::vector<AffineForOp>> loopnests;
  int loopnestc = 0;
  std::vector<SmallVector<DependenceComponent, 2>> dependences;

  // used to hold the id for the for-op induction vars and other variables
  // passed in the access function.
  llvm::DenseMap<Value, int64_t> index_values;
  int index_count = 0;

  // llvm::DenseMap<Operation, unsigned> operation_depth;
  // llvm::DenseMap<AffineForOp, unsigned> ForOp_depth;
  unsigned forcount = 0;
  unsigned opcount = 0;

  Operation *function = getFunction().getOperation();
  handleNestedLoops(*function);
  (*function).walk([&](AffineForOp op) { // innermost loop first order.
    // test if it is a top-level affine for op
    foropvector.push_back(op);
    if (op.hasConstantUpperBound() && op.hasConstantLowerBound()) {
      this->iteration_counts.push_back(
          (op.getConstantUpperBound() - op.getConstantLowerBound()) /
          op.getStep());
    }
    Value loop = op.getInductionVar();
    index_values[loop] = index_count;
    index_count++;

    // identify if the current forOp is the head of a loopnest.

    if ((op.getParentOp()->getName().getStringRef().str() == "func")) {
      std::reverse(foropvector.begin(), foropvector.end());
      loopnests.push_back(foropvector);
      //std::vector<std::vector<int64_t>> dependenceMatrix;

      //getDependenceMatrix(op, foropvector.size(), &dependenceMatrix);
      
      // the loop nest should not have any if statement and should be
      // rectangular in space.
      if (!hasIfStatement(op) && testRectangularForloopNest(foropvector)) {

        // modify the index_values map such that the topmost loop induction var
        // has an id = 0
        for (auto a : index_values)
          index_values[a.first] = index_count - 1 - a.second;

        std::vector<std::vector<unsigned>> validPermutations;
        getValidPermutations(foropvector.size(), op, validPermutations);
        if (validPermutations.size() <= 1) {
          // do not process. go to next loopnest.
          goto newforloop;
        }

        std::vector<std::vector<int64_t>> depComp;

        // get the access matrix for the entire loop nest stored in
        // TODO: Test if the values are consistent.

        getAccessMatrix(op, foropvector, index_values);

        std::vector<std::set<Operation *>> refGroups;
        std::vector<std::vector<int64_t>> dependenceMatrix;
	
        getDependenceMatrix(op, foropvector.size(), &dependenceMatrix);
	    // print the dependence matrix
        // int64_t cost = getPermutationParallelismCost(a,
        // this->loop_dependence_vector, this->iteration_counts);
        buildRefGroups(op, &refGroups, foropvector.size(), 1);
        getLoopCosts(foropvector);

        long double mincost = 999999999999999999999;
        std::vector<unsigned> bestPermutation;

        for (auto a : validPermutations) {
          int64_t parallelCost = getPermutationParallelismCost(
              a, this->loop_dependence_vector, this->iteration_counts);
          long double localitycost = getPermutationCostLocality(
              a, foropvector, this->loopcost, this->r);
          long double cost = 100 * parallelCost + localitycost;

          if (cost < mincost) {
            mincost = cost;
            bestPermutation = a;
          }
        }

        permuteLoops(MutableArrayRef<AffineForOp>(foropvector),
                     ArrayRef<unsigned>(bestPermutation));
      }

    // for every loop nest - whether or not it has to be processed.
    // clear the foropvector so that new for loops in the loop nest get added
    // again. index_values: these are valid only for this loop nest since the
    // new for nest will have new block arguments and hence need new id
    // index_count : used to insert id values in the index_values.
    newforloop:
      foropvector.clear();
      index_values.clear();
      index_count = 0;
      forcount = 0;
      this->iteration_counts.clear();
      this->loop_dependence_vector.clear();
      this->loopcost.clear();
      this->groups.clear();
      this->elementSize.clear();
      this->LoopAccessMatrix.clear();
      this->const_vector.clear();
    }

  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopInterchange() {
  return std::make_unique<LoopInterchange>();
}
