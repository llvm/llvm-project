//===- ReductionNode.h - Reduction Node Implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the reduction nodes which are used to track of the metadata
// for a specific generated variant within a reduction pass and are the building
// blocks of the reduction tree structure. A reduction tree is used to keep
// track of the different generated variants throughout a reduction pass in the
// MLIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONNODE_H
#define MLIR_REDUCER_REDUCTIONNODE_H

#include <queue>
#include <vector>

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

class ModuleOp;
class Region;

/// Defines the traversal method options to be used in the reduction tree
/// traversal.
enum TraversalMode { SinglePath, Backtrack, MultiPath };

/// ReductionTreePass will build a reduction tree during module reduction and
/// the ReductionNode represents the vertex of the tree. A ReductionNode records
/// the information such as the reduced module, how this node is reduced from
/// the parent node, etc. This information will be used to construct a reduction
/// path to reduce the certain module.
class ReductionNode {
public:
  template <TraversalMode mode>
  class iterator;

  using Range = std::pair<int, int>;

  ReductionNode(ReductionNode *parent, std::vector<Range> range,
                llvm::SpecificBumpPtrAllocator<ReductionNode> &allocator);

  ReductionNode *getParent() const { return parent; }

  /// If the ReductionNode hasn't been tested the interestingness, it'll be the
  /// same module as the one in the parent node. Otherwise, the returned module
  /// will have been applied certain reduction strategies. Note that it's not
  /// necessary to be an interesting case or a reduced module (has smaller size
  /// than parent's).
  ModuleOp getModule() const { return module.get(); }

  /// Return the region we're reducing.
  Region &getRegion() const { return *region; }

  /// Return the size of the module.
  size_t getSize() const { return size; }

  /// Returns true if the module exhibits the interesting behavior.
  Tester::Interestingness isInteresting() const { return interesting; }

  /// Return the range information that how this node is reduced from the parent
  /// node.
  ArrayRef<Range> getStartRanges() const { return startRanges; }

  /// Return the range set we are using to generate variants.
  ArrayRef<Range> getRanges() const { return ranges; }

  /// Return the generated variants(the child nodes).
  ArrayRef<ReductionNode *> getVariants() const { return variants; }

  /// Split the ranges and generate new variants.
  ArrayRef<ReductionNode *> generateNewVariants();

  /// Update the interestingness result from tester.
  void update(std::pair<Tester::Interestingness, size_t> result);

  /// Each Reduction Node contains a copy of module for applying rewrite
  /// patterns. In addition, we only apply rewrite patterns in a certain region.
  /// In init(), we will duplicate the module from parent node and locate the
  /// corresponding region.
  LogicalResult initialize(ModuleOp parentModule, Region &parentRegion);

private:
  /// A custom BFS iterator. The difference between
  /// llvm/ADT/BreadthFirstIterator.h is the graph we're exploring is dynamic.
  /// We may explore more neighbors at certain node if we didn't find interested
  /// event. As a result, we defer pushing adjacent nodes until poping the last
  /// visited node. The graph exploration strategy will be put in
  /// getNeighbors().
  ///
  /// Subclass BaseIterator and implement traversal strategy in getNeighbors().
  template <typename T>
  class BaseIterator {
  public:
    BaseIterator(ReductionNode *node) { visitQueue.push(node); }
    BaseIterator(const BaseIterator &) = default;
    BaseIterator() = default;

    static BaseIterator end() { return BaseIterator(); }

    bool operator==(const BaseIterator &i) {
      return visitQueue == i.visitQueue;
    }
    bool operator!=(const BaseIterator &i) { return !(*this == i); }

    BaseIterator &operator++() {
      ReductionNode *top = visitQueue.front();
      visitQueue.pop();
      for (ReductionNode *node : getNeighbors(top))
        visitQueue.push(node);
      return *this;
    }

    BaseIterator operator++(int) {
      BaseIterator tmp = *this;
      ++*this;
      return tmp;
    }

    ReductionNode &operator*() const { return *(visitQueue.front()); }
    ReductionNode *operator->() const { return visitQueue.front(); }

  protected:
    ArrayRef<ReductionNode *> getNeighbors(ReductionNode *node) {
      return static_cast<T *>(this)->getNeighbors(node);
    }

  private:
    std::queue<ReductionNode *> visitQueue;
  };

  /// This is a copy of module from parent node. All the reducer patterns will
  /// be applied to this instance.
  OwningOpRef<ModuleOp> module;

  /// The region of certain operation we're reducing in the module
  Region *region;

  /// The node we are reduced from. It means we will be in variants of parent
  /// node.
  ReductionNode *parent;

  /// The size of module after applying the reducer patterns with range
  /// constraints. This is only valid while the interestingness has been tested.
  size_t size;

  /// This is true if the module has been evaluated and it exhibits the
  /// interesting behavior.
  Tester::Interestingness interesting;

  /// `ranges` represents the selected subset of operations in the region. We
  /// implictly number each operation in the region and ReductionTreePass will
  /// apply reducer patterns on the operation falls into the `ranges`. We will
  /// generate new ReductionNode with subset of `ranges` to see if we can do
  /// further reduction. we may split the element in the `ranges` so that we can
  /// have more subset variants from `ranges`.
  /// Note that after applying the reducer patterns the number of operation in
  /// the region may have changed, we need to update the `ranges` after that.
  std::vector<Range> ranges;

  /// `startRanges` records the ranges of operations selected from the parent
  /// node to produce this ReductionNode. It can be used to construct the
  /// reduction path from the root. I.e., if we apply the same reducer patterns
  /// and `startRanges` selection on the parent region, we will get the same
  /// module as this node.
  const std::vector<Range> startRanges;

  /// This points to the child variants that were created using this node as a
  /// starting point.
  std::vector<ReductionNode *> variants;

  llvm::SpecificBumpPtrAllocator<ReductionNode> &allocator;
};

// Specialized iterator for SinglePath traversal
template <>
class ReductionNode::iterator<SinglePath>
    : public BaseIterator<iterator<SinglePath>> {
  friend BaseIterator<iterator<SinglePath>>;
  using BaseIterator::BaseIterator;
  ArrayRef<ReductionNode *> getNeighbors(ReductionNode *node);
};

} // end namespace mlir

#endif // MLIR_REDUCER_REDUCTIONNODE_H
