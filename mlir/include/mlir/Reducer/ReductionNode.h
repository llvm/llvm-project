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

#include "mlir/Reducer/Tester.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

/// Defines the traversal method options to be used in the reduction tree
/// traversal.
enum TraversalMode { SinglePath, Backtrack, MultiPath };

/// This class defines the ReductionNode which is used to generate variant and
/// keep track of the necessary metadata for the reduction pass. The nodes are
/// linked together in a reduction tree structure which defines the relationship
/// between all the different generated variants.
class ReductionNode {
public:
  template <TraversalMode mode>
  class iterator;

  using Range = std::pair<int, int>;

  ReductionNode(ReductionNode *parent, std::vector<Range> range,
                llvm::SpecificBumpPtrAllocator<ReductionNode> &allocator);

  ReductionNode *getParent() const;

  size_t getSize() const;

  /// Returns true if the module exhibits the interesting behavior.
  Tester::Interestingness isInteresting() const;

  std::vector<Range> getRanges() const;

  std::vector<ReductionNode *> &getVariants();

  /// Split the ranges and generate new variants.
  std::vector<ReductionNode *> generateNewVariants();

  /// Update the interestingness result from tester.
  void update(std::pair<Tester::Interestingness, size_t> result);

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
      std::vector<ReductionNode *> neighbors = getNeighbors(top);
      for (ReductionNode *node : neighbors)
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
    std::vector<ReductionNode *> getNeighbors(ReductionNode *node) {
      return static_cast<T *>(this)->getNeighbors(node);
    }

  private:
    std::queue<ReductionNode *> visitQueue;
  };

  /// The size of module after applying the range constraints.
  size_t size;

  /// This is true if the module has been evaluated and it exhibits the
  /// interesting behavior.
  Tester::Interestingness interesting;

  ReductionNode *parent;

  /// We will only keep the operation with index falls into the ranges.
  /// For example, number each function in a certain module and then we will
  /// remove the functions with index outside the ranges and see if the
  /// resulting module is still interesting.
  std::vector<Range> ranges;

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
  std::vector<ReductionNode *> getNeighbors(ReductionNode *node);
};

} // end namespace mlir

#endif
