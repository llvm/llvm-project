//===- PredicateTree.cpp - Predicate tree merging -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PredicateTree.h"
#include "RootOrdering.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "pdl-predicate-tree"

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

//===----------------------------------------------------------------------===//
// Predicate List Building
//===----------------------------------------------------------------------===//

static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              Position *pos);

/// Compares the depths of two positions.
static bool comparePosDepth(Position *lhs, Position *rhs) {
  return lhs->getOperationDepth() < rhs->getOperationDepth();
}

/// Returns the number of non-range elements within `values`.
static unsigned getNumNonRangeValues(ValueRange values) {
  return llvm::count_if(values.getTypes(),
                        [](Type type) { return !isa<pdl::RangeType>(type); });
}

static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              AttributePosition *pos) {
  assert(isa<pdl::AttributeType>(val.getType()) && "expected attribute type");
  predList.emplace_back(pos, builder.getIsNotNull());

  if (auto attr = dyn_cast<pdl::AttributeOp>(val.getDefiningOp())) {
    // If the attribute has a type or value, add a constraint.
    if (Value type = attr.getValueType())
      getTreePredicates(predList, type, builder, inputs, builder.getType(pos));
    else if (Attribute value = attr.getValueAttr())
      predList.emplace_back(pos, builder.getAttributeConstraint(value));
  }
}

/// Collect all of the predicates for the given operand position.
static void getOperandTreePredicates(std::vector<PositionalPredicate> &predList,
                                     Value val, PredicateBuilder &builder,
                                     DenseMap<Value, Position *> &inputs,
                                     Position *pos) {
  Type valueType = val.getType();
  bool isVariadic = isa<pdl::RangeType>(valueType);

  // If this is a typed operand, add a type constraint.
  TypeSwitch<Operation *>(val.getDefiningOp())
      .Case<pdl::OperandOp, pdl::OperandsOp>([&](auto op) {
        // Prevent traversal into a null value if the operand has a proper
        // index.
        if (std::is_same<pdl::OperandOp, decltype(op)>::value ||
            cast<OperandGroupPosition>(pos)->getOperandGroupNumber())
          predList.emplace_back(pos, builder.getIsNotNull());

        if (Value type = op.getValueType())
          getTreePredicates(predList, type, builder, inputs,
                            builder.getType(pos));
      })
      .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto op) {
        std::optional<unsigned> index = op.getIndex();

        // Prevent traversal into a null value if the result has a proper index.
        if (index)
          predList.emplace_back(pos, builder.getIsNotNull());

        // Get the parent operation of this operand.
        OperationPosition *parentPos = builder.getOperandDefiningOp(pos);
        predList.emplace_back(parentPos, builder.getIsNotNull());

        // Ensure that the operands match the corresponding results of the
        // parent operation.
        Position *resultPos = nullptr;
        if (std::is_same<pdl::ResultOp, decltype(op)>::value)
          resultPos = builder.getResult(parentPos, *index);
        else
          resultPos = builder.getResultGroup(parentPos, index, isVariadic);
        predList.emplace_back(resultPos, builder.getEqualTo(pos));

        // Collect the predicates of the parent operation.
        getTreePredicates(predList, op.getParent(), builder, inputs,
                          (Position *)parentPos);
      });
}

static void
getTreePredicates(std::vector<PositionalPredicate> &predList, Value val,
                  PredicateBuilder &builder,
                  DenseMap<Value, Position *> &inputs, OperationPosition *pos,
                  std::optional<unsigned> ignoreOperand = std::nullopt) {
  assert(isa<pdl::OperationType>(val.getType()) && "expected operation");
  pdl::OperationOp op = cast<pdl::OperationOp>(val.getDefiningOp());
  OperationPosition *opPos = cast<OperationPosition>(pos);

  // Ensure getDefiningOp returns a non-null operation.
  if (!opPos->isRoot())
    predList.emplace_back(pos, builder.getIsNotNull());

  // Check that this is the correct root operation.
  if (std::optional<StringRef> opName = op.getOpName())
    predList.emplace_back(pos, builder.getOperationName(*opName));

  // Check that the operation has the proper number of operands. If there are
  // any variable length operands, we check a minimum instead of an exact count.
  OperandRange operands = op.getOperandValues();
  unsigned minOperands = getNumNonRangeValues(operands);
  if (minOperands != operands.size()) {
    if (minOperands)
      predList.emplace_back(pos, builder.getOperandCountAtLeast(minOperands));
  } else {
    predList.emplace_back(pos, builder.getOperandCount(minOperands));
  }

  // Check that the operation has the proper number of results. If there are
  // any variable length results, we check a minimum instead of an exact count.
  OperandRange types = op.getTypeValues();
  unsigned minResults = getNumNonRangeValues(types);
  if (minResults == types.size())
    predList.emplace_back(pos, builder.getResultCount(types.size()));
  else if (minResults)
    predList.emplace_back(pos, builder.getResultCountAtLeast(minResults));

  // Recurse into any attributes, operands, or results.
  for (auto [attrName, attr] :
       llvm::zip(op.getAttributeValueNames(), op.getAttributeValues())) {
    getTreePredicates(
        predList, attr, builder, inputs,
        builder.getAttribute(opPos, cast<StringAttr>(attrName).getValue()));
  }

  // Process the operands and results of the operation. For all values up to
  // the first variable length value, we use the concrete operand/result
  // number. After that, we use the "group" given that we can't know the
  // concrete indices until runtime. If there is only one variadic operand
  // group, we treat it as all of the operands/results of the operation.
  /// Operands.
  if (operands.size() == 1 && isa<pdl::RangeType>(operands[0].getType())) {
    // Ignore the operands if we are performing an upward traversal (in that
    // case, they have already been visited).
    if (opPos->isRoot() || opPos->isOperandDefiningOp())
      getTreePredicates(predList, operands.front(), builder, inputs,
                        builder.getAllOperands(opPos));
  } else {
    bool foundVariableLength = false;
    for (const auto &operandIt : llvm::enumerate(operands)) {
      bool isVariadic = isa<pdl::RangeType>(operandIt.value().getType());
      foundVariableLength |= isVariadic;

      // Ignore the specified operand, usually because this position was
      // visited in an upward traversal via an iterative choice.
      if (ignoreOperand && *ignoreOperand == operandIt.index())
        continue;

      Position *pos =
          foundVariableLength
              ? builder.getOperandGroup(opPos, operandIt.index(), isVariadic)
              : builder.getOperand(opPos, operandIt.index());
      getTreePredicates(predList, operandIt.value(), builder, inputs, pos);
    }
  }
  /// Results.
  if (types.size() == 1 && isa<pdl::RangeType>(types[0].getType())) {
    getTreePredicates(predList, types.front(), builder, inputs,
                      builder.getType(builder.getAllResults(opPos)));
    return;
  }

  bool foundVariableLength = false;
  for (auto [idx, typeValue] : llvm::enumerate(types)) {
    bool isVariadic = isa<pdl::RangeType>(typeValue.getType());
    foundVariableLength |= isVariadic;

    auto *resultPos = foundVariableLength
                          ? builder.getResultGroup(pos, idx, isVariadic)
                          : builder.getResult(pos, idx);
    predList.emplace_back(resultPos, builder.getIsNotNull());
    getTreePredicates(predList, typeValue, builder, inputs,
                      builder.getType(resultPos));
  }
}

static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              TypePosition *pos) {
  // Check for a constraint on a constant type.
  if (pdl::TypeOp typeOp = val.getDefiningOp<pdl::TypeOp>()) {
    if (Attribute type = typeOp.getConstantTypeAttr())
      predList.emplace_back(pos, builder.getTypeConstraint(type));
  } else if (pdl::TypesOp typeOp = val.getDefiningOp<pdl::TypesOp>()) {
    if (Attribute typeAttr = typeOp.getConstantTypesAttr())
      predList.emplace_back(pos, builder.getTypeConstraint(typeAttr));
  }
}

/// Collect the tree predicates anchored at the given value.
static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              Position *pos) {
  // Make sure this input value is accessible to the rewrite.
  auto it = inputs.try_emplace(val, pos);
  if (!it.second) {
    // If this is an input value that has been visited in the tree, add a
    // constraint to ensure that both instances refer to the same value.
    if (isa<pdl::AttributeOp, pdl::OperandOp, pdl::OperandsOp, pdl::OperationOp,
            pdl::TypeOp>(val.getDefiningOp())) {
      auto minMaxPositions =
          std::minmax(pos, it.first->second, comparePosDepth);
      predList.emplace_back(minMaxPositions.second,
                            builder.getEqualTo(minMaxPositions.first));
    }
    return;
  }

  TypeSwitch<Position *>(pos)
      .Case<AttributePosition, OperationPosition, TypePosition>([&](auto *pos) {
        getTreePredicates(predList, val, builder, inputs, pos);
      })
      .Case<OperandPosition, OperandGroupPosition>([&](auto *pos) {
        getOperandTreePredicates(predList, val, builder, inputs, pos);
      })
      .Default([](auto *) { llvm_unreachable("unexpected position kind"); });
}

static void getAttributePredicates(pdl::AttributeOp op,
                                   std::vector<PositionalPredicate> &predList,
                                   PredicateBuilder &builder,
                                   DenseMap<Value, Position *> &inputs) {
  Position *&attrPos = inputs[op];
  if (attrPos)
    return;
  Attribute value = op.getValueAttr();
  assert(value && "expected non-tree `pdl.attribute` to contain a value");
  attrPos = builder.getAttributeLiteral(value);
}

static void getConstraintPredicates(pdl::ApplyNativeConstraintOp op,
                                    std::vector<PositionalPredicate> &predList,
                                    PredicateBuilder &builder,
                                    DenseMap<Value, Position *> &inputs) {
  OperandRange arguments = op.getArgs();

  std::vector<Position *> allPositions;
  allPositions.reserve(arguments.size());
  for (Value arg : arguments)
    allPositions.push_back(inputs.lookup(arg));

  // Push the constraint to the furthest position.
  Position *pos = *llvm::max_element(allPositions, comparePosDepth);
  ResultRange results = op.getResults();
  PredicateBuilder::Predicate pred = builder.getConstraint(
      op.getName(), allPositions, SmallVector<Type>(results.getTypes()),
      op.getIsNegated());

  // For each result register a position so it can be used later
  for (auto [i, result] : llvm::enumerate(results)) {
    ConstraintQuestion *q = cast<ConstraintQuestion>(pred.first);
    ConstraintPosition *pos = builder.getConstraintPosition(q, i);
    auto [it, inserted] = inputs.try_emplace(result, pos);
    // If this is an input value that has been visited in the tree, add a
    // constraint to ensure that both instances refer to the same value.
    if (!inserted) {
      Position *first = pos;
      Position *second = it->second;
      if (comparePosDepth(second, first))
        std::tie(second, first) = std::make_pair(first, second);

      predList.emplace_back(second, builder.getEqualTo(first));
    }
  }
  predList.emplace_back(pos, pred);
}

static void getResultPredicates(pdl::ResultOp op,
                                std::vector<PositionalPredicate> &predList,
                                PredicateBuilder &builder,
                                DenseMap<Value, Position *> &inputs) {
  Position *&resultPos = inputs[op];
  if (resultPos)
    return;

  // Ensure that the result isn't null.
  auto *parentPos = cast<OperationPosition>(inputs.lookup(op.getParent()));
  resultPos = builder.getResult(parentPos, op.getIndex());
  predList.emplace_back(resultPos, builder.getIsNotNull());
}

static void getResultPredicates(pdl::ResultsOp op,
                                std::vector<PositionalPredicate> &predList,
                                PredicateBuilder &builder,
                                DenseMap<Value, Position *> &inputs) {
  Position *&resultPos = inputs[op];
  if (resultPos)
    return;

  // Ensure that the result isn't null if the result has an index.
  auto *parentPos = cast<OperationPosition>(inputs.lookup(op.getParent()));
  bool isVariadic = isa<pdl::RangeType>(op.getType());
  std::optional<unsigned> index = op.getIndex();
  resultPos = builder.getResultGroup(parentPos, index, isVariadic);
  if (index)
    predList.emplace_back(resultPos, builder.getIsNotNull());
}

static void getTypePredicates(Value typeValue,
                              function_ref<Attribute()> typeAttrFn,
                              PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs) {
  Position *&typePos = inputs[typeValue];
  if (typePos)
    return;
  Attribute typeAttr = typeAttrFn();
  assert(typeAttr &&
         "expected non-tree `pdl.type`/`pdl.types` to contain a value");
  typePos = builder.getTypeLiteral(typeAttr);
}

/// Collect all of the predicates that cannot be determined via walking the
/// tree.
static void getNonTreePredicates(pdl::PatternOp pattern,
                                 std::vector<PositionalPredicate> &predList,
                                 PredicateBuilder &builder,
                                 DenseMap<Value, Position *> &inputs) {
  for (Operation &op : pattern.getBodyRegion().getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case([&](pdl::AttributeOp attrOp) {
          getAttributePredicates(attrOp, predList, builder, inputs);
        })
        .Case<pdl::ApplyNativeConstraintOp>([&](auto constraintOp) {
          getConstraintPredicates(constraintOp, predList, builder, inputs);
        })
        .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto resultOp) {
          getResultPredicates(resultOp, predList, builder, inputs);
        })
        .Case([&](pdl::TypeOp typeOp) {
          getTypePredicates(
              typeOp, [&] { return typeOp.getConstantTypeAttr(); }, builder,
              inputs);
        })
        .Case([&](pdl::TypesOp typeOp) {
          getTypePredicates(
              typeOp, [&] { return typeOp.getConstantTypesAttr(); }, builder,
              inputs);
        });
  }
}

namespace {

/// An op accepting a value at an optional index.
struct OpIndex {
  Value parent;
  std::optional<unsigned> index;
};

/// The parent and operand index of each operation for each root, stored
/// as a nested map [root][operation].
using ParentMaps = DenseMap<Value, DenseMap<Value, OpIndex>>;

} // namespace

/// Given a pattern, determines the set of roots present in this pattern.
/// These are the operations whose results are not consumed by other operations.
static SmallVector<Value> detectRoots(pdl::PatternOp pattern) {
  // First, collect all the operations that are used as operands
  // to other operations. These are not roots by default.
  DenseSet<Value> used;
  for (auto operationOp : pattern.getBodyRegion().getOps<pdl::OperationOp>()) {
    for (Value operand : operationOp.getOperandValues())
      TypeSwitch<Operation *>(operand.getDefiningOp())
          .Case<pdl::ResultOp, pdl::ResultsOp>(
              [&used](auto resultOp) { used.insert(resultOp.getParent()); });
  }

  // Remove the specified root from the use set, so that we can
  // always select it as a root, even if it is used by other operations.
  if (Value root = pattern.getRewriter().getRoot())
    used.erase(root);

  // Finally, collect all the unused operations.
  SmallVector<Value> roots;
  for (Value operationOp : pattern.getBodyRegion().getOps<pdl::OperationOp>())
    if (!used.contains(operationOp))
      roots.push_back(operationOp);

  return roots;
}

/// Given a list of candidate roots, builds the cost graph for connecting them.
/// The graph is formed by traversing the DAG of operations starting from each
/// root and marking the depth of each connector value (operand). Then we join
/// the candidate roots based on the common connector values, taking the one
/// with the minimum depth. Along the way, we compute, for each candidate root,
/// a mapping from each operation (in the DAG underneath this root) to its
/// parent operation and the corresponding operand index.
static void buildCostGraph(ArrayRef<Value> roots, RootOrderingGraph &graph,
                           ParentMaps &parentMaps) {

  // The entry of a queue. The entry consists of the following items:
  // * the value in the DAG underneath the root;
  // * the parent of the value;
  // * the operand index of the value in its parent;
  // * the depth of the visited value.
  struct Entry {
    Entry(Value value, Value parent, std::optional<unsigned> index,
          unsigned depth)
        : value(value), parent(parent), index(index), depth(depth) {}

    Value value;
    Value parent;
    std::optional<unsigned> index;
    unsigned depth;
  };

  // A root of a value and its depth (distance from root to the value).
  struct RootDepth {
    Value root;
    unsigned depth = 0;
  };

  // Map from candidate connector values to their roots and depths. Using a
  // small vector with 1 entry because most values belong to a single root.
  llvm::MapVector<Value, SmallVector<RootDepth, 1>> connectorsRootsDepths;

  // Perform a breadth-first traversal of the op DAG rooted at each root.
  for (Value root : roots) {
    // The queue of visited values. A value may be present multiple times in
    // the queue, for multiple parents. We only accept the first occurrence,
    // which is guaranteed to have the lowest depth.
    std::queue<Entry> toVisit;
    toVisit.emplace(root, Value(), 0, 0);

    // The map from value to its parent for the current root.
    DenseMap<Value, OpIndex> &parentMap = parentMaps[root];

    while (!toVisit.empty()) {
      Entry entry = toVisit.front();
      toVisit.pop();
      // Skip if already visited.
      if (!parentMap.insert({entry.value, {entry.parent, entry.index}}).second)
        continue;

      // Mark the root and depth of the value.
      connectorsRootsDepths[entry.value].push_back({root, entry.depth});

      // Traverse the operands of an operation and result ops.
      // We intentionally do not traverse attributes and types, because those
      // are expensive to join on.
      TypeSwitch<Operation *>(entry.value.getDefiningOp())
          .Case<pdl::OperationOp>([&](auto operationOp) {
            OperandRange operands = operationOp.getOperandValues();
            // Special case when we pass all the operands in one range.
            // For those, the index is empty.
            if (operands.size() == 1 &&
                isa<pdl::RangeType>(operands[0].getType())) {
              toVisit.emplace(operands[0], entry.value, std::nullopt,
                              entry.depth + 1);
              return;
            }

            // Default case: visit all the operands.
            for (const auto &p :
                 llvm::enumerate(operationOp.getOperandValues()))
              toVisit.emplace(p.value(), entry.value, p.index(),
                              entry.depth + 1);
          })
          .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto resultOp) {
            toVisit.emplace(resultOp.getParent(), entry.value,
                            resultOp.getIndex(), entry.depth);
          });
    }
  }

  // Now build the cost graph.
  // This is simply a minimum over all depths for the target root.
  unsigned nextID = 0;
  for (const auto &connectorRootsDepths : connectorsRootsDepths) {
    Value value = connectorRootsDepths.first;
    ArrayRef<RootDepth> rootsDepths = connectorRootsDepths.second;
    // If there is only one root for this value, this will not trigger
    // any edges in the cost graph (a perf optimization).
    if (rootsDepths.size() == 1)
      continue;

    for (const RootDepth &p : rootsDepths) {
      for (const RootDepth &q : rootsDepths) {
        if (&p == &q)
          continue;
        // Insert or retrieve the property of edge from p to q.
        RootOrderingEntry &entry = graph[q.root][p.root];
        if (!entry.connector /* new edge */ || entry.cost.first > q.depth) {
          if (!entry.connector)
            entry.cost.second = nextID++;
          entry.cost.first = q.depth;
          entry.connector = value;
        }
      }
    }
  }

  assert((llvm::hasSingleElement(roots) || graph.size() == roots.size()) &&
         "the pattern contains a candidate root disconnected from the others");
}

/// Returns true if the operand at the given index needs to be queried using an
/// operand group, i.e., if it is variadic itself or follows a variadic operand.
static bool useOperandGroup(pdl::OperationOp op, unsigned index) {
  OperandRange operands = op.getOperandValues();
  assert(index < operands.size() && "operand index out of range");
  for (unsigned i = 0; i <= index; ++i)
    if (isa<pdl::RangeType>(operands[i].getType()))
      return true;
  return false;
}

/// Visit a node during upward traversal.
static void visitUpward(std::vector<PositionalPredicate> &predList,
                        OpIndex opIndex, PredicateBuilder &builder,
                        DenseMap<Value, Position *> &valueToPosition,
                        Position *&pos, unsigned rootID) {
  Value value = opIndex.parent;
  TypeSwitch<Operation *>(value.getDefiningOp())
      .Case<pdl::OperationOp>([&](auto operationOp) {
        LLVM_DEBUG(llvm::dbgs() << "  * Value: " << value << "\n");

        // Get users and iterate over them.
        Position *usersPos = builder.getUsers(pos, /*useRepresentative=*/true);
        Position *foreachPos = builder.getForEach(usersPos, rootID);
        OperationPosition *opPos = builder.getPassthroughOp(foreachPos);

        // Compare the operand(s) of the user against the input value(s).
        Position *operandPos;
        if (!opIndex.index) {
          // We are querying all the operands of the operation.
          operandPos = builder.getAllOperands(opPos);
        } else if (useOperandGroup(operationOp, *opIndex.index)) {
          // We are querying an operand group.
          Type type = operationOp.getOperandValues()[*opIndex.index].getType();
          bool variadic = isa<pdl::RangeType>(type);
          operandPos = builder.getOperandGroup(opPos, opIndex.index, variadic);
        } else {
          // We are querying an individual operand.
          operandPos = builder.getOperand(opPos, *opIndex.index);
        }
        predList.emplace_back(operandPos, builder.getEqualTo(pos));

        // Guard against duplicate upward visits. These are not possible,
        // because if this value was already visited, it would have been
        // cheaper to start the traversal at this value rather than at the
        // `connector`, violating the optimality of our spanning tree.
        bool inserted = valueToPosition.try_emplace(value, opPos).second;
        (void)inserted;
        assert(inserted && "duplicate upward visit");

        // Obtain the tree predicates at the current value.
        getTreePredicates(predList, value, builder, valueToPosition, opPos,
                          opIndex.index);

        // Update the position
        pos = opPos;
      })
      .Case<pdl::ResultOp>([&](auto resultOp) {
        // Traverse up an individual result.
        auto *opPos = dyn_cast<OperationPosition>(pos);
        assert(opPos && "operations and results must be interleaved");
        pos = builder.getResult(opPos, *opIndex.index);

        // Insert the result position in case we have not visited it yet.
        valueToPosition.try_emplace(value, pos);
      })
      .Case<pdl::ResultsOp>([&](auto resultOp) {
        // Traverse up a group of results.
        auto *opPos = dyn_cast<OperationPosition>(pos);
        assert(opPos && "operations and results must be interleaved");
        bool isVariadic = isa<pdl::RangeType>(value.getType());
        if (opIndex.index)
          pos = builder.getResultGroup(opPos, opIndex.index, isVariadic);
        else
          pos = builder.getAllResults(opPos);

        // Insert the result position in case we have not visited it yet.
        valueToPosition.try_emplace(value, pos);
      });
}

/// Given a pattern operation, build the set of matcher predicates necessary to
/// match this pattern.
static Value buildPredicateList(pdl::PatternOp pattern,
                                PredicateBuilder &builder,
                                std::vector<PositionalPredicate> &predList,
                                DenseMap<Value, Position *> &valueToPosition) {
  SmallVector<Value> roots = detectRoots(pattern);

  // Build the root ordering graph and compute the parent maps.
  RootOrderingGraph graph;
  ParentMaps parentMaps;
  buildCostGraph(roots, graph, parentMaps);
  LLVM_DEBUG({
    llvm::dbgs() << "Graph:\n";
    for (auto &target : graph) {
      llvm::dbgs() << "  * " << target.first.getLoc() << " " << target.first
                   << "\n";
      for (auto &source : target.second) {
        RootOrderingEntry &entry = source.second;
        llvm::dbgs() << "      <- " << source.first << ": " << entry.cost.first
                     << ":" << entry.cost.second << " via "
                     << entry.connector.getLoc() << "\n";
      }
    }
  });

  // Solve the optimal branching problem for each candidate root, or use the
  // provided one.
  Value bestRoot = pattern.getRewriter().getRoot();
  OptimalBranching::EdgeList bestEdges;
  if (!bestRoot) {
    unsigned bestCost = 0;
    LLVM_DEBUG(llvm::dbgs() << "Candidate roots:\n");
    for (Value root : roots) {
      OptimalBranching solver(graph, root);
      unsigned cost = solver.solve();
      LLVM_DEBUG(llvm::dbgs() << "  * " << root << ": " << cost << "\n");
      if (!bestRoot || bestCost > cost) {
        bestCost = cost;
        bestRoot = root;
        bestEdges = solver.preOrderTraversal(roots);
      }
    }
  } else {
    OptimalBranching solver(graph, bestRoot);
    solver.solve();
    bestEdges = solver.preOrderTraversal(roots);
  }

  // Print the best solution.
  LLVM_DEBUG({
    llvm::dbgs() << "Best tree:\n";
    for (const std::pair<Value, Value> &edge : bestEdges) {
      llvm::dbgs() << "  * " << edge.first;
      if (edge.second)
        llvm::dbgs() << " <- " << edge.second;
      llvm::dbgs() << "\n";
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Calling key getTreePredicates:\n");
  LLVM_DEBUG(llvm::dbgs() << "  * Value: " << bestRoot << "\n");

  // The best root is the starting point for the traversal. Get the tree
  // predicates for the DAG rooted at bestRoot.
  getTreePredicates(predList, bestRoot, builder, valueToPosition,
                    builder.getRoot());

  // Traverse the selected optimal branching. For all edges in order, traverse
  // up starting from the connector, until the candidate root is reached, and
  // call getTreePredicates at every node along the way.
  for (const auto &it : llvm::enumerate(bestEdges)) {
    Value target = it.value().first;
    Value source = it.value().second;

    // Check if we already visited the target root. This happens in two cases:
    // 1) the initial root (bestRoot);
    // 2) a root that is dominated by (contained in the subtree rooted at) an
    //    already visited root.
    if (valueToPosition.count(target))
      continue;

    // Determine the connector.
    Value connector = graph[target][source].connector;
    assert(connector && "invalid edge");
    LLVM_DEBUG(llvm::dbgs() << "  * Connector: " << connector.getLoc() << "\n");
    DenseMap<Value, OpIndex> parentMap = parentMaps.lookup(target);
    Position *pos = valueToPosition.lookup(connector);
    assert(pos && "connector has not been traversed yet");

    // Traverse from the connector upwards towards the target root.
    for (Value value = connector; value != target;) {
      OpIndex opIndex = parentMap.lookup(value);
      assert(opIndex.parent && "missing parent");
      visitUpward(predList, opIndex, builder, valueToPosition, pos, it.index());
      value = opIndex.parent;
    }
  }

  getNonTreePredicates(pattern, predList, builder, valueToPosition);

  return bestRoot;
}

//===----------------------------------------------------------------------===//
// Pattern Predicate Tree Merging
//===----------------------------------------------------------------------===//

namespace {

/// This class represents a specific predicate applied to a position, and
/// provides hashing and ordering operators. This class allows for computing a
/// frequence sum and ordering predicates based on a cost model.
struct OrderedPredicate {
  OrderedPredicate(const std::pair<Position *, Qualifier *> &ip)
      : position(ip.first), question(ip.second) {}
  OrderedPredicate(const PositionalPredicate &ip)
      : position(ip.position), question(ip.question) {}

  /// The position this predicate is applied to.
  Position *position;

  /// The question that is applied by this predicate onto the position.
  Qualifier *question;

  /// The first and second order benefit sums.
  /// The primary sum is the number of occurrences of this predicate among all
  /// of the patterns.
  unsigned primary = 0;
  /// The secondary sum is a squared summation of the primary sum of all of the
  /// predicates within each pattern that contains this predicate. This allows
  /// for favoring predicates that are more commonly shared within a pattern, as
  /// opposed to those shared across patterns.
  unsigned secondary = 0;

  /// The tie breaking ID, used to preserve a deterministic (insertion) order
  /// among all the predicates with the same priority, depth, and position /
  /// predicate dependency.
  unsigned id = 0;

  /// A map between a pattern operation and the answer to the predicate question
  /// within that pattern.
  DenseMap<Operation *, Qualifier *> patternToAnswer;

  /// Returns true if this predicate is ordered before `rhs`, based on the cost
  /// model.
  bool operator<(const OrderedPredicate &rhs) const {
    // Sort by:
    // * higher first and secondary order sums
    // * lower depth
    // * lower position dependency
    // * lower predicate dependency
    // * lower tie breaking ID
    auto *rhsPos = rhs.position;
    return std::make_tuple(primary, secondary, rhsPos->getOperationDepth(),
                           rhsPos->getKind(), rhs.question->getKind(), rhs.id) >
           std::make_tuple(rhs.primary, rhs.secondary,
                           position->getOperationDepth(), position->getKind(),
                           question->getKind(), id);
  }
};

/// A DenseMapInfo for OrderedPredicate based solely on the position and
/// question.
struct OrderedPredicateDenseInfo {
  using Base = DenseMapInfo<std::pair<Position *, Qualifier *>>;

  static OrderedPredicate getEmptyKey() { return Base::getEmptyKey(); }
  static OrderedPredicate getTombstoneKey() { return Base::getTombstoneKey(); }
  static bool isEqual(const OrderedPredicate &lhs,
                      const OrderedPredicate &rhs) {
    return lhs.position == rhs.position && lhs.question == rhs.question;
  }
  static unsigned getHashValue(const OrderedPredicate &p) {
    return llvm::hash_combine(p.position, p.question);
  }
};

/// This class wraps a set of ordered predicates that are used within a specific
/// pattern operation.
struct OrderedPredicateList {
  OrderedPredicateList(pdl::PatternOp pattern, Value root)
      : pattern(pattern), root(root) {}

  pdl::PatternOp pattern;
  Value root;
  DenseSet<OrderedPredicate *> predicates;
};
} // namespace

/// Returns true if the given matcher refers to the same predicate as the given
/// ordered predicate. This means that the position and questions of the two
/// match.
static bool isSamePredicate(MatcherNode *node, OrderedPredicate *predicate) {
  return node->getPosition() == predicate->position &&
         node->getQuestion() == predicate->question;
}

/// Get or insert a child matcher for the given parent switch node, given a
/// predicate and parent pattern.
std::unique_ptr<MatcherNode> &getOrCreateChild(SwitchNode *node,
                                               OrderedPredicate *predicate,
                                               pdl::PatternOp pattern) {
  assert(isSamePredicate(node, predicate) &&
         "expected matcher to equal the given predicate");

  auto it = predicate->patternToAnswer.find(pattern);
  assert(it != predicate->patternToAnswer.end() &&
         "expected pattern to exist in predicate");
  return node->getChildren()[it->second];
}

/// Build the matcher CFG by "pushing" patterns through by sorted predicate
/// order. A pattern will traverse as far as possible using common predicates
/// and then either diverge from the CFG or reach the end of a branch and start
/// creating new nodes.
static void propagatePattern(std::unique_ptr<MatcherNode> &node,
                             OrderedPredicateList &list,
                             std::vector<OrderedPredicate *>::iterator current,
                             std::vector<OrderedPredicate *>::iterator end) {
  if (current == end) {
    // We've hit the end of a pattern, so create a successful result node.
    node =
        std::make_unique<SuccessNode>(list.pattern, list.root, std::move(node));

    // If the pattern doesn't contain this predicate, ignore it.
  } else if (!list.predicates.contains(*current)) {
    propagatePattern(node, list, std::next(current), end);

    // If the current matcher node is invalid, create a new one for this
    // position and continue propagation.
  } else if (!node) {
    // Create a new node at this position and continue
    node = std::make_unique<SwitchNode>((*current)->position,
                                        (*current)->question);
    propagatePattern(
        getOrCreateChild(cast<SwitchNode>(&*node), *current, list.pattern),
        list, std::next(current), end);

    // If the matcher has already been created, and it is for this predicate we
    // continue propagation to the child.
  } else if (isSamePredicate(node.get(), *current)) {
    propagatePattern(
        getOrCreateChild(cast<SwitchNode>(&*node), *current, list.pattern),
        list, std::next(current), end);

    // If the matcher doesn't match the current predicate, insert a branch as
    // the common set of matchers has diverged.
  } else {
    propagatePattern(node->getFailureNode(), list, current, end);
  }
}

/// Fold any switch nodes nested under `node` to boolean nodes when possible.
/// `node` is updated in-place if it is a switch.
static void foldSwitchToBool(std::unique_ptr<MatcherNode> &node) {
  if (!node)
    return;

  if (SwitchNode *switchNode = dyn_cast<SwitchNode>(&*node)) {
    SwitchNode::ChildMapT &children = switchNode->getChildren();
    for (auto &it : children)
      foldSwitchToBool(it.second);

    // If the node only contains one child, collapse it into a boolean predicate
    // node.
    if (children.size() == 1) {
      auto *childIt = children.begin();
      node = std::make_unique<BoolNode>(
          node->getPosition(), node->getQuestion(), childIt->first,
          std::move(childIt->second), std::move(node->getFailureNode()));
    }
  } else if (BoolNode *boolNode = dyn_cast<BoolNode>(&*node)) {
    foldSwitchToBool(boolNode->getSuccessNode());
  }

  foldSwitchToBool(node->getFailureNode());
}

/// Insert an exit node at the end of the failure path of the `root`.
static void insertExitNode(std::unique_ptr<MatcherNode> *root) {
  while (*root)
    root = &(*root)->getFailureNode();
  *root = std::make_unique<ExitNode>();
}

/// Sorts the range begin/end with the partial order given by cmp.
template <typename Iterator, typename Compare>
static void stableTopologicalSort(Iterator begin, Iterator end, Compare cmp) {
  while (begin != end) {
    // Cannot compute sortBeforeOthers in the predicate of stable_partition
    // because stable_partition will not keep the [begin, end) range intact
    // while it runs.
    llvm::SmallPtrSet<typename Iterator::value_type, 16> sortBeforeOthers;
    for (auto i = begin; i != end; ++i) {
      if (std::none_of(begin, end, [&](auto const &b) { return cmp(b, *i); }))
        sortBeforeOthers.insert(*i);
    }

    auto const next = std::stable_partition(begin, end, [&](auto const &a) {
      return sortBeforeOthers.contains(a);
    });
    assert(next != begin && "not a partial ordering");
    begin = next;
  }
}

/// Returns true if 'b' depends on a result of 'a'.
static bool dependsOn(OrderedPredicate *a, OrderedPredicate *b) {
  auto *cqa = dyn_cast<ConstraintQuestion>(a->question);
  if (!cqa)
    return false;

  auto positionDependsOnA = [&](Position *p) {
    auto *cp = dyn_cast<ConstraintPosition>(p);
    return cp && cp->getQuestion() == cqa;
  };

  if (auto *cqb = dyn_cast<ConstraintQuestion>(b->question)) {
    // Does any argument of b use a?
    return llvm::any_of(cqb->getArgs(), positionDependsOnA);
  }
  if (auto *equalTo = dyn_cast<EqualToQuestion>(b->question)) {
    return positionDependsOnA(b->position) ||
           positionDependsOnA(equalTo->getValue());
  }
  return positionDependsOnA(b->position);
}

/// Given a module containing PDL pattern operations, generate a matcher tree
/// using the patterns within the given module and return the root matcher node.
std::unique_ptr<MatcherNode>
MatcherNode::generateMatcherTree(ModuleOp module, PredicateBuilder &builder,
                                 DenseMap<Value, Position *> &valueToPosition) {
  // The set of predicates contained within the pattern operations of the
  // module.
  struct PatternPredicates {
    PatternPredicates(pdl::PatternOp pattern, Value root,
                      std::vector<PositionalPredicate> predicates)
        : pattern(pattern), root(root), predicates(std::move(predicates)) {}

    /// A pattern.
    pdl::PatternOp pattern;

    /// A root of the pattern chosen among the candidate roots in pdl.rewrite.
    Value root;

    /// The extracted predicates for this pattern and root.
    std::vector<PositionalPredicate> predicates;
  };

  SmallVector<PatternPredicates, 16> patternsAndPredicates;
  for (pdl::PatternOp pattern : module.getOps<pdl::PatternOp>()) {
    std::vector<PositionalPredicate> predicateList;
    Value root =
        buildPredicateList(pattern, builder, predicateList, valueToPosition);
    patternsAndPredicates.emplace_back(pattern, root, std::move(predicateList));
  }

  // Associate a pattern result with each unique predicate.
  DenseSet<OrderedPredicate, OrderedPredicateDenseInfo> uniqued;
  for (auto &patternAndPredList : patternsAndPredicates) {
    for (auto &predicate : patternAndPredList.predicates) {
      auto it = uniqued.insert(predicate);
      it.first->patternToAnswer.try_emplace(patternAndPredList.pattern,
                                            predicate.answer);
      // Mark the insertion order (0-based indexing).
      if (it.second)
        it.first->id = uniqued.size() - 1;
    }
  }

  // Associate each pattern to a set of its ordered predicates for later lookup.
  std::vector<OrderedPredicateList> lists;
  lists.reserve(patternsAndPredicates.size());
  for (auto &patternAndPredList : patternsAndPredicates) {
    OrderedPredicateList list(patternAndPredList.pattern,
                              patternAndPredList.root);
    for (auto &predicate : patternAndPredList.predicates) {
      OrderedPredicate *orderedPredicate = &*uniqued.find(predicate);
      list.predicates.insert(orderedPredicate);

      // Increment the primary sum for each reference to a particular predicate.
      ++orderedPredicate->primary;
    }
    lists.push_back(std::move(list));
  }

  // For a particular pattern, get the total primary sum and add it to the
  // secondary sum of each predicate. Square the primary sums to emphasize
  // shared predicates within rather than across patterns.
  for (auto &list : lists) {
    unsigned total = 0;
    for (auto *predicate : list.predicates)
      total += predicate->primary * predicate->primary;
    for (auto *predicate : list.predicates)
      predicate->secondary += total;
  }

  // Sort the set of predicates now that the cost primary and secondary sums
  // have been computed.
  std::vector<OrderedPredicate *> ordered;
  ordered.reserve(uniqued.size());
  for (auto &ip : uniqued)
    ordered.push_back(&ip);
  llvm::sort(ordered, [](OrderedPredicate *lhs, OrderedPredicate *rhs) {
    return *lhs < *rhs;
  });

  // Mostly keep the now established order, but also ensure that
  // ConstraintQuestions come after the results they use.
  stableTopologicalSort(ordered.begin(), ordered.end(), dependsOn);

  // Build the matchers for each of the pattern predicate lists.
  std::unique_ptr<MatcherNode> root;
  for (OrderedPredicateList &list : lists)
    propagatePattern(root, list, ordered.begin(), ordered.end());

  // Collapse the graph and insert the exit node.
  foldSwitchToBool(root);
  insertExitNode(&root);
  return root;
}

//===----------------------------------------------------------------------===//
// MatcherNode
//===----------------------------------------------------------------------===//

MatcherNode::MatcherNode(TypeID matcherTypeID, Position *p, Qualifier *q,
                         std::unique_ptr<MatcherNode> failureNode)
    : position(p), question(q), failureNode(std::move(failureNode)),
      matcherTypeID(matcherTypeID) {}

//===----------------------------------------------------------------------===//
// BoolNode
//===----------------------------------------------------------------------===//

BoolNode::BoolNode(Position *position, Qualifier *question, Qualifier *answer,
                   std::unique_ptr<MatcherNode> successNode,
                   std::unique_ptr<MatcherNode> failureNode)
    : MatcherNode(TypeID::get<BoolNode>(), position, question,
                  std::move(failureNode)),
      answer(answer), successNode(std::move(successNode)) {}

//===----------------------------------------------------------------------===//
// SuccessNode
//===----------------------------------------------------------------------===//

SuccessNode::SuccessNode(pdl::PatternOp pattern, Value root,
                         std::unique_ptr<MatcherNode> failureNode)
    : MatcherNode(TypeID::get<SuccessNode>(), /*position=*/nullptr,
                  /*question=*/nullptr, std::move(failureNode)),
      pattern(pattern), root(root) {}

//===----------------------------------------------------------------------===//
// SwitchNode
//===----------------------------------------------------------------------===//

SwitchNode::SwitchNode(Position *position, Qualifier *question)
    : MatcherNode(TypeID::get<SwitchNode>(), position, question) {}
