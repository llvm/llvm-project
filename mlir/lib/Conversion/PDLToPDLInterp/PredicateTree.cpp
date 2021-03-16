//===- PredicateTree.cpp - Predicate tree merging -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PredicateTree.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/TypeSwitch.h"

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
                        [](Type type) { return !type.isa<pdl::RangeType>(); });
}

static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              AttributePosition *pos) {
  assert(val.getType().isa<pdl::AttributeType>() && "expected attribute type");
  pdl::AttributeOp attr = cast<pdl::AttributeOp>(val.getDefiningOp());
  predList.emplace_back(pos, builder.getIsNotNull());

  // If the attribute has a type or value, add a constraint.
  if (Value type = attr.type())
    getTreePredicates(predList, type, builder, inputs, builder.getType(pos));
  else if (Attribute value = attr.valueAttr())
    predList.emplace_back(pos, builder.getAttributeConstraint(value));
}

/// Collect all of the predicates for the given operand position.
static void getOperandTreePredicates(std::vector<PositionalPredicate> &predList,
                                     Value val, PredicateBuilder &builder,
                                     DenseMap<Value, Position *> &inputs,
                                     Position *pos) {
  Type valueType = val.getType();
  bool isVariadic = valueType.isa<pdl::RangeType>();

  // If this is a typed operand, add a type constraint.
  TypeSwitch<Operation *>(val.getDefiningOp())
      .Case<pdl::OperandOp, pdl::OperandsOp>([&](auto op) {
        // Prevent traversal into a null value if the operand has a proper
        // index.
        if (std::is_same<pdl::OperandOp, decltype(op)>::value ||
            cast<OperandGroupPosition>(pos)->getOperandGroupNumber())
          predList.emplace_back(pos, builder.getIsNotNull());

        if (Value type = op.type())
          getTreePredicates(predList, type, builder, inputs,
                            builder.getType(pos));
      })
      .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto op) {
        Optional<unsigned> index = op.index();

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
        getTreePredicates(predList, op.parent(), builder, inputs, parentPos);
      });
}

static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              OperationPosition *pos) {
  assert(val.getType().isa<pdl::OperationType>() && "expected operation");
  pdl::OperationOp op = cast<pdl::OperationOp>(val.getDefiningOp());
  OperationPosition *opPos = cast<OperationPosition>(pos);

  // Ensure getDefiningOp returns a non-null operation.
  if (!opPos->isRoot())
    predList.emplace_back(pos, builder.getIsNotNull());

  // Check that this is the correct root operation.
  if (Optional<StringRef> opName = op.name())
    predList.emplace_back(pos, builder.getOperationName(*opName));

  // Check that the operation has the proper number of operands. If there are
  // any variable length operands, we check a minimum instead of an exact count.
  OperandRange operands = op.operands();
  unsigned minOperands = getNumNonRangeValues(operands);
  if (minOperands != operands.size()) {
    if (minOperands)
      predList.emplace_back(pos, builder.getOperandCountAtLeast(minOperands));
  } else {
    predList.emplace_back(pos, builder.getOperandCount(minOperands));
  }

  // Check that the operation has the proper number of results. If there are
  // any variable length results, we check a minimum instead of an exact count.
  OperandRange types = op.types();
  unsigned minResults = getNumNonRangeValues(types);
  if (minResults == types.size())
    predList.emplace_back(pos, builder.getResultCount(types.size()));
  else if (minResults)
    predList.emplace_back(pos, builder.getResultCountAtLeast(minResults));

  // Recurse into any attributes, operands, or results.
  for (auto it : llvm::zip(op.attributeNames(), op.attributes())) {
    getTreePredicates(
        predList, std::get<1>(it), builder, inputs,
        builder.getAttribute(opPos,
                             std::get<0>(it).cast<StringAttr>().getValue()));
  }

  // Process the operands and results of the operation. For all values up to
  // the first variable length value, we use the concrete operand/result
  // number. After that, we use the "group" given that we can't know the
  // concrete indices until runtime. If there is only one variadic operand
  // group, we treat it as all of the operands/results of the operation.
  /// Operands.
  if (operands.size() == 1 && operands[0].getType().isa<pdl::RangeType>()) {
    getTreePredicates(predList, operands.front(), builder, inputs,
                      builder.getAllOperands(opPos));
  } else {
    bool foundVariableLength = false;
    for (auto operandIt : llvm::enumerate(operands)) {
      bool isVariadic = operandIt.value().getType().isa<pdl::RangeType>();
      foundVariableLength |= isVariadic;

      Position *pos =
          foundVariableLength
              ? builder.getOperandGroup(opPos, operandIt.index(), isVariadic)
              : builder.getOperand(opPos, operandIt.index());
      getTreePredicates(predList, operandIt.value(), builder, inputs, pos);
    }
  }
  /// Results.
  if (types.size() == 1 && types[0].getType().isa<pdl::RangeType>()) {
    getTreePredicates(predList, types.front(), builder, inputs,
                      builder.getType(builder.getAllResults(opPos)));
  } else {
    bool foundVariableLength = false;
    for (auto &resultIt : llvm::enumerate(types)) {
      bool isVariadic = resultIt.value().getType().isa<pdl::RangeType>();
      foundVariableLength |= isVariadic;

      auto *resultPos =
          foundVariableLength
              ? builder.getResultGroup(pos, resultIt.index(), isVariadic)
              : builder.getResult(pos, resultIt.index());
      predList.emplace_back(resultPos, builder.getIsNotNull());
      getTreePredicates(predList, resultIt.value(), builder, inputs,
                        builder.getType(resultPos));
    }
  }
}

static void getTreePredicates(std::vector<PositionalPredicate> &predList,
                              Value val, PredicateBuilder &builder,
                              DenseMap<Value, Position *> &inputs,
                              TypePosition *pos) {
  // Check for a constraint on a constant type.
  if (pdl::TypeOp typeOp = val.getDefiningOp<pdl::TypeOp>()) {
    if (Attribute type = typeOp.typeAttr())
      predList.emplace_back(pos, builder.getTypeConstraint(type));
  } else if (pdl::TypesOp typeOp = val.getDefiningOp<pdl::TypesOp>()) {
    if (Attribute typeAttr = typeOp.typesAttr())
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

/// Collect all of the predicates related to constraints within the given
/// pattern operation.
static void getConstraintPredicates(pdl::ApplyNativeConstraintOp op,
                                    std::vector<PositionalPredicate> &predList,
                                    PredicateBuilder &builder,
                                    DenseMap<Value, Position *> &inputs) {
  OperandRange arguments = op.args();
  ArrayAttr parameters = op.constParamsAttr();

  std::vector<Position *> allPositions;
  allPositions.reserve(arguments.size());
  for (Value arg : arguments)
    allPositions.push_back(inputs.lookup(arg));

  // Push the constraint to the furthest position.
  Position *pos = *std::max_element(allPositions.begin(), allPositions.end(),
                                    comparePosDepth);
  PredicateBuilder::Predicate pred =
      builder.getConstraint(op.name(), std::move(allPositions), parameters);
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
  auto *parentPos = cast<OperationPosition>(inputs.lookup(op.parent()));
  resultPos = builder.getResult(parentPos, op.index());
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
  auto *parentPos = cast<OperationPosition>(inputs.lookup(op.parent()));
  bool isVariadic = op.getType().isa<pdl::RangeType>();
  Optional<unsigned> index = op.index();
  resultPos = builder.getResultGroup(parentPos, index, isVariadic);
  if (index)
    predList.emplace_back(resultPos, builder.getIsNotNull());
}

/// Collect all of the predicates that cannot be determined via walking the
/// tree.
static void getNonTreePredicates(pdl::PatternOp pattern,
                                 std::vector<PositionalPredicate> &predList,
                                 PredicateBuilder &builder,
                                 DenseMap<Value, Position *> &inputs) {
  for (Operation &op : pattern.body().getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case<pdl::ApplyNativeConstraintOp>([&](auto constraintOp) {
          getConstraintPredicates(constraintOp, predList, builder, inputs);
        })
        .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto resultOp) {
          getResultPredicates(resultOp, predList, builder, inputs);
        });
  }
}

/// Given a pattern operation, build the set of matcher predicates necessary to
/// match this pattern.
static void buildPredicateList(pdl::PatternOp pattern,
                               PredicateBuilder &builder,
                               std::vector<PositionalPredicate> &predList,
                               DenseMap<Value, Position *> &valueToPosition) {
  getTreePredicates(predList, pattern.getRewriter().root(), builder,
                    valueToPosition, builder.getRoot());
  getNonTreePredicates(pattern, predList, builder, valueToPosition);
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
    auto *rhsPos = rhs.position;
    return std::make_tuple(primary, secondary, rhsPos->getOperationDepth(),
                           rhsPos->getKind(), rhs.question->getKind()) >
           std::make_tuple(rhs.primary, rhs.secondary,
                           position->getOperationDepth(), position->getKind(),
                           question->getKind());
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
  OrderedPredicateList(pdl::PatternOp pattern) : pattern(pattern) {}

  pdl::PatternOp pattern;
  DenseSet<OrderedPredicate *> predicates;
};
} // end anonymous namespace

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
  return node->getChildren().insert({it->second, nullptr}).first->second;
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
    node = std::make_unique<SuccessNode>(list.pattern, std::move(node));

    // If the pattern doesn't contain this predicate, ignore it.
  } else if (list.predicates.find(*current) == list.predicates.end()) {
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
      auto childIt = children.begin();
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

/// Given a module containing PDL pattern operations, generate a matcher tree
/// using the patterns within the given module and return the root matcher node.
std::unique_ptr<MatcherNode>
MatcherNode::generateMatcherTree(ModuleOp module, PredicateBuilder &builder,
                                 DenseMap<Value, Position *> &valueToPosition) {
  // Collect the set of predicates contained within the pattern operations of
  // the module.
  SmallVector<std::pair<pdl::PatternOp, std::vector<PositionalPredicate>>, 16>
      patternsAndPredicates;
  for (pdl::PatternOp pattern : module.getOps<pdl::PatternOp>()) {
    std::vector<PositionalPredicate> predicateList;
    buildPredicateList(pattern, builder, predicateList, valueToPosition);
    patternsAndPredicates.emplace_back(pattern, std::move(predicateList));
  }

  // Associate a pattern result with each unique predicate.
  DenseSet<OrderedPredicate, OrderedPredicateDenseInfo> uniqued;
  for (auto &patternAndPredList : patternsAndPredicates) {
    for (auto &predicate : patternAndPredList.second) {
      auto it = uniqued.insert(predicate);
      it.first->patternToAnswer.try_emplace(patternAndPredList.first,
                                            predicate.answer);
    }
  }

  // Associate each pattern to a set of its ordered predicates for later lookup.
  std::vector<OrderedPredicateList> lists;
  lists.reserve(patternsAndPredicates.size());
  for (auto &patternAndPredList : patternsAndPredicates) {
    OrderedPredicateList list(patternAndPredList.first);
    for (auto &predicate : patternAndPredList.second) {
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
  std::stable_sort(
      ordered.begin(), ordered.end(),
      [](OrderedPredicate *lhs, OrderedPredicate *rhs) { return *lhs < *rhs; });

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

SuccessNode::SuccessNode(pdl::PatternOp pattern,
                         std::unique_ptr<MatcherNode> failureNode)
    : MatcherNode(TypeID::get<SuccessNode>(), /*position=*/nullptr,
                  /*question=*/nullptr, std::move(failureNode)),
      pattern(pattern) {}

//===----------------------------------------------------------------------===//
// SwitchNode
//===----------------------------------------------------------------------===//

SwitchNode::SwitchNode(Position *position, Qualifier *question)
    : MatcherNode(TypeID::get<SwitchNode>(), position, question) {}
