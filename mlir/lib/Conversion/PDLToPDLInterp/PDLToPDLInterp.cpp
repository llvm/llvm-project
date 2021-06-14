//===- PDLToPDLInterp.cpp - Lower a PDL module to the interpreter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "../PassDetail.h"
#include "PredicateTree.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

//===----------------------------------------------------------------------===//
// PatternLowering
//===----------------------------------------------------------------------===//

namespace {
/// This class generators operations within the PDL Interpreter dialect from a
/// given module containing PDL pattern operations.
struct PatternLowering {
public:
  PatternLowering(FuncOp matcherFunc, ModuleOp rewriterModule);

  /// Generate code for matching and rewriting based on the pattern operations
  /// within the module.
  void lower(ModuleOp module);

private:
  using ValueMap = llvm::ScopedHashTable<Position *, Value>;
  using ValueMapScope = llvm::ScopedHashTableScope<Position *, Value>;

  /// Generate interpreter operations for the tree rooted at the given matcher
  /// node.
  Block *generateMatcher(MatcherNode &node);

  /// Get or create an access to the provided positional value within the
  /// current block.
  Value getValueAt(Block *cur, Position *pos);

  /// Create an interpreter predicate operation, branching to the provided true
  /// and false destinations.
  void generatePredicate(Block *currentBlock, Qualifier *question,
                         Qualifier *answer, Value val, Block *trueDest,
                         Block *falseDest);

  /// Create an interpreter switch predicate operation, with a provided default
  /// and several case destinations.
  void generateSwitch(SwitchNode *switchNode, Block *currentBlock,
                      Qualifier *question, Value val, Block *defaultDest);

  /// Create the interpreter operations to record a successful pattern match.
  void generateRecordMatch(Block *currentBlock, Block *nextBlock,
                           pdl::PatternOp pattern);

  /// Generate a rewriter function for the given pattern operation, and returns
  /// a reference to that function.
  SymbolRefAttr generateRewriter(pdl::PatternOp pattern,
                                 SmallVectorImpl<Position *> &usedMatchValues);

  /// Generate the rewriter code for the given operation.
  void generateRewriter(pdl::ApplyNativeRewriteOp rewriteOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::AttributeOp attrOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::EraseOp eraseOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::OperationOp operationOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::ReplaceOp replaceOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::ResultOp resultOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::ResultsOp resultOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::TypeOp typeOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);
  void generateRewriter(pdl::TypesOp typeOp,
                        DenseMap<Value, Value> &rewriteValues,
                        function_ref<Value(Value)> mapRewriteValue);

  /// Generate the values used for resolving the result types of an operation
  /// created within a dag rewriter region.
  void generateOperationResultTypeRewriter(
      pdl::OperationOp op, SmallVectorImpl<Value> &types,
      DenseMap<Value, Value> &rewriteValues,
      function_ref<Value(Value)> mapRewriteValue);

  /// A builder to use when generating interpreter operations.
  OpBuilder builder;

  /// The matcher function used for all match related logic within PDL patterns.
  FuncOp matcherFunc;

  /// The rewriter module containing the all rewrite related logic within PDL
  /// patterns.
  ModuleOp rewriterModule;

  /// The symbol table of the rewriter module used for insertion.
  SymbolTable rewriterSymbolTable;

  /// A scoped map connecting a position with the corresponding interpreter
  /// value.
  ValueMap values;

  /// A stack of blocks used as the failure destination for matcher nodes that
  /// don't have an explicit failure path.
  SmallVector<Block *, 8> failureBlockStack;

  /// A mapping between values defined in a pattern match, and the corresponding
  /// positional value.
  DenseMap<Value, Position *> valueToPosition;

  /// The set of operation values whose whose location will be used for newly
  /// generated operations.
  SetVector<Value> locOps;
};
} // end anonymous namespace

PatternLowering::PatternLowering(FuncOp matcherFunc, ModuleOp rewriterModule)
    : builder(matcherFunc.getContext()), matcherFunc(matcherFunc),
      rewriterModule(rewriterModule), rewriterSymbolTable(rewriterModule) {}

void PatternLowering::lower(ModuleOp module) {
  PredicateUniquer predicateUniquer;
  PredicateBuilder predicateBuilder(predicateUniquer, module.getContext());

  // Define top-level scope for the arguments to the matcher function.
  ValueMapScope topLevelValueScope(values);

  // Insert the root operation, i.e. argument to the matcher, at the root
  // position.
  Block *matcherEntryBlock = matcherFunc.addEntryBlock();
  values.insert(predicateBuilder.getRoot(), matcherEntryBlock->getArgument(0));

  // Generate a root matcher node from the provided PDL module.
  std::unique_ptr<MatcherNode> root = MatcherNode::generateMatcherTree(
      module, predicateBuilder, valueToPosition);
  Block *firstMatcherBlock = generateMatcher(*root);

  // After generation, merged the first matched block into the entry.
  matcherEntryBlock->getOperations().splice(matcherEntryBlock->end(),
                                            firstMatcherBlock->getOperations());
  firstMatcherBlock->erase();
}

Block *PatternLowering::generateMatcher(MatcherNode &node) {
  // Push a new scope for the values used by this matcher.
  Block *block = matcherFunc.addBlock();
  ValueMapScope scope(values);

  // If this is the return node, simply insert the corresponding interpreter
  // finalize.
  if (isa<ExitNode>(node)) {
    builder.setInsertionPointToEnd(block);
    builder.create<pdl_interp::FinalizeOp>(matcherFunc.getLoc());
    return block;
  }

  // If this node contains a position, get the corresponding value for this
  // block.
  Position *position = node.getPosition();
  Value val = position ? getValueAt(block, position) : Value();

  // Get the next block in the match sequence.
  std::unique_ptr<MatcherNode> &failureNode = node.getFailureNode();
  Block *nextBlock;
  if (failureNode) {
    nextBlock = generateMatcher(*failureNode);
    failureBlockStack.push_back(nextBlock);
  } else {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    nextBlock = failureBlockStack.back();
  }

  // If this value corresponds to an operation, record that we are going to use
  // its location as part of a fused location.
  bool isOperationValue = val && val.getType().isa<pdl::OperationType>();
  if (isOperationValue)
    locOps.insert(val);

  // Generate code for a boolean predicate node.
  if (auto *boolNode = dyn_cast<BoolNode>(&node)) {
    auto *child = generateMatcher(*boolNode->getSuccessNode());
    generatePredicate(block, node.getQuestion(), boolNode->getAnswer(), val,
                      child, nextBlock);

    // Generate code for a switch node.
  } else if (auto *switchNode = dyn_cast<SwitchNode>(&node)) {
    generateSwitch(switchNode, block, node.getQuestion(), val, nextBlock);

    // Generate code for a success node.
  } else if (auto *successNode = dyn_cast<SuccessNode>(&node)) {
    generateRecordMatch(block, nextBlock, successNode->getPattern());
  }

  if (failureNode)
    failureBlockStack.pop_back();
  if (isOperationValue)
    locOps.remove(val);
  return block;
}

Value PatternLowering::getValueAt(Block *cur, Position *pos) {
  if (Value val = values.lookup(pos))
    return val;

  // Get the value for the parent position.
  Value parentVal = getValueAt(cur, pos->getParent());

  // TODO: Use a location from the position.
  Location loc = parentVal.getLoc();
  builder.setInsertionPointToEnd(cur);
  Value value;
  switch (pos->getKind()) {
  case Predicates::OperationPos:
    value = builder.create<pdl_interp::GetDefiningOpOp>(
        loc, builder.getType<pdl::OperationType>(), parentVal);
    break;
  case Predicates::OperandPos: {
    auto *operandPos = cast<OperandPosition>(pos);
    value = builder.create<pdl_interp::GetOperandOp>(
        loc, builder.getType<pdl::ValueType>(), parentVal,
        operandPos->getOperandNumber());
    break;
  }
  case Predicates::OperandGroupPos: {
    auto *operandPos = cast<OperandGroupPosition>(pos);
    Type valueTy = builder.getType<pdl::ValueType>();
    value = builder.create<pdl_interp::GetOperandsOp>(
        loc, operandPos->isVariadic() ? pdl::RangeType::get(valueTy) : valueTy,
        parentVal, operandPos->getOperandGroupNumber());
    break;
  }
  case Predicates::AttributePos: {
    auto *attrPos = cast<AttributePosition>(pos);
    value = builder.create<pdl_interp::GetAttributeOp>(
        loc, builder.getType<pdl::AttributeType>(), parentVal,
        attrPos->getName().strref());
    break;
  }
  case Predicates::TypePos: {
    if (parentVal.getType().isa<pdl::AttributeType>())
      value = builder.create<pdl_interp::GetAttributeTypeOp>(loc, parentVal);
    else
      value = builder.create<pdl_interp::GetValueTypeOp>(loc, parentVal);
    break;
  }
  case Predicates::ResultPos: {
    auto *resPos = cast<ResultPosition>(pos);
    value = builder.create<pdl_interp::GetResultOp>(
        loc, builder.getType<pdl::ValueType>(), parentVal,
        resPos->getResultNumber());
    break;
  }
  case Predicates::ResultGroupPos: {
    auto *resPos = cast<ResultGroupPosition>(pos);
    Type valueTy = builder.getType<pdl::ValueType>();
    value = builder.create<pdl_interp::GetResultsOp>(
        loc, resPos->isVariadic() ? pdl::RangeType::get(valueTy) : valueTy,
        parentVal, resPos->getResultGroupNumber());
    break;
  }
  default:
    llvm_unreachable("Generating unknown Position getter");
    break;
  }
  values.insert(pos, value);
  return value;
}

void PatternLowering::generatePredicate(Block *currentBlock,
                                        Qualifier *question, Qualifier *answer,
                                        Value val, Block *trueDest,
                                        Block *falseDest) {
  builder.setInsertionPointToEnd(currentBlock);
  Location loc = val.getLoc();
  Predicates::Kind kind = question->getKind();
  switch (kind) {
  case Predicates::IsNotNullQuestion:
    builder.create<pdl_interp::IsNotNullOp>(loc, val, trueDest, falseDest);
    break;
  case Predicates::OperationNameQuestion: {
    auto *opNameAnswer = cast<OperationNameAnswer>(answer);
    builder.create<pdl_interp::CheckOperationNameOp>(
        loc, val, opNameAnswer->getValue().getStringRef(), trueDest, falseDest);
    break;
  }
  case Predicates::TypeQuestion: {
    auto *ans = cast<TypeAnswer>(answer);
    if (val.getType().isa<pdl::RangeType>())
      builder.create<pdl_interp::CheckTypesOp>(
          loc, val, ans->getValue().cast<ArrayAttr>(), trueDest, falseDest);
    else
      builder.create<pdl_interp::CheckTypeOp>(
          loc, val, ans->getValue().cast<TypeAttr>(), trueDest, falseDest);
    break;
  }
  case Predicates::AttributeQuestion: {
    auto *ans = cast<AttributeAnswer>(answer);
    builder.create<pdl_interp::CheckAttributeOp>(loc, val, ans->getValue(),
                                                 trueDest, falseDest);
    break;
  }
  case Predicates::OperandCountAtLeastQuestion:
  case Predicates::OperandCountQuestion:
    builder.create<pdl_interp::CheckOperandCountOp>(
        loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::OperandCountAtLeastQuestion,
        trueDest, falseDest);
    break;
  case Predicates::ResultCountAtLeastQuestion:
  case Predicates::ResultCountQuestion:
    builder.create<pdl_interp::CheckResultCountOp>(
        loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::ResultCountAtLeastQuestion,
        trueDest, falseDest);
    break;
  case Predicates::EqualToQuestion: {
    auto *equalToQuestion = cast<EqualToQuestion>(question);
    builder.create<pdl_interp::AreEqualOp>(
        loc, val, getValueAt(currentBlock, equalToQuestion->getValue()),
        trueDest, falseDest);
    break;
  }
  case Predicates::ConstraintQuestion: {
    auto *cstQuestion = cast<ConstraintQuestion>(question);
    SmallVector<Value, 2> args;
    for (Position *position : std::get<1>(cstQuestion->getValue()))
      args.push_back(getValueAt(currentBlock, position));
    builder.create<pdl_interp::ApplyConstraintOp>(
        loc, std::get<0>(cstQuestion->getValue()), args,
        std::get<2>(cstQuestion->getValue()).cast<ArrayAttr>(), trueDest,
        falseDest);
    break;
  }
  default:
    llvm_unreachable("Generating unknown Predicate operation");
  }
}

template <typename OpT, typename PredT, typename ValT = typename PredT::KeyTy>
static void createSwitchOp(Value val, Block *defaultDest, OpBuilder &builder,
                           llvm::MapVector<Qualifier *, Block *> &dests) {
  std::vector<ValT> values;
  std::vector<Block *> blocks;
  values.reserve(dests.size());
  blocks.reserve(dests.size());
  for (const auto &it : dests) {
    blocks.push_back(it.second);
    values.push_back(cast<PredT>(it.first)->getValue());
  }
  builder.create<OpT>(val.getLoc(), val, values, defaultDest, blocks);
}

void PatternLowering::generateSwitch(SwitchNode *switchNode,
                                     Block *currentBlock, Qualifier *question,
                                     Value val, Block *defaultDest) {
  // If the switch question is not an exact answer, i.e. for the `at_least`
  // cases, we generate a special block sequence.
  Predicates::Kind kind = question->getKind();
  if (kind == Predicates::OperandCountAtLeastQuestion ||
      kind == Predicates::ResultCountAtLeastQuestion) {
    // Order the children such that the cases are in reverse numerical order.
    SmallVector<unsigned> sortedChildren = llvm::to_vector<16>(
        llvm::seq<unsigned>(0, switchNode->getChildren().size()));
    llvm::sort(sortedChildren, [&](unsigned lhs, unsigned rhs) {
      return cast<UnsignedAnswer>(switchNode->getChild(lhs).first)->getValue() >
             cast<UnsignedAnswer>(switchNode->getChild(rhs).first)->getValue();
    });

    // Build the destination for each child using the next highest child as a
    // a failure destination. This essentially creates the following control
    // flow:
    //
    // if (operand_count < 1)
    //   goto failure
    // if (child1.match())
    //   ...
    //
    // if (operand_count < 2)
    //   goto failure
    // if (child2.match())
    //   ...
    //
    // failure:
    //   ...
    //
    failureBlockStack.push_back(defaultDest);
    for (unsigned idx : sortedChildren) {
      auto &child = switchNode->getChild(idx);
      Block *childBlock = generateMatcher(*child.second);
      Block *predicateBlock = builder.createBlock(childBlock);
      generatePredicate(predicateBlock, question, child.first, val, childBlock,
                        defaultDest);
      failureBlockStack.back() = predicateBlock;
    }
    Block *firstPredicateBlock = failureBlockStack.pop_back_val();
    currentBlock->getOperations().splice(currentBlock->end(),
                                         firstPredicateBlock->getOperations());
    firstPredicateBlock->erase();
    return;
  }

  // Otherwise, generate each of the children and generate an interpreter
  // switch.
  llvm::MapVector<Qualifier *, Block *> children;
  for (auto &it : switchNode->getChildren())
    children.insert({it.first, generateMatcher(*it.second)});
  builder.setInsertionPointToEnd(currentBlock);

  switch (question->getKind()) {
  case Predicates::OperandCountQuestion:
    return createSwitchOp<pdl_interp::SwitchOperandCountOp, UnsignedAnswer,
                          int32_t>(val, defaultDest, builder, children);
  case Predicates::ResultCountQuestion:
    return createSwitchOp<pdl_interp::SwitchResultCountOp, UnsignedAnswer,
                          int32_t>(val, defaultDest, builder, children);
  case Predicates::OperationNameQuestion:
    return createSwitchOp<pdl_interp::SwitchOperationNameOp,
                          OperationNameAnswer>(val, defaultDest, builder,
                                               children);
  case Predicates::TypeQuestion:
    if (val.getType().isa<pdl::RangeType>()) {
      return createSwitchOp<pdl_interp::SwitchTypesOp, TypeAnswer>(
          val, defaultDest, builder, children);
    }
    return createSwitchOp<pdl_interp::SwitchTypeOp, TypeAnswer>(
        val, defaultDest, builder, children);
  case Predicates::AttributeQuestion:
    return createSwitchOp<pdl_interp::SwitchAttributeOp, AttributeAnswer>(
        val, defaultDest, builder, children);
  default:
    llvm_unreachable("Generating unknown switch predicate.");
  }
}

void PatternLowering::generateRecordMatch(Block *currentBlock, Block *nextBlock,
                                          pdl::PatternOp pattern) {
  // Generate a rewriter for the pattern this success node represents, and track
  // any values used from the match region.
  SmallVector<Position *, 8> usedMatchValues;
  SymbolRefAttr rewriterFuncRef = generateRewriter(pattern, usedMatchValues);

  // Process any values used in the rewrite that are defined in the match.
  std::vector<Value> mappedMatchValues;
  mappedMatchValues.reserve(usedMatchValues.size());
  for (Position *position : usedMatchValues)
    mappedMatchValues.push_back(getValueAt(currentBlock, position));

  // Collect the set of operations generated by the rewriter.
  SmallVector<StringRef, 4> generatedOps;
  for (auto op : pattern.getRewriter().body().getOps<pdl::OperationOp>())
    generatedOps.push_back(*op.name());
  ArrayAttr generatedOpsAttr;
  if (!generatedOps.empty())
    generatedOpsAttr = builder.getStrArrayAttr(generatedOps);

  // Grab the root kind if present.
  StringAttr rootKindAttr;
  if (Optional<StringRef> rootKind = pattern.getRootKind())
    rootKindAttr = builder.getStringAttr(*rootKind);

  builder.setInsertionPointToEnd(currentBlock);
  builder.create<pdl_interp::RecordMatchOp>(
      pattern.getLoc(), mappedMatchValues, locOps.getArrayRef(),
      rewriterFuncRef, rootKindAttr, generatedOpsAttr, pattern.benefitAttr(),
      nextBlock);
}

SymbolRefAttr PatternLowering::generateRewriter(
    pdl::PatternOp pattern, SmallVectorImpl<Position *> &usedMatchValues) {
  FuncOp rewriterFunc =
      FuncOp::create(pattern.getLoc(), "pdl_generated_rewriter",
                     builder.getFunctionType(llvm::None, llvm::None));
  rewriterSymbolTable.insert(rewriterFunc);

  // Generate the rewriter function body.
  builder.setInsertionPointToEnd(rewriterFunc.addEntryBlock());

  // Map an input operand of the pattern to a generated interpreter value.
  DenseMap<Value, Value> rewriteValues;
  auto mapRewriteValue = [&](Value oldValue) {
    Value &newValue = rewriteValues[oldValue];
    if (newValue)
      return newValue;

    // Prefer materializing constants directly when possible.
    Operation *oldOp = oldValue.getDefiningOp();
    if (pdl::AttributeOp attrOp = dyn_cast<pdl::AttributeOp>(oldOp)) {
      if (Attribute value = attrOp.valueAttr()) {
        return newValue = builder.create<pdl_interp::CreateAttributeOp>(
                   attrOp.getLoc(), value);
      }
    } else if (pdl::TypeOp typeOp = dyn_cast<pdl::TypeOp>(oldOp)) {
      if (TypeAttr type = typeOp.typeAttr()) {
        return newValue = builder.create<pdl_interp::CreateTypeOp>(
                   typeOp.getLoc(), type);
      }
    } else if (pdl::TypesOp typeOp = dyn_cast<pdl::TypesOp>(oldOp)) {
      if (ArrayAttr type = typeOp.typesAttr()) {
        return newValue = builder.create<pdl_interp::CreateTypesOp>(
                   typeOp.getLoc(), typeOp.getType(), type);
      }
    }

    // Otherwise, add this as an input to the rewriter.
    Position *inputPos = valueToPosition.lookup(oldValue);
    assert(inputPos && "expected value to be a pattern input");
    usedMatchValues.push_back(inputPos);
    return newValue = rewriterFunc.front().addArgument(oldValue.getType());
  };

  // If this is a custom rewriter, simply dispatch to the registered rewrite
  // method.
  pdl::RewriteOp rewriter = pattern.getRewriter();
  if (StringAttr rewriteName = rewriter.nameAttr()) {
    auto mappedArgs = llvm::map_range(rewriter.externalArgs(), mapRewriteValue);
    SmallVector<Value, 4> args(1, mapRewriteValue(rewriter.root()));
    args.append(mappedArgs.begin(), mappedArgs.end());
    builder.create<pdl_interp::ApplyRewriteOp>(
        rewriter.getLoc(), /*resultTypes=*/TypeRange(), rewriteName, args,
        rewriter.externalConstParamsAttr());
  } else {
    // Otherwise this is a dag rewriter defined using PDL operations.
    for (Operation &rewriteOp : *rewriter.getBody()) {
      llvm::TypeSwitch<Operation *>(&rewriteOp)
          .Case<pdl::ApplyNativeRewriteOp, pdl::AttributeOp, pdl::EraseOp,
                pdl::OperationOp, pdl::ReplaceOp, pdl::ResultOp, pdl::ResultsOp,
                pdl::TypeOp, pdl::TypesOp>([&](auto op) {
            this->generateRewriter(op, rewriteValues, mapRewriteValue);
          });
    }
  }

  // Update the signature of the rewrite function.
  rewriterFunc.setType(builder.getFunctionType(
      llvm::to_vector<8>(rewriterFunc.front().getArgumentTypes()),
      /*results=*/llvm::None));

  builder.create<pdl_interp::FinalizeOp>(rewriter.getLoc());
  return builder.getSymbolRefAttr(
      pdl_interp::PDLInterpDialect::getRewriterModuleName(),
      builder.getSymbolRefAttr(rewriterFunc));
}

void PatternLowering::generateRewriter(
    pdl::ApplyNativeRewriteOp rewriteOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 2> arguments;
  for (Value argument : rewriteOp.args())
    arguments.push_back(mapRewriteValue(argument));
  auto interpOp = builder.create<pdl_interp::ApplyRewriteOp>(
      rewriteOp.getLoc(), rewriteOp.getResultTypes(), rewriteOp.nameAttr(),
      arguments, rewriteOp.constParamsAttr());
  for (auto it : llvm::zip(rewriteOp.results(), interpOp.results()))
    rewriteValues[std::get<0>(it)] = std::get<1>(it);
}

void PatternLowering::generateRewriter(
    pdl::AttributeOp attrOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  Value newAttr = builder.create<pdl_interp::CreateAttributeOp>(
      attrOp.getLoc(), attrOp.valueAttr());
  rewriteValues[attrOp] = newAttr;
}

void PatternLowering::generateRewriter(
    pdl::EraseOp eraseOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  builder.create<pdl_interp::EraseOp>(eraseOp.getLoc(),
                                      mapRewriteValue(eraseOp.operation()));
}

void PatternLowering::generateRewriter(
    pdl::OperationOp operationOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 4> operands;
  for (Value operand : operationOp.operands())
    operands.push_back(mapRewriteValue(operand));

  SmallVector<Value, 4> attributes;
  for (Value attr : operationOp.attributes())
    attributes.push_back(mapRewriteValue(attr));

  SmallVector<Value, 2> types;
  generateOperationResultTypeRewriter(operationOp, types, rewriteValues,
                                      mapRewriteValue);

  // Create the new operation.
  Location loc = operationOp.getLoc();
  Value createdOp = builder.create<pdl_interp::CreateOperationOp>(
      loc, *operationOp.name(), types, operands, attributes,
      operationOp.attributeNames());
  rewriteValues[operationOp.op()] = createdOp;

  // Generate accesses for any results that have their types constrained.
  // Handle the case where there is a single range representing all of the
  // result types.
  OperandRange resultTys = operationOp.types();
  if (resultTys.size() == 1 && resultTys[0].getType().isa<pdl::RangeType>()) {
    Value &type = rewriteValues[resultTys[0]];
    if (!type) {
      auto results = builder.create<pdl_interp::GetResultsOp>(loc, createdOp);
      type = builder.create<pdl_interp::GetValueTypeOp>(loc, results);
    }
    return;
  }

  // Otherwise, populate the individual results.
  bool seenVariableLength = false;
  Type valueTy = builder.getType<pdl::ValueType>();
  Type valueRangeTy = pdl::RangeType::get(valueTy);
  for (auto it : llvm::enumerate(resultTys)) {
    Value &type = rewriteValues[it.value()];
    if (type)
      continue;
    bool isVariadic = it.value().getType().isa<pdl::RangeType>();
    seenVariableLength |= isVariadic;

    // After a variable length result has been seen, we need to use result
    // groups because the exact index of the result is not statically known.
    Value resultVal;
    if (seenVariableLength)
      resultVal = builder.create<pdl_interp::GetResultsOp>(
          loc, isVariadic ? valueRangeTy : valueTy, createdOp, it.index());
    else
      resultVal = builder.create<pdl_interp::GetResultOp>(
          loc, valueTy, createdOp, it.index());
    type = builder.create<pdl_interp::GetValueTypeOp>(loc, resultVal);
  }
}

void PatternLowering::generateRewriter(
    pdl::ReplaceOp replaceOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 4> replOperands;

  // If the replacement was another operation, get its results. `pdl` allows
  // for using an operation for simplicitly, but the interpreter isn't as
  // user facing.
  if (Value replOp = replaceOp.replOperation()) {
    // Don't use replace if we know the replaced operation has no results.
    auto opOp = replaceOp.operation().getDefiningOp<pdl::OperationOp>();
    if (!opOp || !opOp.types().empty()) {
      replOperands.push_back(builder.create<pdl_interp::GetResultsOp>(
          replOp.getLoc(), mapRewriteValue(replOp)));
    }
  } else {
    for (Value operand : replaceOp.replValues())
      replOperands.push_back(mapRewriteValue(operand));
  }

  // If there are no replacement values, just create an erase instead.
  if (replOperands.empty()) {
    builder.create<pdl_interp::EraseOp>(replaceOp.getLoc(),
                                        mapRewriteValue(replaceOp.operation()));
    return;
  }

  builder.create<pdl_interp::ReplaceOp>(
      replaceOp.getLoc(), mapRewriteValue(replaceOp.operation()), replOperands);
}

void PatternLowering::generateRewriter(
    pdl::ResultOp resultOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  rewriteValues[resultOp] = builder.create<pdl_interp::GetResultOp>(
      resultOp.getLoc(), builder.getType<pdl::ValueType>(),
      mapRewriteValue(resultOp.parent()), resultOp.index());
}

void PatternLowering::generateRewriter(
    pdl::ResultsOp resultOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  rewriteValues[resultOp] = builder.create<pdl_interp::GetResultsOp>(
      resultOp.getLoc(), resultOp.getType(), mapRewriteValue(resultOp.parent()),
      resultOp.index());
}

void PatternLowering::generateRewriter(
    pdl::TypeOp typeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (TypeAttr typeAttr = typeOp.typeAttr()) {
    rewriteValues[typeOp] =
        builder.create<pdl_interp::CreateTypeOp>(typeOp.getLoc(), typeAttr);
  }
}

void PatternLowering::generateRewriter(
    pdl::TypesOp typeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (ArrayAttr typeAttr = typeOp.typesAttr()) {
    rewriteValues[typeOp] = builder.create<pdl_interp::CreateTypesOp>(
        typeOp.getLoc(), typeOp.getType(), typeAttr);
  }
}

void PatternLowering::generateOperationResultTypeRewriter(
    pdl::OperationOp op, SmallVectorImpl<Value> &types,
    DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // Look for an operation that was replaced by `op`. The result types will be
  // inferred from the results that were replaced.
  Block *rewriterBlock = op->getBlock();
  Value replacedOp;
  for (OpOperand &use : op.op().getUses()) {
    // Check that the use corresponds to a ReplaceOp and that it is the
    // replacement value, not the operation being replaced.
    pdl::ReplaceOp replOpUser = dyn_cast<pdl::ReplaceOp>(use.getOwner());
    if (!replOpUser || use.getOperandNumber() == 0)
      continue;
    // Make sure the replaced operation was defined before this one.
    Value replOpVal = replOpUser.operation();
    Operation *replacedOp = replOpVal.getDefiningOp();
    if (replacedOp->getBlock() == rewriterBlock &&
        !replacedOp->isBeforeInBlock(op))
      continue;

    Value replacedOpResults = builder.create<pdl_interp::GetResultsOp>(
        replacedOp->getLoc(), mapRewriteValue(replOpVal));
    types.push_back(builder.create<pdl_interp::GetValueTypeOp>(
        replacedOp->getLoc(), replacedOpResults));
    return;
  }

  // Check if the operation has type inference support.
  if (op.hasTypeInference()) {
    types.push_back(builder.create<pdl_interp::InferredTypesOp>(op.getLoc()));
    return;
  }

  // Otherwise, handle inference for each of the result types individually.
  OperandRange resultTypeValues = op.types();
  types.reserve(resultTypeValues.size());
  for (auto it : llvm::enumerate(resultTypeValues)) {
    Value resultType = it.value();

    // Check for an already translated value.
    if (Value existingRewriteValue = rewriteValues.lookup(resultType)) {
      types.push_back(existingRewriteValue);
      continue;
    }

    // Check for an input from the matcher.
    if (resultType.getDefiningOp()->getBlock() != rewriterBlock) {
      types.push_back(mapRewriteValue(resultType));
      continue;
    }

    // The verifier asserts that the result types of each pdl.operation can be
    // inferred. If we reach here, there is a bug either in the logic above or
    // in the verifier for pdl.operation.
    op->emitOpError() << "unable to infer result type for operation";
    llvm_unreachable("unable to infer result type for operation");
  }
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PDLToPDLInterpPass
    : public ConvertPDLToPDLInterpBase<PDLToPDLInterpPass> {
  void runOnOperation() final;
};
} // namespace

/// Convert the given module containing PDL pattern operations into a PDL
/// Interpreter operations.
void PDLToPDLInterpPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create the main matcher function This function contains all of the match
  // related functionality from patterns in the module.
  OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
  FuncOp matcherFunc = builder.create<FuncOp>(
      module.getLoc(), pdl_interp::PDLInterpDialect::getMatcherFunctionName(),
      builder.getFunctionType(builder.getType<pdl::OperationType>(),
                              /*results=*/llvm::None),
      /*attrs=*/llvm::None);

  // Create a nested module to hold the functions invoked for rewriting the IR
  // after a successful match.
  ModuleOp rewriterModule = builder.create<ModuleOp>(
      module.getLoc(), pdl_interp::PDLInterpDialect::getRewriterModuleName());

  // Generate the code for the patterns within the module.
  PatternLowering generator(matcherFunc, rewriterModule);
  generator.lower(module);

  // After generation, delete all of the pattern operations.
  for (pdl::PatternOp pattern :
       llvm::make_early_inc_range(module.getOps<pdl::PatternOp>()))
    pattern.erase();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createPDLToPDLInterpPass() {
  return std::make_unique<PDLToPDLInterpPass>();
}
