//===- PDLToPDLInterp.cpp - Lower a PDL module to the interpreter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"

#include "PredicateTree.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

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
  PatternLowering(pdl_interp::FuncOp matcherFunc, ModuleOp rewriterModule,
                  DenseMap<Operation *, PDLPatternConfigSet *> *configMap);

  /// Generate code for matching and rewriting based on the pattern operations
  /// within the module.
  void lower(ModuleOp module);

private:
  using ValueMap = llvm::ScopedHashTable<Position *, Value>;
  using ValueMapScope = llvm::ScopedHashTableScope<Position *, Value>;

  /// Generate interpreter operations for the tree rooted at the given matcher
  /// node, in the specified region.
  Block *generateMatcher(MatcherNode &node, Region &region,
                         Block *block = nullptr);

  /// Get or create an access to the provided positional value in the current
  /// block. This operation may mutate the provided block pointer if nested
  /// regions (i.e., pdl_interp.iterate) are required.
  Value getValueAt(Block *&currentBlock, Position *pos);

  /// Create the interpreter predicate operations. This operation may mutate the
  /// provided current block pointer if nested regions (iterates) are required.
  void generate(BoolNode *boolNode, Block *&currentBlock, Value val);

  /// Create the interpreter switch / predicate operations, with several case
  /// destinations. This operation never mutates the provided current block
  /// pointer, because the switch operation does not need Values beyond `val`.
  void generate(SwitchNode *switchNode, Block *currentBlock, Value val);

  /// Create the interpreter operations to record a successful pattern match
  /// using the contained root operation. This operation may mutate the current
  /// block pointer if nested regions (i.e., pdl_interp.iterate) are required.
  void generate(SuccessNode *successNode, Block *&currentBlock);

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
  void generateRewriter(pdl::RangeOp rangeOp,
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
  /// created within a dag rewriter region. If the result types of the operation
  /// should be inferred, `hasInferredResultTypes` is set to true.
  void generateOperationResultTypeRewriter(
      pdl::OperationOp op, function_ref<Value(Value)> mapRewriteValue,
      SmallVectorImpl<Value> &types, DenseMap<Value, Value> &rewriteValues,
      bool &hasInferredResultTypes);

  /// A builder to use when generating interpreter operations.
  OpBuilder builder;

  /// The matcher function used for all match related logic within PDL patterns.
  pdl_interp::FuncOp matcherFunc;

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

  /// The set of operation values whose location will be used for newly
  /// generated operations.
  SetVector<Value> locOps;

  /// A mapping between pattern operations and the corresponding configuration
  /// set.
  DenseMap<Operation *, PDLPatternConfigSet *> *configMap;

  /// A mapping from a constraint question to the ApplyConstraintOp
  /// that implements it.
  DenseMap<ConstraintQuestion *, pdl_interp::ApplyConstraintOp> constraintOpMap;
};
} // namespace

PatternLowering::PatternLowering(
    pdl_interp::FuncOp matcherFunc, ModuleOp rewriterModule,
    DenseMap<Operation *, PDLPatternConfigSet *> *configMap)
    : builder(matcherFunc.getContext()), matcherFunc(matcherFunc),
      rewriterModule(rewriterModule), rewriterSymbolTable(rewriterModule),
      configMap(configMap) {}

void PatternLowering::lower(ModuleOp module) {
  PredicateUniquer predicateUniquer;
  PredicateBuilder predicateBuilder(predicateUniquer, module.getContext());

  // Define top-level scope for the arguments to the matcher function.
  ValueMapScope topLevelValueScope(values);

  // Insert the root operation, i.e. argument to the matcher, at the root
  // position.
  Block *matcherEntryBlock = &matcherFunc.front();
  values.insert(predicateBuilder.getRoot(), matcherEntryBlock->getArgument(0));

  // Generate a root matcher node from the provided PDL module.
  std::unique_ptr<MatcherNode> root = MatcherNode::generateMatcherTree(
      module, predicateBuilder, valueToPosition);
  Block *firstMatcherBlock = generateMatcher(*root, matcherFunc.getBody());
  assert(failureBlockStack.empty() && "failed to empty the stack");

  // After generation, merged the first matched block into the entry.
  matcherEntryBlock->getOperations().splice(matcherEntryBlock->end(),
                                            firstMatcherBlock->getOperations());
  firstMatcherBlock->erase();
}

Block *PatternLowering::generateMatcher(MatcherNode &node, Region &region,
                                        Block *block) {
  // Push a new scope for the values used by this matcher.
  if (!block)
    block = &region.emplaceBlock();
  ValueMapScope scope(values);

  // If this is the return node, simply insert the corresponding interpreter
  // finalize.
  if (isa<ExitNode>(node)) {
    builder.setInsertionPointToEnd(block);
    pdl_interp::FinalizeOp::create(builder, matcherFunc.getLoc());
    return block;
  }

  // Get the next block in the match sequence.
  // This is intentionally executed first, before we get the value for the
  // position associated with the node, so that we preserve an "there exist"
  // semantics: if getting a value requires an upward traversal (going from a
  // value to its consumers), we want to perform the check on all the consumers
  // before we pass control to the failure node.
  std::unique_ptr<MatcherNode> &failureNode = node.getFailureNode();
  Block *failureBlock;
  if (failureNode) {
    failureBlock = generateMatcher(*failureNode, region);
    failureBlockStack.push_back(failureBlock);
  } else {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    failureBlock = failureBlockStack.back();
  }

  // If this node contains a position, get the corresponding value for this
  // block.
  Block *currentBlock = block;
  Position *position = node.getPosition();
  Value val = position ? getValueAt(currentBlock, position) : Value();

  // If this value corresponds to an operation, record that we are going to use
  // its location as part of a fused location.
  bool isOperationValue = val && isa<pdl::OperationType>(val.getType());
  if (isOperationValue)
    locOps.insert(val);

  // Dispatch to the correct method based on derived node type.
  TypeSwitch<MatcherNode *>(&node)
      .Case<BoolNode, SwitchNode>([&](auto *derivedNode) {
        this->generate(derivedNode, currentBlock, val);
      })
      .Case([&](SuccessNode *successNode) {
        generate(successNode, currentBlock);
      });

  // Pop all the failure blocks that were inserted due to nesting of
  // pdl_interp.iterate.
  while (failureBlockStack.back() != failureBlock) {
    failureBlockStack.pop_back();
    assert(!failureBlockStack.empty() && "unable to locate failure block");
  }

  // Pop the new failure block.
  if (failureNode)
    failureBlockStack.pop_back();

  if (isOperationValue)
    locOps.remove(val);

  return block;
}

Value PatternLowering::getValueAt(Block *&currentBlock, Position *pos) {
  if (Value val = values.lookup(pos))
    return val;

  // Get the value for the parent position.
  Value parentVal;
  if (Position *parent = pos->getParent())
    parentVal = getValueAt(currentBlock, parent);

  // TODO: Use a location from the position.
  Location loc = parentVal ? parentVal.getLoc() : builder.getUnknownLoc();
  builder.setInsertionPointToEnd(currentBlock);
  Value value;
  switch (pos->getKind()) {
  case Predicates::OperationPos: {
    auto *operationPos = cast<OperationPosition>(pos);
    if (operationPos->isOperandDefiningOp())
      // Standard (downward) traversal which directly follows the defining op.
      value = pdl_interp::GetDefiningOpOp::create(
          builder, loc, builder.getType<pdl::OperationType>(), parentVal);
    else
      // A passthrough operation position.
      value = parentVal;
    break;
  }
  case Predicates::UsersPos: {
    auto *usersPos = cast<UsersPosition>(pos);

    // The first operation retrieves the representative value of a range.
    // This applies only when the parent is a range of values and we were
    // requested to use a representative value (e.g., upward traversal).
    if (isa<pdl::RangeType>(parentVal.getType()) &&
        usersPos->useRepresentative())
      value = pdl_interp::ExtractOp::create(builder, loc, parentVal, 0);
    else
      value = parentVal;

    // The second operation retrieves the users.
    value = pdl_interp::GetUsersOp::create(builder, loc, value);
    break;
  }
  case Predicates::ForEachPos: {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    auto foreach = pdl_interp::ForEachOp::create(
        builder, loc, parentVal, failureBlockStack.back(), /*initLoop=*/true);
    value = foreach.getLoopVariable();

    // Create the continuation block.
    Block *continueBlock = builder.createBlock(&foreach.getRegion());
    pdl_interp::ContinueOp::create(builder, loc);
    failureBlockStack.push_back(continueBlock);

    currentBlock = &foreach.getRegion().front();
    break;
  }
  case Predicates::OperandPos: {
    auto *operandPos = cast<OperandPosition>(pos);
    value = pdl_interp::GetOperandOp::create(
        builder, loc, builder.getType<pdl::ValueType>(), parentVal,
        operandPos->getOperandNumber());
    break;
  }
  case Predicates::OperandGroupPos: {
    auto *operandPos = cast<OperandGroupPosition>(pos);
    Type valueTy = builder.getType<pdl::ValueType>();
    value = pdl_interp::GetOperandsOp::create(
        builder, loc,
        operandPos->isVariadic() ? pdl::RangeType::get(valueTy) : valueTy,
        parentVal, operandPos->getOperandGroupNumber());
    break;
  }
  case Predicates::AttributePos: {
    auto *attrPos = cast<AttributePosition>(pos);
    value = pdl_interp::GetAttributeOp::create(
        builder, loc, builder.getType<pdl::AttributeType>(), parentVal,
        attrPos->getName().strref());
    break;
  }
  case Predicates::TypePos: {
    if (isa<pdl::AttributeType>(parentVal.getType()))
      value = pdl_interp::GetAttributeTypeOp::create(builder, loc, parentVal);
    else
      value = pdl_interp::GetValueTypeOp::create(builder, loc, parentVal);
    break;
  }
  case Predicates::ResultPos: {
    auto *resPos = cast<ResultPosition>(pos);
    value = pdl_interp::GetResultOp::create(
        builder, loc, builder.getType<pdl::ValueType>(), parentVal,
        resPos->getResultNumber());
    break;
  }
  case Predicates::ResultGroupPos: {
    auto *resPos = cast<ResultGroupPosition>(pos);
    Type valueTy = builder.getType<pdl::ValueType>();
    value = pdl_interp::GetResultsOp::create(
        builder, loc,
        resPos->isVariadic() ? pdl::RangeType::get(valueTy) : valueTy,
        parentVal, resPos->getResultGroupNumber());
    break;
  }
  case Predicates::AttributeLiteralPos: {
    auto *attrPos = cast<AttributeLiteralPosition>(pos);
    value = pdl_interp::CreateAttributeOp::create(builder, loc,
                                                  attrPos->getValue());
    break;
  }
  case Predicates::TypeLiteralPos: {
    auto *typePos = cast<TypeLiteralPosition>(pos);
    Attribute rawTypeAttr = typePos->getValue();
    if (TypeAttr typeAttr = dyn_cast<TypeAttr>(rawTypeAttr))
      value = pdl_interp::CreateTypeOp::create(builder, loc, typeAttr);
    else
      value = pdl_interp::CreateTypesOp::create(builder, loc,
                                                cast<ArrayAttr>(rawTypeAttr));
    break;
  }
  case Predicates::ConstraintResultPos: {
    // Due to the order of traversal, the ApplyConstraintOp has already been
    // created and we can find it in constraintOpMap.
    auto *constrResPos = cast<ConstraintPosition>(pos);
    auto i = constraintOpMap.find(constrResPos->getQuestion());
    assert(i != constraintOpMap.end());
    value = i->second->getResult(constrResPos->getIndex());
    break;
  }
  default:
    llvm_unreachable("Generating unknown Position getter");
    break;
  }

  values.insert(pos, value);
  return value;
}

void PatternLowering::generate(BoolNode *boolNode, Block *&currentBlock,
                               Value val) {
  Location loc = val.getLoc();
  Qualifier *question = boolNode->getQuestion();
  Qualifier *answer = boolNode->getAnswer();
  Region *region = currentBlock->getParent();

  // Execute the getValue queries first, so that we create success
  // matcher in the correct (possibly nested) region.
  SmallVector<Value> args;
  if (auto *equalToQuestion = dyn_cast<EqualToQuestion>(question)) {
    args = {getValueAt(currentBlock, equalToQuestion->getValue())};
  } else if (auto *cstQuestion = dyn_cast<ConstraintQuestion>(question)) {
    for (Position *position : cstQuestion->getArgs())
      args.push_back(getValueAt(currentBlock, position));
  }

  // Generate a new block as success successor and get the failure successor.
  Block *success = &region->emplaceBlock();
  Block *failure = failureBlockStack.back();

  // Create the predicate.
  builder.setInsertionPointToEnd(currentBlock);
  Predicates::Kind kind = question->getKind();
  switch (kind) {
  case Predicates::IsNotNullQuestion:
    pdl_interp::IsNotNullOp::create(builder, loc, val, success, failure);
    break;
  case Predicates::OperationNameQuestion: {
    auto *opNameAnswer = cast<OperationNameAnswer>(answer);
    pdl_interp::CheckOperationNameOp::create(
        builder, loc, val, opNameAnswer->getValue().getStringRef(), success,
        failure);
    break;
  }
  case Predicates::TypeQuestion: {
    auto *ans = cast<TypeAnswer>(answer);
    if (isa<pdl::RangeType>(val.getType()))
      pdl_interp::CheckTypesOp::create(builder, loc, val,
                                       llvm::cast<ArrayAttr>(ans->getValue()),
                                       success, failure);
    else
      pdl_interp::CheckTypeOp::create(builder, loc, val,
                                      llvm::cast<TypeAttr>(ans->getValue()),
                                      success, failure);
    break;
  }
  case Predicates::AttributeQuestion: {
    auto *ans = cast<AttributeAnswer>(answer);
    pdl_interp::CheckAttributeOp::create(builder, loc, val, ans->getValue(),
                                         success, failure);
    break;
  }
  case Predicates::OperandCountAtLeastQuestion:
  case Predicates::OperandCountQuestion:
    pdl_interp::CheckOperandCountOp::create(
        builder, loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::OperandCountAtLeastQuestion,
        success, failure);
    break;
  case Predicates::ResultCountAtLeastQuestion:
  case Predicates::ResultCountQuestion:
    pdl_interp::CheckResultCountOp::create(
        builder, loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::ResultCountAtLeastQuestion,
        success, failure);
    break;
  case Predicates::EqualToQuestion: {
    bool trueAnswer = isa<TrueAnswer>(answer);
    pdl_interp::AreEqualOp::create(builder, loc, val, args.front(),
                                   trueAnswer ? success : failure,
                                   trueAnswer ? failure : success);
    break;
  }
  case Predicates::ConstraintQuestion: {
    auto *cstQuestion = cast<ConstraintQuestion>(question);
    auto applyConstraintOp = pdl_interp::ApplyConstraintOp::create(
        builder, loc, cstQuestion->getResultTypes(), cstQuestion->getName(),
        args, cstQuestion->getIsNegated(), success, failure);

    constraintOpMap.insert({cstQuestion, applyConstraintOp});
    break;
  }
  default:
    llvm_unreachable("Generating unknown Predicate operation");
  }

  // Generate the matcher in the current (potentially nested) region.
  // This might use the results of the current predicate.
  generateMatcher(*boolNode->getSuccessNode(), *region, success);
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
  OpT::create(builder, val.getLoc(), val, values, defaultDest, blocks);
}

void PatternLowering::generate(SwitchNode *switchNode, Block *currentBlock,
                               Value val) {
  Qualifier *question = switchNode->getQuestion();
  Region *region = currentBlock->getParent();
  Block *defaultDest = failureBlockStack.back();

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
    Location loc = val.getLoc();
    for (unsigned idx : sortedChildren) {
      auto &child = switchNode->getChild(idx);
      Block *childBlock = generateMatcher(*child.second, *region);
      Block *predicateBlock = builder.createBlock(childBlock);
      builder.setInsertionPointToEnd(predicateBlock);
      unsigned ans = cast<UnsignedAnswer>(child.first)->getValue();
      switch (kind) {
      case Predicates::OperandCountAtLeastQuestion:
        pdl_interp::CheckOperandCountOp::create(builder, loc, val, ans,
                                                /*compareAtLeast=*/true,
                                                childBlock, defaultDest);
        break;
      case Predicates::ResultCountAtLeastQuestion:
        pdl_interp::CheckResultCountOp::create(builder, loc, val, ans,
                                               /*compareAtLeast=*/true,
                                               childBlock, defaultDest);
        break;
      default:
        llvm_unreachable("Generating invalid AtLeast operation");
      }
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
    children.insert({it.first, generateMatcher(*it.second, *region)});
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
    if (isa<pdl::RangeType>(val.getType())) {
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

void PatternLowering::generate(SuccessNode *successNode, Block *&currentBlock) {
  pdl::PatternOp pattern = successNode->getPattern();
  Value root = successNode->getRoot();

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
  for (auto op :
       pattern.getRewriter().getBodyRegion().getOps<pdl::OperationOp>())
    generatedOps.push_back(*op.getOpName());
  ArrayAttr generatedOpsAttr;
  if (!generatedOps.empty())
    generatedOpsAttr = builder.getStrArrayAttr(generatedOps);

  // Grab the root kind if present.
  StringAttr rootKindAttr;
  if (pdl::OperationOp rootOp = root.getDefiningOp<pdl::OperationOp>())
    if (std::optional<StringRef> rootKind = rootOp.getOpName())
      rootKindAttr = builder.getStringAttr(*rootKind);

  builder.setInsertionPointToEnd(currentBlock);
  auto matchOp = pdl_interp::RecordMatchOp::create(
      builder, pattern.getLoc(), mappedMatchValues, locOps.getArrayRef(),
      rewriterFuncRef, rootKindAttr, generatedOpsAttr, pattern.getBenefitAttr(),
      failureBlockStack.back());

  // Set the config of the lowered match to the parent pattern.
  if (configMap)
    configMap->try_emplace(matchOp, configMap->lookup(pattern));
}

SymbolRefAttr PatternLowering::generateRewriter(
    pdl::PatternOp pattern, SmallVectorImpl<Position *> &usedMatchValues) {
  builder.setInsertionPointToEnd(rewriterModule.getBody());
  auto rewriterFunc = pdl_interp::FuncOp::create(
      builder, pattern.getLoc(), "pdl_generated_rewriter",
      builder.getFunctionType({}, {}));
  rewriterSymbolTable.insert(rewriterFunc);

  // Generate the rewriter function body.
  builder.setInsertionPointToEnd(&rewriterFunc.front());

  // Map an input operand of the pattern to a generated interpreter value.
  DenseMap<Value, Value> rewriteValues;
  auto mapRewriteValue = [&](Value oldValue) {
    Value &newValue = rewriteValues[oldValue];
    if (newValue)
      return newValue;

    // Prefer materializing constants directly when possible.
    Operation *oldOp = oldValue.getDefiningOp();
    if (pdl::AttributeOp attrOp = dyn_cast<pdl::AttributeOp>(oldOp)) {
      if (Attribute value = attrOp.getValueAttr()) {
        return newValue = pdl_interp::CreateAttributeOp::create(
                   builder, attrOp.getLoc(), value);
      }
    } else if (pdl::TypeOp typeOp = dyn_cast<pdl::TypeOp>(oldOp)) {
      if (TypeAttr type = typeOp.getConstantTypeAttr()) {
        return newValue = pdl_interp::CreateTypeOp::create(
                   builder, typeOp.getLoc(), type);
      }
    } else if (pdl::TypesOp typeOp = dyn_cast<pdl::TypesOp>(oldOp)) {
      if (ArrayAttr type = typeOp.getConstantTypesAttr()) {
        return newValue = pdl_interp::CreateTypesOp::create(
                   builder, typeOp.getLoc(), typeOp.getType(), type);
      }
    }

    // Otherwise, add this as an input to the rewriter.
    Position *inputPos = valueToPosition.lookup(oldValue);
    assert(inputPos && "expected value to be a pattern input");
    usedMatchValues.push_back(inputPos);
    return newValue = rewriterFunc.front().addArgument(oldValue.getType(),
                                                       oldValue.getLoc());
  };

  // If this is a custom rewriter, simply dispatch to the registered rewrite
  // method.
  pdl::RewriteOp rewriter = pattern.getRewriter();
  if (StringAttr rewriteName = rewriter.getNameAttr()) {
    SmallVector<Value> args;
    if (rewriter.getRoot())
      args.push_back(mapRewriteValue(rewriter.getRoot()));
    auto mappedArgs =
        llvm::map_range(rewriter.getExternalArgs(), mapRewriteValue);
    args.append(mappedArgs.begin(), mappedArgs.end());
    pdl_interp::ApplyRewriteOp::create(builder, rewriter.getLoc(),
                                       /*results=*/TypeRange(), rewriteName,
                                       args);
  } else {
    // Otherwise this is a dag rewriter defined using PDL operations.
    for (Operation &rewriteOp : *rewriter.getBody()) {
      llvm::TypeSwitch<Operation *>(&rewriteOp)
          .Case<pdl::ApplyNativeRewriteOp, pdl::AttributeOp, pdl::EraseOp,
                pdl::OperationOp, pdl::RangeOp, pdl::ReplaceOp, pdl::ResultOp,
                pdl::ResultsOp, pdl::TypeOp, pdl::TypesOp>([&](auto op) {
            this->generateRewriter(op, rewriteValues, mapRewriteValue);
          });
    }
  }

  // Update the signature of the rewrite function.
  rewriterFunc.setType(builder.getFunctionType(
      llvm::to_vector<8>(rewriterFunc.front().getArgumentTypes()),
      /*results=*/{}));

  pdl_interp::FinalizeOp::create(builder, rewriter.getLoc());
  return SymbolRefAttr::get(
      builder.getContext(),
      pdl_interp::PDLInterpDialect::getRewriterModuleName(),
      SymbolRefAttr::get(rewriterFunc));
}

void PatternLowering::generateRewriter(
    pdl::ApplyNativeRewriteOp rewriteOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 2> arguments;
  for (Value argument : rewriteOp.getArgs())
    arguments.push_back(mapRewriteValue(argument));
  auto interpOp = pdl_interp::ApplyRewriteOp::create(
      builder, rewriteOp.getLoc(), rewriteOp.getResultTypes(),
      rewriteOp.getNameAttr(), arguments);
  for (auto it : llvm::zip(rewriteOp.getResults(), interpOp.getResults()))
    rewriteValues[std::get<0>(it)] = std::get<1>(it);
}

void PatternLowering::generateRewriter(
    pdl::AttributeOp attrOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  Value newAttr = pdl_interp::CreateAttributeOp::create(
      builder, attrOp.getLoc(), attrOp.getValueAttr());
  rewriteValues[attrOp] = newAttr;
}

void PatternLowering::generateRewriter(
    pdl::EraseOp eraseOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  pdl_interp::EraseOp::create(builder, eraseOp.getLoc(),
                              mapRewriteValue(eraseOp.getOpValue()));
}

void PatternLowering::generateRewriter(
    pdl::OperationOp operationOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 4> operands;
  for (Value operand : operationOp.getOperandValues())
    operands.push_back(mapRewriteValue(operand));

  SmallVector<Value, 4> attributes;
  for (Value attr : operationOp.getAttributeValues())
    attributes.push_back(mapRewriteValue(attr));

  bool hasInferredResultTypes = false;
  SmallVector<Value, 2> types;
  generateOperationResultTypeRewriter(operationOp, mapRewriteValue, types,
                                      rewriteValues, hasInferredResultTypes);

  // Create the new operation.
  Location loc = operationOp.getLoc();
  Value createdOp = pdl_interp::CreateOperationOp::create(
      builder, loc, *operationOp.getOpName(), types, hasInferredResultTypes,
      operands, attributes, operationOp.getAttributeValueNames());
  rewriteValues[operationOp.getOp()] = createdOp;

  // Generate accesses for any results that have their types constrained.
  // Handle the case where there is a single range representing all of the
  // result types.
  OperandRange resultTys = operationOp.getTypeValues();
  if (resultTys.size() == 1 && isa<pdl::RangeType>(resultTys[0].getType())) {
    Value &type = rewriteValues[resultTys[0]];
    if (!type) {
      auto results = pdl_interp::GetResultsOp::create(builder, loc, createdOp);
      type = pdl_interp::GetValueTypeOp::create(builder, loc, results);
    }
    return;
  }

  // Otherwise, populate the individual results.
  bool seenVariableLength = false;
  Type valueTy = builder.getType<pdl::ValueType>();
  Type valueRangeTy = pdl::RangeType::get(valueTy);
  for (const auto &it : llvm::enumerate(resultTys)) {
    Value &type = rewriteValues[it.value()];
    if (type)
      continue;
    bool isVariadic = isa<pdl::RangeType>(it.value().getType());
    seenVariableLength |= isVariadic;

    // After a variable length result has been seen, we need to use result
    // groups because the exact index of the result is not statically known.
    Value resultVal;
    if (seenVariableLength)
      resultVal = pdl_interp::GetResultsOp::create(
          builder, loc, isVariadic ? valueRangeTy : valueTy, createdOp,
          it.index());
    else
      resultVal = pdl_interp::GetResultOp::create(builder, loc, valueTy,
                                                  createdOp, it.index());
    type = pdl_interp::GetValueTypeOp::create(builder, loc, resultVal);
  }
}

void PatternLowering::generateRewriter(
    pdl::RangeOp rangeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 4> replOperands;
  for (Value operand : rangeOp.getArguments())
    replOperands.push_back(mapRewriteValue(operand));
  rewriteValues[rangeOp] = pdl_interp::CreateRangeOp::create(
      builder, rangeOp.getLoc(), rangeOp.getType(), replOperands);
}

void PatternLowering::generateRewriter(
    pdl::ReplaceOp replaceOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  SmallVector<Value, 4> replOperands;

  // If the replacement was another operation, get its results. `pdl` allows
  // for using an operation for simplicitly, but the interpreter isn't as
  // user facing.
  if (Value replOp = replaceOp.getReplOperation()) {
    // Don't use replace if we know the replaced operation has no results.
    auto opOp = replaceOp.getOpValue().getDefiningOp<pdl::OperationOp>();
    if (!opOp || !opOp.getTypeValues().empty()) {
      replOperands.push_back(pdl_interp::GetResultsOp::create(
          builder, replOp.getLoc(), mapRewriteValue(replOp)));
    }
  } else {
    for (Value operand : replaceOp.getReplValues())
      replOperands.push_back(mapRewriteValue(operand));
  }

  // If there are no replacement values, just create an erase instead.
  if (replOperands.empty()) {
    pdl_interp::EraseOp::create(builder, replaceOp.getLoc(),
                                mapRewriteValue(replaceOp.getOpValue()));
    return;
  }

  pdl_interp::ReplaceOp::create(builder, replaceOp.getLoc(),
                                mapRewriteValue(replaceOp.getOpValue()),
                                replOperands);
}

void PatternLowering::generateRewriter(
    pdl::ResultOp resultOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  rewriteValues[resultOp] = pdl_interp::GetResultOp::create(
      builder, resultOp.getLoc(), builder.getType<pdl::ValueType>(),
      mapRewriteValue(resultOp.getParent()), resultOp.getIndex());
}

void PatternLowering::generateRewriter(
    pdl::ResultsOp resultOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  rewriteValues[resultOp] = pdl_interp::GetResultsOp::create(
      builder, resultOp.getLoc(), resultOp.getType(),
      mapRewriteValue(resultOp.getParent()), resultOp.getIndex());
}

void PatternLowering::generateRewriter(
    pdl::TypeOp typeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (TypeAttr typeAttr = typeOp.getConstantTypeAttr()) {
    rewriteValues[typeOp] =
        pdl_interp::CreateTypeOp::create(builder, typeOp.getLoc(), typeAttr);
  }
}

void PatternLowering::generateRewriter(
    pdl::TypesOp typeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (ArrayAttr typeAttr = typeOp.getConstantTypesAttr()) {
    rewriteValues[typeOp] = pdl_interp::CreateTypesOp::create(
        builder, typeOp.getLoc(), typeOp.getType(), typeAttr);
  }
}

void PatternLowering::generateOperationResultTypeRewriter(
    pdl::OperationOp op, function_ref<Value(Value)> mapRewriteValue,
    SmallVectorImpl<Value> &types, DenseMap<Value, Value> &rewriteValues,
    bool &hasInferredResultTypes) {
  Block *rewriterBlock = op->getBlock();

  // Try to handle resolution for each of the result types individually. This is
  // preferred over type inferrence because it will allow for us to use existing
  // types directly, as opposed to trying to rebuild the type list.
  OperandRange resultTypeValues = op.getTypeValues();
  auto tryResolveResultTypes = [&] {
    types.reserve(resultTypeValues.size());
    for (const auto &it : llvm::enumerate(resultTypeValues)) {
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

      // Otherwise, we couldn't infer the result types. Bail out here to see if
      // we can infer the types for this operation from another way.
      types.clear();
      return failure();
    }
    return success();
  };
  if (!resultTypeValues.empty() && succeeded(tryResolveResultTypes()))
    return;

  // Otherwise, check if the operation has type inference support itself.
  if (op.hasTypeInference()) {
    hasInferredResultTypes = true;
    return;
  }

  // Look for an operation that was replaced by `op`. The result types will be
  // inferred from the results that were replaced.
  for (OpOperand &use : op.getOp().getUses()) {
    // Check that the use corresponds to a ReplaceOp and that it is the
    // replacement value, not the operation being replaced.
    pdl::ReplaceOp replOpUser = dyn_cast<pdl::ReplaceOp>(use.getOwner());
    if (!replOpUser || use.getOperandNumber() == 0)
      continue;
    // Make sure the replaced operation was defined before this one. PDL
    // rewrites only have single block regions, so if the op isn't in the
    // rewriter block (i.e. the current block of the operation) we already know
    // it dominates (i.e. it's in the matcher).
    Value replOpVal = replOpUser.getOpValue();
    Operation *replacedOp = replOpVal.getDefiningOp();
    if (replacedOp->getBlock() == rewriterBlock &&
        !replacedOp->isBeforeInBlock(op))
      continue;

    Value replacedOpResults = pdl_interp::GetResultsOp::create(
        builder, replacedOp->getLoc(), mapRewriteValue(replOpVal));
    types.push_back(pdl_interp::GetValueTypeOp::create(
        builder, replacedOp->getLoc(), replacedOpResults));
    return;
  }

  // If the types could not be inferred from any context and there weren't any
  // explicit result types, assume the user actually meant for the operation to
  // have no results.
  if (resultTypeValues.empty())
    return;

  // The verifier asserts that the result types of each pdl.getOperation can be
  // inferred. If we reach here, there is a bug either in the logic above or
  // in the verifier for pdl.getOperation.
  op->emitOpError() << "unable to infer result type for operation";
  llvm_unreachable("unable to infer result type for operation");
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PDLToPDLInterpPass
    : public impl::ConvertPDLToPDLInterpPassBase<PDLToPDLInterpPass> {
  PDLToPDLInterpPass() = default;
  PDLToPDLInterpPass(const PDLToPDLInterpPass &rhs) = default;
  PDLToPDLInterpPass(DenseMap<Operation *, PDLPatternConfigSet *> &configMap)
      : configMap(&configMap) {}
  void runOnOperation() final;

  /// A map containing the configuration for each pattern.
  DenseMap<Operation *, PDLPatternConfigSet *> *configMap = nullptr;
};
} // namespace

/// Convert the given module containing PDL pattern operations into a PDL
/// Interpreter operations.
void PDLToPDLInterpPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create the main matcher function This function contains all of the match
  // related functionality from patterns in the module.
  OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
  auto matcherFunc = pdl_interp::FuncOp::create(
      builder, module.getLoc(),
      pdl_interp::PDLInterpDialect::getMatcherFunctionName(),
      builder.getFunctionType(builder.getType<pdl::OperationType>(),
                              /*results=*/{}),
      /*attrs=*/ArrayRef<NamedAttribute>());

  // Create a nested module to hold the functions invoked for rewriting the IR
  // after a successful match.
  ModuleOp rewriterModule =
      ModuleOp::create(builder, module.getLoc(),
                       pdl_interp::PDLInterpDialect::getRewriterModuleName());

  // Generate the code for the patterns within the module.
  PatternLowering generator(matcherFunc, rewriterModule, configMap);
  generator.lower(module);

  // After generation, delete all of the pattern operations.
  for (pdl::PatternOp pattern :
       llvm::make_early_inc_range(module.getOps<pdl::PatternOp>())) {
    // Drop the now dead config mappings.
    if (configMap)
      configMap->erase(pattern);

    pattern.erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertPDLToPDLInterpPass(
    DenseMap<Operation *, PDLPatternConfigSet *> &configMap) {
  return std::make_unique<PDLToPDLInterpPass>(configMap);
}
