//===- PDLToPDLInterp.cpp - Lower a PDL module to the interpreter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"

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

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLTOPDLINTERP
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
  Block *generateMatcher(MatcherNode &node, Region &region);

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

  /// The set of operation values whose whose location will be used for newly
  /// generated operations.
  SetVector<Value> locOps;

  /// A mapping between pattern operations and the corresponding configuration
  /// set.
  DenseMap<Operation *, PDLPatternConfigSet *> *configMap;
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

Block *PatternLowering::generateMatcher(MatcherNode &node, Region &region) {
  // Push a new scope for the values used by this matcher.
  Block *block = &region.emplaceBlock();
  ValueMapScope scope(values);

  // If this is the return node, simply insert the corresponding interpreter
  // finalize.
  if (isa<ExitNode>(node)) {
    builder.setInsertionPointToEnd(block);
    builder.create<pdl_interp::FinalizeOp>(matcherFunc.getLoc());
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
  bool isOperationValue = val && val.getType().isa<pdl::OperationType>();
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
      value = builder.create<pdl_interp::GetDefiningOpOp>(
          loc, builder.getType<pdl::OperationType>(), parentVal);
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
    if (parentVal.getType().isa<pdl::RangeType>() &&
        usersPos->useRepresentative())
      value = builder.create<pdl_interp::ExtractOp>(loc, parentVal, 0);
    else
      value = parentVal;

    // The second operation retrieves the users.
    value = builder.create<pdl_interp::GetUsersOp>(loc, value);
    break;
  }
  case Predicates::ForEachPos: {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    auto foreach = builder.create<pdl_interp::ForEachOp>(
        loc, parentVal, failureBlockStack.back(), /*initLoop=*/true);
    value = foreach.getLoopVariable();

    // Create the continuation block.
    Block *continueBlock = builder.createBlock(&foreach.getRegion());
    builder.create<pdl_interp::ContinueOp>(loc);
    failureBlockStack.push_back(continueBlock);

    currentBlock = &foreach.getRegion().front();
    break;
  }
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
  case Predicates::AttributeLiteralPos: {
    auto *attrPos = cast<AttributeLiteralPosition>(pos);
    value =
        builder.create<pdl_interp::CreateAttributeOp>(loc, attrPos->getValue());
    break;
  }
  case Predicates::TypeLiteralPos: {
    auto *typePos = cast<TypeLiteralPosition>(pos);
    Attribute rawTypeAttr = typePos->getValue();
    if (TypeAttr typeAttr = rawTypeAttr.dyn_cast<TypeAttr>())
      value = builder.create<pdl_interp::CreateTypeOp>(loc, typeAttr);
    else
      value = builder.create<pdl_interp::CreateTypesOp>(
          loc, rawTypeAttr.cast<ArrayAttr>());
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

  // Generate the matcher in the current (potentially nested) region
  // and get the failure successor.
  Block *success = generateMatcher(*boolNode->getSuccessNode(), *region);
  Block *failure = failureBlockStack.back();

  // Finally, create the predicate.
  builder.setInsertionPointToEnd(currentBlock);
  Predicates::Kind kind = question->getKind();
  switch (kind) {
  case Predicates::IsNotNullQuestion:
    builder.create<pdl_interp::IsNotNullOp>(loc, val, success, failure);
    break;
  case Predicates::OperationNameQuestion: {
    auto *opNameAnswer = cast<OperationNameAnswer>(answer);
    builder.create<pdl_interp::CheckOperationNameOp>(
        loc, val, opNameAnswer->getValue().getStringRef(), success, failure);
    break;
  }
  case Predicates::TypeQuestion: {
    auto *ans = cast<TypeAnswer>(answer);
    if (val.getType().isa<pdl::RangeType>())
      builder.create<pdl_interp::CheckTypesOp>(
          loc, val, ans->getValue().cast<ArrayAttr>(), success, failure);
    else
      builder.create<pdl_interp::CheckTypeOp>(
          loc, val, ans->getValue().cast<TypeAttr>(), success, failure);
    break;
  }
  case Predicates::AttributeQuestion: {
    auto *ans = cast<AttributeAnswer>(answer);
    builder.create<pdl_interp::CheckAttributeOp>(loc, val, ans->getValue(),
                                                 success, failure);
    break;
  }
  case Predicates::OperandCountAtLeastQuestion:
  case Predicates::OperandCountQuestion:
    builder.create<pdl_interp::CheckOperandCountOp>(
        loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::OperandCountAtLeastQuestion,
        success, failure);
    break;
  case Predicates::ResultCountAtLeastQuestion:
  case Predicates::ResultCountQuestion:
    builder.create<pdl_interp::CheckResultCountOp>(
        loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::ResultCountAtLeastQuestion,
        success, failure);
    break;
  case Predicates::EqualToQuestion: {
    bool trueAnswer = isa<TrueAnswer>(answer);
    builder.create<pdl_interp::AreEqualOp>(loc, val, args.front(),
                                           trueAnswer ? success : failure,
                                           trueAnswer ? failure : success);
    break;
  }
  case Predicates::ConstraintQuestion: {
    auto *cstQuestion = cast<ConstraintQuestion>(question);
    builder.create<pdl_interp::ApplyConstraintOp>(loc, cstQuestion->getName(),
                                                  args, success, failure);
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
        builder.create<pdl_interp::CheckOperandCountOp>(
            loc, val, ans, /*compareAtLeast=*/true, childBlock, defaultDest);
        break;
      case Predicates::ResultCountAtLeastQuestion:
        builder.create<pdl_interp::CheckResultCountOp>(
            loc, val, ans, /*compareAtLeast=*/true, childBlock, defaultDest);
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
    if (Optional<StringRef> rootKind = rootOp.getOpName())
      rootKindAttr = builder.getStringAttr(*rootKind);

  builder.setInsertionPointToEnd(currentBlock);
  auto matchOp = builder.create<pdl_interp::RecordMatchOp>(
      pattern.getLoc(), mappedMatchValues, locOps.getArrayRef(),
      rewriterFuncRef, rootKindAttr, generatedOpsAttr, pattern.getBenefitAttr(),
      failureBlockStack.back());

  // Set the config of the lowered match to the parent pattern.
  if (configMap)
    configMap->try_emplace(matchOp, configMap->lookup(pattern));
}

SymbolRefAttr PatternLowering::generateRewriter(
    pdl::PatternOp pattern, SmallVectorImpl<Position *> &usedMatchValues) {
  builder.setInsertionPointToEnd(rewriterModule.getBody());
  auto rewriterFunc = builder.create<pdl_interp::FuncOp>(
      pattern.getLoc(), "pdl_generated_rewriter",
      builder.getFunctionType(llvm::None, llvm::None));
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
        return newValue = builder.create<pdl_interp::CreateAttributeOp>(
                   attrOp.getLoc(), value);
      }
    } else if (pdl::TypeOp typeOp = dyn_cast<pdl::TypeOp>(oldOp)) {
      if (TypeAttr type = typeOp.getConstantTypeAttr()) {
        return newValue = builder.create<pdl_interp::CreateTypeOp>(
                   typeOp.getLoc(), type);
      }
    } else if (pdl::TypesOp typeOp = dyn_cast<pdl::TypesOp>(oldOp)) {
      if (ArrayAttr type = typeOp.getConstantTypesAttr()) {
        return newValue = builder.create<pdl_interp::CreateTypesOp>(
                   typeOp.getLoc(), typeOp.getType(), type);
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
    builder.create<pdl_interp::ApplyRewriteOp>(
        rewriter.getLoc(), /*resultTypes=*/TypeRange(), rewriteName, args);
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
  auto interpOp = builder.create<pdl_interp::ApplyRewriteOp>(
      rewriteOp.getLoc(), rewriteOp.getResultTypes(), rewriteOp.getNameAttr(),
      arguments);
  for (auto it : llvm::zip(rewriteOp.getResults(), interpOp.getResults()))
    rewriteValues[std::get<0>(it)] = std::get<1>(it);
}

void PatternLowering::generateRewriter(
    pdl::AttributeOp attrOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  Value newAttr = builder.create<pdl_interp::CreateAttributeOp>(
      attrOp.getLoc(), attrOp.getValueAttr());
  rewriteValues[attrOp] = newAttr;
}

void PatternLowering::generateRewriter(
    pdl::EraseOp eraseOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  builder.create<pdl_interp::EraseOp>(eraseOp.getLoc(),
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
  Value createdOp = builder.create<pdl_interp::CreateOperationOp>(
      loc, *operationOp.getOpName(), types, hasInferredResultTypes, operands,
      attributes, operationOp.getAttributeValueNames());
  rewriteValues[operationOp.getOp()] = createdOp;

  // Generate accesses for any results that have their types constrained.
  // Handle the case where there is a single range representing all of the
  // result types.
  OperandRange resultTys = operationOp.getTypeValues();
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
  for (const auto &it : llvm::enumerate(resultTys)) {
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
  if (Value replOp = replaceOp.getReplOperation()) {
    // Don't use replace if we know the replaced operation has no results.
    auto opOp = replaceOp.getOpValue().getDefiningOp<pdl::OperationOp>();
    if (!opOp || !opOp.getTypeValues().empty()) {
      replOperands.push_back(builder.create<pdl_interp::GetResultsOp>(
          replOp.getLoc(), mapRewriteValue(replOp)));
    }
  } else {
    for (Value operand : replaceOp.getReplValues())
      replOperands.push_back(mapRewriteValue(operand));
  }

  // If there are no replacement values, just create an erase instead.
  if (replOperands.empty()) {
    builder.create<pdl_interp::EraseOp>(
        replaceOp.getLoc(), mapRewriteValue(replaceOp.getOpValue()));
    return;
  }

  builder.create<pdl_interp::ReplaceOp>(replaceOp.getLoc(),
                                        mapRewriteValue(replaceOp.getOpValue()),
                                        replOperands);
}

void PatternLowering::generateRewriter(
    pdl::ResultOp resultOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  rewriteValues[resultOp] = builder.create<pdl_interp::GetResultOp>(
      resultOp.getLoc(), builder.getType<pdl::ValueType>(),
      mapRewriteValue(resultOp.getParent()), resultOp.getIndex());
}

void PatternLowering::generateRewriter(
    pdl::ResultsOp resultOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  rewriteValues[resultOp] = builder.create<pdl_interp::GetResultsOp>(
      resultOp.getLoc(), resultOp.getType(),
      mapRewriteValue(resultOp.getParent()), resultOp.getIndex());
}

void PatternLowering::generateRewriter(
    pdl::TypeOp typeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (TypeAttr typeAttr = typeOp.getConstantTypeAttr()) {
    rewriteValues[typeOp] =
        builder.create<pdl_interp::CreateTypeOp>(typeOp.getLoc(), typeAttr);
  }
}

void PatternLowering::generateRewriter(
    pdl::TypesOp typeOp, DenseMap<Value, Value> &rewriteValues,
    function_ref<Value(Value)> mapRewriteValue) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (ArrayAttr typeAttr = typeOp.getConstantTypesAttr()) {
    rewriteValues[typeOp] = builder.create<pdl_interp::CreateTypesOp>(
        typeOp.getLoc(), typeOp.getType(), typeAttr);
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

    Value replacedOpResults = builder.create<pdl_interp::GetResultsOp>(
        replacedOp->getLoc(), mapRewriteValue(replOpVal));
    types.push_back(builder.create<pdl_interp::GetValueTypeOp>(
        replacedOp->getLoc(), replacedOpResults));
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
    : public impl::ConvertPDLToPDLInterpBase<PDLToPDLInterpPass> {
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
  auto matcherFunc = builder.create<pdl_interp::FuncOp>(
      module.getLoc(), pdl_interp::PDLInterpDialect::getMatcherFunctionName(),
      builder.getFunctionType(builder.getType<pdl::OperationType>(),
                              /*results=*/llvm::None),
      /*attrs=*/llvm::None);

  // Create a nested module to hold the functions invoked for rewriting the IR
  // after a successful match.
  ModuleOp rewriterModule = builder.create<ModuleOp>(
      module.getLoc(), pdl_interp::PDLInterpDialect::getRewriterModuleName());

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

std::unique_ptr<OperationPass<ModuleOp>> mlir::createPDLToPDLInterpPass() {
  return std::make_unique<PDLToPDLInterpPass>();
}
std::unique_ptr<OperationPass<ModuleOp>> mlir::createPDLToPDLInterpPass(
    DenseMap<Operation *, PDLPatternConfigSet *> &configMap) {
  return std::make_unique<PDLToPDLInterpPass>(configMap);
}
