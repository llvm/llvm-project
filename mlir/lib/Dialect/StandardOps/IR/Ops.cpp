//===- Ops.cpp - Standard MLIR Operations ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/StandardOps/IR/OpsDialect.cpp.inc"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// StandardOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with standard
/// operations.
struct StdInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within standard ops can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = dyn_cast<ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// StandardOpsDialect
//===----------------------------------------------------------------------===//

/// A custom unary operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op should have one operand");
  assert(op->getNumResults() == 1 && "unary op should have one result");

  int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0).getType();
}

/// A custom binary operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op should have two operands");
  assert(op->getNumResults() == 1 && "binary op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (op->getOperand(0).getType() != resultType ||
      op->getOperand(1).getType() != resultType) {
    p.printGenericOp(op);
    return;
  }

  int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0).getType();
}

/// A custom ternary operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardTernaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 3 && "ternary op should have three operands");
  assert(op->getNumResults() == 1 && "ternary op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (op->getOperand(0).getType() != resultType ||
      op->getOperand(1).getType() != resultType ||
      op->getOperand(2).getType() != resultType) {
    p.printGenericOp(op);
    return;
  }

  int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << op->getOperand(0) << ", " << op->getOperand(1) << ", "
    << op->getOperand(2);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0).getType();
}

/// A custom cast operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardCastOp(Operation *op, OpAsmPrinter &p) {
  int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << op->getOperand(0) << " : " << op->getOperand(0).getType() << " to "
    << op->getResult(0).getType();
}

void StandardOpsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/StandardOps/IR/Ops.cpp.inc"
      >();
  addInterfaces<StdInlinerInterface>();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *StandardOpsDialect::materializeConstant(OpBuilder &builder,
                                                   Attribute value, Type type,
                                                   Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Common cast compatibility check for vector types.
//===----------------------------------------------------------------------===//

/// This method checks for cast compatibility of vector types.
/// If 'a' and 'b' are vector types, and they are cast compatible,
/// it calls the 'areElementsCastCompatible' function to check for
/// element cast compatibility.
/// Returns 'true' if the vector types are cast compatible,  and 'false'
/// otherwise.
static bool areVectorCastSimpleCompatible(
    Type a, Type b,
    function_ref<bool(TypeRange, TypeRange)> areElementsCastCompatible) {
  if (auto va = a.dyn_cast<VectorType>())
    if (auto vb = b.dyn_cast<VectorType>())
      return va.getShape().equals(vb.getShape()) &&
             areElementsCastCompatible(va.getElementType(),
                                       vb.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

OpFoldResult AddFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult AddIOp::fold(ArrayRef<Attribute> operands) {
  /// addi(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

/// Canonicalize a sum of a constant and (constant - something) to simply be
/// a sum of constants minus something. This transformation does similar
/// transformations for additions of a constant with a subtract/add of
/// a constant. This may result in some operations being reordered (but should
/// remain equivalent).
struct AddConstantReorder : public OpRewritePattern<AddIOp> {
  using OpRewritePattern<AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddIOp addop,
                                PatternRewriter &rewriter) const override {
    for (int i = 0; i < 2; i++) {
      APInt origConst;
      APInt midConst;
      if (matchPattern(addop.getOperand(i), m_ConstantInt(&origConst))) {
        if (auto midAddOp = addop.getOperand(1 - i).getDefiningOp<AddIOp>()) {
          for (int j = 0; j < 2; j++) {
            if (matchPattern(midAddOp.getOperand(j),
                             m_ConstantInt(&midConst))) {
              auto nextConstant = rewriter.create<ConstantOp>(
                  addop.getLoc(), rewriter.getIntegerAttr(
                                      addop.getType(), origConst + midConst));
              rewriter.replaceOpWithNewOp<AddIOp>(addop, nextConstant,
                                                  midAddOp.getOperand(1 - j));
              return success();
            }
          }
        }
        if (auto midSubOp = addop.getOperand(1 - i).getDefiningOp<SubIOp>()) {
          if (matchPattern(midSubOp.getOperand(0), m_ConstantInt(&midConst))) {
            auto nextConstant = rewriter.create<ConstantOp>(
                addop.getLoc(),
                rewriter.getIntegerAttr(addop.getType(), origConst + midConst));
            rewriter.replaceOpWithNewOp<SubIOp>(addop, nextConstant,
                                                midSubOp.getOperand(1));
            return success();
          }
          if (matchPattern(midSubOp.getOperand(1), m_ConstantInt(&midConst))) {
            auto nextConstant = rewriter.create<ConstantOp>(
                addop.getLoc(),
                rewriter.getIntegerAttr(addop.getType(), origConst - midConst));
            rewriter.replaceOpWithNewOp<AddIOp>(addop, nextConstant,
                                                midSubOp.getOperand(0));
            return success();
          }
        }
      }
    }
    return failure();
  }
};

void AddIOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<AddConstantReorder>(context);
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  /// and(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// and(x, allOnes) -> x
  APInt intValue;
  if (matchPattern(rhs(), m_ConstantInt(&intValue)) &&
      intValue.isAllOnesValue())
    return lhs();
  /// and(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  // Erase assertion if argument is constant true.
  if (matchPattern(op.arg(), m_One())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AtomicRMWOp op) {
  if (op.getMemRefType().getRank() != op.getNumOperands() - 2)
    return op.emitOpError(
        "expects the number of subscripts to be equal to memref rank");
  switch (op.kind()) {
  case AtomicRMWKind::addf:
  case AtomicRMWKind::maxf:
  case AtomicRMWKind::minf:
  case AtomicRMWKind::mulf:
    if (!op.value().getType().isa<FloatType>())
      return op.emitOpError()
             << "with kind '" << stringifyAtomicRMWKind(op.kind())
             << "' expects a floating-point type";
    break;
  case AtomicRMWKind::addi:
  case AtomicRMWKind::maxs:
  case AtomicRMWKind::maxu:
  case AtomicRMWKind::mins:
  case AtomicRMWKind::minu:
  case AtomicRMWKind::muli:
    if (!op.value().getType().isa<IntegerType>())
      return op.emitOpError()
             << "with kind '" << stringifyAtomicRMWKind(op.kind())
             << "' expects an integer type";
    break;
  default:
    break;
  }
  return success();
}

/// Returns the identity value attribute associated with an AtomicRMWKind op.
Attribute mlir::getIdentityValueAttr(AtomicRMWKind kind, Type resultType,
                                     OpBuilder &builder, Location loc) {
  switch (kind) {
  case AtomicRMWKind::addf:
  case AtomicRMWKind::addi:
    return builder.getZeroAttr(resultType);
  case AtomicRMWKind::muli:
    return builder.getIntegerAttr(resultType, 1);
  case AtomicRMWKind::mulf:
    return builder.getFloatAttr(resultType, 1);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

/// Returns the identity value associated with an AtomicRMWKind op.
Value mlir::getIdentityValue(AtomicRMWKind op, Type resultType,
                             OpBuilder &builder, Location loc) {
  Attribute attr = getIdentityValueAttr(op, resultType, builder, loc);
  return builder.create<ConstantOp>(loc, attr);
}

/// Return the value obtained by applying the reduction operation kind
/// associated with a binary AtomicRMWKind op to `lhs` and `rhs`.
Value mlir::getReductionOp(AtomicRMWKind op, OpBuilder &builder, Location loc,
                           Value lhs, Value rhs) {
  switch (op) {
  case AtomicRMWKind::addf:
    return builder.create<AddFOp>(loc, lhs, rhs);
  case AtomicRMWKind::addi:
    return builder.create<AddIOp>(loc, lhs, rhs);
  case AtomicRMWKind::mulf:
    return builder.create<MulFOp>(loc, lhs, rhs);
  case AtomicRMWKind::muli:
    return builder.create<MulIOp>(loc, lhs, rhs);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// GenericAtomicRMWOp
//===----------------------------------------------------------------------===//

void GenericAtomicRMWOp::build(OpBuilder &builder, OperationState &result,
                               Value memref, ValueRange ivs) {
  result.addOperands(memref);
  result.addOperands(ivs);

  if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
    Type elementType = memrefType.getElementType();
    result.addTypes(elementType);

    Region *bodyRegion = result.addRegion();
    bodyRegion->push_back(new Block());
    bodyRegion->addArgument(elementType);
  }
}

static LogicalResult verify(GenericAtomicRMWOp op) {
  auto &body = op.body();
  if (body.getNumArguments() != 1)
    return op.emitOpError("expected single number of entry block arguments");

  if (op.getResult().getType() != body.getArgument(0).getType())
    return op.emitOpError(
        "expected block argument of the same type result type");

  bool hasSideEffects =
      body.walk([&](Operation *nestedOp) {
            if (MemoryEffectOpInterface::hasNoEffect(nestedOp))
              return WalkResult::advance();
            nestedOp->emitError("body of 'generic_atomic_rmw' should contain "
                                "only operations with no side effects");
            return WalkResult::interrupt();
          })
          .wasInterrupted();
  return hasSideEffects ? failure() : success();
}

static ParseResult parseGenericAtomicRMWOp(OpAsmParser &parser,
                                           OperationState &result) {
  OpAsmParser::OperandType memref;
  Type memrefType;
  SmallVector<OpAsmParser::OperandType, 4> ivs;

  Type indexType = parser.getBuilder().getIndexType();
  if (parser.parseOperand(memref) ||
      parser.parseOperandList(ivs, OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(memrefType) ||
      parser.resolveOperand(memref, memrefType, result.operands) ||
      parser.resolveOperands(ivs, indexType, result.operands))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.types.push_back(memrefType.cast<MemRefType>().getElementType());
  return success();
}

static void print(OpAsmPrinter &p, GenericAtomicRMWOp op) {
  p << op.getOperationName() << ' ' << op.memref() << "[" << op.indices()
    << "] : " << op.memref().getType();
  p.printRegion(op.body());
  p.printOptionalAttrDict(op->getAttrs());
}

//===----------------------------------------------------------------------===//
// AtomicYieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AtomicYieldOp op) {
  Type parentType = op->getParentOp()->getResultTypes().front();
  Type resultType = op.result().getType();
  if (parentType != resultType)
    return op.emitOpError() << "types mismatch between yield op: " << resultType
                            << " and its parent: " << parentType;
  return success();
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

/// Given a successor, try to collapse it to a new destination if it only
/// contains a passthrough unconditional branch. If the successor is
/// collapsable, `successor` and `successorOperands` are updated to reference
/// the new destination and values. `argStorage` is used as storage if operands
/// to the collapsed successor need to be remapped. It must outlive uses of
/// successorOperands.
static LogicalResult collapseBranch(Block *&successor,
                                    ValueRange &successorOperands,
                                    SmallVectorImpl<Value> &argStorage) {
  // Check that the successor only contains a unconditional branch.
  if (std::next(successor->begin()) != successor->end())
    return failure();
  // Check that the terminator is an unconditional branch.
  BranchOp successorBranch = dyn_cast<BranchOp>(successor->getTerminator());
  if (!successorBranch)
    return failure();
  // Check that the arguments are only used within the terminator.
  for (BlockArgument arg : successor->getArguments()) {
    for (Operation *user : arg.getUsers())
      if (user != successorBranch)
        return failure();
  }
  // Don't try to collapse branches to infinite loops.
  Block *successorDest = successorBranch.getDest();
  if (successorDest == successor)
    return failure();

  // Update the operands to the successor. If the branch parent has no
  // arguments, we can use the branch operands directly.
  OperandRange operands = successorBranch.getOperands();
  if (successor->args_empty()) {
    successor = successorDest;
    successorOperands = operands;
    return success();
  }

  // Otherwise, we need to remap any argument operands.
  for (Value operand : operands) {
    BlockArgument argOperand = operand.dyn_cast<BlockArgument>();
    if (argOperand && argOperand.getOwner() == successor)
      argStorage.push_back(successorOperands[argOperand.getArgNumber()]);
    else
      argStorage.push_back(operand);
  }
  successor = successorDest;
  successorOperands = argStorage;
  return success();
}

/// Simplify a branch to a block that has a single predecessor. This effectively
/// merges the two blocks.
static LogicalResult
simplifyBrToBlockWithSinglePred(BranchOp op, PatternRewriter &rewriter) {
  // Check that the successor block has a single predecessor.
  Block *succ = op.getDest();
  Block *opParent = op->getBlock();
  if (succ == opParent || !llvm::hasSingleElement(succ->getPredecessors()))
    return failure();

  // Merge the successor into the current block and erase the branch.
  rewriter.mergeBlocks(succ, opParent, op.getOperands());
  rewriter.eraseOp(op);
  return success();
}

///   br ^bb1
/// ^bb1
///   br ^bbN(...)
///
///  -> br ^bbN(...)
///
static LogicalResult simplifyPassThroughBr(BranchOp op,
                                           PatternRewriter &rewriter) {
  Block *dest = op.getDest();
  ValueRange destOperands = op.getOperands();
  SmallVector<Value, 4> destOperandStorage;

  // Try to collapse the successor if it points somewhere other than this
  // block.
  if (dest == op->getBlock() ||
      failed(collapseBranch(dest, destOperands, destOperandStorage)))
    return failure();

  // Create a new branch with the collapsed successor.
  rewriter.replaceOpWithNewOp<BranchOp>(op, dest, destOperands);
  return success();
}

LogicalResult BranchOp::canonicalize(BranchOp op, PatternRewriter &rewriter) {
  return success(succeeded(simplifyBrToBlockWithSinglePred(op, rewriter)) ||
                 succeeded(simplifyPassThroughBr(op, rewriter)));
}

Block *BranchOp::getDest() { return getSuccessor(); }

void BranchOp::setDest(Block *block) { return setSuccessor(block); }

void BranchOp::eraseOperand(unsigned index) { (*this)->eraseOperand(index); }

Optional<MutableOperandRange>
BranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
}

Block *BranchOp::getSuccessorForOperands(ArrayRef<Attribute>) { return dest(); }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i))
      return emitOpError("result type mismatch");

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

/// Fold indirect calls that have a constant function as the callee operand.
LogicalResult CallIndirectOp::canonicalize(CallIndirectOp indirectCall,
                                           PatternRewriter &rewriter) {
  // Check that the callee is a constant callee.
  SymbolRefAttr calledFn;
  if (!matchPattern(indirectCall.getCallee(), m_Constant(&calledFn)))
    return failure();

  // Replace with a direct call.
  rewriter.replaceOpWithNewOp<CallOp>(indirectCall, calledFn,
                                      indirectCall.getResultTypes(),
                                      indirectCall.getArgOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

static void buildCmpIOp(OpBuilder &build, OperationState &result,
                        CmpIPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  result.addAttribute(CmpIOp::getPredicateAttrName(),
                      build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
// comparison predicates.
bool mlir::applyCmpPredicate(CmpIPredicate predicate, const APInt &lhs,
                             const APInt &rhs) {
  switch (predicate) {
  case CmpIPredicate::eq:
    return lhs.eq(rhs);
  case CmpIPredicate::ne:
    return lhs.ne(rhs);
  case CmpIPredicate::slt:
    return lhs.slt(rhs);
  case CmpIPredicate::sle:
    return lhs.sle(rhs);
  case CmpIPredicate::sgt:
    return lhs.sgt(rhs);
  case CmpIPredicate::sge:
    return lhs.sge(rhs);
  case CmpIPredicate::ult:
    return lhs.ult(rhs);
  case CmpIPredicate::ule:
    return lhs.ule(rhs);
  case CmpIPredicate::ugt:
    return lhs.ugt(rhs);
  case CmpIPredicate::uge:
    return lhs.uge(rhs);
  }
  llvm_unreachable("unknown comparison predicate");
}

// Returns true if the predicate is true for two equal operands.
static bool applyCmpPredicateToEqualOperands(CmpIPredicate predicate) {
  switch (predicate) {
  case CmpIPredicate::eq:
  case CmpIPredicate::sle:
  case CmpIPredicate::sge:
  case CmpIPredicate::ule:
  case CmpIPredicate::uge:
    return true;
  case CmpIPredicate::ne:
  case CmpIPredicate::slt:
  case CmpIPredicate::sgt:
  case CmpIPredicate::ult:
  case CmpIPredicate::ugt:
    return false;
  }
  llvm_unreachable("unknown comparison predicate");
}

// Constant folding hook for comparisons.
OpFoldResult CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two arguments");

  if (lhs() == rhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return BoolAttr::get(getContext(), val);
  }

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

static void buildCmpFOp(OpBuilder &build, OperationState &result,
                        CmpFPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  result.addAttribute(CmpFOp::getPredicateAttrName(),
                      build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool mlir::applyCmpPredicate(CmpFPredicate predicate, const APFloat &lhs,
                             const APFloat &rhs) {
  auto cmpResult = lhs.compare(rhs);
  switch (predicate) {
  case CmpFPredicate::AlwaysFalse:
    return false;
  case CmpFPredicate::OEQ:
    return cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::OGT:
    return cmpResult == APFloat::cmpGreaterThan;
  case CmpFPredicate::OGE:
    return cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::OLT:
    return cmpResult == APFloat::cmpLessThan;
  case CmpFPredicate::OLE:
    return cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::ONE:
    return cmpResult != APFloat::cmpUnordered && cmpResult != APFloat::cmpEqual;
  case CmpFPredicate::ORD:
    return cmpResult != APFloat::cmpUnordered;
  case CmpFPredicate::UEQ:
    return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::UGT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan;
  case CmpFPredicate::UGE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::ULT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan;
  case CmpFPredicate::ULE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case CmpFPredicate::UNE:
    return cmpResult != APFloat::cmpEqual;
  case CmpFPredicate::UNO:
    return cmpResult == APFloat::cmpUnordered;
  case CmpFPredicate::AlwaysTrue:
    return true;
  }
  llvm_unreachable("unknown comparison predicate");
}

// Constant folding hook for comparisons.
OpFoldResult CmpFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpf takes two arguments");

  auto lhs = operands.front().dyn_cast_or_null<FloatAttr>();
  auto rhs = operands.back().dyn_cast_or_null<FloatAttr>();

  // TODO: We could actually do some intelligent things if we know only one
  // of the operands, but it's inf or nan.
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return IntegerAttr::get(IntegerType::get(getContext(), 1), APInt(1, val));
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

namespace {
/// cond_br true, ^bb1, ^bb2
///  -> br ^bb1
/// cond_br false, ^bb1, ^bb2
///  -> br ^bb2
///
struct SimplifyConstCondBranchPred : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(condbr.getCondition(), m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getTrueDest(),
                                            condbr.getTrueOperands());
      return success();
    } else if (matchPattern(condbr.getCondition(), m_Zero())) {
      // False branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getFalseDest(),
                                            condbr.getFalseOperands());
      return success();
    }
    return failure();
  }
};

///   cond_br %cond, ^bb1, ^bb2
/// ^bb1
///   br ^bbN(...)
/// ^bb2
///   br ^bbK(...)
///
///  -> cond_br %cond, ^bbN(...), ^bbK(...)
///
struct SimplifyPassThroughCondBranch : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    Block *trueDest = condbr.trueDest(), *falseDest = condbr.falseDest();
    ValueRange trueDestOperands = condbr.getTrueOperands();
    ValueRange falseDestOperands = condbr.getFalseOperands();
    SmallVector<Value, 4> trueDestOperandStorage, falseDestOperandStorage;

    // Try to collapse one of the current successors.
    LogicalResult collapsedTrue =
        collapseBranch(trueDest, trueDestOperands, trueDestOperandStorage);
    LogicalResult collapsedFalse =
        collapseBranch(falseDest, falseDestOperands, falseDestOperandStorage);
    if (failed(collapsedTrue) && failed(collapsedFalse))
      return failure();

    // Create a new branch with the collapsed successors.
    rewriter.replaceOpWithNewOp<CondBranchOp>(condbr, condbr.getCondition(),
                                              trueDest, trueDestOperands,
                                              falseDest, falseDestOperands);
    return success();
  }
};

/// cond_br %cond, ^bb1(A, ..., N), ^bb1(A, ..., N)
///  -> br ^bb1(A, ..., N)
///
/// cond_br %cond, ^bb1(A), ^bb1(B)
///  -> %select = select %cond, A, B
///     br ^bb1(%select)
///
struct SimplifyCondBranchIdenticalSuccessors
    : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that the true and false destinations are the same and have the same
    // operands.
    Block *trueDest = condbr.trueDest();
    if (trueDest != condbr.falseDest())
      return failure();

    // If all of the operands match, no selects need to be generated.
    OperandRange trueOperands = condbr.getTrueOperands();
    OperandRange falseOperands = condbr.getFalseOperands();
    if (trueOperands == falseOperands) {
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, trueOperands);
      return success();
    }

    // Otherwise, if the current block is the only predecessor insert selects
    // for any mismatched branch operands.
    if (trueDest->getUniquePredecessor() != condbr->getBlock())
      return failure();

    // Generate a select for any operands that differ between the two.
    SmallVector<Value, 8> mergedOperands;
    mergedOperands.reserve(trueOperands.size());
    Value condition = condbr.getCondition();
    for (auto it : llvm::zip(trueOperands, falseOperands)) {
      if (std::get<0>(it) == std::get<1>(it))
        mergedOperands.push_back(std::get<0>(it));
      else
        mergedOperands.push_back(rewriter.create<SelectOp>(
            condbr.getLoc(), condition, std::get<0>(it), std::get<1>(it)));
    }

    rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, mergedOperands);
    return success();
  }
};

///   ...
///   cond_br %cond, ^bb1(...), ^bb2(...)
/// ...
/// ^bb1: // has single predecessor
///   ...
///   cond_br %cond, ^bb3(...), ^bb4(...)
///
/// ->
///
///   ...
///   cond_br %cond, ^bb1(...), ^bb2(...)
/// ...
/// ^bb1: // has single predecessor
///   ...
///   br ^bb3(...)
///
struct SimplifyCondBranchFromCondBranchOnSameCondition
    : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that we have a single distinct predecessor.
    Block *currentBlock = condbr->getBlock();
    Block *predecessor = currentBlock->getSinglePredecessor();
    if (!predecessor)
      return failure();

    // Check that the predecessor terminates with a conditional branch to this
    // block and that it branches on the same condition.
    auto predBranch = dyn_cast<CondBranchOp>(predecessor->getTerminator());
    if (!predBranch || condbr.getCondition() != predBranch.getCondition())
      return failure();

    // Fold this branch to an unconditional branch.
    if (currentBlock == predBranch.trueDest())
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.trueDest(),
                                            condbr.trueDestOperands());
    else
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.falseDest(),
                                            condbr.falseDestOperands());
    return success();
  }
};

///   cond_br %arg0, ^trueB, ^falseB
///
/// ^trueB:
///   "test.consumer1"(%arg0) : (i1) -> ()
///    ...
///
/// ^falseB:
///   "test.consumer2"(%arg0) : (i1) -> ()
///   ...
///
/// ->
///
///   cond_br %arg0, ^trueB, ^falseB
/// ^trueB:
///   "test.consumer1"(%true) : (i1) -> ()
///   ...
///
/// ^falseB:
///   "test.consumer2"(%false) : (i1) -> ()
///   ...
struct CondBranchTruthPropagation : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that we have a single distinct predecessor.
    bool replaced = false;
    Type ty = rewriter.getI1Type();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    // TODO These checks can be expanded to encompas any use with only
    // either the true of false edge as a predecessor. For now, we fall
    // back to checking the single predecessor is given by the true/fasle
    // destination, thereby ensuring that only that edge can reach the
    // op.
    if (condbr.getTrueDest()->getSinglePredecessor()) {
      for (OpOperand &use :
           llvm::make_early_inc_range(condbr.condition().getUses())) {
        if (use.getOwner()->getBlock() == condbr.getTrueDest()) {
          replaced = true;

          if (!constantTrue)
            constantTrue = rewriter.create<mlir::ConstantOp>(
                condbr.getLoc(), ty, rewriter.getBoolAttr(true));

          rewriter.updateRootInPlace(use.getOwner(),
                                     [&] { use.set(constantTrue); });
        }
      }
    }
    if (condbr.getFalseDest()->getSinglePredecessor()) {
      for (OpOperand &use :
           llvm::make_early_inc_range(condbr.condition().getUses())) {
        if (use.getOwner()->getBlock() == condbr.getFalseDest()) {
          replaced = true;

          if (!constantFalse)
            constantFalse = rewriter.create<mlir::ConstantOp>(
                condbr.getLoc(), ty, rewriter.getBoolAttr(false));

          rewriter.updateRootInPlace(use.getOwner(),
                                     [&] { use.set(constantFalse); });
        }
      }
    }
    return success(replaced);
  }
};
} // end anonymous namespace

void CondBranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<SimplifyConstCondBranchPred, SimplifyPassThroughCondBranch,
              SimplifyCondBranchIdenticalSuccessors,
              SimplifyCondBranchFromCondBranchOnSameCondition,
              CondBranchTruthPropagation>(context);
}

Optional<MutableOperandRange>
CondBranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? trueDestOperandsMutable()
                            : falseDestOperandsMutable();
}

Block *CondBranchOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    return condAttr.getValue().isOneValue() ? trueDest() : falseDest();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Constant*Op
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ConstantOp &op) {
  p << "constant ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1)
    p << ' ';
  p << op.getValue();

  // If the value is a symbol reference or Array, print a trailing type.
  if (op.getValue().isa<SymbolRefAttr, ArrayAttr>())
    p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // If the attribute is a symbol reference or array, then we expect a trailing
  // type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr, ArrayAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
static LogicalResult verify(ConstantOp &op) {
  auto value = op.getValue();
  if (!value)
    return op.emitOpError("requires a 'value' attribute");

  Type type = op.getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";

  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    if (type.isa<IndexType>() || value.isa<BoolAttr>())
      return success();
    IntegerType intType = type.cast<IntegerType>();
    if (!intType.isSignless())
      return op.emitOpError("requires integer result types to be signless");

    // If the type has a known bitwidth we verify that the value can be
    // represented with the given bitwidth.
    unsigned bitwidth = intType.getWidth();
    APInt intVal = intAttr.getValue();
    if (!intVal.isSignedIntN(bitwidth) && !intVal.isIntN(bitwidth))
      return op.emitOpError("requires 'value' to be an integer within the "
                            "range of the integer result type");
    return success();
  }

  if (auto complexTy = type.dyn_cast<ComplexType>()) {
    auto arrayAttr = value.dyn_cast<ArrayAttr>();
    if (!complexTy || arrayAttr.size() != 2)
      return op.emitOpError(
          "requires 'value' to be a complex constant, represented as array of "
          "two values");
    auto complexEltTy = complexTy.getElementType();
    if (complexEltTy != arrayAttr[0].getType() ||
        complexEltTy != arrayAttr[1].getType()) {
      return op.emitOpError()
             << "requires attribute's element types (" << arrayAttr[0].getType()
             << ", " << arrayAttr[1].getType()
             << ") to match the element type of the op's return type ("
             << complexEltTy << ")";
    }
    return success();
  }

  if (type.isa<FloatType>()) {
    if (!value.isa<FloatAttr>())
      return op.emitOpError("requires 'value' to be a floating point constant");
    return success();
  }

  if (type.isa<ShapedType>()) {
    if (!value.isa<ElementsAttr>())
      return op.emitOpError("requires 'value' to be a shaped constant");
    return success();
  }

  if (type.isa<FunctionType>()) {
    auto fnAttr = value.dyn_cast<FlatSymbolRefAttr>();
    if (!fnAttr)
      return op.emitOpError("requires 'value' to be a function reference");

    // Try to find the referenced function.
    auto fn =
        op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
    if (!fn)
      return op.emitOpError()
             << "reference to undefined function '" << fnAttr.getValue() << "'";

    // Check that the referenced function has the correct type.
    if (fn.getType() != type)
      return op.emitOpError("reference to function with mismatched type");

    return success();
  }

  if (type.isa<NoneType>() && value.isa<UnitAttr>())
    return success();

  return op.emitOpError("unsupported 'value' attribute: ") << value;
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  Type type = getType();
  if (auto intCst = getValue().dyn_cast<IntegerAttr>()) {
    IntegerType intTy = type.dyn_cast<IntegerType>();

    // Sugar i1 constants with 'true' and 'false'.
    if (intTy && intTy.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getInt();
    if (intTy)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());

  } else if (type.isa<FunctionType>()) {
    setNameFn(getResult(), "f");
  } else {
    setNameFn(getResult(), "cst");
  }
}

/// Returns true if a constant operation can be built with the given value and
/// result type.
bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  // SymbolRefAttr can only be used with a function type.
  if (value.isa<SymbolRefAttr>())
    return type.isa<FunctionType>();
  // The attribute must have the same type as 'type'.
  if (!value.getType().isa<NoneType>() && value.getType() != type)
    return false;
  // If the type is an integer type, it must be signless.
  if (IntegerType integerTy = type.dyn_cast<IntegerType>())
    if (!integerTy.isSignless())
      return false;
  // Finally, check that the attribute kind is handled.
  if (auto arrAttr = value.dyn_cast<ArrayAttr>()) {
    auto complexTy = type.dyn_cast<ComplexType>();
    if (!complexTy)
      return false;
    auto complexEltTy = complexTy.getElementType();
    return arrAttr.size() == 2 && arrAttr[0].getType() == complexEltTy &&
           arrAttr[1].getType() == complexEltTy;
  }
  return value.isa<IntegerAttr, FloatAttr, ElementsAttr, UnitAttr>();
}

void ConstantFloatOp::build(OpBuilder &builder, OperationState &result,
                            const APFloat &value, FloatType type) {
  ConstantOp::build(builder, result, type, builder.getFloatAttr(type, value));
}

bool ConstantFloatOp::classof(Operation *op) {
  return ConstantOp::classof(op) && op->getResult(0).getType().isa<FloatType>();
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::classof(Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isSignlessInteger();
}

void ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                          int64_t value, unsigned width) {
  Type type = builder.getIntegerType(width);
  ConstantOp::build(builder, result, type, builder.getIntegerAttr(type, value));
}

/// Build a constant int op producing an integer with the specified type,
/// which must be an integer type.
void ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                          int64_t value, Type type) {
  assert(type.isSignlessInteger() &&
         "ConstantIntOp can only have signless integer type");
  ConstantOp::build(builder, result, type, builder.getIntegerAttr(type, value));
}

/// ConstantIndexOp only matches values whose result type is Index.
bool ConstantIndexOp::classof(Operation *op) {
  return ConstantOp::classof(op) && op->getResult(0).getType().isIndex();
}

void ConstantIndexOp::build(OpBuilder &builder, OperationState &result,
                            int64_t value) {
  Type type = builder.getIndexType();
  ConstantOp::build(builder, result, type, builder.getIntegerAttr(type, value));
}

// ---------------------------------------------------------------------------
// DivFOp
// ---------------------------------------------------------------------------

OpFoldResult DivFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a / b; });
}

//===----------------------------------------------------------------------===//
// FPExtOp
//===----------------------------------------------------------------------===//

bool FPExtOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() < fb.getWidth();
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

bool FPToSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (a.isa<FloatType>() && b.isSignlessInteger())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

bool FPToUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (a.isa<FloatType>() && b.isSignlessInteger())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// FPTruncOp
//===----------------------------------------------------------------------===//

bool FPTruncOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() > fb.getWidth();
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast is applicable from index to integer and backwards.
bool IndexCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (a.isa<ShapedType>() && b.isa<ShapedType>()) {
    auto aShaped = a.cast<ShapedType>();
    auto bShaped = b.cast<ShapedType>();

    return (aShaped.getShape() == bShaped.getShape()) &&
           areCastCompatible(aShaped.getElementType(),
                             bShaped.getElementType());
  }

  return (a.isIndex() && b.isSignlessInteger()) ||
         (a.isSignlessInteger() && b.isIndex());
}

OpFoldResult IndexCastOp::fold(ArrayRef<Attribute> cstOperands) {
  // Fold IndexCast(IndexCast(x)) -> x
  auto cast = getOperand().getDefiningOp<IndexCastOp>();
  if (cast && cast.getOperand().getType() == getType())
    return cast.getOperand();

  // Fold IndexCast(constant) -> constant
  // A little hack because we go through int.  Otherwise, the size
  // of the constant might need to change.
  if (auto value = cstOperands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(getType(), value.getInt());

  return {};
}

namespace {
///  index_cast(sign_extend x) => index_cast(x)
struct IndexCastOfSExt : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {

    if (auto extop = op.getOperand().getDefiningOp<SignExtendIOp>()) {
      op.setOperand(extop.getOperand());
      return success();
    }
    return failure();
  }
};

} // namespace

void IndexCastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<IndexCastOfSExt>(context);
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

OpFoldResult MulFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

OpFoldResult MulIOp::fold(ArrayRef<Attribute> operands) {
  /// muli(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// muli(x, 1) -> x
  if (matchPattern(rhs(), m_One()))
    return getOperand(0);

  // TODO: Handle the overflow case.
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  /// or(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// or(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

OpFoldResult RankOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold rank when the rank of the operand is known.
  auto type = getOperand().getType();
  if (auto shapedType = type.dyn_cast<ShapedType>())
    if (shapedType.hasRank())
      return IntegerAttr::get(IndexType::get(getContext()),
                              shapedType.getRank());
  return IntegerAttr();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReturnOp op) {
  auto function = cast<FuncOp>(op->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")"
             << " in function @" << function.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Transforms a select to a not, where relevant.
//
//  select %arg, %false, %true
//
//  becomes
//
//  xor %arg, %true
struct SelectToNot : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchPattern(op.getTrueValue(), m_Zero()))
      return failure();

    if (!matchPattern(op.getFalseValue(), m_One()))
      return failure();

    if (!op.getType().isInteger(1))
      return failure();

    rewriter.replaceOpWithNewOp<XOrOp>(op, op.condition(), op.getFalseValue());
    return success();
  }
};

void SelectOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SelectToNot>(context);
}

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  auto trueVal = getTrueValue();
  auto falseVal = getFalseValue();
  if (trueVal == falseVal)
    return trueVal;

  auto condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(condition, m_One()))
    return trueVal;

  // select false, %0, %1 => %1
  if (matchPattern(condition, m_Zero()))
    return falseVal;

  if (auto cmp = dyn_cast_or_null<CmpIOp>(condition.getDefiningOp())) {
    auto pred = cmp.predicate();
    if (pred == mlir::CmpIPredicate::eq || pred == mlir::CmpIPredicate::ne) {
      auto cmpLhs = cmp.lhs();
      auto cmpRhs = cmp.rhs();

      // %0 = cmpi eq, %arg0, %arg1
      // %1 = select %0, %arg0, %arg1 => %arg1

      // %0 = cmpi ne, %arg0, %arg1
      // %1 = select %0, %arg0, %arg1 => %arg0

      if ((cmpLhs == trueVal && cmpRhs == falseVal) ||
          (cmpRhs == trueVal && cmpLhs == falseVal))
        return pred == mlir::CmpIPredicate::ne ? trueVal : falseVal;
    }
  }
  return nullptr;
}

static void print(OpAsmPrinter &p, SelectOp op) {
  p << "select " << op.getOperands();
  p.printOptionalAttrDict(op->getAttrs());
  p << " : ";
  if (ShapedType condType = op.getCondition().getType().dyn_cast<ShapedType>())
    p << condType << ", ";
  p << op.getType();
}

static ParseResult parseSelectOp(OpAsmParser &parser, OperationState &result) {
  Type conditionType, resultType;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();

  // Check for the explicit condition type if this is a masked tensor or vector.
  if (succeeded(parser.parseOptionalComma())) {
    conditionType = resultType;
    if (parser.parseType(resultType))
      return failure();
  } else {
    conditionType = parser.getBuilder().getI1Type();
  }

  result.addTypes(resultType);
  return parser.resolveOperands(operands,
                                {conditionType, resultType, resultType},
                                parser.getNameLoc(), result.operands);
}

static LogicalResult verify(SelectOp op) {
  Type conditionType = op.getCondition().getType();
  if (conditionType.isSignlessInteger(1))
    return success();

  // If the result type is a vector or tensor, the type can be a mask with the
  // same elements.
  Type resultType = op.getType();
  if (!resultType.isa<TensorType, VectorType>())
    return op.emitOpError()
           << "expected condition to be a signless i1, but got "
           << conditionType;
  Type shapedConditionType = getI1SameShape(resultType);
  if (conditionType != shapedConditionType)
    return op.emitOpError()
           << "expected condition type to have the same shape "
              "as the result type, expected "
           << shapedConditionType << ", but got " << conditionType;
  return success();
}

//===----------------------------------------------------------------------===//
// SignExtendIOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(SignExtendIOp op) {
  // Get the scalar type (which is either directly the type of the operand
  // or the vector's/tensor's element type.
  auto srcType = getElementTypeOrSelf(op.getOperand().getType());
  auto dstType = getElementTypeOrSelf(op.getType());

  // For now, index is forbidden for the source and the destination type.
  if (srcType.isa<IndexType>())
    return op.emitError() << srcType << " is not a valid operand type";
  if (dstType.isa<IndexType>())
    return op.emitError() << dstType << " is not a valid result type";

  if (srcType.cast<IntegerType>().getWidth() >=
      dstType.cast<IntegerType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

OpFoldResult SignExtendIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary operation takes one operand");

  if (!operands[0])
    return {};

  if (auto lhs = operands[0].dyn_cast<IntegerAttr>()) {
    return IntegerAttr::get(
        getType(), lhs.getValue().sext(getType().getIntOrFloatBitWidth()));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// SignedDivIOp
//===----------------------------------------------------------------------===//

OpFoldResult SignedDivIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    return a.sdiv_ov(b, overflowOrDiv0);
  });

  // Fold out division by one. Assumes all tensors of all ones are splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// SignedFloorDivIOp
//===----------------------------------------------------------------------===//

static APInt signedCeilNonnegInputs(APInt a, APInt b, bool &overflow) {
  // Returns (a-1)/b + 1
  APInt one(a.getBitWidth(), 1, true); // Signed value 1.
  APInt val = a.ssub_ov(one, overflow).sdiv_ov(b, overflow);
  return val.sadd_ov(one, overflow);
}

OpFoldResult SignedFloorDivIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    unsigned bits = a.getBitWidth();
    APInt zero = APInt::getNullValue(bits);
    if (a.sge(zero) && b.sgt(zero)) {
      // Both positive (or a is zero), return a / b.
      return a.sdiv_ov(b, overflowOrDiv0);
    } else if (a.sle(zero) && b.slt(zero)) {
      // Both negative (or a is zero), return -a / -b.
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt posB = zero.ssub_ov(b, overflowOrDiv0);
      return posA.sdiv_ov(posB, overflowOrDiv0);
    } else if (a.slt(zero) && b.sgt(zero)) {
      // A is negative, b is positive, return - ceil(-a, b).
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt ceil = signedCeilNonnegInputs(posA, b, overflowOrDiv0);
      return zero.ssub_ov(ceil, overflowOrDiv0);
    } else {
      // A is positive, b is negative, return - ceil(a, -b).
      APInt posB = zero.ssub_ov(b, overflowOrDiv0);
      APInt ceil = signedCeilNonnegInputs(a, posB, overflowOrDiv0);
      return zero.ssub_ov(ceil, overflowOrDiv0);
    }
  });

  // Fold out floor division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// SignedCeilDivIOp
//===----------------------------------------------------------------------===//

OpFoldResult SignedCeilDivIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    unsigned bits = a.getBitWidth();
    APInt zero = APInt::getNullValue(bits);
    if (a.sgt(zero) && b.sgt(zero)) {
      // Both positive, return ceil(a, b).
      return signedCeilNonnegInputs(a, b, overflowOrDiv0);
    } else if (a.slt(zero) && b.slt(zero)) {
      // Both negative, return ceil(-a, -b).
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt posB = zero.ssub_ov(b, overflowOrDiv0);
      return signedCeilNonnegInputs(posA, posB, overflowOrDiv0);
    } else if (a.slt(zero) && b.sgt(zero)) {
      // A is negative, b is positive, return - ( -a / b).
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt div = posA.sdiv_ov(b, overflowOrDiv0);
      return zero.ssub_ov(div, overflowOrDiv0);
    } else {
      // A is positive (or zero), b is negative, return - (a / -b).
      APInt posB = zero.ssub_ov(b, overflowOrDiv0);
      APInt div = a.sdiv_ov(posB, overflowOrDiv0);
      return zero.ssub_ov(div, overflowOrDiv0);
    }
  });

  // Fold out floor division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// SignedRemIOp
//===----------------------------------------------------------------------===//

OpFoldResult SignedRemIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "remi_signed takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().srem(rhsValue));
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

// sitofp is applicable from integer types to float types.
bool SIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (a.isSignlessInteger() && b.isa<FloatType>())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(SplatOp op) {
  // TODO: we could replace this by a trait.
  if (op.getOperand().getType() !=
      op.getType().cast<ShapedType>().getElementType())
    return op.emitError("operand should be of elemental type of result type");

  return success();
}

// Constant folding hook for SplatOp.
OpFoldResult SplatOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "splat takes one operand");

  auto constOperand = operands.front();
  if (!constOperand || !constOperand.isa<IntegerAttr, FloatAttr>())
    return {};

  auto shapedType = getType().cast<ShapedType>();
  assert(shapedType.getElementType() == constOperand.getType() &&
         "incorrect input attribute type for folding");

  // SplatElementsAttr::get treats single value for second arg as being a splat.
  return SplatElementsAttr::get(shapedType, {constOperand});
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

OpFoldResult SubFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

OpFoldResult SubIOp::fold(ArrayRef<Attribute> operands) {
  // subi(x,x) -> 0
  if (getOperand(0) == getOperand(1))
    return Builder(getContext()).getZeroAttr(getType());
  // subi(x,0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

/// Canonicalize a sub of a constant and (constant +/- something) to simply be
/// a single operation that merges the two constants.
struct SubConstantReorder : public OpRewritePattern<SubIOp> {
  using OpRewritePattern<SubIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIOp subOp,
                                PatternRewriter &rewriter) const override {
    APInt origConst;
    APInt midConst;

    if (matchPattern(subOp.getOperand(0), m_ConstantInt(&origConst))) {
      if (auto midAddOp = subOp.getOperand(1).getDefiningOp<AddIOp>()) {
        // origConst - (midConst + something) == (origConst - midConst) -
        // something
        for (int j = 0; j < 2; j++) {
          if (matchPattern(midAddOp.getOperand(j), m_ConstantInt(&midConst))) {
            auto nextConstant = rewriter.create<ConstantOp>(
                subOp.getLoc(),
                rewriter.getIntegerAttr(subOp.getType(), origConst - midConst));
            rewriter.replaceOpWithNewOp<SubIOp>(subOp, nextConstant,
                                                midAddOp.getOperand(1 - j));
            return success();
          }
        }
      }

      if (auto midSubOp = subOp.getOperand(0).getDefiningOp<SubIOp>()) {
        if (matchPattern(midSubOp.getOperand(0), m_ConstantInt(&midConst))) {
          // (midConst - something) - origConst == (midConst - origConst) -
          // something
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), midConst - origConst));
          rewriter.replaceOpWithNewOp<SubIOp>(subOp, nextConstant,
                                              midSubOp.getOperand(1));
          return success();
        }

        if (matchPattern(midSubOp.getOperand(1), m_ConstantInt(&midConst))) {
          // (something - midConst) - origConst == something - (origConst +
          // midConst)
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), origConst + midConst));
          rewriter.replaceOpWithNewOp<SubIOp>(subOp, midSubOp.getOperand(0),
                                              nextConstant);
          return success();
        }
      }

      if (auto midSubOp = subOp.getOperand(1).getDefiningOp<SubIOp>()) {
        if (matchPattern(midSubOp.getOperand(0), m_ConstantInt(&midConst))) {
          // origConst - (midConst - something) == (origConst - midConst) +
          // something
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), origConst - midConst));
          rewriter.replaceOpWithNewOp<AddIOp>(subOp, nextConstant,
                                              midSubOp.getOperand(1));
          return success();
        }

        if (matchPattern(midSubOp.getOperand(1), m_ConstantInt(&midConst))) {
          // origConst - (something - midConst) == (origConst + midConst) -
          // something
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), origConst + midConst));
          rewriter.replaceOpWithNewOp<SubIOp>(subOp, nextConstant,
                                              midSubOp.getOperand(0));
          return success();
        }
      }
    }

    if (matchPattern(subOp.getOperand(1), m_ConstantInt(&origConst))) {
      if (auto midAddOp = subOp.getOperand(0).getDefiningOp<AddIOp>()) {
        // (midConst + something) - origConst == (midConst - origConst) +
        // something
        for (int j = 0; j < 2; j++) {
          if (matchPattern(midAddOp.getOperand(j), m_ConstantInt(&midConst))) {
            auto nextConstant = rewriter.create<ConstantOp>(
                subOp.getLoc(),
                rewriter.getIntegerAttr(subOp.getType(), midConst - origConst));
            rewriter.replaceOpWithNewOp<AddIOp>(subOp, nextConstant,
                                                midAddOp.getOperand(1 - j));
            return success();
          }
        }
      }

      if (auto midSubOp = subOp.getOperand(0).getDefiningOp<SubIOp>()) {
        if (matchPattern(midSubOp.getOperand(0), m_ConstantInt(&midConst))) {
          // (midConst - something) - origConst == (midConst - origConst) -
          // something
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), midConst - origConst));
          rewriter.replaceOpWithNewOp<SubIOp>(subOp, nextConstant,
                                              midSubOp.getOperand(1));
          return success();
        }

        if (matchPattern(midSubOp.getOperand(1), m_ConstantInt(&midConst))) {
          // (something - midConst) - origConst == something - (midConst +
          // origConst)
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), midConst + origConst));
          rewriter.replaceOpWithNewOp<SubIOp>(subOp, midSubOp.getOperand(0),
                                              nextConstant);
          return success();
        }
      }

      if (auto midSubOp = subOp.getOperand(1).getDefiningOp<SubIOp>()) {
        if (matchPattern(midSubOp.getOperand(0), m_ConstantInt(&midConst))) {
          // origConst - (midConst - something) == (origConst - midConst) +
          // something
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), origConst - midConst));
          rewriter.replaceOpWithNewOp<AddIOp>(subOp, nextConstant,
                                              midSubOp.getOperand(1));
          return success();
        }
        if (matchPattern(midSubOp.getOperand(1), m_ConstantInt(&midConst))) {
          // origConst - (something - midConst) == (origConst - midConst) -
          // something
          auto nextConstant = rewriter.create<ConstantOp>(
              subOp.getLoc(),
              rewriter.getIntegerAttr(subOp.getType(), origConst - midConst));
          rewriter.replaceOpWithNewOp<SubIOp>(subOp, nextConstant,
                                              midSubOp.getOperand(0));
          return success();
        }
      }
    }
    return failure();
  }
};

void SubIOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<SubConstantReorder>(context);
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

// uitofp is applicable from integer types to float types.
bool UIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (a.isSignlessInteger() && b.isa<FloatType>())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     DenseIntElementsAttr caseValues,
                     BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands) {
  SmallVector<Value> flattenedCaseOperands;
  SmallVector<int32_t> caseOperandOffsets;
  int32_t offset = 0;
  for (ValueRange operands : caseOperands) {
    flattenedCaseOperands.append(operands.begin(), operands.end());
    caseOperandOffsets.push_back(offset);
    offset += operands.size();
  }
  DenseIntElementsAttr caseOperandOffsetsAttr;
  if (!caseOperandOffsets.empty())
    caseOperandOffsetsAttr = builder.getI32VectorAttr(caseOperandOffsets);

  build(builder, result, value, defaultOperands, flattenedCaseOperands,
        caseValues, caseOperandOffsetsAttr, defaultDestination,
        caseDestinations);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<APInt> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }
  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands);
}

/// <cases> ::= `default` `:` bb-id (`(` ssa-use-and-type-list `)`)?
///             ( `,` integer `:` bb-id (`(` ssa-use-and-type-list `)`)? )*
static ParseResult
parseSwitchOpCases(OpAsmParser &parser, Type &flagType,
                   Block *&defaultDestination,
                   SmallVectorImpl<OpAsmParser::OperandType> &defaultOperands,
                   SmallVectorImpl<Type> &defaultOperandTypes,
                   DenseIntElementsAttr &caseValues,
                   SmallVectorImpl<Block *> &caseDestinations,
                   SmallVectorImpl<OpAsmParser::OperandType> &caseOperands,
                   SmallVectorImpl<Type> &caseOperandTypes,
                   DenseIntElementsAttr &caseOperandOffsets) {
  if (failed(parser.parseKeyword("default")) || failed(parser.parseColon()) ||
      failed(parser.parseSuccessor(defaultDestination)))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseRegionArgumentList(defaultOperands)) ||
        failed(parser.parseColonTypeList(defaultOperandTypes)) ||
        failed(parser.parseRParen()))
      return failure();
  }

  SmallVector<APInt> values;
  SmallVector<int32_t> offsets;
  unsigned bitWidth = flagType.getIntOrFloatBitWidth();
  int64_t offset = 0;
  while (succeeded(parser.parseOptionalComma())) {
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();
    values.push_back(APInt(bitWidth, value));

    Block *destination;
    SmallVector<OpAsmParser::OperandType> operands;
    if (failed(parser.parseColon()) ||
        failed(parser.parseSuccessor(destination)))
      return failure();
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseRegionArgumentList(operands)) ||
          failed(parser.parseColonTypeList(caseOperandTypes)) ||
          failed(parser.parseRParen()))
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.append(operands.begin(), operands.end());
    offsets.push_back(offset);
    offset += operands.size();
  }

  if (values.empty())
    return success();

  Builder &builder = parser.getBuilder();
  ShapedType caseValueType =
      VectorType::get(static_cast<int64_t>(values.size()), flagType);
  caseValues = DenseIntElementsAttr::get(caseValueType, values);
  caseOperandOffsets = builder.getI32VectorAttr(offsets);

  return success();
}

static void printSwitchOpCases(
    OpAsmPrinter &p, SwitchOp op, Type flagType, Block *defaultDestination,
    OperandRange defaultOperands, TypeRange defaultOperandTypes,
    DenseIntElementsAttr caseValues, SuccessorRange caseDestinations,
    OperandRange caseOperands, TypeRange caseOperandTypes,
    ElementsAttr caseOperandOffsets) {
  p << "  default: ";
  p.printSuccessorAndUseList(defaultDestination, defaultOperands);

  if (!caseValues)
    return;

  for (int64_t i = 0, size = caseValues.size(); i < size; ++i) {
    p << ',';
    p.printNewline();
    p << "  ";
    p << caseValues.getValue<APInt>(i).getLimitedValue();
    p << ": ";
    p.printSuccessorAndUseList(caseDestinations[i], op.getCaseOperands(i));
  }
  p.printNewline();
}

static LogicalResult verify(SwitchOp op) {
  auto caseValues = op.case_values();
  auto caseDestinations = op.caseDestinations();

  if (!caseValues && caseDestinations.empty())
    return success();

  Type flagType = op.flag().getType();
  Type caseValueType = caseValues->getType().getElementType();
  if (caseValueType != flagType)
    return op.emitOpError()
           << "'flag' type (" << flagType << ") should match case value type ("
           << caseValueType << ")";

  if (caseValues &&
      caseValues->size() != static_cast<int64_t>(caseDestinations.size()))
    return op.emitOpError() << "number of case values (" << caseValues->size()
                            << ") should match number of "
                               "case destinations ("
                            << caseDestinations.size() << ")";
  return success();
}

OperandRange SwitchOp::getCaseOperands(unsigned index) {
  return getCaseOperandsMutable(index);
}

MutableOperandRange SwitchOp::getCaseOperandsMutable(unsigned index) {
  MutableOperandRange caseOperands = caseOperandsMutable();
  if (!case_operand_offsets()) {
    assert(caseOperands.size() == 0 &&
           "non-empty case operands must have offsets");
    return caseOperands;
  }

  ElementsAttr offsets = case_operand_offsets().getValue();
  assert(index < offsets.size() && "invalid case operand offset index");

  int64_t begin = offsets.getValue(index).cast<IntegerAttr>().getInt();
  int64_t end = index + 1 == offsets.size()
                    ? caseOperands.size()
                    : offsets.getValue(index + 1).cast<IntegerAttr>().getInt();
  return caseOperandsMutable().slice(begin, end - begin);
}

Optional<MutableOperandRange>
SwitchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == 0 ? defaultOperandsMutable()
                    : getCaseOperandsMutable(index - 1);
}

Block *SwitchOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  Optional<DenseIntElementsAttr> caseValues = case_values();

  if (!caseValues)
    return defaultDestination();

  SuccessorRange caseDests = caseDestinations();
  if (auto value = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    for (int64_t i = 0, size = case_values()->size(); i < size; ++i)
      if (value == caseValues->getValue<IntegerAttr>(i))
        return caseDests[i];
    return defaultDestination();
  }
  return nullptr;
}

/// switch %flag : i32, [
///   default:  ^bb1
/// ]
///  -> br ^bb1
static LogicalResult simplifySwitchWithOnlyDefault(SwitchOp op,
                                                   PatternRewriter &rewriter) {
  if (!op.caseDestinations().empty())
    return failure();

  rewriter.replaceOpWithNewOp<BranchOp>(op, op.defaultDestination(),
                                        op.defaultOperands());
  return success();
}

/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb1,
///   43: ^bb2
/// ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   43: ^bb2
/// ]
static LogicalResult
dropSwitchCasesThatMatchDefault(SwitchOp op, PatternRewriter &rewriter) {
  SmallVector<Block *> newCaseDestinations;
  SmallVector<ValueRange> newCaseOperands;
  SmallVector<APInt> newCaseValues;
  bool requiresChange = false;
  auto caseValues = op.case_values();
  auto caseDests = op.caseDestinations();

  for (int64_t i = 0, size = caseValues->size(); i < size; ++i) {
    if (caseDests[i] == op.defaultDestination() &&
        op.getCaseOperands(i) == op.defaultOperands()) {
      requiresChange = true;
      continue;
    }
    newCaseDestinations.push_back(caseDests[i]);
    newCaseOperands.push_back(op.getCaseOperands(i));
    newCaseValues.push_back(caseValues->getValue<APInt>(i));
  }

  if (!requiresChange)
    return failure();

  rewriter.replaceOpWithNewOp<SwitchOp>(op, op.flag(), op.defaultDestination(),
                                        op.defaultOperands(), newCaseValues,
                                        newCaseDestinations, newCaseOperands);
  return success();
}

/// Helper for folding a switch with a constant value.
/// switch %c_42 : i32, [
///   default: ^bb1 ,
///   42: ^bb2,
///   43: ^bb3
/// ]
/// -> br ^bb2
static void foldSwitch(SwitchOp op, PatternRewriter &rewriter,
                       APInt caseValue) {
  auto caseValues = op.case_values();
  for (int64_t i = 0, size = caseValues->size(); i < size; ++i) {
    if (caseValues->getValue<APInt>(i) == caseValue) {
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.caseDestinations()[i],
                                            op.getCaseOperands(i));
      return;
    }
  }
  rewriter.replaceOpWithNewOp<BranchOp>(op, op.defaultDestination(),
                                        op.defaultOperands());
}

/// switch %c_42 : i32, [
///   default: ^bb1,
///   42: ^bb2,
///   43: ^bb3
/// ]
/// -> br ^bb2
static LogicalResult simplifyConstSwitchValue(SwitchOp op,
                                              PatternRewriter &rewriter) {
  APInt caseValue;
  if (!matchPattern(op.flag(), m_ConstantInt(&caseValue)))
    return failure();

  foldSwitch(op, rewriter, caseValue);
  return success();
}

/// switch %c_42 : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   br ^bb3
/// ->
/// switch %c_42 : i32, [
///   default: ^bb1,
///   42: ^bb3,
/// ]
static LogicalResult simplifyPassThroughSwitch(SwitchOp op,
                                               PatternRewriter &rewriter) {
  SmallVector<Block *> newCaseDests;
  SmallVector<ValueRange> newCaseOperands;
  SmallVector<SmallVector<Value>> argStorage;
  auto caseValues = op.case_values();
  auto caseDests = op.caseDestinations();
  bool requiresChange = false;
  for (int64_t i = 0, size = caseValues->size(); i < size; ++i) {
    Block *caseDest = caseDests[i];
    ValueRange caseOperands = op.getCaseOperands(i);
    argStorage.emplace_back();
    if (succeeded(collapseBranch(caseDest, caseOperands, argStorage.back())))
      requiresChange = true;

    newCaseDests.push_back(caseDest);
    newCaseOperands.push_back(caseOperands);
  }

  Block *defaultDest = op.defaultDestination();
  ValueRange defaultOperands = op.defaultOperands();
  argStorage.emplace_back();

  if (succeeded(
          collapseBranch(defaultDest, defaultOperands, argStorage.back())))
    requiresChange = true;

  if (!requiresChange)
    return failure();

  rewriter.replaceOpWithNewOp<SwitchOp>(op, op.flag(), defaultDest,
                                        defaultOperands, caseValues.getValue(),
                                        newCaseDests, newCaseOperands);
  return success();
}

/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   switch %flag : i32, [
///     default: ^bb3,
///     42: ^bb4
///   ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   br ^bb4
///
///  and
///
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   switch %flag : i32, [
///     default: ^bb3,
///     43: ^bb4
///   ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   br ^bb3
static LogicalResult
simplifySwitchFromSwitchOnSameCondition(SwitchOp op,
                                        PatternRewriter &rewriter) {
  // Check that we have a single distinct predecessor.
  Block *currentBlock = op->getBlock();
  Block *predecessor = currentBlock->getSinglePredecessor();
  if (!predecessor)
    return failure();

  // Check that the predecessor terminates with a switch branch to this block
  // and that it branches on the same condition and that this branch isn't the
  // default destination.
  auto predSwitch = dyn_cast<SwitchOp>(predecessor->getTerminator());
  if (!predSwitch || op.flag() != predSwitch.flag() ||
      predSwitch.defaultDestination() == currentBlock)
    return failure();

  // Fold this switch to an unconditional branch.
  APInt caseValue;
  bool isDefault = true;
  SuccessorRange predDests = predSwitch.caseDestinations();
  Optional<DenseIntElementsAttr> predCaseValues = predSwitch.case_values();
  for (int64_t i = 0, size = predCaseValues->size(); i < size; ++i) {
    if (currentBlock == predDests[i]) {
      caseValue = predCaseValues->getValue<APInt>(i);
      isDefault = false;
      break;
    }
  }
  if (isDefault)
    rewriter.replaceOpWithNewOp<BranchOp>(op, op.defaultDestination(),
                                          op.defaultOperands());
  else
    foldSwitch(op, rewriter, caseValue);
  return success();
}

/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2
/// ]
/// ^bb1:
///   switch %flag : i32, [
///     default: ^bb3,
///     42: ^bb4,
///     43: ^bb5
///   ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb1:
///   switch %flag : i32, [
///     default: ^bb3,
///     43: ^bb5
///   ]
static LogicalResult
simplifySwitchFromDefaultSwitchOnSameCondition(SwitchOp op,
                                               PatternRewriter &rewriter) {
  // Check that we have a single distinct predecessor.
  Block *currentBlock = op->getBlock();
  Block *predecessor = currentBlock->getSinglePredecessor();
  if (!predecessor)
    return failure();

  // Check that the predecessor terminates with a switch branch to this block
  // and that it branches on the same condition and that this branch is the
  // default destination.
  auto predSwitch = dyn_cast<SwitchOp>(predecessor->getTerminator());
  if (!predSwitch || op.flag() != predSwitch.flag() ||
      predSwitch.defaultDestination() != currentBlock)
    return failure();

  // Delete case values that are not possible here.
  DenseSet<APInt> caseValuesToRemove;
  auto predDests = predSwitch.caseDestinations();
  auto predCaseValues = predSwitch.case_values();
  for (int64_t i = 0, size = predCaseValues->size(); i < size; ++i)
    if (currentBlock != predDests[i])
      caseValuesToRemove.insert(predCaseValues->getValue<APInt>(i));

  SmallVector<Block *> newCaseDestinations;
  SmallVector<ValueRange> newCaseOperands;
  SmallVector<APInt> newCaseValues;
  bool requiresChange = false;

  auto caseValues = op.case_values();
  auto caseDests = op.caseDestinations();
  for (int64_t i = 0, size = caseValues->size(); i < size; ++i) {
    if (caseValuesToRemove.contains(caseValues->getValue<APInt>(i))) {
      requiresChange = true;
      continue;
    }
    newCaseDestinations.push_back(caseDests[i]);
    newCaseOperands.push_back(op.getCaseOperands(i));
    newCaseValues.push_back(caseValues->getValue<APInt>(i));
  }

  if (!requiresChange)
    return failure();

  rewriter.replaceOpWithNewOp<SwitchOp>(op, op.flag(), op.defaultDestination(),
                                        op.defaultOperands(), newCaseValues,
                                        newCaseDestinations, newCaseOperands);
  return success();
}

void SwitchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(&simplifySwitchWithOnlyDefault)
      .add(&dropSwitchCasesThatMatchDefault)
      .add(&simplifyConstSwitchValue)
      .add(&simplifyPassThroughSwitch)
      .add(&simplifySwitchFromSwitchOnSameCondition)
      .add(&simplifySwitchFromDefaultSwitchOnSameCondition);
}

//===----------------------------------------------------------------------===//
// TruncateIOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(TruncateIOp op) {
  auto srcType = getElementTypeOrSelf(op.getOperand().getType());
  auto dstType = getElementTypeOrSelf(op.getType());

  if (srcType.isa<IndexType>())
    return op.emitError() << srcType << " is not a valid operand type";
  if (dstType.isa<IndexType>())
    return op.emitError() << dstType << " is not a valid result type";

  if (srcType.cast<IntegerType>().getWidth() <=
      dstType.cast<IntegerType>().getWidth())
    return op.emitError("operand type ")
           << srcType << " must be wider than result type " << dstType;

  return success();
}

OpFoldResult TruncateIOp::fold(ArrayRef<Attribute> operands) {
  // trunci(zexti(a)) -> a
  // trunci(sexti(a)) -> a
  if (matchPattern(getOperand(), m_Op<ZeroExtendIOp>()) ||
      matchPattern(getOperand(), m_Op<SignExtendIOp>()))
    return getOperand().getDefiningOp()->getOperand(0);

  assert(operands.size() == 1 && "unary operation takes one operand");

  if (!operands[0])
    return {};

  if (auto lhs = operands[0].dyn_cast<IntegerAttr>()) {

    return IntegerAttr::get(
        getType(), lhs.getValue().trunc(getType().getIntOrFloatBitWidth()));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// UnsignedDivIOp
//===----------------------------------------------------------------------===//

OpFoldResult UnsignedDivIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (div0 || !b) {
      div0 = true;
      return a;
    }
    return a.udiv(b);
  });

  // Fold out division by one. Assumes all tensors of all ones are splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return div0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// UnsignedRemIOp
//===----------------------------------------------------------------------===//

OpFoldResult UnsignedRemIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "remi_unsigned takes two operands");

  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().urem(rhsValue));
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

OpFoldResult XOrOp::fold(ArrayRef<Attribute> operands) {
  /// xor(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// xor(x,x) -> 0
  if (lhs() == rhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

namespace {
/// Replace a not of a comparison operation, for example: not(cmp eq A, B) =>
/// cmp ne A, B. Note that a logical not is implemented as xor 1, val.
struct NotICmp : public OpRewritePattern<XOrOp> {
  using OpRewritePattern<XOrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XOrOp op,
                                PatternRewriter &rewriter) const override {
    // Commutative ops (such as xor) have the constant appear second, which
    // we assume here.

    APInt constValue;
    if (!matchPattern(op.getOperand(1), m_ConstantInt(&constValue)))
      return failure();

    if (constValue != 1)
      return failure();

    auto prev = op.getOperand(0).getDefiningOp<CmpIOp>();
    if (!prev)
      return failure();

    switch (prev.predicate()) {
    case CmpIPredicate::eq:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ne, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::ne:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::eq, prev.lhs(),
                                          prev.rhs());
      return success();

    case CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sge, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::sle:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sgt, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sle, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::sge:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::slt, prev.lhs(),
                                          prev.rhs());
      return success();

    case CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::uge, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::ule:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ugt, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ule, prev.lhs(),
                                          prev.rhs());
      return success();
    case CmpIPredicate::uge:
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ult, prev.lhs(),
                                          prev.rhs());
      return success();
    }
    return failure();
  }
};
} // namespace

void XOrOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<NotICmp>(context);
}

//===----------------------------------------------------------------------===//
// ZeroExtendIOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ZeroExtendIOp op) {
  auto srcType = getElementTypeOrSelf(op.getOperand().getType());
  auto dstType = getElementTypeOrSelf(op.getType());

  if (srcType.isa<IndexType>())
    return op.emitError() << srcType << " is not a valid operand type";
  if (dstType.isa<IndexType>())
    return op.emitError() << dstType << " is not a valid result type";

  if (srcType.cast<IntegerType>().getWidth() >=
      dstType.cast<IntegerType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/StandardOps/IR/Ops.cpp.inc"
