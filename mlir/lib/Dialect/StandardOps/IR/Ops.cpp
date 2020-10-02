//===- Ops.cpp - Standard MLIR Operations ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

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

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(Operation *, Region *,
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

/// A custom cast operation printer that omits the "std." prefix from the
/// operation names.
static void printStandardCastOp(Operation *op, OpAsmPrinter &p) {
  int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op->getName().getStringRef().drop_front(stdDotLen) << ' '
    << op->getOperand(0) << " : " << op->getOperand(0).getType() << " to "
    << op->getResult(0).getType();
}

/// A custom cast operation verifier.
template <typename T>
static LogicalResult verifyCastOp(T op) {
  auto opType = op.getOperand().getType();
  auto resType = op.getType();
  if (!T::areCastCompatible(opType, resType))
    return op.emitError("operand type ") << opType << " and result type "
                                         << resType << " are cast incompatible";

  return success();
}

void StandardOpsDialect::initialize() {
  addOperations<DmaStartOp, DmaWaitOp,
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

void mlir::printDimAndSymbolList(Operation::operand_iterator begin,
                                 Operation::operand_iterator end,
                                 unsigned numDims, OpAsmPrinter &p) {
  Operation::operand_range operands(begin, end);
  p << '(' << operands.take_front(numDims) << ')';
  if (operands.size() != numDims)
    p << '[' << operands.drop_front(numDims) << ']';
}

// Parses dimension and symbol list, and sets 'numDims' to the number of
// dimension operands parsed.
// Returns 'false' on success and 'true' on error.
ParseResult mlir::parseDimAndSymbolList(OpAsmParser &parser,
                                        SmallVectorImpl<Value> &operands,
                                        unsigned &numDims) {
  SmallVector<OpAsmParser::OperandType, 8> opInfos;
  if (parser.parseOperandList(opInfos, OpAsmParser::Delimiter::Paren))
    return failure();
  // Store number of dimensions for validation by caller.
  numDims = opInfos.size();

  // Parse the optional symbol operands.
  auto indexTy = parser.getBuilder().getIndexType();
  if (parser.parseOperandList(opInfos,
                              OpAsmParser::Delimiter::OptionalSquare) ||
      parser.resolveOperands(opInfos, indexTy, operands))
    return failure();
  return success();
}

/// Matches a ConstantIndexOp.
/// TODO: This should probably just be a general matcher that uses m_Constant
/// and checks the operation for an index type.
static detail::op_matcher<ConstantIndexOp> m_ConstantIndex() {
  return detail::op_matcher<ConstantIndexOp>();
}

//===----------------------------------------------------------------------===//
// Common canonicalization pattern support logic
//===----------------------------------------------------------------------===//

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<MemRefCastOp>();
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
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
    Type a, Type b, function_ref<bool(Type, Type)> areElementsCastCompatible) {
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

//===----------------------------------------------------------------------===//
// AllocOp / AllocaOp
//===----------------------------------------------------------------------===//

template <typename AllocLikeOp>
static void printAllocLikeOp(OpAsmPrinter &p, AllocLikeOp op, StringRef name) {
  static_assert(llvm::is_one_of<AllocLikeOp, AllocOp, AllocaOp>::value,
                "applies to only alloc or alloca");
  p << name;

  // Print dynamic dimension operands.
  MemRefType type = op.getType();
  printDimAndSymbolList(op.operand_begin(), op.operand_end(),
                        type.getNumDynamicDims(), p);
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"map"});
  p << " : " << type;
}

static void print(OpAsmPrinter &p, AllocOp op) {
  printAllocLikeOp(p, op, "alloc");
}

static void print(OpAsmPrinter &p, AllocaOp op) {
  printAllocLikeOp(p, op, "alloca");
}

static ParseResult parseAllocLikeOp(OpAsmParser &parser,
                                    OperationState &result) {
  MemRefType type;

  // Parse the dimension operands and optional symbol operands, followed by a
  // memref type.
  unsigned numDimOperands;
  if (parseDimAndSymbolList(parser, result.operands, numDimOperands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  // Check numDynamicDims against number of question marks in memref type.
  // Note: this check remains here (instead of in verify()), because the
  // partition between dim operands and symbol operands is lost after parsing.
  // Verification still checks that the total number of operands matches
  // the number of symbols in the affine map, plus the number of dynamic
  // dimensions in the memref.
  if (numDimOperands != type.getNumDynamicDims())
    return parser.emitError(parser.getNameLoc())
           << "dimension operand count does not equal memref dynamic dimension "
              "count";
  result.types.push_back(type);
  return success();
}

template <typename AllocLikeOp>
static LogicalResult verify(AllocLikeOp op) {
  static_assert(llvm::is_one_of<AllocLikeOp, AllocOp, AllocaOp>::value,
                "applies to only alloc or alloca");
  auto memRefType = op.getResult().getType().template dyn_cast<MemRefType>();
  if (!memRefType)
    return op.emitOpError("result must be a memref");

  unsigned numSymbols = 0;
  if (!memRefType.getAffineMaps().empty()) {
    // Store number of symbols used in affine map (used in subsequent check).
    AffineMap affineMap = memRefType.getAffineMaps()[0];
    numSymbols = affineMap.getNumSymbols();
  }

  // Check that the total number of operands matches the number of symbols in
  // the affine map, plus the number of dynamic dimensions specified in the
  // memref type.
  unsigned numDynamicDims = memRefType.getNumDynamicDims();
  if (op.getNumOperands() != numDynamicDims + numSymbols)
    return op.emitOpError(
        "operand count does not equal dimension plus symbol operand count");

  // Verify that all operands are of type Index.
  for (auto operandType : op.getOperandTypes())
    if (!operandType.isIndex())
      return op.emitOpError("requires operands to be of type Index");

  if (std::is_same<AllocLikeOp, AllocOp>::value)
    return success();

  // An alloca op needs to have an ancestor with an allocation scope trait.
  if (!op.template getParentWithTrait<OpTrait::AutomaticAllocationScope>())
    return op.emitOpError(
        "requires an ancestor op with AutomaticAllocationScope trait");

  return success();
}

namespace {
/// Fold constant dimensions into an alloc like operation.
template <typename AllocLikeOp>
struct SimplifyAllocConst : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp alloc,
                                PatternRewriter &rewriter) const override {
    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    if (llvm::none_of(alloc.getOperands(), [](Value operand) {
          return matchPattern(operand, m_ConstantIndex());
        }))
      return failure();

    auto memrefType = alloc.getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());
    SmallVector<Value, 4> newOperands;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType.getRank(); dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (dimSize != -1) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = alloc.getOperand(dynamicDimPos).getDefiningOp();
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.getValue());
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(-1);
        newOperands.push_back(alloc.getOperand(dynamicDimPos));
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType).setShape(newShapeConstants);
    assert(static_cast<int64_t>(newOperands.size()) ==
           newMemRefType.getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    auto newAlloc = rewriter.create<AllocLikeOp>(alloc.getLoc(), newMemRefType,
                                                 newOperands, IntegerAttr());
    // Insert a cast so we have the same type as the old alloc.
    auto resultCast = rewriter.create<MemRefCastOp>(alloc.getLoc(), newAlloc,
                                                    alloc.getType());

    rewriter.replaceOp(alloc, {resultCast});
    return success();
  }
};

/// Fold alloc operations with no uses. Alloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadAlloc : public OpRewritePattern<AllocOp> {
  using OpRewritePattern<AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc.use_empty()) {
      rewriter.eraseOp(alloc);
      return success();
    }
    return failure();
  }
};
} // end anonymous namespace.

void AllocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<SimplifyAllocConst<AllocOp>, SimplifyDeadAlloc>(context);
}

void AllocaOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyAllocConst<AllocaOp>>(context);
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

namespace {
struct EraseRedundantAssertions : public OpRewritePattern<AssertOp> {
  using OpRewritePattern<AssertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AssertOp op,
                                PatternRewriter &rewriter) const override {
    // Erase assertion if argument is constant true.
    if (matchPattern(op.arg(), m_One())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
} // namespace

void AssertOp::getCanonicalizationPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context) {
  patterns.insert<EraseRedundantAssertions>(context);
}

//===----------------------------------------------------------------------===//
// AssumeAlignmentOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AssumeAlignmentOp op) {
  unsigned alignment = op.alignment();
  if (!llvm::isPowerOf2_32(alignment))
    return op.emitOpError("alignment must be power of 2");
  return success();
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
  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// AtomicYieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AtomicYieldOp op) {
  Type parentType = op.getParentOp()->getResultTypes().front();
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
/// the new destination and values. `argStorage` is an optional storage to use
/// if operands to the collapsed successor need to be remapped.
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

namespace {
/// Simplify a branch to a block that has a single predecessor. This effectively
/// merges the two blocks.
struct SimplifyBrToBlockWithSinglePred : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    // Check that the successor block has a single predecessor.
    Block *succ = op.getDest();
    Block *opParent = op.getOperation()->getBlock();
    if (succ == opParent || !llvm::hasSingleElement(succ->getPredecessors()))
      return failure();

    // Merge the successor into the current block and erase the branch.
    rewriter.mergeBlocks(succ, opParent, op.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

///   br ^bb1
/// ^bb1
///   br ^bbN(...)
///
///  -> br ^bbN(...)
///
struct SimplifyPassThroughBr : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    Block *dest = op.getDest();
    ValueRange destOperands = op.getOperands();
    SmallVector<Value, 4> destOperandStorage;

    // Try to collapse the successor if it points somewhere other than this
    // block.
    if (dest == op.getOperation()->getBlock() ||
        failed(collapseBranch(dest, destOperands, destOperandStorage)))
      return failure();

    // Create a new branch with the collapsed successor.
    rewriter.replaceOpWithNewOp<BranchOp>(op, dest, destOperands);
    return success();
  }
};
} // end anonymous namespace.

Block *BranchOp::getDest() { return getSuccessor(); }

void BranchOp::setDest(Block *block) { return setSuccessor(block); }

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

void BranchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyBrToBlockWithSinglePred, SimplifyPassThroughBr>(
      context);
}

Optional<MutableOperandRange>
BranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
}

Block *BranchOp::getSuccessorForOperands(ArrayRef<Attribute>) { return dest(); }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(CallOp op) {
  // Check that the callee attribute was specified.
  auto fnAttr = op.getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");
  auto fn =
      op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
  if (!fn)
    return op.emitOpError() << "'" << fnAttr.getValue()
                            << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != op.getNumOperands())
    return op.emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (op.getOperand(i).getType() != fnType.getInput(i))
      return op.emitOpError("operand type mismatch");

  if (fnType.getNumResults() != op.getNumResults())
    return op.emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (op.getResult(i).getType() != fnType.getResult(i))
      return op.emitOpError("result type mismatch");

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getOperandTypes(), getResultTypes(), getContext());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//
namespace {
/// Fold indirect calls that have a constant function as the callee operand.
struct SimplifyIndirectCallWithKnownCallee
    : public OpRewritePattern<CallIndirectOp> {
  using OpRewritePattern<CallIndirectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CallIndirectOp indirectCall,
                                PatternRewriter &rewriter) const override {
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
};
} // end anonymous namespace.

void CallIndirectOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyIndirectCallWithKnownCallee>(context);
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(1, type.getContext());
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

// Constant folding hook for comparisons.
OpFoldResult CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two arguments");

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return IntegerAttr::get(IntegerType::get(1, getContext()), APInt(1, val));
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
  return IntegerAttr::get(IntegerType::get(1, getContext()), APInt(1, val));
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
    if (trueDest->getUniquePredecessor() != condbr.getOperation()->getBlock())
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
} // end anonymous namespace

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyConstCondBranchPred, SimplifyPassThroughCondBranch,
                 SimplifyCondBranchIdenticalSuccessors>(context);
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
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

  if (op.getAttrs().size() > 1)
    p << ' ';
  p << op.getValue();

  // If the value is a symbol reference, print a trailing type.
  if (op.getValue().isa<SymbolRefAttr>())
    p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr>())
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

  auto type = op.getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";

  if (type.isa<IndexType>() || value.isa<BoolAttr>())
    return success();

  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    // If the type has a known bitwidth we verify that the value can be
    // represented with the given bitwidth.
    auto bitwidth = type.cast<IntegerType>().getWidth();
    auto intVal = intAttr.getValue();
    if (!intVal.isSignedIntN(bitwidth) && !intVal.isIntN(bitwidth))
      return op.emitOpError("requires 'value' to be an integer within the "
                            "range of the integer result type");
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
        op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
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
  // Otherwise, the attribute must have the same type as 'type'.
  if (value.getType() != type)
    return false;
  // Finally, check that the attribute kind is handled.
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

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//
namespace {
/// Fold Dealloc operations that are deallocating an AllocOp that is only used
/// by other Dealloc operations.
struct SimplifyDeadDealloc : public OpRewritePattern<DeallocOp> {
  using OpRewritePattern<DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp dealloc,
                                PatternRewriter &rewriter) const override {
    // Check that the memref operand's defining operation is an AllocOp.
    Value memref = dealloc.memref();
    if (!isa_and_nonnull<AllocOp>(memref.getDefiningOp()))
      return failure();

    // Check that all of the uses of the AllocOp are other DeallocOps.
    for (auto *user : memref.getUsers())
      if (!isa<DeallocOp>(user))
        return failure();

    // Erase the dealloc operation.
    rewriter.eraseOp(dealloc);
    return success();
  }
};
} // end anonymous namespace.

static LogicalResult verify(DeallocOp op) {
  if (!op.memref().getType().isa<MemRefType>())
    return op.emitOpError("operand must be a memref");
  return success();
}

void DeallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<SimplifyDeadDealloc>(context);
}

LogicalResult DeallocOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dealloc(memrefcast) -> dealloc
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

void DimOp::build(OpBuilder &builder, OperationState &result,
                  Value memrefOrTensor, int64_t index) {
  auto loc = result.location;
  Value indexValue = builder.create<ConstantIndexOp>(loc, index);
  build(builder, result, memrefOrTensor, indexValue);
}

void DimOp::build(OpBuilder &builder, OperationState &result,
                  Value memrefOrTensor, Value index) {
  auto indexTy = builder.getIndexType();
  build(builder, result, indexTy, memrefOrTensor, index);
}

Optional<int64_t> DimOp::getConstantIndex() {
  if (auto constantOp = index().getDefiningOp<ConstantOp>())
    return constantOp.getValue().cast<IntegerAttr>().getInt();
  return {};
}

static LogicalResult verify(DimOp op) {
  // Assume unknown index to be in range.
  Optional<int64_t> index = op.getConstantIndex();
  if (!index.hasValue())
    return success();

  // Check that constant index is not knowingly out of range.
  auto type = op.memrefOrTensor().getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    if (index.getValue() >= tensorType.getRank())
      return op.emitOpError("index is out of range");
  } else if (auto memrefType = type.dyn_cast<MemRefType>()) {
    if (index.getValue() >= memrefType.getRank())
      return op.emitOpError("index is out of range");
  } else if (type.isa<UnrankedTensorType>() || type.isa<UnrankedMemRefType>()) {
    // Assume index to be in range.
  } else {
    llvm_unreachable("expected operand with tensor or memref type");
  }

  return success();
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  auto index = operands[1].dyn_cast_or_null<IntegerAttr>();

  // All forms of folding require a known index.
  if (!index)
    return {};

  auto argTy = memrefOrTensor().getType();
  // Fold if the shape extent along the given index is known.
  if (auto shapedTy = argTy.dyn_cast<ShapedType>()) {
    // Folding for unranked types (UnrankedMemRefType, UnrankedTensorType) is
    // not supported.
    if (!shapedTy.hasRank())
      return {};
    if (!shapedTy.isDynamicDim(index.getInt())) {
      Builder builder(getContext());
      return builder.getIndexAttr(shapedTy.getShape()[index.getInt()]);
    }
  }

  // Fold dim to the size argument for an `AllocOp`, `ViewOp`, or `SubViewOp`.
  auto memrefType = argTy.dyn_cast<MemRefType>();
  if (!memrefType)
    return {};

  // The size at the given index is now known to be a dynamic size of a memref.
  auto *memref = memrefOrTensor().getDefiningOp();
  unsigned unsignedIndex = index.getValue().getZExtValue();
  if (auto alloc = dyn_cast_or_null<AllocOp>(memref))
    return *(alloc.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(unsignedIndex));

  if (auto view = dyn_cast_or_null<ViewOp>(memref))
    return *(view.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(unsignedIndex));

  if (auto subview = dyn_cast_or_null<SubViewOp>(memref)) {
    assert(subview.isDynamicSize(unsignedIndex) &&
           "Expected dynamic subview size");
    return subview.getDynamicSize(unsignedIndex);
  }

  // dim(memrefcast) -> dim
  if (succeeded(foldMemRefCast(*this)))
    return getResult();

  return {};
}

// ---------------------------------------------------------------------------
// DmaStartOp
// ---------------------------------------------------------------------------

void DmaStartOp::build(OpBuilder &builder, OperationState &result,
                       Value srcMemRef, ValueRange srcIndices, Value destMemRef,
                       ValueRange destIndices, Value numElements,
                       Value tagMemRef, ValueRange tagIndices, Value stride,
                       Value elementsPerStride) {
  result.addOperands(srcMemRef);
  result.addOperands(srcIndices);
  result.addOperands(destMemRef);
  result.addOperands(destIndices);
  result.addOperands({numElements, tagMemRef});
  result.addOperands(tagIndices);
  if (stride)
    result.addOperands({stride, elementsPerStride});
}

void DmaStartOp::print(OpAsmPrinter &p) {
  p << "dma_start " << getSrcMemRef() << '[' << getSrcIndices() << "], "
    << getDstMemRef() << '[' << getDstIndices() << "], " << getNumElements()
    << ", " << getTagMemRef() << '[' << getTagIndices() << ']';
  if (isStrided())
    p << ", " << getStride() << ", " << getNumElementsPerStride();

  p.printOptionalAttrDict(getAttrs());
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType()
    << ", " << getTagMemRef().getType();
}

// Parse DmaStartOp.
// Ex:
//   %dma_id = dma_start %src[%i, %j], %dst[%k, %l], %size,
//                       %tag[%index], %stride, %num_elt_per_stride :
//                     : memref<3076 x f32, 0>,
//                       memref<1024 x f32, 2>,
//                       memref<1 x i32>
//
ParseResult DmaStartOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> srcIndexInfos;
  OpAsmParser::OperandType dstMemRefInfo;
  SmallVector<OpAsmParser::OperandType, 4> dstIndexInfos;
  OpAsmParser::OperandType numElementsInfo;
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> tagIndexInfos;
  SmallVector<OpAsmParser::OperandType, 2> strideInfo;

  SmallVector<Type, 3> types;
  auto indexType = parser.getBuilder().getIndexType();

  // Parse and resolve the following list of operands:
  // *) source memref followed by its indices (in square brackets).
  // *) destination memref followed by its indices (in square brackets).
  // *) dma size in KiB.
  if (parser.parseOperand(srcMemRefInfo) ||
      parser.parseOperandList(srcIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(dstMemRefInfo) ||
      parser.parseOperandList(dstIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(numElementsInfo) ||
      parser.parseComma() || parser.parseOperand(tagMemrefInfo) ||
      parser.parseOperandList(tagIndexInfos, OpAsmParser::Delimiter::Square))
    return failure();

  // Parse optional stride and elements per stride.
  if (parser.parseTrailingOperandList(strideInfo))
    return failure();

  bool isStrided = strideInfo.size() == 2;
  if (!strideInfo.empty() && !isStrided) {
    return parser.emitError(parser.getNameLoc(),
                            "expected two stride related operands");
  }

  if (parser.parseColonTypeList(types))
    return failure();
  if (types.size() != 3)
    return parser.emitError(parser.getNameLoc(), "fewer/more types expected");

  if (parser.resolveOperand(srcMemRefInfo, types[0], result.operands) ||
      parser.resolveOperands(srcIndexInfos, indexType, result.operands) ||
      parser.resolveOperand(dstMemRefInfo, types[1], result.operands) ||
      parser.resolveOperands(dstIndexInfos, indexType, result.operands) ||
      // size should be an index.
      parser.resolveOperand(numElementsInfo, indexType, result.operands) ||
      parser.resolveOperand(tagMemrefInfo, types[2], result.operands) ||
      // tag indices should be index.
      parser.resolveOperands(tagIndexInfos, indexType, result.operands))
    return failure();

  if (isStrided) {
    if (parser.resolveOperands(strideInfo, indexType, result.operands))
      return failure();
  }

  return success();
}

LogicalResult DmaStartOp::verify() {
  unsigned numOperands = getNumOperands();

  // Mandatory non-variadic operands are: src memref, dst memref, tag memref and
  // the number of elements.
  if (numOperands < 4)
    return emitOpError("expected at least 4 operands");

  // Check types of operands. The order of these calls is important: the later
  // calls rely on some type properties to compute the operand position.
  // 1. Source memref.
  if (!getSrcMemRef().getType().isa<MemRefType>())
    return emitOpError("expected source to be of memref type");
  if (numOperands < getSrcMemRefRank() + 4)
    return emitOpError() << "expected at least " << getSrcMemRefRank() + 4
                         << " operands";
  if (!getSrcIndices().empty() &&
      !llvm::all_of(getSrcIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError("expected source indices to be of index type");

  // 2. Destination memref.
  if (!getDstMemRef().getType().isa<MemRefType>())
    return emitOpError("expected destination to be of memref type");
  unsigned numExpectedOperands = getSrcMemRefRank() + getDstMemRefRank() + 4;
  if (numOperands < numExpectedOperands)
    return emitOpError() << "expected at least " << numExpectedOperands
                         << " operands";
  if (!getDstIndices().empty() &&
      !llvm::all_of(getDstIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError("expected destination indices to be of index type");

  // 3. Number of elements.
  if (!getNumElements().getType().isIndex())
    return emitOpError("expected num elements to be of index type");

  // 4. Tag memref.
  if (!getTagMemRef().getType().isa<MemRefType>())
    return emitOpError("expected tag to be of memref type");
  numExpectedOperands += getTagMemRefRank();
  if (numOperands < numExpectedOperands)
    return emitOpError() << "expected at least " << numExpectedOperands
                         << " operands";
  if (!getTagIndices().empty() &&
      !llvm::all_of(getTagIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError("expected tag indices to be of index type");

  // DMAs from different memory spaces supported.
  if (getSrcMemorySpace() == getDstMemorySpace())
    return emitOpError("DMA should be between different memory spaces");

  // Optional stride-related operands must be either both present or both
  // absent.
  if (numOperands != numExpectedOperands &&
      numOperands != numExpectedOperands + 2)
    return emitOpError("incorrect number of operands");

  // 5. Strides.
  if (isStrided()) {
    if (!getStride().getType().isIndex() ||
        !getNumElementsPerStride().getType().isIndex())
      return emitOpError(
          "expected stride and num elements per stride to be of type index");
  }

  return success();
}

LogicalResult DmaStartOp::fold(ArrayRef<Attribute> cstOperands,
                               SmallVectorImpl<OpFoldResult> &results) {
  /// dma_start(memrefcast) -> dma_start
  return foldMemRefCast(*this);
}

// ---------------------------------------------------------------------------
// DmaWaitOp
// ---------------------------------------------------------------------------

void DmaWaitOp::build(OpBuilder &builder, OperationState &result,
                      Value tagMemRef, ValueRange tagIndices,
                      Value numElements) {
  result.addOperands(tagMemRef);
  result.addOperands(tagIndices);
  result.addOperands(numElements);
}

void DmaWaitOp::print(OpAsmPrinter &p) {
  p << "dma_wait " << getTagMemRef() << '[' << getTagIndices() << "], "
    << getNumElements();
  p.printOptionalAttrDict(getAttrs());
  p << " : " << getTagMemRef().getType();
}

// Parse DmaWaitOp.
// Eg:
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 4>
//
ParseResult DmaWaitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType tagMemrefInfo;
  SmallVector<OpAsmParser::OperandType, 2> tagIndexInfos;
  Type type;
  auto indexType = parser.getBuilder().getIndexType();
  OpAsmParser::OperandType numElementsInfo;

  // Parse tag memref, its indices, and dma size.
  if (parser.parseOperand(tagMemrefInfo) ||
      parser.parseOperandList(tagIndexInfos, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(numElementsInfo) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(tagMemrefInfo, type, result.operands) ||
      parser.resolveOperands(tagIndexInfos, indexType, result.operands) ||
      parser.resolveOperand(numElementsInfo, indexType, result.operands))
    return failure();

  return success();
}

LogicalResult DmaWaitOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dma_wait(memrefcast) -> dma_wait
  return foldMemRefCast(*this);
}

LogicalResult DmaWaitOp::verify() {
  // Mandatory non-variadic operands are tag and the number of elements.
  if (getNumOperands() < 2)
    return emitOpError() << "expected at least 2 operands";

  // Check types of operands. The order of these calls is important: the later
  // calls rely on some type properties to compute the operand position.
  if (!getTagMemRef().getType().isa<MemRefType>())
    return emitOpError() << "expected tag to be of memref type";

  if (getNumOperands() != 2 + getTagMemRefRank())
    return emitOpError() << "expected " << 2 + getTagMemRefRank()
                         << " operands";

  if (!getTagIndices().empty() &&
      !llvm::all_of(getTagIndices().getTypes(),
                    [](Type t) { return t.isIndex(); }))
    return emitOpError() << "expected tag indices to be of index type";

  if (!getNumElements().getType().isIndex())
    return emitOpError()
           << "expected the number of elements to be of index type";

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicTensorFromElementsOp
//===----------------------------------------------------------------------===//

static ParseResult parseDynamicTensorFromElementsOp(OpAsmParser &parser,
                                                    OperationState &result) {
  // Parse operands.
  SmallVector<OpAsmParser::OperandType, 4> dynamicExtents;
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.parseOperandList(dynamicExtents) ||
      parser.resolveOperands(dynamicExtents, indexTy, result.operands))
    return failure();

  // Parse body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}, {}))
    return failure();

  // Parse result type.
  Type resultType;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();
  result.addTypes(resultType);

  return success();
}

static void print(OpAsmPrinter &p, DynamicTensorFromElementsOp op) {
  p << "dynamic_tensor_from_elements " << op.dynamicExtents();
  p.printRegion(op.body());
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getType();
}

static LogicalResult verify(DynamicTensorFromElementsOp op) {
  // Ensure that the tensor type has as many dynamic dimensions as are specified
  // by the operands.
  RankedTensorType resultTy = op.getType().cast<RankedTensorType>();
  if (op.getNumOperands() != resultTy.getNumDynamicDims())
    return op.emitError("must have as many index operands as dynamic extents "
                        "in the result type");

  // Ensure that region arguments span the index space.
  if (!llvm::all_of(op.body().getArgumentTypes(),
                    [](Type ty) { return ty.isIndex(); }))
    return op.emitError("all body arguments must be index");
  if (op.body().getNumArguments() != resultTy.getRank())
    return op.emitError("must have one body argument per input dimension");

  // Ensure that the region yields an element of the right type.
  auto yieldOp =
      llvm::cast<YieldOp>(op.body().getBlocks().front().getTerminator());
  if (yieldOp.value().getType() != resultTy.getElementType())
    return op.emitOpError(
        "body must be terminated with a `yield` operation of the tensor "
        "element type");

  return success();
}

void DynamicTensorFromElementsOp::build(
    OpBuilder &b, OperationState &result, Type resultTy,
    ValueRange dynamicExtents,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  build(b, result, resultTy, dynamicExtents);

  // Build and populate body.
  OpBuilder::InsertionGuard guard(b);
  Region *bodyRegion = result.regions.front().get();
  auto rank = resultTy.cast<RankedTensorType>().getRank();
  SmallVector<Type, 2> argumentTypes(rank, b.getIndexType());
  Block *bodyBlock =
      b.createBlock(bodyRegion, bodyRegion->end(), argumentTypes);
  bodyBuilder(b, result.location, bodyBlock->getArguments());
}

namespace {

/// Canonicalizes dynamic_tensor_from_elements operations with a constant
/// operand into the equivalent operation with the operand expressed in the
/// result type, instead. We also insert a type cast to make sure that the
/// resulting IR is still well-typed.
struct StaticDynamicTensorFromElements
    : public OpRewritePattern<DynamicTensorFromElementsOp> {
  using OpRewritePattern<DynamicTensorFromElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicTensorFromElementsOp tensorFromElements,
                                PatternRewriter &rewriter) const final {
    auto resultType =
        tensorFromElements.getResult().getType().cast<RankedTensorType>();

    if (resultType.hasStaticShape())
      return failure();

    SmallVector<Value, 4> newOperands;
    SmallVector<int64_t, 4> newShape;
    auto operandsIt = tensorFromElements.dynamicExtents().begin();

    for (int64_t dim : resultType.getShape()) {
      if (dim != RankedTensorType::kDynamicSize) {
        newShape.push_back(dim);
        continue;
      }
      APInt index;
      if (!matchPattern(*operandsIt, m_ConstantInt(&index))) {
        newShape.push_back(RankedTensorType::kDynamicSize);
        newOperands.push_back(*operandsIt++);
        continue;
      }
      newShape.push_back(index.getSExtValue());
      operandsIt++;
    }

    if (newOperands.size() == tensorFromElements.dynamicExtents().size())
      return failure();

    auto loc = tensorFromElements.getLoc();
    auto newOp = rewriter.create<DynamicTensorFromElementsOp>(
        loc, RankedTensorType::get(newShape, resultType.getElementType()),
        newOperands);
    rewriter.inlineRegionBefore(tensorFromElements.body(), newOp.body(),
                                newOp.body().begin());
    rewriter.replaceOpWithNewOp<TensorCastOp>(tensorFromElements, resultType,
                                              newOp);
    return success();
  }
};

/// Canonicalizes the pattern of the form
///
/// %tensor = dynamic_tensor_from_elements %x {
///   ^bb0(%arg0: index):  // no predecessors
///   <computation>
///   yield %1 : index
/// } : tensor<?xindex>
/// %extracted_element = extract_element %tensor[%c0] : tensor<?xi32>
///
/// to just <computation> with %arg0 replaced by %c0. We only do this if the
/// dynamic_tensor_from_elements operation has no side-effects.
struct ExtractElementFromDynamicTensorFromElements
    : public OpRewritePattern<ExtractElementOp> {
  using OpRewritePattern<ExtractElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractElementOp extract,
                                PatternRewriter &rewriter) const final {
    auto tensorFromElements =
        extract.aggregate().getDefiningOp<DynamicTensorFromElementsOp>();
    if (!tensorFromElements || !wouldOpBeTriviallyDead(tensorFromElements))
      return failure();

    BlockAndValueMapping mapping;
    Block *body = tensorFromElements.getBody();
    mapping.map(body->getArguments(), extract.indices());
    for (auto &op : body->without_terminator())
      rewriter.clone(op, mapping);

    auto yield = cast<YieldOp>(body->getTerminator());

    rewriter.replaceOp(extract, mapping.lookupOrDefault(yield.value()));
    return success();
  }
};

} // namespace

void DynamicTensorFromElementsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExtractElementFromDynamicTensorFromElements,
                 StaticDynamicTensorFromElements>(context);
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ExtractElementOp op) {
  // Verify the # indices match if we have a ranked type.
  auto aggregateType = op.getAggregate().getType().cast<ShapedType>();
  if (aggregateType.hasRank() &&
      aggregateType.getRank() != op.getNumOperands() - 1)
    return op.emitOpError("incorrect number of indices for extract_element");

  return success();
}

OpFoldResult ExtractElementOp::fold(ArrayRef<Attribute> operands) {
  assert(!operands.empty() && "extract_element takes at least one operand");

  // The aggregate operand must be a known constant.
  Attribute aggregate = operands.front();
  if (!aggregate)
    return {};

  // If this is a splat elements attribute, simply return the value. All of the
  // elements of a splat attribute are the same.
  if (auto splatAggregate = aggregate.dyn_cast<SplatElementsAttr>())
    return splatAggregate.getSplatValue();

  // Otherwise, collect the constant indices into the aggregate.
  SmallVector<uint64_t, 8> indices;
  for (Attribute indice : llvm::drop_begin(operands, 1)) {
    if (!indice || !indice.isa<IntegerAttr>())
      return {};
    indices.push_back(indice.cast<IntegerAttr>().getInt());
  }

  // If this is an elements attribute, query the value at the given indices.
  auto elementsAttr = aggregate.dyn_cast<ElementsAttr>();
  if (elementsAttr && elementsAttr.isValidIndex(indices))
    return elementsAttr.getValue(indices);
  return {};
}

//===----------------------------------------------------------------------===//
// TensorFromElementsOp
//===----------------------------------------------------------------------===//

void TensorFromElementsOp::build(OpBuilder &builder, OperationState &result,
                                 Type elementType, ValueRange elements) {
  Type resultTy = RankedTensorType::get({static_cast<int64_t>(elements.size())},
                                        elementType);
  result.addOperands(elements);
  result.addTypes(resultTy);
}

void TensorFromElementsOp::build(OpBuilder &builder, OperationState &result,
                                 ValueRange elements) {
  assert(!elements.empty() && "expected at least one element");
  build(builder, result, elements.front().getType(), elements);
}

namespace {

// Canonicalizes the pattern of the form
//
// %tensor = "tensor_from_elements(%element) : (i32) -> tensor<1xi32>
// %extracted_element = extract_element %tensor[%c0] : tensor<1xi32>
//
// to just %element.
struct ExtractElementFromTensorFromElements
    : public OpRewritePattern<ExtractElementOp> {
  using OpRewritePattern<ExtractElementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractElementOp extract,
                                PatternRewriter &rewriter) const final {
    if (extract.indices().size() != 1)
      return failure();

    auto tensorFromElements = dyn_cast_or_null<TensorFromElementsOp>(
        extract.aggregate().getDefiningOp());
    if (tensorFromElements == nullptr)
      return failure();

    APInt index;
    if (!matchPattern(*extract.indices().begin(), m_ConstantInt(&index)))
      return failure();
    rewriter.replaceOp(extract,
                       tensorFromElements.getOperand(index.getZExtValue()));
    return success();
  }
};

} // namespace

void TensorFromElementsOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExtractElementFromTensorFromElements>(context);
}

//===----------------------------------------------------------------------===//
// FPExtOp
//===----------------------------------------------------------------------===//

bool FPExtOp::areCastCompatible(Type a, Type b) {
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() < fb.getWidth();
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

bool FPToSIOp::areCastCompatible(Type a, Type b) {
  if (a.isa<FloatType>() && b.isSignlessInteger())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

bool FPToUIOp::areCastCompatible(Type a, Type b) {
  if (a.isa<FloatType>() && b.isSignlessInteger())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// FPTruncOp
//===----------------------------------------------------------------------===//

bool FPTruncOp::areCastCompatible(Type a, Type b) {
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() > fb.getWidth();
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast is applicable from index to integer and backwards.
bool IndexCastOp::areCastCompatible(Type a, Type b) {
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

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(LoadOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("incorrect number of indices for load");
  return success();
}

OpFoldResult LoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// load(memrefcast) -> load
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// MemRefCastOp
//===----------------------------------------------------------------------===//

Value MemRefCastOp::getViewSource() { return source(); }

bool MemRefCastOp::areCastCompatible(Type a, Type b) {
  auto aT = a.dyn_cast<MemRefType>();
  auto bT = b.dyn_cast<MemRefType>();

  auto uaT = a.dyn_cast<UnrankedMemRefType>();
  auto ubT = b.dyn_cast<UnrankedMemRefType>();

  if (aT && bT) {
    if (aT.getElementType() != bT.getElementType())
      return false;
    if (aT.getAffineMaps() != bT.getAffineMaps()) {
      int64_t aOffset, bOffset;
      SmallVector<int64_t, 4> aStrides, bStrides;
      if (failed(getStridesAndOffset(aT, aStrides, aOffset)) ||
          failed(getStridesAndOffset(bT, bStrides, bOffset)) ||
          aStrides.size() != bStrides.size())
        return false;

      // Strides along a dimension/offset are compatible if the value in the
      // source memref is static and the value in the target memref is the
      // same. They are also compatible if either one is dynamic (see
      // description of MemRefCastOp for details).
      auto checkCompatible = [](int64_t a, int64_t b) {
        return (a == MemRefType::getDynamicStrideOrOffset() ||
                b == MemRefType::getDynamicStrideOrOffset() || a == b);
      };
      if (!checkCompatible(aOffset, bOffset))
        return false;
      for (auto aStride : enumerate(aStrides))
        if (!checkCompatible(aStride.value(), bStrides[aStride.index()]))
          return false;
    }
    if (aT.getMemorySpace() != bT.getMemorySpace())
      return false;

    // They must have the same rank, and any specified dimensions must match.
    if (aT.getRank() != bT.getRank())
      return false;

    for (unsigned i = 0, e = aT.getRank(); i != e; ++i) {
      int64_t aDim = aT.getDimSize(i), bDim = bT.getDimSize(i);
      if (aDim != -1 && bDim != -1 && aDim != bDim)
        return false;
    }
    return true;
  } else {
    if (!aT && !uaT)
      return false;
    if (!bT && !ubT)
      return false;
    // Unranked to unranked casting is unsupported
    if (uaT && ubT)
      return false;

    auto aEltType = (aT) ? aT.getElementType() : uaT.getElementType();
    auto bEltType = (bT) ? bT.getElementType() : ubT.getElementType();
    if (aEltType != bEltType)
      return false;

    auto aMemSpace = (aT) ? aT.getMemorySpace() : uaT.getMemorySpace();
    auto bMemSpace = (bT) ? bT.getMemorySpace() : ubT.getMemorySpace();
    if (aMemSpace != bMemSpace)
      return false;

    return true;
  }

  return false;
}

OpFoldResult MemRefCastOp::fold(ArrayRef<Attribute> operands) {
  return impl::foldCastOp(*this);
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
// PrefetchOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, PrefetchOp op) {
  p << PrefetchOp::getOperationName() << " " << op.memref() << '[';
  p.printOperands(op.indices());
  p << ']' << ", " << (op.isWrite() ? "write" : "read");
  p << ", locality<" << op.localityHint();
  p << ">, " << (op.isDataCache() ? "data" : "instr");
  p.printOptionalAttrDict(
      op.getAttrs(),
      /*elidedAttrs=*/{"localityHint", "isWrite", "isDataCache"});
  p << " : " << op.getMemRefType();
}

static ParseResult parsePrefetchOp(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::OperandType memrefInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  IntegerAttr localityHint;
  MemRefType type;
  StringRef readOrWrite, cacheType;

  auto indexTy = parser.getBuilder().getIndexType();
  auto i32Type = parser.getBuilder().getIntegerType(32);
  if (parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseKeyword(&readOrWrite) ||
      parser.parseComma() || parser.parseKeyword("locality") ||
      parser.parseLess() ||
      parser.parseAttribute(localityHint, i32Type, "localityHint",
                            result.attributes) ||
      parser.parseGreater() || parser.parseComma() ||
      parser.parseKeyword(&cacheType) || parser.parseColonType(type) ||
      parser.resolveOperand(memrefInfo, type, result.operands) ||
      parser.resolveOperands(indexInfo, indexTy, result.operands))
    return failure();

  if (!readOrWrite.equals("read") && !readOrWrite.equals("write"))
    return parser.emitError(parser.getNameLoc(),
                            "rw specifier has to be 'read' or 'write'");
  result.addAttribute(
      PrefetchOp::getIsWriteAttrName(),
      parser.getBuilder().getBoolAttr(readOrWrite.equals("write")));

  if (!cacheType.equals("data") && !cacheType.equals("instr"))
    return parser.emitError(parser.getNameLoc(),
                            "cache type has to be 'data' or 'instr'");

  result.addAttribute(
      PrefetchOp::getIsDataCacheAttrName(),
      parser.getBuilder().getBoolAttr(cacheType.equals("data")));

  return success();
}

static LogicalResult verify(PrefetchOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("too few indices");

  return success();
}

LogicalResult PrefetchOp::fold(ArrayRef<Attribute> cstOperands,
                               SmallVectorImpl<OpFoldResult> &results) {
  // prefetch(memrefcast) -> prefetch
  return foldMemRefCast(*this);
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
  auto function = cast<FuncOp>(op.getParentOp());

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

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  auto condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(condition, m_One()))
    return getTrueValue();

  // select false, %0, %1 => %1
  if (matchPattern(condition, m_Zero()))
    return getFalseValue();
  return nullptr;
}

static void print(OpAsmPrinter &p, SelectOp op) {
  p << "select " << op.getOperands();
  p.printOptionalAttrDict(op.getAttrs());
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
bool SIToFPOp::areCastCompatible(Type a, Type b) {
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
// StoreOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(StoreOp op) {
  if (op.getNumOperands() != 2 + op.getMemRefType().getRank())
    return op.emitOpError("store index operand count not equal to memref rank");

  return success();
}

LogicalResult StoreOp::fold(ArrayRef<Attribute> cstOperands,
                            SmallVectorImpl<OpFoldResult> &results) {
  /// store(memrefcast) -> store
  return foldMemRefCast(*this);
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

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

// uitofp is applicable from integer types to float types.
bool UIToFPOp::areCastCompatible(Type a, Type b) {
  if (a.isSignlessInteger() && b.isa<FloatType>())
    return true;
  return areVectorCastSimpleCompatible(a, b, areCastCompatible);
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

/// Print a list with either (1) the static integer value in `arrayAttr` if
/// `isDynamic` evaluates to false or (2) the next value otherwise.
/// This allows idiomatic printing of mixed value and integer attributes in a
/// list. E.g. `[%arg0, 7, 42, %arg42]`.
static void printSubViewListOfOperandsOrIntegers(
    OpAsmPrinter &p, ValueRange values, ArrayAttr arrayAttr,
    llvm::function_ref<bool(int64_t)> isDynamic) {
  p << "[";
  unsigned idx = 0;
  llvm::interleaveComma(arrayAttr, p, [&](Attribute a) {
    int64_t val = a.cast<IntegerAttr>().getInt();
    if (isDynamic(val))
      p << values[idx++];
    else
      p << val;
  });
  p << "] ";
}

/// Parse a mixed list with either (1) static integer values or (2) SSA values.
/// Fill `result` with the integer ArrayAttr named `attrName` where `dynVal`
/// encode the position of SSA values. Add the parsed SSA values to `ssa`
/// in-order.
//
/// E.g. after parsing "[%arg0, 7, 42, %arg42]":
///   1. `result` is filled with the i64 ArrayAttr "[`dynVal`, 7, 42, `dynVal`]"
///   2. `ssa` is filled with "[%arg0, %arg1]".
static ParseResult
parseListOfOperandsOrIntegers(OpAsmParser &parser, OperationState &result,
                              StringRef attrName, int64_t dynVal,
                              SmallVectorImpl<OpAsmParser::OperandType> &ssa) {
  if (failed(parser.parseLSquare()))
    return failure();
  // 0-D.
  if (succeeded(parser.parseOptionalRSquare()))
    return success();

  SmallVector<int64_t, 4> attrVals;
  while (true) {
    OpAsmParser::OperandType operand;
    auto res = parser.parseOptionalOperand(operand);
    if (res.hasValue() && succeeded(res.getValue())) {
      ssa.push_back(operand);
      attrVals.push_back(dynVal);
    } else {
      Attribute attr;
      NamedAttrList placeholder;
      if (failed(parser.parseAttribute(attr, "_", placeholder)) ||
          !attr.isa<IntegerAttr>())
        return parser.emitError(parser.getNameLoc())
               << "expected SSA value or integer";
      attrVals.push_back(attr.cast<IntegerAttr>().getInt());
    }

    if (succeeded(parser.parseOptionalComma()))
      continue;
    if (failed(parser.parseRSquare()))
      return failure();
    else
      break;
  }

  auto arrayAttr = parser.getBuilder().getI64ArrayAttr(attrVals);
  result.addAttribute(attrName, arrayAttr);
  return success();
}

namespace {
/// Helpers to write more idiomatic operations.
namespace saturated_arith {
struct Wrapper {
  explicit Wrapper(int64_t v) : v(v) {}
  operator int64_t() { return v; }
  int64_t v;
};
Wrapper operator+(Wrapper a, int64_t b) {
  if (ShapedType::isDynamicStrideOrOffset(a) ||
      ShapedType::isDynamicStrideOrOffset(b))
    return Wrapper(ShapedType::kDynamicStrideOrOffset);
  return Wrapper(a.v + b);
}
Wrapper operator*(Wrapper a, int64_t b) {
  if (ShapedType::isDynamicStrideOrOffset(a) ||
      ShapedType::isDynamicStrideOrOffset(b))
    return Wrapper(ShapedType::kDynamicStrideOrOffset);
  return Wrapper(a.v * b);
}
} // end namespace saturated_arith
} // end namespace

/// A subview result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubViewOp::inferResultType(MemRefType sourceMemRefType,
                                ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes,
                                ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceMemRefType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank &&
         "unexpected staticOffsets size mismatch");
  assert(staticSizes.size() == rank && "unexpected staticSizes size mismatch");
  assert(staticStrides.size() == rank &&
         "unexpected staticStrides size mismatch");

  // Extract source offset and strides.
  int64_t sourceOffset;
  SmallVector<int64_t, 4> sourceStrides;
  auto res = getStridesAndOffset(sourceMemRefType, sourceStrides, sourceOffset);
  assert(succeeded(res) && "SubViewOp expected strided memref type");
  (void)res;

  // Compute target offset whose value is:
  //   `sourceOffset + sum_i(staticOffset_i * sourceStrides_i)`.
  int64_t targetOffset = sourceOffset;
  for (auto it : llvm::zip(staticOffsets, sourceStrides)) {
    auto staticOffset = std::get<0>(it), targetStride = std::get<1>(it);
    using namespace saturated_arith;
    targetOffset = Wrapper(targetOffset) + Wrapper(staticOffset) * targetStride;
  }

  // Compute target stride whose value is:
  //   `sourceStrides_i * staticStrides_i`.
  SmallVector<int64_t, 4> targetStrides;
  targetStrides.reserve(staticOffsets.size());
  for (auto it : llvm::zip(sourceStrides, staticStrides)) {
    auto sourceStride = std::get<0>(it), staticStride = std::get<1>(it);
    using namespace saturated_arith;
    targetStrides.push_back(Wrapper(sourceStride) * staticStride);
  }

  // The type is now known.
  return MemRefType::get(
      staticSizes, sourceMemRefType.getElementType(),
      makeStridedLinearLayoutMap(targetStrides, targetOffset,
                                 sourceMemRefType.getContext()),
      sourceMemRefType.getMemorySpace());
}

/// Print SubViewOp in the form:
/// ```
///   subview ssa-name `[` offset-list `]` `[` size-list `]` `[` stride-list `]`
///     `:` strided-memref-type `to` strided-memref-type
/// ```
template <typename OpType>
static void printOpWithOffsetsSizesAndStrides(
    OpAsmPrinter &p, OpType op,
    llvm::function_ref<void(OpAsmPrinter &p, OpType op)> printExtraOperands =
        [](OpAsmPrinter &p, OpType op) {},
    StringLiteral resultTypeKeyword = "to") {
  int stdDotLen = StandardOpsDialect::getDialectNamespace().size() + 1;
  p << op.getOperation()->getName().getStringRef().drop_front(stdDotLen) << ' ';
  p << op.source();
  printExtraOperands(p, op);
  printSubViewListOfOperandsOrIntegers(p, op.offsets(), op.static_offsets(),
                                       ShapedType::isDynamicStrideOrOffset);
  printSubViewListOfOperandsOrIntegers(p, op.sizes(), op.static_sizes(),
                                       ShapedType::isDynamic);
  printSubViewListOfOperandsOrIntegers(p, op.strides(), op.static_strides(),
                                       ShapedType::isDynamicStrideOrOffset);
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{OpType::getSpecialAttrNames()});
  p << " : " << op.getSourceType() << " " << resultTypeKeyword << " "
    << op.getType();
}

static void print(OpAsmPrinter &p, SubViewOp op) {
  return printOpWithOffsetsSizesAndStrides<SubViewOp>(p, op);
}

/// Parse of the form:
/// ```
///   `name` ssa-name (extra-operands)?
///       `[` offset-list `]` `[` size-list `]` `[` stride-list `]`
///     `:` strided-memref-type `resultTypeKeyword strided-memref-type
/// ```
template <typename OpType>
static ParseResult parseOpWithOffsetsSizesAndStrides(
    OpAsmParser &parser, OperationState &result,
    std::function<ParseResult(OpAsmParser &p,
                              OpAsmParser::OperandType &dstInfo)>
        parseExtraOperand = nullptr,
    StringLiteral resultTypeKeyword = "to") {
  OpAsmParser::OperandType srcInfo, dstInfo;
  SmallVector<OpAsmParser::OperandType, 4> offsetsInfo, sizesInfo, stridesInfo;
  auto indexType = parser.getBuilder().getIndexType();
  Type srcType, dstType;
  if (parser.parseOperand(srcInfo))
    return failure();
  if (parseExtraOperand && parseExtraOperand(parser, dstInfo))
    return failure();
  if (parseListOfOperandsOrIntegers(
          parser, result, OpType::getStaticOffsetsAttrName(),
          ShapedType::kDynamicStrideOrOffset, offsetsInfo) ||
      parseListOfOperandsOrIntegers(parser, result,
                                    OpType::getStaticSizesAttrName(),
                                    ShapedType::kDynamicSize, sizesInfo) ||
      parseListOfOperandsOrIntegers(
          parser, result, OpType::getStaticStridesAttrName(),
          ShapedType::kDynamicStrideOrOffset, stridesInfo))
    return failure();

  // Handle segment sizes.
  auto b = parser.getBuilder();
  SmallVector<int, 4> segmentSizes = {1, static_cast<int>(offsetsInfo.size()),
                                      static_cast<int>(sizesInfo.size()),
                                      static_cast<int>(stridesInfo.size())};
  // If we parse an extra operand it needs to appear in the segmentSizes
  if (parseExtraOperand)
    segmentSizes.insert(segmentSizes.begin(), 1);
  result.addAttribute(OpType::getOperandSegmentSizeAttr(),
                      b.getI32VectorAttr(segmentSizes));

  return failure(
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.parseKeywordType(resultTypeKeyword.str().c_str(), dstType) ||
      parser.resolveOperand(srcInfo, srcType, result.operands) ||
      (parseExtraOperand &&
       parser.resolveOperand(dstInfo, dstType, result.operands)) ||
      parser.resolveOperands(offsetsInfo, indexType, result.operands) ||
      parser.resolveOperands(sizesInfo, indexType, result.operands) ||
      parser.resolveOperands(stridesInfo, indexType, result.operands) ||
      parser.addTypeToList(dstType, result.types));
}

static ParseResult parseSubViewOp(OpAsmParser &parser, OperationState &result) {
  return parseOpWithOffsetsSizesAndStrides<SubViewOp>(parser, result);
}

void mlir::SubViewOp::build(OpBuilder &b, OperationState &result, Value source,
                            ArrayRef<int64_t> staticOffsets,
                            ArrayRef<int64_t> staticSizes,
                            ArrayRef<int64_t> staticStrides, ValueRange offsets,
                            ValueRange sizes, ValueRange strides,
                            ArrayRef<NamedAttribute> attrs) {
  auto sourceMemRefType = source.getType().cast<MemRefType>();
  auto resultType = inferResultType(sourceMemRefType, staticOffsets,
                                    staticSizes, staticStrides);
  build(b, result, resultType, source, offsets, sizes, strides,
        b.getI64ArrayAttr(staticOffsets), b.getI64ArrayAttr(staticSizes),
        b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build a SubViewOp with all dynamic entries: `staticOffsets`, `staticSizes`
/// and `staticStrides` are  automatically filled with source-memref-rank
/// sentinel values that encode dynamic entries.
void mlir::SubViewOp::build(OpBuilder &b, OperationState &result, Value source,
                            ValueRange offsets, ValueRange sizes,
                            ValueRange strides,
                            ArrayRef<NamedAttribute> attrs) {
  auto sourceMemRefType = source.getType().cast<MemRefType>();
  unsigned rank = sourceMemRefType.getRank();
  SmallVector<int64_t, 4> staticOffsetsVector;
  staticOffsetsVector.assign(rank, ShapedType::kDynamicStrideOrOffset);
  SmallVector<int64_t, 4> staticSizesVector;
  staticSizesVector.assign(rank, ShapedType::kDynamicSize);
  SmallVector<int64_t, 4> staticStridesVector;
  staticStridesVector.assign(rank, ShapedType::kDynamicStrideOrOffset);
  build(b, result, source, staticOffsetsVector, staticSizesVector,
        staticStridesVector, offsets, sizes, strides, attrs);
}

/// Build a SubViewOp as above but with custom result type.
void mlir::SubViewOp::build(OpBuilder &b, OperationState &result,
                            MemRefType resultType, Value source,
                            ArrayRef<int64_t> staticOffsets,
                            ArrayRef<int64_t> staticSizes,
                            ArrayRef<int64_t> staticStrides, ValueRange offsets,
                            ValueRange sizes, ValueRange strides,
                            ArrayRef<NamedAttribute> attrs) {
  build(b, result, resultType, source, offsets, sizes, strides,
        b.getI64ArrayAttr(staticOffsets), b.getI64ArrayAttr(staticSizes),
        b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build a SubViewOp as above but with custom result type.
void mlir::SubViewOp::build(OpBuilder &b, OperationState &result,
                            MemRefType resultType, Value source,
                            ValueRange offsets, ValueRange sizes,
                            ValueRange strides,
                            ArrayRef<NamedAttribute> attrs) {
  auto sourceMemRefType = source.getType().cast<MemRefType>();
  unsigned rank = sourceMemRefType.getRank();
  SmallVector<int64_t, 4> staticOffsetsVector;
  staticOffsetsVector.assign(rank, ShapedType::kDynamicStrideOrOffset);
  SmallVector<int64_t, 4> staticSizesVector;
  staticSizesVector.assign(rank, ShapedType::kDynamicSize);
  SmallVector<int64_t, 4> staticStridesVector;
  staticStridesVector.assign(rank, ShapedType::kDynamicStrideOrOffset);
  build(b, result, resultType, source, staticOffsetsVector, staticSizesVector,
        staticStridesVector, offsets, sizes, strides, attrs);
}

/// For ViewLikeOpInterface.
Value SubViewOp::getViewSource() { return source(); }

/// Verify that a particular offset/size/stride static attribute is well-formed.
template <typename OpType>
static LogicalResult verifyOpWithOffsetSizesAndStridesPart(
    OpType op, StringRef name, StringRef attrName, ArrayAttr attr,
    llvm::function_ref<bool(int64_t)> isDynamic, ValueRange values) {
  /// Check static and dynamic offsets/sizes/strides breakdown.
  if (attr.size() != op.getSourceRank())
    return op.emitError("expected ")
           << op.getSourceRank() << " " << name << " values";
  unsigned expectedNumDynamicEntries =
      llvm::count_if(attr.getValue(), [&](Attribute attr) {
        return isDynamic(attr.cast<IntegerAttr>().getInt());
      });
  if (values.size() != expectedNumDynamicEntries)
    return op.emitError("expected ")
           << expectedNumDynamicEntries << " dynamic " << name << " values";
  return success();
}

/// Helper function extracts int64_t from the assumedArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

/// Checks if `original` Type type can be rank reduced to `reduced` type.
/// This function is slight variant of `is subsequence` algorithm where
/// not matching dimension must be 1.
static bool isRankReducedType(Type originalType, Type reducedType) {
  if (originalType == reducedType)
    return true;
  if (!originalType.isa<RankedTensorType>() && !originalType.isa<MemRefType>())
    return true;
  if (originalType.isa<RankedTensorType>() &&
      !reducedType.isa<RankedTensorType>())
    return true;
  if (originalType.isa<MemRefType>() && !reducedType.isa<MemRefType>())
    return true;

  ShapedType originalShapedType = originalType.cast<ShapedType>();
  ShapedType reducedShapedType = reducedType.cast<ShapedType>();

  // Rank and size logic is valid for all ShapedTypes.
  ArrayRef<int64_t> originalShape = originalShapedType.getShape();
  ArrayRef<int64_t> reducedShape = reducedShapedType.getShape();
  unsigned originalRank = originalShape.size(),
           reducedRank = reducedShape.size();
  if (reducedRank > originalRank)
    return false;

  unsigned reducedIdx = 0;
  SmallVector<bool, 4> keepMask(originalRank);
  for (unsigned originalIdx = 0; originalIdx < originalRank; ++originalIdx) {
    // -2 is never used as a dim size so it will never match.
    int reducedVal = reducedIdx < reducedRank ? reducedShape[reducedIdx] : -2;
    // Skip matching dims greedily.
    if ((keepMask[originalIdx] = originalShape[originalIdx] == reducedVal))
      reducedIdx++;
    // 1 is the only non-matching allowed.
    else if (originalShape[originalIdx] != 1)
      return false;
  }
  // Must match the reduced rank.
  if (reducedIdx != reducedRank)
    return false;

  // We are done for the tensor case.
  if (originalType.isa<RankedTensorType>())
    return true;

  // Strided layout logic is relevant for MemRefType only.
  MemRefType original = originalType.cast<MemRefType>();
  MemRefType reduced = reducedType.cast<MemRefType>();
  MLIRContext *c = original.getContext();
  int64_t originalOffset, symCounter = 0, dimCounter = 0;
  SmallVector<int64_t, 4> originalStrides;
  getStridesAndOffset(original, originalStrides, originalOffset);
  auto getSymbolOrConstant = [&](int64_t offset) {
    return offset == ShapedType::kDynamicStrideOrOffset
               ? getAffineSymbolExpr(symCounter++, c)
               : getAffineConstantExpr(offset, c);
  };

  AffineExpr expr = getSymbolOrConstant(originalOffset);
  for (unsigned i = 0, e = originalStrides.size(); i < e; i++) {
    if (keepMask[i])
      expr = expr + getSymbolOrConstant(originalStrides[i]) *
                        getAffineDimExpr(dimCounter++, c);
  }

  auto reducedMap = AffineMap::get(dimCounter, symCounter, expr, c);
  return original.getElementType() == reduced.getElementType() &&
         original.getMemorySpace() == reduced.getMemorySpace() &&
         (reduced.getAffineMaps().empty() ||
          reducedMap == reduced.getAffineMaps().front());
}

template <typename OpType>
static LogicalResult verifyOpWithOffsetSizesAndStrides(OpType op) {
  // Verify static attributes offsets/sizes/strides.
  if (failed(verifyOpWithOffsetSizesAndStridesPart(
          op, "offset", op.getStaticOffsetsAttrName(), op.static_offsets(),
          ShapedType::isDynamicStrideOrOffset, op.offsets())))
    return failure();

  if (failed(verifyOpWithOffsetSizesAndStridesPart(
          op, "size", op.getStaticSizesAttrName(), op.static_sizes(),
          ShapedType::isDynamic, op.sizes())))
    return failure();
  if (failed(verifyOpWithOffsetSizesAndStridesPart(
          op, "stride", op.getStaticStridesAttrName(), op.static_strides(),
          ShapedType::isDynamicStrideOrOffset, op.strides())))
    return failure();
  return success();
}

/// Verifier for SubViewOp.
static LogicalResult verify(SubViewOp op) {
  MemRefType baseType = op.getSourceType();
  MemRefType subViewType = op.getType();

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != subViewType.getMemorySpace())
    return op.emitError("different memory spaces specified for base memref "
                        "type ")
           << baseType << " and subview memref type " << subViewType;

  // Verify that the base memref type has a strided layout map.
  if (!isStrided(baseType))
    return op.emitError("base type ") << baseType << " is not strided";

  if (failed(verifyOpWithOffsetSizesAndStrides(op)))
    return failure();

  // Verify result type against inferred type.
  auto expectedType = SubViewOp::inferResultType(
      baseType, extractFromI64ArrayAttr(op.static_offsets()),
      extractFromI64ArrayAttr(op.static_sizes()),
      extractFromI64ArrayAttr(op.static_strides()));
  if (!isRankReducedType(expectedType, subViewType))
    return op.emitError("expected result type to be ")
           << expectedType << " or a rank-reduced version.";

  return success();
}

raw_ostream &mlir::operator<<(raw_ostream &os, Range &range) {
  return os << "range " << range.offset << ":" << range.size << ":"
            << range.stride;
}

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
template <typename OpType>
static SmallVector<Range, 8> getOrCreateRangesImpl(OpType op, OpBuilder &b,
                                                   Location loc) {
  SmallVector<Range, 8> res;
  unsigned rank = op.getSourceRank();
  res.reserve(rank);
  for (unsigned idx = 0; idx < rank; ++idx) {
    Value offset =
        op.isDynamicOffset(idx)
            ? op.getDynamicOffset(idx)
            : b.create<ConstantIndexOp>(loc, op.getStaticOffset(idx));
    Value size = op.isDynamicSize(idx)
                     ? op.getDynamicSize(idx)
                     : b.create<ConstantIndexOp>(loc, op.getStaticSize(idx));
    Value stride =
        op.isDynamicStride(idx)
            ? op.getDynamicStride(idx)
            : b.create<ConstantIndexOp>(loc, op.getStaticStride(idx));
    res.emplace_back(Range{offset, size, stride});
  }
  return res;
}

SmallVector<Range, 8> SubViewOp::getOrCreateRanges(OpBuilder &b, Location loc) {
  return ::getOrCreateRangesImpl(*this, b, loc);
}

namespace {

/// Take a list of `values` with potential new constant to extract and a list
/// of `constantValues` with`values.size()` sentinel that evaluate to true by
/// applying `isDynamic`.
/// Detects the `values` produced by a ConstantIndexOp and places the new
/// constant in place of the corresponding sentinel value.
void canonicalizeSubViewPart(SmallVectorImpl<Value> &values,
                             SmallVectorImpl<int64_t> &constantValues,
                             llvm::function_ref<bool(int64_t)> isDynamic) {
  bool hasNewStaticValue = llvm::any_of(
      values, [](Value val) { return matchPattern(val, m_ConstantIndex()); });
  if (hasNewStaticValue) {
    for (unsigned cstIdx = 0, valIdx = 0, e = constantValues.size();
         cstIdx != e; ++cstIdx) {
      // Was already static, skip.
      if (!isDynamic(constantValues[cstIdx]))
        continue;
      // Newly static, move from Value to constant.
      if (matchPattern(values[valIdx], m_ConstantIndex())) {
        constantValues[cstIdx] =
            cast<ConstantIndexOp>(values[valIdx].getDefiningOp()).getValue();
        // Erase for impl. simplicity. Reverse iterator if we really must.
        values.erase(std::next(values.begin(), valIdx));
        continue;
      }
      // Remains dynamic move to next value.
      ++valIdx;
    }
  }
}

static void replaceWithNewOp(PatternRewriter &rewriter, SubViewOp op,
                             SubViewOp newOp) {
  rewriter.replaceOpWithNewOp<MemRefCastOp>(op, newOp, op.getType());
}

static void replaceWithNewOp(PatternRewriter &rewriter, SubTensorOp op,
                             SubTensorOp newOp) {
  rewriter.replaceOpWithNewOp<TensorCastOp>(op, newOp, op.getType());
}

/// Pattern to rewrite a subview op with constant arguments.
template <typename OpType>
class OpWithOffsetSizesAndStridesConstantArgumentFolder final
    : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // No constant operand, just return;
    if (llvm::none_of(op.getOperands(), [](Value operand) {
          return matchPattern(operand, m_ConstantIndex());
        }))
      return failure();

    // At least one of offsets/sizes/strides is a new constant.
    // Form the new list of operands and constant attributes from the existing.
    SmallVector<Value, 8> newOffsets(op.offsets());
    SmallVector<int64_t, 8> newStaticOffsets =
        extractFromI64ArrayAttr(op.static_offsets());
    assert(newStaticOffsets.size() == op.getSourceRank());
    canonicalizeSubViewPart(newOffsets, newStaticOffsets,
                            ShapedType::isDynamicStrideOrOffset);

    SmallVector<Value, 8> newSizes(op.sizes());
    SmallVector<int64_t, 8> newStaticSizes =
        extractFromI64ArrayAttr(op.static_sizes());
    assert(newStaticOffsets.size() == op.getSourceRank());
    canonicalizeSubViewPart(newSizes, newStaticSizes, ShapedType::isDynamic);

    SmallVector<Value, 8> newStrides(op.strides());
    SmallVector<int64_t, 8> newStaticStrides =
        extractFromI64ArrayAttr(op.static_strides());
    assert(newStaticOffsets.size() == op.getSourceRank());
    canonicalizeSubViewPart(newStrides, newStaticStrides,
                            ShapedType::isDynamicStrideOrOffset);

    // Create the new op in canonical form.
    auto newOp = rewriter.create<OpType>(
        op.getLoc(), op.source(), newStaticOffsets, newStaticSizes,
        newStaticStrides, newOffsets, newSizes, newStrides);

    replaceWithNewOp(rewriter, op, newOp);

    return success();
  }
};

} // end anonymous namespace

/// Determines whether MemRefCastOp casts to a more dynamic version of the
/// source memref. This is useful to to fold a memref_cast into a consuming op
/// and implement canonicalization patterns for ops in different dialects that
/// may consume the results of memref_cast operations. Such foldable memref_cast
/// operations are typically inserted as `view` and `subview` ops are
/// canonicalized, to preserve the type compatibility of their uses.
///
/// Returns true when all conditions are met:
/// 1. source and result are ranked memrefs with strided semantics and same
/// element type and rank.
/// 2. each of the source's size, offset or stride has more static information
/// than the corresponding result's size, offset or stride.
///
/// Example 1:
/// ```mlir
///   %1 = memref_cast %0 : memref<8x16xf32> to memref<?x?xf32>
///   %2 = consumer %1 ... : memref<?x?xf32> ...
/// ```
///
/// may fold into:
///
/// ```mlir
///   %2 = consumer %0 ... : memref<8x16xf32> ...
/// ```
///
/// Example 2:
/// ```
///   %1 = memref_cast %0 : memref<?x16xf32, affine_map<(i, j)->(16 * i + j)>>
///          to memref<?x?xf32>
///   consumer %1 : memref<?x?xf32> ...
/// ```
///
/// may fold into:
///
/// ```
///   consumer %0 ... : memref<?x16xf32, affine_map<(i, j)->(16 * i + j)>>
/// ```
bool mlir::canFoldIntoConsumerOp(MemRefCastOp castOp) {
  MemRefType sourceType = castOp.source().getType().dyn_cast<MemRefType>();
  MemRefType resultType = castOp.getType().dyn_cast<MemRefType>();

  // Requires ranked MemRefType.
  if (!sourceType || !resultType)
    return false;

  // Requires same elemental type.
  if (sourceType.getElementType() != resultType.getElementType())
    return false;

  // Requires same rank.
  if (sourceType.getRank() != resultType.getRank())
    return false;

  // Only fold casts between strided memref forms.
  int64_t sourceOffset, resultOffset;
  SmallVector<int64_t, 4> sourceStrides, resultStrides;
  if (failed(getStridesAndOffset(sourceType, sourceStrides, sourceOffset)) ||
      failed(getStridesAndOffset(resultType, resultStrides, resultOffset)))
    return false;

  // If cast is towards more static sizes along any dimension, don't fold.
  for (auto it : llvm::zip(sourceType.getShape(), resultType.getShape())) {
    auto ss = std::get<0>(it), st = std::get<1>(it);
    if (ss != st)
      if (MemRefType::isDynamic(ss) && !MemRefType::isDynamic(st))
        return false;
  }

  // If cast is towards more static offset along any dimension, don't fold.
  if (sourceOffset != resultOffset)
    if (MemRefType::isDynamicStrideOrOffset(sourceOffset) &&
        !MemRefType::isDynamicStrideOrOffset(resultOffset))
      return false;

  // If cast is towards more static strides along any dimension, don't fold.
  for (auto it : llvm::zip(sourceStrides, resultStrides)) {
    auto ss = std::get<0>(it), st = std::get<1>(it);
    if (ss != st)
      if (MemRefType::isDynamicStrideOrOffset(ss) &&
          !MemRefType::isDynamicStrideOrOffset(st))
        return false;
  }

  return true;
}

namespace {
/// Pattern to rewrite a subview op with MemRefCast arguments.
/// This essentially pushes memref_cast past its consuming subview when
/// `canFoldIntoConsumerOp` is true.
///
/// Example:
/// ```
///   %0 = memref_cast %V : memref<16x16xf32> to memref<?x?xf32>
///   %1 = subview %0[0, 0][3, 4][1, 1] :
///     memref<?x?xf32> to memref<3x4xf32, offset:?, strides:[?, 1]>
/// ```
/// is rewritten into:
/// ```
///   %0 = subview %V: memref<16x16xf32> to memref<3x4xf32, #[[map0]]>
///   %1 = memref_cast %0: memref<3x4xf32, offset:0, strides:[16, 1]> to
///     memref<3x4xf32, offset:?, strides:[?, 1]>
/// ```
class SubViewOpMemRefCastFolder final : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    // Any constant operand, just return to let SubViewOpConstantFolder kick in.
    if (llvm::any_of(subViewOp.getOperands(), [](Value operand) {
          return matchPattern(operand, m_ConstantIndex());
        }))
      return failure();

    auto castOp = subViewOp.source().getDefiningOp<MemRefCastOp>();
    if (!castOp)
      return failure();

    if (!canFoldIntoConsumerOp(castOp))
      return failure();

    /// Deduce the resultType of the SubViewOp using `inferSubViewResultType` on
    /// the cast source operand type and the SubViewOp static information. This
    /// is the resulting type if the MemRefCastOp were folded.
    Type resultType = SubViewOp::inferResultType(
        castOp.source().getType().cast<MemRefType>(),
        extractFromI64ArrayAttr(subViewOp.static_offsets()),
        extractFromI64ArrayAttr(subViewOp.static_sizes()),
        extractFromI64ArrayAttr(subViewOp.static_strides()));
    Value newSubView = rewriter.create<SubViewOp>(
        subViewOp.getLoc(), resultType, castOp.source(), subViewOp.offsets(),
        subViewOp.sizes(), subViewOp.strides(), subViewOp.static_offsets(),
        subViewOp.static_sizes(), subViewOp.static_strides());
    rewriter.replaceOpWithNewOp<MemRefCastOp>(subViewOp, subViewOp.getType(),
                                              newSubView);
    return success();
  }
};
} // namespace

void SubViewOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<OpWithOffsetSizesAndStridesConstantArgumentFolder<SubViewOp>,
                 SubViewOpMemRefCastFolder>(context);
}

//===----------------------------------------------------------------------===//
// SubTensorOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, SubTensorOp op) {
  return printOpWithOffsetsSizesAndStrides<SubTensorOp>(p, op);
}

static ParseResult parseSubTensorOp(OpAsmParser &parser,
                                    OperationState &result) {
  return parseOpWithOffsetsSizesAndStrides<SubTensorOp>(parser, result);
}

/// A subtensor result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubTensorOp::inferResultType(RankedTensorType sourceRankedTensorType,
                                  ArrayRef<int64_t> staticOffsets,
                                  ArrayRef<int64_t> staticSizes,
                                  ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceRankedTensorType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank &&
         "unexpected staticOffsets size mismatch");
  assert(staticSizes.size() == rank && "unexpected staticSizes size mismatch");
  assert(staticStrides.size() == rank &&
         "unexpected staticStrides size mismatch");
  return RankedTensorType::get(staticSizes,
                               sourceRankedTensorType.getElementType());
}

void mlir::SubTensorOp::build(OpBuilder &b, OperationState &result,
                              Value source, ArrayRef<int64_t> staticOffsets,
                              ArrayRef<int64_t> staticSizes,
                              ArrayRef<int64_t> staticStrides,
                              ValueRange offsets, ValueRange sizes,
                              ValueRange strides,
                              ArrayRef<NamedAttribute> attrs) {
  auto sourceRankedTensorType = source.getType().cast<RankedTensorType>();
  auto resultType = inferResultType(sourceRankedTensorType, staticOffsets,
                                    staticSizes, staticStrides);
  build(b, result, resultType, source, offsets, sizes, strides,
        b.getI64ArrayAttr(staticOffsets), b.getI64ArrayAttr(staticSizes),
        b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build a SubTensorOp with all dynamic entries: `staticOffsets`, `staticSizes`
/// and `staticStrides` are automatically filled with sentinel values that
/// encode dynamic entries.
void mlir::SubTensorOp::build(OpBuilder &b, OperationState &result,
                              Value source, ValueRange offsets,
                              ValueRange sizes, ValueRange strides,
                              ArrayRef<NamedAttribute> attrs) {
  auto sourceRankedTensorType = source.getType().cast<RankedTensorType>();
  unsigned rank = sourceRankedTensorType.getRank();
  SmallVector<int64_t, 4> staticOffsetsVector(
      rank, ShapedType::kDynamicStrideOrOffset);
  SmallVector<int64_t, 4> staticSizesVector(rank, ShapedType::kDynamicSize);
  SmallVector<int64_t, 4> staticStridesVector(
      rank, ShapedType::kDynamicStrideOrOffset);
  build(b, result, source, staticOffsetsVector, staticSizesVector,
        staticStridesVector, offsets, sizes, strides, attrs);
}

SmallVector<Range, 8> SubTensorOp::getOrCreateRanges(OpBuilder &b,
                                                     Location loc) {
  return ::getOrCreateRangesImpl(*this, b, loc);
}

/// Verifier for SubTensorOp.
static LogicalResult verify(SubTensorOp op) {
  if (failed(verifyOpWithOffsetSizesAndStrides(op)))
    return failure();

  // Verify result type against inferred type.
  auto expectedType = SubTensorOp::inferResultType(
      op.getSourceType(), extractFromI64ArrayAttr(op.static_offsets()),
      extractFromI64ArrayAttr(op.static_sizes()),
      extractFromI64ArrayAttr(op.static_strides()));
  if (!isRankReducedType(expectedType, op.getType()))
    return op.emitError("expected result type to be ")
           << expectedType << " or a rank-reduced version.";

  return success();
}

void SubTensorOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results
      .insert<OpWithOffsetSizesAndStridesConstantArgumentFolder<SubTensorOp>>(
          context);
}

//===----------------------------------------------------------------------===//
// SubTensorInsertOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, SubTensorInsertOp op) {
  return printOpWithOffsetsSizesAndStrides<SubTensorInsertOp>(
      p, op,
      [](OpAsmPrinter &p, SubTensorInsertOp op) { p << " into " << op.dest(); },
      /*resultTypeKeyword=*/"into");
}

static ParseResult parseSubTensorInsertOp(OpAsmParser &parser,
                                          OperationState &result) {
  return parseOpWithOffsetsSizesAndStrides<SubTensorInsertOp>(
      parser, result,
      [](OpAsmParser &parser, OpAsmParser::OperandType &dstInfo) {
        return failure(parser.parseKeyword("into") ||
                       parser.parseOperand(dstInfo));
      },
      "into");
}

void mlir::SubTensorInsertOp::build(
    OpBuilder &b, OperationState &result, Value source, Value dest,
    ArrayRef<int64_t> staticOffsets, ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides, ValueRange offsets, ValueRange sizes,
    ValueRange strides, ArrayRef<NamedAttribute> attrs) {
  build(b, result, dest.getType(), source, dest, offsets, sizes, strides,
        b.getI64ArrayAttr(staticOffsets), b.getI64ArrayAttr(staticSizes),
        b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// Build a SubViewOp with all dynamic entries: `staticOffsets`, `staticSizes`
/// and `staticStrides` are  automatically filled with source-memref-rank
/// sentinel values that encode dynamic entries.
void mlir::SubTensorInsertOp::build(OpBuilder &b, OperationState &result,
                                    Value source, Value dest,
                                    ValueRange offsets, ValueRange sizes,
                                    ValueRange strides,
                                    ArrayRef<NamedAttribute> attrs) {
  auto sourceRankedTensorType = source.getType().cast<RankedTensorType>();
  unsigned rank = sourceRankedTensorType.getRank();
  SmallVector<int64_t, 4> staticOffsetsVector(
      rank, ShapedType::kDynamicStrideOrOffset);
  SmallVector<int64_t, 4> staticSizesVector(rank, ShapedType::kDynamicSize);
  SmallVector<int64_t, 4> staticStridesVector(
      rank, ShapedType::kDynamicStrideOrOffset);
  build(b, result, source, dest, staticOffsetsVector, staticSizesVector,
        staticStridesVector, offsets, sizes, strides, attrs);
}

SmallVector<Range, 8> SubTensorInsertOp::getOrCreateRanges(OpBuilder &b,
                                                           Location loc) {
  return ::getOrCreateRangesImpl(*this, b, loc);
}

/// Verifier for SubViewOp.
static LogicalResult verify(SubTensorInsertOp op) {
  if (failed(verifyOpWithOffsetSizesAndStrides(op)))
    return failure();
  if (op.getType() != op.dest().getType())
    return op.emitError("expected result type to be ") << op.dest().getType();
  return success();
}

//===----------------------------------------------------------------------===//
// TensorCastOp
//===----------------------------------------------------------------------===//

bool TensorCastOp::areCastCompatible(Type a, Type b) {
  auto aT = a.dyn_cast<TensorType>();
  auto bT = b.dyn_cast<TensorType>();
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return succeeded(verifyCompatibleShape(aT, bT));
}

OpFoldResult TensorCastOp::fold(ArrayRef<Attribute> operands) {
  return impl::foldCastOp(*this);
}

/// Compute a TensorType that has the joined shape knowledge of the two
/// given TensorTypes. The element types need to match.
static TensorType joinShapes(TensorType one, TensorType two) {
  assert(one.getElementType() == two.getElementType());

  if (!one.hasRank())
    return two;
  if (!two.hasRank())
    return one;

  int64_t rank = one.getRank();
  if (rank != two.getRank())
    return {};

  SmallVector<int64_t, 4> join;
  join.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (one.isDynamicDim(i)) {
      join.push_back(two.getDimSize(i));
      continue;
    }
    if (two.isDynamicDim(i)) {
      join.push_back(one.getDimSize(i));
      continue;
    }
    if (one.getDimSize(i) != two.getDimSize(i))
      return {};
    join.push_back(one.getDimSize(i));
  }
  return RankedTensorType::get(join, one.getElementType());
}

namespace {

/// Replaces chains of two tensor_cast operations by a single tensor_cast
/// operation if doing so does not remove runtime constraints.
struct ChainedTensorCast : public OpRewritePattern<TensorCastOp> {
  using OpRewritePattern<TensorCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorCastOp tensorCast,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOperand =
        tensorCast.getOperand().getDefiningOp<TensorCastOp>();

    if (!tensorCastOperand)
      return failure();

    auto sourceType =
        tensorCastOperand.getOperand().getType().cast<TensorType>();
    auto intermediateType = tensorCastOperand.getType().cast<TensorType>();
    auto resultType = tensorCast.getType().cast<TensorType>();

    // We can remove the intermediate cast if joining all three produces the
    // same result as just joining the source and result shapes.
    auto firstJoin =
        joinShapes(joinShapes(sourceType, intermediateType), resultType);

    // The join might not exist if the cast sequence would fail at runtime.
    if (!firstJoin)
      return failure();

    // The newJoin always exists if the above join exists, it might just contain
    // less information. If so, we cannot drop the intermediate cast, as doing
    // so would remove runtime checks.
    auto newJoin = joinShapes(sourceType, resultType);
    if (firstJoin != newJoin)
      return failure();

    rewriter.replaceOpWithNewOp<TensorCastOp>(tensorCast, resultType,
                                              tensorCastOperand.getOperand());
    return success();
  }
};

} // namespace

void TensorCastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ChainedTensorCast>(context);
}

//===----------------------------------------------------------------------===//
// Helpers for Tensor[Load|Store]Op
//===----------------------------------------------------------------------===//

static Type getTensorTypeFromMemRefType(Type type) {
  if (auto memref = type.dyn_cast<MemRefType>())
    return RankedTensorType::get(memref.getShape(), memref.getElementType());
  if (auto memref = type.dyn_cast<UnrankedMemRefType>())
    return UnrankedTensorType::get(memref.getElementType());
  return NoneType::get(type.getContext());
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
// ViewOp
//===----------------------------------------------------------------------===//

static ParseResult parseViewOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType srcInfo;
  SmallVector<OpAsmParser::OperandType, 1> offsetInfo;
  SmallVector<OpAsmParser::OperandType, 4> sizesInfo;
  auto indexType = parser.getBuilder().getIndexType();
  Type srcType, dstType;
  llvm::SMLoc offsetLoc;
  if (parser.parseOperand(srcInfo) || parser.getCurrentLocation(&offsetLoc) ||
      parser.parseOperandList(offsetInfo, OpAsmParser::Delimiter::Square))
    return failure();

  if (offsetInfo.size() != 1)
    return parser.emitError(offsetLoc) << "expects 1 offset operand";

  return failure(
      parser.parseOperandList(sizesInfo, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(srcInfo, srcType, result.operands) ||
      parser.resolveOperands(offsetInfo, indexType, result.operands) ||
      parser.resolveOperands(sizesInfo, indexType, result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types));
}

static void print(OpAsmPrinter &p, ViewOp op) {
  p << op.getOperationName() << ' ' << op.getOperand(0) << '[';
  p.printOperand(op.byte_shift());
  p << "][" << op.sizes() << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperand(0).getType() << " to " << op.getType();
}

static LogicalResult verify(ViewOp op) {
  auto baseType = op.getOperand(0).getType().cast<MemRefType>();
  auto viewType = op.getType();

  // The base memref should have identity layout map (or none).
  if (baseType.getAffineMaps().size() > 1 ||
      (baseType.getAffineMaps().size() == 1 &&
       !baseType.getAffineMaps()[0].isIdentity()))
    return op.emitError("unsupported map for base memref type ") << baseType;

  // The result memref should have identity layout map (or none).
  if (viewType.getAffineMaps().size() > 1 ||
      (viewType.getAffineMaps().size() == 1 &&
       !viewType.getAffineMaps()[0].isIdentity()))
    return op.emitError("unsupported map for result memref type ") << viewType;

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != viewType.getMemorySpace())
    return op.emitError("different memory spaces specified for base memref "
                        "type ")
           << baseType << " and view memref type " << viewType;

  // Verify that we have the correct number of sizes for the result type.
  unsigned numDynamicDims = viewType.getNumDynamicDims();
  if (op.sizes().size() != numDynamicDims)
    return op.emitError("incorrect number of size operands for type ")
           << viewType;

  return success();
}

Value ViewOp::getViewSource() { return source(); }

namespace {

struct ViewOpShapeFolder : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp viewOp,
                                PatternRewriter &rewriter) const override {
    // Return if none of the operands are constants.
    if (llvm::none_of(viewOp.getOperands(), [](Value operand) {
          return matchPattern(operand, m_ConstantIndex());
        }))
      return failure();

    // Get result memref type.
    auto memrefType = viewOp.getType();

    // Get offset from old memref view type 'memRefType'.
    int64_t oldOffset;
    SmallVector<int64_t, 4> oldStrides;
    if (failed(getStridesAndOffset(memrefType, oldStrides, oldOffset)))
      return failure();
    assert(oldOffset == 0 && "Expected 0 offset");

    SmallVector<Value, 4> newOperands;

    // Offset cannot be folded into result type.

    // Fold any dynamic dim operands which are produced by a constant.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());

    unsigned dynamicDimPos = 0;
    unsigned rank = memrefType.getRank();
    for (unsigned dim = 0, e = rank; dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (!ShapedType::isDynamic(dimSize)) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = viewOp.sizes()[dynamicDimPos].getDefiningOp();
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.getValue());
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(dimSize);
        newOperands.push_back(viewOp.sizes()[dynamicDimPos]);
      }
      dynamicDimPos++;
    }

    // Create new memref type with constant folded dims.
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType).setShape(newShapeConstants);
    // Nothing new, don't fold.
    if (newMemRefType == memrefType)
      return failure();

    // Create new ViewOp.
    auto newViewOp = rewriter.create<ViewOp>(viewOp.getLoc(), newMemRefType,
                                             viewOp.getOperand(0),
                                             viewOp.byte_shift(), newOperands);
    // Insert a cast so we have the same type as the old memref type.
    rewriter.replaceOpWithNewOp<MemRefCastOp>(viewOp, newViewOp,
                                              viewOp.getType());
    return success();
  }
};

struct ViewOpMemrefCastFolder : public OpRewritePattern<ViewOp> {
  using OpRewritePattern<ViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ViewOp viewOp,
                                PatternRewriter &rewriter) const override {
    Value memrefOperand = viewOp.getOperand(0);
    MemRefCastOp memrefCastOp = memrefOperand.getDefiningOp<MemRefCastOp>();
    if (!memrefCastOp)
      return failure();
    Value allocOperand = memrefCastOp.getOperand();
    AllocOp allocOp = allocOperand.getDefiningOp<AllocOp>();
    if (!allocOp)
      return failure();
    rewriter.replaceOpWithNewOp<ViewOp>(viewOp, viewOp.getType(), allocOperand,
                                        viewOp.byte_shift(), viewOp.sizes());
    return success();
  }
};

} // end anonymous namespace

void ViewOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ViewOpShapeFolder, ViewOpMemrefCastFolder>(context);
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
