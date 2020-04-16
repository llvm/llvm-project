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

StandardOpsDialect::StandardOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
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
    auto cast = dyn_cast_or_null<MemRefCastOp>(operand.get().getDefiningOp());
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
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
  /// and(x,x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// AssumeAlignmentOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AssumeAlignmentOp op) {
  unsigned alignment = op.alignment().getZExtValue();
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
// BranchOp
//===----------------------------------------------------------------------===//

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
} // end anonymous namespace.

Block *BranchOp::getDest() { return getSuccessor(); }

void BranchOp::setDest(Block *block) { return setSuccessor(block); }

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

void BranchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SimplifyBrToBlockWithSinglePred>(context);
}

Optional<OperandRange> BranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return getOperands();
}

bool BranchOp::canEraseSuccessorOperand() { return true; }

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
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, getResultTypes(), getContext());
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
static Type getCheckedI1SameShape(Type type) {
  auto i1Type = IntegerType::get(1, type.getContext());
  if (type.isSignlessIntOrIndexOrFloat())
    return i1Type;
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type);
  return Type();
}

static Type getI1SameShape(Type type) {
  Type res = getCheckedI1SameShape(type);
  assert(res && "expected type with valid i1 shape");
  return res;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

static void buildCmpIOp(Builder *build, OperationState &result,
                        CmpIPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  result.addAttribute(
      CmpIOp::getPredicateAttrName(),
      build->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
// comparison predicates.
static bool applyCmpPredicate(CmpIPredicate predicate, const APInt &lhs,
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

static void buildCmpFOp(Builder *build, OperationState &result,
                        CmpFPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  result.addAttribute(
      CmpFOp::getPredicateAttrName(),
      build->getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
static bool applyCmpPredicate(CmpFPredicate predicate, const APFloat &lhs,
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

  // TODO(gcmn) We could actually do some intelligent things if we know only one
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
/// cond_br true, ^bb1, ^bb2 -> br ^bb1
/// cond_br false, ^bb1, ^bb2 -> br ^bb2
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
} // end anonymous namespace.

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyConstCondBranchPred>(context);
}

Optional<OperandRange> CondBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? getTrueOperands() : getFalseOperands();
}

bool CondBranchOp::canEraseSuccessorOperand() { return true; }

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
      return op.emitOpError("reference to undefined function 'bar'");

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
  return value.isa<BoolAttr>() || value.isa<IntegerAttr>() ||
         value.isa<FloatAttr>() || value.isa<ElementsAttr>() ||
         value.isa<UnitAttr>();
}

void ConstantFloatOp::build(Builder *builder, OperationState &result,
                            const APFloat &value, FloatType type) {
  ConstantOp::build(builder, result, type, builder->getFloatAttr(type, value));
}

bool ConstantFloatOp::classof(Operation *op) {
  return ConstantOp::classof(op) && op->getResult(0).getType().isa<FloatType>();
}

/// ConstantIntOp only matches values whose result type is an IntegerType.
bool ConstantIntOp::classof(Operation *op) {
  return ConstantOp::classof(op) &&
         op->getResult(0).getType().isSignlessInteger();
}

void ConstantIntOp::build(Builder *builder, OperationState &result,
                          int64_t value, unsigned width) {
  Type type = builder->getIntegerType(width);
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

/// Build a constant int op producing an integer with the specified type,
/// which must be an integer type.
void ConstantIntOp::build(Builder *builder, OperationState &result,
                          int64_t value, Type type) {
  assert(type.isSignlessInteger() &&
         "ConstantIntOp can only have signless integer type");
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
}

/// ConstantIndexOp only matches values whose result type is Index.
bool ConstantIndexOp::classof(Operation *op) {
  return ConstantOp::classof(op) && op->getResult(0).getType().isIndex();
}

void ConstantIndexOp::build(Builder *builder, OperationState &result,
                            int64_t value) {
  Type type = builder->getIndexType();
  ConstantOp::build(builder, result, type,
                    builder->getIntegerAttr(type, value));
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

static void print(OpAsmPrinter &p, DimOp op) {
  p << "dim " << op.getOperand() << ", " << op.getIndex();
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"index"});
  p << " : " << op.getOperand().getType();
}

static ParseResult parseDimOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  Type type;
  Type indexType = parser.getBuilder().getIndexType();

  return failure(
      parser.parseOperand(operandInfo) || parser.parseComma() ||
      parser.parseAttribute(indexAttr, indexType, "index", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(operandInfo, type, result.operands) ||
      parser.addTypeToList(indexType, result.types));
}

static LogicalResult verify(DimOp op) {
  // Check that we have an integer index operand.
  auto indexAttr = op.getAttrOfType<IntegerAttr>("index");
  if (!indexAttr)
    return op.emitOpError("requires an integer attribute named 'index'");
  int64_t index = indexAttr.getValue().getSExtValue();

  auto type = op.getOperand().getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    if (index >= tensorType.getRank())
      return op.emitOpError("index is out of range");
  } else if (auto memrefType = type.dyn_cast<MemRefType>()) {
    if (index >= memrefType.getRank())
      return op.emitOpError("index is out of range");

  } else if (type.isa<UnrankedTensorType>()) {
    // ok, assumed to be in-range.
  } else {
    return op.emitOpError("requires an operand with tensor or memref type");
  }

  return success();
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  // Constant fold dim when the size along the index referred to is a constant.
  auto opType = memrefOrTensor().getType();
  int64_t dimSize = ShapedType::kDynamicSize;
  if (auto tensorType = opType.dyn_cast<RankedTensorType>())
    dimSize = tensorType.getShape()[getIndex()];
  else if (auto memrefType = opType.dyn_cast<MemRefType>())
    dimSize = memrefType.getShape()[getIndex()];

  if (!ShapedType::isDynamic(dimSize))
    return IntegerAttr::get(IndexType::get(getContext()), dimSize);

  // Fold dim to the size argument for an AllocOp/ViewOp/SubViewOp.
  auto memrefType = opType.dyn_cast<MemRefType>();
  if (!memrefType)
    return {};

  // The size at getIndex() is now a dynamic size of a memref.
  auto memref = memrefOrTensor().getDefiningOp();
  if (auto alloc = dyn_cast_or_null<AllocOp>(memref))
    return *(alloc.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(getIndex()));

  if (auto view = dyn_cast_or_null<ViewOp>(memref))
    return *(view.getDynamicSizes().begin() +
             memrefType.getDynamicDimIndex(getIndex()));

  // The subview op here is expected to have rank dynamic sizes now.
  if (auto subview = dyn_cast_or_null<SubViewOp>(memref)) {
    auto sizes = subview.sizes();
    if (!sizes.empty())
      return *(sizes.begin() + getIndex());
  }

  /// dim(memrefcast) -> dim
  if (succeeded(foldMemRefCast(*this)))
    return getResult();

  return {};
}

// ---------------------------------------------------------------------------
// DmaStartOp
// ---------------------------------------------------------------------------

void DmaStartOp::build(Builder *builder, OperationState &result,
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

  auto memrefType0 = types[0].dyn_cast<MemRefType>();
  if (!memrefType0)
    return parser.emitError(parser.getNameLoc(),
                            "expected source to be of memref type");

  auto memrefType1 = types[1].dyn_cast<MemRefType>();
  if (!memrefType1)
    return parser.emitError(parser.getNameLoc(),
                            "expected destination to be of memref type");

  auto memrefType2 = types[2].dyn_cast<MemRefType>();
  if (!memrefType2)
    return parser.emitError(parser.getNameLoc(),
                            "expected tag to be of memref type");

  if (isStrided) {
    if (parser.resolveOperands(strideInfo, indexType, result.operands))
      return failure();
  }

  // Check that source/destination index list size matches associated rank.
  if (static_cast<int64_t>(srcIndexInfos.size()) != memrefType0.getRank() ||
      static_cast<int64_t>(dstIndexInfos.size()) != memrefType1.getRank())
    return parser.emitError(parser.getNameLoc(),
                            "memref rank not equal to indices count");
  if (static_cast<int64_t>(tagIndexInfos.size()) != memrefType2.getRank())
    return parser.emitError(parser.getNameLoc(),
                            "tag memref rank not equal to indices count");

  return success();
}

LogicalResult DmaStartOp::verify() {
  // DMAs from different memory spaces supported.
  if (getSrcMemorySpace() == getDstMemorySpace())
    return emitOpError("DMA should be between different memory spaces");

  if (getNumOperands() != getTagMemRefRank() + getSrcMemRefRank() +
                              getDstMemRefRank() + 3 + 1 &&
      getNumOperands() != getTagMemRefRank() + getSrcMemRefRank() +
                              getDstMemRefRank() + 3 + 1 + 2) {
    return emitOpError("incorrect number of operands");
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

void DmaWaitOp::build(Builder *builder, OperationState &result, Value tagMemRef,
                      ValueRange tagIndices, Value numElements) {
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

  auto memrefType = type.dyn_cast<MemRefType>();
  if (!memrefType)
    return parser.emitError(parser.getNameLoc(),
                            "expected tag to be of memref type");

  if (static_cast<int64_t>(tagIndexInfos.size()) != memrefType.getRank())
    return parser.emitError(parser.getNameLoc(),
                            "tag memref rank not equal to indices count");

  return success();
}

LogicalResult DmaWaitOp::fold(ArrayRef<Attribute> cstOperands,
                              SmallVectorImpl<OpFoldResult> &results) {
  /// dma_wait(memrefcast) -> dma_wait
  return foldMemRefCast(*this);
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
// FPExtOp
//===----------------------------------------------------------------------===//

bool FPExtOp::areCastCompatible(Type a, Type b) {
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() < fb.getWidth();
  if (auto va = a.dyn_cast<VectorType>())
    if (auto vb = b.dyn_cast<VectorType>())
      return va.getShape().equals(vb.getShape()) &&
             areCastCompatible(va.getElementType(), vb.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// FPTruncOp
//===----------------------------------------------------------------------===//

bool FPTruncOp::areCastCompatible(Type a, Type b) {
  if (auto fa = a.dyn_cast<FloatType>())
    if (auto fb = b.dyn_cast<FloatType>())
      return fa.getWidth() > fb.getWidth();
  if (auto va = a.dyn_cast<VectorType>())
    if (auto vb = b.dyn_cast<VectorType>())
      return va.getShape().equals(vb.getShape()) &&
             areCastCompatible(va.getElementType(), vb.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast is applicable from index to integer and backwards.
bool IndexCastOp::areCastCompatible(Type a, Type b) {
  return (a.isIndex() && b.isSignlessInteger()) ||
         (a.isSignlessInteger() && b.isIndex());
}

OpFoldResult IndexCastOp::fold(ArrayRef<Attribute> cstOperands) {
  // Fold IndexCast(IndexCast(x)) -> x
  auto cast = dyn_cast_or_null<IndexCastOp>(getOperand().getDefiningOp());
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
  // Constant fold rank when the rank of the tensor is known.
  auto type = getOperand().getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return IntegerAttr::get(IndexType::get(getContext()), tensorType.getRank());
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
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")";

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
  return a.isSignlessInteger() && b.isa<FloatType>();
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
  if (!constOperand ||
      (!constOperand.isa<IntegerAttr>() && !constOperand.isa<FloatAttr>()))
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

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

// Returns a MemRefType with dynamic sizes and offset and the same stride as the
// `memRefType` passed as argument.
// TODO(andydavis,ntv) Evolve to a more powerful inference that can also keep
// sizes and offset static.
static Type inferSubViewResultType(MemRefType memRefType) {
  auto rank = memRefType.getRank();
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto res = getStridesAndOffset(memRefType, strides, offset);
  assert(succeeded(res) && "SubViewOp expected strided memref type");
  (void)res;

  // Assume sizes and offset are fully dynamic for now until canonicalization
  // occurs on the ranges. Typed strides don't change though.
  offset = MemRefType::getDynamicStrideOrOffset();
  // Overwrite strides because verifier will not pass.
  // TODO(b/144419106): don't force degrade the strides to fully dynamic.
  for (auto &stride : strides)
    stride = MemRefType::getDynamicStrideOrOffset();
  auto stridedLayout =
      makeStridedLinearLayoutMap(strides, offset, memRefType.getContext());
  SmallVector<int64_t, 4> sizes(rank, ShapedType::kDynamicSize);
  return MemRefType::Builder(memRefType)
      .setShape(sizes)
      .setAffineMaps(stridedLayout);
}

void mlir::SubViewOp::build(Builder *b, OperationState &result, Value source,
                            ValueRange offsets, ValueRange sizes,
                            ValueRange strides, Type resultType,
                            ArrayRef<NamedAttribute> attrs) {
  if (!resultType)
    resultType = inferSubViewResultType(source.getType().cast<MemRefType>());
  build(b, result, resultType, source, offsets, sizes, strides);
  result.addAttributes(attrs);
}

void mlir::SubViewOp::build(Builder *b, OperationState &result, Type resultType,
                            Value source) {
  build(b, result, source, /*offsets=*/{}, /*sizes=*/{}, /*strides=*/{},
        resultType);
}

static LogicalResult verify(SubViewOp op) {
  auto baseType = op.getBaseMemRefType().cast<MemRefType>();
  auto subViewType = op.getType();

  // The rank of the base and result subview must match.
  if (baseType.getRank() != subViewType.getRank()) {
    return op.emitError(
        "expected rank of result type to match rank of base type ");
  }

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != subViewType.getMemorySpace())
    return op.emitError("different memory spaces specified for base memref "
                        "type ")
           << baseType << " and subview memref type " << subViewType;

  // Verify that the base memref type has a strided layout map.
  int64_t baseOffset;
  SmallVector<int64_t, 4> baseStrides;
  if (failed(getStridesAndOffset(baseType, baseStrides, baseOffset)))
    return op.emitError("base type ") << subViewType << " is not strided";

  // Verify that the result memref type has a strided layout map.
  int64_t subViewOffset;
  SmallVector<int64_t, 4> subViewStrides;
  if (failed(getStridesAndOffset(subViewType, subViewStrides, subViewOffset)))
    return op.emitError("result type ") << subViewType << " is not strided";

  // Num offsets should either be zero or rank of memref.
  if (op.getNumOffsets() != 0 && op.getNumOffsets() != subViewType.getRank()) {
    return op.emitError("expected number of dynamic offsets specified to match "
                        "the rank of the result type ")
           << subViewType;
  }

  // Num sizes should either be zero or rank of memref.
  if (op.getNumSizes() != 0 && op.getNumSizes() != subViewType.getRank()) {
    return op.emitError("expected number of dynamic sizes specified to match "
                        "the rank of the result type ")
           << subViewType;
  }

  // Num strides should either be zero or rank of memref.
  if (op.getNumStrides() != 0 && op.getNumStrides() != subViewType.getRank()) {
    return op.emitError("expected number of dynamic strides specified to match "
                        "the rank of the result type ")
           << subViewType;
  }

  // Verify that if the shape of the subview type is static, then sizes are not
  // dynamic values, and vice versa.
  if ((subViewType.hasStaticShape() && op.getNumSizes() != 0) ||
      (op.getNumSizes() == 0 && !subViewType.hasStaticShape())) {
    return op.emitError("invalid to specify dynamic sizes when subview result "
                        "type is statically shaped and viceversa");
  }

  // Verify that if dynamic sizes are specified, then the result memref type
  // have full dynamic dimensions.
  if (op.getNumSizes() > 0) {
    if (llvm::any_of(subViewType.getShape(), [](int64_t dim) {
          return dim != ShapedType::kDynamicSize;
        })) {
      // TODO: This is based on the assumption that number of size arguments are
      // either 0, or the rank of the result type. It is possible to have more
      // fine-grained verification where only particular dimensions are
      // dynamic. That probably needs further changes to the shape op
      // specification.
      return op.emitError("expected shape of result type to be fully dynamic "
                          "when sizes are specified");
    }
  }

  // Verify that if dynamic offsets are specified or base memref has dynamic
  // offset or base memref has dynamic strides, then the subview offset is
  // dynamic.
  if ((op.getNumOffsets() > 0 ||
       baseOffset == MemRefType::getDynamicStrideOrOffset() ||
       llvm::is_contained(baseStrides,
                          MemRefType::getDynamicStrideOrOffset())) &&
      subViewOffset != MemRefType::getDynamicStrideOrOffset()) {
    return op.emitError(
        "expected result memref layout map to have dynamic offset");
  }

  // For now, verify that if dynamic strides are specified, then all the result
  // memref type have dynamic strides.
  if (op.getNumStrides() > 0) {
    if (llvm::any_of(subViewStrides, [](int64_t stride) {
          return stride != MemRefType::getDynamicStrideOrOffset();
        })) {
      return op.emitError("expected result type to have dynamic strides");
    }
  }

  // If any of the base memref has dynamic stride, then the corresponding
  // stride of the subview must also have dynamic stride.
  assert(baseStrides.size() == subViewStrides.size());
  for (auto stride : enumerate(baseStrides)) {
    if (stride.value() == MemRefType::getDynamicStrideOrOffset() &&
        subViewStrides[stride.index()] !=
            MemRefType::getDynamicStrideOrOffset()) {
      return op.emitError(
          "expected result type to have dynamic stride along a dimension if "
          "the base memref type has dynamic stride along that dimension");
    }
  }
  return success();
}

raw_ostream &mlir::operator<<(raw_ostream &os, SubViewOp::Range &range) {
  return os << "range " << range.offset << ":" << range.size << ":"
            << range.stride;
}

SmallVector<SubViewOp::Range, 8> SubViewOp::getRanges() {
  SmallVector<Range, 8> res;
  unsigned rank = getType().getRank();
  res.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    res.emplace_back(Range{*(offsets().begin() + i), *(sizes().begin() + i),
                           *(strides().begin() + i)});
  return res;
}

LogicalResult
SubViewOp::getStaticStrides(SmallVectorImpl<int64_t> &staticStrides) {
  // If the strides are dynamic return failure.
  if (getNumStrides())
    return failure();

  // When static, the stride operands can be retrieved by taking the strides of
  // the result of the subview op, and dividing the strides of the base memref.
  int64_t resultOffset, baseOffset;
  SmallVector<int64_t, 2> resultStrides, baseStrides;
  if (failed(
          getStridesAndOffset(getBaseMemRefType(), baseStrides, baseOffset)) ||
      llvm::is_contained(baseStrides, MemRefType::getDynamicStrideOrOffset()) ||
      failed(getStridesAndOffset(getType(), resultStrides, resultOffset)))
    return failure();

  assert(static_cast<int64_t>(resultStrides.size()) == getType().getRank() &&
         baseStrides.size() == resultStrides.size() &&
         "base and result memrefs must have the same rank");
  assert(!llvm::is_contained(resultStrides,
                             MemRefType::getDynamicStrideOrOffset()) &&
         "strides of subview op must be static, when there are no dynamic "
         "strides specified");
  staticStrides.resize(getType().getRank());
  for (auto resultStride : enumerate(resultStrides)) {
    auto baseStride = baseStrides[resultStride.index()];
    // The result stride is expected to be a multiple of the base stride. Abort
    // if that is not the case.
    if (resultStride.value() < baseStride ||
        resultStride.value() % baseStride != 0)
      return failure();
    staticStrides[resultStride.index()] = resultStride.value() / baseStride;
  }
  return success();
}

namespace {

/// Pattern to rewrite a subview op with constant size arguments.
class SubViewOpShapeFolder final : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    MemRefType subViewType = subViewOp.getType();
    // Follow all or nothing approach for shapes for now. If all the operands
    // for sizes are constants then fold it into the type of the result memref.
    if (subViewType.hasStaticShape() ||
        llvm::any_of(subViewOp.sizes(), [](Value operand) {
          return !matchPattern(operand, m_ConstantIndex());
        })) {
      return failure();
    }
    SmallVector<int64_t, 4> staticShape(subViewOp.getNumSizes());
    for (auto size : llvm::enumerate(subViewOp.sizes())) {
      auto defOp = size.value().getDefiningOp();
      assert(defOp);
      staticShape[size.index()] = cast<ConstantIndexOp>(defOp).getValue();
    }
    MemRefType newMemRefType =
        MemRefType::Builder(subViewType).setShape(staticShape);
    auto newSubViewOp = rewriter.create<SubViewOp>(
        subViewOp.getLoc(), subViewOp.source(), subViewOp.offsets(),
        ArrayRef<Value>(), subViewOp.strides(), newMemRefType);
    // Insert a memref_cast for compatibility of the uses of the op.
    rewriter.replaceOpWithNewOp<MemRefCastOp>(subViewOp, newSubViewOp,
                                              subViewOp.getType());
    return success();
  }
};

// Pattern to rewrite a subview op with constant stride arguments.
class SubViewOpStrideFolder final : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    if (subViewOp.getNumStrides() == 0) {
      return failure();
    }
    // Follow all or nothing approach for strides for now. If all the operands
    // for strides are constants then fold it into the strides of the result
    // memref.
    int64_t baseOffset, resultOffset;
    SmallVector<int64_t, 4> baseStrides, resultStrides;
    MemRefType subViewType = subViewOp.getType();
    if (failed(getStridesAndOffset(subViewOp.getBaseMemRefType(), baseStrides,
                                   baseOffset)) ||
        failed(getStridesAndOffset(subViewType, resultStrides, resultOffset)) ||
        llvm::is_contained(baseStrides,
                           MemRefType::getDynamicStrideOrOffset()) ||
        llvm::any_of(subViewOp.strides(), [](Value stride) {
          return !matchPattern(stride, m_ConstantIndex());
        })) {
      return failure();
    }

    SmallVector<int64_t, 4> staticStrides(subViewOp.getNumStrides());
    for (auto stride : llvm::enumerate(subViewOp.strides())) {
      auto defOp = stride.value().getDefiningOp();
      assert(defOp);
      assert(baseStrides[stride.index()] > 0);
      staticStrides[stride.index()] =
          cast<ConstantIndexOp>(defOp).getValue() * baseStrides[stride.index()];
    }
    AffineMap layoutMap = makeStridedLinearLayoutMap(
        staticStrides, resultOffset, rewriter.getContext());
    MemRefType newMemRefType =
        MemRefType::Builder(subViewType).setAffineMaps(layoutMap);
    auto newSubViewOp = rewriter.create<SubViewOp>(
        subViewOp.getLoc(), subViewOp.source(), subViewOp.offsets(),
        subViewOp.sizes(), ArrayRef<Value>(), newMemRefType);
    // Insert a memref_cast for compatibility of the uses of the op.
    rewriter.replaceOpWithNewOp<MemRefCastOp>(subViewOp, newSubViewOp,
                                              subViewOp.getType());
    return success();
  }
};

// Pattern to rewrite a subview op with constant offset arguments.
class SubViewOpOffsetFolder final : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    if (subViewOp.getNumOffsets() == 0) {
      return failure();
    }
    // Follow all or nothing approach for offsets for now. If all the operands
    // for offsets are constants then fold it into the offset of the result
    // memref.
    int64_t baseOffset, resultOffset;
    SmallVector<int64_t, 4> baseStrides, resultStrides;
    MemRefType subViewType = subViewOp.getType();
    if (failed(getStridesAndOffset(subViewOp.getBaseMemRefType(), baseStrides,
                                   baseOffset)) ||
        failed(getStridesAndOffset(subViewType, resultStrides, resultOffset)) ||
        llvm::is_contained(baseStrides,
                           MemRefType::getDynamicStrideOrOffset()) ||
        baseOffset == MemRefType::getDynamicStrideOrOffset() ||
        llvm::any_of(subViewOp.offsets(), [](Value stride) {
          return !matchPattern(stride, m_ConstantIndex());
        })) {
      return failure();
    }

    auto staticOffset = baseOffset;
    for (auto offset : llvm::enumerate(subViewOp.offsets())) {
      auto defOp = offset.value().getDefiningOp();
      assert(defOp);
      assert(baseStrides[offset.index()] > 0);
      staticOffset +=
          cast<ConstantIndexOp>(defOp).getValue() * baseStrides[offset.index()];
    }

    AffineMap layoutMap = makeStridedLinearLayoutMap(
        resultStrides, staticOffset, rewriter.getContext());
    MemRefType newMemRefType =
        MemRefType::Builder(subViewType).setAffineMaps(layoutMap);
    auto newSubViewOp = rewriter.create<SubViewOp>(
        subViewOp.getLoc(), subViewOp.source(), ArrayRef<Value>(),
        subViewOp.sizes(), subViewOp.strides(), newMemRefType);
    // Insert a memref_cast for compatibility of the uses of the op.
    rewriter.replaceOpWithNewOp<MemRefCastOp>(subViewOp, newSubViewOp,
                                              subViewOp.getType());
    return success();
  }
};

} // end anonymous namespace

void SubViewOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<SubViewOpShapeFolder, SubViewOpStrideFolder,
                 SubViewOpOffsetFolder>(context);
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

//===----------------------------------------------------------------------===//
// Helpers for Tensor[Load|Store]Op
//===----------------------------------------------------------------------===//

static Type getTensorTypeFromMemRefType(Type type) {
  if (auto memref = type.dyn_cast<MemRefType>())
    return RankedTensorType::get(memref.getShape(), memref.getElementType());
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

  if (offsetInfo.size() > 1)
    return parser.emitError(offsetLoc) << "expects 0 or 1 offset operand";

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
  auto dynamicOffset = op.getDynamicOffset();
  if (dynamicOffset != nullptr)
    p.printOperand(dynamicOffset);
  p << "][" << op.getDynamicSizes() << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperand(0).getType() << " to " << op.getType();
}

Value ViewOp::getDynamicOffset() {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto result =
      succeeded(mlir::getStridesAndOffset(getType(), strides, offset));
  assert(result);
  if (result && offset == MemRefType::getDynamicStrideOrOffset())
    return getOperand(1);
  return nullptr;
}

static LogicalResult verifyDynamicStrides(MemRefType memrefType,
                                          ArrayRef<int64_t> strides) {
  unsigned rank = memrefType.getRank();
  assert(rank == strides.size());
  bool dynamicStrides = false;
  for (int i = rank - 2; i >= 0; --i) {
    // If size at dim 'i + 1' is dynamic, set the 'dynamicStrides' flag.
    if (memrefType.isDynamicDim(i + 1))
      dynamicStrides = true;
    // If stride at dim 'i' is not dynamic, return error.
    if (dynamicStrides && strides[i] != MemRefType::getDynamicStrideOrOffset())
      return failure();
  }
  return success();
}

static LogicalResult verify(ViewOp op) {
  auto baseType = op.getOperand(0).getType().cast<MemRefType>();
  auto viewType = op.getResult().getType().cast<MemRefType>();

  // The base memref should have identity layout map (or none).
  if (baseType.getAffineMaps().size() > 1 ||
      (baseType.getAffineMaps().size() == 1 &&
       !baseType.getAffineMaps()[0].isIdentity()))
    return op.emitError("unsupported map for base memref type ") << baseType;

  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != viewType.getMemorySpace())
    return op.emitError("different memory spaces specified for base memref "
                        "type ")
           << baseType << " and view memref type " << viewType;

  // Verify that the result memref type has a strided layout map.
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(viewType, strides, offset)))
    return op.emitError("result type ") << viewType << " is not strided";

  // Verify that we have the correct number of operands for the result type.
  unsigned memrefOperandCount = 1;
  unsigned numDynamicDims = viewType.getNumDynamicDims();
  unsigned dynamicOffsetCount =
      offset == MemRefType::getDynamicStrideOrOffset() ? 1 : 0;
  if (op.getNumOperands() !=
      memrefOperandCount + numDynamicDims + dynamicOffsetCount)
    return op.emitError("incorrect number of operands for type ") << viewType;

  // Verify dynamic strides symbols were added to correct dimensions based
  // on dynamic sizes.
  if (failed(verifyDynamicStrides(viewType, strides)))
    return op.emitError("incorrect dynamic strides in view memref type ")
           << viewType;
  return success();
}

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
    if (memrefType.getAffineMaps().size() > 1)
      return failure();
    auto map = memrefType.getAffineMaps().empty()
                   ? AffineMap::getMultiDimIdentityMap(memrefType.getRank(),
                                                       rewriter.getContext())
                   : memrefType.getAffineMaps()[0];

    // Get offset from old memref view type 'memRefType'.
    int64_t oldOffset;
    SmallVector<int64_t, 4> oldStrides;
    if (failed(getStridesAndOffset(memrefType, oldStrides, oldOffset)))
      return failure();

    SmallVector<Value, 4> newOperands;

    // Fold dynamic offset operand if it is produced by a constant.
    auto dynamicOffset = viewOp.getDynamicOffset();
    int64_t newOffset = oldOffset;
    unsigned dynamicOffsetOperandCount = 0;
    if (dynamicOffset != nullptr) {
      auto *defOp = dynamicOffset.getDefiningOp();
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic offset will be folded into the map.
        newOffset = constantIndexOp.getValue();
      } else {
        // Unable to fold dynamic offset. Add it to 'newOperands' list.
        newOperands.push_back(dynamicOffset);
        dynamicOffsetOperandCount = 1;
      }
    }

    // Fold any dynamic dim operands which are produced by a constant.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());

    unsigned dynamicDimPos = viewOp.getDynamicSizesOperandStart();
    unsigned rank = memrefType.getRank();
    for (unsigned dim = 0, e = rank; dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (!ShapedType::isDynamic(dimSize)) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto *defOp = viewOp.getOperand(dynamicDimPos).getDefiningOp();
      if (auto constantIndexOp = dyn_cast_or_null<ConstantIndexOp>(defOp)) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constantIndexOp.getValue());
      } else {
        // Dynamic shape dimension not folded; copy operand from old memref.
        newShapeConstants.push_back(dimSize);
        newOperands.push_back(viewOp.getOperand(dynamicDimPos));
      }
      dynamicDimPos++;
    }

    // Compute new strides based on 'newShapeConstants'.
    SmallVector<int64_t, 4> newStrides(rank);
    newStrides[rank - 1] = 1;
    bool dynamicStrides = false;
    for (int i = rank - 2; i >= 0; --i) {
      if (ShapedType::isDynamic(newShapeConstants[i + 1]))
        dynamicStrides = true;
      if (dynamicStrides)
        newStrides[i] = MemRefType::getDynamicStrideOrOffset();
      else
        newStrides[i] = newShapeConstants[i + 1] * newStrides[i + 1];
    }

    // Regenerate strided layout map with 'newStrides' and 'newOffset'.
    map = makeStridedLinearLayoutMap(newStrides, newOffset,
                                     rewriter.getContext());

    // Create new memref type with constant folded dims and/or offset/strides.
    MemRefType newMemRefType = MemRefType::Builder(memrefType)
                                   .setShape(newShapeConstants)
                                   .setAffineMaps({map});
    (void)dynamicOffsetOperandCount; // unused in opt mode
    assert(static_cast<int64_t>(newOperands.size()) ==
           dynamicOffsetOperandCount + newMemRefType.getNumDynamicDims());

    // Create new ViewOp.
    auto newViewOp = rewriter.create<ViewOp>(viewOp.getLoc(), newMemRefType,
                                             viewOp.getOperand(0), newOperands);
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
    MemRefCastOp memrefCastOp =
        dyn_cast_or_null<MemRefCastOp>(memrefOperand.getDefiningOp());
    if (!memrefCastOp)
      return failure();
    Value allocOperand = memrefCastOp.getOperand();
    AllocOp allocOp = dyn_cast_or_null<AllocOp>(allocOperand.getDefiningOp());
    if (!allocOp)
      return failure();
    rewriter.replaceOpWithNewOp<ViewOp>(viewOp, viewOp.getType(), allocOperand,
                                        viewOp.operands());
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
