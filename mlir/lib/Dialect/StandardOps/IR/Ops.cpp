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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.cpp.inc"

using namespace mlir;

/// Helper function to dispatch an OpFoldResult into either the `dynamicVec` if
/// it is a Value or into `staticVec` if it is an IntegerAttr.
/// In the case of a Value, a copy of the `sentinel` value is also pushed to
/// `staticVec`. This is useful to extract mixed static and dynamic entries that
/// come from an AttrSizedOperandSegments trait.
static void dispatchIndexOpFoldResult(OpFoldResult ofr,
                                      SmallVectorImpl<Value> &dynamicVec,
                                      SmallVectorImpl<int64_t> &staticVec,
                                      int64_t sentinel) {
  if (auto v = ofr.dyn_cast<Value>()) {
    dynamicVec.push_back(v);
    staticVec.push_back(sentinel);
    return;
  }
  APInt apInt = ofr.dyn_cast<Attribute>().cast<IntegerAttr>().getValue();
  staticVec.push_back(apInt.getSExtValue());
}

static void dispatchIndexOpFoldResults(ArrayRef<OpFoldResult> ofrs,
                                       SmallVectorImpl<Value> &dynamicVec,
                                       SmallVectorImpl<int64_t> &staticVec,
                                       int64_t sentinel) {
  for (auto ofr : ofrs)
    dispatchIndexOpFoldResult(ofr, dynamicVec, staticVec, sentinel);
}

/// Return true if ofr1 and ofr2 are the same integer constant attribute values
/// or the same SSA value.
/// Ignore integer bitwitdh and type mismatch that come from the fact there is
/// no IndexAttr and that IndexType have no bitwidth.
bool mlir::isEqualConstantIntOrValue(OpFoldResult op1, OpFoldResult op2) {
  auto getConstantIntValue = [](OpFoldResult ofr) -> llvm::Optional<int64_t> {
    Attribute attr = ofr.dyn_cast<Attribute>();
    // Note: isa+cast-like pattern allows writing the condition below as 1 line.
    if (!attr && ofr.get<Value>().getDefiningOp<ConstantOp>())
      attr = ofr.get<Value>().getDefiningOp<ConstantOp>().getValue();
    if (auto intAttr = attr.dyn_cast_or_null<IntegerAttr>())
      return intAttr.getValue().getSExtValue();
    return llvm::None;
  };
  auto cst1 = getConstantIntValue(op1), cst2 = getConstantIntValue(op2);
  if (cst1 && cst2 && *cst1 == *cst2)
    return true;
  auto v1 = op1.dyn_cast<Value>(), v2 = op2.dyn_cast<Value>();
  return v1 && v2 && v1 == v2;
}

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
  getContext()->loadDialect<tensor::TensorDialect>();
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

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
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
} // end anonymous namespace

void CondBranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<SimplifyConstCondBranchPred, SimplifyPassThroughCondBranch,
              SimplifyCondBranchIdenticalSuccessors,
              SimplifyCondBranchFromCondBranchOnSameCondition>(context);
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
  if (value.getType() != type)
    return false;
  // If the type is an integer type, it must be signless.
  if (IntegerType integerTy = type.dyn_cast<IntegerType>())
    if (!integerTy.isSignless())
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
// SubTensorOp
//===----------------------------------------------------------------------===//

/// A subtensor result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubTensorOp::inferResultType(RankedTensorType sourceRankedTensorType,
                                  ArrayRef<int64_t> leadingStaticOffsets,
                                  ArrayRef<int64_t> leadingStaticSizes,
                                  ArrayRef<int64_t> leadingStaticStrides) {
  // A subtensor may specify only a leading subset of offset/sizes/strides in
  // which case we complete with offset=0, sizes from memref type and strides=1.
  unsigned rank = sourceRankedTensorType.getRank();
  assert(leadingStaticSizes.size() <= rank &&
         "unexpected leadingStaticSizes overflow");
  auto staticSizes = llvm::to_vector<4>(leadingStaticSizes);
  unsigned numTrailingSizes = rank - staticSizes.size();
  llvm::append_range(staticSizes, sourceRankedTensorType.getShape().take_back(
                                      numTrailingSizes));
  return RankedTensorType::get(staticSizes,
                               sourceRankedTensorType.getElementType());
}

Type SubTensorOp::inferResultType(RankedTensorType sourceRankedTensorType,
                                  ArrayRef<OpFoldResult> leadingStaticOffsets,
                                  ArrayRef<OpFoldResult> leadingStaticSizes,
                                  ArrayRef<OpFoldResult> leadingStaticStrides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(leadingStaticOffsets, dynamicOffsets,
                             staticOffsets, ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(leadingStaticSizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(leadingStaticStrides, dynamicStrides,
                             staticStrides, ShapedType::kDynamicStrideOrOffset);
  return SubTensorOp::inferResultType(sourceRankedTensorType, staticOffsets,
                                      staticSizes, staticStrides);
}

/// A subtensor result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubTensorOp::inferRankReducedResultType(
    unsigned resultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<int64_t> leadingStaticOffsets,
    ArrayRef<int64_t> leadingStaticSizes,
    ArrayRef<int64_t> leadingStaticStrides) {
  auto inferredType =
      inferResultType(sourceRankedTensorType, leadingStaticOffsets,
                      leadingStaticSizes, leadingStaticStrides)
          .cast<RankedTensorType>();
  int rankDiff = inferredType.getRank() - resultRank;
  if (rankDiff > 0) {
    auto shape = inferredType.getShape();
    llvm::SmallDenseSet<unsigned> dimsToProject;
    mlir::getPositionsOfShapeOne(rankDiff, shape, dimsToProject);
    SmallVector<int64_t> projectedShape;
    for (unsigned pos = 0, e = shape.size(); pos < e; ++pos)
      if (!dimsToProject.contains(pos))
        projectedShape.push_back(shape[pos]);
    inferredType =
        RankedTensorType::get(projectedShape, inferredType.getElementType());
  }
  return inferredType;
}

Type SubTensorOp::inferRankReducedResultType(
    unsigned resultRank, RankedTensorType sourceRankedTensorType,
    ArrayRef<OpFoldResult> leadingStaticOffsets,
    ArrayRef<OpFoldResult> leadingStaticSizes,
    ArrayRef<OpFoldResult> leadingStaticStrides) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(leadingStaticOffsets, dynamicOffsets,
                             staticOffsets, ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(leadingStaticSizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(leadingStaticStrides, dynamicStrides,
                             staticStrides, ShapedType::kDynamicStrideOrOffset);
  return SubTensorOp::inferRankReducedResultType(
      resultRank, sourceRankedTensorType, staticOffsets, staticSizes,
      staticStrides);
}

// Build a SubTensorOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void mlir::SubTensorOp::build(OpBuilder &b, OperationState &result,
                              RankedTensorType resultType, Value source,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides,
                              ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  auto sourceRankedTensorType = source.getType().cast<RankedTensorType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType =
        SubTensorOp::inferResultType(sourceRankedTensorType, staticOffsets,
                                     staticSizes, staticStrides)
            .cast<RankedTensorType>();
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubTensorOp with mixed static and dynamic entries and inferred result
// type.
void mlir::SubTensorOp::build(OpBuilder &b, OperationState &result,
                              Value source, ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides,
                              ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), source, offsets, sizes, strides, attrs);
}

// Build a SubTensorOp with dynamic entries and custom result type. If the type
// passed is nullptr, it is inferred.
void mlir::SubTensorOp::build(OpBuilder &b, OperationState &result,
                              RankedTensorType resultType, Value source,
                              ValueRange offsets, ValueRange sizes,
                              ValueRange strides,
                              ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

// Build a SubTensorOp with dynamic entries and inferred result type.
void mlir::SubTensorOp::build(OpBuilder &b, OperationState &result,
                              Value source, ValueRange offsets,
                              ValueRange sizes, ValueRange strides,
                              ArrayRef<NamedAttribute> attrs) {
  build(b, result, RankedTensorType(), source, offsets, sizes, strides, attrs);
}

enum SubTensorVerificationResult {
  Success,
  RankTooLarge,
  SizeMismatch,
  ElemTypeMismatch,
};

/// Checks if `original` Type type can be rank reduced to `reduced` type.
/// This function is slight variant of `is subsequence` algorithm where
/// not matching dimension must be 1.
static SubTensorVerificationResult
isRankReducedType(Type originalType, Type candidateReducedType,
                  std::string *errMsg = nullptr) {
  if (originalType == candidateReducedType)
    return SubTensorVerificationResult::Success;
  if (!originalType.isa<RankedTensorType>())
    return SubTensorVerificationResult::Success;
  if (originalType.isa<RankedTensorType>() &&
      !candidateReducedType.isa<RankedTensorType>())
    return SubTensorVerificationResult::Success;

  ShapedType originalShapedType = originalType.cast<ShapedType>();
  ShapedType candidateReducedShapedType =
      candidateReducedType.cast<ShapedType>();

  // Rank and size logic is valid for all ShapedTypes.
  ArrayRef<int64_t> originalShape = originalShapedType.getShape();
  ArrayRef<int64_t> candidateReducedShape =
      candidateReducedShapedType.getShape();
  unsigned originalRank = originalShape.size(),
           candidateReducedRank = candidateReducedShape.size();
  if (candidateReducedRank > originalRank)
    return SubTensorVerificationResult::RankTooLarge;

  auto optionalUnusedDimsMask =
      computeRankReductionMask(originalShape, candidateReducedShape);

  // Sizes cannot be matched in case empty vector is returned.
  if (!optionalUnusedDimsMask.hasValue())
    return SubTensorVerificationResult::SizeMismatch;

  if (originalShapedType.getElementType() !=
      candidateReducedShapedType.getElementType())
    return SubTensorVerificationResult::ElemTypeMismatch;

  // We are done for the tensor case.
  if (originalType.isa<RankedTensorType>())
    return SubTensorVerificationResult::Success;

  return SubTensorVerificationResult::Success;
}

template <typename OpTy>
static LogicalResult
produceSubTensorErrorMsg(SubTensorVerificationResult result, OpTy op,
                         Type expectedType, StringRef errMsg = "") {
  auto memrefType = expectedType.cast<ShapedType>();
  switch (result) {
  case SubTensorVerificationResult::Success:
    return success();
  case SubTensorVerificationResult::RankTooLarge:
    return op.emitError("expected result rank to be smaller or equal to ")
           << "the source rank. " << errMsg;
  case SubTensorVerificationResult::SizeMismatch:
    return op.emitError("expected result type to be ")
           << expectedType
           << " or a rank-reduced version. (mismatch of result sizes) "
           << errMsg;
  case SubTensorVerificationResult::ElemTypeMismatch:
    return op.emitError("expected result element type to be ")
           << memrefType.getElementType() << errMsg;
  }
  llvm_unreachable("unexpected subtensor verification result");
}
/// Verifier for SubTensorOp.
static LogicalResult verify(SubTensorOp op) {
  // Verify result type against inferred type.
  auto expectedType = SubTensorOp::inferResultType(
      op.getSourceType(), extractFromI64ArrayAttr(op.static_offsets()),
      extractFromI64ArrayAttr(op.static_sizes()),
      extractFromI64ArrayAttr(op.static_strides()));
  auto result = isRankReducedType(expectedType, op.getType());
  return produceSubTensorErrorMsg(result, op, expectedType);
}

namespace {
/// Pattern to rewrite a subtensor op with tensor::Cast arguments.
/// This essentially pushes memref_cast past its consuming subtensor when
/// `canFoldIntoConsumerOp` is true.
///
/// Example:
/// ```
///   %0 = tensorcast %V : tensor<16x16xf32> to tensor<?x?xf32>
///   %1 = subtensor %0[0, 0][3, 4][1, 1] : tensor<?x?xf32> to tensor<3x4xf32>
/// ```
/// is rewritten into:
/// ```
///   %0 = subtensor %V[0, 0][3, 4][1, 1] : tensor<16x16xf32> to tensor<3x4xf32>
///   %1 = tensor.cast %0: tensor<3x4xf32> to tensor<3x4xf32>
/// ```
class SubTensorOpCastFolder final : public OpRewritePattern<SubTensorOp> {
public:
  using OpRewritePattern<SubTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorOp subTensorOp,
                                PatternRewriter &rewriter) const override {
    // Any constant operand, just return to let SubViewOpConstantFolder kick in.
    if (llvm::any_of(subTensorOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    auto castOp = subTensorOp.source().getDefiningOp<tensor::CastOp>();
    if (!castOp)
      return failure();

    if (!canFoldIntoConsumerOp(castOp))
      return failure();

    /// Deduce the resultType of SubTensorOp with `inferRankReducedResultType`
    /// on the cast source operand type and the SubTensorOp static information.
    /// This is the resulting type if the tensor::CastOp were folded and
    /// rank-reduced to the desired result rank.
    auto resultType = SubTensorOp::inferRankReducedResultType(
        subTensorOp.getType().getRank(),
        castOp.source().getType().cast<RankedTensorType>(),
        subTensorOp.getMixedOffsets(), subTensorOp.getMixedSizes(),
        subTensorOp.getMixedStrides());
    Value newSubTensor = rewriter.create<SubTensorOp>(
        subTensorOp.getLoc(), resultType, castOp.source(),
        subTensorOp.offsets(), subTensorOp.sizes(), subTensorOp.strides(),
        subTensorOp.static_offsets(), subTensorOp.static_sizes(),
        subTensorOp.static_strides());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        subTensorOp, subTensorOp.getType(), newSubTensor);
    return success();
  }
};
} // namespace

/// A canonicalizer wrapper to replace SubTensorOps.
struct SubTensorCanonicalizer {
  void operator()(PatternRewriter &rewriter, SubTensorOp op,
                  SubTensorOp newOp) {
    Value replacement = newOp.getResult();
    if (replacement.getType() != op.getType())
      replacement = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                    replacement);
    rewriter.replaceOp(op, replacement);
  }
};

void SubTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<OpWithOffsetSizesAndStridesConstantArgumentFolder<
                  SubTensorOp, SubTensorCanonicalizer>,
              SubTensorOpCastFolder>(context);
}

//
static LogicalResult
foldIdentityOffsetSizeAndStrideOpInterface(OffsetSizeAndStrideOpInterface op,
                                           ShapedType shapedType) {
  OpBuilder b(op.getContext());
  for (OpFoldResult ofr : op.getMixedOffsets())
    if (!isEqualConstantIntOrValue(ofr, b.getIndexAttr(0)))
      return failure();
  // Rank-reducing noops only need to inspect the leading dimensions: llvm::zip
  // is appropriate.
  auto shape = shapedType.getShape();
  for (auto it : llvm::zip(op.getMixedSizes(), shape))
    if (!isEqualConstantIntOrValue(std::get<0>(it),
                                   b.getIndexAttr(std::get<1>(it))))
      return failure();
  for (OpFoldResult ofr : op.getMixedStrides())
    if (!isEqualConstantIntOrValue(ofr, b.getIndexAttr(1)))
      return failure();
  return success();
}

OpFoldResult SubTensorOp::fold(ArrayRef<Attribute>) {
  if (getSourceType() == getType() &&
      succeeded(foldIdentityOffsetSizeAndStrideOpInterface(*this, getType())))
    return this->source();
  return OpFoldResult();
}

//===----------------------------------------------------------------------===//
// SubTensorInsertOp
//===----------------------------------------------------------------------===//

// Build a SubTensorInsertOp with mixed static and dynamic entries.
void mlir::SubTensorInsertOp::build(OpBuilder &b, OperationState &result,
                                    Value source, Value dest,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    ArrayRef<OpFoldResult> strides,
                                    ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  build(b, result, dest.getType(), source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubTensorInsertOp with dynamic entries.
void mlir::SubTensorInsertOp::build(OpBuilder &b, OperationState &result,
                                    Value source, Value dest,
                                    ValueRange offsets, ValueRange sizes,
                                    ValueRange strides,
                                    ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

OpFoldResult SubTensorInsertOp::fold(ArrayRef<Attribute>) {
  if (getSourceType().hasStaticShape() && getType().hasStaticShape() &&
      getSourceType() == getType() &&
      succeeded(foldIdentityOffsetSizeAndStrideOpInterface(*this, getType())))
    return this->source();
  return OpFoldResult();
}

namespace {
/// Pattern to rewrite a subtensor_insert op with constant arguments.
class SubTensorInsertOpConstantArgumentFolder final
    : public OpRewritePattern<SubTensorInsertOp> {
public:
  using OpRewritePattern<SubTensorInsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorInsertOp subTensorInsertOp,
                                PatternRewriter &rewriter) const override {
    // No constant operand, just return.
    if (llvm::none_of(subTensorInsertOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    // At least one of offsets/sizes/strides is a new constant.
    // Form the new list of operands and constant attributes from the
    // existing.
    SmallVector<OpFoldResult> mixedOffsets(subTensorInsertOp.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(subTensorInsertOp.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(subTensorInsertOp.getMixedStrides());
    canonicalizeSubViewPart(mixedOffsets, ShapedType::isDynamicStrideOrOffset);
    canonicalizeSubViewPart(mixedSizes, ShapedType::isDynamic);
    canonicalizeSubViewPart(mixedStrides, ShapedType::isDynamicStrideOrOffset);

    // Create the new op in canonical form.
    Value source = subTensorInsertOp.source();
    RankedTensorType sourceType = source.getType().cast<RankedTensorType>();
    SmallVector<int64_t, 4> shape = llvm::to_vector<4>(
        llvm::map_range(mixedSizes, [](OpFoldResult valueOrAttr) -> int64_t {
          if (auto attr = valueOrAttr.dyn_cast<Attribute>())
            return attr.cast<IntegerAttr>().getInt();
          return ShapedType::kDynamicSize;
        }));
    RankedTensorType newSourceType =
        RankedTensorType::get(shape, sourceType.getElementType());
    Location loc = subTensorInsertOp.getLoc();
    if (sourceType != newSourceType)
      source = rewriter.create<tensor::CastOp>(loc, newSourceType, source);
    rewriter.replaceOpWithNewOp<SubTensorInsertOp>(
        subTensorInsertOp, source, subTensorInsertOp.dest(), mixedOffsets,
        mixedSizes, mixedStrides);
    return success();
  }
};

/// Fold tensor_casts with subtensor_insert operations.
struct SubTensorInsertOpCastFolder final
    : public OpRewritePattern<SubTensorInsertOp> {
  using OpRewritePattern<SubTensorInsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubTensorInsertOp subTensorInsertOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(subTensorInsertOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    auto getSourceOfCastOp = [](Value v) -> Optional<Value> {
      auto castOp = v.getDefiningOp<tensor::CastOp>();
      if (!castOp || !canFoldIntoConsumerOp(castOp))
        return llvm::None;
      return castOp.source();
    };
    Optional<Value> sourceCastSource =
        getSourceOfCastOp(subTensorInsertOp.source());
    Optional<Value> destCastSource =
        getSourceOfCastOp(subTensorInsertOp.dest());
    if (!sourceCastSource && !destCastSource)
      return failure();

    Value replacement = rewriter.create<SubTensorInsertOp>(
        subTensorInsertOp.getLoc(),
        (sourceCastSource ? *sourceCastSource : subTensorInsertOp.source()),
        (destCastSource ? *destCastSource : subTensorInsertOp.dest()),
        subTensorInsertOp.getMixedOffsets(), subTensorInsertOp.getMixedSizes(),
        subTensorInsertOp.getMixedStrides());

    if (replacement.getType() != subTensorInsertOp.getType()) {
      replacement = rewriter.create<tensor::CastOp>(
          subTensorInsertOp.getLoc(), subTensorInsertOp.getType(), replacement);
    }
    rewriter.replaceOp(subTensorInsertOp, replacement);
    return success();
  }
};
} // namespace

void SubTensorInsertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<SubTensorInsertOpConstantArgumentFolder,
              SubTensorInsertOpCastFolder>(context);
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
  if (matchPattern(getOperand(), m_Op<ZeroExtendIOp>()))
    return getOperand().getDefiningOp()->getOperand(0);

  return nullptr;
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
