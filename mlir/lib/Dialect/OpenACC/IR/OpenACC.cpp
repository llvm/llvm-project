//===- OpenACC.cpp - OpenACC MLIR Operations ------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.cpp.inc"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace acc;

#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// OpenACC operations
//===----------------------------------------------------------------------===//

void OpenACCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.cpp.inc"
      >();
}

template <typename StructureOp>
static ParseResult parseRegions(OpAsmParser &parser, OperationState &state,
                                unsigned nRegions = 1) {

  SmallVector<Region *, 2> regions;
  for (unsigned i = 0; i < nRegions; ++i)
    regions.push_back(state.addRegion());

  for (Region *region : regions) {
    if (parser.parseRegion(*region, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
  }

  return success();
}

static bool isComputeOperation(Operation *op) {
  return isa<acc::ParallelOp>(op) || isa<acc::LoopOp>(op);
}

namespace {
/// Pattern to remove operation without region that have constant false `ifCond`
/// and remove the condition from the operation if the `ifCond` is a true
/// constant.
template <typename OpTy>
struct RemoveConstantIfCondition : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early return if there is no condition.
    Value ifCond = op.getIfCond();
    if (!ifCond)
      return failure();

    IntegerAttr constAttr;
    if (!matchPattern(ifCond, m_Constant(&constAttr)))
      return failure();
    if (constAttr.getInt())
      rewriter.updateRootInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
    else
      rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

unsigned ParallelOp::getNumDataOperands() {
  return getReductionOperands().size() + getCopyOperands().size() +
         getCopyinOperands().size() + getCopyinReadonlyOperands().size() +
         getCopyoutOperands().size() + getCopyoutZeroOperands().size() +
         getCreateOperands().size() + getCreateZeroOperands().size() +
         getNoCreateOperands().size() + getPresentOperands().size() +
         getDevicePtrOperands().size() + getAttachOperands().size() +
         getGangPrivateOperands().size() + getGangFirstPrivateOperands().size();
}

Value ParallelOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync() ? 1 : 0;
  numOptional += getNumGangs() ? 1 : 0;
  numOptional += getNumWorkers() ? 1 : 0;
  numOptional += getVectorLength() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

//===----------------------------------------------------------------------===//
// SerialOp
//===----------------------------------------------------------------------===//

unsigned SerialOp::getNumDataOperands() {
  return getReductionOperands().size() + getCopyOperands().size() +
         getCopyinOperands().size() + getCopyinReadonlyOperands().size() +
         getCopyoutOperands().size() + getCopyoutZeroOperands().size() +
         getCreateOperands().size() + getCreateZeroOperands().size() +
         getNoCreateOperands().size() + getPresentOperands().size() +
         getDevicePtrOperands().size() + getAttachOperands().size() +
         getGangPrivateOperands().size() + getGangFirstPrivateOperands().size();
}

Value SerialOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

//===----------------------------------------------------------------------===//
// KernelsOp
//===----------------------------------------------------------------------===//

unsigned KernelsOp::getNumDataOperands() {
  return getCopyOperands().size() + getCopyinOperands().size() +
         getCopyinReadonlyOperands().size() + getCopyoutOperands().size() +
         getCopyoutZeroOperands().size() + getCreateOperands().size() +
         getCreateZeroOperands().size() + getNoCreateOperands().size() +
         getPresentOperands().size() + getDevicePtrOperands().size() +
         getAttachOperands().size();
}

Value KernelsOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

static ParseResult
parseGangClause(OpAsmParser &parser,
                std::optional<OpAsmParser::UnresolvedOperand> &gangNum,
                Type &gangNumType,
                std::optional<OpAsmParser::UnresolvedOperand> &gangStatic,
                Type &gangStaticType, UnitAttr &hasGang) {
  hasGang = UnitAttr::get(parser.getBuilder().getContext());
  // optional gang operands
  if (succeeded(parser.parseOptionalLParen())) {
    if (succeeded(parser.parseOptionalKeyword(LoopOp::getGangNumKeyword()))) {
      if (parser.parseEqual())
        return failure();
      gangNum = OpAsmParser::UnresolvedOperand{};
      if (parser.parseOperand(*gangNum) || parser.parseColonType(gangNumType))
        return failure();
    } else {
      gangNum = std::nullopt;
    }
    // FIXME: Comma should require subsequent operands.
    (void)parser.parseOptionalComma();
    if (succeeded(
            parser.parseOptionalKeyword(LoopOp::getGangStaticKeyword()))) {
      gangStatic = OpAsmParser::UnresolvedOperand{};
      if (parser.parseEqual())
        return failure();
      gangStatic = OpAsmParser::UnresolvedOperand{};
      if (parser.parseOperand(*gangStatic) ||
          parser.parseColonType(gangStaticType))
        return failure();
    }
    // FIXME: Why allow optional last commas?
    (void)parser.parseOptionalComma();
    if (failed(parser.parseRParen()))
      return failure();
  }
  return success();
}

void printGangClause(OpAsmPrinter &p, Operation *op, Value gangNum,
                     Type gangNumType, Value gangStatic, Type gangStaticType,
                     UnitAttr hasGang) {
  if (gangNum || gangStatic) {
    p << "(";
    if (gangNum) {
      p << LoopOp::getGangNumKeyword() << "=" << gangNum << " : "
        << gangNumType;
      if (gangStatic)
        p << ", ";
    }
    if (gangStatic)
      p << LoopOp::getGangStaticKeyword() << "=" << gangStatic << " : "
        << gangStaticType;
    p << ")";
  }
}

static ParseResult
parseWorkerClause(OpAsmParser &parser,
                  std::optional<OpAsmParser::UnresolvedOperand> &workerNum,
                  Type &workerNumType, UnitAttr &hasWorker) {
  hasWorker = UnitAttr::get(parser.getBuilder().getContext());
  if (succeeded(parser.parseOptionalLParen())) {
    workerNum = OpAsmParser::UnresolvedOperand{};
    if (parser.parseOperand(*workerNum) ||
        parser.parseColonType(workerNumType) || parser.parseRParen())
      return failure();
  }
  return success();
}

void printWorkerClause(OpAsmPrinter &p, Operation *op, Value workerNum,
                       Type workerNumType, UnitAttr hasWorker) {
  if (workerNum)
    p << "(" << workerNum << " : " << workerNumType << ")";
}

static ParseResult
parseVectorClause(OpAsmParser &parser,
                  std::optional<OpAsmParser::UnresolvedOperand> &vectorLength,
                  Type &vectorLengthType, UnitAttr &hasVector) {
  hasVector = UnitAttr::get(parser.getBuilder().getContext());
  if (succeeded(parser.parseOptionalLParen())) {
    vectorLength = OpAsmParser::UnresolvedOperand{};
    if (parser.parseOperand(*vectorLength) ||
        parser.parseColonType(vectorLengthType) || parser.parseRParen())
      return failure();
  }
  return success();
}

void printVectorClause(OpAsmPrinter &p, Operation *op, Value vectorLength,
                       Type vectorLengthType, UnitAttr hasVector) {
  if (vectorLength)
    p << "(" << vectorLength << " : " << vectorLengthType << ")";
}

LogicalResult acc::LoopOp::verify() {
  // auto, independent and seq attribute are mutually exclusive.
  if ((getAuto_() && (getIndependent() || getSeq())) ||
      (getIndependent() && getSeq())) {
    return emitError() << "only one of \"" << acc::LoopOp::getAutoAttrStrName()
                       << "\", " << getIndependentAttrName() << ", "
                       << getSeqAttrName()
                       << " can be present at the same time";
  }

  // Gang, worker and vector are incompatible with seq.
  if (getSeq() && (getHasGang() || getHasWorker() || getHasVector()))
    return emitError("gang, worker or vector cannot appear with the seq attr");

  // Check non-empty body().
  if (getRegion().empty())
    return emitError("expected non-empty body.");

  return success();
}

//===----------------------------------------------------------------------===//
// DataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DataOp::verify() {
  // 2.6.5. Data Construct restriction
  // At least one copy, copyin, copyout, create, no_create, present, deviceptr,
  // attach, or default clause must appear on a data construct.
  if (getOperands().empty() && !getDefaultAttr())
    return emitError("at least one operand or the default attribute "
                     "must appear on the data operation");
  return success();
}

unsigned DataOp::getNumDataOperands() {
  return getCopyOperands().size() + getCopyinOperands().size() +
         getCopyinReadonlyOperands().size() + getCopyoutOperands().size() +
         getCopyoutZeroOperands().size() + getCreateOperands().size() +
         getCreateZeroOperands().size() + getNoCreateOperands().size() +
         getPresentOperands().size() + getDeviceptrOperands().size() +
         getAttachOperands().size();
}

Value DataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  return getOperand(numOptional + i);
}

//===----------------------------------------------------------------------===//
// ExitDataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::ExitDataOp::verify() {
  // 2.6.6. Data Exit Directive restriction
  // At least one copyout, delete, or detach clause must appear on an exit data
  // directive.
  if (getCopyoutOperands().empty() && getDeleteOperands().empty() &&
      getDetachOperands().empty())
    return emitError(
        "at least one operand in copyout, delete or detach must appear on the "
        "exit data operation");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!getWaitOperands().empty() && getWait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned ExitDataOp::getNumDataOperands() {
  return getCopyoutOperands().size() + getDeleteOperands().size() +
         getDetachOperands().size();
}

Value ExitDataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  numOptional += getAsyncOperand() ? 1 : 0;
  numOptional += getWaitDevnum() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

void ExitDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<RemoveConstantIfCondition<ExitDataOp>>(context);
}

//===----------------------------------------------------------------------===//
// EnterDataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::EnterDataOp::verify() {
  // 2.6.6. Data Enter Directive restriction
  // At least one copyin, create, or attach clause must appear on an enter data
  // directive.
  if (getCopyinOperands().empty() && getCreateOperands().empty() &&
      getCreateZeroOperands().empty() && getAttachOperands().empty())
    return emitError(
        "at least one operand in copyin, create, "
        "create_zero or attach must appear on the enter data operation");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!getWaitOperands().empty() && getWait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned EnterDataOp::getNumDataOperands() {
  return getCopyinOperands().size() + getCreateOperands().size() +
         getCreateZeroOperands().size() + getAttachOperands().size();
}

Value EnterDataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  numOptional += getAsyncOperand() ? 1 : 0;
  numOptional += getWaitDevnum() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

void EnterDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<RemoveConstantIfCondition<EnterDataOp>>(context);
}

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

LogicalResult acc::InitOp::verify() {
  Operation *currOp = *this;
  while ((currOp = currOp->getParentOp()))
    if (isComputeOperation(currOp))
      return emitOpError("cannot be nested in a compute operation");
  return success();
}

//===----------------------------------------------------------------------===//
// ShutdownOp
//===----------------------------------------------------------------------===//

LogicalResult acc::ShutdownOp::verify() {
  Operation *currOp = *this;
  while ((currOp = currOp->getParentOp()))
    if (isComputeOperation(currOp))
      return emitOpError("cannot be nested in a compute operation");
  return success();
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

LogicalResult acc::UpdateOp::verify() {
  // At least one of host or device should have a value.
  if (getHostOperands().empty() && getDeviceOperands().empty())
    return emitError(
        "at least one value must be present in hostOperands or deviceOperands");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!getWaitOperands().empty() && getWait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned UpdateOp::getNumDataOperands() {
  return getHostOperands().size() + getDeviceOperands().size();
}

Value UpdateOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsyncOperand() ? 1 : 0;
  numOptional += getWaitDevnum() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + getDeviceTypeOperands().size() +
                    numOptional + i);
}

void UpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<RemoveConstantIfCondition<UpdateOp>>(context);
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

LogicalResult acc::WaitOp::verify() {
  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.cpp.inc"
