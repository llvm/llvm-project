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

static ParseResult
parseOperandList(OpAsmParser &parser, StringRef keyword,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &args,
                 SmallVectorImpl<Type> &argTypes, OperationState &result) {
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  if (failed(parser.parseLParen()))
    return failure();

  // Exit early if the list is empty.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  if (failed(parser.parseCommaSeparatedList([&]() {
        OpAsmParser::UnresolvedOperand arg;
        Type type;

        if (parser.parseOperand(arg, /*allowResultNumber=*/false) ||
            parser.parseColonType(type))
          return failure();

        args.push_back(arg);
        argTypes.push_back(type);
        return success();
      })) ||
      failed(parser.parseRParen()))
    return failure();

  return parser.resolveOperands(args, argTypes, parser.getCurrentLocation(),
                                result.operands);
}

static void printOperandList(Operation::operand_range operands,
                             StringRef listName, OpAsmPrinter &printer) {

  if (!operands.empty()) {
    printer << " " << listName << "(";
    llvm::interleaveComma(operands, printer, [&](Value op) {
      printer << op << ": " << op.getType();
    });
    printer << ")";
  }
}

static ParseResult parseOptionalOperand(OpAsmParser &parser, StringRef keyword,
                                        OpAsmParser::UnresolvedOperand &operand,
                                        Type type, bool &hasOptional,
                                        OperationState &result) {
  hasOptional = false;
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    hasOptional = true;
    if (parser.parseLParen() || parser.parseOperand(operand) ||
        parser.resolveOperand(operand, type, result.operands) ||
        parser.parseRParen())
      return failure();
  }
  return success();
}

static ParseResult parseOperandAndType(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  Type type;
  if (parser.parseOperand(operand) || parser.parseColonType(type) ||
      parser.resolveOperand(operand, type, result.operands))
    return failure();
  return success();
}

/// Parse optional operand and its type wrapped in parenthesis prefixed with
/// a keyword.
/// Example:
///   keyword `(` %vectorLength: i64 `)`
static OptionalParseResult parseOptionalOperandAndType(OpAsmParser &parser,
                                                       StringRef keyword,
                                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    return failure(parser.parseLParen() ||
                   parseOperandAndType(parser, result) || parser.parseRParen());
  }
  return std::nullopt;
}

/// Parse optional operand and its type wrapped in parenthesis.
/// Example:
///   `(` %vectorLength: i64 `)`
static OptionalParseResult parseOptionalOperandAndType(OpAsmParser &parser,
                                                       OperationState &result) {
  if (succeeded(parser.parseOptionalLParen())) {
    return failure(parseOperandAndType(parser, result) || parser.parseRParen());
  }
  return std::nullopt;
}

/// Parse optional operand with its type prefixed with prefixKeyword `=`.
/// Example:
///   num=%gangNum: i32
static OptionalParseResult parserOptionalOperandAndTypeWithPrefix(
    OpAsmParser &parser, OperationState &result, StringRef prefixKeyword) {
  if (succeeded(parser.parseOptionalKeyword(prefixKeyword))) {
    if (parser.parseEqual() || parseOperandAndType(parser, result))
      return failure();
    return success();
  }
  return std::nullopt;
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
      return success();

    IntegerAttr constAttr;
    if (matchPattern(ifCond, m_Constant(&constAttr))) {
      if (constAttr.getInt())
        rewriter.updateRootInPlace(op,
                                   [&]() { op.getIfCondMutable().erase(0); });
      else
        rewriter.eraseOp(op);
    }

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

/// Parse acc.parallel operation
/// operation := `acc.parallel` `async` `(` index `)`?
///                             `wait` `(` index-list `)`?
///                             `num_gangs` `(` value `)`?
///                             `num_workers` `(` value `)`?
///                             `vector_length` `(` value `)`?
///                             `if` `(` value `)`?
///                             `self` `(` value `)`?
///                             `reduction` `(` value-list `)`?
///                             `copy` `(` value-list `)`?
///                             `copyin` `(` value-list `)`?
///                             `copyin_readonly` `(` value-list `)`?
///                             `copyout` `(` value-list `)`?
///                             `copyout_zero` `(` value-list `)`?
///                             `create` `(` value-list `)`?
///                             `create_zero` `(` value-list `)`?
///                             `no_create` `(` value-list `)`?
///                             `present` `(` value-list `)`?
///                             `deviceptr` `(` value-list `)`?
///                             `attach` `(` value-list `)`?
///                             `private` `(` value-list `)`?
///                             `firstprivate` `(` value-list `)`?
///                             region attr-dict?
ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  SmallVector<OpAsmParser::UnresolvedOperand, 8> privateOperands,
      firstprivateOperands, copyOperands, copyinOperands,
      copyinReadonlyOperands, copyoutOperands, copyoutZeroOperands,
      createOperands, createZeroOperands, noCreateOperands, presentOperands,
      devicePtrOperands, attachOperands, waitOperands, reductionOperands;
  SmallVector<Type, 8> waitOperandTypes, reductionOperandTypes,
      copyOperandTypes, copyinOperandTypes, copyinReadonlyOperandTypes,
      copyoutOperandTypes, copyoutZeroOperandTypes, createOperandTypes,
      createZeroOperandTypes, noCreateOperandTypes, presentOperandTypes,
      deviceptrOperandTypes, attachOperandTypes, privateOperandTypes,
      firstprivateOperandTypes;

  SmallVector<Type, 8> operandTypes;
  OpAsmParser::UnresolvedOperand ifCond, selfCond;
  bool hasIfCond = false, hasSelfCond = false;
  OptionalParseResult async, numGangs, numWorkers, vectorLength;
  Type i1Type = builder.getI1Type();

  // getAsync()?
  async = parseOptionalOperandAndType(parser, ParallelOp::getAsyncKeyword(),
                                      result);
  if (async.has_value() && failed(*async))
    return failure();

  // getWait()?
  if (failed(parseOperandList(parser, ParallelOp::getWaitKeyword(),
                              waitOperands, waitOperandTypes, result)))
    return failure();

  // num_gangs(value)?
  numGangs = parseOptionalOperandAndType(
      parser, ParallelOp::getNumGangsKeyword(), result);
  if (numGangs.has_value() && failed(*numGangs))
    return failure();

  // num_workers(value)?
  numWorkers = parseOptionalOperandAndType(
      parser, ParallelOp::getNumWorkersKeyword(), result);
  if (numWorkers.has_value() && failed(*numWorkers))
    return failure();

  // vector_length(value)?
  vectorLength = parseOptionalOperandAndType(
      parser, ParallelOp::getVectorLengthKeyword(), result);
  if (vectorLength.has_value() && failed(*vectorLength))
    return failure();

  // if()?
  if (failed(parseOptionalOperand(parser, ParallelOp::getIfKeyword(), ifCond,
                                  i1Type, hasIfCond, result)))
    return failure();

  // self()?
  if (failed(parseOptionalOperand(parser, ParallelOp::getSelfKeyword(),
                                  selfCond, i1Type, hasSelfCond, result)))
    return failure();

  // reduction()?
  if (failed(parseOperandList(parser, ParallelOp::getReductionKeyword(),
                              reductionOperands, reductionOperandTypes,
                              result)))
    return failure();

  // copy()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyKeyword(),
                              copyOperands, copyOperandTypes, result)))
    return failure();

  // copyin()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyinKeyword(),
                              copyinOperands, copyinOperandTypes, result)))
    return failure();

  // copyin_readonly()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyinReadonlyKeyword(),
                              copyinReadonlyOperands,
                              copyinReadonlyOperandTypes, result)))
    return failure();

  // copyout()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyoutKeyword(),
                              copyoutOperands, copyoutOperandTypes, result)))
    return failure();

  // copyout_zero()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyoutZeroKeyword(),
                              copyoutZeroOperands, copyoutZeroOperandTypes,
                              result)))
    return failure();

  // create()?
  if (failed(parseOperandList(parser, ParallelOp::getCreateKeyword(),
                              createOperands, createOperandTypes, result)))
    return failure();

  // create_zero()?
  if (failed(parseOperandList(parser, ParallelOp::getCreateZeroKeyword(),
                              createZeroOperands, createZeroOperandTypes,
                              result)))
    return failure();

  // no_create()?
  if (failed(parseOperandList(parser, ParallelOp::getNoCreateKeyword(),
                              noCreateOperands, noCreateOperandTypes, result)))
    return failure();

  // present()?
  if (failed(parseOperandList(parser, ParallelOp::getPresentKeyword(),
                              presentOperands, presentOperandTypes, result)))
    return failure();

  // deviceptr()?
  if (failed(parseOperandList(parser, ParallelOp::getDevicePtrKeyword(),
                              devicePtrOperands, deviceptrOperandTypes,
                              result)))
    return failure();

  // attach()?
  if (failed(parseOperandList(parser, ParallelOp::getAttachKeyword(),
                              attachOperands, attachOperandTypes, result)))
    return failure();

  // private()?
  if (failed(parseOperandList(parser, ParallelOp::getPrivateKeyword(),
                              privateOperands, privateOperandTypes, result)))
    return failure();

  // firstprivate()?
  if (failed(parseOperandList(parser, ParallelOp::getFirstPrivateKeyword(),
                              firstprivateOperands, firstprivateOperandTypes,
                              result)))
    return failure();

  // Parallel op region
  if (failed(parseRegions<ParallelOp>(parser, result)))
    return failure();

  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(async.has_value() ? 1 : 0),
           static_cast<int32_t>(waitOperands.size()),
           static_cast<int32_t>(numGangs.has_value() ? 1 : 0),
           static_cast<int32_t>(numWorkers.has_value() ? 1 : 0),
           static_cast<int32_t>(vectorLength.has_value() ? 1 : 0),
           static_cast<int32_t>(hasIfCond ? 1 : 0),
           static_cast<int32_t>(hasSelfCond ? 1 : 0),
           static_cast<int32_t>(reductionOperands.size()),
           static_cast<int32_t>(copyOperands.size()),
           static_cast<int32_t>(copyinOperands.size()),
           static_cast<int32_t>(copyinReadonlyOperands.size()),
           static_cast<int32_t>(copyoutOperands.size()),
           static_cast<int32_t>(copyoutZeroOperands.size()),
           static_cast<int32_t>(createOperands.size()),
           static_cast<int32_t>(createZeroOperands.size()),
           static_cast<int32_t>(noCreateOperands.size()),
           static_cast<int32_t>(presentOperands.size()),
           static_cast<int32_t>(devicePtrOperands.size()),
           static_cast<int32_t>(attachOperands.size()),
           static_cast<int32_t>(privateOperands.size()),
           static_cast<int32_t>(firstprivateOperands.size())}));

  // Additional attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

void ParallelOp::print(OpAsmPrinter &printer) {
  // getAsync()?
  if (Value async = getAsync())
    printer << " " << ParallelOp::getAsyncKeyword() << "(" << async << ": "
            << async.getType() << ")";

  // getWait()?
  printOperandList(getWaitOperands(), ParallelOp::getWaitKeyword(), printer);

  // num_gangs()?
  if (Value numGangs = getNumGangs())
    printer << " " << ParallelOp::getNumGangsKeyword() << "(" << numGangs
            << ": " << numGangs.getType() << ")";

  // num_workers()?
  if (Value numWorkers = getNumWorkers())
    printer << " " << ParallelOp::getNumWorkersKeyword() << "(" << numWorkers
            << ": " << numWorkers.getType() << ")";

  // vector_length()?
  if (Value vectorLength = getVectorLength())
    printer << " " << ParallelOp::getVectorLengthKeyword() << "("
            << vectorLength << ": " << vectorLength.getType() << ")";

  // if()?
  if (Value ifCond = getIfCond())
    printer << " " << ParallelOp::getIfKeyword() << "(" << ifCond << ")";

  // self()?
  if (Value selfCond = getSelfCond())
    printer << " " << ParallelOp::getSelfKeyword() << "(" << selfCond << ")";

  // reduction()?
  printOperandList(getReductionOperands(), ParallelOp::getReductionKeyword(),
                   printer);

  // copy()?
  printOperandList(getCopyOperands(), ParallelOp::getCopyKeyword(), printer);

  // copyin()?
  printOperandList(getCopyinOperands(), ParallelOp::getCopyinKeyword(),
                   printer);

  // copyin_readonly()?
  printOperandList(getCopyinReadonlyOperands(),
                   ParallelOp::getCopyinReadonlyKeyword(), printer);

  // copyout()?
  printOperandList(getCopyoutOperands(), ParallelOp::getCopyoutKeyword(),
                   printer);

  // copyout_zero()?
  printOperandList(getCopyoutZeroOperands(),
                   ParallelOp::getCopyoutZeroKeyword(), printer);

  // create()?
  printOperandList(getCreateOperands(), ParallelOp::getCreateKeyword(),
                   printer);

  // create_zero()?
  printOperandList(getCreateZeroOperands(), ParallelOp::getCreateZeroKeyword(),
                   printer);

  // no_create()?
  printOperandList(getNoCreateOperands(), ParallelOp::getNoCreateKeyword(),
                   printer);

  // present()?
  printOperandList(getPresentOperands(), ParallelOp::getPresentKeyword(),
                   printer);

  // deviceptr()?
  printOperandList(getDevicePtrOperands(), ParallelOp::getDevicePtrKeyword(),
                   printer);

  // attach()?
  printOperandList(getAttachOperands(), ParallelOp::getAttachKeyword(),
                   printer);

  // private()?
  printOperandList(getGangPrivateOperands(), ParallelOp::getPrivateKeyword(),
                   printer);

  // firstprivate()?
  printOperandList(getGangFirstPrivateOperands(),
                   ParallelOp::getFirstPrivateKeyword(), printer);

  printer << ' ';
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), ParallelOp::getOperandSegmentSizeAttr());
}

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
// LoopOp
//===----------------------------------------------------------------------===//

/// Parse acc.loop operation
/// operation := `acc.loop`
///              (`gang` ( `(` (`num=` value)? (`,` `static=` value `)`)? )? )?
///              (`vector` ( `(` value `)` )? )? (`worker` (`(` value `)`)? )?
///              (`vector_length` `(` value `)`)?
///              (`tile` `(` value-list `)`)?
///              (`private` `(` value-list `)`)?
///              (`reduction` `(` value-list `)`)?
///              region attr-dict?
ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  unsigned executionMapping = OpenACCExecMapping::NONE;
  SmallVector<Type, 8> operandTypes;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> privateOperands,
      reductionOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> tileOperands;
  OptionalParseResult gangNum, gangStatic, worker, vector;

  // gang?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getGangKeyword())))
    executionMapping |= OpenACCExecMapping::GANG;

  // optional gang operand
  if (succeeded(parser.parseOptionalLParen())) {
    gangNum = parserOptionalOperandAndTypeWithPrefix(
        parser, result, LoopOp::getGangNumKeyword());
    if (gangNum.has_value() && failed(*gangNum))
      return failure();
    // FIXME: Comma should require subsequent operands.
    (void)parser.parseOptionalComma();
    gangStatic = parserOptionalOperandAndTypeWithPrefix(
        parser, result, LoopOp::getGangStaticKeyword());
    if (gangStatic.has_value() && failed(*gangStatic))
      return failure();
    // FIXME: Why allow optional last commas?
    (void)parser.parseOptionalComma();
    if (failed(parser.parseRParen()))
      return failure();
  }

  // worker?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getWorkerKeyword())))
    executionMapping |= OpenACCExecMapping::WORKER;

  // optional worker operand
  worker = parseOptionalOperandAndType(parser, result);
  if (worker.has_value() && failed(*worker))
    return failure();

  // vector?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getVectorKeyword())))
    executionMapping |= OpenACCExecMapping::VECTOR;

  // optional vector operand
  vector = parseOptionalOperandAndType(parser, result);
  if (vector.has_value() && failed(*vector))
    return failure();

  // tile()?
  if (failed(parseOperandList(parser, LoopOp::getTileKeyword(), tileOperands,
                              operandTypes, result)))
    return failure();

  // private()?
  if (failed(parseOperandList(parser, LoopOp::getPrivateKeyword(),
                              privateOperands, operandTypes, result)))
    return failure();

  // reduction()?
  if (failed(parseOperandList(parser, LoopOp::getReductionKeyword(),
                              reductionOperands, operandTypes, result)))
    return failure();

  if (executionMapping != acc::OpenACCExecMapping::NONE)
    result.addAttribute(LoopOp::getExecutionMappingAttrStrName(),
                        builder.getI64IntegerAttr(executionMapping));

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  if (failed(parseRegions<LoopOp>(parser, result)))
    return failure();

  result.addAttribute(LoopOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(gangNum.has_value() ? 1 : 0),
                           static_cast<int32_t>(gangStatic.has_value() ? 1 : 0),
                           static_cast<int32_t>(worker.has_value() ? 1 : 0),
                           static_cast<int32_t>(vector.has_value() ? 1 : 0),
                           static_cast<int32_t>(tileOperands.size()),
                           static_cast<int32_t>(privateOperands.size()),
                           static_cast<int32_t>(reductionOperands.size())}));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

void LoopOp::print(OpAsmPrinter &printer) {
  unsigned execMapping = getExecMapping();
  if (execMapping & OpenACCExecMapping::GANG) {
    printer << " " << LoopOp::getGangKeyword();
    Value gangNum = getGangNum();
    Value gangStatic = getGangStatic();

    // Print optional gang operands
    if (gangNum || gangStatic) {
      printer << "(";
      if (gangNum) {
        printer << LoopOp::getGangNumKeyword() << "=" << gangNum << ": "
                << gangNum.getType();
        if (gangStatic)
          printer << ", ";
      }
      if (gangStatic)
        printer << LoopOp::getGangStaticKeyword() << "=" << gangStatic << ": "
                << gangStatic.getType();
      printer << ")";
    }
  }

  if (execMapping & OpenACCExecMapping::WORKER) {
    printer << " " << LoopOp::getWorkerKeyword();

    // Print optional worker operand if present
    if (Value workerNum = getWorkerNum())
      printer << "(" << workerNum << ": " << workerNum.getType() << ")";
  }

  if (execMapping & OpenACCExecMapping::VECTOR) {
    printer << " " << LoopOp::getVectorKeyword();

    // Print optional vector operand if present
    if (Value vectorLength = this->getVectorLength())
      printer << "(" << vectorLength << ": " << vectorLength.getType() << ")";
  }

  // tile()?
  printOperandList(getTileOperands(), LoopOp::getTileKeyword(), printer);

  // private()?
  printOperandList(getPrivateOperands(), LoopOp::getPrivateKeyword(), printer);

  // reduction()?
  printOperandList(getReductionOperands(), LoopOp::getReductionKeyword(),
                   printer);

  if (getNumResults() > 0)
    printer << " -> (" << getResultTypes() << ")";

  printer << ' ';
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {LoopOp::getExecutionMappingAttrStrName(),
                            LoopOp::getOperandSegmentSizeAttr()});
}

LogicalResult acc::LoopOp::verify() {
  // auto, independent and seq attribute are mutually exclusive.
  if ((getAuto_() && (getIndependent() || getSeq())) ||
      (getIndependent() && getSeq())) {
    return emitError("only one of " + acc::LoopOp::getAutoAttrStrName() + ", " +
                     acc::LoopOp::getIndependentAttrStrName() + ", " +
                     acc::LoopOp::getSeqAttrStrName() +
                     " can be present at the same time");
  }

  // Gang, worker and vector are incompatible with seq.
  if (getSeq() && getExecMapping() != OpenACCExecMapping::NONE)
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
