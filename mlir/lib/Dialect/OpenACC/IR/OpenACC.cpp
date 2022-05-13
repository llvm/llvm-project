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
  return llvm::None;
}

/// Parse optional operand and its type wrapped in parenthesis.
/// Example:
///   `(` %vectorLength: i64 `)`
static OptionalParseResult parseOptionalOperandAndType(OpAsmParser &parser,
                                                       OperationState &result) {
  if (succeeded(parser.parseOptionalLParen())) {
    return failure(parseOperandAndType(parser, result) || parser.parseRParen());
  }
  return llvm::None;
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
  return llvm::None;
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
    Value ifCond = op.ifCond();
    if (!ifCond)
      return success();

    IntegerAttr constAttr;
    if (matchPattern(ifCond, m_Constant(&constAttr))) {
      if (constAttr.getInt())
        rewriter.updateRootInPlace(op, [&]() { op.ifCondMutable().erase(0); });
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

  // async()?
  async = parseOptionalOperandAndType(parser, ParallelOp::getAsyncKeyword(),
                                      result);
  if (async.hasValue() && failed(*async))
    return failure();

  // wait()?
  if (failed(parseOperandList(parser, ParallelOp::getWaitKeyword(),
                              waitOperands, waitOperandTypes, result)))
    return failure();

  // num_gangs(value)?
  numGangs = parseOptionalOperandAndType(
      parser, ParallelOp::getNumGangsKeyword(), result);
  if (numGangs.hasValue() && failed(*numGangs))
    return failure();

  // num_workers(value)?
  numWorkers = parseOptionalOperandAndType(
      parser, ParallelOp::getNumWorkersKeyword(), result);
  if (numWorkers.hasValue() && failed(*numWorkers))
    return failure();

  // vector_length(value)?
  vectorLength = parseOptionalOperandAndType(
      parser, ParallelOp::getVectorLengthKeyword(), result);
  if (vectorLength.hasValue() && failed(*vectorLength))
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
      builder.getI32VectorAttr(
          {static_cast<int32_t>(async.hasValue() ? 1 : 0),
           static_cast<int32_t>(waitOperands.size()),
           static_cast<int32_t>(numGangs.hasValue() ? 1 : 0),
           static_cast<int32_t>(numWorkers.hasValue() ? 1 : 0),
           static_cast<int32_t>(vectorLength.hasValue() ? 1 : 0),
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
  // async()?
  if (Value async = this->async())
    printer << " " << ParallelOp::getAsyncKeyword() << "(" << async << ": "
            << async.getType() << ")";

  // wait()?
  printOperandList(waitOperands(), ParallelOp::getWaitKeyword(), printer);

  // num_gangs()?
  if (Value numGangs = this->numGangs())
    printer << " " << ParallelOp::getNumGangsKeyword() << "(" << numGangs
            << ": " << numGangs.getType() << ")";

  // num_workers()?
  if (Value numWorkers = this->numWorkers())
    printer << " " << ParallelOp::getNumWorkersKeyword() << "(" << numWorkers
            << ": " << numWorkers.getType() << ")";

  // vector_length()?
  if (Value vectorLength = this->vectorLength())
    printer << " " << ParallelOp::getVectorLengthKeyword() << "("
            << vectorLength << ": " << vectorLength.getType() << ")";

  // if()?
  if (Value ifCond = this->ifCond())
    printer << " " << ParallelOp::getIfKeyword() << "(" << ifCond << ")";

  // self()?
  if (Value selfCond = this->selfCond())
    printer << " " << ParallelOp::getSelfKeyword() << "(" << selfCond << ")";

  // reduction()?
  printOperandList(reductionOperands(), ParallelOp::getReductionKeyword(),
                   printer);

  // copy()?
  printOperandList(copyOperands(), ParallelOp::getCopyKeyword(), printer);

  // copyin()?
  printOperandList(copyinOperands(), ParallelOp::getCopyinKeyword(), printer);

  // copyin_readonly()?
  printOperandList(copyinReadonlyOperands(),
                   ParallelOp::getCopyinReadonlyKeyword(), printer);

  // copyout()?
  printOperandList(copyoutOperands(), ParallelOp::getCopyoutKeyword(), printer);

  // copyout_zero()?
  printOperandList(copyoutZeroOperands(), ParallelOp::getCopyoutZeroKeyword(),
                   printer);

  // create()?
  printOperandList(createOperands(), ParallelOp::getCreateKeyword(), printer);

  // create_zero()?
  printOperandList(createZeroOperands(), ParallelOp::getCreateZeroKeyword(),
                   printer);

  // no_create()?
  printOperandList(noCreateOperands(), ParallelOp::getNoCreateKeyword(),
                   printer);

  // present()?
  printOperandList(presentOperands(), ParallelOp::getPresentKeyword(), printer);

  // deviceptr()?
  printOperandList(devicePtrOperands(), ParallelOp::getDevicePtrKeyword(),
                   printer);

  // attach()?
  printOperandList(attachOperands(), ParallelOp::getAttachKeyword(), printer);

  // private()?
  printOperandList(gangPrivateOperands(), ParallelOp::getPrivateKeyword(),
                   printer);

  // firstprivate()?
  printOperandList(gangFirstPrivateOperands(),
                   ParallelOp::getFirstPrivateKeyword(), printer);

  printer << ' ';
  printer.printRegion(region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), ParallelOp::getOperandSegmentSizeAttr());
}

unsigned ParallelOp::getNumDataOperands() {
  return reductionOperands().size() + copyOperands().size() +
         copyinOperands().size() + copyinReadonlyOperands().size() +
         copyoutOperands().size() + copyoutZeroOperands().size() +
         createOperands().size() + createZeroOperands().size() +
         noCreateOperands().size() + presentOperands().size() +
         devicePtrOperands().size() + attachOperands().size() +
         gangPrivateOperands().size() + gangFirstPrivateOperands().size();
}

Value ParallelOp::getDataOperand(unsigned i) {
  unsigned numOptional = async() ? 1 : 0;
  numOptional += numGangs() ? 1 : 0;
  numOptional += numWorkers() ? 1 : 0;
  numOptional += vectorLength() ? 1 : 0;
  numOptional += ifCond() ? 1 : 0;
  numOptional += selfCond() ? 1 : 0;
  return getOperand(waitOperands().size() + numOptional + i);
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
    if (gangNum.hasValue() && failed(*gangNum))
      return failure();
    // FIXME: Comma should require subsequent operands.
    (void)parser.parseOptionalComma();
    gangStatic = parserOptionalOperandAndTypeWithPrefix(
        parser, result, LoopOp::getGangStaticKeyword());
    if (gangStatic.hasValue() && failed(*gangStatic))
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
  if (worker.hasValue() && failed(*worker))
    return failure();

  // vector?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getVectorKeyword())))
    executionMapping |= OpenACCExecMapping::VECTOR;

  // optional vector operand
  vector = parseOptionalOperandAndType(parser, result);
  if (vector.hasValue() && failed(*vector))
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
    result.addAttribute(LoopOp::getExecutionMappingAttrName(),
                        builder.getI64IntegerAttr(executionMapping));

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  if (failed(parseRegions<LoopOp>(parser, result)))
    return failure();

  result.addAttribute(LoopOp::getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(
                          {static_cast<int32_t>(gangNum.hasValue() ? 1 : 0),
                           static_cast<int32_t>(gangStatic.hasValue() ? 1 : 0),
                           static_cast<int32_t>(worker.hasValue() ? 1 : 0),
                           static_cast<int32_t>(vector.hasValue() ? 1 : 0),
                           static_cast<int32_t>(tileOperands.size()),
                           static_cast<int32_t>(privateOperands.size()),
                           static_cast<int32_t>(reductionOperands.size())}));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

void LoopOp::print(OpAsmPrinter &printer) {
  unsigned execMapping = exec_mapping();
  if (execMapping & OpenACCExecMapping::GANG) {
    printer << " " << LoopOp::getGangKeyword();
    Value gangNum = this->gangNum();
    Value gangStatic = this->gangStatic();

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
    if (Value workerNum = this->workerNum())
      printer << "(" << workerNum << ": " << workerNum.getType() << ")";
  }

  if (execMapping & OpenACCExecMapping::VECTOR) {
    printer << " " << LoopOp::getVectorKeyword();

    // Print optional vector operand if present
    if (Value vectorLength = this->vectorLength())
      printer << "(" << vectorLength << ": " << vectorLength.getType() << ")";
  }

  // tile()?
  printOperandList(tileOperands(), LoopOp::getTileKeyword(), printer);

  // private()?
  printOperandList(privateOperands(), LoopOp::getPrivateKeyword(), printer);

  // reduction()?
  printOperandList(reductionOperands(), LoopOp::getReductionKeyword(), printer);

  if (getNumResults() > 0)
    printer << " -> (" << getResultTypes() << ")";

  printer << ' ';
  printer.printRegion(region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {LoopOp::getExecutionMappingAttrName(),
                            LoopOp::getOperandSegmentSizeAttr()});
}

LogicalResult acc::LoopOp::verify() {
  // auto, independent and seq attribute are mutually exclusive.
  if ((auto_() && (independent() || seq())) || (independent() && seq())) {
    return emitError("only one of " + acc::LoopOp::getAutoAttrName() + ", " +
                     acc::LoopOp::getIndependentAttrName() + ", " +
                     acc::LoopOp::getSeqAttrName() +
                     " can be present at the same time");
  }

  // Gang, worker and vector are incompatible with seq.
  if (seq() && exec_mapping() != OpenACCExecMapping::NONE)
    return emitError("gang, worker or vector cannot appear with the seq attr");

  // Check non-empty body().
  if (region().empty())
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
  if (getOperands().empty() && !defaultAttr())
    return emitError("at least one operand or the default attribute "
                     "must appear on the data operation");
  return success();
}

unsigned DataOp::getNumDataOperands() {
  return copyOperands().size() + copyinOperands().size() +
         copyinReadonlyOperands().size() + copyoutOperands().size() +
         copyoutZeroOperands().size() + createOperands().size() +
         createZeroOperands().size() + noCreateOperands().size() +
         presentOperands().size() + deviceptrOperands().size() +
         attachOperands().size();
}

Value DataOp::getDataOperand(unsigned i) {
  unsigned numOptional = ifCond() ? 1 : 0;
  return getOperand(numOptional + i);
}

//===----------------------------------------------------------------------===//
// ExitDataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::ExitDataOp::verify() {
  // 2.6.6. Data Exit Directive restriction
  // At least one copyout, delete, or detach clause must appear on an exit data
  // directive.
  if (copyoutOperands().empty() && deleteOperands().empty() &&
      detachOperands().empty())
    return emitError(
        "at least one operand in copyout, delete or detach must appear on the "
        "exit data operation");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (asyncOperand() && async())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!waitOperands().empty() && wait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (waitDevnum() && waitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned ExitDataOp::getNumDataOperands() {
  return copyoutOperands().size() + deleteOperands().size() +
         detachOperands().size();
}

Value ExitDataOp::getDataOperand(unsigned i) {
  unsigned numOptional = ifCond() ? 1 : 0;
  numOptional += asyncOperand() ? 1 : 0;
  numOptional += waitDevnum() ? 1 : 0;
  return getOperand(waitOperands().size() + numOptional + i);
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
  if (copyinOperands().empty() && createOperands().empty() &&
      createZeroOperands().empty() && attachOperands().empty())
    return emitError(
        "at least one operand in copyin, create, "
        "create_zero or attach must appear on the enter data operation");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (asyncOperand() && async())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!waitOperands().empty() && wait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (waitDevnum() && waitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned EnterDataOp::getNumDataOperands() {
  return copyinOperands().size() + createOperands().size() +
         createZeroOperands().size() + attachOperands().size();
}

Value EnterDataOp::getDataOperand(unsigned i) {
  unsigned numOptional = ifCond() ? 1 : 0;
  numOptional += asyncOperand() ? 1 : 0;
  numOptional += waitDevnum() ? 1 : 0;
  return getOperand(waitOperands().size() + numOptional + i);
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
  if (hostOperands().empty() && deviceOperands().empty())
    return emitError(
        "at least one value must be present in hostOperands or deviceOperands");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (asyncOperand() && async())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!waitOperands().empty() && wait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (waitDevnum() && waitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned UpdateOp::getNumDataOperands() {
  return hostOperands().size() + deviceOperands().size();
}

Value UpdateOp::getDataOperand(unsigned i) {
  unsigned numOptional = asyncOperand() ? 1 : 0;
  numOptional += waitDevnum() ? 1 : 0;
  numOptional += ifCond() ? 1 : 0;
  return getOperand(waitOperands().size() + deviceTypeOperands().size() +
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
  if (asyncOperand() && async())
    return emitError("async attribute cannot appear with asyncOperand");

  if (waitDevnum() && waitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.cpp.inc"
