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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace acc;

//===----------------------------------------------------------------------===//
// OpenACC operations
//===----------------------------------------------------------------------===//

void OpenACCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"
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
                 SmallVectorImpl<OpAsmParser::OperandType> &args,
                 SmallVectorImpl<Type> &argTypes, OperationState &result) {
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  if (failed(parser.parseLParen()))
    return failure();

  // Exit early if the list is empty.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  do {
    OpAsmParser::OperandType arg;
    Type type;

    if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
      return failure();

    args.push_back(arg);
    argTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (failed(parser.parseRParen()))
    return failure();

  return parser.resolveOperands(args, argTypes, parser.getCurrentLocation(),
                                result.operands);
}

static void printOperandList(Operation::operand_range operands,
                             StringRef listName, OpAsmPrinter &printer) {

  if (operands.size() > 0) {
    printer << " " << listName << "(";
    llvm::interleaveComma(operands, printer, [&](Value op) {
      printer << op << ": " << op.getType();
    });
    printer << ")";
  }
}

static ParseResult parseOptionalOperand(OpAsmParser &parser, StringRef keyword,
                                        OpAsmParser::OperandType &operand,
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
///                             `copyout` `(` value-list `)`?
///                             `create` `(` value-list `)`?
///                             `no_create` `(` value-list `)`?
///                             `present` `(` value-list `)`?
///                             `deviceptr` `(` value-list `)`?
///                             `attach` `(` value-list `)`?
///                             `private` `(` value-list `)`?
///                             `firstprivate` `(` value-list `)`?
///                             region attr-dict?
static ParseResult parseParallelOp(OpAsmParser &parser,
                                   OperationState &result) {
  Builder &builder = parser.getBuilder();
  SmallVector<OpAsmParser::OperandType, 8> privateOperands,
      firstprivateOperands, createOperands, copyOperands, copyinOperands,
      copyoutOperands, noCreateOperands, presentOperands, devicePtrOperands,
      attachOperands, waitOperands, reductionOperands;
  SmallVector<Type, 8> operandTypes;
  OpAsmParser::OperandType async, numGangs, numWorkers, vectorLength, ifCond,
      selfCond;
  bool hasAsync = false, hasNumGangs = false, hasNumWorkers = false;
  bool hasVectorLength = false, hasIfCond = false, hasSelfCond = false;

  Type indexType = builder.getIndexType();
  Type i1Type = builder.getI1Type();

  // async()?
  if (failed(parseOptionalOperand(parser, ParallelOp::getAsyncKeyword(), async,
                                  indexType, hasAsync, result)))
    return failure();

  // wait()?
  if (failed(parseOperandList(parser, ParallelOp::getWaitKeyword(),
                              waitOperands, operandTypes, result)))
    return failure();

  // num_gangs(value)?
  if (failed(parseOptionalOperand(parser, ParallelOp::getNumGangsKeyword(),
                                  numGangs, indexType, hasNumGangs, result)))
    return failure();

  // num_workers(value)?
  if (failed(parseOptionalOperand(parser, ParallelOp::getNumWorkersKeyword(),
                                  numWorkers, indexType, hasNumWorkers,
                                  result)))
    return failure();

  // vector_length(value)?
  if (failed(parseOptionalOperand(parser, ParallelOp::getVectorLengthKeyword(),
                                  vectorLength, indexType, hasVectorLength,
                                  result)))
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
                              reductionOperands, operandTypes, result)))
    return failure();

  // copy()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyKeyword(),
                              copyOperands, operandTypes, result)))
    return failure();

  // copyin()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyinKeyword(),
                              copyinOperands, operandTypes, result)))
    return failure();

  // copyout()?
  if (failed(parseOperandList(parser, ParallelOp::getCopyoutKeyword(),
                              copyoutOperands, operandTypes, result)))
    return failure();

  // create()?
  if (failed(parseOperandList(parser, ParallelOp::getCreateKeyword(),
                              createOperands, operandTypes, result)))
    return failure();

  // no_create()?
  if (failed(parseOperandList(parser, ParallelOp::getNoCreateKeyword(),
                              noCreateOperands, operandTypes, result)))
    return failure();

  // present()?
  if (failed(parseOperandList(parser, ParallelOp::getPresentKeyword(),
                              presentOperands, operandTypes, result)))
    return failure();

  // deviceptr()?
  if (failed(parseOperandList(parser, ParallelOp::getDevicePtrKeyword(),
                              devicePtrOperands, operandTypes, result)))
    return failure();

  // attach()?
  if (failed(parseOperandList(parser, ParallelOp::getAttachKeyword(),
                              attachOperands, operandTypes, result)))
    return failure();

  // private()?
  if (failed(parseOperandList(parser, ParallelOp::getPrivateKeyword(),
                              privateOperands, operandTypes, result)))
    return failure();

  // firstprivate()?
  if (failed(parseOperandList(parser, ParallelOp::getFirstPrivateKeyword(),
                              firstprivateOperands, operandTypes, result)))
    return failure();

  // Parallel op region
  if (failed(parseRegions<ParallelOp>(parser, result)))
    return failure();

  result.addAttribute(ParallelOp::getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(
                          {static_cast<int32_t>(hasAsync ? 1 : 0),
                           static_cast<int32_t>(waitOperands.size()),
                           static_cast<int32_t>(hasNumGangs ? 1 : 0),
                           static_cast<int32_t>(hasNumWorkers ? 1 : 0),
                           static_cast<int32_t>(hasVectorLength ? 1 : 0),
                           static_cast<int32_t>(hasIfCond ? 1 : 0),
                           static_cast<int32_t>(hasSelfCond ? 1 : 0),
                           static_cast<int32_t>(reductionOperands.size()),
                           static_cast<int32_t>(copyOperands.size()),
                           static_cast<int32_t>(copyinOperands.size()),
                           static_cast<int32_t>(copyoutOperands.size()),
                           static_cast<int32_t>(createOperands.size()),
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

static void print(OpAsmPrinter &printer, ParallelOp &op) {
  printer << ParallelOp::getOperationName();

  // async()?
  if (auto async = op.async())
    printer << " " << ParallelOp::getAsyncKeyword() << "(" << async << ")";

  // wait()?
  printOperandList(op.waitOperands(), ParallelOp::getWaitKeyword(), printer);

  // num_gangs()?
  if (auto numGangs = op.numGangs())
    printer << " " << ParallelOp::getNumGangsKeyword() << "(" << numGangs
            << ")";

  // num_workers()?
  if (auto numWorkers = op.numWorkers())
    printer << " " << ParallelOp::getNumWorkersKeyword() << "(" << numWorkers
            << ")";

  // if()?
  if (Value ifCond = op.ifCond())
    printer << " " << ParallelOp::getIfKeyword() << "(" << ifCond << ")";

  // self()?
  if (Value selfCond = op.selfCond())
    printer << " " << ParallelOp::getSelfKeyword() << "(" << selfCond << ")";

  // reduction()?
  printOperandList(op.reductionOperands(), ParallelOp::getReductionKeyword(),
                   printer);

  // copy()?
  printOperandList(op.copyOperands(), ParallelOp::getCopyKeyword(), printer);

  // copyin()?
  printOperandList(op.copyinOperands(), ParallelOp::getCopyinKeyword(),
                   printer);

  // copyout()?
  printOperandList(op.copyoutOperands(), ParallelOp::getCopyoutKeyword(),
                   printer);

  // create()?
  printOperandList(op.createOperands(), ParallelOp::getCreateKeyword(),
                   printer);

  // no_create()?
  printOperandList(op.noCreateOperands(), ParallelOp::getNoCreateKeyword(),
                   printer);

  // present()?
  printOperandList(op.presentOperands(), ParallelOp::getPresentKeyword(),
                   printer);

  // deviceptr()?
  printOperandList(op.devicePtrOperands(), ParallelOp::getDevicePtrKeyword(),
                   printer);

  // attach()?
  printOperandList(op.attachOperands(), ParallelOp::getAttachKeyword(),
                   printer);

  // private()?
  printOperandList(op.gangPrivateOperands(), ParallelOp::getPrivateKeyword(),
                   printer);

  // firstprivate()?
  printOperandList(op.gangFirstPrivateOperands(),
                   ParallelOp::getFirstPrivateKeyword(), printer);

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(), ParallelOp::getOperandSegmentSizeAttr());
}

//===----------------------------------------------------------------------===//
// DataOp
//===----------------------------------------------------------------------===//

/// Parse acc.data operation
/// operation := `acc.parallel` `present` `(` value-list `)`?
///                             `copy` `(` value-list `)`?
///                             `copyin` `(` value-list `)`?
///                             `copyout` `(` value-list `)`?
///                             `create` `(` value-list `)`?
///                             `no_create` `(` value-list `)`?
///                             `delete` `(` value-list `)`?
///                             `attach` `(` value-list `)`?
///                             `detach` `(` value-list `)`?
///                             region attr-dict?
static ParseResult parseDataOp(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  SmallVector<OpAsmParser::OperandType, 8> presentOperands, copyOperands,
      copyinOperands, copyoutOperands, createOperands, noCreateOperands,
      deleteOperands, attachOperands, detachOperands;
  SmallVector<Type, 8> operandsTypes;

  // present(value-list)?
  if (failed(parseOperandList(parser, DataOp::getPresentKeyword(),
                              presentOperands, operandsTypes, result)))
    return failure();

  // copy(value-list)?
  if (failed(parseOperandList(parser, DataOp::getCopyKeyword(), copyOperands,
                              operandsTypes, result)))
    return failure();

  // copyin(value-list)?
  if (failed(parseOperandList(parser, DataOp::getCopyinKeyword(),
                              copyinOperands, operandsTypes, result)))
    return failure();

  // copyout(value-list)?
  if (failed(parseOperandList(parser, DataOp::getCopyoutKeyword(),
                              copyoutOperands, operandsTypes, result)))
    return failure();

  // create(value-list)?
  if (failed(parseOperandList(parser, DataOp::getCreateKeyword(),
                              createOperands, operandsTypes, result)))
    return failure();

  // no_create(value-list)?
  if (failed(parseOperandList(parser, DataOp::getCreateKeyword(),
                              noCreateOperands, operandsTypes, result)))
    return failure();

  // delete(value-list)?
  if (failed(parseOperandList(parser, DataOp::getDeleteKeyword(),
                              deleteOperands, operandsTypes, result)))
    return failure();

  // attach(value-list)?
  if (failed(parseOperandList(parser, DataOp::getAttachKeyword(),
                              attachOperands, operandsTypes, result)))
    return failure();

  // detach(value-list)?
  if (failed(parseOperandList(parser, DataOp::getDetachKeyword(),
                              detachOperands, operandsTypes, result)))
    return failure();

  // Data op region
  if (failed(parseRegions<ParallelOp>(parser, result)))
    return failure();

  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(presentOperands.size()),
                                static_cast<int32_t>(copyOperands.size()),
                                static_cast<int32_t>(copyinOperands.size()),
                                static_cast<int32_t>(copyoutOperands.size()),
                                static_cast<int32_t>(createOperands.size()),
                                static_cast<int32_t>(noCreateOperands.size()),
                                static_cast<int32_t>(deleteOperands.size()),
                                static_cast<int32_t>(attachOperands.size()),
                                static_cast<int32_t>(detachOperands.size())}));

  // Additional attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

static void print(OpAsmPrinter &printer, DataOp &op) {
  printer << DataOp::getOperationName();

  // present(value-list)?
  printOperandList(op.presentOperands(), DataOp::getPresentKeyword(), printer);

  // copy(value-list)?
  printOperandList(op.copyOperands(), DataOp::getCopyKeyword(), printer);

  // copyin(value-list)?
  printOperandList(op.copyinOperands(), DataOp::getCopyinKeyword(), printer);

  // copyout(value-list)?
  printOperandList(op.copyoutOperands(), DataOp::getCopyoutKeyword(), printer);

  // create(value-list)?
  printOperandList(op.createOperands(), DataOp::getCreateKeyword(), printer);

  // no_create(value-list)?
  printOperandList(op.noCreateOperands(), DataOp::getNoCreateKeyword(),
                   printer);

  // delete(value-list)?
  printOperandList(op.deleteOperands(), DataOp::getDeleteKeyword(), printer);

  // attach(value-list)?
  printOperandList(op.attachOperands(), DataOp::getAttachKeyword(), printer);

  // detach(value-list)?
  printOperandList(op.detachOperands(), DataOp::getDetachKeyword(), printer);

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(), ParallelOp::getOperandSegmentSizeAttr());
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

/// Parse acc.loop operation
/// operation := `acc.loop` `gang`? `vector`? `worker`? `seq`?
///                         `private` `(` value-list `)`?
///                         `reduction` `(` value-list `)`?
///                         region attr-dict?
static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {

  Builder &builder = parser.getBuilder();
  unsigned executionMapping = 0;
  SmallVector<Type, 8> operandTypes;
  SmallVector<OpAsmParser::OperandType, 8> privateOperands, reductionOperands;

  // gang?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getGangAttrName())))
    executionMapping |= OpenACCExecMapping::GANG;

  // vector?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getVectorAttrName())))
    executionMapping |= OpenACCExecMapping::VECTOR;

  // worker?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getWorkerAttrName())))
    executionMapping |= OpenACCExecMapping::WORKER;

  // seq?
  if (succeeded(parser.parseOptionalKeyword(LoopOp::getSeqAttrName())))
    executionMapping |= OpenACCExecMapping::SEQ;

  // private()?
  if (failed(parseOperandList(parser, LoopOp::getPrivateKeyword(),
                              privateOperands, operandTypes, result)))
    return failure();

  // reduction()?
  if (failed(parseOperandList(parser, LoopOp::getReductionKeyword(),
                              reductionOperands, operandTypes, result)))
    return failure();

  if (executionMapping != 0)
    result.addAttribute(LoopOp::getExecutionMappingAttrName(),
                        builder.getI64IntegerAttr(executionMapping));

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  if (failed(parseRegions<LoopOp>(parser, result)))
    return failure();

  result.addAttribute(LoopOp::getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(
                          {static_cast<int32_t>(privateOperands.size()),
                           static_cast<int32_t>(reductionOperands.size())}));

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  return success();
}

static void print(OpAsmPrinter &printer, LoopOp &op) {
  printer << LoopOp::getOperationName();

  unsigned execMapping =
      (op.getAttrOfType<IntegerAttr>(LoopOp::getExecutionMappingAttrName()) !=
       nullptr)
          ? op.getAttrOfType<IntegerAttr>(LoopOp::getExecutionMappingAttrName())
                .getInt()
          : 0;
  if ((execMapping & OpenACCExecMapping::GANG) == OpenACCExecMapping::GANG)
    printer << " " << LoopOp::getGangAttrName();

  if ((execMapping & OpenACCExecMapping::WORKER) == OpenACCExecMapping::WORKER)
    printer << " " << LoopOp::getWorkerAttrName();

  if ((execMapping & OpenACCExecMapping::VECTOR) == OpenACCExecMapping::VECTOR)
    printer << " " << LoopOp::getVectorAttrName();

  if ((execMapping & OpenACCExecMapping::SEQ) == OpenACCExecMapping::SEQ)
    printer << " " << LoopOp::getSeqAttrName();

  // private()?
  printOperandList(op.privateOperands(), LoopOp::getPrivateKeyword(), printer);

  // reduction()?
  printOperandList(op.reductionOperands(), LoopOp::getReductionKeyword(),
                   printer);

  if (op.getNumResults() > 0)
    printer << " -> (" << op.getResultTypes() << ")";

  printer.printRegion(op.region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(), {LoopOp::getExecutionMappingAttrName(),
                      LoopOp::getOperandSegmentSizeAttr()});
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"
