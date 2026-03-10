//===- OpenACCCG.cpp - OpenACC codegen ops, attributes, and types ---------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for OpenACC codegen operations, attributes, and types.
// These correspond to the definitions in OpenACCCG*.td tablegen files
// and are kept in a separate file because they do not represent direct mappings
// of OpenACC language constructs; they are intermediate representations used
// when decomposing and lowering primary `acc` dialect operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace acc;

namespace {

/// Generic helper for single-region OpenACC ops that execute their body once
/// and then return to the parent operation with their results (if any).
static void
getSingleRegionOpSuccessorRegions(Operation *op, Region &region,
                                  RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&region));
    return;
  }
  regions.push_back(RegionSuccessor::parent());
}

static ValueRange getSingleRegionSuccessorInputs(Operation *op,
                                                 RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(op->getResults()) : ValueRange();
}

/// Remove empty acc.kernel_environment operations. If the operation has wait
/// operands, create a acc.wait operation to preserve synchronization.
struct RemoveEmptyKernelEnvironment
    : public OpRewritePattern<acc::KernelEnvironmentOp> {
  using OpRewritePattern<acc::KernelEnvironmentOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(acc::KernelEnvironmentOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumRegions() == 1 && "expected op to have one region");

    Block &block = op.getRegion().front();
    if (!block.empty())
      return failure();

    // Conservatively disable canonicalization of empty acc.kernel_environment
    // operations if the wait operands in the kernel_environment cannot be fully
    // represented by acc.wait operation.

    // Disable canonicalization if device type is not the default
    if (auto deviceTypeAttr = op.getWaitOperandsDeviceTypeAttr()) {
      for (auto attr : deviceTypeAttr) {
        if (auto dtAttr = mlir::dyn_cast<acc::DeviceTypeAttr>(attr)) {
          if (dtAttr.getValue() != mlir::acc::DeviceType::None)
            return failure();
        }
      }
    }

    // Disable canonicalization if any wait segment has a devnum
    if (auto hasDevnumAttr = op.getHasWaitDevnumAttr()) {
      for (auto attr : hasDevnumAttr) {
        if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr)) {
          if (boolAttr.getValue())
            return failure();
        }
      }
    }

    // Disable canonicalization if there are multiple wait segments
    if (auto segmentsAttr = op.getWaitOperandsSegmentsAttr()) {
      if (segmentsAttr.size() > 1)
        return failure();
    }

    // Remove empty kernel environment.
    // Preserve synchronization by creating acc.wait operation if needed.
    if (!op.getWaitOperands().empty() || op.getWaitOnlyAttr())
      rewriter.replaceOpWithNewOp<acc::WaitOp>(op, op.getWaitOperands(),
                                               /*asyncOperand=*/Value(),
                                               /*waitDevnum=*/Value(),
                                               /*async=*/nullptr,
                                               /*ifCond=*/Value());
    else
      rewriter.eraseOp(op);

    return success();
  }
};

template <typename EffectTy>
static void addOperandEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    const MutableOperandRange &operand) {
  for (unsigned i = 0, e = operand.size(); i < e; ++i)
    effects.emplace_back(EffectTy::get(), &operand[i]);
}

template <typename EffectTy>
static void addResultEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    Value result) {
  effects.emplace_back(EffectTy::get(), mlir::cast<mlir::OpResult>(result));
}

static int64_t gpuProcessorIndex(gpu::Processor p) {
  switch (p) {
  case gpu::Processor::Sequential:
    return 0;
  case gpu::Processor::ThreadX:
    return 1;
  case gpu::Processor::ThreadY:
    return 2;
  case gpu::Processor::ThreadZ:
    return 3;
  case gpu::Processor::BlockX:
    return 4;
  case gpu::Processor::BlockY:
    return 5;
  case gpu::Processor::BlockZ:
    return 6;
  }
  llvm_unreachable("unhandled gpu::Processor");
}

static gpu::Processor indexToGpuProcessor(int64_t idx) {
  switch (idx) {
  case 0:
    return gpu::Processor::Sequential;
  case 1:
    return gpu::Processor::ThreadX;
  case 2:
    return gpu::Processor::ThreadY;
  case 3:
    return gpu::Processor::ThreadZ;
  case 4:
    return gpu::Processor::BlockX;
  case 5:
    return gpu::Processor::BlockY;
  case 6:
    return gpu::Processor::BlockZ;
  default:
    return gpu::Processor::Sequential;
  }
}

static GPUParallelDimAttr intToParDim(MLIRContext *context, int64_t dimInt) {
  return GPUParallelDimAttr::get(
      context, IntegerAttr::get(IndexType::get(context), dimInt));
}

static GPUParallelDimAttr processorParDim(MLIRContext *context,
                                          gpu::Processor proc) {
  return GPUParallelDimAttr::get(
      context,
      IntegerAttr::get(IndexType::get(context), gpuProcessorIndex(proc)));
}

static ParseResult parseProcessorValue(AsmParser &parser,
                                       GPUParallelDimAttr &dim) {
  std::string keyword;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseKeywordOrString(&keyword)))
    return failure();
  auto maybeProcessor = gpu::symbolizeProcessor(keyword);
  if (!maybeProcessor)
    return parser.emitError(loc)
           << "expected one of ::mlir::gpu::Processor enum names";
  dim = intToParDim(parser.getContext(), gpuProcessorIndex(*maybeProcessor));
  return success();
}

static void printProcessorValue(AsmPrinter &printer,
                                const GPUParallelDimAttr &attr) {
  gpu::Processor processor = indexToGpuProcessor(attr.getValue().getInt());
  printer << gpu::stringifyProcessor(processor);
}

} // namespace

//===----------------------------------------------------------------------===//
// KernelEnvironmentOp
//===----------------------------------------------------------------------===//

void KernelEnvironmentOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  getSingleRegionOpSuccessorRegions(getOperation(), getRegion(), point,
                                    regions);
}

ValueRange KernelEnvironmentOp::getSuccessorInputs(RegionSuccessor successor) {
  return getSingleRegionSuccessorInputs(getOperation(), successor);
}

void KernelEnvironmentOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveEmptyKernelEnvironment>(context);
}

template <typename ComputeConstructT>
KernelEnvironmentOp
KernelEnvironmentOp::createAndPopulate(ComputeConstructT computeConstruct,
                                       OpBuilder &builder) {
  auto kernelEnvironment = KernelEnvironmentOp::create(
      builder, computeConstruct->getLoc(),
      computeConstruct.getDataClauseOperands(),
      computeConstruct.getAsyncOperands(),
      computeConstruct.getAsyncOperandsDeviceTypeAttr(),
      computeConstruct.getAsyncOnlyAttr(), computeConstruct.getWaitOperands(),
      computeConstruct.getWaitOperandsSegmentsAttr(),
      computeConstruct.getWaitOperandsDeviceTypeAttr(),
      computeConstruct.getHasWaitDevnumAttr(),
      computeConstruct.getWaitOnlyAttr());
  Block &block = kernelEnvironment.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(&block);
  return kernelEnvironment;
}

template KernelEnvironmentOp
KernelEnvironmentOp::createAndPopulate<ParallelOp>(ParallelOp, OpBuilder &);
template KernelEnvironmentOp
KernelEnvironmentOp::createAndPopulate<KernelsOp>(KernelsOp, OpBuilder &);
template KernelEnvironmentOp
KernelEnvironmentOp::createAndPopulate<SerialOp>(SerialOp, OpBuilder &);

//===----------------------------------------------------------------------===//
// FirstprivateMapInitialOp
//===----------------------------------------------------------------------===//

LogicalResult FirstprivateMapInitialOp::verify() {
  if (getDataClause() != acc::DataClause::acc_firstprivate)
    return emitError("data clause associated with firstprivate operation must "
                     "match its intent");
  if (!getVar())
    return emitError("must have var operand");
  if (!mlir::isa<mlir::acc::PointerLikeType>(getVar().getType()) &&
      !mlir::isa<mlir::acc::MappableType>(getVar().getType()))
    return emitError("var must be mappable or pointer-like");
  if (mlir::isa<mlir::acc::PointerLikeType>(getVar().getType()) &&
      getVarType() == getVar().getType())
    return emitError("varType must capture the element type of var");
  if (getModifiers() != acc::DataClauseModifier::none)
    return emitError("no data clause modifiers are allowed");
  return success();
}

void FirstprivateMapInitialOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       acc::CurrentDeviceIdResource::get());
  addOperandEffect<MemoryEffects::Read>(effects, getVarMutable());
  addResultEffect<MemoryEffects::Write>(effects, getAccVar());
}

//===----------------------------------------------------------------------===//
// ReductionInitOp
//===----------------------------------------------------------------------===//

void ReductionInitOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  getSingleRegionOpSuccessorRegions(getOperation(), getRegion(), point,
                                    regions);
}

void ReductionInitOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.emplace_back(1, 1);
}

ValueRange ReductionInitOp::getSuccessorInputs(RegionSuccessor successor) {
  return getSingleRegionSuccessorInputs(getOperation(), successor);
}

LogicalResult ReductionInitOp::verify() {
  Block &block = getRegion().front();
  if (auto yieldOp = dyn_cast<acc::YieldOp>(block.getTerminator())) {
    if (yieldOp.getNumOperands() != 1)
      return emitOpError(
          "region must yield exactly one value (private storage)");
    if (yieldOp.getOperand(0).getType() != getVar().getType())
      return emitOpError("yielded value type must match var type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReductionCombineRegionOp
//===----------------------------------------------------------------------===//

void ReductionCombineRegionOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  getSingleRegionOpSuccessorRegions(getOperation(), getRegion(), point,
                                    regions);
}

void ReductionCombineRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.emplace_back(1, 1);
}

ValueRange
ReductionCombineRegionOp::getSuccessorInputs(RegionSuccessor successor) {
  return getSingleRegionSuccessorInputs(getOperation(), successor);
}

LogicalResult ReductionCombineRegionOp::verify() {
  Block &block = getRegion().front();
  if (auto yieldOp = dyn_cast<acc::YieldOp>(block.getTerminator())) {
    if (yieldOp.getNumOperands() != 0)
      return emitOpError("region must be terminated by acc.yield with no "
                         "operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReductionCombineOp
//===----------------------------------------------------------------------===//

void ReductionCombineOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMemrefMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getDestMemrefMutable(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMemrefMutable(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ComputeRegionOp
//===----------------------------------------------------------------------===//

static ParWidthOp getParWidthOpForLaunchArg(ComputeRegionOp op,
                                            GPUParallelDimAttr parDim) {
  for (auto launchArg : op.getLaunchArgs()) {
    auto parOp = launchArg.getDefiningOp<ParWidthOp>();
    if (!parOp)
      continue;
    auto launchArgDim = cast<GPUParallelDimAttr>(parOp.getParDim());
    if (launchArgDim == parDim)
      return parOp;
  }
  return nullptr;
}

std::optional<Value> ComputeRegionOp::getLaunchArg(GPUParallelDimAttr parDim) {
  if (auto parWidthOp = getParWidthOpForLaunchArg(*this, parDim))
    return parWidthOp.getResult();
  return {};
}

std::optional<Value>
ComputeRegionOp::getKnownLaunchArg(GPUParallelDimAttr parDim) {
  if (auto parWidthOp = getParWidthOpForLaunchArg(*this, parDim))
    if (parWidthOp.getLaunchArg())
      return parWidthOp.getLaunchArg();
  return {};
}

std::optional<uint64_t>
ComputeRegionOp::getKnownConstantLaunchArg(GPUParallelDimAttr parDim) {
  auto knownParWidth = getKnownLaunchArg(parDim);
  if (knownParWidth.has_value())
    return getConstantIntValue(knownParWidth.value());
  return {};
}

BlockArgument ComputeRegionOp::appendInputArg(Value value) {
  getInputArgsMutable().append(value);
  return getBody()->addArgument(value.getType(), getLoc());
}

bool ComputeRegionOp::isEffectivelySerial() {
  auto *ctx = getContext();

  if (getLaunchArg(GPUParallelDimAttr::seqDim(ctx)))
    return true;

  auto checkDim = [&](GPUParallelDimAttr dim) -> bool {
    auto val = getKnownConstantLaunchArg(dim);
    return val && *val == 1;
  };

  return checkDim(GPUParallelDimAttr::threadXDim(ctx)) &&
         checkDim(GPUParallelDimAttr::threadYDim(ctx)) &&
         checkDim(GPUParallelDimAttr::threadZDim(ctx)) &&
         checkDim(GPUParallelDimAttr::blockXDim(ctx)) &&
         checkDim(GPUParallelDimAttr::blockYDim(ctx)) &&
         checkDim(GPUParallelDimAttr::blockZDim(ctx));
}

BlockArgument ComputeRegionOp::parDimToWidth(GPUParallelDimAttr parDim) {
  for (auto [pos, launchArg] : llvm::enumerate(getLaunchArgs())) {
    auto parOp = launchArg.getDefiningOp<ParWidthOp>();
    assert(parOp);
    auto launchArgDim = cast<GPUParallelDimAttr>(parOp.getParDim());
    if (launchArgDim == parDim) {
      assert(pos < getRegion().front().getNumArguments() &&
             "launch arg position out of range");
      return getRegion().front().getArgument(pos);
    }
  }
  llvm_unreachable("attempting to get unspecified parDim");
}

SmallVector<GPUParallelDimAttr> ComputeRegionOp::getLaunchParDims() {
  SmallVector<GPUParallelDimAttr> parDims;
  for (auto launchArg : getLaunchArgs()) {
    auto parOp = launchArg.getDefiningOp<ParWidthOp>();
    auto launchArgDim = cast<GPUParallelDimAttr>(parOp.getParDim());
    int64_t dimInt = launchArgDim.getValue().getInt();
    parDims.push_back(intToParDim(getContext(), dimInt));
  }
  return parDims;
}

Value ComputeRegionOp::getOperand(BlockArgument blockArg) {
  unsigned argNumber = blockArg.getArgNumber();
  unsigned numLaunchArgs = getLaunchArgs().size();
  unsigned numInputArgs = getInputArgs().size();
  assert(argNumber < (numLaunchArgs + numInputArgs) &&
         "invalid block argument");
  if (argNumber < numLaunchArgs)
    return getLaunchArgs()[argNumber];
  return getInputArgs()[argNumber - numLaunchArgs];
}

BlockArgument ComputeRegionOp::gpuParWidth(gpu::Processor processor) {
  return parDimToWidth(GPUParallelDimAttr::get(getContext(), processor));
}

LogicalResult ComputeRegionOp::verify() {
  unsigned expectedBlockArgs = getLaunchArgs().size() + getInputArgs().size();
  unsigned actualBlockArgs = getRegion().front().getNumArguments();
  if (expectedBlockArgs != actualBlockArgs)
    return emitOpError("expected ")
           << expectedBlockArgs << " block arguments (launch + input), got "
           << actualBlockArgs;

  return success();
}

void ComputeRegionOp::print(OpAsmPrinter &p) {
  ValueRange regionArgs = getBody()->getArguments();
  ValueRange launchArgs = getLaunchArgs();
  ValueRange inputArgs = getInputArgs();

  assert(regionArgs.size() == (launchArgs.size() + inputArgs.size()) &&
         "region args mismatch");

  if (getStream())
    p << " stream(" << getStream() << " : " << getStream().getType() << ")";

  size_t i = 0;
  if (!launchArgs.empty()) {
    p << " launch(";
    for (size_t j = 0; j < launchArgs.size(); ++j, ++i) {
      p << regionArgs[i] << " = " << launchArgs[j];
      if (j < launchArgs.size() - 1)
        p << ", ";
    }
    p << ")";
  }
  if (!inputArgs.empty()) {
    p << " ins(";
    for (size_t j = 0; j < inputArgs.size(); ++j, ++i) {
      p << regionArgs[i] << " = " << inputArgs[j];
      if (j < inputArgs.size() - 1)
        p << ", ";
    }
    p << ") : (";
    for (size_t j = 0; j < inputArgs.size(); ++j) {
      p << inputArgs[j].getType();
      if (j < inputArgs.size() - 1)
        p << ", ";
    }
    p << ")";
  }
  p.printOptionalArrowTypeList(getResultTypes());
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/getOperandSegmentSizeAttr());
}

ParseResult ComputeRegionOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::Argument> regionArgs;
  OpAsmParser::UnresolvedOperand streamOperand;
  Type streamType;
  SmallVector<OpAsmParser::UnresolvedOperand> launchOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  SmallVector<Type> types;

  bool hasStream = false;
  if (succeeded(parser.parseOptionalKeyword("stream"))) {
    hasStream = true;
    if (parser.parseLParen() || parser.parseOperand(streamOperand) ||
        parser.parseColon() || parser.parseType(streamType) ||
        parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("launch"))) {
    if (parser.parseAssignmentList(regionArgs, launchOperands))
      return failure();
    auto parWidthType = acc::ParWidthType::get(builder.getContext());
    for (size_t i = 0; i < regionArgs.size(); ++i)
      types.push_back(parWidthType);
  }

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseAssignmentList(regionArgs, inputOperands) ||
        parser.parseColon() || parser.parseLParen() ||
        parser.parseTypeList(types) || parser.parseRParen())
      return failure();
  }

  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  for (auto [iterArg, type] : llvm::zip_equal(regionArgs, types))
    iterArg.type = type;

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  const size_t numLaunchOperands = launchOperands.size();
  const size_t numInputOperands = inputOperands.size();
  assert(numLaunchOperands + numInputOperands == regionArgs.size() &&
         "compute region args mismatch");

  result.addAttribute(
      ComputeRegionOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(numLaunchOperands),
                                    static_cast<int32_t>(numInputOperands),
                                    hasStream ? 1 : 0}));

  for (size_t i = 0; i < numLaunchOperands; ++i) {
    if (parser.resolveOperand(launchOperands[i], types[i], result.operands))
      return failure();
  }

  for (size_t i = numLaunchOperands; i < regionArgs.size(); ++i) {
    if (parser.resolveOperand(inputOperands[i - numLaunchOperands], types[i],
                              result.operands))
      return failure();
  }

  if (hasStream) {
    if (parser.resolveOperand(streamOperand, streamType, result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// GPUParallelDimAttr
//===----------------------------------------------------------------------===//

GPUParallelDimAttr GPUParallelDimAttr::get(MLIRContext *context,
                                           gpu::Processor proc) {
  return processorParDim(context, proc);
}

GPUParallelDimAttr GPUParallelDimAttr::seqDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::Sequential);
}

GPUParallelDimAttr GPUParallelDimAttr::threadXDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::ThreadX);
}

GPUParallelDimAttr GPUParallelDimAttr::threadYDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::ThreadY);
}

GPUParallelDimAttr GPUParallelDimAttr::threadZDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::ThreadZ);
}

GPUParallelDimAttr GPUParallelDimAttr::blockXDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::BlockX);
}

GPUParallelDimAttr GPUParallelDimAttr::blockYDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::BlockY);
}

GPUParallelDimAttr GPUParallelDimAttr::blockZDim(MLIRContext *context) {
  return processorParDim(context, gpu::Processor::BlockZ);
}

Attribute GPUParallelDimAttr::parse(AsmParser &parser, Type type) {
  GPUParallelDimAttr dim;
  if (parser.parseLess() || parseProcessorValue(parser, dim) ||
      parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected format `<` processor_name `>`");
    return {};
  }
  return dim;
}

void GPUParallelDimAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printProcessorValue(printer, *this);
  printer << ">";
}

GPUParallelDimAttr GPUParallelDimAttr::threadDim(MLIRContext *context,
                                                 unsigned index) {
  assert(index <= 2 && "thread dimension index must be 0, 1, or 2");
  switch (index) {
  case 0:
    return threadXDim(context);
  case 1:
    return threadYDim(context);
  case 2:
    return threadZDim(context);
  }
  llvm_unreachable("validated thread dimension index");
}

GPUParallelDimAttr GPUParallelDimAttr::blockDim(MLIRContext *context,
                                                unsigned index) {
  assert(index <= 2 && "block dimension index must be 0, 1, or 2");
  switch (index) {
  case 0:
    return blockXDim(context);
  case 1:
    return blockYDim(context);
  case 2:
    return blockZDim(context);
  }
  llvm_unreachable("validated block dimension index");
}

gpu::Processor GPUParallelDimAttr::getProcessor() const {
  return indexToGpuProcessor(getValue().getInt());
}

int GPUParallelDimAttr::getOrder() const {
  return gpuProcessorIndex(getProcessor());
}

GPUParallelDimAttr GPUParallelDimAttr::getOneHigher() const {
  int order = getOrder();
  if (order >= 6) // BlockZ is the highest
    return *this;
  return get(getContext(), indexToGpuProcessor(order + 1));
}

GPUParallelDimAttr GPUParallelDimAttr::getOneLower() const {
  int order = getOrder();
  if (order <= 0) // Sequential is the lowest
    return *this;
  return get(getContext(), indexToGpuProcessor(order - 1));
}

bool GPUParallelDimAttr::isSeq() const {
  return getProcessor() == gpu::Processor::Sequential;
}
bool GPUParallelDimAttr::isThreadX() const {
  return getProcessor() == gpu::Processor::ThreadX;
}
bool GPUParallelDimAttr::isThreadY() const {
  return getProcessor() == gpu::Processor::ThreadY;
}
bool GPUParallelDimAttr::isThreadZ() const {
  return getProcessor() == gpu::Processor::ThreadZ;
}
bool GPUParallelDimAttr::isBlockX() const {
  return getProcessor() == gpu::Processor::BlockX;
}
bool GPUParallelDimAttr::isBlockY() const {
  return getProcessor() == gpu::Processor::BlockY;
}
bool GPUParallelDimAttr::isBlockZ() const {
  return getProcessor() == gpu::Processor::BlockZ;
}
bool GPUParallelDimAttr::isAnyThread() const {
  return isThreadX() || isThreadY() || isThreadZ();
}
bool GPUParallelDimAttr::isAnyBlock() const {
  return isBlockX() || isBlockY() || isBlockZ();
}

//===----------------------------------------------------------------------===//
// GPUParallelDimsAttr
//===----------------------------------------------------------------------===//

GPUParallelDimsAttr GPUParallelDimsAttr::seq(MLIRContext *ctx) {
  return GPUParallelDimsAttr::get(ctx, {GPUParallelDimAttr::seqDim(ctx)});
}

bool GPUParallelDimsAttr::isSeq() const {
  assert(!getArray().empty() && "no par_dims found");
  if (getArray().size() == 1) {
    auto parDim = dyn_cast<GPUParallelDimAttr>(getArray()[0]);
    assert(parDim && "expected GPUParallelDimAttr");
    return parDim.isSeq();
  }
  return false;
}

bool GPUParallelDimsAttr::isParallel() const { return !isSeq(); }

bool GPUParallelDimsAttr::isMultiDim() const { return getArray().size() > 1; }

bool GPUParallelDimsAttr::hasAnyBlockLevel() const {
  return llvm::any_of(
      getArray(), [](const GPUParallelDimAttr &p) { return p.isAnyBlock(); });
}

bool GPUParallelDimsAttr::hasOnlyBlockLevel() const {
  return !getArray().empty() &&
         llvm::all_of(getArray(), [](const GPUParallelDimAttr &p) {
           return p.isAnyBlock();
         });
}

bool GPUParallelDimsAttr::hasOnlyThreadYLevel() const {
  return !getArray().empty() &&
         llvm::all_of(getArray(), [](const GPUParallelDimAttr &p) {
           return p.isThreadY();
         });
}

bool GPUParallelDimsAttr::hasOnlyThreadXLevel() const {
  return !getArray().empty() &&
         llvm::all_of(getArray(), [](const GPUParallelDimAttr &p) {
           return p.isThreadX();
         });
}

Attribute GPUParallelDimsAttr::parse(AsmParser &parser, Type type) {
  auto delimiter = AsmParser::Delimiter::Square;
  SmallVector<GPUParallelDimAttr> parDims;
  auto parseParDim = [&]() -> ParseResult {
    GPUParallelDimAttr dim;
    if (parseProcessorValue(parser, dim))
      return failure();
    parDims.push_back(dim);
    return success();
  };
  if (parser.parseCommaSeparatedList(delimiter, parseParDim,
                                     "list of OpenACC GPU parallel dimensions"))
    return {};
  return GPUParallelDimsAttr::get(parser.getContext(), parDims);
}

void GPUParallelDimsAttr::print(AsmPrinter &printer) const {
  printer << "[";
  llvm::interleaveComma(getArray(), printer,
                        [&printer](const GPUParallelDimAttr &p) {
                          printProcessorValue(printer, p);
                        });
  printer << "]";
}
