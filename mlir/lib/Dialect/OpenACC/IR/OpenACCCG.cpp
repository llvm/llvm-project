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
