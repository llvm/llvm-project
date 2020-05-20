//===- ConvertGPUToSPIRV.cpp - Convert GPU ops to SPIR-V dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion patterns from GPU ops to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Module.h"

using namespace mlir;

namespace {

/// Pattern to convert a scf::ForOp within kernel functions into spirv::LoopOp.
class ForOpConversion final : public SPIRVOpLowering<scf::ForOp> {
public:
  using SPIRVOpLowering<scf::ForOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a scf::IfOp within kernel functions into
/// spirv::SelectionOp.
class IfOpConversion final : public SPIRVOpLowering<scf::IfOp> {
public:
  using SPIRVOpLowering<scf::IfOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(scf::IfOp IfOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to erase a scf::YieldOp.
class TerminatorOpConversion final : public SPIRVOpLowering<scf::YieldOp> {
public:
  using SPIRVOpLowering<scf::YieldOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(scf::YieldOp terminatorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(terminatorOp);
    return success();
  }
};

/// Pattern lowering GPU block/thread size/id to loading SPIR-V invocation
/// builtin variables.
template <typename SourceOp, spirv::BuiltIn builtin>
class LaunchConfigConversion : public SPIRVOpLowering<SourceOp> {
public:
  using SPIRVOpLowering<SourceOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// This is separate because in Vulkan workgroup size is exposed to shaders via
/// a constant with WorkgroupSize decoration. So here we cannot generate a
/// builtin variable; instead the information in the `spv.entry_point_abi`
/// attribute on the surrounding FuncOp is used to replace the gpu::BlockDimOp.
class WorkGroupSizeConversion : public SPIRVOpLowering<gpu::BlockDimOp> {
public:
  using SPIRVOpLowering<gpu::BlockDimOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(gpu::BlockDimOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a kernel function in GPU dialect within a spv.module.
class GPUFuncOpConversion final : public SPIRVOpLowering<gpu::GPUFuncOp> {
public:
  using SPIRVOpLowering<gpu::GPUFuncOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;

private:
  SmallVector<int32_t, 3> workGroupSizeAsInt32;
};

/// Pattern to convert a gpu.module to a spv.module.
class GPUModuleConversion final : public SPIRVOpLowering<gpu::GPUModuleOp> {
public:
  using SPIRVOpLowering<gpu::GPUModuleOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp moduleOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a gpu.return into a SPIR-V return.
// TODO: This can go to DRR when GPU return has operands.
class GPUReturnOpConversion final : public SPIRVOpLowering<gpu::ReturnOp> {
public:
  using SPIRVOpLowering<gpu::ReturnOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// scf::ForOp.
//===----------------------------------------------------------------------===//

LogicalResult
ForOpConversion::matchAndRewrite(scf::ForOp forOp, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // scf::ForOp can be lowered to the structured control flow represented by
  // spirv::LoopOp by making the continue block of the spirv::LoopOp the loop
  // latch and the merge block the exit block. The resulting spirv::LoopOp has a
  // single back edge from the continue to header block, and a single exit from
  // header to merge.
  scf::ForOpOperandAdaptor forOperands(operands);
  auto loc = forOp.getLoc();
  auto loopControl = rewriter.getI32IntegerAttr(
      static_cast<uint32_t>(spirv::LoopControl::None));
  auto loopOp = rewriter.create<spirv::LoopOp>(loc, loopControl);
  loopOp.addEntryAndMergeBlock();

  OpBuilder::InsertionGuard guard(rewriter);
  // Create the block for the header.
  auto header = new Block();
  // Insert the header.
  loopOp.body().getBlocks().insert(std::next(loopOp.body().begin(), 1), header);

  // Create the new induction variable to use.
  BlockArgument newIndVar =
      header->addArgument(forOperands.lowerBound().getType());
  Block *body = forOp.getBody();

  // Apply signature conversion to the body of the forOp. It has a single block,
  // with argument which is the induction variable. That has to be replaced with
  // the new induction variable.
  TypeConverter::SignatureConversion signatureConverter(
      body->getNumArguments());
  signatureConverter.remapInput(0, newIndVar);
  body = rewriter.applySignatureConversion(&forOp.getLoopBody(),
                                           signatureConverter);

  // Delete the loop terminator.
  rewriter.eraseOp(body->getTerminator());

  // Move the blocks from the forOp into the loopOp. This is the body of the
  // loopOp.
  rewriter.inlineRegionBefore(forOp.getOperation()->getRegion(0), loopOp.body(),
                              std::next(loopOp.body().begin(), 2));

  // Branch into it from the entry.
  rewriter.setInsertionPointToEnd(&(loopOp.body().front()));
  rewriter.create<spirv::BranchOp>(loc, header, forOperands.lowerBound());

  // Generate the rest of the loop header.
  rewriter.setInsertionPointToEnd(header);
  auto mergeBlock = loopOp.getMergeBlock();
  auto cmpOp = rewriter.create<spirv::SLessThanOp>(
      loc, rewriter.getI1Type(), newIndVar, forOperands.upperBound());
  rewriter.create<spirv::BranchConditionalOp>(
      loc, cmpOp, body, ArrayRef<Value>(), mergeBlock, ArrayRef<Value>());

  // Generate instructions to increment the step of the induction variable and
  // branch to the header.
  Block *continueBlock = loopOp.getContinueBlock();
  rewriter.setInsertionPointToEnd(continueBlock);

  // Add the step to the induction variable and branch to the header.
  Value updatedIndVar = rewriter.create<spirv::IAddOp>(
      loc, newIndVar.getType(), newIndVar, forOperands.step());
  rewriter.create<spirv::BranchOp>(loc, header, updatedIndVar);

  rewriter.eraseOp(forOp);
  return success();
}

//===----------------------------------------------------------------------===//
// scf::IfOp.
//===----------------------------------------------------------------------===//

LogicalResult
IfOpConversion::matchAndRewrite(scf::IfOp ifOp, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
  // When lowering `scf::IfOp` we explicitly create a selection header block
  // before the control flow diverges and a merge block where control flow
  // subsequently converges.
  scf::IfOpOperandAdaptor ifOperands(operands);
  auto loc = ifOp.getLoc();

  // Create `spv.selection` operation, selection header block and merge block.
  auto selectionControl = rewriter.getI32IntegerAttr(
      static_cast<uint32_t>(spirv::SelectionControl::None));
  auto selectionOp = rewriter.create<spirv::SelectionOp>(loc, selectionControl);
  selectionOp.addMergeBlock();
  auto *mergeBlock = selectionOp.getMergeBlock();

  OpBuilder::InsertionGuard guard(rewriter);
  auto *selectionHeaderBlock = new Block();
  selectionOp.body().getBlocks().push_front(selectionHeaderBlock);

  // Inline `then` region before the merge block and branch to it.
  auto &thenRegion = ifOp.thenRegion();
  auto *thenBlock = &thenRegion.front();
  rewriter.setInsertionPointToEnd(&thenRegion.back());
  rewriter.create<spirv::BranchOp>(loc, mergeBlock);
  rewriter.inlineRegionBefore(thenRegion, mergeBlock);

  auto *elseBlock = mergeBlock;
  // If `else` region is not empty, inline that region before the merge block
  // and branch to it.
  if (!ifOp.elseRegion().empty()) {
    auto &elseRegion = ifOp.elseRegion();
    elseBlock = &elseRegion.front();
    rewriter.setInsertionPointToEnd(&elseRegion.back());
    rewriter.create<spirv::BranchOp>(loc, mergeBlock);
    rewriter.inlineRegionBefore(elseRegion, mergeBlock);
  }

  // Create a `spv.BranchConditional` operation for selection header block.
  rewriter.setInsertionPointToEnd(selectionHeaderBlock);
  rewriter.create<spirv::BranchConditionalOp>(loc, ifOperands.condition(),
                                              thenBlock, ArrayRef<Value>(),
                                              elseBlock, ArrayRef<Value>());

  rewriter.eraseOp(ifOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Builtins.
//===----------------------------------------------------------------------===//

static Optional<int32_t> getLaunchConfigIndex(Operation *op) {
  auto dimAttr = op->getAttrOfType<StringAttr>("dimension");
  if (!dimAttr) {
    return {};
  }
  if (dimAttr.getValue() == "x") {
    return 0;
  } else if (dimAttr.getValue() == "y") {
    return 1;
  } else if (dimAttr.getValue() == "z") {
    return 2;
  }
  return {};
}

template <typename SourceOp, spirv::BuiltIn builtin>
LogicalResult LaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto index = getLaunchConfigIndex(op);
  if (!index)
    return failure();

  // SPIR-V invocation builtin variables are a vector of type <3xi32>
  auto spirvBuiltin = spirv::getBuiltinVariableValue(op, builtin, rewriter);
  rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
      op, rewriter.getIntegerType(32), spirvBuiltin,
      rewriter.getI32ArrayAttr({index.getValue()}));
  return success();
}

LogicalResult WorkGroupSizeConversion::matchAndRewrite(
    gpu::BlockDimOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto index = getLaunchConfigIndex(op);
  if (!index)
    return failure();

  auto workGroupSizeAttr = spirv::lookupLocalWorkGroupSize(op);
  auto val = workGroupSizeAttr.getValue<int32_t>(index.getValue());
  auto convertedType = typeConverter.convertType(op.getResult().getType());
  if (!convertedType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
      op, convertedType, IntegerAttr::get(convertedType, val));
  return success();
}

//===----------------------------------------------------------------------===//
// GPUFuncOp
//===----------------------------------------------------------------------===//

// Legalizes a GPU function as an entry SPIR-V function.
static spirv::FuncOp
lowerAsEntryFunction(gpu::GPUFuncOp funcOp, SPIRVTypeConverter &typeConverter,
                     ConversionPatternRewriter &rewriter,
                     spirv::EntryPointABIAttr entryPointInfo,
                     ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo) {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults()) {
    funcOp.emitError("SPIR-V lowering only supports entry functions"
                     "with no return values right now");
    return nullptr;
  }
  if (fnType.getNumInputs() != argABIInfo.size()) {
    funcOp.emitError(
        "lowering as entry functions requires ABI info for all arguments");
    return nullptr;
  }
  // Update the signature to valid SPIR-V types and add the ABI
  // attributes. These will be "materialized" by using the
  // LowerABIAttributesPass.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  {
    for (auto argType : enumerate(funcOp.getType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None));
  for (const auto &namedAttr : funcOp.getAttrs()) {
    if (namedAttr.first == impl::getTypeAttrName() ||
        namedAttr.first == SymbolTable::getSymbolAttrName())
      continue;
    newFuncOp.setAttr(namedAttr.first, namedAttr.second);
  }
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);

  spirv::setABIAttrs(newFuncOp, entryPointInfo, argABIInfo);
  return newFuncOp;
}

/// Populates `argABI` with spv.interface_var_abi attributes for lowering
/// gpu.func to spv.func if no arguments have the attributes set
/// already. Returns failure if any argument has the ABI attribute set already.
static LogicalResult
getDefaultABIAttrs(MLIRContext *context, gpu::GPUFuncOp funcOp,
                   SmallVectorImpl<spirv::InterfaceVarABIAttr> &argABI) {
  for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    if (funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
            argIndex, spirv::getInterfaceVarABIAttrName()))
      return failure();
    // Vulkan's interface variable requirements needs scalars to be wrapped in a
    // struct. The struct held in storage buffer.
    Optional<spirv::StorageClass> sc;
    if (funcOp.getArgument(argIndex).getType().isIntOrIndexOrFloat())
      sc = spirv::StorageClass::StorageBuffer;
    argABI.push_back(spirv::getInterfaceVarABIAttr(0, argIndex, sc, context));
  }
  return success();
}

LogicalResult GPUFuncOpConversion::matchAndRewrite(
    gpu::GPUFuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!gpu::GPUDialect::isKernel(funcOp))
    return failure();

  SmallVector<spirv::InterfaceVarABIAttr, 4> argABI;
  if (failed(getDefaultABIAttrs(rewriter.getContext(), funcOp, argABI))) {
    argABI.clear();
    for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
      // If the ABI is already specified, use it.
      auto abiAttr = funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
          argIndex, spirv::getInterfaceVarABIAttrName());
      if (!abiAttr) {
        funcOp.emitRemark(
            "match failure: missing 'spv.interface_var_abi' attribute at "
            "argument ")
            << argIndex;
        return failure();
      }
      argABI.push_back(abiAttr);
    }
  }

  auto entryPointAttr = spirv::lookupEntryPointABI(funcOp);
  if (!entryPointAttr) {
    funcOp.emitRemark("match failure: missing 'spv.entry_point_abi' attribute");
    return failure();
  }
  spirv::FuncOp newFuncOp = lowerAsEntryFunction(
      funcOp, typeConverter, rewriter, entryPointAttr, argABI);
  if (!newFuncOp)
    return failure();
  newFuncOp.removeAttr(Identifier::get(gpu::GPUDialect::getKernelFuncAttrName(),
                                       rewriter.getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// ModuleOp with gpu.module.
//===----------------------------------------------------------------------===//

LogicalResult GPUModuleConversion::matchAndRewrite(
    gpu::GPUModuleOp moduleOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto spvModule = rewriter.create<spirv::ModuleOp>(
      moduleOp.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::GLSL450);

  // Move the region from the module op into the SPIR-V module.
  Region &spvModuleRegion = spvModule.body();
  rewriter.inlineRegionBefore(moduleOp.body(), spvModuleRegion,
                              spvModuleRegion.begin());
  // The spv.module build method adds a block with a terminator. Remove that
  // block. The terminator of the module op in the remaining block will be
  // legalized later.
  rewriter.eraseBlock(&spvModuleRegion.back());
  rewriter.eraseOp(moduleOp);
  return success();
}

//===----------------------------------------------------------------------===//
// GPU return inside kernel functions to SPIR-V return.
//===----------------------------------------------------------------------===//

LogicalResult GPUReturnOpConversion::matchAndRewrite(
    gpu::ReturnOp returnOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!operands.empty())
    return failure();

  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  return success();
}

//===----------------------------------------------------------------------===//
// GPU To SPIRV Patterns.
//===----------------------------------------------------------------------===//

namespace {
#include "GPUToSPIRV.cpp.inc"
}

void mlir::populateGPUToSPIRVPatterns(MLIRContext *context,
                                      SPIRVTypeConverter &typeConverter,
                                      OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
  patterns.insert<
      ForOpConversion, GPUFuncOpConversion, GPUModuleConversion,
      GPUReturnOpConversion, IfOpConversion,
      LaunchConfigConversion<gpu::BlockIdOp, spirv::BuiltIn::WorkgroupId>,
      LaunchConfigConversion<gpu::GridDimOp, spirv::BuiltIn::NumWorkgroups>,
      LaunchConfigConversion<gpu::ThreadIdOp,
                             spirv::BuiltIn::LocalInvocationId>,
      TerminatorOpConversion, WorkGroupSizeConversion>(context, typeConverter);
}
