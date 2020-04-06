//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

using namespace mlir;

namespace {


struct GPUShuffleOpLowering : public ConvertToLLVMPattern {
  explicit GPUShuffleOpLowering(LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(gpu::ShuffleOp::getOperationName(),
                             lowering_.getDialect()->getContext(), lowering_) {}

  /// Lowers a shuffle to the corresponding NVVM op.
  ///
  /// Convert the `width` argument into an activeMask (a bitmask which specifies
  /// which threads participate in the shuffle) and a maskAndClamp (specifying
  /// the highest lane which participates in the shuffle).
  ///
  ///     %one = llvm.constant(1 : i32) : !llvm.i32
  ///     %shl = llvm.shl %one, %width : !llvm.i32
  ///     %active_mask = llvm.sub %shl, %one : !llvm.i32
  ///     %mask_and_clamp = llvm.sub %width, %one : !llvm.i32
  ///     %shfl = nvvm.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : !llvm<"{ float, i1 }">
  ///     %shfl_value = llvm.extractvalue %shfl[0 : index] :
  ///         !llvm<"{ float, i1 }">
  ///     %shfl_pred = llvm.extractvalue %shfl[1 : index] :
  ///         !llvm<"{ float, i1 }">
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    gpu::ShuffleOpOperandAdaptor adaptor(operands);

    auto dialect = typeConverter.getDialect();
    auto valueTy = adaptor.value().getType().cast<LLVM::LLVMType>();
    auto int32Type = LLVM::LLVMType::getInt32Ty(dialect);
    auto predTy = LLVM::LLVMType::getInt1Ty(dialect);
    auto resultTy = LLVM::LLVMType::getStructTy(dialect, {valueTy, predTy});

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(1));
    // Bit mask of active lanes: `(1 << activeWidth) - 1`.
    Value activeMask = rewriter.create<LLVM::SubOp>(
        loc, int32Type,
        rewriter.create<LLVM::ShlOp>(loc, int32Type, one, adaptor.width()),
        one);
    // Clamp lane: `activeWidth - 1`
    Value maskAndClamp =
        rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.width(), one);

    auto returnValueAndIsValidAttr = rewriter.getUnitAttr();
    Value shfl = rewriter.create<NVVM::ShflBflyOp>(
        loc, resultTy, activeMask, adaptor.value(), adaptor.offset(),
        maskAndClamp, returnValueAndIsValidAttr);
    Value shflValue = rewriter.create<LLVM::ExtractValueOp>(
        loc, valueTy, shfl, rewriter.getIndexArrayAttr(0));
    Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(
        loc, predTy, shfl, rewriter.getIndexArrayAttr(1));

    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    return success();
  }
};

struct GPUFuncOpLowering : ConvertToLLVMPattern {
  explicit GPUFuncOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(gpu::GPUFuncOp::getOperationName(),
                             typeConverter.getDialect()->getContext(),
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.empty() && "func op is not expected to have operands");
    auto gpuFuncOp = cast<gpu::GPUFuncOp>(op);
    Location loc = gpuFuncOp.getLoc();

    SmallVector<LLVM::GlobalOp, 3> workgroupBuffers;
    workgroupBuffers.reserve(gpuFuncOp.getNumWorkgroupAttributions());
    for (auto en : llvm::enumerate(gpuFuncOp.getWorkgroupAttributions())) {
      Value attribution = en.value();

      auto type = attribution.getType().dyn_cast<MemRefType>();
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      uint64_t numElements = type.getNumElements();

      auto elementType = typeConverter.convertType(type.getElementType())
                             .cast<LLVM::LLVMType>();
      auto arrayType = LLVM::LLVMType::getArrayTy(elementType, numElements);
      std::string name = std::string(
          llvm::formatv("__wg_{0}_{1}", gpuFuncOp.getName(), en.index()));
      auto globalOp = rewriter.create<LLVM::GlobalOp>(
          gpuFuncOp.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::Internal, name, /*value=*/Attribute(),
          gpu::GPUDialect::getWorkgroupAddressSpace());
      workgroupBuffers.push_back(globalOp);
    }

    // Rewrite the original GPU function to an LLVM function.
    auto funcType = typeConverter.convertType(gpuFuncOp.getType())
                        .cast<LLVM::LLVMType>()
                        .getPointerElementTy();

    // Remap proper input types.
    TypeConverter::SignatureConversion signatureConversion(
        gpuFuncOp.front().getNumArguments());
    typeConverter.convertFunctionSignature(
        gpuFuncOp.getType(), /*isVariadic=*/false, signatureConversion);

    // Create the new function operation. Only copy those attributes that are
    // not specific to function modeling.
    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : gpuFuncOp.getAttrs()) {
      if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
          attr.first.is(impl::getTypeAttrName()) ||
          attr.first.is(gpu::GPUFuncOp::getNumWorkgroupAttributionsAttrName()))
        continue;
      attributes.push_back(attr);
    }
    auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), funcType,
        LLVM::Linkage::External, attributes);

    {
      // Insert operations that correspond to converted workgroup and private
      // memory attributions to the body of the function. This must operate on
      // the original function, before the body region is inlined in the new
      // function to maintain the relation between block arguments and the
      // parent operation that assigns their semantics.
      OpBuilder::InsertionGuard guard(rewriter);

      // Rewrite workgroup memory attributions to addresses of global buffers.
      rewriter.setInsertionPointToStart(&gpuFuncOp.front());
      unsigned numProperArguments = gpuFuncOp.getNumArguments();
      auto i32Type = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

      Value zero = nullptr;
      if (!workgroupBuffers.empty())
        zero = rewriter.create<LLVM::ConstantOp>(loc, i32Type,
                                                 rewriter.getI32IntegerAttr(0));
      for (auto en : llvm::enumerate(workgroupBuffers)) {
        LLVM::GlobalOp global = en.value();
        Value address = rewriter.create<LLVM::AddressOfOp>(loc, global);
        auto elementType = global.getType().getArrayElementType();
        Value memory = rewriter.create<LLVM::GEPOp>(
            loc, elementType.getPointerTo(global.addr_space().getZExtValue()),
            address, ArrayRef<Value>{zero, zero});

        // Build a memref descriptor pointing to the buffer to plug with the
        // existing memref infrastructure. This may use more registers than
        // otherwise necessary given that memref sizes are fixed, but we can try
        // and canonicalize that away later.
        Value attribution = gpuFuncOp.getWorkgroupAttributions()[en.index()];
        auto type = attribution.getType().cast<MemRefType>();
        auto descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, typeConverter, type, memory);
        signatureConversion.remapInput(numProperArguments + en.index(), descr);
      }

      // Rewrite private memory attributions to alloca'ed buffers.
      unsigned numWorkgroupAttributions =
          gpuFuncOp.getNumWorkgroupAttributions();
      auto int64Ty = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());
      for (auto en : llvm::enumerate(gpuFuncOp.getPrivateAttributions())) {
        Value attribution = en.value();
        auto type = attribution.getType().cast<MemRefType>();
        assert(type && type.hasStaticShape() &&
               "unexpected type in attribution");

        // Explicitly drop memory space when lowering private memory
        // attributions since NVVM models it as `alloca`s in the default
        // memory space and does not support `alloca`s with addrspace(5).
        auto ptrType = typeConverter.convertType(type.getElementType())
                           .cast<LLVM::LLVMType>()
                           .getPointerTo();
        Value numElements = rewriter.create<LLVM::ConstantOp>(
            gpuFuncOp.getLoc(), int64Ty,
            rewriter.getI64IntegerAttr(type.getNumElements()));
        Value allocated = rewriter.create<LLVM::AllocaOp>(
            gpuFuncOp.getLoc(), ptrType, numElements, /*alignment=*/0);
        auto descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, typeConverter, type, allocated);
        signatureConversion.remapInput(
            numProperArguments + numWorkgroupAttributions + en.index(), descr);
      }
    }

    // Move the region to the new function, update the entry block signature.
    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), llvmFuncOp.getBody(),
                                llvmFuncOp.end());
    rewriter.applySignatureConversion(&llvmFuncOp.getBody(),
                                      signatureConversion);

    rewriter.eraseOp(gpuFuncOp);
    return success();
  }
};

struct GPUReturnOpLowering : public ConvertToLLVMPattern {
  GPUReturnOpLowering(LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(gpu::ReturnOp::getOperationName(),
                             typeConverter.getDialect()->getContext(),
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return success();
  }
};

/// Import the GPU Ops to NVVM Patterns.
#include "GPUToNVVM.cpp.inc"

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
class LowerGpuOpsToNVVMOpsPass
    : public OperationPass<LowerGpuOpsToNVVMOpsPass, gpu::GPUModuleOp> {
public:
/// Include the generated pass utilities.
#define GEN_PASS_ConvertGpuOpsToNVVMOps
#include "mlir/Conversion/Passes.h.inc"

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    /// MemRef conversion for GPU to NVVM lowering. The GPU dialect uses memory
    /// space 5 for private memory attributions, but NVVM represents private
    /// memory allocations as local `alloca`s in the default address space. This
    /// converter drops the private memory space to support the use case above.
    LLVMTypeConverter converter(m.getContext());
    converter.addConversion([&](MemRefType type) -> Optional<Type> {
      if (type.getMemorySpace() != gpu::GPUDialect::getPrivateAddressSpace())
        return llvm::None;
      return converter.convertType(MemRefType::Builder(type).setMemorySpace(0));
    });

    OwningRewritePatternList patterns;

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    populateGpuRewritePatterns(m.getContext(), patterns);
    applyPatternsGreedily(m, patterns);
    patterns.clear();

    populateStdToLLVMConversionPatterns(converter, patterns);
    populateGpuToNVVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<gpu::GPUDialect>();
    target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                        LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op>();
    target.addIllegalOp<FuncOp>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    // TODO(csigg): Remove once we support replacing non-root ops.
    target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
    if (failed(applyPartialConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};

} // anonymous namespace

void mlir::populateGpuToNVVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateWithGenerated(converter.getDialect()->getContext(), &patterns);
  patterns
      .insert<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                          NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
              GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                          NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
              GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                          NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
              GPUIndexIntrinsicOpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                          NVVM::GridDimYOp, NVVM::GridDimZOp>,
              GPUShuffleOpLowering, GPUFuncOpLowering, GPUReturnOpLowering>(
          converter);
  patterns.insert<OpToFuncCallLowering<AbsFOp>>(converter, "__nv_fabsf",
                                                "__nv_fabs");
  patterns.insert<OpToFuncCallLowering<CeilFOp>>(converter, "__nv_ceilf",
                                                 "__nv_ceil");
  patterns.insert<OpToFuncCallLowering<CosOp>>(converter, "__nv_cosf",
                                               "__nv_cos");
  patterns.insert<OpToFuncCallLowering<ExpOp>>(converter, "__nv_expf",
                                               "__nv_exp");
  patterns.insert<OpToFuncCallLowering<LogOp>>(converter, "__nv_logf",
                                               "__nv_log");
  patterns.insert<OpToFuncCallLowering<Log10Op>>(converter, "__nv_log10f",
                                                 "__nv_log10");
  patterns.insert<OpToFuncCallLowering<Log2Op>>(converter, "__nv_log2f",
                                                "__nv_log2");
  patterns.insert<OpToFuncCallLowering<TanhOp>>(converter, "__nv_tanhf",
                                                "__nv_tanh");
}

std::unique_ptr<OpPassBase<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToNVVMOpsPass() {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>();
}
