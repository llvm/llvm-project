//===- GPUToLLVMSPV.cpp - Convert GPU operations to LLVM dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOLLVMSPVOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static LLVM::LLVMFuncOp lookupOrCreateSPIRVFn(Operation *symbolTable,
                                              StringRef name,
                                              ArrayRef<Type> paramTypes,
                                              Type resultType) {
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTable, name));
  if (!func) {
    OpBuilder b(symbolTable->getRegion(0));
    func = b.create<LLVM::LLVMFuncOp>(
        symbolTable->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes));
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  }
  return func;
}

static LLVM::CallOp createSPIRVBuiltinCall(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           LLVM::LLVMFuncOp func,
                                           ValueRange args) {
  auto call = rewriter.create<LLVM::CallOp>(loc, func, args);
  call.setCConv(func.getCConv());
  return call;
}

namespace {
//===----------------------------------------------------------------------===//
// Barriers
//===----------------------------------------------------------------------===//

/// Replace `gpu.barrier` with an `llvm.call` to `barrier` with
/// `CLK_LOCAL_MEM_FENCE` argument, indicating work-group memory scope:
/// ```
/// // gpu.barrier
/// %c1 = llvm.mlir.constant(1: i32) : i32
/// llvm.call spir_funccc @_Z7barrierj(%c1) : (i32) -> ()
/// ```
struct GPUBarrierConversion final : ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    constexpr StringLiteral funcName = "_Z7barrierj";

    Operation *moduleOp = op->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "Expecting module");
    Type flagTy = rewriter.getI32Type();
    Type voidTy = rewriter.getType<LLVM::LLVMVoidType>();
    LLVM::LLVMFuncOp func =
        lookupOrCreateSPIRVFn(moduleOp, funcName, flagTy, voidTy);

    // Value used by SPIR-V backend to represent `CLK_LOCAL_MEM_FENCE`.
    // See `llvm/lib/Target/SPIRV/SPIRVBuiltins.td`.
    constexpr int64_t localMemFenceFlag = 1;
    Location loc = op->getLoc();
    Value flag =
        rewriter.create<LLVM::ConstantOp>(loc, flagTy, localMemFenceFlag);
    rewriter.replaceOp(op, createSPIRVBuiltinCall(loc, rewriter, func, flag));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SPIR-V Builtins
//===----------------------------------------------------------------------===//

/// Replace `gpu.*` with an `llvm.call` to the corresponding SPIR-V builtin with
/// a constant argument for the `dimension` attribute. Return type will depend
/// on index width option:
/// ```
/// // %thread_id_y = gpu.thread_id y
/// %c1 = llvm.mlir.constant(1: i32) : i32
/// %0 = llvm.call spir_funccc @_Z12get_local_idj(%c1) : (i32) -> i64
/// ```
struct LaunchConfigConversion : ConvertToLLVMPattern {
  LaunchConfigConversion(StringRef funcName, StringRef rootOpName,
                         MLIRContext *context,
                         const LLVMTypeConverter &typeConverter,
                         PatternBenefit benefit)
      : ConvertToLLVMPattern(rootOpName, context, typeConverter, benefit),
        funcName(funcName) {}

  virtual gpu::Dimension getDimension(Operation *op) const = 0;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *moduleOp = op->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "Expecting module");
    Type dimTy = rewriter.getI32Type();
    Type indexTy = getTypeConverter()->getIndexType();
    LLVM::LLVMFuncOp func =
        lookupOrCreateSPIRVFn(moduleOp, funcName, dimTy, indexTy);

    Location loc = op->getLoc();
    gpu::Dimension dim = getDimension(op);
    Value dimVal = rewriter.create<LLVM::ConstantOp>(loc, dimTy,
                                                     static_cast<int64_t>(dim));
    rewriter.replaceOp(op, createSPIRVBuiltinCall(loc, rewriter, func, dimVal));
    return success();
  }

  StringRef funcName;
};

template <typename SourceOp>
struct LaunchConfigOpConversion final : LaunchConfigConversion {
  static StringRef getFuncName();

  explicit LaunchConfigOpConversion(const LLVMTypeConverter &typeConverter,
                                    PatternBenefit benefit = 1)
      : LaunchConfigConversion(getFuncName(), SourceOp::getOperationName(),
                               &typeConverter.getContext(), typeConverter,
                               benefit) {}

  gpu::Dimension getDimension(Operation *op) const final {
    return cast<SourceOp>(op).getDimension();
  }
};

template <>
StringRef LaunchConfigOpConversion<gpu::BlockIdOp>::getFuncName() {
  return "_Z12get_group_idj";
}

template <>
StringRef LaunchConfigOpConversion<gpu::GridDimOp>::getFuncName() {
  return "_Z14get_num_groupsj";
}

template <>
StringRef LaunchConfigOpConversion<gpu::BlockDimOp>::getFuncName() {
  return "_Z14get_local_sizej";
}

template <>
StringRef LaunchConfigOpConversion<gpu::ThreadIdOp>::getFuncName() {
  return "_Z12get_local_idj";
}

template <>
StringRef LaunchConfigOpConversion<gpu::GlobalIdOp>::getFuncName() {
  return "_Z13get_global_idj";
}

//===----------------------------------------------------------------------===//
// Shuffles
//===----------------------------------------------------------------------===//

/// Replace `gpu.shuffle` with an `llvm.call` to the corresponding SPIR-V
/// builtin for `shuffleResult`, keeping `value` and `offset` arguments, and a
/// `true` constant for the `valid` result type. Conversion will only take place
/// if `width` is constant and equal to the `subgroup` pass option:
/// ```
/// // %0 = gpu.shuffle idx %value, %offset, %width : f64
/// %0 = llvm.call spir_funccc @_Z17sub_group_shuffledj(%value, %offset)
///     : (f64, i32) -> f64
/// ```
struct GPUShuffleConversion final : ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static StringRef getBaseName(gpu::ShuffleMode mode) {
    switch (mode) {
    case gpu::ShuffleMode::IDX:
      return "sub_group_shuffle";
    case gpu::ShuffleMode::XOR:
      return "sub_group_shuffle_xor";
    case gpu::ShuffleMode::UP:
      return "sub_group_shuffle_up";
    case gpu::ShuffleMode::DOWN:
      return "sub_group_shuffle_down";
    }
    llvm_unreachable("Unhandled shuffle mode");
  }

  static StringRef getTypeMangling(Type type) {
    return TypeSwitch<Type, StringRef>(type)
        .Case<Float32Type>([](auto) { return "fj"; })
        .Case<Float64Type>([](auto) { return "dj"; })
        .Case<IntegerType>([](auto intTy) {
          switch (intTy.getWidth()) {
          case 32:
            return "ij";
          case 64:
            return "lj";
          }
          llvm_unreachable("Invalid integer width");
        });
  }

  static std::string getFuncName(gpu::ShuffleOp op) {
    StringRef baseName = getBaseName(op.getMode());
    StringRef typeMangling = getTypeMangling(op.getType(0));
    return llvm::formatv("_Z{0}{1}{2}", baseName.size(), baseName,
                         typeMangling);
  }

  /// Get the subgroup size from the target or return a default.
  static int getSubgroupSize(Operation *op) {
    return spirv::lookupTargetEnvOrDefault(op)
        .getResourceLimits()
        .getSubgroupSize();
  }

  static bool hasValidWidth(gpu::ShuffleOp op) {
    llvm::APInt val;
    Value width = op.getWidth();
    return matchPattern(width, m_ConstantInt(&val)) &&
           val == getSubgroupSize(op);
  }

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!hasValidWidth(op))
      return rewriter.notifyMatchFailure(
          op, "shuffle width and subgroup size mismatch");

    std::string funcName = getFuncName(op);

    Operation *moduleOp = op->getParentWithTrait<OpTrait::SymbolTable>();
    assert(moduleOp && "Expecting module");
    Type valueType = adaptor.getValue().getType();
    Type offsetType = adaptor.getOffset().getType();
    Type resultType = valueType;
    LLVM::LLVMFuncOp func = lookupOrCreateSPIRVFn(
        moduleOp, funcName, {valueType, offsetType}, resultType);

    Location loc = op->getLoc();
    std::array<Value, 2> args{adaptor.getValue(), adaptor.getOffset()};
    Value result =
        createSPIRVBuiltinCall(loc, rewriter, func, args).getResult();
    Value trueVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(), true);
    rewriter.replaceOp(op, {result, trueVal});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GPU To LLVM-SPV Pass.
//===----------------------------------------------------------------------===//

struct GPUToLLVMSPVConversionPass final
    : impl::ConvertGpuOpsToLLVMSPVOpsBase<GPUToLLVMSPVConversionPass> {
  using Base::Base;

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    LowerToLLVMOptions options(context);
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(context, options);
    LLVMConversionTarget target(*context);

    target.addIllegalOp<gpu::BarrierOp, gpu::BlockDimOp, gpu::BlockIdOp,
                        gpu::GlobalIdOp, gpu::GridDimOp, gpu::ShuffleOp,
                        gpu::ThreadIdOp>();

    populateGpuToLLVMSPVConversionPatterns(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// GPU To LLVM-SPV Patterns.
//===----------------------------------------------------------------------===//

namespace mlir {
void populateGpuToLLVMSPVConversionPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns.add<GPUBarrierConversion, GPUShuffleConversion,
               LaunchConfigOpConversion<gpu::BlockIdOp>,
               LaunchConfigOpConversion<gpu::GridDimOp>,
               LaunchConfigOpConversion<gpu::BlockDimOp>,
               LaunchConfigOpConversion<gpu::ThreadIdOp>,
               LaunchConfigOpConversion<gpu::GlobalIdOp>>(typeConverter);
}
} // namespace mlir
