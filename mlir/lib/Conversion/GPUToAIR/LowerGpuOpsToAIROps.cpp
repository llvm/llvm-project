//===- LowerGpuOpsToAIROps.cpp - MLIR GPU to AIR lowering passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that lowers GPU modules to the LLVM IR the AIR
// target compiles, in AIR's address spaces (thread 0, device 1, constant 2,
// threadgroup 3), with calls to AIR's functions:
//
//   gpu.thread_id         air.thread_position_in_threadgroup
//   gpu.block_id          air.threadgroup_position_in_grid
//   gpu.block_dim         air.threads_per_threadgroup
//   gpu.grid_dim          air.threadgroups_per_grid
//   gpu.global_id         air.thread_position_in_grid
//   gpu.lane_id           air.thread_index_in_simdgroup
//   gpu.subgroup_id       air.simdgroup_index_in_threadgroup
//   gpu.num_subgroups     air.simdgroups_per_threadgroup
//   gpu.subgroup_size     32
//   gpu.barrier           air.wg.barrier
//   gpu.shuffle           air.simd_shuffle
//   gpu.ballot            air.simd_ballot
//   gpu.subgroup_reduce   air.simd_<op>
//   air.simdgroup_matrix_* air.simdgroup_matrix_8x8_*
//   gpu.func              functions marked "air-kernel"
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAIR/GPUToAIRPass.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOAIROPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

const char *const mlir::kAIRDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
    "f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-"
    "v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-"
    "n8:16:32";

namespace {
// AIR's address spaces.
enum : unsigned {
  kThreadAddressSpace = 0,
  kDeviceAddressSpace = 1,
  kConstantAddressSpace = 2,
  kThreadgroupAddressSpace = 3,
};

/// The discardable attribute that marks kernels during the conversion; they
/// end up with the "air-kernel" function attribute.
constexpr StringLiteral kKernelMarker = "air.kernel";
} // namespace

/// A call of the AIR function `name`, declared in the GPU module as needed.
static Value callAIR(ConversionPatternRewriter &rewriter, Operation *op,
                     StringRef name, Type result, ValueRange args) {
  Operation *module = op->getParentWithTrait<OpTrait::SymbolTable>();
  SmallVector<Type> params(args.getTypes());
  Type resultType = result ? result : LLVM::LLVMVoidType::get(op->getContext());
  FailureOr<LLVM::LLVMFuncOp> fn =
      LLVM::lookupOrCreateFn(rewriter, module, name, params, resultType);
  if (failed(fn))
    return Value();
  auto call = LLVM::CallOp::create(rewriter, op->getLoc(), *fn, args);
  return result ? call.getResult() : Value();
}

/// Extend or truncate an i32 to the index type.
static Value toIndexBitwidth(ConversionPatternRewriter &rewriter, Location loc,
                             Value value, unsigned indexBitwidth) {
  Type indexType = rewriter.getIntegerType(indexBitwidth);
  if (indexBitwidth > 32)
    return LLVM::ZExtOp::create(rewriter, loc, indexType, value);
  if (indexBitwidth < 32)
    return LLVM::TruncOp::create(rewriter, loc, indexType, value);
  return value;
}

/// AIR's name suffix for a float or 32-bit integer type ("f32", "s.i32").
static std::string typeSuffix(Type type, bool isUnsigned = false) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  return (isUnsigned ? "u.i" : "s.i") + std::to_string(type.getIntOrFloatBitWidth());
}

namespace {

/// A system value: a call of the AIR function, of one component (x, y or z)
/// of the three-dimensional ones (XYZ).
template <typename SourceOp, bool XYZ>
struct GPUSystemValueLowering : ConvertOpToLLVMPattern<SourceOp> {
  GPUSystemValueLowering(const LLVMTypeConverter &converter, StringRef function)
      : ConvertOpToLLVMPattern<SourceOp>(converter), function(function) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Value value;
    if constexpr (XYZ) {
      Value xyz = callAIR(rewriter, op, (function + ".v3i32").str(),
                          VectorType::get({3}, i32), {});
      Value dim = LLVM::ConstantOp::create(
          rewriter, loc, i32, static_cast<int32_t>(op.getDimension()));
      value = LLVM::ExtractElementOp::create(rewriter, loc, xyz, dim);
    } else {
      value = callAIR(rewriter, op, function, i32, {});
    }
    rewriter.replaceOp(
        op, toIndexBitwidth(rewriter, loc, value,
                            this->getTypeConverter()->getIndexTypeBitwidth()));
    return success();
  }

  StringRef function;
};

struct GPUSubgroupSizeOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupSizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type indexType = rewriter.getIntegerType(
        getTypeConverter()->getIndexTypeBitwidth());
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, indexType, 32);
    return success();
  }
};

/// gpu.barrier orders device and threadgroup memory across the threadgroup.
struct GPUBarrierOpLowering : ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    if (adaptor.getNamedBarrier())
      return rewriter.notifyMatchFailure(op, "named barrier");
    // mem_flags: device (1) | threadgroup (2), or the memory of `memfence`
    // (none for `memfence []`, which only synchronizes).
    int64_t memFlags = 3;
    if (std::optional<ArrayAttr> spaces = op.getAddressSpaces()) {
      memFlags = 0;
      for (Attribute a : *spaces)
        switch (cast<gpu::AddressSpaceAttr>(a).getValue()) {
        case gpu::AddressSpace::Global:
          memFlags |= 1;
          break;
        case gpu::AddressSpace::Workgroup:
          memFlags |= 2;
          break;
        case gpu::AddressSpace::Private:
          break;
        }
    }
    Value flags = LLVM::ConstantOp::create(rewriter, loc, i32, memFlags);
    gpu::BarrierScope barrierScope = op.getScope();
    if (barrierScope != gpu::BarrierScope::Workgroup &&
        barrierScope != gpu::BarrierScope::Subgroup)
      return rewriter.notifyMatchFailure(op, "barrier scope beyond workgroup");
    // thread_scope_threadgroup (1).
    Value scope = LLVM::ConstantOp::create(rewriter, loc, i32, 1);
    callAIR(rewriter, op,
            barrierScope == gpu::BarrierScope::Subgroup
                ? "air.simdgroup.barrier"
                : "air.wg.barrier",
            nullptr, {flags, scope});
    rewriter.eraseOp(op);
    return success();
  }
};

/// gpu.shuffle: the source lane (lane - offset, lane + offset, lane ^ offset
/// or offset), valid within the thread's segment of `width` lanes and the
/// thread's own lane outside of it, and an index shuffle of each 32-bit part.
struct GPUShuffleOpLowering : ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Value lane = callAIR(rewriter, op, "air.thread_index_in_simdgroup", i32, {});
    Value width = adaptor.getWidth();
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i32, 0);
    Value negWidth = LLVM::SubOp::create(rewriter, loc, i32, zero, width);
    Value add = LLVM::AddOp::create(rewriter, loc, i32, lane, width);
    Value segmentEnd = LLVM::AndOp::create(rewriter, loc, i32, add, negWidth);
    Value srcLane;
    switch (op.getMode()) {
    case gpu::ShuffleMode::UP:
      srcLane = LLVM::SubOp::create(rewriter, loc, i32, lane,
                                    adaptor.getOffset());
      break;
    case gpu::ShuffleMode::DOWN:
      srcLane = LLVM::AddOp::create(rewriter, loc, i32, lane,
                                    adaptor.getOffset());
      break;
    case gpu::ShuffleMode::XOR:
      srcLane = LLVM::XOrOp::create(rewriter, loc, i32, lane,
                                    adaptor.getOffset());
      break;
    case gpu::ShuffleMode::IDX:
      srcLane = adaptor.getOffset();
      break;
    }
    // The source lane must be in the thread's segment of `width` lanes.
    Value segmentStart = LLVM::AndOp::create(rewriter, loc, i32, lane, negWidth);
    Value valid = LLVM::AndOp::create(
        rewriter, loc,
        LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge, srcLane,
                             segmentStart),
        LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt, srcLane,
                             segmentEnd));
    Value selected =
        LLVM::SelectOp::create(rewriter, loc, valid, srcLane, lane);
    Value lane16 =
        LLVM::TruncOp::create(rewriter, loc, rewriter.getI16Type(), selected);

    SmallVector<Value> parts;
    if (failed(LLVM::decomposeValue(rewriter, loc, adaptor.getValue(), i32,
                                    parts)))
      return rewriter.notifyMatchFailure(op, "cannot split value into i32");
    SmallVector<Value> shuffled;
    for (Value part : parts)
      shuffled.push_back(callAIR(rewriter, op, "air.simd_shuffle.u.i32", i32,
                                 {part, lane16}));
    Value result = LLVM::composeValue(rewriter, loc, shuffled,
                                      adaptor.getValue().getType());
    rewriter.replaceOp(op, {result, valid});
    return success();
  }
};

struct GPUBallotOpLowering : ConvertOpToLLVMPattern<gpu::BallotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BallotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto intType = dyn_cast<IntegerType>(op.getType());
    if (!intType || intType.getWidth() < 32)
      return rewriter.notifyMatchFailure(op, "result must be at least 32 bits");
    Type i64 = rewriter.getI64Type();
    Value mask = callAIR(rewriter, op, "air.simd_ballot.i64", i64,
                         {adaptor.getPredicate()});
    if (intType.getWidth() < 64)
      mask = LLVM::TruncOp::create(rewriter, loc, intType, mask);
    rewriter.replaceOp(op, mask);
    return success();
  }
};

/// gpu.subgroup_reduce over the whole SIMD-group.
struct GPUSubgroupReduceOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getClusterSize())
      return rewriter.notifyMatchFailure(op, "clustered reductions");
    Type type = adaptor.getValue().getType();
    bool isFloat = isa<FloatType>(type);
    if (!isFloat && !type.isInteger(32))
      return rewriter.notifyMatchFailure(op, "only i32 and floats");
    StringRef fn;
    bool isUnsigned = false;
    switch (op.getOp()) {
    case gpu::AllReduceOperation::ADD:
      fn = "sum";
      break;
    case gpu::AllReduceOperation::MUL:
      fn = "product";
      break;
    case gpu::AllReduceOperation::MINSI:
    case gpu::AllReduceOperation::MINNUMF:
      fn = "min";
      break;
    case gpu::AllReduceOperation::MAXSI:
    case gpu::AllReduceOperation::MAXNUMF:
      fn = "max";
      break;
    case gpu::AllReduceOperation::MINUI:
      fn = "min", isUnsigned = true;
      break;
    case gpu::AllReduceOperation::MAXUI:
      fn = "max", isUnsigned = true;
      break;
    case gpu::AllReduceOperation::AND:
      fn = "and";
      break;
    case gpu::AllReduceOperation::OR:
      fn = "or";
      break;
    case gpu::AllReduceOperation::XOR:
      fn = "xor";
      break;
    default:
      return rewriter.notifyMatchFailure(op, "unsupported reduction");
    }
    if (isFloat && (fn == "and" || fn == "or" || fn == "xor"))
      return rewriter.notifyMatchFailure(op, "bitwise reduction of floats");
    std::string name =
        ("air.simd_" + fn + "." + typeSuffix(type, isUnsigned)).str();
    rewriter.replaceOp(op, callAIR(rewriter, op, name, type,
                                   {adaptor.getValue()}));
    return success();
  }
};

/// The dimensions (8 x 8), strides and origin (0, 0) of AIR's matrix load
/// and store: the matrix element (r, c) is at c * strides.x + r * strides.y
/// from the pointer.
static SmallVector<Value, 3>
matrixShape(ConversionPatternRewriter &rewriter, Location loc,
            Value descriptor, bool transpose) {
  Type i64 = rewriter.getI64Type();
  auto v2i64 = VectorType::get({2}, i64);
  MemRefDescriptor memref(descriptor);
  Value rowStride = memref.stride(rewriter, loc, 0);
  Value colStride = memref.stride(rewriter, loc, 1);
  auto toI64 = [&](Value v) -> Value {
    if (v.getType() == i64)
      return v;
    return LLVM::ZExtOp::create(rewriter, loc, i64, v);
  };
  Value x = toI64(transpose ? rowStride : colStride);
  Value y = toI64(transpose ? colStride : rowStride);
  Value strides = LLVM::PoisonOp::create(rewriter, loc, v2i64);
  strides = LLVM::InsertElementOp::create(
      rewriter, loc, strides, x, LLVM::ConstantOp::create(rewriter, loc, i64, 0));
  strides = LLVM::InsertElementOp::create(
      rewriter, loc, strides, y, LLVM::ConstantOp::create(rewriter, loc, i64, 1));
  Value dims = LLVM::ConstantOp::create(
      rewriter, loc, v2i64,
      DenseElementsAttr::get(v2i64, ArrayRef<int64_t>{8, 8}));
  Value origin = LLVM::ConstantOp::create(
      rewriter, loc, v2i64,
      DenseElementsAttr::get(v2i64, ArrayRef<int64_t>{0, 0}));
  return {dims, strides, origin};
}

struct MatrixLoadOpLowering : ConvertOpToLLVMPattern<air::MatrixLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MemRefType type = op.getBase().getType();
    Value ptr = getStridedElementPtr(rewriter, loc, type, adaptor.getBase(),
                                     adaptor.getIndices());
    SmallVector<Value, 3> shape =
        matrixShape(rewriter, loc, adaptor.getBase(), op.getTranspose());
    std::string elt = typeSuffix(type.getElementType());
    unsigned as = cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace();
    std::string name = "air.simdgroup_matrix_8x8_load.v64" + elt + ".p" +
                       std::to_string(as) + elt;
    rewriter.replaceOp(op, callAIR(rewriter, op, name, op.getType(),
                                   {ptr, shape[0], shape[1], shape[2]}));
    return success();
  }
};

struct MatrixStoreOpLowering : ConvertOpToLLVMPattern<air::MatrixStoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MemRefType type = op.getBase().getType();
    Value ptr = getStridedElementPtr(rewriter, loc, type, adaptor.getBase(),
                                     adaptor.getIndices());
    SmallVector<Value, 3> shape =
        matrixShape(rewriter, loc, adaptor.getBase(), op.getTranspose());
    std::string elt = typeSuffix(type.getElementType());
    unsigned as = cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace();
    std::string name = "air.simdgroup_matrix_8x8_store.v64" + elt + ".p" +
                       std::to_string(as) + elt;
    callAIR(rewriter, op, name, nullptr,
            {adaptor.getValue(), ptr, shape[0], shape[1], shape[2]});
    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixMultiplyAccumulateOpLowering
    : ConvertOpToLLVMPattern<air::MatrixMultiplyAccumulateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixMultiplyAccumulateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string ab = "v64" + typeSuffix(op.getA().getType().getElementType());
    std::string cd = "v64" + typeSuffix(op.getType().getElementType());
    std::string name = "air.simdgroup_matrix_8x8_multiply_accumulate." + cd +
                       "." + ab + "." + ab + "." + cd;
    rewriter.replaceOp(op, callAIR(rewriter, op, name, op.getType(),
                                   {adaptor.getA(), adaptor.getB(),
                                    adaptor.getC()}));
    return success();
  }
};

struct MatrixFillOpLowering : ConvertOpToLLVMPattern<air::MatrixFillOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixFillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string elt = typeSuffix(op.getValue().getType());
    std::string name =
        "air.simdgroup_matrix_8x8_init_filled.v64" + elt + "." + elt;
    rewriter.replaceOp(op, callAIR(rewriter, op, name, op.getType(),
                                   {adaptor.getValue()}));
    return success();
  }
};

/// AIR has no generic address space: its address space 0 is the thread's.
/// A memref without a memory space is device memory (the GPU dialect's
/// buffers), except a memref.alloca, which is the thread's.
static void assignDefaultMemorySpaces(gpu::GPUModuleOp m) {
  MLIRContext *ctx = m.getContext();
  auto space = [&](gpu::AddressSpace s) {
    return gpu::AddressSpaceAttr::get(ctx, s);
  };
  m.walk([&](memref::AllocaOp op) {
    MemRefType type = op.getType();
    if (!type.getMemorySpace())
      op.getResult().setType(
          MemRefType::get(type.getShape(), type.getElementType(),
                          type.getLayout(), space(gpu::AddressSpace::Private)));
  });
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](BaseMemRefType type) -> std::optional<Type> {
    if (type.getMemorySpace())
      return std::nullopt;
    Attribute device = space(gpu::AddressSpace::Global);
    if (auto ranked = dyn_cast<MemRefType>(type))
      return MemRefType::get(ranked.getShape(), ranked.getElementType(),
                             ranked.getLayout(), device);
    return UnrankedMemRefType::get(type.getElementType(), device);
  });
  replacer.recursivelyReplaceElementsIn(m, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

struct LowerGpuOpsToAIROpsPass final
    : public impl::ConvertGpuOpsToAIROpsBase<LowerGpuOpsToAIROpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    auto llvmDataLayout = m->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getDataLayoutAttrName());
    if (!llvmDataLayout) {
      llvmDataLayout = StringAttr::get(ctx, kAIRDataLayout);
      m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(), llvmDataLayout);
    }

    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.dataLayout = llvm::DataLayout(llvmDataLayout.getValue());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.useBarePtrCallConv = useBarePtrCallConv;

    assignDefaultMemorySpaces(m);

    // In-dialect lowering of gpu.all_reduce and friends first.
    {
      RewritePatternSet patterns(ctx);
      populateGpuRewritePatterns(patterns);
      (void)applyPatternsGreedily(m, std::move(patterns));
    }

    LLVMTypeConverter converter(ctx, options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return kDeviceAddressSpace;
          case gpu::AddressSpace::Workgroup:
            return kThreadgroupAddressSpace;
          case gpu::AddressSpace::Private:
            return kThreadAddressSpace;
          case gpu::AddressSpace::Constant:
            return kConstantAddressSpace;
          }
          llvm_unreachable("unknown address space enum value");
        });

    RewritePatternSet llvmPatterns(ctx);
    LLVMConversionTarget target(getContext());
    for (Dialect *dialect : ctx->getLoadedDialects())
      if (auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect))
        iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                       llvmPatterns);
    // Structured control flow (loops in kernels) becomes branches.
    populateSCFToControlFlowConversionPatterns(llvmPatterns);
    target.addIllegalDialect<scf::SCFDialect>();
    populateGpuToAIRConversionPatterns(converter, llvmPatterns);
    configureGpuToAIRConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      return signalPassFailure();

    // Kernels carry the "air-kernel" function attribute, and the threadgroup
    // size when it is known as an AGX hint, which AIR ignores (the AGX
    // backend budgets registers and barriers by it).
    auto *gpuDialect = ctx->getLoadedDialect<gpu::GPUDialect>();
    m.walk([&](LLVM::LLVMFuncOp f) {
      if (!f->hasAttr(kKernelMarker))
        return;
      f->removeAttr(kKernelMarker);
      SmallVector<Attribute> passthrough;
      if (ArrayAttr old = f.getPassthroughAttr())
        passthrough.append(old.begin(), old.end());
      passthrough.push_back(StringAttr::get(ctx, "air-kernel"));
      auto knownBlockSize = gpuDialect->getKnownBlockSizeAttrHelper();
      if (DenseI32ArrayAttr size = knownBlockSize.getAttr(f)) {
        int64_t threads = 1;
        for (int32_t d : size.asArrayRef())
          threads *= d;
        passthrough.push_back(ArrayAttr::get(
            ctx, {StringAttr::get(ctx, "agx-max-threads-per-threadgroup"),
                  StringAttr::get(ctx, std::to_string(threads))}));
      }
      f.setPassthroughAttr(ArrayAttr::get(ctx, passthrough));
    });
  }
};

} // namespace

void mlir::configureGpuToAIRConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalDialect<air::AIRDialect>();
  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp>();
}

void mlir::populateGpuToAIRConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GPUSystemValueLowering<gpu::ThreadIdOp, true>>(
      converter, "air.thread_position_in_threadgroup");
  patterns.add<GPUSystemValueLowering<gpu::BlockIdOp, true>>(
      converter, "air.threadgroup_position_in_grid");
  patterns.add<GPUSystemValueLowering<gpu::BlockDimOp, true>>(
      converter, "air.threads_per_threadgroup");
  patterns.add<GPUSystemValueLowering<gpu::GridDimOp, true>>(
      converter, "air.threadgroups_per_grid");
  patterns.add<GPUSystemValueLowering<gpu::GlobalIdOp, true>>(
      converter, "air.thread_position_in_grid");
  patterns.add<GPUSystemValueLowering<gpu::LaneIdOp, false>>(
      converter, "air.thread_index_in_simdgroup");
  patterns.add<GPUSystemValueLowering<gpu::SubgroupIdOp, false>>(
      converter, "air.simdgroup_index_in_threadgroup");
  patterns.add<GPUSystemValueLowering<gpu::NumSubgroupsOp, false>>(
      converter, "air.simdgroups_per_threadgroup");
  patterns.add<GPUSubgroupSizeOpLowering, GPUBarrierOpLowering,
               GPUShuffleOpLowering, GPUBallotOpLowering,
               GPUSubgroupReduceOpLowering, MatrixLoadOpLowering,
               MatrixStoreOpLowering, MatrixMultiplyAccumulateOpLowering,
               MatrixFillOpLowering>(converter);
  patterns.add<GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      GPUFuncOpLoweringOptions{
          /*allocaAddrSpace=*/kThreadAddressSpace,
          /*workgroupAddrSpace=*/kThreadgroupAddressSpace,
          StringAttr::get(&converter.getContext(), kKernelMarker),
          /*kernelBlockSizeAttributeName=*/{},
          /*kernelClusterSizeAttributeName=*/{}});
}
