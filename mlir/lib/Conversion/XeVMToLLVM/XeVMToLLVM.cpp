//===-- XeVMToLLVM.cpp - XeVM to LLVM dialect conversion --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/XeVMToLLVM/XeVMToLLVM.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTXEVMTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace xevm;

namespace {

struct LLVMFuncAttributeOptions {
  bool isConvergent = false;
  bool isNoUnwind = false;
  bool isWillReturn = false;
  LLVM::MemoryEffectsAttr memEffectsAttr{};
};
static constexpr LLVMFuncAttributeOptions noUnwindAttrs = {
    false, true, false, {}};
static constexpr LLVMFuncAttributeOptions noUnwindWillReturnAttrs = {
    false, true, true, {}};
static constexpr LLVMFuncAttributeOptions convergentNoUnwindWillReturnAttrs = {
    true, true, true, {}};

std::string getTypeMangling(Type ty, bool isUnsigned = false) {
  return TypeSwitch<Type, std::string>(ty)
      .Case([isUnsigned](VectorType ty) -> std::string {
        return "Dv" + std::to_string(ty.getNumElements()) + "_" +
               getTypeMangling(ty.getElementType(), isUnsigned);
      })
      .Case([](Float16Type) -> std::string { return "Dh"; })
      .Case([](Float32Type) -> std::string { return "f"; })
      .Case([](Float64Type) -> std::string { return "d"; })
      .Case([isUnsigned](IntegerType ty) -> std::string {
        switch (ty.getWidth()) {
        case 8:
          return isUnsigned ? "h" : "c";
        case 16:
          return isUnsigned ? "t" : "s";
        case 32:
          return isUnsigned ? "j" : "i";
        case 64:
          return isUnsigned ? "m" : "l";
        default:
          llvm_unreachable("unhandled integer type");
        }
      })
      .DefaultUnreachable("unhandled type for mangling");
}

std::string mangle(StringRef baseName, ArrayRef<Type> types,
                   ArrayRef<bool> isUnsigned = {}) {
  assert((isUnsigned.empty() || isUnsigned.size() == types.size()) &&
         "Signedness info doesn't match");
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::SmallDenseMap<Type, unsigned> substitutions;
  os << "_Z" << baseName.size() << baseName;
  for (auto [idx, type] : llvm::enumerate(types)) {
    auto it = substitutions.find(type);
    if (it != substitutions.end()) {
      os << "S";
      // First substitution is `S_`, second is `S0_`, and so on.
      if (unsigned firstIdx = it->getSecond(); firstIdx > 0)
        os << firstIdx - 1;
      os << "_";
    } else {
      if (!type.isIntOrFloat())
        substitutions[type] = substitutions.size();
      os << getTypeMangling(type, isUnsigned.empty() ? false : isUnsigned[idx]);
    }
  }
  return os.str();
}

static int32_t getL1CacheControl(LoadCacheControl cc) {
  int32_t control = 0;
  switch (cc) {
  case LoadCacheControl::L1UC_L2UC_L3UC:
  case LoadCacheControl::L1UC_L2UC_L3C:
  case LoadCacheControl::L1UC_L2C_L3UC:
  case LoadCacheControl::L1UC_L2C_L3C:
    control = 1;
    break;
  case LoadCacheControl::L1C_L2UC_L3UC:
  case LoadCacheControl::L1C_L2UC_L3C:
  case LoadCacheControl::L1C_L2C_L3UC:
  case LoadCacheControl::L1C_L2C_L3C:
    control = 2;
    break;
  case LoadCacheControl::L1S_L2UC_L3UC:
  case LoadCacheControl::L1S_L2UC_L3C:
  case LoadCacheControl::L1S_L2C_L3UC:
  case LoadCacheControl::L1S_L2C_L3C:
    control = 3;
    break;
  case LoadCacheControl::INVALIDATE_READ:
    control = 4;
    break;
  }
  return control;
}

static int32_t getL1CacheControl(StoreCacheControl cc) {
  int32_t control = 0;
  switch (cc) {
  case StoreCacheControl::L1UC_L2UC_L3UC:
  case StoreCacheControl::L1UC_L2UC_L3WB:
  case StoreCacheControl::L1UC_L2WB_L3UC:
  case StoreCacheControl::L1UC_L2WB_L3WB:
    control = 1;
    break;
  case StoreCacheControl::L1WT_L2UC_L3UC:
  case StoreCacheControl::L1WT_L2UC_L3WB:
  case StoreCacheControl::L1WT_L2WB_L3UC:
  case StoreCacheControl::L1WT_L2WB_L3WB:
    control = 2;
    break;
  case StoreCacheControl::L1S_L2UC_L3UC:
  case StoreCacheControl::L1S_L2UC_L3WB:
  case StoreCacheControl::L1S_L2WB_L3UC:
  case StoreCacheControl::L1S_L2WB_L3WB:
    control = 3;
    break;
  case StoreCacheControl::L1WB_L2UC_L3UC:
  case StoreCacheControl::L1WB_L2WB_L3UC:
  case StoreCacheControl::L1WB_L2UC_L3WB:
    control = 4;
    break;
  }
  return control;
}

static int32_t getL3CacheControl(LoadCacheControl cc) {
  int32_t control = 0;
  switch (cc) {
  case LoadCacheControl::L1UC_L2UC_L3UC:
  case LoadCacheControl::L1UC_L2C_L3UC:
  case LoadCacheControl::L1C_L2UC_L3UC:
  case LoadCacheControl::L1C_L2C_L3UC:
  case LoadCacheControl::L1S_L2UC_L3UC:
  case LoadCacheControl::L1S_L2C_L3UC:
    control = 1;
    break;
  case LoadCacheControl::L1UC_L2UC_L3C:
  case LoadCacheControl::L1UC_L2C_L3C:
  case LoadCacheControl::L1C_L2UC_L3C:
  case LoadCacheControl::L1C_L2C_L3C:
  case LoadCacheControl::L1S_L2UC_L3C:
  case LoadCacheControl::L1S_L2C_L3C:
    control = 2;
    break;
  case LoadCacheControl::INVALIDATE_READ:
    control = 4;
    break;
  }
  return control;
}

static int32_t getL3CacheControl(StoreCacheControl cc) {
  int32_t control = 0;
  switch (cc) {
  case StoreCacheControl::L1UC_L2UC_L3UC:
  case StoreCacheControl::L1UC_L2WB_L3UC:
  case StoreCacheControl::L1WT_L2UC_L3UC:
  case StoreCacheControl::L1WT_L2WB_L3UC:
  case StoreCacheControl::L1S_L2UC_L3UC:
  case StoreCacheControl::L1S_L2WB_L3UC:
  case StoreCacheControl::L1WB_L2UC_L3UC:
  case StoreCacheControl::L1WB_L2WB_L3UC:
    control = 1;
    break;
  case StoreCacheControl::L1UC_L2UC_L3WB:
  case StoreCacheControl::L1UC_L2WB_L3WB:
  case StoreCacheControl::L1WT_L2UC_L3WB:
  case StoreCacheControl::L1WT_L2WB_L3WB:
  case StoreCacheControl::L1S_L2UC_L3WB:
  case StoreCacheControl::L1S_L2WB_L3WB:
  case StoreCacheControl::L1WB_L2UC_L3WB:
    control = 2;
    break;
  }
  return control;
}

static std::optional<LoadCacheControl> getCacheControl(PrefetchOp op) {
  return op.getCacheControl();
}

static std::optional<LoadCacheControl> getCacheControl(BlockLoad2dOp op) {
  return op.getCacheControl();
}

static std::optional<LoadCacheControl> getCacheControl(BlockLoadOp op) {
  return op.getCacheControl();
}

static std::optional<LoadCacheControl> getCacheControl(BlockPrefetch2dOp op) {
  return op.getCacheControl();
}

static std::optional<StoreCacheControl> getCacheControl(BlockStore2dOp op) {
  return op.getCacheControl();
}

static std::optional<StoreCacheControl> getCacheControl(BlockStoreOp op) {
  return op.getCacheControl();
}

static std::optional<LoadCacheControl> getCacheControl(LLVM::LoadOp op) {
  if (op->hasAttr("cache_control")) {
    auto attr = op->getAttrOfType<xevm::LoadCacheControlAttr>("cache_control");
    if (!attr)
      return std::nullopt;
    return std::optional<LoadCacheControl>(attr.getValue());
  }
  return std::nullopt;
}

static std::optional<StoreCacheControl> getCacheControl(LLVM::StoreOp op) {
  if (op->hasAttr("cache_control")) {
    auto attr = op->getAttrOfType<xevm::StoreCacheControlAttr>("cache_control");
    if (!attr)
      return std::nullopt;
    return std::optional<StoreCacheControl>(attr.getValue());
  }
  return std::nullopt;
}

template <typename OpType>
int32_t getL1CacheControl(OpType op) {
  return getL1CacheControl(*getCacheControl(op));
}

template <typename OpType>
int32_t getL3CacheControl(OpType op) {
  return getL3CacheControl(*getCacheControl(op));
}

template <typename OpType>
static std::optional<ArrayAttr>
getCacheControlMetadata(ConversionPatternRewriter &rewriter, OpType op) {
  if (!getCacheControl(op))
    return {};
  constexpr int32_t decorationCacheControlArity{4};
  constexpr int32_t loadCacheControlKey{6442};
  constexpr int32_t storeCacheControlKey{6443};
  constexpr bool isLoad = std::is_same_v<OpType, BlockLoad2dOp> ||
                          std::is_same_v<OpType, BlockPrefetch2dOp> ||
                          std::is_same_v<OpType, LLVM::LoadOp> ||
                          std::is_same_v<OpType, BlockLoadOp> ||
                          std::is_same_v<OpType, PrefetchOp>;
  const int32_t controlKey{isLoad ? loadCacheControlKey : storeCacheControlKey};
  SmallVector<int32_t, decorationCacheControlArity> decorationsL1{
      controlKey, 0, getL1CacheControl<OpType>(op), 0};
  SmallVector<int32_t, decorationCacheControlArity> decorationsL3{
      controlKey, 1, getL3CacheControl<OpType>(op), 0};
  auto arrayAttrL1 = rewriter.getI32ArrayAttr(decorationsL1);
  auto arrayAttrL3 = rewriter.getI32ArrayAttr(decorationsL3);

  SmallVector<Attribute, 2> combinedAttrs = {arrayAttrL1, arrayAttrL3};
  return rewriter.getArrayAttr(combinedAttrs);
}

static LLVM::CallOp createDeviceFunctionCall(
    ConversionPatternRewriter &rewriter, StringRef funcName, Type retType,
    ArrayRef<Type> argTypes, ArrayRef<Value> args,
    mlir::ArrayRef<std::pair<unsigned, mlir::StringRef>> paramAttrs,
    LLVMFuncAttributeOptions funcAttributeOptions, Operation *op) {
  auto moduleOp = op->getParentWithTrait<OpTrait::SymbolTable>();
  assert(moduleOp && "Expecting module");
  Location loc = op->getLoc();

  auto funcOpRes =
      LLVM::lookupOrCreateFn(rewriter, moduleOp, funcName, argTypes, retType);
  assert(!failed(funcOpRes));
  LLVM::LLVMFuncOp funcOp = funcOpRes.value();
  funcOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  funcOp.setConvergent(funcAttributeOptions.isConvergent);
  funcOp.setNoUnwind(funcAttributeOptions.isNoUnwind);
  funcOp.setWillReturn(funcAttributeOptions.isWillReturn);

  if (funcAttributeOptions.memEffectsAttr)
    funcOp.setMemoryEffectsAttr(funcAttributeOptions.memEffectsAttr);

  for (auto [idx, attrName] : paramAttrs)
    funcOp.setArgAttr(idx, attrName, rewriter.getUnitAttr());

  auto callOp = LLVM::CallOp::create(rewriter, loc, funcOp, args);
  callOp->setAttrs(funcOp->getAttrs());

  return callOp;
}

class MMAToOCLPattern : public OpConversionPattern<xevm::MMAOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(xevm::MMAOp op, xevm::MMAOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getC()) {
      return rewriter.notifyMatchFailure(op, "OCL requires C operand");
    }
    auto precisionA = op.getTypes().getA();
    auto precisionB = op.getTypes().getB();
    auto precisionC = op.getTypes().getC();
    auto precisionD = op.getTypes().getD();
    if (precisionC != precisionD) {
      return rewriter.notifyMatchFailure(op, "type of C and D need to match");
    }
    if (precisionC != xevm::ElemType::S32 &&
        precisionC != xevm::ElemType::F32 &&
        precisionC != xevm::ElemType::F16 &&
        precisionC != xevm::ElemType::BF16) {
      return rewriter.notifyMatchFailure(
          op, "type of C and D must be S32, F32, F16 or BF16");
    }
    if (precisionA == xevm::ElemType::S32 ||
        precisionA == xevm::ElemType::F32) {
      return rewriter.notifyMatchFailure(op, "type of A cannot be S32 or F32");
    }
    if (precisionB == xevm::ElemType::S32 ||
        precisionB == xevm::ElemType::F32) {
      return rewriter.notifyMatchFailure(op, "type of B cannot be S32 or F32");
    }
    constexpr uint32_t bitWidthPackedA{16};
    constexpr uint32_t bitWidthPackedB{32};
    auto loc = op.getLoc();

    auto castIfNeeded = [&](Value val, Type packedType) -> Value {
      VectorType origTy = cast<VectorType>(val.getType());
      const uint32_t vecBitSize =
          origTy.getNumElements() *
          origTy.getElementType().getIntOrFloatBitWidth();
      VectorType newTy = VectorType::get(
          vecBitSize / packedType.getIntOrFloatBitWidth(), packedType);
      if (origTy != newTy)
        val = LLVM::BitcastOp::create(rewriter, loc, newTy, val);
      return val;
    };

    Value a = op.getA();
    Type packedAType = (op.getTypes().getA() == xevm::ElemType::TF32)
                           ? cast<Type>(rewriter.getF32Type())
                           : rewriter.getIntegerType(bitWidthPackedA);
    a = castIfNeeded(a, packedAType);

    Value b = op.getB();
    Type packedBType = (op.getTypes().getB() == xevm::ElemType::TF32)
                           ? cast<Type>(rewriter.getF32Type())
                           : rewriter.getIntegerType(bitWidthPackedB);
    b = castIfNeeded(b, packedBType);

    Value c = op.getC();
    VectorType cOrigTy = cast<VectorType>(c.getType());
    VectorType resOrigTy = cast<VectorType>(op->getResultTypes()[0]);
    assert(cOrigTy == resOrigTy && "Accumulator and result type mismatch");
    // OCL builtins encode bfloat16 as int16
    VectorType cTy =
        cOrigTy.getElementType().isBF16()
            ? VectorType::get(cOrigTy.getShape(), rewriter.getIntegerType(16))
            : cOrigTy;
    VectorType resTy = cTy;
    if (cOrigTy != cTy)
      c = LLVM::BitcastOp::create(rewriter, loc, cTy, c);

    constexpr int32_t systolicDepth{8};
    std::string fnName =
        llvm::formatv("intel_sub_group_{0}_{1}_matrix_mad_k{2}",
                      stringifyElemType(op.getTypes().getA()).str(),
                      stringifyElemType(op.getTypes().getB()).str(),
                      systolicDepth *
                          getNumOperandsPerDword(op.getTypes().getA()))
            .str();
    SmallVector<Type> argTypes{a.getType(), b.getType(), cTy};
    fnName = mangle(fnName, argTypes);
    SmallVector<Value> args{a, b, c};

    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::NoModRef,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    auto funcAttrs = convergentNoUnwindWillReturnAttrs;
    funcAttrs.memEffectsAttr = memAttr;
    Value result =
        createDeviceFunctionCall(rewriter, fnName, resTy, argTypes, args, {},
                                 funcAttrs, op.getOperation())
            ->getResult(0);

    if (resOrigTy != resTy)
      result = LLVM::BitcastOp::create(rewriter, loc, resOrigTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static unsigned getNumOperandsPerDword(xevm::ElemType pTy) {
    switch (pTy) {
    case xevm::ElemType::TF32:
      return 1;
    case xevm::ElemType::BF16:
    case xevm::ElemType::F16:
      return 2;
    case xevm::ElemType::U8:
    case xevm::ElemType::S8:
      return 4;
    default:
      llvm_unreachable("unsupported xevm::ElemType");
    }
  }
};

class PrefetchToOCLPattern : public OpConversionPattern<PrefetchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrefetchOp op, PrefetchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const std::string fnName{"_Z8prefetchPU3AS1Kcm"};
    Value one =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 1);
    SmallVector<Value> args{op.getPtr(), one};
    SmallVector<Type> argTypes;
    for (auto arg : args)
      argTypes.push_back(arg.getType());
    auto funcAttr = noUnwindAttrs;
    auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
        /*other=*/LLVM::ModRefInfo::NoModRef,
        /*argMem=*/LLVM::ModRefInfo::Ref,
        /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
    funcAttr.memEffectsAttr = memAttr;

    LLVM::CallOp call = createDeviceFunctionCall(
        rewriter, fnName, LLVM::LLVMVoidType::get(rewriter.getContext()),
        argTypes, args, {}, funcAttr, op.getOperation());
    if (std::optional<ArrayAttr> optCacheControls =
            getCacheControlMetadata(rewriter, op))
      call->setAttr(XeVMDialect::getCacheControlsAttrName(), *optCacheControls);
    rewriter.eraseOp(op);
    return success();
  }
};

class MemfenceToOCLPattern : public OpConversionPattern<MemfenceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MemfenceOp op, MemfenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const std::string fnName{"atomic_work_item_fence"};
    int memScope, addrSpace;
    switch (op.getAddrspace()) {
    case xevm::AddrSpace::SHARED:
      addrSpace = 1; // CLK_LOCAL_MEM_FENCE
      break;
    case xevm::AddrSpace::GLOBAL:
      addrSpace = 2; // CLK_GLOBAL_MEM_FENCE
      break;
    default:
      // GENERIC is not supported in OpenCL
      return rewriter.notifyMatchFailure(
          op, "Fence only supports global and shared address spaces.");
    }
    switch (op.getScope()) {
    case xevm::MemScope::WORKGROUP:
      memScope = 1;
      break;
    case xevm::MemScope::DEVICE:
      memScope = 2;
      break;
    default:
      // CLUSTER and SYSTEM are not supported in OpenCL
      return rewriter.notifyMatchFailure(
          op, "Fence only supports workgroup and device memory scopes.");
    }
    Type i32Type = rewriter.getI32Type();
    Value acqRel = LLVM::ConstantOp::create(rewriter, loc, i32Type, 4);
    Value memScopeConst =
        LLVM::ConstantOp::create(rewriter, loc, i32Type, memScope);
    Value addrSpaceConst =
        LLVM::ConstantOp::create(rewriter, loc, i32Type, addrSpace);
    SmallVector<Value> args{addrSpaceConst, acqRel, memScopeConst};
    SmallVector<Type> argTypes{3, i32Type};
    createDeviceFunctionCall(rewriter, mangle(fnName, argTypes),
                             LLVM::LLVMVoidType::get(rewriter.getContext()),
                             argTypes, args, {}, noUnwindAttrs,
                             op.getOperation());
    rewriter.eraseOp(op);
    return success();
  }
};
template <typename OpType>
class LoadStorePrefetchToOCLPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isLoad = std::is_same_v<OpType, BlockLoad2dOp>;
    constexpr bool isPrefetch = std::is_same_v<OpType, BlockPrefetch2dOp>;

    auto loc = op.getLoc();
    VectorType vecType;
    bool packReg = false;
    bool transpose = false;
    if constexpr (isLoad) {
      vecType = op.getRes().getType();
      packReg = op.getPackRegister();
      transpose = op.getTranspose();
    } else if constexpr (!isPrefetch) {
      vecType = op.getStoredVal().getType();
    }

    auto i32Type = rewriter.getI32Type();
    Value byteCoord =
        LLVM::UndefOp::create(rewriter, loc, VectorType::get(2, i32Type));
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i32Type, 0);
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Type, 1);
    byteCoord = LLVM::InsertElementOp::create(
        rewriter, loc, VectorType::get(2, i32Type), byteCoord, op.getX(), zero);
    byteCoord = LLVM::InsertElementOp::create(
        rewriter, loc, VectorType::get(2, i32Type), byteCoord, op.getY(), one);
    SmallVector<Value> args{op.getPtr(), op.getBaseWidth(), op.getBaseHeight(),
                            op.getBasePitch(), byteCoord};
    SmallVector<Type> retTypes;
    Value spvLoadDstPtr;
    std::string funcName{"intel_sub_group_2d_block_"};
    std::string bitWidthId;
    LLVMFuncAttributeOptions funcAttr{noUnwindWillReturnAttrs};
    SmallVector<std::pair<unsigned, StringRef>, 4> paramAttrs;
    if constexpr (isPrefetch) { // Prefetch
      funcName += "prefetch";
      paramAttrs = {std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName())};
      auto memAttr = rewriter.getAttr<LLVM::MemoryEffectsAttr>(
          /*other=*/LLVM::ModRefInfo::NoModRef,
          /*argMem=*/LLVM::ModRefInfo::Ref,
          /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef);
      funcAttr = noUnwindAttrs;
      funcAttr.memEffectsAttr = memAttr;
    } else {
      auto vecElemType = vecType.getElementType();
      auto vecElemBitWidth = vecElemType.getIntOrFloatBitWidth();
      Value numElems = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                                vecType.getNumElements());
      auto dstOrSrcPtr = LLVM::AllocaOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
          vecElemType, numElems);
      args.push_back(dstOrSrcPtr);
      if constexpr (isLoad) { // Load
        funcName += "read";
        bitWidthId = getTypeMangling(vecElemType, /*isUnsigned=*/true);
        if (packReg)
          funcName += "_transform";
        else if (transpose)
          funcName += "_transpose";
        spvLoadDstPtr = dstOrSrcPtr;
        retTypes.push_back(vecType);
        paramAttrs = {
            std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(0, LLVM::LLVMDialect::getReadonlyAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getWriteOnlyAttrName()),
        };
      } else { // Store
        funcName += "write";
        bitWidthId = (vecElemBitWidth == 32)
                         ? "j"
                         : ((vecElemBitWidth == 16) ? "t" : "h");
        LLVM::StoreOp::create(rewriter, loc, op.getStoredVal(), dstOrSrcPtr);
        paramAttrs = {
            std::make_pair(0, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(0, LLVM::LLVMDialect::getWriteOnlyAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getNonNullAttrName()),
            std::make_pair(5, LLVM::LLVMDialect::getReadonlyAttrName()),
        };
      }
    }

    funcName =
        llvm::formatv("{0}_{1}b_{2}r{3}x{4}c", funcName, op.getElemSizeInBits(),
                      op.getTileHeight(), op.getTileWidth(), op.getVBlocks())
            .str();
    std::string prefetchCode("");
    if (!isPrefetch)
      prefetchCode += "P";
    funcName = llvm::formatv("_Z{0}{1}PU3AS1viiiDv2_i{2}{3}", funcName.size(),
                             funcName, prefetchCode, bitWidthId)
                   .str();
    SmallVector<Type> argTypes;
    for (auto arg : args) {
      argTypes.push_back(arg.getType());
    }
    LLVM::CallOp call = createDeviceFunctionCall(
        rewriter, funcName, LLVM::LLVMVoidType::get(rewriter.getContext()),
        argTypes, args, paramAttrs, funcAttr, op.getOperation());
    if (std::optional<ArrayAttr> optCacheControls =
            getCacheControlMetadata(rewriter, op)) {
      call->setAttr(XeVMDialect::getCacheControlsAttrName(), *optCacheControls);
    }
    if constexpr (isLoad)
      rewriter.replaceOp(
          op, LLVM::LoadOp::create(rewriter, loc, vecType, spvLoadDstPtr));
    else
      rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpType>
class BlockLoadStore1DToOCLPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr bool isStore = std::is_same_v<OpType, xevm::BlockStoreOp>;
    // Get OpenCL function name
    // https://registry.khronos.org/OpenCL/extensions/
    //         intel/cl_intel_subgroup_local_block_io.html
    std::string funcName{"intel_sub_group_block_"};
    // Value or Result type can be vector or scalar
    Type valOrResTy;
    if constexpr (isStore) {
      funcName += "write_u";
      valOrResTy = op.getVal().getType();
    } else {
      funcName += "read_u";
      valOrResTy = op.getType();
    }
    // Get element type of the vector/scalar
    VectorType vecTy = dyn_cast<VectorType>(valOrResTy);
    Type elemType = vecTy ? vecTy.getElementType() : valOrResTy;
    funcName += getTypeMangling(elemType);
    if (vecTy)
      funcName += std::to_string(vecTy.getNumElements());
    SmallVector<Type, 2> argTypes{};
    // XeVM BlockLoad/StoreOp always use signless integer types
    // but OpenCL builtins expect unsigned types
    // use unsigned types for mangling
    SmallVector<bool, 2> isUnsigned{};
    // arg0: pointer to the src/dst address
    // arg1 - only if store : vector to store
    // Prepare arguments
    SmallVector<Value, 2> args{};
    args.push_back(op.getPtr());
    argTypes.push_back(op.getPtr().getType());
    isUnsigned.push_back(true);
    Type retType;
    if constexpr (isStore) {
      args.push_back(op.getVal());
      argTypes.push_back(op.getVal().getType());
      isUnsigned.push_back(true);
      retType = LLVM::LLVMVoidType::get(rewriter.getContext());
    } else {
      retType = valOrResTy;
    }
    funcName = std::string("_Z") + std::to_string(funcName.size()) + funcName +
               "PU3AS" +
               std::to_string(op.getPtr().getType().getAddressSpace());
    funcName += getTypeMangling(elemType, /*isUnsigned=*/true);
    if constexpr (isStore)
      funcName += getTypeMangling(valOrResTy, /*isUnsigned=*/true);
    LLVMFuncAttributeOptions funcAttr{noUnwindWillReturnAttrs};

    LLVM::CallOp call =
        createDeviceFunctionCall(rewriter, funcName, retType, argTypes, args,
                                 {}, funcAttr, op.getOperation());
    if (std::optional<ArrayAttr> optCacheControls =
            getCacheControlMetadata(rewriter, op)) {
      call->setAttr(XeVMDialect::getCacheControlsAttrName(), *optCacheControls);
    }
    if constexpr (isStore)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, call->getResult(0));
    return success();
  }
};

template <typename OpType>
class LLVMLoadStoreToOCLPattern : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr("cache_control"))
      return failure();
    std::optional<ArrayAttr> optCacheControls =
        getCacheControlMetadata(rewriter, op);
    op->setAttr(XeVMDialect::getCacheControlsAttrName(), *optCacheControls);
    op->removeAttr("cache_control");
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertXeVMToLLVMPass
    : public impl::ConvertXeVMToLLVMPassBase<ConvertXeVMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, XeVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    populateXeVMToLLVMConversionPatterns(target, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert XeVM to LLVM.
struct XeVMToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateXeVMToLLVMConversionPatterns(target, patterns);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void ::mlir::populateXeVMToLLVMConversionPatterns(ConversionTarget &target,
                                                  RewritePatternSet &patterns) {
  target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(
      [](Operation *op) { return !op->hasAttr("cache_control"); });
  target.addIllegalDialect<XeVMDialect>();
  patterns.add<LoadStorePrefetchToOCLPattern<BlockLoad2dOp>,
               LoadStorePrefetchToOCLPattern<BlockStore2dOp>,
               LoadStorePrefetchToOCLPattern<BlockPrefetch2dOp>,
               MMAToOCLPattern, MemfenceToOCLPattern, PrefetchToOCLPattern,
               LLVMLoadStoreToOCLPattern<LLVM::LoadOp>,
               LLVMLoadStoreToOCLPattern<LLVM::StoreOp>,
               BlockLoadStore1DToOCLPattern<BlockLoadOp>,
               BlockLoadStore1DToOCLPattern<BlockStoreOp>>(
      patterns.getContext());
}

void ::mlir::registerConvertXeVMToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, XeVMDialect *dialect) {
    dialect->addInterfaces<XeVMToLLVMDialectInterface>();
  });
}
