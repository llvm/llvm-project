//===-- MathToXeVM.cpp - conversion from Math to XeVM ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToXeVM/MathToXeVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOXEVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "math-to-xevm"

// GPUCommon/OpToFunctionCallLowering is not used here, as it doesn't handle
// native functions/intrinsics that take vector operands.

/// Convert math ops marked with `fast` (`afn`) to native OpenCL intrinsics.
template <typename Op>
struct ConvertNativeFuncPattern final : public OpConversionPattern<Op> {

  ConvertNativeFuncPattern(MLIRContext *context, StringRef nativeFunc,
                           PatternBenefit benefit = 1)
      : OpConversionPattern<Op>(context, benefit), nativeFunc(nativeFunc) {}

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: OCL doesn't provide native int intrinsics, but check what happens
    // when IGC receives a native_exp on ints anyway
    // TODO: what about vectorization?
    if (!isSPIRVCompatibleFloatOrVec(op.getType()))
      return failure();

    arith::FastMathFlags fastFlags = op.getFastmath();
    if (!((uint32_t)fastFlags & (uint32_t)arith::FastMathFlags::afn))
      return failure();

    // FIXME: Implement handling for vector sizes/dimensions that are not
    // supported by SPIRV
    SmallVector<Type, 1> operandTypes;
    for (auto operand : adaptor.getOperands()) {
      if (!isSPIRVCompatibleFloatOrVec(operand.getType()))
        return failure();
      operandTypes.push_back(operand.getType());
    }
    LLVM::LLVMFuncOp funcOp = appendOrGetFuncOp(op, operandTypes);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp,
                                              adaptor.getOperands());
    return success();
  }

  inline bool isSPIRVCompatibleFloatOrVec(Type type) const {
    if (type.isFloat()) {
      return true;
    } else if (auto vecType = dyn_cast<VectorType>(type)) {
      if (!vecType.getElementType().isFloat())
        return false;
      // SPIRV distinguishes between vectors and matrices: OpenCL native math
      // intrsinics are not compatible with matrices.
      ArrayRef<int64_t> shape = vecType.getShape();
      if (shape.size() != 1)
        return false;
      // SPIRV only allows vectors of size 2, 3, 4, 8, 16.
      if (shape[0] == 2 || shape[0] == 3 || shape[0] == 4 || shape[0] == 8 ||
          shape[0] == 16)
        return true;
    }
    return false;
  }

  LLVM::LLVMFuncOp
  appendOrGetFuncOp(Op &op, const SmallVector<Type, 1> &operandTypes) const {
    // This function assumes op types have already been validated using
    // isSPIRVCompatibleFloatOrVec.
    using LLVM::LLVMFuncOp;

    std::string mangledNativeFunc =
        "_Z" + std::to_string(nativeFunc.size()) + nativeFunc.str();

    auto appendFloatToMangledFunc = [&mangledNativeFunc](Type type) {
      if (type.isF32())
        mangledNativeFunc += "f";
      else if (type.isF16())
        mangledNativeFunc += "Dh";
      else if (type.isF64())
        mangledNativeFunc += "d";
    };

    for (auto type : operandTypes) {
      if (auto vecType = dyn_cast<VectorType>(type)) {
        mangledNativeFunc += "Dv" + std::to_string(vecType.getShape()[0]) + "_";
        appendFloatToMangledFunc(vecType.getElementType());
      } else
        appendFloatToMangledFunc(type);
    }

    auto funcAttr = StringAttr::get(op->getContext(), mangledNativeFunc);
    auto funcOp =
        SymbolTable::lookupNearestSymbolFrom<LLVMFuncOp>(op, funcAttr);
    if (funcOp)
      return funcOp;

    auto parentFunc = op->template getParentOfType<FunctionOpInterface>();
    assert(parentFunc && "expected there to be a parent function");
    OpBuilder b(parentFunc);

    // Create a valid global location removing any metadata attached to the
    // location as debug info metadata inside of a function cannot be used
    // outside of that function.
    auto funcType = LLVM::LLVMFunctionType::get(op.getType(), operandTypes);
    auto globalloc =
        op->getLoc()->template findInstanceOfOrUnknown<FileLineColLoc>();
    return LLVMFuncOp::create(b, globalloc, mangledNativeFunc, funcType);
  }

  const StringRef nativeFunc;
};

void mlir::populateMathToXeVMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertNativeFuncPattern<math::ExpOp>>(patterns.getContext(),
                                                      "__spirv_ocl_native_exp");
}

namespace {
struct ConvertMathToXeVMPass
    : public impl::ConvertMathToXeVMBase<ConvertMathToXeVMPass> {
  ConvertMathToXeVMPass() = default;
  void runOnOperation() override;
};
} // namespace

void ConvertMathToXeVMPass::runOnOperation() {
  auto m = getOperation();
  // MLIRContext *ctx = m.getContext();

  RewritePatternSet patterns(&getContext());
  populateMathToXeVMConversionPatterns(patterns);
  ConversionTarget target(getContext());
  target.addLegalDialect<BuiltinDialect, func::FuncDialect,
                         vector::VectorDialect, LLVM::LLVMDialect>();
  if (failed(applyPartialConversion(m, target, std::move(patterns))))
    signalPassFailure();
}
