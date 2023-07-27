//===- NVVMToLLVM.cpp - NVVM to LLVM dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation NVVM ops which is not supported in LLVM
// core.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <regex>
#include <string>

#define DEBUG_TYPE "nvvm-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
#define GEN_PASS_DEF_CONVERTNVVMTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace NVVM;

#include "mlir/Dialect/LLVMIR/NVVMOpsInterface.cpp.inc"
namespace {

class PtxBuilder {
  Operation *op;
  PatternRewriter &rewriter;
  std::string asmStr;
  SmallVector<Value> asmVals;
  std::string asmConstraints;
  bool sideEffects;
  bool hasResult = false;

  // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  char getRegisterType(Value v) {
    if (v.getDefiningOp<LLVM::ConstantOp>())
      return 'n';
    if (v.getType().isInteger(16))
      return 'h';
    if (v.getType().isInteger(32))
      return 'r';
    if (v.getType().isInteger(64))
      return 'l';
    if (v.getType().isF32())
      return 'f';
    if (v.getType().isF64())
      return 'd';
    if (auto ptr = v.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      // Shared address spaces is addressed with 32-bit pointers.
      if (ptr.getAddressSpace() == NVVM::kSharedMemorySpace) {
        return 'r';
      }
      return 'l';
    }
    assert(false && "Register type is not handled yet");
    return ' ';
  }

public:
  PtxBuilder(Operation *op, PatternRewriter &rewriter, std::string ptxAsm,
             bool sideEffects = false)
      : op(op), rewriter(rewriter), asmStr(std::move(ptxAsm)),
        sideEffects(sideEffects) {}

  void insertValue(Value v, PTXRegisterMod itype = PTXRegisterMod::Read) {
    llvm::raw_string_ostream ss(asmConstraints);
    if (itype == PTXRegisterMod::Read) {
      asmVals.push_back(v);
    } else if (itype == PTXRegisterMod::ReadWrite) {
      asmVals.push_back(v);
      ss << "+";
      hasResult = true;
    } else if (itype == PTXRegisterMod::Write) {
      ss << "=";
      hasResult = true;
    }
    ss << getRegisterType(v) << ",";
    ss.flush();
  }

  LLVM::InlineAsmOp build() {
    auto asmDialectAttr =
        LLVM::AsmDialectAttr::get(op->getContext(), LLVM::AsmDialect::AD_ATT);
    Type resultType = hasResult ? op->getResult(0).getType()
                                : LLVM::LLVMVoidType::get(op->getContext());

    // Remove the last comma from the constraints string.
    if (!asmConstraints.empty() &&
        asmConstraints[asmConstraints.size() - 1] == ',')
      asmConstraints.pop_back();

    // asm keywords expects %, but inline assembly uses $. Replace all % with $
    std::replace(asmStr.begin(), asmStr.end(), '%', '$');

    return rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(), resultType,
        /*operands=*/asmVals,
        /*asm_string=*/llvm::StringRef(asmStr),
        /*constraints=*/asmConstraints.data(),
        /*has_side_effects=*/sideEffects,
        /*is_align_stack=*/false,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
  }

  void buildAndReplaceOp() {
    LLVM::InlineAsmOp inlineAsmOp = build();
    LLVM_DEBUG(DBGS() << "\n Generated PTX \n\t" << inlineAsmOp << "\n");
    if (inlineAsmOp->getNumResults() == op->getNumResults())
      rewriter.replaceOp(op, inlineAsmOp);
    else
      rewriter.eraseOp(op);
  }
};

struct PtxLowering
    : public OpInterfaceRewritePattern<NVVM::BasicPtxBuilderInterface> {
  using OpInterfaceRewritePattern<
      NVVM::BasicPtxBuilderInterface>::OpInterfaceRewritePattern;

  PtxLowering(MLIRContext *context, PatternBenefit benefit = 2)
      : OpInterfaceRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(NVVM::BasicPtxBuilderInterface op,
                                PatternRewriter &rewriter) const override {
    if (op.hasIntrinsic()) {
      LLVM_DEBUG(DBGS() << "Ptx Builder does not lower \n\t" << op << "\n");
      return failure();
    }

    SmallVector<std::pair<Value, PTXRegisterMod>> asmValues;
    PtxBuilder generator(op, rewriter, op.getPtx(), op.hasSideEffect());

    op.getAsmValues(rewriter, asmValues);
    for (auto &[asmValue, modifier] : asmValues) {
      LLVM_DEBUG(DBGSNL() << asmValue << "\t Modifier : " << modifier);
      generator.insertValue(asmValue, modifier);
    }

    generator.buildAndReplaceOp();
    return success();
  }
};

struct ConvertNVVMToLLVMPass
    : public impl::ConvertNVVMToLLVMPassBase<ConvertNVVMToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    RewritePatternSet pattern(&getContext());
    mlir::populateNVVMToLLVMConversionPatterns(pattern);
    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateNVVMToLLVMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<PtxLowering>(patterns.getContext());
}
