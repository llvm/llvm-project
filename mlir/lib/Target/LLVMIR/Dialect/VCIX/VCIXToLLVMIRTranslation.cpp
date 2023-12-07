//===- VCIXToLLVMIRTranslation.cpp - Translate VCIX to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the VCIX dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/VCIX/VCIXToLLVMIRTranslation.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/VCIX/VCIXDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;
using mlir::LLVM::detail::getLLVMConstant;

/// Return unary intrinsics that produces a vector
static llvm::Intrinsic::ID getUnaryIntrinsicId(Type opType,
                                               VectorType vecType) {
  return opType.isInteger(5) ? llvm::Intrinsic::riscv_sf_vc_v_i
                             : llvm::Intrinsic::riscv_sf_vc_v_x;
}

/// Return unary intrinsics that does not produce any vector
static llvm::Intrinsic::ID getUnaryROIntrinsicId(vcix::SewLmul sewLmul,
                                                 Type opType) {
  switch (sewLmul) {
#define SEW_LMUL_TO_INTRIN(SEW_LMUL)                                           \
  case vcix::SewLmul::SEW_LMUL:                                                \
    return opType.isInteger(5) ? llvm::Intrinsic::riscv_sf_vc_x_se_##SEW_LMUL  \
                               : llvm::Intrinsic::riscv_sf_vc_i_se_##SEW_LMUL;

    SEW_LMUL_TO_INTRIN(e8mf8);
    SEW_LMUL_TO_INTRIN(e8mf4);
    SEW_LMUL_TO_INTRIN(e8mf2);
    SEW_LMUL_TO_INTRIN(e8m1);
    SEW_LMUL_TO_INTRIN(e8m2);
    SEW_LMUL_TO_INTRIN(e8m4);
    SEW_LMUL_TO_INTRIN(e8m8);

    SEW_LMUL_TO_INTRIN(e16mf4);
    SEW_LMUL_TO_INTRIN(e16mf2);
    SEW_LMUL_TO_INTRIN(e16m1);
    SEW_LMUL_TO_INTRIN(e16m2);
    SEW_LMUL_TO_INTRIN(e16m4);
    SEW_LMUL_TO_INTRIN(e16m8);

    SEW_LMUL_TO_INTRIN(e32mf2);
    SEW_LMUL_TO_INTRIN(e32m1);
    SEW_LMUL_TO_INTRIN(e32m2);
    SEW_LMUL_TO_INTRIN(e32m4);
    SEW_LMUL_TO_INTRIN(e32m8);

    SEW_LMUL_TO_INTRIN(e64m1);
    SEW_LMUL_TO_INTRIN(e64m2);
    SEW_LMUL_TO_INTRIN(e64m4);
    SEW_LMUL_TO_INTRIN(e64m8);
  }
  llvm_unreachable("unknown redux kind");
}

/// Return binary intrinsics that produces any vector
static llvm::Intrinsic::ID getBinaryIntrinsicId(Type opType) {
  if (auto intTy = opType.dyn_cast<IntegerType>())
    return intTy.getWidth() == 5 ? llvm::Intrinsic::riscv_sf_vc_v_iv_se
                                 : llvm::Intrinsic::riscv_sf_vc_v_xv_se;

  if (opType.isa<FloatType>())
    return llvm::Intrinsic::riscv_sf_vc_v_fv_se;

  assert(opType.isa<VectorType>() &&
         "First operand should either be imm, float or vector ");
  return llvm::Intrinsic::riscv_sf_vc_v_vv_se;
}

/// Return binary intrinsics that does not produce any vector
static llvm::Intrinsic::ID getBinaryROIntrinsicId(Type opType) {
  if (auto intTy = opType.dyn_cast<IntegerType>())
    return intTy.getWidth() == 5 ? llvm::Intrinsic::riscv_sf_vc_iv_se
                                 : llvm::Intrinsic::riscv_sf_vc_xv_se;

  if (opType.isa<FloatType>())
    return llvm::Intrinsic::riscv_sf_vc_fv_se;

  assert(opType.isa<VectorType>() &&
         "First operand should either be imm, float or vector ");
  return llvm::Intrinsic::riscv_sf_vc_vv_se;
}

/// Return ternary intrinsics that produces any vector
static llvm::Intrinsic::ID getTernaryIntrinsicId(Type opType) {
  if (auto intTy = opType.dyn_cast<IntegerType>())
    return intTy.getWidth() == 5 ? llvm::Intrinsic::riscv_sf_vc_v_ivv_se
                                 : llvm::Intrinsic::riscv_sf_vc_v_xvv_se;

  if (opType.isa<FloatType>())
    return llvm::Intrinsic::riscv_sf_vc_v_fvv_se;

  assert(opType.isa<VectorType>() &&
         "First operand should either be imm, float or vector ");
  return llvm::Intrinsic::riscv_sf_vc_v_vvv_se;
}

/// Return ternary intrinsics that does not produce any vector
static llvm::Intrinsic::ID getTernaryROIntrinsicId(Type opType) {
  if (auto intTy = opType.dyn_cast<IntegerType>())
    return intTy.getWidth() == 5 ? llvm::Intrinsic::riscv_sf_vc_ivv_se
                                 : llvm::Intrinsic::riscv_sf_vc_xvv_se;

  if (opType.isa<FloatType>())
    return llvm::Intrinsic::riscv_sf_vc_fvv_se;

  assert(opType.isa<VectorType>() &&
         "First operand should either be imm, float or vector ");
  return llvm::Intrinsic::riscv_sf_vc_vvv_se;
}

/// Return wide ternary intrinsics that produces any vector
static llvm::Intrinsic::ID getWideTernaryIntrinsicId(Type opType) {
  if (auto intTy = opType.dyn_cast<IntegerType>())
    return intTy.getWidth() == 5 ? llvm::Intrinsic::riscv_sf_vc_v_ivw_se
                                 : llvm::Intrinsic::riscv_sf_vc_v_xvw_se;

  if (opType.isa<FloatType>())
    return llvm::Intrinsic::riscv_sf_vc_v_fvw_se;

  assert(opType.isa<VectorType>() &&
         "First operand should either be imm, float or vector ");
  return llvm::Intrinsic::riscv_sf_vc_v_vvw_se;
}

/// Return wide ternary intrinsics that does not produce any vector
static llvm::Intrinsic::ID getWideTernaryROIntrinsicId(Type opType) {
  if (auto intTy = opType.dyn_cast<IntegerType>())
    return intTy.getWidth() == 5 ? llvm::Intrinsic::riscv_sf_vc_ivw_se
                                 : llvm::Intrinsic::riscv_sf_vc_xvw_se;

  if (opType.isa<FloatType>())
    return llvm::Intrinsic::riscv_sf_vc_fvw_se;

  assert(opType.isa<VectorType>() &&
         "First operand should either be imm, float or vector ");
  return llvm::Intrinsic::riscv_sf_vc_vvw_se;
}

/// Return RVL for VCIX intrinsic. If rvl was previously set, return it,
/// otherwise construct a constant using fixed vector type
static llvm::Value *convertRvl(llvm::IRBuilderBase &builder, llvm::Value *rvl,
                               VectorType vtype, llvm::Type *xlen, Location loc,
                               LLVM::ModuleTranslation &moduleTranslation) {
  if (rvl) {
    assert(vtype.isScalable() &&
           "rvl parameter must be set for scalable vectors");
    return builder.CreateZExtOrTrunc(rvl, xlen);
  }

  assert(vtype.getRank() == 1 && "Only 1-d fixed vectors are supported");
  return getLLVMConstant(
      xlen,
      IntegerAttr::get(IntegerType::get(&moduleTranslation.getContext(), 64),
                       vtype.getShape()[0]),
      loc, moduleTranslation);
}

/// Infer Xlen width from opcode's type. This is done to avoid passing target
/// option around
static unsigned getXlenFromOpcode(Attribute opcodeAttr) {
  auto intAttr = opcodeAttr.cast<IntegerAttr>();
  return intAttr.getType().cast<IntegerType>().getWidth();
}

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the VCIX dialect to LLVM IR.
class VCIXDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "mlir/Dialect/VCIX/VCIXConversions.inc"
    return failure();
  }
};
} // namespace

void mlir::registerVCIXDialectTranslation(DialectRegistry &registry) {
  registry.insert<vcix::VCIXDialect>();
  registry.addExtension(+[](MLIRContext *ctx, vcix::VCIXDialect *dialect) {
    dialect->addInterfaces<VCIXDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerVCIXDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerVCIXDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
