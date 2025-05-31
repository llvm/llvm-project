//===- VCIXToLLVMIRTranslation.cpp - Translate VCIX to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR VCIX dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/VCIX/VCIXToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/VCIXDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::createIntrinsicCall;

/// Infer XLen type from opcode's type. This is done to avoid passing target
/// option around.
static llvm::Type *getXlenType(Attribute opcodeAttr,
                               LLVM::ModuleTranslation &moduleTranslation) {
  auto intAttr = cast<IntegerAttr>(opcodeAttr);
  unsigned xlenWidth = cast<IntegerType>(intAttr.getType()).getWidth();
  return llvm::Type::getIntNTy(moduleTranslation.getLLVMContext(), xlenWidth);
}

/// Return VL for VCIX intrinsic. If vl was previously set, return it,
/// otherwise construct a constant using fixed vector type.
static llvm::Value *createVL(llvm::IRBuilderBase &builder, llvm::Value *vl,
                             VectorType vtype, llvm::Type *xlen, Location loc,
                             LLVM::ModuleTranslation &moduleTranslation) {
  if (vl) {
    assert(vtype.isScalable() &&
           "vl parameter must be set for scalable vectors");
    return builder.CreateZExtOrTrunc(vl, xlen);
  }

  assert(vtype.getRank() == 1 && "Only 1-d fixed vectors are supported");
  return mlir::LLVM::detail::getLLVMConstant(
      xlen,
      IntegerAttr::get(IntegerType::get(&moduleTranslation.getContext(), 64),
                       vtype.getShape()[0]),
      loc, moduleTranslation);
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
#include "mlir/Dialect/LLVMIR/VCIXConversions.inc"

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
