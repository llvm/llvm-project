//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods from LLVMImportInterface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/LLVMImportInterface.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

LogicalResult mlir::LLVMImportInterface::convertUnregisteredIntrinsic(
    OpBuilder &builder, llvm::CallInst *inst,
    LLVM::ModuleImport &moduleImport) {
  StringRef intrinName = inst->getCalledFunction()->getName();

  SmallVector<llvm::Value *> args(inst->args());
  ArrayRef<llvm::Value *> llvmOperands(args);

  SmallVector<llvm::OperandBundleUse> llvmOpBundles;
  llvmOpBundles.reserve(inst->getNumOperandBundles());
  for (unsigned i = 0; i < inst->getNumOperandBundles(); ++i)
    llvmOpBundles.push_back(inst->getOperandBundleAt(i));

  SmallVector<Value> mlirOperands;
  SmallVector<NamedAttribute> mlirAttrs;
  if (failed(moduleImport.convertIntrinsicArguments(
          llvmOperands, llvmOpBundles, false, {}, {}, mlirOperands, mlirAttrs)))
    return failure();

  Type resultType = moduleImport.convertType(inst->getType());
  auto op = builder.create<::mlir::LLVM::CallIntrinsicOp>(
      moduleImport.translateLoc(inst->getDebugLoc()),
      isa<LLVMVoidType>(resultType) ? TypeRange{} : TypeRange{resultType},
      StringAttr::get(builder.getContext(), intrinName),
      ValueRange{mlirOperands}, FastmathFlagsAttr{});

  moduleImport.setFastmathFlagsAttr(inst, op);

  ArrayAttr argsAttr, resAttr;
  moduleImport.convertParameterAttributes(inst, argsAttr, resAttr, builder);
  op.setArgAttrsAttr(argsAttr);
  op.setResAttrsAttr(resAttr);

  // Update importer tracking of results.
  unsigned numRes = op.getNumResults();
  if (numRes == 1)
    moduleImport.mapValue(inst) = op.getResult(0);
  else if (numRes == 0)
    moduleImport.mapNoResultOp(inst);
  else
    return op.emitError(
        "expected at most one result from target intrinsic call");

  return success();
}

/// Converts the LLVM intrinsic to an MLIR operation if a conversion exists.
/// Returns failure otherwise.
LogicalResult mlir::LLVMImportInterface::convertIntrinsic(
    OpBuilder &builder, llvm::CallInst *inst,
    LLVM::ModuleImport &moduleImport) const {
  // Lookup the dialect interface for the given intrinsic.
  // Verify the intrinsic identifier maps to an actual intrinsic.
  llvm::Intrinsic::ID intrinId = inst->getIntrinsicID();
  assert(intrinId != llvm::Intrinsic::not_intrinsic);

  // First lookup the intrinsic across different dialects for known
  // supported conversions, examples include arm-neon, nvm-sve, etc.
  Dialect *dialect = nullptr;

  if (!moduleImport.useUnregisteredIntrinsicsOnly())
    dialect = intrinsicToDialect.lookup(intrinId);

  // No specialized (supported) intrinsics, attempt to generate a generic
  // version via llvm.call_intrinsic (if available).
  if (!dialect)
    return convertUnregisteredIntrinsic(builder, inst, moduleImport);

  // Dispatch the conversion to the dialect interface.
  const LLVMImportDialectInterface *iface = getInterfaceFor(dialect);
  assert(iface && "expected to find a dialect interface");
  return iface->convertIntrinsic(builder, inst, moduleImport);
}
