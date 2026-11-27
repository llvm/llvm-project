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
#include "mlir/Dialect/LLVMIR/LLVMCallOps.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

LogicalResult mlir::LLVMImportInterface::convertUnregisteredIntrinsic(
    OpBuilder &builder, llvm::CallInst *inst,
    LLVM::ModuleImport &moduleImport) {
  Location loc = moduleImport.translateLoc(inst->getDebugLoc());
  SmallVector<llvm::Value *> args(inst->args());
  SmallVector<Value> mlirOperands;
  SmallVector<NamedAttribute> mlirAttrs;
  if (failed(moduleImport.convertIntrinsicArguments(
          args, llvm::to_vector(inst->operand_bundles()),
          /*requiresOpBundles=*/true,
          /*immArgPositions=*/{}, /*immArgAttrNames=*/{}, mlirOperands,
          mlirAttrs)))
    return failure();

  CallIntrinsicOp::Properties properties{};
  properties.setIntrin(
      builder.getStringAttr(inst->getCalledFunction()->getName()));
  properties.setOperandSegmentSizes(
      {static_cast<int32_t>(args.size()),
       static_cast<int32_t>(mlirOperands.size() - args.size())});
  if (failed(CallIntrinsicOp::setPropertiesFromAttr(
          properties, builder.getDictionaryAttr(mlirAttrs),
          [&] { return emitError(loc); })))
    return failure();

  Type resultType = moduleImport.convertType(inst->getType());
  auto op = CallIntrinsicOp::create(
      builder, loc,
      isa<LLVMVoidType>(resultType) ? TypeRange{} : TypeRange{resultType},
      mlirOperands, properties, /*discardableAttributes=*/{});

  moduleImport.setFastmathFlagsAttr(inst, op);
  moduleImport.convertArgAndResultAttrs(inst, op);

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
