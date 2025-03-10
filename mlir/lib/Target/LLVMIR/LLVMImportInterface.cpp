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

  Type results = moduleImport.convertType(inst->getType());
  auto op = builder.create<::mlir::LLVM::CallIntrinsicOp>(
      moduleImport.translateLoc(inst->getDebugLoc()), results,
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
