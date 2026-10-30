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
#include "mlir/Target/LLVMIR/ModuleImport.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

LogicalResult mlir::LLVMImportInterface::convertUnregisteredIntrinsic(
    OpBuilder &builder, llvm::CallInst *inst,
    LLVM::ModuleImport &moduleImport) {
  StringRef intrinName = inst->getCalledFunction()->getName();
  Location loc = moduleImport.translateLoc(inst->getDebugLoc());

  SmallVector<Value> args;
  SmallVector<int32_t> positions;
  SmallVector<Attribute> values;
  for (auto [index, arg] : llvm::enumerate(inst->args())) {
    // paramHasAttr also checks the intrinsic declaration's attributes.
    if (inst->paramHasAttr(index, llvm::Attribute::ImmArg)) {
      auto *constant = dyn_cast<llvm::Constant>(arg.get());
      TypedAttr attr =
          constant ? moduleImport.convertScalarConstant(constant) : TypedAttr{};
      if (!attr)
        return emitError(loc) << "expected a scalar integer or floating-point "
                                 "constant for intrinsic argument #"
                              << index;
      positions.push_back(index);
      values.push_back(attr);
      continue;
    }
    FailureOr<Value> value = moduleImport.convertValue(arg.get());
    if (failed(value))
      return failure();
    args.push_back(*value);
  }

  SmallVector<SmallVector<Value>> bundleOperands;
  SmallVector<Attribute> bundleTags;
  for (unsigned i = 0; i < inst->getNumOperandBundles(); ++i) {
    llvm::OperandBundleUse bundle = inst->getOperandBundleAt(i);
    SmallVector<Value> operands;
    for (const llvm::Use &operand : bundle.Inputs) {
      FailureOr<Value> value = moduleImport.convertValue(operand.get());
      if (failed(value))
        return failure();
      operands.push_back(*value);
    }
    bundleOperands.push_back(std::move(operands));
    bundleTags.push_back(builder.getStringAttr(bundle.getTagName()));
  }
  SmallVector<ValueRange> bundles;
  for (const SmallVector<Value> &operands : bundleOperands)
    bundles.push_back(operands);

  Type resultType = moduleImport.convertType(inst->getType());
  auto op = CallIntrinsicOp::create(
      builder, loc,
      isa<LLVMVoidType>(resultType) ? TypeRange{} : TypeRange{resultType},
      builder.getStringAttr(intrinName), args, FastmathFlagsAttr{}, bundles,
      bundleTags.empty() ? ArrayAttr{} : builder.getArrayAttr(bundleTags),
      /*arg_attrs=*/ArrayAttr{},
      /*res_attrs=*/ArrayAttr{}, builder.getDenseI32ArrayAttr(positions),
      builder.getArrayAttr(values));

  moduleImport.setFastmathFlagsAttr(inst, op);
  SmallVector<unsigned> immArgPositions(positions.begin(), positions.end());
  moduleImport.convertArgAndResultAttrs(inst, op, immArgPositions);

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
