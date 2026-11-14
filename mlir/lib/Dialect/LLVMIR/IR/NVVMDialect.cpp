//===- NVVMDialect.cpp - NVVM IR dialect registration ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NVVM IR dialect in MLIR and registers it.
//
// The NVVM dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

// Any op partition header declares the hook that registers the ops of all op
// partitions. NVVMMiscSRegOps.h is the cheapest to include.
#include "mlir/Dialect/LLVMIR/NVVMMiscSRegOps.h"

using namespace mlir;
using namespace mlir::NVVM;

#include "mlir/Dialect/LLVMIR/NVVMOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// NVVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

namespace {
struct NVVMInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

// TODO: This should be the llvm.nvvm dialect once this is supported.
void NVVMDialect::initialize() {
  registerNVVMDialectOperations(this);
  registerAttributes();

  // Support unknown operations because not all NVVM operations are
  // registered.
  allowUnknownOperations();
  addInterfaces<NVVMInlinerInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, NVVMDialect>();
  declarePromisedInterface<gpu::TargetAttrInterface, NVVMTargetAttr>();
}

LogicalResult NVVMDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  StringAttr attrName = attr.getName();
  // Kernel function attribute should be attached to functions.
  if (attrName == NVVMDialect::getKernelFuncAttrName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << NVVMDialect::getKernelFuncAttrName()
                             << "' attribute attached to unexpected op";
    }
  }
  // If maxntid / reqntid / cluster_dim exist, it must be an array with max 3
  // dim
  if (attrName == NVVMDialect::getMaxntidAttrName() ||
      attrName == NVVMDialect::getReqntidAttrName() ||
      attrName == NVVMDialect::getClusterDimAttrName()) {
    auto values = llvm::dyn_cast<DenseI32ArrayAttr>(attr.getValue());
    if (!values || values.empty() || values.size() > 3) {
      return op->emitError()
             << "'" << attrName
             << "' attribute must be integer array with maximum 3 index";
    }
  }
  // If minctasm / maxnreg / cluster_max_blocks exist, it must be an integer
  // attribute
  if (attrName == NVVMDialect::getMinctasmAttrName() ||
      attrName == NVVMDialect::getMaxnregAttrName() ||
      attrName == NVVMDialect::getClusterMaxBlocksAttrName()) {
    if (!llvm::dyn_cast<IntegerAttr>(attr.getValue())) {
      return op->emitError()
             << "'" << attrName << "' attribute must be integer constant";
    }
  }
  // blocksareclusters must be used along with reqntid and cluster_dim
  if (attrName == NVVMDialect::getBlocksAreClustersAttrName()) {
    if (!op->hasDiscardableAttr(NVVMDialect::getReqntidAttrName()) ||
        !op->hasDiscardableAttr(NVVMDialect::getClusterDimAttrName())) {
      return op->emitError()
             << "'" << attrName << "' attribute must be used along with " << "'"
             << NVVMDialect::getReqntidAttrName() << "' and " << "'"
             << NVVMDialect::getClusterDimAttrName() << "'";
    }
  }

  return success();
}

LogicalResult NVVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIndex,
                                                    unsigned argIndex,
                                                    NamedAttribute argAttr) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();

  bool isKernel = op->hasDiscardableAttr(NVVMDialect::getKernelFuncAttrName());
  StringAttr attrName = argAttr.getName();
  if (attrName == NVVM::NVVMDialect::getGridConstantAttrName()) {
    if (!isKernel) {
      return op->emitError()
             << "'" << attrName
             << "' attribute must be present only on kernel arguments";
    }
    if (!isa<UnitAttr>(argAttr.getValue()))
      return op->emitError() << "'" << attrName << "' must be a unit attribute";
    if (!funcOp.getArgAttr(argIndex, LLVM::LLVMDialect::getByValAttrName())) {
      return op->emitError()
             << "'" << attrName
             << "' attribute requires the argument to also have attribute '"
             << LLVM::LLVMDialect::getByValAttrName() << "'";
    }
  }

  return success();
}
