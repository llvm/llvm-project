//===- ToLLVMInterface.cpp - MLIR LLVM Conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

void mlir::populateConversionTargetFromOperation(
    Operation *root, ConversionTarget &target, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  DenseSet<Dialect *> dialects;
  root->walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (!dialects.insert(dialect).second)
      return;
    // First time we encounter this dialect: if it implements the interface,
    // let's populate patterns !
    auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
    if (!iface)
      return;
    iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                   patterns);
  });
}

void mlir::populateOpConvertToLLVMConversionPatterns(
    Operation *op, ConversionTarget &target, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  auto iface = dyn_cast<ConvertToLLVMOpInterface>(op);
  if (!iface)
    iface = op->getParentOfType<ConvertToLLVMOpInterface>();
  if (!iface)
    return;
  SmallVector<ConvertToLLVMAttrInterface, 12> attrs;
  iface.getConvertToLLVMConversionAttrs(attrs);
  for (ConvertToLLVMAttrInterface attr : attrs)
    attr.populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                 patterns);
}

#include "mlir/Conversion/ConvertToLLVM/ToLLVMAttrInterface.cpp.inc"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMOpInterface.cpp.inc"
