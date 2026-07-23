//===- LowerFunctionDiscardablesToLLVM.cpp - Func discardables to llvm ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/LowerFunctionDiscardablesToLLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"

using namespace mlir;

#define DEBUG_TYPE "lower-function-discardables-to-llvm"

FailureOr<LoweredLLVMFuncAttrs>
mlir::lowerDiscardableAttrsForLLVMFunc(FunctionOpInterface funcOp,
                                       Type llvmFuncType) {
  MLIRContext *ctx = funcOp->getContext();
  LoweredLLVMFuncAttrs result;

  result.properties.sym_name = StringAttr::get(ctx, funcOp.getName());
  result.properties.function_type = TypeAttr::get(llvmFuncType);

  llvm::SmallDenseSet<StringRef> odsAttrNames(
      LLVM::LLVMFuncOp::getAttributeNames().begin(),
      LLVM::LLVMFuncOp::getAttributeNames().end());

  NamedAttrList inherentAttrs;

  for (const NamedAttribute &attr : funcOp->getDiscardableAttrs()) {
    StringRef attrName = attr.getName().strref();

    if (odsAttrNames.contains(attrName)) {
      LDBG() << "LLVM specific attributes: " << attrName
             << "should use llvm.* prefix, discarding it";
      continue;
    }

    StringRef inherent = attrName;
    if (inherent.consume_front("llvm.") && odsAttrNames.contains(inherent))
      inherentAttrs.set(inherent, attr.getValue()); // collect inherent attrs
    else
      result.discardableAttrs.push_back(attr);
  }

  // Convert collected inherent attrs into typed properties.
  if (!inherentAttrs.empty()) {
    DictionaryAttr dict = inherentAttrs.getDictionary(ctx);
    auto emitError = [&] {
      return funcOp.emitOpError("invalid llvm.func property");
    };
    if (failed(LLVM::LLVMFuncOp::setPropertiesFromAttr(result.properties, dict,
                                                       emitError))) {
      return failure();
    }
  }
  return result;
}
