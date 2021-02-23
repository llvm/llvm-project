//===- ExecutionEngine.cpp - C API for MLIR JIT ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/ExecutionEngine.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Target/LLVMIR.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

extern "C" MlirExecutionEngine mlirExecutionEngineCreate(MlirModule op) {
  static bool init_once = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)init_once;

  mlir::registerLLVMDialectTranslation(*unwrap(op)->getContext());
  auto jitOrError = ExecutionEngine::create(unwrap(op));
  if (!jitOrError) {
    consumeError(jitOrError.takeError());
    return MlirExecutionEngine{nullptr};
  }
  return wrap(jitOrError->release());
}

extern "C" void mlirExecutionEngineDestroy(MlirExecutionEngine jit) {
  delete (unwrap(jit));
}

extern "C" MlirLogicalResult
mlirExecutionEngineInvokePacked(MlirExecutionEngine jit, MlirStringRef name,
                                void **arguments) {
  const std::string ifaceName = ("_mlir_ciface_" + unwrap(name)).str();
  llvm::Error error = unwrap(jit)->invokePacked(
      ifaceName, MutableArrayRef<void *>{arguments, (size_t)0});
  if (error)
    return wrap(failure());
  return wrap(success());
}

extern "C" void *mlirExecutionEngineLookup(MlirExecutionEngine jit,
                                           MlirStringRef name) {
  auto expectedFPtr = unwrap(jit)->lookup(unwrap(name));
  if (!expectedFPtr)
    return nullptr;
  return reinterpret_cast<void *>(*expectedFPtr);
}
