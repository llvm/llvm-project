//===- OpenACCRuntimeUtils.cpp - OpenACC runtime call utilities -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCRuntimeUtils.h"

#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace mlir;
using namespace mlir::acc;

StringRef acc::getRuntimeFunctionName(RuntimeFunction fn) {
  switch (fn) {
#define ACC_RTL(Enum, Str, ...)                                                \
  case RuntimeFunction::Enum:                                                  \
    return Str;
#include "mlir/Dialect/OpenACC/OpenACCRuntimeFunctions.def"
  }
  llvm_unreachable("unknown ACC runtime function");
}

LLVM::LLVMFunctionType acc::getRuntimeFunctionType(MLIRContext *ctx,
                                                   RuntimeFunction fn) {
  Type Void = LLVM::LLVMVoidType::get(ctx);
  Type Ptr = LLVM::LLVMPointerType::get(ctx);
  Type Int32 = IntegerType::get(ctx, 32);
  Type Int64 = IntegerType::get(ctx, 64);

  switch (fn) {
#define ACC_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case RuntimeFunction::Enum:                                                  \
    return LLVM::LLVMFunctionType::get(ReturnType,                             \
                                       ArrayRef<Type>{__VA_ARGS__}, IsVarArg);
#include "mlir/Dialect/OpenACC/OpenACCRuntimeFunctions.def"
  }
  llvm_unreachable("unknown ACC runtime function");
}

void ACCRuntimeCallConfig::setName(RuntimeFunction fn, StringRef name) {
  overrides[fn] = name.str();
}

StringRef ACCRuntimeCallConfig::getName(RuntimeFunction fn) const {
  if (auto it = overrides.find(fn); it != overrides.end())
    return it->second;
  return getRuntimeFunctionName(fn);
}

void ACCRuntimeCallConfig::setFunctionDisplayNameFn(FunctionDisplayNameFn fn) {
  functionDisplayNameFn = std::move(fn);
}

std::string
ACCRuntimeCallConfig::getFunctionDisplayName(StringRef mangledOrSymbol) const {
  if (functionDisplayNameFn)
    return functionDisplayNameFn(mangledOrSymbol);
  return mangledOrSymbol.str();
}

void ACCRuntimeCallConfig::setDeviceTypeRuntimeValue(DeviceType type,
                                                     int64_t runtimeValue) {
  deviceTypeRuntimeValues[type] = runtimeValue;
}

int64_t ACCRuntimeCallConfig::getDeviceTypeRuntimeValue(DeviceType type) const {
  if (auto it = deviceTypeRuntimeValues.find(type);
      it != deviceTypeRuntimeValues.end())
    return it->second;
  llvm::report_fatal_error(
      llvm::Twine("missing OpenACC runtime device-type mapping for ") +
      stringifyDeviceType(type));
}

void ACCRuntimeCallConfig::setAsyncSyncRuntimeValue(int64_t runtimeValue) {
  asyncSyncRuntimeValue = runtimeValue;
}

int64_t ACCRuntimeCallConfig::getAsyncSyncRuntimeValue() const {
  return asyncSyncRuntimeValue;
}

void ACCRuntimeCallConfig::setAsyncNoValueRuntimeValue(int64_t runtimeValue) {
  asyncNoValueRuntimeValue = runtimeValue;
}

int64_t ACCRuntimeCallConfig::getAsyncNoValueRuntimeValue() const {
  return asyncNoValueRuntimeValue;
}

void acc::populateDialectIdentityDeviceTypeMapping(
    ACCRuntimeCallConfig &config) {
  for (uint32_t value = 0; value <= getMaxEnumValForDeviceType(); ++value)
    if (std::optional<DeviceType> type = symbolizeDeviceType(value))
      config.setDeviceTypeRuntimeValue(*type, value);
}

FailureOr<LLVM::CallOp>
acc::createRuntimeCall(Location loc, OpBuilder &builder, ModuleOp module,
                       RuntimeFunction fn, const ACCRuntimeCallConfig &config,
                       ArrayRef<Value> arguments) {
  MLIRContext *ctx = builder.getContext();
  LLVM::LLVMFunctionType fnTy = getRuntimeFunctionType(ctx, fn);
  StringRef symbolName = config.getName(fn);

  SymbolTable symbolTable(module);
  auto func = symbolTable.lookup<LLVM::LLVMFuncOp>(symbolName);
  if (func) {
    // An existing declaration with a different signature cannot be called with
    // the arguments expected by the runtime entry point.
    if (func.getFunctionType() != fnTy)
      return emitError(loc) << "OpenACC runtime function '" << symbolName
                            << "' is already declared with signature "
                            << func.getFunctionType() << ", expected " << fnTy;
  } else {
    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(module.getBody());
    func = LLVM::LLVMFuncOp::create(moduleBuilder, loc, symbolName, fnTy);
  }

  return LLVM::CallOp::create(builder, loc, func, arguments);
}
