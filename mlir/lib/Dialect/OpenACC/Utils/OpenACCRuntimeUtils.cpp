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

#include <cassert>
#include <optional>

using namespace mlir;
using namespace mlir::acc;

ACCRuntimeCallConfig::ACCRuntimeCallConfig() {
  populateDialectIdentityDeviceTypeMapping(*this);
  populateDialectIdentityMapFlagsMapping(*this);
}

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

StringRef acc::getDataDescriptorName(DataDescriptor desc) {
  switch (desc) {
#define ACC_DESC_BEGIN(Enum, NameStr, DescKind)                                \
  case DataDescriptor::Enum:                                                   \
    return NameStr;
#include "mlir/Dialect/OpenACC/OpenACCRuntimeDescriptors.def"
  }
  llvm_unreachable("unknown ACC data descriptor");
}

DataDescKind acc::getDataDescriptorKind(DataDescriptor desc) {
  switch (desc) {
#define ACC_DESC_BEGIN(Enum, NameStr, DescKind)                                \
  case DataDescriptor::Enum:                                                   \
    return DescKind;
#include "mlir/Dialect/OpenACC/OpenACCRuntimeDescriptors.def"
  }
  llvm_unreachable("unknown ACC data descriptor");
}

LLVM::LLVMStructType acc::getDataDescriptorType(MLIRContext *ctx,
                                                DataDescriptor desc,
                                                Type baseType) {
  Type Int8 = IntegerType::get(ctx, 8);
  Type Int32 = IntegerType::get(ctx, 32);
  Type Int64 = IntegerType::get(ctx, 64);
  Type Ptr = LLVM::LLVMPointerType::get(ctx);
  Type Base = baseType;

  switch (desc) {
#define ACC_DESC_BEGIN(Enum, NameStr, DescKind)                                \
  case DataDescriptor::Enum: {                                                 \
    SmallVector<Type> fields;
#define ACC_DESC_FIELD(Enum, Field, TypeToken)                                 \
  assert(TypeToken && "no type for descriptor field " #Field);                 \
  fields.push_back(TypeToken);
#define ACC_DESC_END(Enum)                                                     \
  return LLVM::LLVMStructType::getLiteral(ctx, fields);                        \
  }
#include "mlir/Dialect/OpenACC/OpenACCRuntimeDescriptors.def"
  }
  llvm_unreachable("unknown ACC data descriptor");
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

void ACCRuntimeCallConfig::setMapFlagRuntimeValue(MapFlags flag,
                                                  int64_t runtimeValue) {
  mapFlagRuntimeValues[flag] = runtimeValue;
}

int64_t ACCRuntimeCallConfig::getMapFlagsRuntimeValue(MapFlags flags) const {
  int64_t runtimeValue = 0;
  for (unsigned bit = 0; bit != 32; ++bit) {
    auto flag = static_cast<MapFlags>(1u << bit);
    if (!bitEnumContainsAny(flags, flag))
      continue;
    auto it = mapFlagRuntimeValues.find(flag);
    if (it == mapFlagRuntimeValues.end())
      llvm::report_fatal_error(
          llvm::Twine("missing OpenACC runtime map-flag mapping for ") +
          stringifyMapFlags(flag));
    runtimeValue |= it->second;
  }
  return runtimeValue;
}

void ACCRuntimeCallConfig::setMapFlagsPostProcessFn(MapFlagsPostProcessFn fn) {
  mapFlagsPostProcessFn = std::move(fn);
}

MapFlags ACCRuntimeCallConfig::postProcessMapFlags(Operation *mapOp,
                                                   MapFlags flags) const {
  if (mapFlagsPostProcessFn)
    return mapFlagsPostProcessFn(mapOp, flags);
  return flags;
}

std::string ACCRuntimeCallConfig::formatMapFlags(MapFlags flags) const {
  int64_t runtimeValue = getMapFlagsRuntimeValue(flags);
  std::string rendered = stringifyMapFlags(flags);
  rendered += " (";
  rendered += std::to_string(runtimeValue);
  rendered += " / 0x";
  rendered += llvm::Twine::utohexstr(runtimeValue).str();
  rendered += ")";
  return rendered;
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

void acc::populateDialectIdentityMapFlagsMapping(ACCRuntimeCallConfig &config) {
  for (unsigned bit = 0; bit != 32; ++bit) {
    uint32_t value = 1u << bit;
    if (std::optional<MapFlags> flag = symbolizeMapFlags(value))
      config.setMapFlagRuntimeValue(*flag, value);
  }
}

FailureOr<LLVM::CallOp>
acc::createRuntimeCall(Location loc, OpBuilder &builder,
                       Region &globalSymbolRegion, SymbolTable &symbolTable,
                       RuntimeFunction fn, const ACCRuntimeCallConfig &config,
                       ArrayRef<Value> arguments) {
  MLIRContext *ctx = builder.getContext();
  LLVM::LLVMFunctionType fnTy = getRuntimeFunctionType(ctx, fn);
  StringRef symbolName = config.getName(fn);

  auto func = symbolTable.lookup<LLVM::LLVMFuncOp>(symbolName);
  if (func) {
    // An existing declaration with a different signature cannot be called with
    // the arguments expected by the runtime entry point.
    if (func.getFunctionType() != fnTy)
      return emitError(loc) << "OpenACC runtime function '" << symbolName
                            << "' is already declared with signature "
                            << func.getFunctionType() << ", expected " << fnTy;
  } else {
    OpBuilder moduleBuilder =
        OpBuilder::atBlockEnd(&globalSymbolRegion.front());
    func = LLVM::LLVMFuncOp::create(moduleBuilder, loc, symbolName, fnTy);
    symbolTable.insert(func);
  }

  return LLVM::CallOp::create(builder, loc, func, arguments);
}
