//===- OpenACCRuntimeUtils.h - OpenACC runtime call utilities ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for resolving OpenACC compiler-to-runtime entry points declared in
// OpenACCRuntimeFunctions.def.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCRUNTIMEUTILS_H
#define MLIR_DIALECT_OPENACC_OPENACCRUNTIMEUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <functional>
#include <string>

namespace mlir {
namespace acc {

/// IDs for OpenACC compiler-to-runtime entry points (`__tgt_acc_*`).
enum class RuntimeFunction {
#define ACC_RTL(Enum, ...) Enum,
#include "mlir/Dialect/OpenACC/OpenACCRuntimeFunctions.def"
};

/// Returns the default runtime symbol name for \p fn.
StringRef getRuntimeFunctionName(RuntimeFunction fn);

/// Builds the LLVM function type for \p fn in \p ctx.
LLVM::LLVMFunctionType getRuntimeFunctionType(MLIRContext *ctx,
                                              RuntimeFunction fn);

/// Optional overrides for OpenACC to LLVM runtime lowering.
class ACCRuntimeCallConfig {
public:
  using FunctionDisplayNameFn = std::function<std::string(StringRef)>;

  void setName(RuntimeFunction fn, StringRef name);
  StringRef getName(RuntimeFunction fn) const;

  void setFunctionDisplayNameFn(FunctionDisplayNameFn fn);
  std::string getFunctionDisplayName(StringRef mangledOrSymbol) const;

  /// Map an OpenACC dialect \p DeviceType to the integer encoding expected by
  /// the target runtime. Dialect ordinals and runtime ABI values are not
  /// required to match; callers must install a mapping that matches their
  /// runtime. Querying an unmapped type is an error.
  void setDeviceTypeRuntimeValue(DeviceType type, int64_t runtimeValue);
  int64_t getDeviceTypeRuntimeValue(DeviceType type) const;

  /// Runtime encoding of `acc_async_sync`, used when an operation carries no
  /// `async` clause. OpenACC defines the name of this queue but leaves its
  /// value to the implementation, so it is part of the runtime ABI.
  void setAsyncSyncRuntimeValue(int64_t runtimeValue);
  int64_t getAsyncSyncRuntimeValue() const;

  /// Runtime encoding of `acc_async_noval`, used for an `async` clause without
  /// an argument. As with `acc_async_sync`, the value is implementation-defined
  void setAsyncNoValueRuntimeValue(int64_t runtimeValue);
  int64_t getAsyncNoValueRuntimeValue() const;

private:
  DenseMap<RuntimeFunction, std::string> overrides;
  DenseMap<DeviceType, int64_t> deviceTypeRuntimeValues;
  FunctionDisplayNameFn functionDisplayNameFn;
  // Default to the encodings used by openacc.h (`acc_async_sync` /
  // `acc_async_noval`).
  int64_t asyncSyncRuntimeValue = -1;
  int64_t asyncNoValueRuntimeValue = -4;
};

/// Install a device-type mapping that uses OpenACC dialect enum ordinals as the
/// runtime encoding. This is only correct when the target runtime happens to
/// use the same numbering; runtimes with a different ABI must install their
/// own mapping via \c setDeviceTypeRuntimeValue.
void populateDialectIdentityDeviceTypeMapping(ACCRuntimeCallConfig &config);

/// Declares (if needed) and returns a call to the runtime function identified
/// by \p fn using the name from \p config. Fails and emits a diagnostic if the
/// symbol is already declared with a signature the runtime cannot be called
/// through.
FailureOr<LLVM::CallOp> createRuntimeCall(Location loc, OpBuilder &builder,
                                          ModuleOp module, RuntimeFunction fn,
                                          const ACCRuntimeCallConfig &config,
                                          ArrayRef<Value> arguments);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCRUNTIMEUTILS_H
