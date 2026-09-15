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
#include "mlir/IR/Region.h"
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

/// IDs for the argument descriptors of the OpenACC data entry points, declared
/// in OpenACCRuntimeDescriptors.def.
enum class DataDescriptor {
#define ACC_DESC_BEGIN(Enum, ...) Enum,
#include "mlir/Dialect/OpenACC/OpenACCRuntimeDescriptors.def"
};

/// Field indices of each descriptor, in declaration order.
#define ACC_DESC_BEGIN(Enum, ...) enum class Enum##Field : int64_t {
#define ACC_DESC_FIELD(Enum, Field, TypeToken) Field,
#define ACC_DESC_END(Enum)                                                     \
  }                                                                            \
  ;
#include "mlir/Dialect/OpenACC/OpenACCRuntimeDescriptors.def"

/// The index an insert or extract of \p field addresses.
template <typename FieldEnum>
int64_t getDataDescriptorFieldIndex(FieldEnum field) {
  return static_cast<int64_t>(field);
}

/// Returns the name of the runtime type \p desc materializes.
StringRef getDataDescriptorName(DataDescriptor desc);

/// Returns the descriptor kind the runtime reads from the version field of
/// \p desc. An overlay contributes its kind to the descriptor it nests.
DataDescKind getDataDescriptorKind(DataDescriptor desc);

/// Builds the LLVM type of \p desc in \p ctx. A descriptor that nests another
/// one takes it as \p baseType, which is ignored otherwise.
LLVM::LLVMStructType getDataDescriptorType(MLIRContext *ctx,
                                           DataDescriptor desc,
                                           Type baseType = {});

/// Configuration for OpenACC to LLVM runtime lowering. Device types and map
/// flags use the OpenACC dialect encodings by default.
class ACCRuntimeCallConfig {
public:
  using FunctionDisplayNameFn = std::function<std::string(StringRef)>;

  ACCRuntimeCallConfig();

  void setName(RuntimeFunction fn, StringRef name);
  StringRef getName(RuntimeFunction fn) const;

  void setFunctionDisplayNameFn(FunctionDisplayNameFn fn);
  std::string getFunctionDisplayName(StringRef mangledOrSymbol) const;

  /// Map an OpenACC dialect \p DeviceType to the integer encoding expected by
  /// the target runtime. Dialect ordinals and runtime ABI values are not
  /// required to match. A target whose ABI differs from the default dialect
  /// encoding must install its mapping here. Querying an unmapped type is an
  /// error.
  void setDeviceTypeRuntimeValue(DeviceType type, int64_t runtimeValue);
  int64_t getDeviceTypeRuntimeValue(DeviceType type) const;

  /// Map a single OpenACC dialect \p MapFlags bit to the bit the target runtime
  /// gives the same meaning. getMapFlagsRuntimeValue combines the bits of a set
  /// of flags into the encoding the runtime reads from an argument-type slot.
  /// As with device types, dialect and runtime encodings are not required to
  /// match. A target whose ABI differs from the default dialect encoding must
  /// install its mapping here, and a set bit with no mapping is an error.
  void setMapFlagRuntimeValue(MapFlags flag, int64_t runtimeValue);
  int64_t getMapFlagsRuntimeValue(MapFlags flags) const;

  /// Adjust the flags of a mapping before they are encoded. The hook is keyed
  /// on the operation that states the mapping and is applied to every mapped
  /// object before anything is derived from the flags. The flags are used as
  /// computed when no hook is installed.
  using MapFlagsPostProcessFn =
      std::function<MapFlags(Operation *mapOp, MapFlags flags)>;
  void setMapFlagsPostProcessFn(MapFlagsPostProcessFn fn);
  MapFlags postProcessMapFlags(Operation *mapOp, MapFlags flags) const;

  /// Renders \p flags for a diagnostic as the names of the set bits and the
  /// decimal and hexadecimal encoding the runtime reads.
  std::string formatMapFlags(MapFlags flags) const;

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
  DenseMap<MapFlags, int64_t> mapFlagRuntimeValues;
  FunctionDisplayNameFn functionDisplayNameFn;
  MapFlagsPostProcessFn mapFlagsPostProcessFn;
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

/// Install a map-flag mapping that uses the OpenACC dialect bit positions as
/// the runtime encoding, with the same caveat as
/// \c populateDialectIdentityDeviceTypeMapping.
void populateDialectIdentityMapFlagsMapping(ACCRuntimeCallConfig &config);

/// Declares (if needed) and returns a call to the runtime function identified
/// by \p fn using the name from \p config. Fails and emits a diagnostic if the
/// symbol is already declared with a signature the runtime cannot be called
/// through. The declaration is created in \p globalSymbolRegion and
/// registered in \p symbolTable.
FailureOr<LLVM::CallOp> createRuntimeCall(Location loc, OpBuilder &builder,
                                          Region &globalSymbolRegion,
                                          SymbolTable &symbolTable,
                                          RuntimeFunction fn,
                                          const ACCRuntimeCallConfig &config,
                                          ArrayRef<Value> arguments);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCRUNTIMEUTILS_H
