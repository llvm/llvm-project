//===- ACCToLLVMUtils.h - OpenACC to LLVM helpers ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVMUTILS_H
#define MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVMUTILS_H

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACCRuntimeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace acc {

/// Unfuses fused locations, returning the last sub-location.
Location unfuseLoc(Location loc);

/// Returns file:line:column location information when available.
std::optional<FileLineColLoc> getFileLineColLoc(Location loc,
                                                bool errorOnInvalidLocation);

/// Returns the symbol name of the function \p op belongs to, or of \p op itself
/// when it is a function.
StringRef getParentFunctionName(Operation *op);

/// Returns the enclosing function symbol name for \p value's defining op.
StringRef getParentFunctionName(Value value);

/// Returns the first non-empty enclosing function name from \p values.
StringRef getParentFunctionName(ValueRange values);

/// Returns the name to give a global that the conversion creates to hold
/// \p detail of \p kind, such as the name of a variable or a source position.
/// The dots make a name no identifier of the program can carry, so these
/// globals are reachable by name without walking the symbols of the module.
/// \p detail keeps letters, digits, `_`, `$` and `.` and folds every other
/// character to an underscore. Distinct details can therefore collide on the
/// same name; getOrCreateGlobalString is what then keeps the globals apart.
std::string getInternalGlobalName(StringRef kind, StringRef detail);

/// Creates or reuses a null-terminated string global in \p globalSymbolRegion.
/// With \p symbolTable, a global already going by \p name is reused when it
/// holds \p value, and a name that two different values arrive under gets a
/// suffix to tell the globals apart. Without it the global is created under
/// \p name as given, which only a caller whose names are unique by
/// construction can ask for.
Value getOrCreateGlobalString(Location loc, OpBuilder &builder, StringRef name,
                              StringRef value, Region &globalSymbolRegion,
                              SymbolTable *symbolTable = nullptr);

/// Returns a pointer to a constant global holding an ident_t for OpenACC
/// runtime calls. \p globalSymbolRegion and \p symbolTable are as in
/// getOrCreateGlobalString; the ident and the source string it points to are
/// named after the position they describe, so leaving out the table creates a
/// set of them per call.
Value createIdent(Location loc, StringRef functionName, OpBuilder &builder,
                  Region &globalSymbolRegion,
                  const ACCRuntimeCallConfig &config,
                  SymbolTable *symbolTable = nullptr);

/// Sign-extends or truncates \p value to the i64 the runtime entry points take
/// for values like queue numbers.
Value castToI64(Location loc, Value value, OpBuilder &builder);

/// Returns the queue an `async` clause selects: the value of the clause when it
/// has one, the queue standing for an `async` clause without a value when
/// \p asyncOnly is set, and the synchronous queue when there is no clause at
/// all. \p asyncOperand must already be converted to the LLVM dialect.
Value getAsyncQueue(Location loc, Value asyncOperand, bool asyncOnly,
                    OpBuilder &builder, const ACCRuntimeCallConfig &config);

/// Emits the runtime call that waits for \p waitOperands on \p asyncQueue,
/// which is what a `wait` clause or an `acc.wait` directive asks for. An empty
/// \p waitOperands waits for every queue, as a `wait` clause without values
/// does. The values must already be converted to the LLVM dialect.
LogicalResult emitWaitCall(Location loc, ValueRange waitOperands,
                           Value asyncQueue, OpBuilder &builder,
                           Region &globalSymbolRegion, SymbolTable &symbolTable,
                           const ACCRuntimeCallConfig &config);

/// Runs \p emitFn guarded by a branch on \p ifCond, or unguarded when there is
/// no condition. Leaves the insertion point after the guarded code, so that a
/// caller can keep emitting into the same block either way.
LogicalResult emitGuardedByIfCond(Location loc, Value ifCond,
                                  RewriterBase &rewriter,
                                  function_ref<LogicalResult()> emitFn);

/// The clauses of a construct can be given once per device type. Of the values
/// that reach a given device type, the ones naming it are the most specific,
/// then the ones naming every device type, then the ones given before any
/// device_type clause.
SmallVector<DeviceType, 3> getDeviceTypesByPrecedence(DeviceType deviceType);

namespace detail {
/// The constructs carrying a `device_type` clause hold their async and wait
/// clauses per device type. The others, such as `acc.enter_data` or
/// `acc.kernel_environment`, hold a single value for each clause, spelling the
/// value-less form either `asyncOnly`/`waitOnly` or `async`/`wait`.
template <typename OpTy>
using has_device_type_clauses_t =
    decltype(std::declval<OpTy>().hasAsyncOnly(DeviceType::None));
template <typename OpTy>
using has_async_only_t = decltype(std::declval<OpTy>().getAsyncOnly());
template <typename OpTy>
using has_wait_only_t = decltype(std::declval<OpTy>().getWaitOnly());
} // namespace detail

/// Returns the value the `async` clause of \p op names for \p deviceType, and
/// sets \p asyncOnly when the clause names no queue. The value is the one the
/// operation holds, so a caller in a conversion has to remap it.
template <typename OpTy>
Value getAsyncClauseValue(OpTy op, DeviceType deviceType, bool &asyncOnly) {
  asyncOnly = false;
  if constexpr (llvm::is_detected<detail::has_device_type_clauses_t,
                                  OpTy>::value) {
    for (DeviceType candidate : getDeviceTypesByPrecedence(deviceType)) {
      if (op.hasAsyncOnly(candidate)) {
        asyncOnly = true;
        return {};
      }
      if (Value asyncValue = op.getAsyncValue(candidate))
        return asyncValue;
    }
    return {};
  } else if constexpr (llvm::is_detected<detail::has_async_only_t,
                                         OpTy>::value) {
    asyncOnly = op.getAsyncOnly();
    return op.getAsyncOperand();
  } else {
    asyncOnly = op.getAsync();
    return op.getAsyncOperand();
  }
}

/// Appends to \p waitValues the queues the `wait` clause of \p op names for
/// \p deviceType, and returns the device type the clause naming them is given
/// for - a clause naming no queue waits for every one of them. Returns
/// std::nullopt when \p op gives no such clause for \p deviceType. The values
/// are the ones the operation holds, so a caller in a conversion has to remap
/// them.
template <typename OpTy>
std::optional<DeviceType>
getWaitClauseValues(OpTy op, DeviceType deviceType,
                    SmallVectorImpl<Value> &waitValues) {
  if constexpr (llvm::is_detected<detail::has_device_type_clauses_t,
                                  OpTy>::value) {
    for (DeviceType candidate : getDeviceTypesByPrecedence(deviceType)) {
      if (op.hasWaitOnly(candidate))
        return candidate;
      auto values = op.getWaitValues(candidate);
      if (!values.empty()) {
        llvm::append_range(waitValues, values);
        return candidate;
      }
    }
    return std::nullopt;
  } else {
    llvm::append_range(waitValues, op.getWaitOperands());
    bool waitsForEveryQueue;
    if constexpr (llvm::is_detected<detail::has_wait_only_t, OpTy>::value)
      waitsForEveryQueue = op.getWaitOnly();
    else
      waitsForEveryQueue = op.getWait();
    if (!waitsForEveryQueue && waitValues.empty())
      return std::nullopt;
    // The operation holds a single wait clause, which no device_type clause
    // narrows to a device type.
    return DeviceType::None;
  }
}

/// Returns whether the `wait` clause \p op gives for \p deviceType carries a
/// devnum modifier, which selects the device the queues belong to. Only the
/// clause given for that device type is asked, so a caller passes the device
/// type getWaitClauseValues took the queues from.
template <typename OpTy>
bool hasWaitDevnum(OpTy op, DeviceType deviceType) {
  if constexpr (llvm::is_detected<detail::has_device_type_clauses_t,
                                  OpTy>::value) {
    return static_cast<bool>(op.getWaitDevnum(deviceType));
  } else {
    return static_cast<bool>(op.getWaitDevnum());
  }
}

/// Returns the queue that the runtime calls of \p op run on, from the `async`
/// clause it gives for \p deviceType.
template <typename OpTy>
Value getAsyncQueue(OpTy op, DeviceType deviceType,
                    ConversionPatternRewriter &rewriter,
                    const ACCRuntimeCallConfig &config) {
  bool asyncOnly = false;
  Value asyncValue = getAsyncClauseValue(op, deviceType, asyncOnly);
  if (asyncValue)
    asyncValue = rewriter.getRemappedValue(asyncValue);
  return getAsyncQueue(op.getLoc(), asyncValue, asyncOnly, rewriter, config);
}

/// Emits the wait that a `wait` clause on \p op asks for before the runtime
/// calls of the construct, waiting on \p asyncQueue for the queues the clause
/// names, or for every queue when it names none. Nothing is emitted when there
/// is no such clause for \p deviceType. A devnum modifier, which selects the
/// device the queues belong to, is reported through \p accSupport as not yet
/// implemented.
template <typename OpTy>
LogicalResult
emitWaitClause(OpTy op, DeviceType deviceType, Value asyncQueue,
               ConversionPatternRewriter &rewriter, OpenACCSupport &accSupport,
               Region &globalSymbolRegion, SymbolTable &symbolTable,
               const ACCRuntimeCallConfig &config) {
  SmallVector<Value> waitValues;
  std::optional<DeviceType> clauseDeviceType =
      getWaitClauseValues(op, deviceType, waitValues);
  if (!clauseDeviceType)
    return success();
  if (hasWaitDevnum(op, *clauseDeviceType)) {
    (void)accSupport.emitNYI(op.getLoc(), "wait clause with a devnum modifier");
    return failure();
  }
  for (Value &waitValue : waitValues)
    waitValue = rewriter.getRemappedValue(waitValue);
  return emitWaitCall(op.getLoc(), waitValues, asyncQueue, rewriter,
                      globalSymbolRegion, symbolTable, config);
}

} // namespace acc
} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_ACCTOLLVMUTILS_H
