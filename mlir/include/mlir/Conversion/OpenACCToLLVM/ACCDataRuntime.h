//===- ACCDataRuntime.h - OpenACC data runtime arguments --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Arguments and descriptors the OpenACC data runtime entry points are called
// with. A conversion that lowers its own construct to those entry points emits
// them with these, rather than restating the argument layout.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACCTOLLVM_ACCDATARUNTIME_H
#define MLIR_CONVERSION_OPENACCTOLLVM_ACCDATARUNTIME_H

#include "mlir/Dialect/OpenACC/OpenACCRuntimeUtils.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
class MemRefType;
class Operation;
class ConversionPatternRewriter;
class RewriterBase;
class Region;
class SymbolTable;

namespace acc {
class OpenACCSupport;
} // namespace acc

/// The arguments every mapping entry point of the OpenACC runtime takes, in the
/// order they are passed. An entry point that maps `argNum` objects reads one
/// element from each of the arrays below per object; an entry point may also
/// take further arguments of its own, which are not part of this set.
struct ACCDataRuntimeArgs {
  /// Source position and enclosing function name of the directive.
  Value ident;
  /// Reserved for per-call runtime flags; no flag is defined yet.
  Value flags;
  /// Device type the directive applies to, in the runtime encoding.
  Value deviceType;
  /// Number of mapped objects, that is, the length of each array below.
  Value argNum;
  /// Array of pointer slots to attach a mapped object to, null when the object
  /// is not attached to anything.
  Value argBasePtrs;
  /// Array of addresses of the mapped objects themselves.
  Value argPtrs;
  /// Array of object sizes in bytes; zero where the descriptor or the bounds
  /// state the extent instead.
  Value argSizes;
  /// Array of map-type flags, in the runtime encoding of `acc::MapFlags`.
  Value argTypes;
  /// Array of variable names for runtime diagnostics, null where unknown.
  Value argNames;
  /// Array of user-defined mappers. OpenACC has none, so this is always null.
  Value argMappers;
  /// Array of descriptors stating layout and bounds, null for whole objects of
  /// a known size.
  Value argDescs;

  /// Returns the fields above in the order the entry points take them, so that
  /// a caller only appends the arguments specific to the one it calls.
  SmallVector<Value> getCallArgs() const;
};

/// Identifies how data runtime arguments will be consumed.
enum class ACCDataCallKind {
  DataEnter,
  DataExit,
};

/// Materialize \p values as a stack-allocated LLVM array.
Value createACCDataArray(Location loc, Type elementType, ArrayRef<Value> values,
                         RewriterBase &rewriter);

/// Wrap \p baseDescriptor in the OpenACC overlay that carries \p bounds.
/// Field zero of \p baseDescriptor is set to \p version with the OpenACC
/// descriptor bit added.
Value createACCDataDescriptor(Location loc, Value baseDescriptor,
                              Type baseDescriptorType, uint32_t version,
                              ValueRange bounds, Value elementSize,
                              ConversionPatternRewriter &rewriter);

/// Build the runtime argument descriptor wrapping an already converted memref.
/// The memref descriptor itself is the one the memref-to-LLVM conversion
/// produced; this points the runtime at it. When \p bounds is non-empty, the
/// wrapper is nested in the OpenACC overlay that carries those bounds instead
/// of being stored on its own.
Value createACCMemRefDescriptorWrapperArg(Location loc, MemRefType memrefType,
                                          Value convertedMemref,
                                          ValueRange bounds, Value elementSize,
                                          ConversionPatternRewriter &rewriter);

/// Build the runtime argument descriptor for \p mapOp, or a null pointer when
/// the mapping needs no descriptor.
Value createACCArgumentDescriptor(Operation *mapOp, Value convertedOperand,
                                  acc::OpenACCSupport &accSupport,
                                  ConversionPatternRewriter &rewriter);

/// Emit the OpenACC data runtime arguments for data-clause operands. The
/// operands may be `acc.map_info` or the data-clause operations it replaces.
/// The emitted arrays name globals after the mapped objects and the position
/// of the directive; they are created in \p globalSymbolRegion, with \p
/// symbolTable as in acc::getOrCreateGlobalString.
LogicalResult emitACCDataRuntimeArgs(
    Location loc, ValueRange mappingOperands, ValueRange convertedOperands,
    ConversionPatternRewriter &rewriter, Region &globalSymbolRegion,
    acc::OpenACCSupport &accSupport, const acc::ACCRuntimeCallConfig &config,
    ACCDataRuntimeArgs &runtimeArgs,
    ACCDataCallKind callKind = ACCDataCallKind::DataEnter,
    SymbolTable *symbolTable = nullptr);

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_ACCDATARUNTIME_H
