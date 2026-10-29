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
  /// Region flags, in the encoding of `ACCDataRegionFlags`.
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

  /// The `argBasePtrs` slot the entry point writes the device address of each
  /// requested operand into, in the order the operands were requested.
  SmallVector<unsigned> returnAddressSlots;

  /// Returns the fields above in the order the entry points take them, so that
  /// a caller only appends the arguments specific to the one it calls.
  SmallVector<Value> getCallArgs() const;
};

/// Identifies how data runtime arguments will be consumed.
enum class ACCDataCallKind {
  DataEnter,
  DataExit,
};

/// Flags of `__tgt_acc_data_begin` and `__tgt_acc_data_end`. They describe the
/// data region.
enum class ACCDataRegionFlags : int64_t {
  /// An explicit data construct.
  None = 0,
  /// The implicit data region of a compute construct.
  Implicit = 1,
};

/// What a data runtime call is asked for besides mapping its operands.
struct ACCDataCallOptions {
  ACCDataCallKind callKind = ACCDataCallKind::DataEnter;
  /// Region flags, in the encoding of `ACCDataRegionFlags`.
  int64_t flags = static_cast<int64_t>(ACCDataRegionFlags::None);
  /// Mapping operands whose device address the entry point returns.
  ValueRange returnAddressOperands;
};

/// The values that describe one object to the runtime, one for each argument
/// array.
struct ACCMappedObjectArgs {
  Value base;
  Value pointer;
  Value size;
  Value type;
  Value name;
  Value descriptor;
};

/// Materialize \p values as a stack-allocated LLVM array.
Value createACCDataArray(Location loc, Type elementType, ArrayRef<Value> values,
                         RewriterBase &rewriter);

/// Materialize the argument arrays of \p objects into the `argNum` through
/// `argDescs` fields of \p runtimeArgs. An `argNum` the caller already set is
/// kept, so the count can be emitted before the object values.
void createACCMappedObjectArrays(Location loc,
                                 ArrayRef<ACCMappedObjectArgs> objects,
                                 ACCDataRuntimeArgs &runtimeArgs,
                                 RewriterBase &rewriter);

/// The address of \p operand as the runtime expects to receive it, given its
/// type converted value \p converted and its \p mapFlags.
Value createACCMappedObjectPointer(Location loc, Value operand, Value converted,
                                   acc::MapFlags mapFlags,
                                   ConversionPatternRewriter &rewriter);

/// The host pointer slot the object \p mapOp maps is attached to, or a null
/// pointer when it is not attached. \p mapOp may be null.
Value createACCMappedObjectBase(Location loc, Operation *mapOp,
                                acc::MapFlags mapFlags,
                                ConversionPatternRewriter &rewriter);

/// The size in bytes of \p operand, or zero where its descriptor or bounds
/// state the extent instead. Fails, after reporting it, when \p operand is
/// passed by value and the size of what is passed is not known.
FailureOr<Value> createACCMappedObjectSize(Location loc, Value operand,
                                           Value converted,
                                           acc::MapFlags mapFlags,
                                           acc::OpenACCSupport &accSupport,
                                           ConversionPatternRewriter &rewriter);

/// The name of \p operand for runtime diagnostics, or a null pointer when it
/// has none. The name global is created as in acc::getOrCreateGlobalString.
Value createACCMappedObjectName(Location loc, Value operand,
                                acc::OpenACCSupport &accSupport,
                                Region &globalSymbolRegion,
                                SymbolTable *symbolTable,
                                ConversionPatternRewriter &rewriter);

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
/// symbolTable as in acc::getOrCreateGlobalString. \p options gives the region
/// flags and the kind of call. Each of `options.returnAddressOperands` has
/// `acc::MapFlags::return_param` set, and its slot is recorded in
/// `returnAddressSlots`.
LogicalResult emitACCDataRuntimeArgs(
    Location loc, ValueRange mappingOperands, ValueRange convertedOperands,
    ConversionPatternRewriter &rewriter, Region &globalSymbolRegion,
    acc::OpenACCSupport &accSupport, const acc::ACCRuntimeCallConfig &config,
    ACCDataRuntimeArgs &runtimeArgs, const ACCDataCallOptions &options = {},
    SymbolTable *symbolTable = nullptr);

/// Emit the call to the mapping entry point \p fn for \p mappingOperands,
/// followed by \p trailingArguments. Emits nothing when there is no operand.
/// Returns the device address of each of `options.returnAddressOperands`.
FailureOr<SmallVector<Value>> emitACCDataRuntimeCall(
    Location loc, acc::RuntimeFunction fn, ValueRange mappingOperands,
    ValueRange convertedOperands, ConversionPatternRewriter &rewriter,
    acc::OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const acc::ACCRuntimeCallConfig &config,
    ArrayRef<Value> trailingArguments, const ACCDataCallOptions &options = {});
} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_ACCDATARUNTIME_H
