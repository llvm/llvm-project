//===- ConvertOpenACCToLLVM.h - OpenACC conversion pass entrypoint --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OPENACCTOLLVM_CONVERTOPENACCTOLLVM_H
#define MLIR_CONVERSION_OPENACCTOLLVM_CONVERTOPENACCTOLLVM_H

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTOPENACCTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

static constexpr unsigned kPtrBasePosInDataDescriptor = 0;
static constexpr unsigned kPtrPosInDataDescriptor = 1;
static constexpr unsigned kSizePosInDataDescriptor = 2;

/// Helper class to produce LLVM dialect operations inserting
/// elements to a Data descriptor. Wraps a Value pointing to the descriptor.
/// The Value may be null, in which case none of the operations are valid.
///
/// The data descriptor holds information needed to perform data operations
/// and movements with the runtime.
/// `BasePointer`: base of the pointer being mapped.
/// `Pointer`: actual pointer of the data being mapped.
/// `Size`: size of the data being mapped.
///
/// Example:
///
/// ```c
/// struct S {
///   int x;
///   int y;
/// };
/// ```
///
/// Mapping `s.y` will result if the following information in the data
/// descriptor:
/// - `BasePointer`: address of `s`
/// - `Pointer`: address of `s.y`
/// - `Size`: size of `s.y`
///
/// For a scalar variable BasePointer and Pointer will be the same.
class DataDescriptor : public StructBuilder {
public:
  /// Construct a helper for the given descriptor value.
  explicit DataDescriptor(Value descriptor);
  /// Builds IR creating an `undef` value of the descriptor type.
  static DataDescriptor undef(OpBuilder &builder, Location loc, Type basePtrTy,
                              Type ptrTy);

  static bool isValid(Value descriptor);

  void setPointer(OpBuilder &builder, Location loc, Value ptr);
  void setBasePointer(OpBuilder &builder, Location loc, Value basePtr);
  void setSize(OpBuilder &builder, Location loc, Value size);
};

/// Collect the patterns to convert from the OpenACC dialect LLVMIR dialect.
void populateOpenACCToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOLLVM_CONVERTOPENACCTOLLVM_H
