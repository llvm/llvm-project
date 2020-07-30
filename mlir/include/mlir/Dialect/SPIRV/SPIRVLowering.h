//===- SPIRVLowering.h - SPIR-V lowering utilities  -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines utilities to use while targeting SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
#define MLIR_DIALECT_SPIRV_SPIRVLOWERING_H

#include "mlir/Dialect/SPIRV/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {

/// Type conversion from standard types to SPIR-V types for shader interface.
///
/// Non-32-bit scalar types require special hardware support that may not exist
/// on all GPUs. This is reflected in SPIR-V as that non-32-bit scalar types
/// require special capabilities or extensions. Right now if a scalar type of a
/// certain bitwidth is not supported in the target environment, we use 32-bit
/// ones unconditionally. This requires the runtime to also feed in data with
/// a matched bitwidth and layout for interface types. The runtime can do that
/// by inspecting the SPIR-V module.
///
/// For memref types, this converter additionally performs type wrapping to
/// satisfy shader interface requirements: shader interface types must be
/// pointers to structs.
///
/// TODO: We might want to introduce a way to control how unsupported bitwidth
/// are handled and explicitly fail if wanted.
class SPIRVTypeConverter : public TypeConverter {
public:
  explicit SPIRVTypeConverter(spirv::TargetEnvAttr targetAttr);

  /// Gets the number of bytes used for a type when converted to SPIR-V
  /// type. Note that it doesnt account for whether the type is legal for a
  /// SPIR-V target (described by spirv::TargetEnvAttr). Returns None on
  /// failure.
  static Optional<int64_t> getConvertedTypeNumBytes(Type);

  /// Gets the SPIR-V correspondence for the standard index type.
  static Type getIndexType(MLIRContext *context);

  /// Returns the corresponding memory space for memref given a SPIR-V storage
  /// class.
  static unsigned getMemorySpaceForStorageClass(spirv::StorageClass);

  /// Returns the SPIR-V storage class given a memory space for memref. Return
  /// llvm::None if the memory space does not map to any SPIR-V storage class.
  static Optional<spirv::StorageClass>
  getStorageClassForMemorySpace(unsigned space);

private:
  spirv::TargetEnv targetEnv;
};

/// Base class to define a conversion pattern to lower `SourceOp` into SPIR-V.
template <typename SourceOp>
class SPIRVOpLowering : public OpConversionPattern<SourceOp> {
public:
  SPIRVOpLowering(MLIRContext *context, SPIRVTypeConverter &typeConverter,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit),
        typeConverter(typeConverter) {}

protected:
  SPIRVTypeConverter &typeConverter;
};

/// Appends to a pattern list additional patterns for translating the builtin
/// `func` op to the SPIR-V dialect. These patterns do not handle shader
/// interface/ABI; they convert function parameters to be of SPIR-V allowed
/// types.
void populateBuiltinFuncToSPIRVPatterns(MLIRContext *context,
                                        SPIRVTypeConverter &typeConverter,
                                        OwningRewritePatternList &patterns);

namespace spirv {
class AccessChainOp;
class FuncOp;

class SPIRVConversionTarget : public ConversionTarget {
public:
  /// Creates a SPIR-V conversion target for the given target environment.
  static std::unique_ptr<SPIRVConversionTarget> get(TargetEnvAttr targetAttr);

private:
  explicit SPIRVConversionTarget(TargetEnvAttr targetAttr);

  // Be explicit that instance of this class cannot be copied or moved: there
  // are lambdas capturing fields of the instance.
  SPIRVConversionTarget(const SPIRVConversionTarget &) = delete;
  SPIRVConversionTarget(SPIRVConversionTarget &&) = delete;
  SPIRVConversionTarget &operator=(const SPIRVConversionTarget &) = delete;
  SPIRVConversionTarget &operator=(SPIRVConversionTarget &&) = delete;

  /// Returns true if the given `op` is legal to use under the current target
  /// environment.
  bool isLegalOp(Operation *op);

  TargetEnv targetEnv;
};

/// Returns the value for the given `builtin` variable. This function gets or
/// inserts the global variable associated for the builtin within the nearest
/// enclosing op that has a symbol table. Returns null Value if such an
/// enclosing op cannot be found.
Value getBuiltinVariableValue(Operation *op, BuiltIn builtin,
                              OpBuilder &builder);

/// Performs the index computation to get to the element at `indices` of the
/// memory pointed to by `basePtr`, using the layout map of `baseType`.

// TODO: This method assumes that the `baseType` is a MemRefType with AffineMap
// that has static strides. Extend to handle dynamic strides.
spirv::AccessChainOp getElementPtr(SPIRVTypeConverter &typeConverter,
                                   MemRefType baseType, Value basePtr,
                                   ValueRange indices, Location loc,
                                   OpBuilder &builder);

/// Sets the InterfaceVarABIAttr and EntryPointABIAttr for a function and its
/// arguments.
LogicalResult setABIAttrs(spirv::FuncOp funcOp,
                          EntryPointABIAttr entryPointInfo,
                          ArrayRef<InterfaceVarABIAttr> argABIInfo);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
