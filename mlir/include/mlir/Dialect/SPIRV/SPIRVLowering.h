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

#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {

/// Type conversion from standard types to SPIR-V types for shader interface.
///
/// For composite types, this converter additionally performs type wrapping to
/// satisfy shader interface requirements: shader interface types must be
/// pointers to structs.
class SPIRVTypeConverter : public TypeConverter {
public:
  SPIRVTypeConverter();

  /// Gets the SPIR-V correspondence for the standard index type.
  static Type getIndexType(MLIRContext *context);
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
  static std::unique_ptr<SPIRVConversionTarget> get(TargetEnvAttr targetEnv,
                                                    MLIRContext *context);

private:
  SPIRVConversionTarget(TargetEnvAttr targetEnv, MLIRContext *context);

  // Be explicit that instance of this class cannot be copied or moved: there
  // are lambdas capturing fields of the instance.
  SPIRVConversionTarget(const SPIRVConversionTarget &) = delete;
  SPIRVConversionTarget(SPIRVConversionTarget &&) = delete;
  SPIRVConversionTarget &operator=(const SPIRVConversionTarget &) = delete;
  SPIRVConversionTarget &operator=(SPIRVConversionTarget &&) = delete;

  /// Returns true if the given `op` is legal to use under the current target
  /// environment.
  bool isLegalOp(Operation *op);

  Version givenVersion;                            /// SPIR-V version to target
  llvm::SmallSet<Extension, 4> givenExtensions;    /// Allowed extensions
  llvm::SmallSet<Capability, 8> givenCapabilities; /// Allowed capabilities
};

/// Returns the value for the given `builtin` variable. This function gets or
/// inserts the global variable associated for the builtin within the nearest
/// enclosing op that has a symbol table. Returns null Value if such an
/// enclosing op cannot be found.
Value getBuiltinVariableValue(Operation *op, BuiltIn builtin,
                              OpBuilder &builder);

/// Performs the index computation to get to the element at `indices` of the
/// memory pointed to by `basePtr`, using the layout map of `baseType`.

// TODO(ravishankarm) : This method assumes that the `baseType` is a MemRefType
// with AffineMap that has static strides. Extend to handle dynamic strides.
spirv::AccessChainOp getElementPtr(SPIRVTypeConverter &typeConverter,
                                   MemRefType baseType, Value basePtr,
                                   ArrayRef<Value> indices, Location loc,
                                   OpBuilder &builder);

/// Sets the InterfaceVarABIAttr and EntryPointABIAttr for a function and its
/// arguments.
LogicalResult setABIAttrs(spirv::FuncOp funcOp,
                          EntryPointABIAttr entryPointInfo,
                          ArrayRef<InterfaceVarABIAttr> argABIInfo);
} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
