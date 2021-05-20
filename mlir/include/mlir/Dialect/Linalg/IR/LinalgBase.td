//===- LinalgBase.td - Linalg dialect base support ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base linear algebra support.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_BASE
#define LINALG_BASE

include "mlir/IR/OpBase.td"

def LinalgOperand: AnyTypeOf<[AnyRankedTensor, AnyStridedMemRef]>;

def Linalg_Dialect : Dialect {
  let name = "linalg";
  let description = [{
    The `linalg` dialect groups together a set of types, operations and
    transformations that are useful to implement a structured abstraction on
    buffers and tensors. These abstractions are useful for transformations and
    can lower to scalar load/store and other operations or to more general
    library calls.

    Additional [Linalg Dialect
    Documentation](https://mlir.llvm.org/docs/Dialects/Linalg) and a
    [Rationale
    Document](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect) are
    are also available and should be read first before going in the details of
    the op semantics.
  }];
  let cppNamespace = "::mlir::linalg";
  let dependentDialects = [
    "AffineDialect", "StandardOpsDialect", "tensor::TensorDialect"
  ];
  let hasOperationAttrVerify = 1;
  let extraClassDeclaration = [{
    /// Attribute name used to to memoize indexing maps for named ops.
    constexpr const static ::llvm::StringLiteral
        kMemoizedIndexingMapsAttrName = "linalg.memoized_indexing_maps";

    /// Attribute name used to mark region arguments that can be bufferized
    /// in-place during linalg comprehensive bufferization.
    constexpr const static ::llvm::StringLiteral
      kInplaceableAttrName = "linalg.inplaceable";

    using RegionBuilderFunType =
      llvm::function_ref<void(ImplicitLocOpBuilder &b, Block &, ValueRange)>;
    RegionBuilderFunType getRegionBuilder(StringRef name) {
      return namedStructuredOpRegionBuilders.lookup(name);
    }
    private:
      llvm::StringMap<RegionBuilderFunType> namedStructuredOpRegionBuilders;
  }];
}

// Whether a type is a RangeType.
def LinalgIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : DialectType<Linalg_Dialect, LinalgIsRangeTypePred, "range">;

#endif // LINALG_BASE
