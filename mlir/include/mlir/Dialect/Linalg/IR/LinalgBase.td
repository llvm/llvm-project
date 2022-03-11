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

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

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
    "arith::ArithmeticDialect",
    "AffineDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];
  let hasCanonicalizer = 1;
  let hasOperationAttrVerify = 1;
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    /// Attribute name used to to memoize indexing maps for named ops.
    constexpr const static ::llvm::StringLiteral
        kMemoizedIndexingMapsAttrName = "linalg.memoized_indexing_maps";

    using RegionBuilderFunType = llvm::function_ref<
      void(ImplicitLocOpBuilder &b, Block &, ArrayRef<NamedAttribute>)>;
    RegionBuilderFunType getRegionBuilder(StringRef name) {
      return namedStructuredOpRegionBuilders.lookup(name);
    }
    private:
      llvm::StringMap<RegionBuilderFunType> namedStructuredOpRegionBuilders;
  }];
}

// Define the function attribute enums matching the OpDSL functions.
def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"exp", 0>,
  I32EnumAttrCase<"log", 1>,
  I32EnumAttrCase<"abs", 2>,
  I32EnumAttrCase<"ceil", 3>,
  I32EnumAttrCase<"floor", 4>,
  I32EnumAttrCase<"negf", 5>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}
def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"add", 0>,
  I32EnumAttrCase<"sub", 1>,
  I32EnumAttrCase<"mul", 2>,
  I32EnumAttrCase<"max_signed", 3>,
  I32EnumAttrCase<"min_signed", 4>,
  I32EnumAttrCase<"max_unsigned", 5>,
  I32EnumAttrCase<"min_unsigned", 6>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}
def TypeFn : I32EnumAttr<"TypeFn", "", [
  I32EnumAttrCase<"cast_signed", 0>,
  I32EnumAttrCase<"cast_unsigned", 1>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}

def UnaryFnAttr : EnumAttr<Linalg_Dialect, UnaryFn, "unary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def BinaryFnAttr : EnumAttr<Linalg_Dialect, BinaryFn, "binary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TypeFnAttr : EnumAttr<Linalg_Dialect, TypeFn, "type_fn"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // LINALG_BASE
