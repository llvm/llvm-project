//===- LinalgOps.td - Linalg dialect ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for linear algebra operations.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_OPS
#define LINALG_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalg_Op<string mnemonic, list<Trait> traits = []> :
    Op<Linalg_Dialect, mnemonic, traits>;

def Linalg_InitTensorOp : Linalg_Op<"init_tensor",
    [NoSideEffect,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]> {
  let summary = "operation to define a tensor of particular shape";

  let description = [{
    `linalg.init_tensor` is an operation that defines a tensor of a particular
    shape. The shape could be dynamic or static. The contents of the tensor are
    unspecified and the only purpose of the op result is to materialize the
    specified shape in IR and make it available to other transformations.

    Note: This op can be lowered to a `bufferization.alloc_tensor`, at which
    point it turns into an explicit buffer allocation.
  }];

  let arguments =
    (ins Variadic<Index>:$sizes, I64ArrayAttr:$static_sizes);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes) attr-dict
    `:` type($result)
  }];

  let extraClassDeclaration = [{
    static StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }

    RankedTensorType getType() {
      return getResult().getType().cast<RankedTensorType>(); }

    // Infer the shape of the result tensor given the static shapes
    // and element type of the result tensor.
    static Type inferResultType(ArrayRef<int64_t> staticSizes, Type elementType,
                                Attribute encoding = {});

    // Return true if the size of the tensor is dynamic at `idx`
    bool isDynamicSize(unsigned idx) {
      APInt v = *(static_sizes().getAsValueRange<IntegerAttr>().begin() + idx);
      return ShapedType::isDynamic(v.getSExtValue());
    }

    // Assert that the size of the result tensor is static at `idx`
    // and return the shape.
    int64_t getStaticSize(unsigned idx) {
      assert(!isDynamicSize(idx) && "expected static size");
      APInt v = *(static_sizes().
          template getAsValueRange<IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }

    // Return the argument position that contains the dynamic size of
    // the tensor at dimension `idx`. Asserts that the shape is
    // dynamic at that `idx`.
    unsigned getIndexOfDynamicSize(unsigned idx) {
      assert(isDynamicSize(idx) && "expected dynamic size");
      return std::count_if(
          static_sizes().getValue().begin(),
          static_sizes().getValue().begin() + idx,
          [&](Attribute attr) {
            return ShapedType::isDynamic(attr.cast<IntegerAttr>().getInt());
          });
    }

    // Return both static and dynamic sizes as a list of `OpFoldResult`.
    SmallVector<OpFoldResult> getMixedSizes();

    // Return the Value of the dynamic size of the tensor at dimension
    // `idx`. Asserts that the shape is dynamic at that `idx.
    Value getDynamicSize(unsigned idx) {
      return getOperand(getIndexOfDynamicSize(idx));
    }
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$shape,
                  "ArrayRef<int64_t>":$staticShape, "Type":$elementType),
    [{
      build($_builder, $_state,
            InitTensorOp::inferResultType(staticShape, elementType),
            shape, $_builder.getI64ArrayAttr(staticShape));
    }]>,
    OpBuilder<(ins "ValueRange":$shape, "Type":$elementType),
    [{
      SmallVector<int64_t, 4> staticShape(
        shape.size(), ShapedType::kDynamicSize);
      build($_builder, $_state, shape, staticShape, elementType);
    }]>,
    OpBuilder<(ins "ArrayRef<int64_t>":$staticShape, "Type":$elementType),
    [{
      build($_builder, $_state, ValueRange{}, staticShape, elementType);
    }]>,
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes, "Type":$elementType,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Linalg_YieldOp : Linalg_Op<"yield", [NoSideEffect, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Linalg yield operation";
  let description = [{
    `linalg.yield` is a special terminator operation for blocks inside regions
    in `linalg` generic ops. It returns values to the immediately enclosing
    `linalg` generic op.

    Example:

    ```mlir
    linalg.yield %f0, %f1 : f32, f32
    ```
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Linalg_IndexOp : Linalg_Op<"index", [NoSideEffect]>,
    Arguments<(ins Confined<I64Attr, [IntMinValue<0>]>:$dim)>,
    Results<(outs Index:$result)> {
  let summary = "linalg index operation";
  let description = [{
    The `linalg.index` operation returns the iteration index of the immediately
    enclosing linalg structured operation for the iteration dimension `dim`. The
    `dim` attribute specifies the position of the accessed dimension in the
    indexing map domain.

    Example:

    ```mlir
    #map = affine_map<(i, j) -> (i, j)>
    linalg.generic {indexing_maps = [#map, #map],
                    iterator_types = ["parallel", "parallel"]}
      outs(%I, %J : memref<?x?xindex>, memref<?x?xindex>) {
      ^bb0(%arg0 : index, %arg1 : index):
      // Access the outer iteration dimension i
      %i = linalg.index 0 : index
      // Access the inner iteration dimension j
      %j = linalg.index 1 : index
      linalg.yield %i, %j : index, index
    }
    ```

    This may lower to IR resembling:

    ```mlir
    %0 = dim %I, %c0 : memref<?x?xindex>
    %1 = dim %I, %c1 : memref<?x?xindex>
    scf.for %i = %c0 to %0 step %c1 {
      scf.for %j = %c0 to %1 step %c1 {
        store %i, %I[%i, %j] : memref<?x?xindex>
        store %j, %J[%i, %j] : memref<?x?xindex>
      }
    }
    ```
  }];

  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
  let hasVerifier = 1;
}

#endif // LINALG_OPS
