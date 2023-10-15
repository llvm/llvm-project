//===-- ArmSMEOps.td - ArmSME dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ArmSME dialect ops. It also defines custom attributes
// and types that are used to define the Ops.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSME_OPS
#define ARMSME_OPS

include "ArmSME.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// ArmSME type definitions
//===----------------------------------------------------------------------===//

class SMETileType<Type datatype, list<int> dims, string description>
  : ShapedContainerType<[datatype],
      And<[IsVectorOfRankPred<[2]>, IsVectorTypeWithAllDimsScalablePred,
           IsVectorOfShape<dims>]>,
  description>;

def nxnxv16i8  : SMETileType<I8,   [16, 16], "vector<[16]x[16]xi8>">;
def nxnxv8i16  : SMETileType<I16,  [8,  8 ], "vector<[8]x[8]xi16>">;
def nxnxv4i32  : SMETileType<I32,  [4,  4 ], "vector<[4]x[4]xi32>">;
def nxnxv2i64  : SMETileType<I64,  [2,  2 ], "vector<[2]x[2]xi64>">;
def nxnxv1i128 : SMETileType<I128, [1,  1 ], "vector<[1]x[1]xi128>">;

def nxnxv8f16  : SMETileType<F16,  [8,  8 ], "vector<[8]x[8]xf16>">;
def nxnxv8bf16 : SMETileType<BF16, [8,  8 ], "vector<[8]x[8]xbf16>">;
def nxnxv4f32  : SMETileType<F32,  [4,  4 ], "vector<[4]x[4]xf32>">;
def nxnxv2f64  : SMETileType<F64,  [2,  2 ], "vector<[2]x[2]xf64>">;

def SMETile : AnyTypeOf<[nxnxv16i8, nxnxv8i16, nxnxv4i32, nxnxv2i64, nxnxv1i128,
                         nxnxv8f16, nxnxv8bf16, nxnxv4f32, nxnxv2f64]>;

// A type constraint that verifies the bitwidth of the scalar integer returned
// from 'arm_sme.get_tile_id' matches the element bitwidth of a "virtual tile".
def TileElementWidthMatchesTileID : TypesMatchWith<
  "`tile_id` has the same number of bits as elements in `vector`",
  "vector", "tile_id",
  "IntegerType::get("
      "$_self.getContext(),"
      "::llvm::isa<IntegerType>(::llvm::cast<VectorType>($_self).getElementType())"
          "? ::llvm::cast<IntegerType>("
                  "::llvm::cast<VectorType>($_self).getElementType())"
                  ".getWidth()"
          ": ::llvm::cast<FloatType>("
                  "::llvm::cast<VectorType>($_self).getElementType())"
                  ".getWidth())">;

//===----------------------------------------------------------------------===//
// ArmSME attr definitions
//===----------------------------------------------------------------------===//

def TileSliceLayout : I32EnumAttr<"TileSliceLayout", "Layout of a tile slice", [
  I32EnumAttrCase<"Horizontal", 0, "horizontal">,
  I32EnumAttrCase<"Vertical", 1, "vertical">,
]> {
  let cppNamespace = "::mlir::arm_sme";
  let genSpecializedAttr = 0;
}

/// An attribute that specifies the layout of a tile slice in a tile.
def ArmSME_TileSliceLayoutAttr : EnumAttr<ArmSME_Dialect, TileSliceLayout,
                                          "layout"> {
  let assemblyFormat = "`<` $value `>`";
  let defaultValue = "TileSliceLayout::Horizontal";
}

//===----------------------------------------------------------------------===//
// ArmSME op definitions
//===----------------------------------------------------------------------===//

class ArmSME_Op<string mnemonic, list<Trait> traits = []> :
  Op<ArmSME_Dialect, mnemonic, traits> {}

def CastTileToVector : ArmSME_Op<"cast_tile_to_vector", [Pure, TileElementWidthMatchesTileID]> {
  let summary = "Cast from tile id to 2-d scalable vector type";
  let description = [{
    A `cast_tile_to_vector` operation does a cast from a tile id to a 2-d
    scalable vector type, which represents an SME "virtual tile". This would
    normally be used when lowering operations that return "virtual tile" vector
    types to model the output. This is required to preserve dataflow as SME
    intrinsics have no return values.

    Example:

    Input:
    ```mlir
    %tile = vector.load %mem1[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
    vector.store %tile, %mem2[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
    ```

    After lowering `vector.load`:
    ```mlir
    %tile_id = arm_sme.get_tile_id : i32
    scf.for %vnum = %c0 to %num_vectors step %c1 {
      // ...
      "arm_sme.intr.ld1w.horiz"(%pg, %ptr, %tile_id, %vnum) : (vector<[4]xi1>, !llvm.ptr, i32, i32) -> ()
    }
    %tile = arm_sme.cast_tile_to_vector %tile_id : i32 to vector<[4]x[4]xi32>
    vector.store %tile, %mem2[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
    ```

    In the example above, the `vector.load` can't be replaced with an SME
    intrinsic that has no outputs since it is used by the `vector.store`.
    However, by inserting a `cast_tile_to_vector` op after the load intrinsics
    the `vector.load` can be replaced. This enables "local" rewrites on
    individual vector ops, rather than "global" rewrites that would have to
    look at the vector op uses and also lower them.

    Canonicalization will look through `arm_sme.cast_tile_to_vector` and fold
    the cast away if it comes from a `arm_sme.cast_vector_to_tile`.
  }];
  let arguments = (ins AnyTypeOf<[I8, I16, I32, I64, I128]>:$tile_id);
  let results = (outs SMETile:$vector);
  let assemblyFormat =
    "$tile_id attr-dict `:` type($tile_id) `to` type($vector)";
  let hasCanonicalizeMethod = 1;
}

def CastVectorToTile : ArmSME_Op<"cast_vector_to_tile", [Pure, TileElementWidthMatchesTileID]> {
  let summary = "Cast from 2-d scalable vector type to tile id";
  let description = [{
    A `cast_vector_to_tile` operation does a cast from a 2-d scalable vector
    type, which represents an SME "virtual tile", to a tile id. This is
    required to preserve dataflow as the SME intrinsics have no return values.

    Example:

    Input:
    ```mlir
    %tile = vector.load %mem1[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
    vector.store %tile, %mem2[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
    ```

    After lowering `vector.store`:
    ```mlir
    %tile = vector.load %mem1[%c0] : memref<?xi32>, vector<[4]x[4]xi32>
    scf.for %vnum = %c0 to %num_vectors step %c1 {
      // ...
      %tile_id = arm_sme.cast_vector_to_tile %tile : (vector<[4]x[4]xi32>) -> i32
      "arm_sme.intr.st1w.horiz"(%pg, %ptr, %tile_id, %vnum) : (vector<[4]xi1>, !llvm.ptr, i32, i32) -> ()
    }
    ```

    Canonicalization will look through `arm_sme.cast_vector_to_tile` and fold
    the cast away if it comes from a `arm_sme.cast_tile_to_vector`.
  }];
  let arguments = (ins SMETile:$vector);
  let results = (outs AnyTypeOf<[I8, I16, I32, I64, I128]>:$tile_id);
  let assemblyFormat =
    "$vector attr-dict `:` type($vector) `to` type($tile_id)";
  let hasCanonicalizeMethod = 1;
}

def GetTileID : ArmSME_Op<"get_tile_id"> {
  let summary = "Returns an SME \"virtual tile\" id";
  let description = [{
    A `get_tile_id` operation returns a scalar integer representing an SME
    "virtual tile" id. The bitwidth of the scalar indicates the element
    bitwidth of the "virtual tile".

    The scope of a tile id is a function and cannot be passed or returned from
    functions.

    Example:
    ```mlir
    // Allocate and return an 8-bit element "virtual tile" id
    %za0_b = arm_sme.get_tile_id : i8
    ```

    Example:
    ```
    // Allocate and return two 16-bit element "virtual tile" ids
    %za0_h = arm_sme.get_tile_id : i16
    %za1_h = arm_sme.get_tile_id : i16
    ```

    Example:
    ```
    // Allocate and return an 128-bit element "virtual tile" id
    %za0_q = arm_sme.get_tile_id : i128
    ```
  }];

  let results = (outs AnyTypeOf<[I8, I16, I32, I64, I128]>:$tile_id);
  let assemblyFormat = "attr-dict `:` type($tile_id)";
}

//
// Tile reset.
//

def ZeroOp : ArmSME_Op<"zero", [Pure]> {
  let summary = "Initialize the two-dimensional ZA array with 0s";
  let results = (outs SMETile:$res);
  let description = [{
    Initialise ZA with 0. This operation is convenient wrapper for the SME
    `zero` intrinsic and instruction. 

    Example 1: Zero an 8-bit element ZA tile.

    ```mlir
    %0 = arm_sme.zero : vector<[16]x[16]xi8>
    ```

    Example 2: Zero a 64-bit element ZA tile.

    ```mlir
    %0 = arm_sme.zero : vector<[2]x[2]xi64>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getRes().getType());
    }
  }];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def TileLoadOp : ArmSME_Op<"tile_load", [
  AttrSizedOperandSegments,
  TypesMatchWith<
    "padding type matches element type of result (if present)",
    "result", "padding",
    "::llvm::cast<VectorType>($_self).getElementType()",
    "!getPadding() || std::equal_to<>()"
  >,
  TypesMatchWith<
    "mask has i1 element type and same shape as result (if present)",
    "result", "mask",
    "VectorType("
      "VectorType::Builder("
        "::llvm::cast<mlir::VectorType>($_self)"
      ").setElementType(IntegerType::get($_self.getContext(), 1)))",
    "!getMask() || std::equal_to<>()"
  >
]> {
  let summary = "Tile load operation";
  let description = [{
    Loads a 2D SME "virtual tile" from memory defined by a base and indices,
    with the shape defined by the 2D scalable vector type of the result tile.
    An optional tile slice layout attribute specifies whether the slices of the
    tile being loaded are horizontal (default) or vertical. The slice of memory
    must be contiguous. The memref must be either rank 1 or rank 2 with dynamic
    dimensions, since the operation is scalable, and the element type must be a
    scalar that matches the element type of the result.

    An optional SSA value `padding` of the same elemental type as the MemRef is
    provided to specify a fallback value in the case of masking.

    An optional SSA value `mask` may be specified to mask out elements read
    from the MemRef. The `mask` type is an `i1` vector with a shape that
    matches how elements are read from the MemRef. Elements whose corresponding
    mask element is `0` are masked out and replaced with `padding`.

    If either `padding` or `mask` are specified, both must be specified.

    Example 1: Load an 8-bit element ZA tile with horizontal layout (default) from memory (ZA0.B).
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
    ```

    Example 2: Load a FP 32-bit element ZA tile with vertical layout from memory.
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0] layout<vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
    ```

    Example 3: Load a 128-bit element ZA tile with horizontal layout (default) from memory.
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0] layout<horizontal> : memref<?x?xi128>, vector<[1]x[1]xi128>
    ```

    Example 4: Masked load of int 32-bit element ZA tile with horizontal layout (default) from memory.
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0], %pad, %mask : memref<?x?xf32>, vector<[4]x[4]xf32>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
    Variadic<Index>:$indices,
    Optional<AnyType>:$padding, Optional<AnyVector>:$mask,
    ArmSME_TileSliceLayoutAttr:$layout
  );
  let results = (outs SMETile:$result);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];

  let builders = [
    OpBuilder<(ins "VectorType":$resultType, "Value":$base,
                   "ValueRange":$indices, "TileSliceLayout":$layout), [{
      build($_builder, $_state, resultType, base, indices, {}, {}, layout);
    }]>,
    OpBuilder<(ins "VectorType":$resultType, "Value":$base,
                   "ValueRange":$indices), [{
      build($_builder, $_state, resultType, base, indices, {}, {}, {});
    }]>,
  ];

  let assemblyFormat =
    "$base `[` $indices `]` (`,` $padding `,` $mask^)? (`layout` `` $layout^)?"
      "attr-dict `:` type($base) `,` type($result)";
}

def TileStoreOp : ArmSME_Op<"tile_store"> {
  let summary = "Tile store operation";
  let description = [{
    Stores a 2D SME "virtual tile" to memory defined by a base and indices,
    with the shape defined by the 2D scalable vector type of the tile being
    stored. An optional tile slice layout attribute specifies whether the
    slices of the tile being stored are horizontal (default) or vertical. The
    slice of memory must be contiguous. The memref must be either rank 1 or
    rank 2 with dynamic dimensions, since the operation is scalable, and the
    element type must be a scalar that matches the element type of the result.

    Example 1: Store an 8-bit element ZA tile with horizontal (default) layout to memory (ZA0.B).
    ```mlir
    arm_sme.tile_store %tile, %base[%c0, %c0] : vector<[16]x[16]xi8>, memref<?x?xi8>
    ```

    Example 2: Store a FP 32-bit element ZA tile with vertical layout to memory.
    ```mlir
    arm_sme.tile_store %tile, %base[%c0, %c0] layout<vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
    ```

    Example 3: Store a 128-bit element ZA tile with horizontal (default) layout to memory.
    ```mlir
    arm_sme.tile_store %tile, %base[%c0, %c0] layout<horizontal> : vector<[1]x[1]xi128>, memref<?x?xi128>
    ```
  }];
  let arguments = (ins SMETile:$valueToStore,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices, ArmSME_TileSliceLayoutAttr:$layout
  );
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
  }];

  let assemblyFormat =
    "$valueToStore `,` $base `[` $indices `]` (`layout` `` $layout^)? attr-dict "
      "`:` type($base) `,` type($valueToStore)";
}

def LoadTileSliceOp : ArmSME_Op<"load_tile_slice", [
    AllTypesMatch<["tile", "result"]>,
    TypesMatchWith<
      "mask has i1 element type and same shape as result",
      "result", "mask",
      "VectorType("
        "VectorType::Builder("
          "::llvm::cast<mlir::VectorType>($_self)"
        ").dropDim(0).setElementType(IntegerType::get($_self.getContext(), 1))"
      ")">,
]> {
  let summary = "Tile slice load and update operation";
  let description = [{
    Loads a 1D tile slice from memory into a 2D SME "virtual tile". The tile
    slice is defined by the dimension of the 2D scalable vector type pointed by
    the index. A tile slice index describes where in the input tile the tile
    slice is loaded to. An optional tile slice layout attribute specifies
    whether the tile slice being loaded at the given index is horizontal
    (default) or vertical. The updated tile is returned as the result.

    The slice of memory read is defined by a base and indices and must be
    contiguous. The memref must be either rank 1 or rank 2, have dynamic
    dimensions since the operation is scalable, and the element type must be a
    scalar that matches the element type of the result.

    An SSA value `mask` specifies to mask out elements read from the MemRef.
    The `mask` type is an `i1` vector with a shape that matches how elements
    are read from the MemRef.

    Example 1: Load a vector<[16]xi8> tile slice from memory into tile horizontally (default) at given index.
    ```mlir
    %tile_update = arm_sme.load_tile_slice %base[%c0], %mask, %tile, %tile_slice_index : memref<?x?xi8>, vector<[16]xi1>, vector<[16]x[16]xi8>
    ```

    Example 2: Load a vector<[4]xf32> tile slice from memory into tile vertically at given index.
    ```mlir
    %tile_update = arm_sme.load_tile_slice %base[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xf32>, vector<[4]xi1>, vector<[4]x[4]xf32>
    ```

    Example 3: Load a vector<[1]xi128> tile slice from memory into tile vertically at given index.
    ```mlir
    %tile_update = arm_sme.load_tile_slice %base[%c0], %mask, %tile, %tile_slice_index layout<vertical> : memref<?x?xi128>, vector<[1]xi1>, vector<[1]x[1]xi128>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from">:$base, AnyVector:$mask,
    SMETile:$tile, Variadic<Index>:$indices, Index:$tile_slice_index,
    ArmSME_TileSliceLayoutAttr:$layout
  );
  let results = (outs SMETile:$result);

  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` `,` $mask `,` $tile `,` $tile_slice_index
      (`layout` `` $layout^)? attr-dict `:` type($base) `,` type($mask) `,`
                                            type($result)
  }];
}

def StoreTileSliceOp : ArmSME_Op<"store_tile_slice"> {
  let summary = "Tile slice store operation";
  let description = [{
    Stores a 1D tile slice from a 2D SME "virtual tile" into memory. The tile
    slice is defined by the dimension of the 2D scalable vector type pointed by
    the index. A tile slice index describes where in the input tile the tile
    slice is stored from. An optional tile slice layout attribute specifies
    whether the tile slice being stored from the given index is horizontal
    (default) or vertical.

    The slice of memory written is defined by a base and indices and must be
    contiguous. The memref must be either rank 1 or rank 2, have dynamic
    dimensions since the operation is scalable, and the element type must be a
    scalar that matches the element type of the input tile.

    Example 1: Store vector<[16]xi8> horizontal (default) tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %base[%c0] : vector<[16]x[16]xi8>, memref<?x?xi8>
    ```

    Example 2: Store vector<[4]xf32> vertical tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %base[%c0] layout<vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
    ```

    Example 3: Store a vector<[1]xi128> vertical tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %base[%c0] layout<vertical> : vector<[1]x[1]xi128>, memref<?x?xi128>
    ```
  }];
  let arguments = (ins SMETile:$tile, Index:$tile_slice_index,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices, ArmSME_TileSliceLayoutAttr:$layout
  );
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getTile().getType());
    }
  }];

  let assemblyFormat = [{
    $tile `,` $tile_slice_index `,` $base `[` $indices `]` (`layout` `` $layout^)?
      attr-dict `:` type($base) `,` type($tile)
  }];
}

def MoveVectorToTileSliceOp : ArmSME_Op<"move_vector_to_tile_slice", [
    AllTypesMatch<["tile", "result"]>,
    TypesMatchWith<
      "type of 'vector' matches type of 'tile' slice",
      "tile", "vector",
      "VectorType::get("
        "::llvm::cast<mlir::VectorType>($_self).getShape().drop_front(),"
        "::llvm::cast<mlir::VectorType>($_self).getElementType(),"
        "/*scalableDims=*/{true})">,
]> {
  let summary = "Move 1-D scalable vector to slice of 2-D tile";
  let description = [{
    The vector to tile slice operation moves a 1-D scalable vector to a slice
    of a 2-D scalable vector tile at the given index. The type of the 1-D
    scalable vector to be moved must match the type of the tile slice. A tile
    slice is a 1-D vector of horizontally or vertically contiguous elements
    within a ZA tile. Horizontal tile slices are currently assumed when
    lowering to intrinsics. The updated tile is returned as the result.

    Example 1: Move a vector<[16]xi8> into tile at given index.
    ```mlir
    %tile_update = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[16]xi8> into vector<[16]x[16]xi8>
    ```

    Example 2: Move a vector<[2]xf64> into tile at given index.
    ```mlir
    %tile_update = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[2]xf64> into vector<[2]x[2]xf64>
    ```
  }];
  let arguments = (ins
      SVEVector:$vector, SMETile:$tile, Index:$tile_slice_index);
  let results = (outs SMETile:$result);

  let extraClassDeclaration = [{
    VectorType getTileType() {
      return ::llvm::cast<VectorType>(getTile().getType());
    }
  }];

  let assemblyFormat = [{
    $vector `,` $tile `,` $tile_slice_index
      attr-dict `:` type($vector) `into` type($result)
  }];
}

def MoveTileSliceToVectorOp : ArmSME_Op<"move_tile_slice_to_vector", [Pure,
    TypesMatchWith<
      "type of 'result' matches type of 'tile' slice",
      "tile", "result",
      "VectorType(VectorType::Builder(::llvm::cast<mlir::VectorType>($_self)).dropDim(0))">,
]> {
  let summary = "Move slice of a 2-D tile to a 1-D scalable vector";
  let description = [{
    The tile slice to vector operation extracts a 1-D scalable slice from a 2-D
    scalable tile at the given index. A tile slice is a 1-D vector of
    horizontally or vertically contiguous elements within a ZA tile. Horizontal
    tile slices are currently assumed when lowering to intrinsics.

    Example 1: Extract `vector<[16]xi8>` from tile at the given index.
    ```mlir
    %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[16]xi8> from vector<[16]x[16]xi8>
    ```

    Example 2: Extract `vector<[2]xf64>` from tile at the given index.
    ```mlir
    %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[2]xf64> from vector<[2]x[2]xf64>
    ```
  }];

  let arguments = (ins SMETile:$tile, Index:$tile_slice_index);
  let results = (outs SVEVector:$result);

  let extraClassDeclaration = [{
    VectorType getSliceType() { return getResult().getType(); }
  }];

  let assemblyFormat = [{
      $tile `[` $tile_slice_index `]` attr-dict
      `:` type($result) `from` type($tile)
  }];
}

#endif // ARMSME_OPS
