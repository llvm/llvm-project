//===-- ArmSME.td - ArmSME dialect operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ArmSME dialect and contains intrinsic ops to lower to
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef ARMSME_OPS
#define ARMSME_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// ArmSME dialect definition
//===----------------------------------------------------------------------===//

def ArmSME_Dialect : Dialect {
  let name = "arm_sme";
  let cppNamespace = "::mlir::arm_sme";
  let summary = "Basic dialect to target Arm SME architectures";
  let description = [{
    This dialect contains the definitions necessary to target Arm SME
    scalable matrix operations.

    Sources:
    https://developer.arm.com/documentation/ddi0616
    https://developer.arm.com/documentation/ddi0602/2023-03/SME-Instructions
  }];
  let dependentDialects = ["scf::SCFDialect", "vector::VectorDialect",
                           "memref::MemRefDialect"];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// ArmSME type definitions
//===----------------------------------------------------------------------===//

class SMETileType<Type datatype, list<int> dims, string description>
  : ShapedContainerType<[datatype],
      And<[IsVectorOfRankPred<[2]>, allDimsScalableVectorTypePred,
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

def SVEVector : ScalableVectorOfRankAndLengthAndType<
  [1], [16, 8, 4, 2, 1], [I8, I16, I32, I64, I128, F16, BF16, F32, F64]>;

def SVEPredicate : ScalableVectorOfRankAndLengthAndType<
  [1], [16, 8, 4, 2, 1], [I1]>;

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

def TileLoadOp : ArmSME_Op<"tile_load"> {
  let summary = "Tile load operation";
  let description = [{
    Loads a 2D SME "virtual tile" from memory defined by a base and indices,
    with the shape defined by the 2D scalable vector type of the result tile.
    An optional tile slice layout attribute specifies whether the slices of the
    tile being loaded are horizontal (default) or vertical. The slice of memory
    must be contiguous. The memref must be either rank 1 or rank 2 with dynamic
    dimensions, since the operation is scalable, and the element type must be a
    scalar that matches the element type of the result.

    Example 1: Load an 8-bit element ZA tile with horizontal layout (default) from memory (ZA0.B).
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0] : memref<?x?xi8>, vector<[16]x[16]xi8>
    ```

    Example 2: Load a FP 32-bit element ZA tile with vertical layout from memory.
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0], <vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
    ```

    Example 3: Load a 128-bit element ZA tile with horizontal layout (default) from memory.
    ```mlir
    %tile = arm_sme.tile_load %base[%c0, %c0], <horizontal> : memref<?x?xi128>, vector<[1]x[1]xi128>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base,
    Variadic<Index>:$indices,
    DefaultValuedAttr<ArmSME_TileSliceLayoutAttr,
                      "::mlir::arm_sme::TileSliceLayout::Horizontal">:$layout
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

  let assemblyFormat =
    "$base `[` $indices `]` (`,` $layout^)? attr-dict "
      "`:` type($base) `,` type($result)";
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
    arm_sme.tile_store %tile, %base[%c0, %c0], <vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
    ```

    Example 3: Store a 128-bit element ZA tile with horizontal (default) layout to memory.
    ```mlir
    arm_sme.tile_store %tile, %base[%c0, %c0], <horizontal> : vector<[1]x[1]xi128>, memref<?x?xi128>
    ```
  }];
  let arguments = (ins SMETile:$valueToStore,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices,
    DefaultValuedAttr<ArmSME_TileSliceLayoutAttr,
                      "::mlir::arm_sme::TileSliceLayout::Horizontal">:$layout
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
    "$valueToStore `,` $base `[` $indices `]` (`,` $layout^)? attr-dict "
      "`:` type($base) `,` type($valueToStore)";
}

def LoadTileSliceOp : ArmSME_Op<"load_tile_slice", [
    AllTypesMatch<["tile", "result"]>
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

    Example 1: Load a vector<[16]xi8> tile slice from memory into tile horizontally (default) at given index.
    ```mlir
    %tile_update = arm_sme.load_tile_slice %base[%c0], %tile, %tile_slice_index : memref<?x?xi8>, vector<[16]x[16]xi8>
    ```

    Example 2: Load a vector<[4]xf32> tile slice from memory into tile vertically at given index.
    ```mlir
    %tile_update = arm_sme.load_tile_slice %base[%c0], %tile, %tile_slice_index, <vertical> : memref<?x?xf32>, vector<[4]x[4]xf32>
    ```

    Example 3: Load a vector<[1]xi128> tile slice from memory into tile vertically at given index.
    ```mlir
    %tile_update = arm_sme.load_tile_slice %base[%c0], %tile, %tile_slice_index, <vertical> : memref<?x?xi128>, vector<[1]x[1]xi128>
    ```
  }];
  let arguments = (ins
    Arg<AnyMemRef, "the reference to load from">:$base,
    SMETile:$tile, Variadic<Index>:$indices, Index:$tile_slice_index,
    DefaultValuedAttr<ArmSME_TileSliceLayoutAttr,
                      "::mlir::arm_sme::TileSliceLayout::Horizontal">:$layout
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
    $base `[` $indices `]` `,` $tile `,` $tile_slice_index (`,` $layout^)?
      attr-dict `:` type($base) `,` type($result)
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
    arm_sme.store_tile_slice %tile, %tile_slice_index, %base[%c0], <vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
    ```

    Example 3: Store a vector<[1]xi128> vertical tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %base[%c0], <vertical> : vector<[1]x[1]xi128>, memref<?x?xi128>
    ```
  }];
  let arguments = (ins SMETile:$tile, Index:$tile_slice_index,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices,
    DefaultValuedAttr<ArmSME_TileSliceLayoutAttr,
                      "::mlir::arm_sme::TileSliceLayout::Horizontal">:$layout
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
    $tile `,` $tile_slice_index `,` $base `[` $indices `]` (`,` $layout^)?
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

//===----------------------------------------------------------------------===//
// ArmSME Intrinsic op definitions
//===----------------------------------------------------------------------===//

def MOPPredicate : ScalableVectorOfLengthAndType<[16, 8, 4, 2], [I1]>;
def MOPVector : ScalableVectorOfLengthAndType<[16, 8, 4, 2],
                                              [I8, I16, BF16, F16, F32, F64]>;
def LDSTPredicate : ScalableVectorOfLengthAndType<[16, 8, 4, 2, 1], [I1]>;

class ArmSME_IntrOp<string mnemonic, list<int> overloadedOperands = [],
                    list<Trait> traits = [], int numResults = 0,
                    list<int> overloadedResults = []>
    : LLVM_IntrOpBase<
          /*Dialect dialect=*/ArmSME_Dialect,
          /*string opName=*/"intr." # mnemonic,
          /*string enumName=*/"aarch64_sme_" # !subst(".", "_", mnemonic),
          /*list<int> overloadedResults=*/overloadedResults,
          /*list<int> overloadedOperands=*/overloadedOperands,
          /*list<Trait> traits=*/traits,
          /*int numResults=*/numResults>;

// Zero
def LLVM_aarch64_sme_zero : ArmSME_IntrOp<"zero">,
                            Arguments<(ins Arg<I32, "Tile mask">)>;

// MOP's
class ArmSME_IntrMopOverloadedOp<string mnemonic>
    : ArmSME_IntrOp<mnemonic, [4]>,
      Arguments<(ins Arg<I32, "Virtual tile ID">,
                 Arg<MOPPredicate, "LHS predicate">,
                 Arg<MOPPredicate, "RHS predicate">,
                 Arg<MOPVector, "LHS vector operand">,
                 Arg<MOPVector, "RHS vector operand">)>;

def LLVM_aarch64_sme_mopa : ArmSME_IntrMopOverloadedOp<"mopa">;
def LLVM_aarch64_sme_mops : ArmSME_IntrMopOverloadedOp<"mops">;
def LLVM_aarch64_sme_mopa_wide : ArmSME_IntrMopOverloadedOp<"mopa.wide">;
def LLVM_aarch64_sme_mops_wide : ArmSME_IntrMopOverloadedOp<"mops.wide">;
def LLVM_aarch64_sme_smopa_wide : ArmSME_IntrMopOverloadedOp<"smopa.wide">;
def LLVM_aarch64_sme_smops_wide : ArmSME_IntrMopOverloadedOp<"smops.wide">;
def LLVM_aarch64_sme_umopa_wide : ArmSME_IntrMopOverloadedOp<"umopa.wide">;
def LLVM_aarch64_sme_umops_wide : ArmSME_IntrMopOverloadedOp<"umops.wide">;
def LLVM_aarch64_sme_sumopa_wide : ArmSME_IntrMopOverloadedOp<"sumopa.wide">;
def LLVM_aarch64_sme_sumops_wide : ArmSME_IntrMopOverloadedOp<"sumops.wide">;
def LLVM_aarch64_sme_usmopa_wide : ArmSME_IntrMopOverloadedOp<"usmopa.wide">;
def LLVM_aarch64_sme_usmops_wide : ArmSME_IntrMopOverloadedOp<"usmops.wide">;

// Loads
class ArmSME_IntrLoadOp<string mnemonic>
    : ArmSME_IntrOp<mnemonic>,
      Arguments<(ins Arg<LDSTPredicate, "Vector predicate">,
                 Arg<LLVM_AnyPointer, "Load address">,
                 Arg<I32, "Virtual tile ID">,
                 Arg<I32, "Tile slice">)>;

def LLVM_aarch64_sme_ld1b_horiz : ArmSME_IntrLoadOp<"ld1b.horiz">;
def LLVM_aarch64_sme_ld1h_horiz : ArmSME_IntrLoadOp<"ld1h.horiz">;
def LLVM_aarch64_sme_ld1w_horiz : ArmSME_IntrLoadOp<"ld1w.horiz">;
def LLVM_aarch64_sme_ld1d_horiz : ArmSME_IntrLoadOp<"ld1d.horiz">;
def LLVM_aarch64_sme_ld1q_horiz : ArmSME_IntrLoadOp<"ld1q.horiz">;
def LLVM_aarch64_sme_ld1b_vert : ArmSME_IntrLoadOp<"ld1b.vert">;
def LLVM_aarch64_sme_ld1h_vert : ArmSME_IntrLoadOp<"ld1h.vert">;
def LLVM_aarch64_sme_ld1w_vert : ArmSME_IntrLoadOp<"ld1w.vert">;
def LLVM_aarch64_sme_ld1d_vert : ArmSME_IntrLoadOp<"ld1d.vert">;
def LLVM_aarch64_sme_ld1q_vert : ArmSME_IntrLoadOp<"ld1q.vert">;

// Stores
class ArmSME_IntrStoreOp<string mnemonic>
    : ArmSME_IntrOp<mnemonic>,
      Arguments<(ins Arg<LDSTPredicate, "Vector predicate">,
                 Arg<LLVM_AnyPointer, "Store address", [MemWrite]>,
                 Arg<I32, "Virtual tile ID">,
                 Arg<I32, "Tile slice">)>;

def LLVM_aarch64_sme_st1b_horiz : ArmSME_IntrStoreOp<"st1b.horiz">;
def LLVM_aarch64_sme_st1h_horiz : ArmSME_IntrStoreOp<"st1h.horiz">;
def LLVM_aarch64_sme_st1w_horiz : ArmSME_IntrStoreOp<"st1w.horiz">;
def LLVM_aarch64_sme_st1d_horiz : ArmSME_IntrStoreOp<"st1d.horiz">;
def LLVM_aarch64_sme_st1q_horiz : ArmSME_IntrStoreOp<"st1q.horiz">;
def LLVM_aarch64_sme_st1b_vert : ArmSME_IntrStoreOp<"st1b.vert">;
def LLVM_aarch64_sme_st1h_vert : ArmSME_IntrStoreOp<"st1h.vert">;
def LLVM_aarch64_sme_st1w_vert : ArmSME_IntrStoreOp<"st1w.vert">;
def LLVM_aarch64_sme_st1d_vert : ArmSME_IntrStoreOp<"st1d.vert">;
def LLVM_aarch64_sme_st1q_vert : ArmSME_IntrStoreOp<"st1q.vert">;

def LLVM_aarch64_sme_str
    : ArmSME_IntrOp<"str">,
      Arguments<(ins Arg<I32, "Index">,
                 Arg<LLVM_AnyPointer, "Store address", [MemWrite]>)>;

// Vector to tile slice
class LLVM_aarch64_sme_write<string direction>
    : ArmSME_IntrOp<"write." # direction, /*overloadedOperands=*/[3],
                    [AllShapesMatch<["pg", "vector"]>]>,
      Arguments<(ins Arg<I32, "Virtual tile ID">,
                     Arg<I32, "Tile slice">,
                     Arg<SVEPredicate, "Vector predicate">:$pg,
                     Arg<SVEVector, "Vector operand">:$vector)>;

// Tile slice to vector
class LLVM_aarch64_sme_read<string direction>
    : ArmSME_IntrOp<"read." # direction, /*overloadedOperands=*/[],
                    [AllShapesMatch<["vector", "pg", "res"]>,
                     AllElementTypesMatch<["vector", "res"]>],
                    /*numResults=*/1, /*overloadedResults=*/[0]>,
      Arguments<(ins Arg<SVEVector, "Vector operand">:$vector,
                     Arg<SVEPredicate, "Vector predicate">:$pg,
                     Arg<I32, "Virtual tile ID">,
                     Arg<I32, "Tile slice">)>;

def LLVM_aarch64_sme_write_horiz : LLVM_aarch64_sme_write<"horiz">;
def LLVM_aarch64_sme_write_vert : LLVM_aarch64_sme_write<"vert">;

def LLVM_aarch64_sme_read_horiz : LLVM_aarch64_sme_read<"horiz">;
def LLVM_aarch64_sme_read_vert : LLVM_aarch64_sme_read<"vert">;

def LLVM_aarch64_sme_za_enable : ArmSME_IntrOp<"za.enable">;
def LLVM_aarch64_sme_za_disable : ArmSME_IntrOp<"za.disable">;

#endif // ARMSME_OPS
