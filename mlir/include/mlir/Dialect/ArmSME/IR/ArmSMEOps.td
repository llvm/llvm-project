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
// ArmSME op interfaces
//===----------------------------------------------------------------------===//

def ArmSMETileType : I32EnumAttr<"ArmSMETileType", "Arm SME tile type",
    [
      I32EnumAttrCase<"ZAB", 0, "za.b">,
      I32EnumAttrCase<"ZAH", 1, "za.h">,
      I32EnumAttrCase<"ZAS", 2, "za.s">,
      I32EnumAttrCase<"ZAD", 3, "za.d">,
      I32EnumAttrCase<"ZAQ", 4, "za.q">,
    ]>{
  let cppNamespace = "mlir::arm_sme";
  let genSpecializedAttr = 0;
}

def ArmSMETileOpInterface : OpInterface<"ArmSMETileOpInterface"> {
  let description = [{
    An interface for operations that use Arm SME tiles. These operations need to
    be assigned a tile ID, an i32 attribute, which specifies which virtual tile
    within the ZA storage to use. The number of tiles available depends on the
    type of the tile. This is summarized below:

    | Tile Vector Types                                                       | Possible Tile IDs   |
    |-------------------------------------------------------------------------|---------------------|
    | `vector<[16]x[16]xi8>`                                                  | 0                   |
    | `vector<[8]x[8]xi16>`, `vector<[8]x[8]xf16>`, or `vector<[8]x[8]xbf16>` | 0 and 1             |
    | `vector<[4]x[4]xi32>` or `vector<[4]x[4]xf32>`                          | 0 to 3 (inclusive)  |
    | `vector<[2]x[2]xi64>` or `vector<[2]x[2]xf64>`                          | 0 to 7 (inclusive)  |
    | `vector<[1]x[1]xi128>`                                                  | 0 to 15 (inclusive) |
  }];
  let methods = [
    InterfaceMethod<
      "Sets the tile ID for this operation.",
      /*returnType=*/"void",
      /*methodName=*/"setTileId",
      /*arguments=*/(ins "mlir::IntegerAttr":$tileId),
      /*methodBody=*/[{}],
      /*defaultImpl=*/ [{
        if (!tileId)
          return;
        ::mlir::Operation* op = this->getOperation();
        op->setAttr("tile_id", tileId);
      }]
    >,
    InterfaceMethod<
      [{
        Returns the tile ID assigned to this operation. This will be null before
        tile allocation.
      }],
      /*returnType=*/"mlir::IntegerAttr",
      /*methodName=*/"getTileId",
      /*arguments=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImpl=*/ [{
        ::mlir::Operation* op = this->getOperation();
        return op->getAttrOfType<mlir::IntegerAttr>("tile_id");
      }]
    >,
    InterfaceMethod<
      "Returns the VectorType of the tile used by this operation.",
      /*returnType=*/"VectorType",
      /*methodName=*/"getTileType"
    >
  ];

  let extraSharedClassDeclaration = [{
    bool isInMemoryTile() {
      auto tileId = getTileId();
      return tileId && tileId.getInt() >= kInMemoryTileIdBase;
    }
  }];

  let verify = [{ return detail::verifyArmSMETileOpInterface($_op); }];
}

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
                         nxnxv8f16, nxnxv8bf16, nxnxv4f32, nxnxv2f64],
                        "a vector type that fits into a SME tile",
                        "VectorType">
{
  let description = [{
    Possible vector types:

    Integer elements:

    * `vector<[16]x[16]xi8>`
    * `vector<[8]x[8]xi16>`
    * `vector<[4]x[4]xi32>`
    * `vector<[2]x[2]xi64>`
    * `vector<[1]x[1]xi128>`

    Floating point elements:

    * `vector<[8]x[8]xf16>`
    * `vector<[8]x[8]xbf16>`
    * `vector<[4]x[4]xf32>`
    * `vector<[2]x[2]xf64>`
  }];
}

class HasMatchingMaskTypeConstraint<string vector, string mask> :
  OptionalTypesMatchWith<
    mask # " has i1 element type and same shape as " # vector,
    vector, mask,
    "::llvm::cast<mlir::VectorType>($_self).cloneWith({}, IntegerType::get($_ctxt, 1))">;

class TileSliceMaskConstraint<string tile, string mask> :
  TypesMatchWith<
    "`" # mask # "` has i1 element type and the shape is a slice of `" # tile # "`",
    tile, mask,
    "VectorType("
      "VectorType::Builder("
        "::llvm::cast<mlir::VectorType>($_self)"
      ").dropDim(0).setElementType(IntegerType::get($_self.getContext(), 1)))">;

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

def CombiningKind : I32EnumAttr<"CombiningKind", "Kind of combining function", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
]> {
  let cppNamespace = "::mlir::arm_sme";
  let genSpecializedAttr = 0;
}

/// An attribute that specifies how to combine a newly produced value with the
/// accumulator. This is similar to vector::CombiningKindAttr, but limited to
/// the functions that are valid for SME outer products. Add corresponds to a
/// MOPA and sub to a MOPS.
/// E.g. For f32:
/// FMOPA: https://developer.arm.com/documentation/ddi0602/2022-03/SME-Instructions/FMOPA--non-widening---Floating-point-outer-product-and-accumulate-
/// FMOPS: https://developer.arm.com/documentation/ddi0602/2022-03/SME-Instructions/FMOPS--non-widening---Floating-point-outer-product-and-subtract-
def ArmSME_CombiningKindAttr : EnumAttr<ArmSME_Dialect, CombiningKind,
                                          "kind"> {
  let assemblyFormat = "`<` $value `>`";
  let defaultValue = "CombiningKind::Add";
}

def TypeSize : I32EnumAttr<"TypeSize", "Size of a vector element type", [
  I32EnumAttrCase<"Byte"  , 0, "byte">,
  I32EnumAttrCase<"Half"  , 1, "half">,
  I32EnumAttrCase<"Word"  , 2, "word">,
  I32EnumAttrCase<"Double", 3, "double">,
]> {
  let cppNamespace = "::mlir::arm_sme";
  let genSpecializedAttr = 0;
}

def ArmSME_TypeSizeAttr : EnumAttr<ArmSME_Dialect, TypeSize,
                                   "type_size"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ArmSME op definitions
//===----------------------------------------------------------------------===//

class ArmSME_Op<string mnemonic, list<Trait> traits = []> :
  Op<ArmSME_Dialect, mnemonic, traits> {}

def GetTileOp : ArmSME_Op<"get_tile", [ArmSMETileOpInterface, Pure]> {
  let summary = "Creates an undefined value of SME virtual tile type";
  let description = [{
    Creates a new SME "virtual tile" value within a function. The contents of
    the tile returned from this operation are undefined.

    Example 1:

    ```mlir
    // Create an 8-bit element "virtual tile" value:
    %za0_b = arm_sme.get_tile: vector<[16]x[16]xi8>
    ```

    Example 2:

    ```mlir
    // Create two 16-bit element "virtual tiles" values:
    %za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
    %za1_h = arm_sme.get_tile : vector<[8]x[8]xi16>
    ```

    Example 3:
    ```mlir
    // Create an 128-bit element "virtual tile" value:
    %za0_q = arm_sme.get_tile : vector<[1]x[1]xi128>
    ```
  }];

  let results = (outs SMETile:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";

  let extraClassDeclaration = [{
    VectorType getTileType() {
      return ::llvm::cast<VectorType>(getTile().getType());
    }
  }];
}

def ZeroOp : ArmSME_Op<"zero", [ArmSMETileOpInterface, Pure]> {
  let summary = "Creates a zero-initialized value of SME virtual tile type";
  let results = (outs SMETile:$res);
  let description = [{
    Creates a new SME "virtual tile" value within a function. The contents of
    the tile returned from this operation are zero-initialized.

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
    VectorType getTileType() {
      return getVectorType();
    }
  }];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def CopyTileOp : ArmSME_Op<"copy_tile", [
  Pure,
  ArmSMETileOpInterface,
  AllTypesMatch<["tile", "result"]>
]> {
  let summary = "Copies an SME tile value";
  let arguments = (ins SMETile:$tile);
  let results = (outs SMETile:$result);
  let description = [{
    Copies an SME "virtual tile" value to a new SSA value. This operation is
    primarily intended to be used to normalize the IR prior to tile allocation.

    Example:

    ```mlir
    %copy = arm_sme.copy_tile %tile : vector<[4]x[4]xf32>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getTileType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let assemblyFormat = "$tile attr-dict `:` type($result)";
}

def TileLoadOp : ArmSME_Op<"tile_load", [
  ArmSMETileOpInterface,
  AttrSizedOperandSegments,
  OptionalTypesMatchWith<
    "padding type matches element type of result",
    "result", "padding",
    "::llvm::cast<VectorType>($_self).getElementType()"
  >,
  HasMatchingMaskTypeConstraint<"result", "mask">,
  PredOpTrait<
    "both `padding` and `mask` should be provided or neither",
    CPred<"bool(getPadding()) == bool(getMask())">
  >,
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
    VectorType getTileType() {
      return getVectorType();
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

def TileStoreOp : ArmSME_Op<"tile_store", [
  ArmSMETileOpInterface,
  AttrSizedOperandSegments,
  HasMatchingMaskTypeConstraint<"valueToStore", "mask">,
]> {
  let summary = "Tile store operation";
  let description = [{
    Stores a 2D SME "virtual tile" to memory defined by a base and indices,
    with the shape defined by the 2D scalable vector type of the tile being
    stored. An optional tile slice layout attribute specifies whether the
    slices of the tile being stored are horizontal (default) or vertical. The
    slice of memory must be contiguous. The memref must be either rank 1 or
    rank 2 with dynamic dimensions, since the operation is scalable, and the
    element type must be a scalar that matches the element type of the result.

    An optional `mask` may be provided, the shape of which corresponds to the
    `tile`, and selects which elements of the tile will be stored.

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

    Example 4: Masked store a int 32-bit element ZA tile with vertical layout to memory.
    ```mlir
    arm_sme.tile_store %tile, %base[%c0, %c0], %mask layout<vertical> : vector<[4]x[4]xf32>, memref<?x?xf32>
    ```
  }];
  let arguments = (ins SMETile:$valueToStore,
    Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base,
    Variadic<Index>:$indices, Optional<AnyVector>:$mask,
    ArmSME_TileSliceLayoutAttr:$layout
  );
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    VectorType getVectorType() {
      return ::llvm::cast<VectorType>(getValueToStore().getType());
    }
    VectorType getTileType() {
      return getVectorType();
    }
  }];

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$base,
                   "ValueRange":$indices), [{
      build($_builder, $_state, valueToStore, base, indices, {});
    }]>,
  ];

  let assemblyFormat =
    "$valueToStore `,` $base `[` $indices `]` (`,` $mask^)? (`layout` `` $layout^)?"
      "attr-dict `:` type($base) `,` type($valueToStore)";
}

def LoadTileSliceOp : ArmSME_Op<"load_tile_slice", [
  ArmSMETileOpInterface,
  AllTypesMatch<["tile", "result"]>, TileSliceMaskConstraint<"result", "mask">
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

    The provided `mask` is used to specify which elements of the tile slice
    will be loaded.

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
    Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base, SVEPredicate:$mask,
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
    VectorType getTileType() {
      return getVectorType();
    }
  }];

  let assemblyFormat = [{
    $base `[` $indices `]` `,` $mask `,` $tile `,` $tile_slice_index
      (`layout` `` $layout^)? attr-dict `:` type($base) `,` type($mask) `,`
                                            type($result)
  }];
}

def StoreTileSliceOp : ArmSME_Op<"store_tile_slice", [
  ArmSMETileOpInterface,
  TileSliceMaskConstraint<"tile", "mask">
]> {
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

    The provided `mask` is used to specify which elements of the tile slice
    will be stored.

    Example 1: Store vector<[16]xi8> horizontal (default) tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %base[%c0] : vector<[16]x[16]xi8>, vector<[16]xi1>, memref<?x?xi8>
    ```

    Example 2: Store vector<[4]xf32> vertical tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %base[%c0] layout<vertical> : vector<[4]x[4]xf32>, vector<[4]xi1>, memref<?x?xf32>
    ```

    Example 3: Store a vector<[1]xi128> vertical tile slice from tile at given index to memory.
    ```mlir
    arm_sme.store_tile_slice %tile, %tile_slice_index, %mask, %base[%c0] layout<vertical> : vector<[1]x[1]xi128>, vector<[1]xi1>, memref<?x?xi128>
    ```
  }];
  let arguments = (ins
    SMETile:$tile, Index:$tile_slice_index, SVEPredicate:$mask,
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
    VectorType getTileType() {
      return getVectorType();
    }
  }];

  let assemblyFormat = [{
    $tile `,` $tile_slice_index `,` $mask `,` $base `[` $indices `]` (`layout` `` $layout^)?
      attr-dict `:` type($base) `,` type($mask) `,` type($tile)
  }];
}

def MoveVectorToTileSliceOp : ArmSME_Op<"move_vector_to_tile_slice", [
    ArmSMETileOpInterface, Pure,
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
    within a ZA tile. The updated tile is returned as the result.

    An optional tile slice layout attribute specifies whether the tile slice is
    horizontal (default) or vertical.

    Example 1: Move a vector<[16]xi8> into tile horizontally (default) at given index.
    ```mlir
    %tile_update = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index : vector<[16]xi8> into vector<[16]x[16]xi8>
    ```

    Example 2: Move a vector<[2]xf64> into tile vertically at given index.
    ```mlir
    %tile_update = arm_sme.move_vector_to_tile_slice %vector, %tile, %tile_slice_index layout<vertical> : vector<[2]xf64> into vector<[2]x[2]xf64>
    ```
  }];
  let arguments = (ins
      SVEVector:$vector, SMETile:$tile, Index:$tile_slice_index,
      ArmSME_TileSliceLayoutAttr:$layout);
  let results = (outs SMETile:$result);

  let extraClassDeclaration = [{
    VectorType getTileType() {
      return ::llvm::cast<VectorType>(getTile().getType());
    }
  }];

  let assemblyFormat = [{
    $vector `,` $tile `,` $tile_slice_index (`layout` `` $layout^)?
      attr-dict `:` type($vector) `into` type($result)
  }];
}

def MoveTileSliceToVectorOp : ArmSME_Op<"move_tile_slice_to_vector", [
    ArmSMETileOpInterface, Pure,
    TypesMatchWith<
      "type of 'result' matches type of 'tile' slice",
      "tile", "result",
      "VectorType(VectorType::Builder(::llvm::cast<mlir::VectorType>($_self)).dropDim(0))">,
]> {
  let summary = "Move slice of a 2-D tile to a 1-D scalable vector";
  let description = [{
    The tile slice to vector operation extracts a 1-D scalable slice from a 2-D
    scalable tile at the given index. A tile slice is a 1-D vector of
    horizontally or vertically contiguous elements within a ZA tile.

    An optional tile slice layout attribute specifies whether the tile slice is
    horizontal (default) or vertical.

    Example 1: Extract `vector<[16]xi8>` from tile horizontally at the given index.
    ```mlir
    %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] : vector<[16]xi8> from vector<[16]x[16]xi8>
    ```

    Example 2: Extract `vector<[2]xf64>` from tile vertically at the given index.
    ```mlir
    %slice = arm_sme.move_tile_slice_to_vector %tile[%tile_slice_index] layout<vertical> : vector<[2]xf64> from vector<[2]x[2]xf64>
    ```
  }];

  let arguments = (ins
    SMETile:$tile, Index:$tile_slice_index,
    ArmSME_TileSliceLayoutAttr:$layout
  );
  let results = (outs SVEVector:$result);

  let extraClassDeclaration = [{
    VectorType getSliceType() { return getResult().getType(); }
    VectorType getTileType() {
      return ::llvm::cast<VectorType>(getTile().getType());
    }
  }];

  let assemblyFormat = [{
      $tile `[` $tile_slice_index `]` (`layout` `` $layout^)? attr-dict
      `:` type($result) `from` type($tile)
  }];
}

class OuterProductResultTileTypeConstraint<string operand> :
  OptionalTypesMatchWith<operand # "type is derived from `lhs` and `rhs`",
    "lhs", operand,
    "[&]{"
    "  auto vectorType = ::llvm::cast<mlir::VectorType>($_self);"
    "  int64_t size = vectorType.getDimSize(0);"
    "  return VectorType::get("
    "    { size, size }, vectorType.getElementType(), { true, true });"
    "}()">;

def OuterProductOp :
  ArmSME_Op<"outerproduct", [
    Pure,
    ArmSMETileOpInterface,
    AttrSizedOperandSegments,
    AllTypesMatch<["lhs", "rhs"]>,
    HasMatchingMaskTypeConstraint<"lhs", "lhsMask">,
    HasMatchingMaskTypeConstraint<"rhs", "rhsMask">,
    PredOpTrait<
      "both `lhsMask` and `rhsMask` should be provided or neither",
      CPred<"bool(getLhsMask()) == bool(getRhsMask())">>,
    OuterProductResultTileTypeConstraint<"result">,
    OuterProductResultTileTypeConstraint<"acc">
  ]>
{
  let summary = "Outer product with optional fused add/sub";

  let description = [{
    This operation represents an outer product that fits within an SME tile.
    All operands must be SVE vectors and the result a SME tile. Unlike
    `vector.outerproduct` masking is on the operands (rather than the result),
    which mirrors the SME instructions.

    Example 1: Unmasked outerproduct (without accumulator)
    ```mlir
    // Not specifying an accumulator implicitly zeros the destination tile.
    %result = arm_sme.outerproduct $lhs, $rhs : vector<[4]xf32>, vector<[4]xf32>
    ```

    Example 2: Unmasked outerproduct (with accumulator)
    ```mlir
    %result = arm_sme.outerproduct $lhs, $rhs acc($accumulator)
                : vector<[4]xf32>, vector<[4]xf32>
    ```

    Example 3: Masked outerproduct
    ```mlir
    %result = arm_sme.outerproduct $lhs, $rhs masks($lhsMask, $rhsMask)
                : vector<[4]xf32>, vector<[4]xf32>
    ```

    Example 4: Masked outerproduct (with accumulator)
    ```mlir
    %result = arm_sme.outerproduct $lhs, $rhs acc($accumulator) masks($lhsMask, $rhsMask)
                : vector<[4]xf32>, vector<[4]xf32>
    ```
  }];

let arguments = (ins
    SVEVector:$lhs, SVEVector:$rhs,
    Optional<SVEPredicate>:$lhsMask,
    Optional<SVEPredicate>:$rhsMask,
    Optional<SMETile>: $acc,
    ArmSME_CombiningKindAttr:$kind);
  let results = (outs SMETile:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    oilist(
        `kind` `` $kind
      | `acc` `` `(` $acc `)`
      | `masks` `` `(` $lhsMask `,` $rhsMask `)`
    ) attr-dict `:` type($lhs) `,` type($rhs)
  }];

  let extraClassDeclaration = [{
    VectorType getLhsType() { return llvm::cast<VectorType>(getLhs().getType()); }
    VectorType getRhsType() { return llvm::cast<VectorType>(getRhs().getType()); }
    VectorType getResultType() { return llvm::cast<VectorType>(getResult().getType()); }
    VectorType getTileType() {
      return getResultType();
    }
  }];
}

class OuterProductWideningBase<string mnemonic,
                               list<Type> allowedInputVectorTypes,
                               list<Type> allowedResultVectorTypes,
                               int numOuterProducts> :
  ArmSME_Op<mnemonic, [
    Pure,
    ArmSMETileOpInterface,
    AttrSizedOperandSegments,
    AllTypesMatch<["lhs", "rhs"]>,
    HasMatchingMaskTypeConstraint<"lhs", "lhsMask">,
    HasMatchingMaskTypeConstraint<"rhs", "rhsMask">,
    PredOpTrait<
      "both `lhsMask` and `rhsMask` should be provided or neither",
      CPred<"bool(getLhsMask()) == bool(getRhsMask())">
    >,
    OptionalTypesMatchWith<"`result` and `acc` have the same type",
                           "result", "acc", "::llvm::cast<Type>($_self)">,
    // This trait ensures the input types match the correct output type for ops
    // that takes multiple inputs and outputs (i.e., 4-way).
    PredOpTrait<
      "tile element size equals input element size * " # numOuterProducts,
      CPred<"getTileType().getElementTypeBitWidth() == "
            "(getLhsType().getElementTypeBitWidth() * " # numOuterProducts # ")">
    >,
  ]> {

  let arguments = (ins
    AnyTypeOf<allowedInputVectorTypes>:$lhs, AnyVector:$rhs,
    Optional<AnyVector>:$lhsMask, Optional<AnyVector>:$rhsMask,
    Optional<AnyVector>:$acc);
  let results = (outs AnyTypeOf<allowedResultVectorTypes>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs
    oilist(
        `acc` `` `(` $acc `)`
      | `masks` `` `(` $lhsMask `,` $rhsMask `)`
    ) attr-dict `:` type($lhs) `,` type($rhs) `into` type($result)
  }];

  let extraClassDeclaration = [{
    VectorType getLhsType() { return llvm::cast<VectorType>(getLhs().getType()); }
    VectorType getRhsType() { return llvm::cast<VectorType>(getRhs().getType()); }
    VectorType getResultType() { return llvm::cast<VectorType>(getResult().getType()); }
    VectorType getTileType() {
      return getResultType();
    }
  }];
}

class OuterProduct2Way<string mnemonic,
                       list<Type> allowedInputVectorTypes,
                       list<Type> allowedResultVectorTypes>
  : OuterProductWideningBase<mnemonic, allowedInputVectorTypes,
                             allowedResultVectorTypes, /*numOuterProducts=*/2>;

def FMopa2WayOp
  : OuterProduct2Way<"fmopa_2way",
      [ScalableVectorOfRankAndLengthAndType<[1], [8], [F16, BF16]>],
      [nxnxv4f32]> {
  let summary = "Floating-point sum of 2 outer products and accumulate";

  let description = [{
    This operation represents a sum of 2 widened outer products. It takes 2 1-D
    scalable vectors as input and a 2-D scalable vector (ZA tile) as output.

    For example (fp16 to fp32):

    ```mlir
    %result = arm_sme.fmopa_2way %lhs, %rhs :
      vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
    ```

    The `lhs` encodes a matrix of shape SVLSx2 and the `rhs` a matrix of
    2xSVLS, where SVLS (spec [1], section B2.1) is the number of 32-bit
    elements in a vector of SVL bits. To illustrate, below is a breakdown of
    this operation for fp16 to fp32, SVL=128 (i.e., vscale=1):

    ```
                          LHS                          RHS
               [A0 A1 A2 A3 A4 A5 A6 A7]    [B0 B1 B2 B3 B4 B5 B6 B7]

    ----------------------------------------------------------------------------

                                  implicit layout

                              [A0 A1]    |
                              [A2 A3]    |    [B0 B2 B4 B6]
                              [A4 A5]    |    [B1 B3 B5 B7]
                              [A6 A7]    |

    ----------------------------------------------------------------------------

                                  2 outer products

                      Acol0 ⊗ Brow0      |           Acol1 ⊗ Brow1
                      -------------      |           -------------
                                         |
                  [B0 B2 B4 B6]          |       [B1 B3 B5 B7]
                                         |
             [A0  [A0B0 A0B2 A0B4 A0B6]  |  [A1  [A1B1 A1B3 A1B5 A1B7]
              A2  [A2B0 A2B2 A2B4 A2B6]  |   A3  [A3B1 A3B3 A3B5 A3B7]
              A4  [A4B0 A4B2 A4B4 A4B6]  |   A5  [A5B1 A5B3 A5B5 A5B7]
              A6] [A6B0 A6B2 A6B4 A6B6]  |   A7] [A7B1 A7B3 A7B5 A7B7]
                                         |

    ----------------------------------------------------------------------------

                              sum of 2 outer products

                           Acol0 ⊗ Brow0 + Acol1 ⊗ Brow1

                 [A0B0 + A1B1 A0B2 + A1B3 A0B4 + A1B5 A0B6 + A1B7]
                 [A2B0 + A3B1 A2B2 + A3B3 A2B4 + A3B5 A2B6 + A3B7]
                 [A4B0 + A5B1 A4B2 + A5B3 A4B4 + A5B5 A4B6 + A5B7]
                 [A6B0 + A7B1 A6B2 + A7B3 A6B4 + A7B5 A6B6 + A7B7]

    ----------------------------------------------------------------------------
    ```

    This operation enables the folding of 2 outer products chained via the
    accumulator into a single outer product.

    For example:

    ```mlir
    %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
    %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
    %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
    %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

    %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
    %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
    ```

    The 2 outer products in the example above can be fused into a single outer
    product as follows:

	```mlir
    %a_packed = "llvm.intr.experimental.vector.interleave2"(%a0, %a1) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
    %b_packed = "llvm.intr.experimental.vector.interleave2"(%b0, %b1) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
    %0 = arm_sme.fmopa_2way %a_packed, %b_packed : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
	```

    This is implemented in the `-arm-sme-outer-product-fusion` pass.

    Example: FP16 to FP32
    ```mlir
    %result = arm_sme.fmopa_2way $lhs, $rhs : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
    ```

    Example: BF16 to FP32
    ```mlir
    %result = arm_sme.fmopa_2way $lhs, $rhs : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
    ```

    | Spec | Features |
    | ---- | -------- |
    | [FMOPA (widening, 2-way, FP16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/FMOPA--widening--2-way--FP16-to-FP32---Half-precision-floating-point-sum-of-outer-products-and-accumulate-) | +sme |
    | [BFMOPA (widening, 2-way, BF16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/BFMOPA--widening---BFloat16-sum-of-outer-products-and-accumulate-) | +sme |

    [1] https://developer.arm.com/documentation/ddi0616
  }];
}

// TODO: support:
// - FMOPA 2-way FP8 to FP16
// - FMOPA 4-way FP16 to FP32
// once intrinsic support lands in the backend.

def FMops2WayOp
  : OuterProduct2Way<"fmops_2way",
      [ScalableVectorOfRankAndLengthAndType<[1], [8], [F16, BF16]>],
      [nxnxv4f32]> {
  let summary = "Floating-point sum of 2 outer products and subtract";
  let description = [{
    Equivalent to `fmopa_2way` but outer products are subtracted from
    destination `result`.

    Example: FP16 to FP32
    ```mlir
    %result = arm_sme.fmops_2way $lhs, $rhs : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
    ```

    Example: BF16 to FP32
    ```mlir
    %result = arm_sme.fmops_2way $lhs, $rhs : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
    ```

    Refer to
    [fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
    description of 2-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [FMOPS (widening, 2-way, FP16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/FMOPS--widening---Half-precision-floating-point-sum-of-outer-products-and-subtract-) | +sme |
    | [BFMOPS (widening, 2-way, BF16 to FP32)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/BMOPS--Bitwise-exclusive-NOR-population-count-outer-product-and-subtract-) | +sme |
  }];
}

def SMopa2WayOp
  : OuterProduct2Way<"smopa_2way",
      [ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32]> {
  let summary = "Signed integer sum of 2 outer products and accumulate";
  let description = [{
    Example:
    ```mlir
    %result = arm_sme.smopa_2way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
    ```

    Refer to
    [fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
    description of 2-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [SMOPA (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPA--2-way---Signed-integer-sum-of-outer-products-and-accumulate-) | +sme2 |
  }];
}

def SMops2WayOp
  : OuterProduct2Way<"smops_2way",
      [ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32]> {
  let summary = "Signed integer sum of 2 outer products and subtract";
  let description = [{
    Example:
    ```mlir
    %result = arm_sme.smops_2way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
    ```

    Refer to
    [fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
    description of 2-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [SMOPS (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPS--2-way---Signed-integer-sum-of-outer-products-and-subtract-) | +sme2 |
  }];
}

def UMopa2WayOp
  : OuterProduct2Way<"umopa_2way",
      [ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32]> {
  let summary = "Unsiged integer sum of 2 outer products and accumulate";
  let description = [{
    Example:
    ```mlir
    %result = arm_sme.umopa_2way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
    ```

    Refer to
    [fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
    description of 2-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [UMOPA (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPA--2-way---Unsigned-integer-sum-of-outer-products-and-accumulate-) | +sme2 |
  }];
}

def UMops2WayOp
  : OuterProduct2Way<"umops_2way",
      [ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32]> {
  let summary = "Unsiged integer sum of 2 outer products and subtract";
  let description = [{
    Example:
    ```mlir
    %result = arm_sme.umops_2way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
    ```

    Refer to
    [fmopa_2way](#arm_smefmopa_2way-arm_smefmopa2wayop) for a detailed
    description of 2-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [UMOPS (2-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPS--2-way---Unsigned-integer-sum-of-outer-products-and-subtract-) | +sme2 |
  }];
}

class OuterProduct4Way<string mnemonic,
                       list<Type> allowedInputVectorTypes,
                       list<Type> allowedResultVectorTypes>
  : OuterProductWideningBase<mnemonic, allowedInputVectorTypes,
                             allowedResultVectorTypes, /*numOuterProducts=*/4>;

def SMopa4WayOp
  : OuterProduct4Way<"smopa_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Signed integer sum of 4 outer products and accumulate";
  let description = [{
    This operation represents a sum of 4 widened outer products. It takes 2 1-D
    scalable vectors as input and a 2-D scalable vector (ZA tile) as output.

    For example (i8 to i32):

    ```mlir
    %result = arm_sme.smopa_4way $lhs, $rhs :
      vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    The `lhs` encodes a matrix of shape SVLSx4 and the `rhs` a matrix of
    4xSVLS, where SVLS (spec [1], section B2.1) is the number of 32-bit
    elements in a vector of SVL bits. To illustrate, below is a breakdown of
    this operation for i8 to i32, SVL=128 (i.e., vscale=1):

    ```
                                        LHS
              [A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A15 A14 A15]

                                        RHS
              [B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12 B13 B14 B15]

    ----------------------------------------------------------------------------

                                  implicit layout

                    [A0   A1  A2  A3]    |    [B0 B4  B8 B12]
                    [A4   A5  A6  A7]    |    [B1 B5  B9 B13]
                    [A8   A9 A10 A11]    |    [B2 B6 B10 B14]
                    [A12 A13 A14 A15]    |    [B3 B7 B11 B15]

    ----------------------------------------------------------------------------

                                  4 outer products

                 Acol0 ⊗ Brow0           |            Acol1 ⊗ Brow1
                 -------------           |            -------------
                                         |
             [B0 B4 B8 B12]              |        [B1 B5 B9 B13]
                                         |
       [A0   [ A0B0  A0B4  A0B8  A0B12]  |  [A1   [ A1B1  A1B5  A1B9  A1B13]
        A4   [ A4B0  A4B4  A4B8  A4B12]  |   A5   [ A5B1  A5B5  A5B9  A5B13]
        A8   [ A8B0  A8B4  A8B8  A8B12]  |   A9   [ A9B1  A9B5  A9B9  A9B13]
        A12] [A12B0 A12B4 A12B8 A12B12]  |   A13] [A13B1 A13B5 A13B9 A13B13]
                                         |
                 Acol2 ⊗ Brow2           |            Acol3 ⊗ Brow3
                 -------------           |            -------------
                                         |
             [B2, B6, B10, B14]          |        [B3 B7 B11 B15]
                                         |
       [A2   [ A2B2  A2B6  A2B10  A2B14] |  [A3   [ A3B3  A3B7  A3B11  A3B15]
        A6   [ A6B2  A6B6  A6B10  A6B14] |   A7   [ A7B3  A7B7  A7B11  A7B15]
        A10  [A10B2 A10B6 A10B10 A10B14] |   A11  [A11B3 A11B7 A11B11 A11B15]
        A14] [A14B2 A14B6 A14B10 A14B14] |   A15] [A15B3 A15B7 A15B11 A15B15]
                                         |

    ----------------------------------------------------------------------------

                              sum of 4 outer products

           Acol0 ⊗ Brow0 + Acol1 ⊗ Brow1 + Acol2 ⊗ Brow2 + Acol3 ⊗ Brow3

     [ A0B0 +  A1B1 +  A2B2 +  A3B3 ... ...  A0B12 +  A1B13 +  A2B14 +  A3B15]
     [ A4B0 +  A5B1 +  A6B2 +  A7B3 ... ...  A4B12 +  A5B13 +  A6B14 +  A7B15]
     [ A8B0 +  A9B1 + A10B2 + A11B3 ... ...  A8B12 +  A9B13 + A10B14 + A11B15]
     [A12B0 + A13B1 + A14B2 + A15B3 ... ... A12B12 + A13B13 + A14B14 + A15B15]

    ----------------------------------------------------------------------------
    ```

    This operation enables the folding of 4 outer products chained via the
    accumulator into a single outer product.

    For example:

    ```mlir
    %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
    %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

    %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
    %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

    %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
    %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

    %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
    %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

    %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
    %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
    %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[4]xi32>, vector<[4]xi32>
    %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[4]xi32>, vector<[4]xi32>
    ```

    The 4 outer products in the example above can be fused into a single outer
    product as follows:

    ```mlir
    %lhs0 = "llvm.intr.experimental.vector.interleave2"(%a0, %a2) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
    %lhs1 = "llvm.intr.experimental.vector.interleave2"(%a1, %a3) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
    %lhs = "llvm.intr.experimental.vector.interleave2"(%lhs0, %lhs1) : (vector<[8]xi8>, vector<[8]xi8>) -> vector<[16]xi8>

    %rhs0 = "llvm.intr.experimental.vector.interleave2"(%b0, %b2) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
    %rhs1 = "llvm.intr.experimental.vector.interleave2"(%b1, %b3) : (vector<[4]xi8>, vector<[4]xi8>) -> vector<[8]xi8>
    %rhs = "llvm.intr.experimental.vector.interleave2"(%rhs0, %rhs1) : (vector<[8]xi8>, vector<[8]xi8>) -> vector<[16]xi8>

    %0 = arm_sme.smopa_4way %lhs, %rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    This is implemented in the `-arm-sme-outer-product-fusion` pass.

    Example: I8 to I32
    ```mlir
    %result = arm_sme.smopa_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.smopa_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    | Spec | Features |
    | ---- | -------- |
    | [SMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPA--4-way---Signed-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def SMops4WayOp
  : OuterProduct4Way<"smops_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Signed integer sum of 4 outer products and subtract";
  let description = [{
    Equivalent to `smopa_4way` but outer products are subtracted from
    destination `result`.

    Example: I8 to I32
    ```mlir
    %result = arm_sme.smops_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.smops_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [SMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SMOPS--4-way---Signed-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def UMopa4WayOp
  : OuterProduct4Way<"umopa_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Unsigned integer sum of 4 outer products and accumulate";
  let description = [{
    Example: I8 to I32
    ```mlir
    %result = arm_sme.umopa_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.umopa_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [UMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPA--4-way---Unsigned-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def UMops4WayOp
  : OuterProduct4Way<"umops_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Unsigned integer sum of 4 outer products and subtract";
  let description = [{
    Example: I8 to I32
    ```mlir
    %result = arm_sme.umops_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.umops_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [UMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/UMOPS--4-way---Unsigned-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def SuMopa4WayOp
  : OuterProduct4Way<"sumopa_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Signed by unsigned integer sum of 4 outer products and accumulate";
  let description = [{
    Example: I8 to I32
    ```mlir
    %result = arm_sme.sumopa_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.sumopa_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [SUMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SUMOPA--Signed-by-unsigned-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def SuMops4WayOp
  : OuterProduct4Way<"sumops_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Signed by unsigned integer sum of 4 outer products and subtract";
  let description = [{
    Example: I8 to I32
    ```mlir
    %result = arm_sme.sumops_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.sumops_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [SUMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/SUMOPS--Signed-by-unsigned-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def UsMopa4WayOp
  : OuterProduct4Way<"usmopa_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Unsigned by signed integer sum of 4 outer products and accumulate";
  let description = [{
    Example: I8 to I32
    ```mlir
    %result = arm_sme.usmopa_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.usmopa_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [USMOPA (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/USMOPA--Unsigned-by-signed-integer-sum-of-outer-products-and-accumulate-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def UsMops4WayOp
  : OuterProduct4Way<"usmops_4way",
      [ScalableVectorOfRankAndLengthAndType<[1], [16], [I8]>,
       ScalableVectorOfRankAndLengthAndType<[1], [8], [I16]>],
      [nxnxv4i32, nxnxv2i64]> {
  let summary = "Unsigned by signed integer sum of 4 outer products and subtract";
  let description = [{
    Example: I8 to I32
    ```mlir
    %result = arm_sme.usmops_4way $lhs, $rhs : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
    ```

    Example: I16 to I64
    ```mlir
    %result = arm_sme.usmops_4way $lhs, $rhs : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
    ```

    Refer to [smopa_4way](#arm_smesmopa_4way-arm_smesmopa4wayop) for a
    detailed description of 4-way outer products.

    | Spec | Features |
    | ---- | -------- |
    | [USMOPS (4-way)](https://developer.arm.com/documentation/ddi0602/2023-09/SME-Instructions/USMOPS--Unsigned-by-signed-integer-sum-of-outer-products-and-subtract-) | +sme (32-bit), +sme-i16i64 (64-bit)|
  }];
}

def StreamingVLOp : ArmSME_Op<"streaming_vl", [Pure]>
{
  let summary = "Query the streaming vector length";

  let description = [{
    This operation returns the streaming vector length (SVL) for a given type
    size. Unlike `vector.vscale` the value returned is invariant to the
    streaming mode.

    Example:
    ```mlir
    // Streaming vector length in:
    // - bytes (8-bit, SVL.B)
    %svl_b = arm_sme.streaming_vl <byte>
    // - half words (16-bit, SVL.H)
    %svl_h = arm_sme.streaming_vl <half>
    // - words (32-bit, SVL.W)
    %svl_w = arm_sme.streaming_vl <word>
    // - double words (64-bit, SVL.D)
    %svl_d = arm_sme.streaming_vl <double>
    ```
  }];

  let arguments = (ins ArmSME_TypeSizeAttr: $type_size);
  let results = (outs Index);

  let assemblyFormat = "$type_size attr-dict";
}

#endif // ARMSME_OPS
