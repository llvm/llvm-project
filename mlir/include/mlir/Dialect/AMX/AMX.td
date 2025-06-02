//===-- AMX.td - AMX dialect operation definitions *- tablegen -*----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the AMX dialect.
//
// The Intel Advanced Matrix Extensions (AMX) provide a tile matrix
// multiply unit (TMUL), a tile control register (TILECFG), and eight
// tile registers TMM0 through TMM7 (TILEDATA).
//
// The AMX dialect provides a bridge between MLIR concepts, such as
// 2-d vector, operations, and memrefs, and the lower level details
// of Intel AMX, such as configuration setup, tile sizes, instructions,
// and tile release.
//
// Note that since configuration changes (implicit at dialect level) are
// costly, it is highly recommended to use the AMX dialect on same-shaped
// vectors, at least within a single method.
//
// https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html
//
//===----------------------------------------------------------------------===//

#ifndef AMX
#define AMX

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/AMX/AMXInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// AMX dialect definition.
//===----------------------------------------------------------------------===//

def AMX_Dialect : Dialect {
  let name = "amx";
  let cppNamespace = "::mlir::amx";
  let description = [{
    The Intel Advanced Matrix Extensions (AMX) provide a tile matrix
    multiply unit (TMUL), a tile control register (TILECFG), and eight
    tile registers TMM0 through TMM7 (TILEDATA).

    This `AMX` dialect provides a bridge between MLIR concepts such as
    vectors and memrefs and the lower level LLVM IR support of AMX.

    Note that since configuration changes (implicit at dialect level) are
    costly, it is highly recommended to use the AMX dialect on same-shaped
    vectors, at least within a single method.

    For details, see the Intel documentation:
    https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html
  }];
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// AMX Tile definition.
//===----------------------------------------------------------------------===//

class AMX_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMX_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

def AMX_TileTypeElementType : AnyTypeOf<[F32, F16, BF16, I32, I8]> {
  let cppFunctionName = "isValidTileTypeElementType";
}

def AMX_TileType : AMX_Type<"Tile", "tile", [ShapedTypeInterface, ValueSemantics]> {
  let summary = "AMX 2D tile to be used by AMX opertaions.";

  let description = [{
    This type is used to represent values in AMX tile registers. All AMX operations
    work on AMX tiles and these tiles cannot be used in other operations directly.
    LLVM IR type for AMX tile is a primitive type, but in MLIR we provide shape and
    element type for IR verification and lowering to LLVMIR dialect.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    AMX_TileTypeElementType:$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this tile type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    TileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const {
      return get(shape.value_or(getShape()), elementType);
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def IsAMXTilePred : And<[CPred<"::llvm::isa<::mlir::amx::TileType>($_self)">,
  CPred<[{::llvm::cast<::mlir::amx::TileType>($_self).getRank() == 2}]>]>;

class AMXTileOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsAMXTilePred, "tile",
                      "::mlir::amx::TileType">;

def AnyAMXTile : AMXTileOf<[F32, F16, BF16, I32, I8]>;

def AMXTileF32 : AMXTileOf<[F32]>;

def AMXTileF16OrBF16 : AMXTileOf<[F16, BF16]>;

def AMXTileI32 : AMXTileOf<[I32]>;

def AMXTileI8 : AMXTileOf<[I8]>;

//===----------------------------------------------------------------------===//
// AMX Op and IntrOp definitions.
//===----------------------------------------------------------------------===//

class AMX_Op<string mnemonic, list<Trait> traits = []> :
  Op<AMX_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// AMX Op definitions
//===----------------------------------------------------------------------===//

//
// Tile reset.
//

def TileZeroOp : AMX_Op<"tile_zero", [Pure,
    AMXIntrinsicOpInterface
  ]> {
  let summary = "tile zero operation";
  let description = [{
    Zeroes the destination tile, with the shape defined by the 2-dim
    vector type of the result. This is eventually lowered into the
    "tilezero" instruction with the corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_zero : !amx.tile<16x16xbf16>
    ```
  }];
  let results = (outs AnyAMXTile:$res);
  let extraClassDeclaration = [{
    TileType getTileType() {
      return ::llvm::cast<TileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      return "llvm.x86.tilezero.internal";
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "attr-dict `:` qualified(type($res))";
  let hasVerifier = 1;
}

//
// Tile memory operations.
//

def TileLoadOp : AMX_Op<"tile_load", [Pure,
    AMXIntrinsicOpInterface
  ]> {
  let summary = "tile load operation";
  let description = [{
    Loads a tile from memory defined by a base and indices, with the
    shape defined by the 2-dim vector type of the result. This is
    eventually lowered into the "tileloadd" instruction with the
    corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_load %arg0[%c0, %c0] : memref<?x?xi8> into !amx.tile<16x64xi8>
    ```
  }];
  let arguments = (ins Arg<AnyMemRef, "load base", [MemRead]>:$base,
                   Variadic<Index>:$indices);
  let results = (outs AnyAMXTile:$res);
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    TileType getTileType() {
      return ::llvm::cast<TileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      return "llvm.x86.tileloadd64.internal";
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$base `[` $indices `]` attr-dict `:` "
                       "type($base) `into` qualified(type($res))";
  let hasVerifier = 1;
}

def TileStoreOp : AMX_Op<"tile_store", [
    AMXIntrinsicOpInterface
  ]> {
  let summary = "tile store operation";
  let description = [{
    Stores a tile to memory defined by a base and indices, with the
    shape defined by the 2-dim vector type of the value. This is
    eventually lowered into the "tilestored" instruction with the
    corresponding tile configuration.

    Example:

    ```mlir
      amx.tile_store %arg1[%c0, %c0], %0 : memref<?x?xi8>, !amx.tile<16x64xi8>
    ```
  }];
  let arguments = (ins Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                   Variadic<Index>:$indices,
                   AnyAMXTile:$val);
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getBase().getType());
    }
    TileType getTileType() {
      return ::llvm::cast<TileType>(getVal().getType());
    }

    std::string getIntrinsicName() {
      return "llvm.x86.tilestored64.internal";
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $val attr-dict `:` "
                       "type($base) `,` qualified(type($val))";
  let hasVerifier = 1;
}

//
// Tile arithmetic operations.
//

def TileMulFOp : AMX_Op<"tile_mulf", [Pure,
    AMXIntrinsicOpInterface,
    AllTypesMatch<["acc", "res"]>
  ]> {
  let summary = "tile multiplication operation (floating-point)";
  let description = [{
    Multiplies a "m x k" tile with a "k x n" tile and accumulates the results
    into a "m x n" destination tile. Supports "f32 <- bf16 x bf16" (with
    pairs of "bf16"). The operation is eventually lowered into the
    "tdpbf16ps" instruction with the corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_mulf %a, %b, %c
        : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
    ```
  }];
  let arguments = (ins AMXTileF16OrBF16:$lhs,
                       AMXTileF16OrBF16:$rhs,
                       AMXTileF32:$acc);
  let results = (outs AMXTileF32:$res);
  let extraClassDeclaration = [{
    TileType getLhsTileType() {
      return ::llvm::cast<TileType>(getLhs().getType());
    }
    TileType getRhsTileType() {
      return ::llvm::cast<TileType>(getRhs().getType());
    }
    TileType getTileType() {
      return ::llvm::cast<TileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.tdp";
      auto elementType =
        getLhsTileType().getElementType();
      intr += elementType.isF16() ? "fp16" : "bf16";
      intr += "ps.internal";
      return intr;
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` "
                       "qualified(type($lhs)) `,` qualified(type($rhs))"
                       " `,` qualified(type($acc)) ";
  let hasVerifier = 1;
}

def TileMulIOp : AMX_Op<"tile_muli", [Pure,
    AMXIntrinsicOpInterface,
    AllTypesMatch<["acc", "res"]>
  ]> {
  let summary = "tile multiplication operation (integer)";
  let description = [{
    Multiplies a "m x k" tile with a "k x n" tile and accumulates the results
    into a "m x n" destination tile. Supports all "si32 <- s/ui8 x s/ui8"
    combinations (4 bytes packed into dwords in the columns of both the
    source operand tiles; the zero or sign extension is specified with
    the attributes and default to sign extended). The operation is eventually
    lowered into one of the "tdpbssd", "tdpbsud", "tdpbusd", or "tdpbuud"
    instructions with the corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_muli %a zext, %b zext, %c
        : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
    ```
  }];
  let arguments = (ins AMXTileI8:$lhs,
                       AMXTileI8:$rhs,
                       AMXTileI32:$acc,
                       UnitAttr:$isZextLhs,
                       UnitAttr:$isZextRhs
                       );
  let results = (outs AMXTileI32:$res);
  let extraClassDeclaration = [{
    TileType getLhsTileType() {
      return ::llvm::cast<TileType>(getLhs().getType());
    }
    TileType getRhsTileType() {
      return ::llvm::cast<TileType>(getRhs().getType());
    }
    TileType getTileType() {
      return ::llvm::cast<TileType>(getRes().getType());
    }

    std::string getIntrinsicName() {
      std::string intr = "llvm.x86.tdpb";
      intr += getIsZextLhs() ? "u" : "s";
      intr += getIsZextRhs() ? "u" : "s";
      intr += "d.internal";
      return intr;
    }
    SmallVector<Value> getIntrinsicOperands(
        ::mlir::ArrayRef<Value> operands,
        const ::mlir::LLVMTypeConverter &typeConverter,
        ::mlir::RewriterBase &rewriter);
  }];
  let assemblyFormat = "$lhs (`zext` $isZextLhs^)? `,` $rhs (`zext` $isZextRhs^)? `,` $acc attr-dict `:` "
                       "qualified(type($lhs)) `,` qualified(type($rhs)) `,` qualified(type($acc)) ";
  let hasVerifier = 1;
}

#endif // AMX
