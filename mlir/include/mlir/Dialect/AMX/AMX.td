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
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    The dialect is split into user-facing AMX ops (AMX_Op) and
    backend-facing intrinsic ops (AMX_IntrOp).

    Note that since configuration changes (implicit at dialect level) are
    costly, it is highly recommended to use the AMX dialect on same-shaped
    vectors, at least within a single method.

    For details, see the Intel documentation:
    https://software.intel.com/content/www/us/en/develop/articles/intel-sdm.html
  }];

  let emitAccessorPrefix = kEmitAccessorPrefix_Both;
}

//===----------------------------------------------------------------------===//
// AMX Op and IntrOp definitions.
//===----------------------------------------------------------------------===//

class AMX_Op<string mnemonic, list<Trait> traits = []> :
  Op<AMX_Dialect, mnemonic, traits> {}

// The "internal" intrinsics are meant for compiler usage.
class AMX_IntrOp<string mnemonic, int numResults, list<Trait> traits = []> :
  LLVM_IntrOpBase<AMX_Dialect, mnemonic,
                  "x86_" # !subst(".", "_", mnemonic) # "_internal",
                  [], [], traits, numResults>;

//===----------------------------------------------------------------------===//
// AMX Op definitions (user facing).
//===----------------------------------------------------------------------===//

//
// Tile reset.
//

def TileZeroOp : AMX_Op<"tile_zero", [NoSideEffect]> {
  let summary = "tile zero operation";
  let description = [{
    Zeroes the destination tile, with the shape defined by the 2-dim
    vector type of the result. This is eventually lowered into the
    "tilezero" instruction with the corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_zero : vector<16x16xbf16>
    ```
  }];
  let results = (outs
    VectorOfRankAndType<[2], [F32, BF16, I32, I8]>:$res);
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return res().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "attr-dict `:` type($res)";
  let hasVerifier = 1;
}

//
// Tile memory operations.
//

def TileLoadOp : AMX_Op<"tile_load", [NoSideEffect]> {
  let summary = "tile load operation";
  let description = [{
    Loads a tile from memory defined by a base and indices, with the
    shape defined by the 2-dim vector type of the result. This is
    eventually lowered into the "tileloadd" instruction with the
    corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_load %arg0[%c0, %c0] : memref<?x?xi8> into vector<16x64xi8>
    ```
  }];
  let arguments = (ins Arg<AnyMemRef, "load base", [MemRead]>:$base,
                   Variadic<Index>:$indices);
  let results = (outs
    VectorOfRankAndType<[2], [F32, BF16, I32, I8]>:$res);
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getVectorType() {
      return res().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$base `[` $indices `]` attr-dict `:` "
                       "type($base) `into` type($res)";
  let hasVerifier = 1;
}

def TileStoreOp : AMX_Op<"tile_store"> {
  let summary = "tile store operation";
  let description = [{
    Stores a tile to memory defined by a base and indices, with the
    shape defined by the 2-dim vector type of the value. This is
    eventually lowered into the "tilestored" instruction with the
    corresponding tile configuration.

    Example:

    ```mlir
      amx.tile_store %arg1[%c0, %c0], %0 : memref<?x?xi8>, vector<16x64xi8>
    ```
  }];
  let arguments = (ins Arg<AnyMemRef, "store base", [MemWrite]>:$base,
                   Variadic<Index>:$indices,
                   VectorOfRankAndType<[2], [F32, BF16, I32, I8]>:$val);
  let extraClassDeclaration = [{
    MemRefType getMemRefType() {
      return base().getType().cast<MemRefType>();
    }
    VectorType getVectorType() {
      return val().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$base `[` $indices `]` `,` $val attr-dict `:` "
                       "type($base) `,` type($val)";
  let hasVerifier = 1;
}

//
// Tile arithmetic operations.
//

def TileMulFOp : AMX_Op<"tile_mulf", [NoSideEffect, AllTypesMatch<["acc", "res"]>]> {
  let summary = "tile multiplication operation (floating-point)";
  let description = [{
    Multiplies a "m x k" tile with a "k x n" tile and accumulates the results
    into a "m x n" destination tile. Supports "f32 <- bf16 x bf16" (with
    pairs of "bf16"). The operation is eventually lowered into the
    "tdpbf16ps" instruction with the corresponding tile configuration.

    Example:

    ```mlir
      %0 = amx.tile_mulf %a, %b, %c
        : vector<16x32xbf16>, vector<16x32xbf16>, vector<16x16xf32>
    ```
  }];
  let arguments = (ins VectorOfRankAndType<[2], [F32, BF16]>:$lhs,
                       VectorOfRankAndType<[2], [F32, BF16]>:$rhs,
                       VectorOfRankAndType<[2], [F32, BF16]>:$acc);
  let results = (outs VectorOfRankAndType<[2], [F32, BF16]>:$res);
  let extraClassDeclaration = [{
    VectorType getLhsVectorType() {
      return lhs().getType().cast<VectorType>();
    }
    VectorType getRhsVectorType() {
      return rhs().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return res().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` "
                       "type($lhs) `,` type($rhs) `,` type($acc) ";
  let hasVerifier = 1;
}

def TileMulIOp : AMX_Op<"tile_muli", [NoSideEffect, AllTypesMatch<["acc", "res"]>]> {
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
        : vector<16x64xi8>, vector<16x64xi8>, vector<16x16xi32>
    ```
  }];
  let arguments = (ins VectorOfRankAndType<[2], [I32, I8]>:$lhs,
                       VectorOfRankAndType<[2], [I32, I8]>:$rhs,
                       VectorOfRankAndType<[2], [I32, I8]>:$acc,
                       UnitAttr:$isZextLhs,
                       UnitAttr:$isZextRhs
                       );
  let results = (outs VectorOfRankAndType<[2], [I32, I8]>:$res);
  let extraClassDeclaration = [{
    VectorType getLhsVectorType() {
      return lhs().getType().cast<VectorType>();
    }
    VectorType getRhsVectorType() {
      return rhs().getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return res().getType().cast<VectorType>();
    }
  }];
  let assemblyFormat = "$lhs (`zext` $isZextLhs^)? `,` $rhs (`zext` $isZextRhs^)? `,` $acc attr-dict `:` "
                       "type($lhs) `,` type($rhs) `,` type($acc) ";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AMX IntrOp definitions (LLVM compiler facing).
//===----------------------------------------------------------------------===//

//
// Tile reset. Parameters define the tile size.
//

def LLVM_x86_amx_tilezero : AMX_IntrOp<"tilezero", 1>,
  Arguments<(ins AnyInteger, AnyInteger)>;

//
// Tile memory operations. Parameters define the tile size,
// base address, and stride between consecutive rows for the
// memory operation.
//

def LLVM_x86_amx_tileloadd64 : AMX_IntrOp<"tileloadd64", 1>,
  Arguments<(ins AnyInteger,
                 AnyInteger, LLVM_AnyPointer, AnyInteger)>;

def LLVM_x86_amx_tilestored64 : AMX_IntrOp<"tilestored64", 0>,
  Arguments<(ins AnyInteger,
                 AnyInteger, LLVM_AnyPointer, AnyInteger, LLVM_Type)>;

//
// Tile multiplication operations (series of dot products). Parameters
// define the tile sizes and source and destination tiles for the
// operation. Note that the prefix "tdp" stands for tile dot product.
//

// Dot product of bf16 tiles into f32 tile.
def LLVM_x86_amx_tdpbf16ps : AMX_IntrOp<"tdpbf16ps", 1>,
  Arguments<(ins AnyInteger,
                 AnyInteger,
		 AnyInteger, LLVM_Type, LLVM_Type, LLVM_Type)>;

// Dot product of i8 tiles into i32 tile (with sign/sign extension).
def LLVM_x86_amx_tdpbssd : AMX_IntrOp<"tdpbssd", 1>,
  Arguments<(ins AnyInteger,
                 AnyInteger,
		 AnyInteger, LLVM_Type, LLVM_Type, LLVM_Type)>;

// Dot product of i8 tiles into i32 tile (with sign/zero extension).
def LLVM_x86_amx_tdpbsud : AMX_IntrOp<"tdpbsud", 1>,
  Arguments<(ins AnyInteger,
                 AnyInteger,
		 AnyInteger, LLVM_Type, LLVM_Type, LLVM_Type)>;

// Dot product of i8 tiles into i32 tile (with zero/sign extension).
def LLVM_x86_amx_tdpbusd : AMX_IntrOp<"tdpbusd", 1>,
  Arguments<(ins AnyInteger,
                 AnyInteger,
		 AnyInteger, LLVM_Type, LLVM_Type, LLVM_Type)>;

// Dot product of i8 tiles into i32 tile (with zero/zero extension).
def LLVM_x86_amx_tdpbuud : AMX_IntrOp<"tdpbuud", 1>,
  Arguments<(ins AnyInteger,
                 AnyInteger,
		 AnyInteger, LLVM_Type, LLVM_Type, LLVM_Type)>;

#endif // AMX
