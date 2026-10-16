//===- TileReducerDialect.td - TileReducer dialect ---------*- tablegen -*-===//
//
// Milestone 1: dialect declaration only. Types arrive in Milestone 2,
// operations in Milestone 3.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_REDUCER_DIALECT
#define TILE_REDUCER_DIALECT

include "mlir/IR/OpBase.td"

def TileReducer_Dialect : Dialect {
  let name = "tr";
  let summary = "Tile-level reduction dialect for GPU compilation.";
  let description = [{
    TileReducer is a tile-level source dialect. Programmers write logical
    tiles and reductions; they do not write threadIdx, lane IDs, or launch
    geometry. Physical GPU mapping is chosen by later lowering stages.

    `tr.program_id` names a logical program instance, not a hardware thread.
    `!tr.tile` is a logical SSA value with no required physical storage.
    `!tr.buffer` is addressable external storage and later lowers to MemRef.
  }];
  let cppNamespace = "::mlir::tr";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class TileReducer_Op<string mnemonic, list<Trait> traits = []> :
    Op<TileReducer_Dialect, mnemonic, traits>;

#endif // TILE_REDUCER_DIALECT
