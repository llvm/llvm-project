//===- TileReducerOps.td - TileReducer operations ----------*- tablegen -*-===//

#ifndef TILE_REDUCER_OPS
#define TILE_REDUCER_OPS

include "TileReducer/TileReducerDialect.td"
include "TileReducer/TileReducerTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// tr.program_id
//===----------------------------------------------------------------------===//

def TileReducer_ProgramIdOp : TileReducer_Op<"program_id", [Pure]> {
  let summary = "Logical program-instance coordinate. Not threadIdx.";
  let description = [{
    `%id = tr.program_id 0 : index` names a logical TileReducer program
    instance. Later GPU lowering decides whether that is a block, a warp,
    or several blocks.
  }];
  let arguments = (ins I64Attr:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "$axis attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tr.dim
//===----------------------------------------------------------------------===//

def TileReducer_DimOp : TileReducer_Op<"dim", [Pure]> {
  let summary = "Query one extent of a buffer.";
  let arguments = (ins AnyType:$buffer, I64Attr:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "$buffer `,` $axis attr-dict `:` type($buffer) `,` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tr.constant
//===----------------------------------------------------------------------===//

def TileReducer_ConstantOp : TileReducer_Op<"constant", [Pure]> {
  let summary = "Splat a scalar attribute onto a logical tile.";
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  // Custom format so `0.0 : !tr.tile<128xf32>` does not parse `0.0` as a
  // typed attribute of tile type (MLIR would otherwise eat the colon).
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tr.load / tr.store
//===----------------------------------------------------------------------===//

def TileReducer_LoadOp : TileReducer_Op<"load", [Pure]> {
  let summary = "Logical tile load. Not a hardware instruction.";
  let description = [{
    `%t = tr.load %in[%row, %kt] : !tr.tile<128x128xf32>`

    Indices are *tile* coordinates into the buffer, one per buffer
    dimension. The compiler may realize this as coalesced loads, a fused
    load/reduce, or nothing materialised at all.
  }];
  let arguments = (ins AnyType:$buffer, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $buffer `[` $indices `]` attr-dict `:` type($buffer) `,` type($result)
  }];
  let hasVerifier = 1;
}

def TileReducer_StoreOp : TileReducer_Op<"store", []> {
  let summary = "Logical tile store into a buffer.";
  let arguments = (ins
    AnyType:$buffer,
    Variadic<Index>:$indices,
    AnyType:$value
  );
  let assemblyFormat = [{
    $buffer `[` $indices `]` `,` $value attr-dict `:` type($buffer) `,` type($value)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tr.add
//===----------------------------------------------------------------------===//

def TileReducer_AddOp : TileReducer_Op<"add",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise tile addition.";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tr.reduce_sum
//===----------------------------------------------------------------------===//

def TileReducer_ReduceSumOp : TileReducer_Op<"reduce_sum", [Pure]> {
  let summary = "Sum-reduce a tile along one axis.";
  let arguments = (ins AnyType:$input, I64Attr:$axis);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input `,` `axis` `=` $axis attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tr.yield / tr.for
//===----------------------------------------------------------------------===//

def TileReducer_YieldOp : TileReducer_Op<"yield",
    [HasParent<"ForOp">, Pure, ReturnLike, Terminator]> {
  let summary = "Yield values from a tr.for body.";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ValueRange());
    }]>
  ];
}

def TileReducer_ForOp : TileReducer_Op<"for",
    [SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects,
     AttrSizedOperandSegments]> {
  let summary = "Tile-level counted loop with iter_args.";
  let description = [{
    Source-level loop. Bounds may be SSA values or integer literals
    (`0`, `1`) so the canonical program parses unchanged. Lowered to
    `scf.for` in a later milestone.
  }];
  let arguments = (ins
    Optional<Index>:$lowerBound,
    Index:$upperBound,
    Optional<Index>:$step,
    OptionalAttr<I64Attr>:$staticLowerBound,
    OptionalAttr<I64Attr>:$staticStep,
    Variadic<AnyType>:$initArgs
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    BlockArgument getInductionVar() { return getRegion().getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getRegion().getArguments().drop_front();
    }
    std::optional<int64_t> getConstantLowerBound();
    std::optional<int64_t> getConstantStep();
  }];
}

#endif // TILE_REDUCER_OPS
