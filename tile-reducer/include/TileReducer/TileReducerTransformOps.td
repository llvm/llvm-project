//===- TileReducerTransformOps.td - transform.tr.* ---------*- tablegen -*-===//
//
// Milestone 13: custom Transform dialect extension.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_REDUCER_TRANSFORM_OPS
#define TILE_REDUCER_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MapRowReductionOp : TransformDialectOp<"tr.map_row_reduction", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Annotate a row reduction with the 8-warp physical map.";
  let description = [{
    Operates on a handle to payload `linalg.generic` ops. Validates that each
    payload op is a row reduction (`parallel, reduction`) and annotates it
    with the baseline physical map:

      warps_per_block    = 8
      warp_size          = 32
      elements_per_lane  = 4

    8 warps x 16 rows = 128 rows; 32 lanes x 4 elems = 128 columns.
    Does not emit GPU thread IDs. Silenceable failure if the handle does not
    point at a row-reduction generic.
  }];
  let arguments = (ins
    TransformHandleTypeInterface:$target,
    DefaultValuedAttr<I64Attr, "8">:$warps_per_block,
    DefaultValuedAttr<I64Attr, "32">:$warp_size,
    DefaultValuedAttr<I64Attr, "4">:$elements_per_lane
  );
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // TILE_REDUCER_TRANSFORM_OPS
