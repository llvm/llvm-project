//===- TileReducerTypes.td - TileReducer types -------------*- tablegen -*-===//
//
// Milestone 2: logical tile and addressable buffer types.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_REDUCER_TYPES
#define TILE_REDUCER_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TileReducer/TileReducerDialect.td"

class TileReducer_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TileReducer_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TileReducer_TileType : TileReducer_Type<"Tile", "tile"> {
  let summary = "Logical static-shape SSA tile. No physical storage.";
  let description = [{
    `!tr.tile<128x128xf32>` is a logical collection of values. It does not
    imply a memref allocation, thread ownership, or a hardware load.

    Dimensions must be static and positive. Rank 0 (`!tr.tile<f32>`) is
    allowed so a full reduction can produce a scalar tile. Dynamic or named
    extents are rejected: those belong on `!tr.buffer`.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];
  let assemblyFormat = "`<` custom<TRShape>($shape, $elementType) `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    int64_t getRank() const { return static_cast<int64_t>(getShape().size()); }
    int64_t getDimSize(int64_t dim) const { return getShape()[dim]; }
    bool hasStaticShape() const { return true; }
  }];
}

def TileReducer_BufferType : TileReducer_Type<"Buffer", "buffer"> {
  let summary = "Addressable external storage. May have dynamic dimensions.";
  let description = [{
    `!tr.buffer<MxKxf32>` is a logical buffer that later lowers to MemRef.
    Extents may be a static integer, `?`, or a name (`M`, `K`) that prints
    back as that name and is stored as a dynamic dimension.

    Named extents are a pretty-print of dynamism, not a new shape lattice.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    ArrayRefParameter<"std::string">:$dimNames
  );
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      CArg<"ArrayRef<std::string>", "{}">:$dimNames
    ), [{
      return $_get(elementType.getContext(), shape, elementType, dimNames);
    }]>
  ];
  let assemblyFormat = "`<` custom<TRBufferShape>($shape, $elementType, $dimNames) `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    int64_t getRank() const { return static_cast<int64_t>(getShape().size()); }
    int64_t getDimSize(int64_t dim) const { return getShape()[dim]; }
    bool isDynamicDim(int64_t dim) const {
      return ::mlir::ShapedType::isDynamic(getShape()[dim]);
    }
  }];
}

#endif // TILE_REDUCER_TYPES
