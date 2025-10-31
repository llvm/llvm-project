//===- XeGPUTypes.td - XeGPU dialect types definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD

include "mlir/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "mlir/Dialect/XeGPU/IR/XeGPUDialect.td"
include "mlir/IR/BuiltinTypes.td"

def XeGPU_IntType: AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;
def XeGPU_FloatType: AnyTypeOf<[F16, F32, F64, BF16, TF32]>;
def XeGPU_ScalarType: AnyTypeOf<[XeGPU_IntType, XeGPU_FloatType]>;
def XeGPU_PointerType : AnyTypeOf<[UI64, UI32, I64, I32]>;
def XeGPU_BaseAddrType
    : AnyTypeOf<[Non0RankedMemRefOf<[XeGPU_ScalarType]>, XeGPU_PointerType]>;
def XeGPU_DpasOprType: FixedVectorOfRankAndType<[1, 2, 3], [XeGPU_ScalarType]>;
def XeGPU_DpasResType: FixedVectorOfRankAndType<[1, 2], [XeGPU_ScalarType]>;
def XeGPU_OffsetType: FixedVectorOfNonZeroRankOf<[Index]>;
def XeGPU_MaskType: FixedVectorOfNonZeroRankOf<[I1]>;
def XeGPU_ValueType: FixedVectorOfNonZeroRankOf<[XeGPU_ScalarType]>;
def XeGPU_VectorType: VectorOfRankAndType<[1,2,3,4,5,6], [XeGPU_ScalarType]>;
def XeGPU_GatherScatterBaseAddrType
    : AnyTypeOf<[MemRefRankOf<[XeGPU_ScalarType], [1]>, XeGPU_PointerType]>;

// common base class for types in XeGPU dialect
class XeGPUTypeDef<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<XeGPU_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def XeGPU_TensorDesc: XeGPUTypeDef<"TensorDesc", "tensor_desc",
        [ShapedTypeInterface], "::mlir::TensorType"> {
  let summary = "TensorDesc describing regions of interested data.";
  let description = [{
    TensorDesc is a type designed to describe regions of interest in data, as well as some features
    unique to Intel hardware. Unlike the built-in tensor type in MLIR, it essentially contains only
    metadata and does not hold the data itself. It is primarily designed to support 2D block load/store
    and DPAS (matrix multiplication instruction) on Intel GPUs. It encodes the following information:

    * shape:  the sizes/shape of the interested data block, e.g., 8x16 means 8 rows
              and each row contains 16 contiguous data element. The rows could be
              either contiguous or not, depends on the encoding attribute. If the
              encoding is a BlockTensorDescAttr, rows are contiguous. If the encoding
              is a ScatterTensorDescAttr, rows are not necessary to be contiguous. If
              encoding is not set, it is considered as a default BlockTensorDescAttr.

    * element_type: the data type of the data element, e.g., f16, f32.

    Similar to the built-in tensor, it also provides optional attributes for encoding
    additional information via either BlockTensorDescAttr or ScatterTensorDescAttr, or
    supporting Workgroup, Subgroup, and workitem (or SIMT) level programmings via the
    Layout attribute. Please check their definition for details.

    Syntax:

    ```
    TensorDesc-type ::= `tensor_desc` `<` dim-list element-type (attr-list)? `>`
    element-type ::= float-type | integer-type | index-type
    dim-list := (static-dim-list `x`)?
    static-dim-list ::= decimal-literal `x` decimal-literal
    attr-list = (, encoding-attr)? (, layout-attr)?
    enconding-attr = (, memory_space = value)? (, arr_len = value)? (, boundary_check = value)? (, scattered = value)?
    layout-attr = (, layout `<`sg_layout = value, sg_data = value, inst_data = value, lane_layout = value, lane_data = value, order = value`>`)?
    ```

    Examples:

    ```mlir
    // A block TensorDesc with 8x16 i32 elements
    xegpu.tensor_desc<8x16xi32>

    // A block TensorDesc with 8x16 f32 elements
    xegpu.tensor_desc<8x16xf32>

    // A TensorDesc with 8x16 f32 elements for a memory region in shared memory space.
    xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_space = slm>>

    // A 1D TensorDesc with a layout for subgroup level programming, each lane access two continuous elements
    xegpu.tensor_desc<32xf32, #xegpu.layout<lane_layout = [16], lane_data = [2]>>

    // A 1D TensorDesc with a layout for subgroup level programming, each lane access two elements with stride = 16
    xegpu.tensor_desc<32xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>

    // A TensorDesc with a layout for subgroup level programming
    xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>

    // A TensorDesc with a layout for workgroup level programming
    xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [1, 16], lane_data = [1, 1]>>

    // A TensorDesc with a layout for workgroup level programming without lane_layout and lane_data
    xegpu.tensor_desc<32x64xf32, #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16]>>

    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">: $shape,
                        "mlir::Type": $elementType,
                        OptionalParameter<"mlir::Attribute">: $encoding,
                        OptionalParameter<"mlir::Attribute">: $layout);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>": $shape,
      "mlir::Type": $elementType,
      CArg<"int", "1">: $array_length,
      CArg<"bool", "true">: $boundary_check,
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space,
      CArg<"mlir::Attribute", "mlir::Attribute()">:$layout)>,
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>": $shape,
      "mlir::Type": $elementType,
      CArg<"int", "1">: $chunk_size,
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space,
      CArg<"mlir::Attribute", "mlir::Attribute()">:$layout)>
  ];

  let extraClassDeclaration = [{
    using mlir::ShapedType::Trait<TensorDescType>::getElementTypeBitWidth;
    using mlir::ShapedType::Trait<TensorDescType>::getRank;
    using mlir::ShapedType::Trait<TensorDescType>::getNumElements;
    using mlir::ShapedType::Trait<TensorDescType>::isDynamicDim;
    using mlir::ShapedType::Trait<TensorDescType>::hasStaticShape;
    using mlir::ShapedType::Trait<TensorDescType>::getNumDynamicDims;
    using mlir::ShapedType::Trait<TensorDescType>::getDimSize;
    using mlir::ShapedType::Trait<TensorDescType>::getDynamicDimIndex;

    TensorDescType clone(::mlir::Type elementType) {
      return llvm::cast<TensorDescType>(cloneWith(getShape(), elementType));
    }

    template <typename T,
              typename = std::enable_if_t<
                            std::is_same_v<T, BlockTensorDescAttr> ||
                            std::is_same_v<T, ScatterTensorDescAttr>>>
    T getEncodingOfType() const {
      return llvm::dyn_cast_if_present<T>(getEncoding());
    }

    LayoutAttr getLayoutAttr() const {
      return llvm::dyn_cast_if_present<LayoutAttr>(getLayout());
    }

    xegpu::MemorySpace getMemorySpace() const {
      if (auto attr = getEncodingOfType<BlockTensorDescAttr>())
        return attr.getMemorySpace().getValue();

      auto attr = getEncodingOfType<ScatterTensorDescAttr>();
      return attr.getMemorySpace().getValue();
    }

    // get the ArrayLength for blocked TensorDesc
    int getArrayLength() {
      auto attr = getEncodingOfType<BlockTensorDescAttr>();
      assert(attr && "invalid on non BlockTensorDescAttr.");
      return attr.getArrayLength().getInt();
    }

    bool getBoundaryCheck() {
      auto attr = getEncodingOfType<BlockTensorDescAttr>();
      assert(attr && "invalid on non BlockTensorDescAttr.");
      return attr.getBoundaryCheck().getValue();
    }

    bool isScattered() {
      return bool(getEncodingOfType<ScatterTensorDescAttr>());
    }

    // get the ChunkSize for scattered TensorDesc
    int getChunkSizeAsInt() {
      auto attr = getEncodingOfType<ScatterTensorDescAttr>();
      assert(attr && "invalid on non ScatterTensorDescAttr.");
      return attr.getChunkSizeAsInt();
    }

    /// Helper to drop all layout information from the TensorDesc type.
    TensorDescType dropLayouts() {
      if (!getLayoutAttr())
        return *this;

      return get(getContext(), getShape(), getElementType(), getEncoding(),
        xegpu::LayoutAttr());
    }
  }];

  let hasCustomAssemblyFormat = true;
  let genVerifyDecl = 1;
}

def XeGPU_GatherScatterSourceType
    : AnyTypeOf<[XeGPU_TensorDesc, XeGPU_GatherScatterBaseAddrType]>;

def XeGPU_Nbarrier: XeGPUTypeDef<"Nbarrier", "nbarrier", [], "mlir::Type"> {
  let summary = "!xegpu.nbarrier a custom XeGPU type representing a barrier.";

  let extraClassDeclaration = [{
    static NbarrierType get(mlir::MLIRContext *context) {
      return Base::get(context);
    };
  }];
}

def XeGPU_MemDesc: XeGPUTypeDef<"MemDesc", "mem_desc", [ShapedTypeInterface], "mlir::Type"> {
  let summary = "MemDesc describing the data in SLM";
  let description = [{
    MemDesc represents a block of data stored in shared local memory.
    By default, unless a layout attribute is provided, the data is stored
    contiguously in row-major order within the region.

    Examples:
    ```mlir
    // A multi-dimensional array stored in column-major order.
    !xegpu.mem_desc<128x128xf16, #xegpu.mem_layout<stride = [1, 128]>>

    // A multi-dimensional array stored in a blocked layout. Elements within the same block
    // are stored contiguously in memory. Blocks are stored in row-major order.
    !xegpu.mem_desc<128x128xf16, #xegpu.mem_layout<block = [8, 8]>>

    // A multi-dimensional array stored in column-major order with blocked layout.
    !xegpu.mem_desc<128x128xf16, #xegpu.mem_layout<stride = [1, 128], block = [8, 8]>>
    ```
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">: $shape,
                        "mlir::Type": $elementType,
                        OptionalParameter<"MemLayoutAttr">: $mem_layout);

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }

    MemDescType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
      return MemDescType::get(getContext(), shape.value_or(getShape()), elementType, getMemLayout());
    }

    ArrayAttr getStrideAttr() {
      auto layout = getMemLayout();
      if (layout && layout.hasAttr("stride")) {
        return layout.getStrideAttr();
      }
      // derive and return default strides
      SmallVector<int64_t> defaultStrides;
      llvm::append_range(defaultStrides, getShape().drop_front());
      llvm::append_values(defaultStrides, 1);
      Builder builder(getContext());
      return builder.getI64ArrayAttr(defaultStrides);
    }

    ArrayAttr getBlockAttr() {
      auto layout = getMemLayout();
      if (layout && layout.hasAttr("block")) {
        return layout.getBlockAttr();
      }
      Builder builder(getContext());
      return builder.getI64ArrayAttr({});
    }

    /// Heuristic to determine if the MemDesc uses column-major layout,
    /// based on the rank and the value of the first stride dimension.
    bool isColMajor() {
      auto dim0 = dyn_cast<IntegerAttr>(getStrideAttr()[0]);
      return getRank() == 2 && dim0.getInt() == 1;
    }

    // Get the Blocking shape for a MemDescType, Which is represented
    // as an attribute in MemDescType. By default it is the shape
    // of the mdescTy
    SmallVector<int64_t> getBlockShape() {
      SmallVector<int64_t> size(getShape());
      ArrayAttr blockAttr = getBlockAttr();
      if (!blockAttr.empty()) {
        size.clear();
        for (auto attr : blockAttr.getValue()) {
          size.push_back(cast<IntegerAttr>(attr).getInt());
        }
      }
      return size;
    }

    // Get strides as vector of integer. 
    // If it contains block attribute, the strides are blocked strides.
    //
    // The blocking is applied to the base matrix shape derived from the
    // memory descriptor's stride information. If the matrix described by
    // the memory descriptor is not contiguous, it is assumed that the base
    // matrix is contiguous and follows the same memory layout.
    //
    // It first computes the original matrix shape using the stride info,
    // then computes the number of blocks in each dimension of original shape,
    // then compute the outer block shape and stride,
    // then combines the inner and outer block shape and stride
    // e.g. for `mem_desc<32x256xf16, @block=[16, 8], @strides=[1, 32]>`
    // its memory layout tuple is ([2,32,16,8],[128,256,1,16])
    // for  `mem_desc<256x32xf16, @block=[8, 16]>` with default @stride[32, 1]
    // its memory layout tuple is ([32,2,8,16],[256,128,16,1])
    SmallVector<int64_t> getStrideShape();
    
    /// Generates instructions to compute the linearize offset 
    //  if the memory descriptor is blocked, it returns linearize offset based on the blocked layout
    //  the strides of memory descriptor is always considered regardless of blocked or not
    Value getLinearOffsets(OpBuilder &builder,
               Location loc, ArrayRef<OpFoldResult> offsets);


  }];

  let hasCustomAssemblyFormat = true;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD
