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
def XeGPU_BaseAddrType: AnyTypeOf<[Non0RankedMemRefOf<[XeGPU_ScalarType]>, UI64, UI32, I64, I32]>;
def XeGPU_DpasOpType: VectorOfRankAndType<[2, 3], [XeGPU_ScalarType]>;
def XeGPU_OffsetType: VectorOfRankAndType<[1], [Index]>;
def XeGPU_MaskType: AnyTypeOf<[VectorOfRankAndType<[1], [I1]>, I1]>;
def XeGPU_ValueType: AnyTypeOf<[VectorOfRankAndType<[1,2,3,4], [XeGPU_ScalarType]>, XeGPU_ScalarType]>;
def XeGPU_Vector2DType: VectorOfRankAndType<[2], [XeGPU_ScalarType]>;

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
    TensorDesc is a type designed to describe regions of the interested data as well as some
    features that are unique to Intel hardware. Different with the builtin tensor type in MLIR,
    it essentially only contains the meta data, and doesn't hold the data by itself. It is designed
    to mainly support 2D block load/store and DPAS (matrix multiplication instruction) on Intel GPU.
    It encodes the following information:

    * shape:  the sizes/shape of the intereted data block, e.g., 8x16 means 8 rows
              and each row contains 16 contiguous data element. The rows could be
              either contiguous or not, depends on whether the encoding attribute
              is set or not.
    * element_type: the data type of the data element, e.g., f16, f32.

    Similar to the builtin tensor, it also provides an optinal attribute to encoding
    the following information via the TensorDescAttr object:
    * memory_space (xegpu::MemorySpace): [optional] where the data is located,
                global memory or shared memory. It is default to Global.
    * array_length (int): [optional] The number of contiguous blocks with size as `shape`,
               that will be loaded by block load at a time. It is default to 1.
    * boundary_check (bool): [optional] indicates whether the operation detects the boundary
                and pads with zero for out-of-boundary access. It is default to do boundary check.


    Syntax:

    ```
    TensorDesc-type ::= `tensor_desc` `<` dim-list element-type (attr-list)? `>`
    element-type ::= float-type | integer-type | index-type
    dim-list := (static-dim-list `x`)?
    static-dim-list ::= decimal-literal `x` decimal-literal
    attr-list = (, memory_space = value)? (, arr_len = value)? (, boundary_check = value)? (, scattered = value)?
    ```

    Examples:

    ```mlir
    // A block TensorDesc with 8x16 i32 elements
    xegpu.tensor_desc<8x16xi32>

    // A block TensorDesc with 8x16 f32 elements
    xegpu.tensor_desc<8x16xf32>

    // A TensorDesc with 8x16 f32 elements for a memory region in shared memory space.
    xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_space = slm>>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">: $shape,
                        "mlir::Type": $elementType,
                        OptionalParameter<"mlir::Attribute">: $encoding);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>": $shape,
      "mlir::Type": $elementType,
      CArg<"int", "1">: $array_length,
      CArg<"bool", "true">: $boundary_check,
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space)>,
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>": $shape,
      "mlir::Type": $elementType,
      CArg<"int", "1">: $chunk_size,
      CArg<"xegpu::MemorySpace", "xegpu::MemorySpace::Global">:$memory_space)>
  ];

  let extraClassDeclaration = [{
    using TensorType::clone;
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

    BlockTensorDescAttr getEncodingAsBlockTensorDescAttr() const {
      return llvm::dyn_cast_if_present<BlockTensorDescAttr>(getEncoding());
    }

    ScatterTensorDescAttr getEncodingAsScatterTensorDescAttr() const {
      return llvm::dyn_cast_if_present<ScatterTensorDescAttr>(getEncoding());
    }

    xegpu::MemorySpace getMemorySpace() const {
      auto block_attr = getEncodingAsBlockTensorDescAttr();
      if (block_attr && block_attr.getMemorySpace())
        return block_attr.getMemorySpace().getValue();

      auto scatter_attr = getEncodingAsScatterTensorDescAttr();
      if (scatter_attr && scatter_attr.getMemorySpace())
        return scatter_attr.getMemorySpace().getValue();

      // return default value
      return MemorySpace::Global;
    }

    int getArrayLength() {
      auto attr = getEncoding();
      auto block_attr = mlir::dyn_cast_if_present<BlockTensorDescAttr>(attr);
      assert((!attr || block_attr) && "invalid on non BlockTensorDescAttr.");
      if (block_attr && block_attr.getArrayLength())
        return block_attr.getArrayLength().getInt();
      // return default value
      return 1;
    }

    bool getBoundaryCheck() {
      auto attr = getEncoding();
      auto block_attr = mlir::dyn_cast_if_present<BlockTensorDescAttr>(attr);
      assert((!attr || block_attr) && "invalid on non BlockTensorDescAttr.");
      if (block_attr && block_attr.getBoundaryCheck())
        return block_attr.getBoundaryCheck().getValue();
      // return default value
      return true;
    }

    bool isScattered() {
      return bool(getEncodingAsScatterTensorDescAttr());
    }

    int getChunkSize() {
      auto attr = getEncoding();
      auto scatter_attr = mlir::dyn_cast_if_present<ScatterTensorDescAttr>(attr);
      assert((!attr || scatter_attr) && "invalid on non ScatterTensorDescAttr.");
      if (scatter_attr && scatter_attr.getChunkSize())
        return scatter_attr.getChunkSize().getInt();
      return 1;
    }
  }];

  let hasCustomAssemblyFormat = true;

}


def XeGPU_Nbarrier: XeGPUTypeDef<"Nbarrier", "nbarrier", [], "mlir::Type"> {
  let summary = "!xegpu.nbarrier a custom XeGPU type representing a barrier.";

  let extraClassDeclaration = [{
    static NbarrierType get(mlir::MLIRContext *context) {
      return Base::get(context);
    };
  }];
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD
