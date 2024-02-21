//===- XeGPUTypes.td - XeGPU dialect types definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD

include "mlir/IR/BuiltinTypes.td"

include "mlir/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "mlir/Dialect/XeGPU/IR/XeGPUDialect.td"

// An Integer array attribute with fixed 2 elements.
def XeGPU_IntType: AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;
def XeGPU_FloatType: AnyTypeOf<[F16, F32, F64, BF16, TF32]>;
def XeGPU_ScalarType: AnyTypeOf<[XeGPU_IntType, XeGPU_FloatType]>;
def XeGPU_BaseAddrType: AnyTypeOf<[MemRefRankOf<[XeGPU_ScalarType], [1, 2]>, UI64, UI32, I64, I32]>;
def XeGPU_DpasOpType: VectorOfRankAndType<[2, 3], [XeGPU_ScalarType]>;
// def XeGPU_OffsetType: AnyTypeOf<[VectorOfRankAndType<[1], [Index]>, Index]>;
def XeGPU_OffsetType: VectorOfRankAndType<[1], [Index]>;
def XeGPU_MaskType: AnyTypeOf<[VectorOfRankAndType<[1,2], [I1]>, I1]>;
def XeGPU_ValueType: AnyTypeOf<[VectorOfRankAndType<[1,2,3,4], [XeGPU_ScalarType]>, XeGPU_ScalarType]>;

def XeGPU_Vector2DType: VectorOfRankAndType<[2], [XeGPU_ScalarType]>;

// common base class for types in XeGPU dialect
class XeGPUTypeDef<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<XeGPU_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

// TensorDesc contains dim and element type info
def XeGPU_TensorDesc: XeGPUTypeDef<"TensorDesc", "tensor_desc",
        [ShapedTypeInterface], "::mlir::TensorType"> {
  let summary = "TensorDesc describing all kinds of memory and tensors, including scatter tensor, 1d tensor, 2d tensor, … 5d tensor";
  let description = [{
    TensorDesc is a type designed to describe all kinds of memory, scatter tensor, 1d tensor, 2d tensor, … 5d tensor.
    Different with the builtin tensor type in MLIR, it essentially only contains the meta data that describes a region
    of the intereted data as well as some features that are unique to intel hardware features. It does not hold the data
    directly by itself. It is designed to mainly support 2d block load/store and DPAS (matrix multiplication instruction)
    on Intel GPU. It majorly encodes the following information:

    * shape:  the sizes/shape of the intereted data block, e.g., 8x16 means 8 rows
              and each row contains 16 continious data element. The rows could be
              either continuous or not, depends on whether the encoding attribute
              is set or not.
    * element_type: the data type of the data element, e.g., f16, f32.

    Similar to the builtin tensor, it also provides an optinal attribute to encoding the following information via the TensorDescAttr object:
    * memory_scope (xegpu::MemoryScope): [optional] where the data is located, global memory or shared memory. It is default to Global.
    * array_length (int): [optional] The number of continuous blocks with size as `shape`,
               that will be loaded by block load at a time. It is default to 1.
    * boundary_check (bool): [optional] indicates whether the operation detects the boundary and pads with zero for out-of-boundary access (default)
    * scattered (xegpu::ScatteredAttr): [optional] It is a unit attribute. It can be only set as empty or ScatteredAttr, indicating
                whether the TensorDesc is blocked (empty, default) or scattered (ScatteredAttr). If it is
                blocked, rows are continuous in the correspoding dimention, otherwise, rows may be not continous.
    * mapping (xegpu::SubGroupMapAttr): [optional] Used to guide compiler to distribute the workload into different threads. It is default to none.

    For convenience, its attribute field can also take either "ScatteredAttr" or "SubGroupMapAttr" directly if and only
    if others are taking default values.

    Syntax:

    ```
    TensorDesc-type ::= `tensor_desc` `<` dim-list element-type (attr-list)? `>`
    element-type ::= float-type | integer-type | index-type
    dim-list := (static-dim-list `x`)?
    static-dim-list ::= decimal-literal `x` decimal-literal
    attr-list = (, memory_scope = value)? (, arr_len = value)? (, ScatteredAttr)? (, mapping)?
    ```

    Examples:

    ```mlir
    // A block TensorDesc with 3x42 i32 elements
    xegpu.tensor_desc<3x42xi32>

    // A block TensorDesc with 4x5 f32 elements
    xegpu.tensor_desc<4x5xf32>

    // A Scattered TensorDesc with 16x4 f32 elements
    xegpu.tensor_desc<16x4xf32, #!xegpu.scattered>

    // A TensorDesc with 8x16 f16 elements.
    // It will be distributed accross 16 hardware threads, organized as [2, 8],
    // and each access 2 continious elements in dim 1.
    xegpu.tensor_desc<8x16xf16, #xegpu.sg_map<wi_layout = [2, 8], wi_data = [1, 2]>>

    // A TensorDesc with 8x16 f32 elements for a memory region in shared memory space.
    xegpu.tensor_desc<8x16xf32, #xegpu.tdesc_attr<memory_scope = slm>>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">: $shape,
                        "mlir::Type": $elementType,
                        OptionalParameter<"mlir::Attribute">: $encoding);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>":$shape, "mlir::Type":$elementType,
      CArg<"mlir::Attribute", "{}"> : $encoding
    )>,
    TypeBuilder<(ins
      "llvm::ArrayRef<int64_t>": $shape, "mlir::Type": $elementType,
      "mlir::xegpu::MemoryScopeKind": $memory_scope, "int": $array_length,
      "bool": $boundary_check, "mlir::xegpu::ScatteredAttr": $scattered,
      "mlir::xegpu::SubGroupMapAttr": $mapping
    )>,
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<int64_t>": $shape, "mlir::Type": $elementType,
      "mlir::xegpu::MemoryScopeKind": $memory_scope, "int": $array_length,
      "bool": $boundary_check, "mlir::xegpu::ScatteredAttr": $scattered,
      "mlir::xegpu::SubGroupMapAttr": $mapping
    )>
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

    TensorDescAttr getEncodingAsTensorDescAttr() const {
      return llvm::dyn_cast_if_present<TensorDescAttr>(getEncoding());
    }

    SubGroupMapAttr getEncodingAsMapAttr() const {
      return llvm::dyn_cast_if_present<SubGroupMapAttr>(getEncoding());
    }

    ScatteredAttr getEncodingAsScatteredAttr() const {
      return llvm::dyn_cast_if_present<ScatteredAttr>(getEncoding());
    }

    xegpu::MemoryScopeKind getMemoryScope();
    int getArrayLength();
    bool getBoundaryCheck();
    xegpu::ScatteredAttr getScattered();
    xegpu::SubGroupMapAttr getMapping();
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
