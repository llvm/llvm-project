//===- XeGPUTransformOps.td - XeGPU transformation ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XEGPU_TRANSFORM_OPS
#define XEGPU_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def SetDescLayoutOp : Op<Transform_Dialect, "xegpu.set_desc_layout", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Set xegpu.layout attribute to a xegpu.create_nd_desc op result.";
  let description = [{
    Given an `xegpu.create_nd_desc` operation, this transform adds `xegpu.layout`
    attribute to the result tensor descriptor. The layout is defined by the
    `sg_layout`, and `sg_data` and optional `inst_data` attributes. Returns a handle
    to the transformed op.
  }];

  let arguments = (ins
                   TransformHandleTypeInterface:$target,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_inst_data
                   );

  let results = (outs TransformHandleTypeInterface:$transformed);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData,
                   "ArrayRef<OpFoldResult>":$mixedInstData
                   )>,
  ];

  let assemblyFormat = [{
    $target
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    (`inst_data` `=` custom<DynamicIndexList>($inst_data, $static_inst_data)^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticSgLayout(), getSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticSgData(), getSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticInstData(), getInstData(), b);
    }
  }];
}

def SetOpLayoutAttrOp : Op<Transform_Dialect, "xegpu.set_op_layout_attr", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Set xegpu.layout attribute of an op.";
  let description = [{
    Sets the `xegpu.layout` attribute of an op. If `result=true`, sets the
    `layout_result_{index}`, otherwise `layout_operand_{index}` attribute. The
    target operand/result value is defined by the `index` argument. The layout
    is defined by the `sg_layout`, `sg_data` and optional `inst_data` attributes.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedOptionalAttr<I64Attr, "0">:$index,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_inst_data,
                   DefaultValuedAttr<UnitAttr, "false">:$result
                   );

  let results = (outs);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "int64_t":$index,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData,
                   "ArrayRef<OpFoldResult>":$mixedInstData,
                   CArg<"bool", "false">:$result
                   )>,
  ];

  let assemblyFormat = [{
    $target (`result` $result^)? (`index` `=` $index^)?
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    (`inst_data` `=` custom<DynamicIndexList>($inst_data, $static_inst_data)^)?
    attr-dict `:` qualified(type(operands))
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticSgLayout(), getSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticSgData(), getSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticInstData(), getInstData(), b);
    }
  }];
}

#endif // XEGPU_TRANSFORM_OPS
