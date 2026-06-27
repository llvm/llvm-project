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

def GetLoadOp : Op<Transform_Dialect, "xegpu.get_load_op", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  NavigationTransformOpTrait, MemoryEffectsOpInterface
]> {

  let summary = "Get a handle to the load_nd op in producer chain of a value.";
  let description = [{
    Traces the producers of the given value until an `xegpu.load_nd` or
    `xegpu.load` op is found. Returns a handle to it. Currently traces
    producers by following only the first operand of producer ops.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$target);

  let results = (outs TransformHandleTypeInterface:$loadNdHandle);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
}

def SetAnchorLayoutOp : Op<Transform_Dialect, "xegpu.set_anchor_layout", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Set anchor layout of an op.";
  let description = [{
    Sets the `xegpu.layout` anchor layout for XeGPU ops that support it. The
    target operand value can be set by the `index` argument (currently only
    applicable to a DPAS op). The layout is defined by the `sg_layout`,
    `sg_data` and optional `inst_data` and `order` attributes. If `slice_dims`
    is provided, the `xegpu.layout` attribute is wrapped in an
    `xegpu.slice<..., dims=slice_dims>` attribute. Emits a silenceable failure
    if the target op does not support anchor layouts.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   DefaultValuedOptionalAttr<I64Attr, "0">:$index,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_inst_data,
                   DefaultValuedOptionalAttr<DenseI32ArrayAttr, "{}">:$order,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$slice_dims
                   );

  let results = (outs);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "int64_t":$index,
                   "ArrayRef<OpFoldResult>":$mixedSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedSgData,
                   "ArrayRef<OpFoldResult>":$mixedInstData,
                   "ArrayRef<int32_t>":$order,
                   "ArrayRef<int64_t>":$sliceDims
                   )>,
  ];

  let assemblyFormat = [{
    $target (`index` `=` $index^)?
    `sg_layout` `=` custom<DynamicIndexList>($sg_layout, $static_sg_layout)
    `sg_data` `=` custom<DynamicIndexList>($sg_data, $static_sg_data)
    (`inst_data` `=` custom<DynamicIndexList>($inst_data, $static_inst_data)^)?
    (`order` `=` $order^)?
    (`slice_dims` `=` $slice_dims^)?
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

def SetGPULaunchThreadsOp
    : Op<Transform_Dialect, "xegpu.set_gpu_launch_threads", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      TransformOpInterface
    ]> {

  let summary = "Set number of threads for a given gpu.launch operation";
  let description = [{
    Overrides the x,y,z threads operands of a given `gpu.launch` operation in-place.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$threads,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_threads
                   );
  let results = (outs);
  let builders = [
    OpBuilder<(ins "Value":$target, "ArrayRef<OpFoldResult>":$mixedThreads)>,
  ];

  let assemblyFormat = [{
    $target
    `threads` `=` custom<DynamicIndexList>($threads, $static_threads)
    attr-dict `:` qualified(type(operands))
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedThreads() {
      Builder b(getContext());
      return getMixedValues(getStaticThreads(), getThreads(), b);
    }
  }];
}

def InsertPrefetchOp : Op<Transform_Dialect, "xegpu.insert_prefetch", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Adds xegpu prefetch ops to a load op.";
  let description = [{
    Inserts `xegpu.prefetch_nd` operations for the given `xegpu.load_nd` op.
    The load op must reside within the `scf.for` loop. Number of prefetch steps
    is set by the `nb_prefetch` argument (default value is 1). Returns a handle
    to the created `xegpu.create_nd_desc` op.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   Optional<TransformAnyParamTypeOrAnyHandle>:$dynamic_nb_prefetch,
                   DefaultValuedOptionalAttr<I64Attr, "1">:$static_nb_prefetch
                   );

  let results = (outs TransformHandleTypeInterface:$desc_op);

  let assemblyFormat = [{
    $target
    `nb_prefetch` `=` ($dynamic_nb_prefetch^):($static_nb_prefetch)?
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    OpFoldResult getNbPrefetch() {
      auto cxt = getContext();
      if (getDynamicNbPrefetch())
        return OpFoldResult(getDynamicNbPrefetch());
      return OpFoldResult(IntegerAttr::get(
                          IntegerType::get(cxt, 64), getStaticNbPrefetch()));
    }
  }];
}

def ConvertLayoutOp : Op<Transform_Dialect, "xegpu.convert_layout", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface
]> {

  let summary = "Convert xegpu.layout attribute for a value.";
  let description = [{
    Adds an `xegpu.convert_layout` op to convert the `xegpu.layout` attribute
    of a value before its first use. The input and target layouts are defined
    by the `*sg_layout`, `*sg_data` and optional `*inst_data` and `*order`
    attributes. Returns a handle to the emitted `xegpu.convert_layout` op.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$target,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$input_sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$input_sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$input_inst_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$target_sg_layout,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$target_sg_data,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$target_inst_data,
                   DefaultValuedOptionalAttr<DenseI32ArrayAttr, "{}">:$input_order,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_input_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_input_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_input_inst_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_target_sg_layout,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_target_sg_data,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_target_inst_data,
                   DefaultValuedOptionalAttr<DenseI32ArrayAttr, "{}">:$target_order
                   );

  let results = (outs TransformHandleTypeInterface:$newConvertOp);
  let builders = [
    OpBuilder<(ins "Value":$target,
                   "ArrayRef<OpFoldResult>":$mixedInputSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedInputSgData,
                   "ArrayRef<OpFoldResult>":$mixedInputInstData,
                   "ArrayRef<int32_t>":$inputOrder,
                   "ArrayRef<OpFoldResult>":$mixedTargetSgLayout,
                   "ArrayRef<OpFoldResult>":$mixedTargetSgData,
                   "ArrayRef<OpFoldResult>":$mixedTargetInstData,
                   "ArrayRef<int32_t>":$targetOrder
                   )>,
  ];

  let assemblyFormat = [{
    $target
    `input_sg_layout` `=` custom<DynamicIndexList>($input_sg_layout, $static_input_sg_layout)
    `input_sg_data` `=` custom<DynamicIndexList>($input_sg_data, $static_input_sg_data)
    (`input_inst_data` `=` custom<DynamicIndexList>($input_inst_data, $static_input_inst_data)^)?
    (`input_order` `=` $input_order^)?
    `target_sg_layout` `=` custom<DynamicIndexList>($target_sg_layout, $static_target_sg_layout)
    `target_sg_data` `=` custom<DynamicIndexList>($target_sg_data, $static_target_sg_data)
    (`target_inst_data` `=` custom<DynamicIndexList>($target_inst_data, $static_target_inst_data)^)?
    (`target_order` `=` $target_order^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInputSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticInputSgLayout(), getInputSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInputSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticInputSgData(), getInputSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedInputInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticInputInstData(), getInputInstData(), b);
    }

    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedTargetSgLayout() {
      Builder b(getContext());
      return getMixedValues(getStaticTargetSgLayout(), getTargetSgLayout(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedTargetSgData() {
      Builder b(getContext());
      return getMixedValues(getStaticTargetSgData(), getTargetSgData(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult> getMixedTargetInstData() {
      Builder b(getContext());
      return getMixedValues(getStaticTargetInstData(), getTargetInstData(), b);
    }
  }];
}

#endif // XEGPU_TRANSFORM_OPS
