//===- FuncTransformOps.td - CF transformation ops -*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FUNC_TRANSFORM_OPS
#define FUNC_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/OpBase.td"

def ApplyFuncToLLVMConversionPatternsOp : Op<Transform_Dialect,
    "apply_conversion_patterns.func.func_to_llvm",
    [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface,
                               ["verifyTypeConverter"]>]> {
  let description = [{
    Collects patterns that convert Func dialect ops to LLVM dialect ops.
    These patterns require an "LLVMTypeConverter".
  }];

  let assemblyFormat = "attr-dict";
}

def CastAndCallOp : Op<Transform_Dialect,
    "func.cast_and_call",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     AttrSizedOperandSegments,
     ReportTrackingListenerFailuresOpTrait]
        # GraphRegionNoTerminator.traits> {
  let summary = "Casts values to the signature of a function and replaces them "
                "with a call";
  let description = [{
    This transform takes value handles to a set of `inputs` and `outputs` and
    attempts to cast them to the function signature of the attached function
    op, then builds a call to the function and replaces the users of the
    outputs. It is the responsibility of the user to ensure that the slice of
    the program replaced by this operation makes sense, i.e. there is no
    verification that the inputs to this operation have any relation to the
    outputs outside of basic dominance requirements needed for the call.

    The casting materialization functions are specified in the graph region of
    this op. They must implement the `TypeConverterBuilderOpInterface`. The
    order of ops within the region is irrelevant.

    The target function can be specified by a symbol name or by a handle to the
    operation.

    This transform only reads the operand handles and only replaces the users of
    the outputs with the results of the call. No handles are consumed and no
    operations are removed. Users are expected to run cleanup separately if
    desired.

    Warning: The replacement of the uses of the outputs could invalidate certain
    restricted value handle types (e.g. `transform.block_arg` if it existed, by
    replacing the use with something not coming from a block argument). The
    value will still exist in such cases but wouldn't verify against the type.
    See the discussion here for more information:
    https://github.com/llvm/llvm-project/pull/78398#discussion_r1455070087

    This transform will emit a silenceable failure if:
     - The set of outputs isn't unique
     - The handle for the insertion point does not include exactly one operation
     - The insertion point op does not dominate any of the output users
     - The insertion point op is not dominated by any of the inputs
     - The function signature does not match the number of inputs/outputs

    This transform will emit a definite failure if it fails to resolve the
    target function, or if it fails to materialize the conversion casts of
    either the inputs to the function argument types, or the call results to
    the output types.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$insertion_point,
    UnitAttr:$insert_after,
    Optional<TransformValueHandleTypeInterface>:$inputs,
    Optional<TransformValueHandleTypeInterface>:$outputs,
    OptionalAttr<SymbolRefAttr>:$function_name,
    Optional<TransformHandleTypeInterface>:$function);
  let results = (outs TransformHandleTypeInterface:$result);
  let regions = (region MaxSizedRegion<1>:$conversions);

  let assemblyFormat = [{
    ($function_name^)? ($function^)?
    ( `(` $inputs^ `)` )?
    ( `->` $outputs^ )?
    (`after` $insert_after^):(`before`)? $insertion_point
    ($conversions^)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // FUNC_TRANSFORM_OPS
