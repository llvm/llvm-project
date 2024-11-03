//===- TestTransformDialectExtension.td --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations that are injected into the Transform
// dialect through the extension mechanism, as a test.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTRANSFORMDIALECTEXTENSION_TD
#define MLIR_TESTTRANSFORMDIALECTEXTENSION_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"

def TestTransformTestDialectHandleType
  : TypeDef<Transform_Dialect, "TestDialectOp",
      [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let description = [{Handle pointing to an op from the Test dialect.}];
  let mnemonic = "test_dialect_op";
  let assemblyFormat = "";
}

def TestTransformTestDialectParamType
  : TypeDef<Transform_Dialect, "TestDialectParam",
      [DeclareTypeInterfaceMethods<TransformParamTypeInterface>]> {
  let description = [{
    Parameter associated with an i32 attribute for testing purposes.
  }];
  let mnemonic = "test_dialect_param";
  let assemblyFormat = "";
}

def TestProduceSelfHandleOrForwardOperandOp
  : Op<Transform_Dialect, "test_produce_self_handle_or_forward_operand",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Optional<PDL_Operation>:$operand);
  let results = (outs PDL_Operation:$res);
  let assemblyFormat = "($operand^)? attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestProduceValueHandleToSelfOperand
  : Op<Transform_Dialect, "test_produce_value_handle_to_self_operand",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$in);
  let results = (outs TransformValueHandleTypeInterface:$out);
  let assemblyFormat = "$in attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  
}

def TestProduceValueHandleToResult
  : Op<Transform_Dialect, "test_produce_value_handle_to_result",
       [TransformEachOpTrait, TransformOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$in,
                       I64Attr:$number);
  let results = (outs TransformValueHandleTypeInterface:$out);
  let assemblyFormat = "$in `,` $number attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
    
def TestProduceValueHandleToArgumentOfParentBlock
  : Op<Transform_Dialect, "test_produce_value_handle_to_argument_of_parent_block",
       [TransformEachOpTrait, TransformOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$in,
                       I64Attr:$number);
  let results = (outs TransformValueHandleTypeInterface:$out);
  let assemblyFormat = "$in `,` $number attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestConsumeOperand : Op<Transform_Dialect, "test_consume_operand",
     [DeclareOpInterfaceMethods<TransformOpInterface>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    Transform_AnyHandleOrParamType:$operand,
    Optional<PDL_Operation>:$second_operand);
  let assemblyFormat = "$operand (`,` $second_operand^)? attr-dict `:` type($operand)";
  let cppNamespace = "::mlir::test";
}

def TestConsumeOperandOfOpKindOrFail
  : Op<Transform_Dialect, "test_consume_operand_of_op_kind_or_fail",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    PDL_Operation:$operand,
    StrAttr:$op_kind);
  let assemblyFormat = "$operand `,` $op_kind attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestPrintRemarkAtOperandOp
  : Op<Transform_Dialect, "test_print_remark_at_operand",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    TransformHandleTypeInterface:$operand,
    StrAttr:$message);
  let assemblyFormat =
    "$operand `,` $message attr-dict `:` type($operand)";
  let cppNamespace = "::mlir::test";
}

def TestPrintRemarkAtOperandValue
  : Op<Transform_Dialect, "test_print_remark_at_operand_value",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformValueHandleTypeInterface:$in,
                       StrAttr:$message);
  let assemblyFormat = "$in `,` $message attr-dict `:` type($in)";
  let cppNamespace = "::mlir::test";
}

def TestAddTestExtensionOp
  : Op<Transform_Dialect, "test_add_test_extension",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        NoMemoryEffect]> {
  let arguments = (ins StrAttr:$message);
  let assemblyFormat = "$message attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestCheckIfTestExtensionPresentOp
  : Op<Transform_Dialect, "test_check_if_test_extension_present",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins PDL_Operation:$operand);
  let assemblyFormat = "$operand attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestRemapOperandPayloadToSelfOp
  : Op<Transform_Dialect, "test_remap_operand_to_self",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins PDL_Operation:$operand);
  let results = (outs Optional<TransformHandleTypeInterface>:$out);        
  let assemblyFormat = "$operand attr-dict (`:` type($out)^)?";
  let cppNamespace = "::mlir::test";
}

def TestRemoveTestExtensionOp
  : Op<Transform_Dialect, "test_remove_test_extension",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        NoMemoryEffect]> {
  let assemblyFormat = "attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestReversePayloadOpsOp
  : Op<Transform_Dialect, "test_reverse_payload_ops",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestTransformOpWithRegions
  : Op<Transform_Dialect, "test_transform_op_with_regions",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let regions = (region AnyRegion:$first, AnyRegion:$second);
  let assemblyFormat = "attr-dict-with-keyword regions";
  let cppNamespace = "::mlir::test";
}

def TestBranchingTransformOpTerminator
  : Op<Transform_Dialect, "test_branching_transform_op_terminator",
       [Terminator, DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let successors = (successor VariadicSuccessor<AnySuccessor>:$succ);
  let cppNamespace = "::mlir::test";
}

def TestEmitRemarkAndEraseOperandOp
  : Op<Transform_Dialect, "test_emit_remark_and_erase_operand",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
  let arguments = (ins PDL_Operation:$target, StrAttr:$remark,
                   UnitAttr:$fail_after_erase);
  let assemblyFormat = "$target `,` $remark attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestWrongNumberOfResultsOp
  : Op<Transform_Dialect, "test_wrong_number_of_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$a,
                      PDL_Operation:$b,
                      PDL_Operation:$c);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestWrongNumberOfMultiResultsOp
  : Op<Transform_Dialect, "test_wrong_number_of_multi_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestCorrectNumberOfMultiResultsOp
  : Op<Transform_Dialect, "test_correct_number_of_multi_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$result1,
                      PDL_Operation:$result2);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestMixedNullAndNonNullResultsOp
  : Op<Transform_Dialect, "test_mixed_null_and_non_null_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$null,
                      PDL_Operation:$non_null);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestMixedSuccessAndSilenceableOp
  : Op<Transform_Dialect, "test_mixed_sucess_and_silenceable",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestPrintNumberOfAssociatedPayloadIROps
  : Op<Transform_Dialect, "test_print_number_of_associated_payload_ir_ops",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins PDL_Operation:$handle);
  let assemblyFormat = "$handle attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestCopyPayloadOp
  : Op<Transform_Dialect, "test_copy_payload",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins PDL_Operation:$handle);
  let results = (outs PDL_Operation:$copy);
  let cppNamespace = "::mlir::test";
  let assemblyFormat = "$handle attr-dict";
}

def TestReportNumberOfTrackedHandlesNestedUnder
  : Op<Transform_Dialect, "test_report_number_of_tracked_handles_nested_under",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins PDL_Operation:$target);
  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestPrintParamOp
  : Op<Transform_Dialect, "test_print_param",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformParamTypeInterface:$param);
  let assemblyFormat = "$param attr-dict `:` type($param)";
  let cppNamespace = "::mlir::test";
}

def TestAddToParamOp
  : Op<Transform_Dialect, "test_add_to_param",
       [MemoryEffectsOpInterface, ParamProducerTransformOpTrait,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins Optional<TestTransformTestDialectParamType>:$param,
                       I32Attr:$addendum);
  let results = (outs TestTransformTestDialectParamType:$result);
  let assemblyFormat = "($param^ `,`)? $addendum attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestProduceParamWithNumberOfTestOps
  : Op<Transform_Dialect, "test_produce_param_with_number_of_test_ops",
       [MemoryEffectsOpInterface, ParamProducerTransformOpTrait,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$handle);
  let results = (outs TestTransformTestDialectParamType:$result);
  let assemblyFormat = "$handle attr-dict `:` type($handle)";
  let cppNamespace = "::mlir::test";
}

def TestProduceIntegerParamWithTypeOp
  : Op<Transform_Dialect, "test_produce_integer_param_with_type",
       [MemoryEffectsOpInterface, ParamProducerTransformOpTrait,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TypeAttr:$type);
  let results = (outs TransformParamTypeInterface:$result);
  let assemblyFormat = "$type attr-dict `:` type($result)";
  let cppNamespace = "::mlir::test";
  let hasVerifier = 1;
}

def TestProduceTransformParamOrForwardOperandOp
  : Op<Transform_Dialect, "test_produce_transform_param_or_forward_operand",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins TransformHandleTypeInterface:$in,
                       UnitAttr:$first_result_is_param,
                       UnitAttr:$first_result_is_null,
                       UnitAttr:$second_result_is_handle);
  let results = (outs AnyType:$out,
                      TransformParamTypeInterface:$param);
  let assemblyFormat = "$in attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestProduceNullPayloadOp
  : Op<Transform_Dialect, "test_produce_null_payload",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let results = (outs TransformHandleTypeInterface:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
  let cppNamespace = "::mlir::test";
}

def TestProduceNullParamOp
  : Op<Transform_Dialect, "test_produce_null_param",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let results = (outs TransformParamTypeInterface:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
  let cppNamespace = "::mlir::test";
}

def TestProduceNullValueOp
  : Op<Transform_Dialect, "test_produce_null_value",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let results = (outs TransformValueHandleTypeInterface:$out);
  let assemblyFormat = "attr-dict `:` type($out)";
  let cppNamespace = "::mlir::test";
}

def TestRequiredMemoryEffectsOp
  : Op<Transform_Dialect, "test_required_memory_effects",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$in,
                       UnitAttr:$has_operand_effect,
                       UnitAttr:$has_result_effect,
                       UnitAttr:$modifies_payload);
  let results = (outs TransformHandleTypeInterface:$out);
  let assemblyFormat = "$in attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
}

#endif // MLIR_TESTTRANSFORMDIALECTEXTENSION_TD
