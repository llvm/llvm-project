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
include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
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
  let arguments = (ins Optional<TransformHandleTypeInterface>:$operand);
  let results = (outs TransformHandleTypeInterface:$res);
  let assemblyFormat =
      "($operand^)? attr-dict `:` functional-type($operand, $res)";
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
        ::mlir::transform::TransformRewriter &rewriter,
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
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestConsumeOperand : Op<Transform_Dialect, "test_consume_operand",
     [DeclareOpInterfaceMethods<TransformOpInterface, ["allowsRepeatedHandleOperands"]>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    Transform_AnyHandleOrParamType:$operand,
    Optional<TransformHandleTypeInterface>:$second_operand,
    UnitAttr:$allow_repeated_handles);
  let assemblyFormat = 
      "$operand (`,` $second_operand^)? attr-dict `:` type($operand)"
      "(`,` type($second_operand)^)?";
  let cppNamespace = "::mlir::test";
}

def TestConsumeOperandEach : Op<Transform_Dialect, "test_consume_operand_each",
     [TransformOpInterface, TransformEachOpTrait,
      MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let assemblyFormat = "$target attr-dict `:` type($target)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state) {
      return ::mlir::DiagnosedSilenceableFailure::success();
    }
  }];
}

def TestConsumeOperandOfOpKindOrFail
  : Op<Transform_Dialect, "test_consume_operand_of_op_kind_or_fail",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    TransformHandleTypeInterface:$operand,
    StrAttr:$op_kind);
  let assemblyFormat =
      "$operand `,` $op_kind attr-dict `:` type($operand)";
  let cppNamespace = "::mlir::test";
}

def TestSucceedIfOperandOfOpKind
  : Op<Transform_Dialect, "test_succeed_if_operand_of_op_kind",
       [MatchOpInterface,
        SingleOpMatcher,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    TransformHandleTypeInterface:$operand_handle,
    StrAttr:$op_kind);
  let assemblyFormat =
      "$operand_handle `,` $op_kind attr-dict `:` type($operand_handle)";
  let extraClassDeclaration = SingleOpMatcher.extraDeclaration;
  let cppNamespace = "::mlir::test";
}

def TestPrintRemarkAtOperandOp
  : Op<Transform_Dialect, "test_print_remark_at_operand",
       [MatchOpInterface,
        DeclareOpInterfaceMethods<TransformOpInterface>,
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
  let arguments = (ins TransformHandleTypeInterface:$operand);
  let assemblyFormat = "$operand attr-dict `:` type($operand)";
  let cppNamespace = "::mlir::test";
}

def TestRemapOperandPayloadToSelfOp
  : Op<Transform_Dialect, "test_remap_operand_to_self",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$operand);
  let results = (outs Optional<TransformHandleTypeInterface>:$out);        
  let assemblyFormat = "$operand attr-dict `:` functional-type($operand, $out)";
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
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
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
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     FunctionalStyleTransformOpTrait]> {
  let arguments = (ins TransformHandleTypeInterface:$target, StrAttr:$remark,
                   UnitAttr:$fail_after_erase);
  let assemblyFormat = "$target `,` $remark attr-dict `:` type($target)";
  let cppNamespace = "::mlir::test";
}

def TestWrongNumberOfResultsOp
  : Op<Transform_Dialect, "test_wrong_number_of_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$a,
                      TransformHandleTypeInterface:$b,
                      TransformHandleTypeInterface:$c);
  let assemblyFormat =
      "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestWrongNumberOfMultiResultsOp
  : Op<Transform_Dialect, "test_wrong_number_of_multi_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target attr-dict `:` functional-type($target, $result)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestCorrectNumberOfMultiResultsOp
  : Op<Transform_Dialect, "test_correct_number_of_multi_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result1,
                      TransformHandleTypeInterface:$result2);
  let assemblyFormat =
      "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestMixedNullAndNonNullResultsOp
  : Op<Transform_Dialect, "test_mixed_null_and_non_null_results",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$null,
                      TransformHandleTypeInterface:$non_null);
  let assemblyFormat =
      "$target attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestMixedSuccessAndSilenceableOp
  : Op<Transform_Dialect, "test_mixed_success_and_silenceable",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     TransformEachOpTrait, TransformOpInterface]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation * target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def TestPrintNumberOfAssociatedPayloadIROps
  : Op<Transform_Dialect, "test_print_number_of_associated_payload_ir_ops",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$handle);
  let assemblyFormat = "$handle attr-dict `:` type($handle)";
  let cppNamespace = "::mlir::test";
}

def TestPrintNumberOfAssociatedPayloadIRValues
  : Op<Transform_Dialect, "test_print_number_of_associated_payload_ir_values",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformValueHandleTypeInterface:$value_handle);
  let assemblyFormat = "$value_handle attr-dict `:` type($value_handle)";
  let cppNamespace = "::mlir::test";
}

def TestPrintNumberOfAssociatedPayloadIRParams
  : Op<Transform_Dialect, "test_print_number_of_associated_payload_ir_params",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformParamTypeInterface:$param);
  let assemblyFormat = "$param attr-dict `:` type($param)";
  let cppNamespace = "::mlir::test";
}

def TestCopyPayloadOp
  : Op<Transform_Dialect, "test_copy_payload",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$handle);
  let results = (outs TransformHandleTypeInterface:$copy);
  let cppNamespace = "::mlir::test";
  let assemblyFormat =
      "$handle attr-dict `:` functional-type(operands, results)";
}

def TestReportNumberOfTrackedHandlesNestedUnder
  : Op<Transform_Dialect, "test_report_number_of_tracked_handles_nested_under",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let assemblyFormat = "$target attr-dict `:` type($target)";
  let cppNamespace = "::mlir::test";
}

def TestPrintParamOp
  : Op<Transform_Dialect, "test_print_param",
       [MatchOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformParamTypeInterface:$param,
                       Optional<TransformHandleTypeInterface>:$anchor,
                       OptionalAttr<StrAttr>:$message);
  let assemblyFormat = "$param (`,` $message^)?  (`at` $anchor^)?"
                       "attr-dict `:` type($param) (`,` type($anchor)^)?";
  let cppNamespace = "::mlir::test";
}

def TestAddToParamOp
  : Op<Transform_Dialect, "test_add_to_param",
       [MatchOpInterface, 
        MemoryEffectsOpInterface,
        ParamProducerTransformOpTrait,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins Optional<TestTransformTestDialectParamType>:$param,
                       I32Attr:$addendum);
  let results = (outs TestTransformTestDialectParamType:$result);
  let assemblyFormat = "($param^ `,`)? $addendum attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestProduceParamWithNumberOfTestOps
  : Op<Transform_Dialect, "test_produce_param_with_number_of_test_ops",
       [MatchOpInterface,
        MemoryEffectsOpInterface,
        ParamProducerTransformOpTrait,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$handle);
  let results = (outs TestTransformTestDialectParamType:$result);
  let assemblyFormat = "$handle attr-dict `:` type($handle)";
  let cppNamespace = "::mlir::test";
}

def TestProduceParamOp
  : Op<Transform_Dialect, "test_produce_param",
       [MemoryEffectsOpInterface, ParamProducerTransformOpTrait,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins AnyAttr:$attr);
  let results = (outs TransformParamTypeInterface:$result);
  let assemblyFormat = "`(` $attr `)` attr-dict `:` type($result)";
  let cppNamespace = "::mlir::test";
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
        ::mlir::transform::TransformRewriter &rewriter,
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

def TestProduceEmptyPayloadOp
  : Op<Transform_Dialect, "test_produce_empty_payload",
      [DeclareOpInterfaceMethods<TransformOpInterface>,
       MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
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

// This op is used as a payload op. It must be a registered op, so that it can
// be created with "RewriterBase::replaceOpWithNewOp" (needed for a test case).
// Since only TransformOpInterface can be injected into the transform dialect,
// this op implements the interface, even though it is not used as a transform
// op.
def TestDummyPayloadOp
  : Op<Transform_Dialect, "test_dummy_payload_op",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       TransformOpInterface]> {
  let arguments = (ins Variadic<AnyType>:$args,
                       UnitAttr:$fail_to_verify);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = "$args attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    DiagnosedSilenceableFailure apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
      llvm_unreachable("op should not be used as a transform");
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  }];
}

def TestTrackedRewriteOp
  : Op<Transform_Dialect, "test_tracked_rewrite",
      [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$in);
  let results = (outs);
  let assemblyFormat = "$in attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
}

def ApplyTestPatternsOp
  : Op<Transform_Dialect, "apply_patterns.transform.test_patterns",
      [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let cppNamespace = "::mlir::test";
}

def ApplyTestConversionPatternsOp
  : Op<Transform_Dialect, "apply_conversion_patterns.transform.test_conversion_patterns",
      [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface>]> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestTypeConverterOp
  : Op<Transform_Dialect, "apply_conversion_patterns.transform.test_type_converter",
      [DeclareOpInterfaceMethods<TypeConverterBuilderOpInterface>]> {
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestReEnterRegionOp
  : Op<Transform_Dialect, "test_re_enter_region",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "($args^ `:` type($args))? attr-dict-with-keyword regions";
  let cppNamespace = "::mlir::test";
  let hasVerifier = 1;
}

def TestNotifyPayloadOpReplacedOp
  : Op<Transform_Dialect, "test_notify_payload_op_replaced",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins TransformHandleTypeInterface:$original,
                       TransformHandleTypeInterface:$replacement);
  let results = (outs);
  let assemblyFormat = "$original `,` $replacement attr-dict `:` functional-type(operands, results)";
  let cppNamespace = "::mlir::test";
}

def TestProduceInvalidIR
  : Op<Transform_Dialect, "test_produce_invalid_ir",
       [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        TransformOpInterface, TransformEachOpTrait]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";
  let cppNamespace = "::mlir::test";
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // MLIR_TESTTRANSFORMDIALECTEXTENSION_TD
