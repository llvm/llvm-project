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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"

def TestTransformTestDialectHandleType
  : TypeDef<Transform_Dialect, "TestDialectOp",
      [DeclareTypeInterfaceMethods<TransformTypeInterface>]> {
  let description = [{Handle pointing to an op from the Test dialect.}];
  let mnemonic = "test_dialect_op";
  let assemblyFormat = "";
}

def TestProduceParamOrForwardOperandOp
  : Op<Transform_Dialect, "test_produce_param_or_forward_operand",
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins 
     Arg<Optional<PDL_Operation>, "", [TransformMappingRead]>:$operand,
     OptionalAttr<I64Attr>:$parameter);
  let results = (outs 
     Res<PDL_Operation, "",
         [TransformMappingAlloc, TransformMappingWrite]>:$res);
  let assemblyFormat = "(`from` $operand^)? ($parameter^)? attr-dict";
  let cppNamespace = "::mlir::test";
  let hasVerifier = 1;
}

def TestConsumeOperand : Op<Transform_Dialect, "test_consume_operand",
     [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins
    Arg<PDL_Operation, "",
        [TransformMappingRead, TransformMappingFree]>:$operand,
    Arg<Optional<PDL_Operation>, "",
        [TransformMappingRead, TransformMappingFree]>:$second_operand);
  let assemblyFormat = "$operand (`,` $second_operand^)? attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestConsumeOperandIfMatchesParamOrFail
  : Op<Transform_Dialect, "test_consume_operand_if_matches_param_or_fail",
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins 
    Arg<PDL_Operation, "",
        [TransformMappingRead, TransformMappingFree]>:$operand,
    I64Attr:$parameter);
  let assemblyFormat = "$operand `[` $parameter `]` attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestPrintRemarkAtOperandOp
  : Op<Transform_Dialect, "test_print_remark_at_operand",
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins 
    Arg<PDL_Operation, "",
        [TransformMappingRead, PayloadIRRead]>:$operand,
    StrAttr:$message);
  let assemblyFormat = "$operand `,` $message attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestAddTestExtensionOp
  : Op<Transform_Dialect, "test_add_test_extension",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        NoSideEffect]> {
  let arguments = (ins StrAttr:$message);
  let assemblyFormat = "$message attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestCheckIfTestExtensionPresentOp
  : Op<Transform_Dialect, "test_check_if_test_extension_present",
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins
    Arg<PDL_Operation, "", [TransformMappingRead, PayloadIRRead]>:$operand);
  let assemblyFormat = "$operand attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestRemapOperandPayloadToSelfOp
  : Op<Transform_Dialect, "test_remap_operand_to_self",
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins
    Arg<PDL_Operation, "",
        [TransformMappingRead, TransformMappingWrite, PayloadIRRead]>:$operand);
  let assemblyFormat = "$operand attr-dict";
  let cppNamespace = "::mlir::test";
}

def TestRemoveTestExtensionOp
  : Op<Transform_Dialect, "test_remove_test_extension",
       [DeclareOpInterfaceMethods<TransformOpInterface>,
        NoSideEffect]> {
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
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
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
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
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
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
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
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
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
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
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
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins Arg<PDL_Operation, "", [TransformMappingRead]>:$handle);
  let results = (outs Res<PDL_Operation, "",
      [TransformMappingAlloc, TransformMappingWrite]>:$copy);
  let cppNamespace = "::mlir::test";
  let assemblyFormat = "$handle attr-dict";
}

#endif // MLIR_TESTTRANSFORMDIALECTEXTENSION_TD
