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

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"

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

def TestConsumeOperandIfMatchesParamOrFail
  : Op<Transform_Dialect, "test_consume_operand_if_matches_param_or_fail",
       [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins 
    Arg<PDL_Operation, "",
        [TransformMappingWrite, TransformMappingFree]>:$operand,
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


#endif // MLIR_TESTTRANSFORMDIALECTEXTENSION_TD
