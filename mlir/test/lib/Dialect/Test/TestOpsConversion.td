//===- TestOpsConversion.td - Test dialect conversion ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_CONVERSION
#define TEST_OPS_CONVERSION

include "TestOpBase.td"

//===----------------------------------------------------------------------===//
// Test Legalization
//===----------------------------------------------------------------------===//

def ILLegalOpA : TEST_Op<"illegal_op_a">, Results<(outs I32)>;
def ILLegalOpB : TEST_Op<"illegal_op_b">, Results<(outs I32)>;
def ILLegalOpC : TEST_Op<"illegal_op_c">, Results<(outs I32)>;
def ILLegalOpD : TEST_Op<"illegal_op_d">, Results<(outs I32)>;
def ILLegalOpE : TEST_Op<"illegal_op_e">, Results<(outs I32)>;
def ILLegalOpF : TEST_Op<"illegal_op_f">, Results<(outs I32)>;
def ILLegalOpG : TEST_Op<"illegal_op_g">, Results<(outs I32)>;
def LegalOpA : TEST_Op<"legal_op_a">,
  Arguments<(ins StrAttr:$status)>, Results<(outs I32)>;
def LegalOpB : TEST_Op<"legal_op_b">, Results<(outs I32)>;
def LegalOpC : TEST_Op<"legal_op_c">,
  Arguments<(ins I32)>, Results<(outs I32)>;
def LegalOpD : TEST_Op<"legal_op_d">, Arguments<(ins AnyType)>;

def ConvertBlockArgsOp : TEST_Op<"convert_block_args", [SingleBlock]> {
  let arguments = (ins UnitAttr:$is_legal, UnitAttr:$replace_with_operand,
                       UnitAttr:$duplicate, Optional<AnyType>:$val);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $val
    (`is_legal` $is_legal^)?
    (`duplicate` $duplicate^)?
    (`replace_with_operand` $replace_with_operand^)?
    $body attr-dict `:` functional-type(operands, results)
  }];
}

// Check that the conversion infrastructure can properly undo the creation of
// operations where an operation was created before its parent, in this case,
// in the parent's builder.
def IllegalOpTerminator : TEST_Op<"illegal_op_terminator", [Terminator]>;
def IllegalOpWithRegion : TEST_Op<"illegal_op_with_region"> {
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins),
    [{
       Region *bodyRegion = $_state.addRegion();
       OpBuilder::InsertionGuard g($_builder);
       Block *body = $_builder.createBlock(bodyRegion);
       $_builder.setInsertionPointToEnd(body);
       IllegalOpTerminator::create($_builder,$_state.location);
    }]>];
}
def IllegalOpWithRegionAnchor : TEST_Op<"illegal_op_with_region_anchor">;

// Check that patterns can specify bounded recursion when rewriting.
def TestRecursiveRewriteOp : TEST_Op<"recursive_rewrite"> {
  let arguments = (ins I64Attr:$depth);
  let assemblyFormat = "$depth attr-dict";
}

// Test legalization pattern: this op will be erase and will also erase the
// producer of its operand.
def BlackHoleOp : TEST_Op<"blackhole">,
  Arguments<(ins AnyType)>;

//===----------------------------------------------------------------------===//
// Test Type Legalization
//===----------------------------------------------------------------------===//

def TestRegionBuilderOp : TEST_Op<"region_builder">;
def TestSegmentedReturnOp : TEST_Op<"segmented_return", [
    Pure, ReturnLike, Terminator, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$first, Variadic<AnyType>:$second);
}

def TestReturnOp : TEST_Op<"return", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>);
  let builders = [OpBuilder<(ins),
    [{ build($_builder, $_state, {}); }]>
  ];
}
// A return-like terminator used to test FuncOp verifier ordering.
// When 'valid' unit-attr is absent the op is malformed; its verifier
// emits an error, but getMutableSuccessorOperands() calls report_fatal_error
// to expose the fact that FuncOp::verify() runs before the region is checked.
def TestCrashingReturnOp : TEST_Op<"crashing_return", [
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
       ["getMutableSuccessorOperands"]>,
    Terminator]> {
  let arguments = (ins Variadic<AnyType>:$args, UnitAttr:$valid);
  let assemblyFormat = "($args^ `:` type($args))? prop-dict attr-dict";
  let hasVerifier = 1;
}
def TestCastOp : TEST_Op<"cast">,
  Arguments<(ins Variadic<AnyType>)>, Results<(outs AnyType)>;
def TestInvalidOp : TEST_Op<"invalid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>)>;
def TestTypeProducerOp : TEST_Op<"type_producer">,
  Results<(outs AnyType)>;
def TestValidProducerOp : TEST_Op<"valid_producer">,
  Results<(outs AnyType)>;
def TestValidConsumerOp : TEST_Op<"valid_consumer">,
  Arguments<(ins AnyType)>;
def TestAnotherTypeProducerOp : TEST_Op<"another_type_producer">,
  Results<(outs AnyType)>;
def TestTypeConsumerOp : TEST_Op<"type_consumer">,
  Arguments<(ins AnyType)>;
def TestTypeChangerOp : TEST_Op<"type_changer">,
  Arguments<(ins AnyType)>, Results<(outs AnyType)>;
def TestValidOp : TEST_Op<"valid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>)>;

def TestMergeBlocksOp : TEST_Op<"merge_blocks"> {
  let summary = "merge_blocks operation";
  let description = [{
    Test op with multiple blocks that are merged with Dialect Conversion
  }];

  let regions = (region AnyRegion:$body);
  let results = (outs Variadic<AnyType>:$result);
}

def TestRemappedValueRegionOp : TEST_Op<"remapped_value_region",
                                        [SingleBlock]> {
  let summary = "remapped_value_region operation";
  let description = [{
    Test op that remaps values that haven't yet been converted in Dialect
    Conversion.
  }];

  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$result);
}

def TestSignatureConversionUndoOp : TEST_Op<"signature_conversion_undo"> {
  let regions = (region AnyRegion);
}

def TestSignatureConversionNoConverterOp
  : TEST_Op<"signature_conversion_no_converter"> {
  let regions = (region AnyRegion);
}

// A return-like operation with override on the successor operand getter
// interface method.
def TestReturnWithIgnoredValueOp : TEST_Op<"return_with_ignored_value", [
    Pure,
    ReturnLike,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, [
      "getMutableSuccessorOperands"
    ]>,
    Terminator
]> {
  let arguments = (ins Variadic<AnyType>:$values, AnyType:$unwanted_value);
}

#endif // TEST_OPS_CONVERSION
