//===- TestOpsPatterns.td - Test dialect patterns ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_PATTERNS
#define TEST_OPS_PATTERNS

include "TestOpBase.td"

//===----------------------------------------------------------------------===//
// Test Patterns
//===----------------------------------------------------------------------===//

def OpA : TEST_Op<"op_a"> {
  let arguments = (ins I32, I32Attr:$attr);
  let results = (outs I32);
}

def OpB : TEST_Op<"op_b"> {
  let arguments = (ins I32, I32Attr:$attr);
  let results = (outs I32);
}

// Test added benefit.
def OpD : TEST_Op<"op_d">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpE : TEST_Op<"op_e">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpF : TEST_Op<"op_f">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpG : TEST_Op<"op_g">, Arguments<(ins I32)>, Results<(outs I32)>;

// Test patterns for zero-result op.
def OpH : TEST_Op<"op_h">, Arguments<(ins I32)>, Results<(outs)>;
def OpI : TEST_Op<"op_i">, Arguments<(ins I32)>, Results<(outs)>;

// Test patterns for zero-input op.
def OpJ : TEST_Op<"op_j">, Arguments<(ins)>, Results<(outs I32)>;
def OpK : TEST_Op<"op_k">, Arguments<(ins)>, Results<(outs I32)>;

// Test that natives calls are only called once during rewrites.
def OpM : TEST_Op<"op_m"> {
  let arguments = (ins I32, OptionalAttr<I32Attr>:$optional_attr);
  let results = (outs I32);
}

def OpN : TEST_Op<"op_n"> {
  let arguments = (ins I32, I32);
  let results = (outs I32);
}

def OpO : TEST_Op<"op_o"> {
  let arguments = (ins I32);
  let results = (outs I32);
}

def OpP : TEST_Op<"op_p"> {
  let arguments = (ins I32, I32, I32, I32, I32, I32);
  let results = (outs I32);
}

def OpQ : TEST_Op<"op_q"> {
  let arguments = (ins AnyType, AnyType);
  let results = (outs AnyType);
}

// Test constant-folding a pattern that maps `(F32) -> I32`.
def SignOp : TEST_Op<"sign", [SameOperandsAndResultShape]> {
  let arguments = (ins RankedTensorOf<[F32]>:$operand);
  let results = (outs RankedTensorOf<[I32]>:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
}

// Test constant-folding a pattern that maps `(F32, F32) -> I1`.
def LessThanOp : TEST_Op<"less_than", [SameOperandsAndResultShape]> {
  let arguments = (ins RankedTensorOf<[F32]>:$lhs, RankedTensorOf<[F32]>:$rhs);
  let results = (outs RankedTensorOf<[I1]>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

// Test for memrefs normalization of an op with normalizable memrefs.
def OpNorm : TEST_Op<"op_norm", [MemRefsNormalizable]> {
  let arguments = (ins AnyMemRef:$X, AnyMemRef:$Y);
}
// Test for memrefs normalization of an op without normalizable memrefs.
def OpNonNorm : TEST_Op<"op_nonnorm"> {
  let arguments = (ins AnyMemRef:$X, AnyMemRef:$Y);
}
// Test for memrefs normalization of an op that has normalizable memref results.
def OpNormRet : TEST_Op<"op_norm_ret", [MemRefsNormalizable]> {
  let arguments = (ins AnyMemRef:$X);
  let results = (outs AnyMemRef:$Y, AnyMemRef:$Z);
}

// Test for memrefs normalization of an op with a reference to a function
// symbol.
def OpFuncRef : TEST_Op<"op_funcref"> {
  let summary = "Test op with a reference to a function symbol";
  let description = [{
    The "test.op_funcref" is a test op with a reference to a function symbol.
  }];
}

// Test `$_` for ignoring op argument match.
def TestIgnoreArgMatchSrcOp : TEST_Op<"ignore_arg_match_src"> {
  let arguments = (ins
    AnyType:$a, AnyType:$b, AnyType:$c,
    AnyAttr:$d, AnyAttr:$e, AnyAttr:$f);
}
def TestIgnoreArgMatchDstOp : TEST_Op<"ignore_arg_match_dst"> {
  let arguments = (ins AnyType:$b, AnyAttr:$f);
}

def OpInterleavedOperandAttribute1 : TEST_Op<"interleaved_operand_attr1"> {
  let arguments = (ins
    I32:$input1,
    I64Attr:$attr1,
    I32:$input2,
    I64Attr:$attr2
  );
}

def OpInterleavedOperandAttribute2 : TEST_Op<"interleaved_operand_attr2"> {
  let arguments = (ins
    I32:$input1,
    I64Attr:$attr1,
    I32:$input2,
    I64Attr:$attr2
  );
}

def ManyArgsOp : TEST_Op<"many_arguments"> {
  let arguments = (ins
    I32:$input1, I32:$input2, I32:$input3, I32:$input4, I32:$input5,
    I32:$input6, I32:$input7, I32:$input8, I32:$input9,
    I64Attr:$attr1, I64Attr:$attr2, I64Attr:$attr3, I64Attr:$attr4,
    I64Attr:$attr5, I64Attr:$attr6, I64Attr:$attr7, I64Attr:$attr8,
    I64Attr:$attr9
  );
}

// Test NativeCodeCall.
def OpNativeCodeCall1 : TEST_Op<"native_code_call1"> {
  let arguments = (ins
    I32:$input1, I32:$input2,
    BoolAttr:$choice,
    I64Attr:$attr1, I64Attr:$attr2
  );
  let results = (outs I32);
}
def OpNativeCodeCall2 : TEST_Op<"native_code_call2"> {
  let arguments = (ins I32:$input, I64ArrayAttr:$attr);
  let results = (outs I32);
}

def OpNativeCodeCall3 : TEST_Op<"native_code_call3"> {
  let arguments = (ins I32:$input);
  let results = (outs I32);
}

def OpNativeCodeCall4 : TEST_Op<"native_code_call4"> {
  let arguments = (ins AnyType:$input1);
  let results = (outs I32:$output1, I32:$output2);
}
def OpNativeCodeCall5 : TEST_Op<"native_code_call5"> {
  let arguments = (ins I32:$input1, I32:$input2);
  let results = (outs I32:$output1, I32:$output2);
}

def OpNativeCodeCall6 : TEST_Op<"native_code_call6"> {
  let arguments = (ins I32:$input1, I32:$input2);
  let results = (outs I32:$output1, I32:$output2);
}
def OpNativeCodeCall7 : TEST_Op<"native_code_call7"> {
  let arguments = (ins I32:$input);
  let results = (outs I32);
}

// Test AllAttrOf.
def OpAllAttrConstraint1 : TEST_Op<"all_attr_constraint_of1"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def OpAllAttrConstraint2 : TEST_Op<"all_attr_constraint_of2"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}

// Op for testing RewritePattern removing op with inner ops.
def TestOpWithRegionPattern : TEST_Op<"op_with_region_pattern"> {
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
}

def TestOpConstant : TEST_Op<"constant", [ConstantLike, NoMemoryEffect]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let hasFolder = 1;
}

def OpR : TEST_Op<"op_r">, Arguments<(ins AnyInteger, AnyInteger)>, Results<(outs AnyInteger)>;
def OpS : TEST_Op<"op_s">, Arguments<(ins AnyInteger, AnyAttr:$value)>, Results<(outs AnyInteger)>;

// Op for testing trivial removal via folding of op with inner ops and no uses.
def TestOpWithRegionFoldNoMemoryEffect : TEST_Op<
    "op_with_region_fold_no_side_effect", [NoMemoryEffect]> {
  let regions = (region SizedRegion<1>:$region);
}

// Op for testing folding of outer op with inner ops.
def TestOpWithRegionFold : TEST_Op<"op_with_region_fold"> {
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType);
  let regions = (region SizedRegion<1>:$region);
  let hasFolder = 1;
}

def TestOpWithVariadicResultsAndFolder: TEST_Op<"op_with_variadic_results_and_folder"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
  let hasFolder = 1;
}

def TestAddIOp : TEST_Op<"addi"> {
  let arguments = (ins AnyTypeOf<[I32, TestI32]>:$op1,
                       AnyTypeOf<[I32, TestI32]>:$op2);
  let results = (outs AnyTypeOf<[I32, TestI32]>);
}

def TestCommutativeOp : TEST_Op<"op_commutative", [Commutative]> {
  let arguments = (ins I32:$op1, I32:$op2, I32:$op3, I32:$op4);
  let results = (outs I32);
}

def TestLargeCommutativeOp : TEST_Op<"op_large_commutative", [Commutative]> {
  let arguments = (ins I32:$op1, I32:$op2, I32:$op3, I32:$op4, I32:$op5, I32:$op6, I32:$op7);
  let results = (outs I32);
}

def TestCommutative2Op : TEST_Op<"op_commutative2", [Commutative]> {
  let arguments = (ins I32:$op1, I32:$op2);
  let results = (outs I32);
}

def TestIdempotentTraitOp
 : TEST_Op<"op_idempotent_trait",
           [SameOperandsAndResultType, NoMemoryEffect, Idempotent]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
}

def TestIdempotentTraitBinaryOp
    : TEST_Op<"op_idempotent_trait_binary",
              [SameOperandsAndResultType, NoMemoryEffect, Idempotent]> {
  let arguments = (ins I32:$op1, I32:$op2);
  let results = (outs I32);
}

def TestInvolutionTraitNoOperationFolderOp
 : TEST_Op<"op_involution_trait_no_operation_fold",
           [SameOperandsAndResultType, NoMemoryEffect, Involution]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
}

def TestInvolutionTraitFailingOperationFolderOp
 : TEST_Op<"op_involution_trait_failing_operation_fold",
           [SameOperandsAndResultType, NoMemoryEffect, Involution]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
  let hasFolder = 1;
}

def TestInvolutionTraitSuccesfulOperationFolderOp
 : TEST_Op<"op_involution_trait_succesful_operation_fold",
           [SameOperandsAndResultType, NoMemoryEffect, Involution]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
  let hasFolder = 1;
}

def TestOpInPlaceFoldAnchor : TEST_Op<"op_in_place_fold_anchor"> {
  let arguments = (ins I32);
  let results = (outs I32);
}

def TestOpInPlaceFold : TEST_Op<"op_in_place_fold"> {
  let arguments = (ins I32:$op, OptionalAttr<I32Attr>:$attr);
  let results = (outs I32);
  let hasFolder = 1;
}

def TestOpInPlaceSelfFold : TEST_Op<"op_in_place_self_fold"> {
  let arguments = (ins UnitAttr:$folded);
  let results = (outs I32);
  let hasFolder = 1;
}

// Test op that simply returns success.
def TestOpInPlaceFoldSuccess : TEST_Op<"op_in_place_fold_success"> {
  let results = (outs Variadic<I1>);
  let hasFolder = 1;
  let extraClassDefinition = [{
    ::llvm::LogicalResult $cppClass::fold(FoldAdaptor adaptor,
        SmallVectorImpl<OpFoldResult> &results) {
      return success();
    }
  }];
}

def TestOpFoldWithFoldAdaptor
  : TEST_Op<"fold_with_fold_adaptor",
      [AttrSizedOperandSegments, NoTerminator]> {
  let arguments = (ins
    I32:$op,
    DenseI32ArrayAttr:$attr,
    Variadic<I32>:$variadic,
    VariadicOfVariadic<I32, "attr">:$var_of_var
  );

  let results = (outs I32:$res);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $op `,` `[` $variadic `]` `,` `{` $var_of_var `}` $body attr-dict-with-keyword
  }];

  let hasFolder = 1;
}

def TestDialectCanonicalizerOp : TEST_Op<"dialect_canonicalizable"> {
  let arguments = (ins);
  let results = (outs I32);
}

//===----------------------------------------------------------------------===//
// Test Patterns (Symbol Binding)
//===----------------------------------------------------------------------===//

// Test symbol binding.
def OpSymbolBindingA : TEST_Op<"symbol_binding_a", []> {
  let arguments = (ins I32:$operand, I64Attr:$attr);
  let results = (outs I32);
}
def OpSymbolBindingB : TEST_Op<"symbol_binding_b", []> {
  let arguments = (ins I32:$operand);
  let results = (outs I32);
}
def OpSymbolBindingC : TEST_Op<"symbol_binding_c", []> {
  let arguments = (ins I32:$operand);
  let results = (outs I32);
  let builders = OpSymbolBindingB.builders;
}
def OpSymbolBindingD : TEST_Op<"symbol_binding_d", []> {
  let arguments = (ins I32:$input1, I32:$input2, I64Attr:$attr);
  let results = (outs I32);
}

def OpSymbolBindingNoResult : TEST_Op<"symbol_binding_no_result", []> {
  let arguments = (ins I32:$operand);
}

//===----------------------------------------------------------------------===//
// Test Patterns (Attributes)
//===----------------------------------------------------------------------===//

// Test matching against op attributes.
def OpAttrMatch1 : TEST_Op<"match_op_attribute1"> {
  let arguments = (ins
    I32Attr:$required_attr,
    OptionalAttr<I32Attr>:$optional_attr,
    DefaultValuedAttr<I32Attr, "42">:$default_valued_attr,
    I32Attr:$more_attr
  );
  let results = (outs I32);
}
def OpAttrMatch2 : TEST_Op<"match_op_attribute2"> {
  let arguments = OpAttrMatch1.arguments;
  let results = (outs I32);
}

// Test unit attrs.
def OpAttrMatch3 : TEST_Op<"match_op_attribute3"> {
  let arguments = (ins UnitAttr:$attr);
  let results = (outs I32);
}
def OpAttrMatch4 : TEST_Op<"match_op_attribute4"> {
  let arguments = (ins UnitAttr:$attr1, UnitAttr:$attr2);
  let results = (outs I32);
}

// Test with constant attr.
def OpC : TEST_Op<"op_c">, Arguments<(ins I32)>, Results<(outs I32)>;

def ThreeResultOp : TEST_Op<"three_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1, F32:$result2, F32:$result3);
}

def AnotherThreeResultOp
    : TEST_Op<"another_three_result",
              [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1, F32:$result2, F32:$result3);
}

def TwoResultOp : TEST_Op<"two_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1, F32:$result2);
}

def AnotherTwoResultOp : TEST_Op<"another_two_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs F32:$result1, F32:$result2);
}

def OneResultOp1 : TEST_Op<"one_result1"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs F32:$result1);
}

def OneResultOp2 : TEST_Op<"one_result2"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1);
}

def OneResultOp3 : TEST_Op<"one_result3"> {
  let arguments = (ins F32);
  let results = (outs I32:$result1);
}

//===----------------------------------------------------------------------===//
// Test Patterns (Variadic Ops)
//===----------------------------------------------------------------------===//

def OneVResOneVOperandOp1 : TEST_Op<"one_variadic_out_one_variadic_in1"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
}
def OneVResOneVOperandOp2 : TEST_Op<"one_variadic_out_one_variadic_in2"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
}

def MixedVOperandOp1 : TEST_Op<"mixed_variadic_in1",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    F32:$input2,
    Variadic<I32>:$input3
  );
}

def MixedVOperandOp2 : TEST_Op<"mixed_variadic_in2",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    F32:$input2,
    Variadic<I32>:$input3
  );
}

def MixedVResultOp1 : TEST_Op<"mixed_variadic_out1", [SameVariadicResultSize]> {
  let results = (outs
    Variadic<I32>:$output1,
    F32:$output2,
    Variadic<I32>:$output3
  );
}

def MixedVResultOp2 : TEST_Op<"mixed_variadic_out2", [SameVariadicResultSize]> {
  let results = (outs
    Variadic<I32>:$output1,
    F32:$output2,
    Variadic<I32>:$output3
  );
}

def OneI32ResultOp : TEST_Op<"one_i32_out"> {
  let results = (outs I32);
}

def MixedVOperandOp3 : TEST_Op<"mixed_variadic_in3",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    I32:$input1,
    Variadic<I32>:$input2,
    Variadic<I32>:$input3,
    I32Attr:$count
  );

  let results = (outs I32);
}

def MixedVResultOp3 : TEST_Op<"mixed_variadic_out3",
                               [SameVariadicResultSize]> {
  let arguments = (ins I32Attr:$count);

  let results = (outs
    I32:$output1,
    Variadic<I32>:$output2,
    Variadic<I32>:$output3
  );

  // We will use this op in a nested result pattern, where we cannot deduce the
  // result type. So need to provide a builder not requiring result types.
  let builders = [
    OpBuilder<(ins "::mlir::IntegerAttr":$count),
    [{
      auto i32Type = $_builder.getIntegerType(32);
      $_state.addTypes(i32Type); // $output1
      SmallVector<Type, 4> types(count.getInt(), i32Type);
      $_state.addTypes(types); // $output2
      $_state.addTypes(types); // $output3
      $_state.addAttribute("count", count);
    }]>
  ];
}

// Variadic structured matching
def MixedVOperandOp4 : TEST_Op<"mixed_variadic_in4"> {
  let arguments = (ins
    Variadic<I32>:$input1,
    I32:$input2,
    I32Attr:$attr1
  );
}

def MixedVOperandOp5 : TEST_Op<"mixed_variadic_in5"> {
  let arguments = (ins
    I32:$input1,
    I32:$input2,
    I32:$input3,
    I32Attr:$attr1,
    StrAttr:$pattern_name
  );
}

// Helper op to test variadic recursive pattern matching
def MixedVOperandInOutI32Op : TEST_Op<"mixed_variadic_in_out_i32"> {
  let arguments = (ins
    I32:$input
  );
  let results = (outs
    I32:$output
  );
}

def MixedVOperandOp6 : TEST_Op<"mixed_variadic_in6",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    Variadic<I32>:$input2,
    I32Attr:$attr1
  );
}

def MixedVOperandOp7 : TEST_Op<"mixed_variadic_optional_in7",
                               [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    Optional<I32>:$input2,
    I32Attr:$attr1
  );
}

//===----------------------------------------------------------------------===//
// Test Patterns (either)
//===----------------------------------------------------------------------===//

def TestEitherOpA : TEST_Op<"either_op_a"> {
  let arguments = (ins AnyInteger:$arg0, AnyInteger:$arg1, AnyInteger:$arg2);
  let results = (outs I32:$output);
}

def TestEitherOpB : TEST_Op<"either_op_b"> {
  let arguments = (ins AnyInteger:$arg0, AnyInteger:$arg1);
  let results = (outs I32:$output);
}

def TestEitherOpC : TEST_Op<"either_op_c"> {
  let arguments = (ins AnyI32Attr:$attr, AnyInteger:$arg0, AnyInteger:$arg1);
  let results = (outs I32:$output);
}

def TestEitherHelperOpA : TEST_Op<"either_helper_op_a"> {
  let arguments = (ins I32:$arg0);
  let results = (outs I32:$output);
}

def TestEitherHelperOpB : TEST_Op<"either_helper_op_b"> {
  let arguments = (ins I32:$arg0);
  let results = (outs I32:$output);
}

//===----------------------------------------------------------------------===//
// Test Patterns (Type Builders)
//===----------------------------------------------------------------------===//

def SourceOp : TEST_Op<"source_op"> {
  let arguments = (ins AnyInteger:$arg, AnyI32Attr:$tag);
  let results = (outs AnyInteger);
}

// An op without return type deduction.
def OpX : TEST_Op<"op_x"> {
  let arguments = (ins AnyInteger:$input);
  let results = (outs AnyInteger);
}

def OneToTwo : TEST_Op<"one_to_two"> {
  let arguments = (ins AnyInteger);
  let results = (outs AnyInteger, AnyInteger);
}

def TwoToOne : TEST_Op<"two_to_one"> {
  let arguments = (ins AnyInteger, AnyInteger);
  let results = (outs AnyInteger);
}

#endif // TEST_OPS_PATTERNS
