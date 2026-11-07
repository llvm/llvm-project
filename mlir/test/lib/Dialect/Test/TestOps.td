//===-- TestOps.td - Test dialect operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS
#define TEST_OPS

// The ops of the test dialect, split by category into separate files, each
// with its own op declarations and definitions. This file additionally holds
// the DRR patterns, which may use ops of all categories.
include "TestOpBase.td"
include "TestOpsCore.td"
include "TestOpsAttributes.td"
include "TestOpsTraits.td"
include "TestOpsPatterns.td"
include "TestOpsConversion.td"
include "TestOpsInterfaces.td"
include "TestOpsProperties.td"

//===----------------------------------------------------------------------===//
// Test Attributes
//===----------------------------------------------------------------------===//

// A pattern that updates dense<[3.0, 4.0]> to dense<[5.0, 6.0]>.
// This tests both matching and generating float elements attributes.
def UpdateFloatElementsAttr : Pat<
  (FloatElementsAttrOp
    ConstantAttr<RankedF32ElementsAttr<[2]>, "{3.0f, 4.0f}">:$f32attr,
    $f64attr),
  (FloatElementsAttrOp
    ConstantAttr<RankedF32ElementsAttr<[2]>, "{5.0f, 6.0f}">:$f32attr,
    $f64attr)>;

//===----------------------------------------------------------------------===//
// Test Enum Attributes
//===----------------------------------------------------------------------===//

// Define a pattern that matches and creates an enum attribute.
def : Pat<(OpWithEnum ConstantEnumCase<TestEnumAttr, "first">:$value,
                      ConstantAttr<I32Attr, "0">:$tag),
          (OpWithEnum ConstantEnumCase<TestEnumAttr, "second">,
                      ConstantAttr<I32Attr, "1">)>;

//===----------------------------------------------------------------------===//
// Test Bit Enum Attributes
//===----------------------------------------------------------------------===//


// Define a pattern that matches and creates a bit enum attribute.
def : Pat<(OpWithBitEnum ConstantEnumCase<TestBitEnumAttr, "write|execute">,
                         ConstantAttr<I32Attr, "0">),
          (OpWithBitEnum ConstantEnumCase<TestBitEnumAttr, "execute|read">,
                         ConstantAttr<I32Attr, "1">)>;

//===----------------------------------------------------------------------===//
// Test Traits
//===----------------------------------------------------------------------===//

def IsNotScalar : Constraint<CPred<"$0.getType().getRank() != 0">>;

def UpdateAttr : Pat<(I32ElementsAttrOp $attr),
                     (I32ElementsAttrOp ConstantAttr<I32ElementsAttr, "0">),
                     [(IsNotScalar $attr)]>;

//===----------------------------------------------------------------------===//
// Test Patterns
//===----------------------------------------------------------------------===//

// Test named pattern.
def TestNamedPatternRule : Pat<(OpA $input, $attr), (OpB $input, $attr)>;

// Test with fused location.
def : Pat<(OpA (OpA $input, $attr), $bttr), (OpB $input, $bttr)>;
// Verify that bumping benefit results in selecting different op.
def : Pat<(OpD $input), (OpE $input)>;
def : Pat<(OpD $input), (OpF $input), [], [], (addBenefit 10)>;
// Verify that patterns with more source nodes are selected before those with fewer.
def : Pat<(OpG $input), (OpB $input, ConstantAttr<I32Attr, "20">:$attr)>;
def : Pat<(OpG (OpG $input)), (OpB $input, ConstantAttr<I32Attr, "34">:$attr)>;
def : Pat<(OpH $input), (OpI $input)>;
def : Pat<(OpJ), (OpK)>;

// Test same operand name enforces equality condition check.
def TestEqualArgsPattern : Pat<(OpN $a, $a), (OpO $a)>;

// Test when equality is enforced at different depth.
def TestNestedOpEqualArgsPattern :
  Pat<(OpN $b, (OpP $a, $b, $c, $d, $e, $f)), (replaceWithValue $b)>;

// Test when equality is enforced on same op and same operand but at different
// depth. We only bound one of the $x to the second operand of outer OpN and
// left another be the default value (which is the value of first operand of
// outer OpN). As a result, it ended up comparing wrong values in some cases.
def TestNestedSameOpAndSameArgEqualityPattern :
  Pat<(OpN (OpN $_, $x), $x), (replaceWithValue $x)>;

// Test multiple equal arguments check enforced.
def TestMultipleEqualArgsPattern :
  Pat<(OpP $a, $b, $a, $a, $b, $c), (OpN $c, $b)>;

// Test equal arguments checks are applied before user provided constraints.
def AssertBinOpEqualArgsAndReturnTrue : Constraint<
  CPred<"assertBinOpEqualArgsAndReturnTrue($0)">>;
def TestEqualArgsCheckBeforeUserConstraintsPattern :
  Pat<(OpQ:$op $x, $x),
      (replaceWithValue $x),
      [(AssertBinOpEqualArgsAndReturnTrue $op)]>;

// Pattern add the argument plus a increasing static number hidden in
// OpMTest function. That value is set into the optional argument.
// That way, we will know if operations is called once or twice.
def OpMGetNullAttr : NativeCodeCall<"Attribute()">;
def OpMAttributeIsNull : Constraint<CPred<"! ($_self)">, "Attribute is null">;
def OpMVal : NativeCodeCall<"opMTest($_builder, $0)">;
def : Pat<(OpM $attr, $optAttr), (OpM $attr, (OpMVal $attr) ),
    [(OpMAttributeIsNull:$optAttr)]>;
def : Pat<(TestIgnoreArgMatchSrcOp $_, $b, I32, I64Attr:$_, $_, $f),
          (TestIgnoreArgMatchDstOp $b, $f)>;

// Test that DRR does not blow up when seeing lots of arguments.
def : Pat<(ManyArgsOp
            $input1, $input2, $input3, $input4, $input5,
            $input6, $input7, $input8, $input9,
            ConstantAttr<I64Attr, "42">,
            $attr2, $attr3, $attr4, $attr5, $attr6,
            $attr7, $attr8, $attr9),
          (ManyArgsOp
            $input1, $input2, $input3, $input4, $input5,
            $input6, $input7, $input8, $input9,
            ConstantAttr<I64Attr, "24">,
            $attr2, $attr3, $attr4, $attr5, $attr6,
            $attr7, $attr8, $attr9)>;

// Test that we can capture and reference interleaved operands and attributes.
def : Pat<(OpInterleavedOperandAttribute1 $input1, $attr1, $input2, $attr2),
          (OpInterleavedOperandAttribute2 $input1, $attr1, $input2, $attr2)>;
// Native code call to invoke a C++ function
def CreateOperand: NativeCodeCall<"chooseOperand($0, $1, $2)">;
// Native code call to invoke a C++ expression
def CreateArrayAttr: NativeCodeCall<"$_builder.getArrayAttr({$0, $1})">;
// Test that we can use NativeCodeCall to create operand and attribute.
// This pattern chooses between $input1 and $input2 according to $choice and
// it combines $attr1 and $attr2 into an array attribute.
def : Pat<(OpNativeCodeCall1 $input1, $input2,
                             ConstBoolAttrTrue:$choice, $attr1, $attr2),
          (OpNativeCodeCall2 (CreateOperand $input1, $input2, $choice),
                             (CreateArrayAttr $attr1, $attr2))>;
// Note: the following is just for testing purpose.
// Should use the replaceWithValue directive instead.
def UseOpResult: NativeCodeCall<"$0">;
// Test that we can use NativeCodeCall to create result.
def : Pat<(OpNativeCodeCall1 $input1, $input2,
                             ConstBoolAttrFalse, $attr1, $attr2),
          (UseOpResult $input2)>;
// Test that NativeCodeCall is not ignored if it is not used to directly
// replace the matched root op.
def : Pattern<(OpNativeCodeCall3 $input),
              [(NativeCodeCallVoid<"createOpI($_builder, $_loc, $0)"> $input),
               (OpK)]>;

def GetFirstI32Result : NativeCodeCall<"success(getFirstI32Result($_self, $0))">;
def BindNativeCodeCallResult : NativeCodeCall<"bindNativeCodeCallResult($0)">;
def : Pat<(OpNativeCodeCall4 (GetFirstI32Result $ret)),
          (OpNativeCodeCall5 (BindNativeCodeCallResult:$native $ret), $native)>;
def BindMultipleNativeCodeCallResult : NativeCodeCall<"bindMultipleNativeCodeCallResult($0, $1)", 2>;
def : Pattern<(OpNativeCodeCall6 $arg1, $arg2),
              [(OpNativeCodeCall7 (BindMultipleNativeCodeCallResult:$native__0 $arg1, $arg2)),
               (OpNativeCodeCall7 $native__1)]>;
def Constraint0 : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<ArrayAttr>($_self)[0]).getInt() == 0">,
    "[0] == 0">;
def Constraint1 : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<ArrayAttr>($_self)[1]).getInt() == 1">,
    "[1] == 1">;
def : Pat<(OpAllAttrConstraint1
            AllAttrOf<[Constraint0, Constraint1]>:$attr),
          (OpAllAttrConstraint2 $attr)>;

def : Pat<(OpR $input1, (ConstantLikeMatcher I32Attr:$input2)),
          (OpS:$unused $input1, $input2)>;
def : Pat<(TestOpInPlaceSelfFold:$op $_),
          (TestOpConstant ConstantAttr<I32Attr, "97">)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Symbol Binding)
//===----------------------------------------------------------------------===//

def HasOneUse: Constraint<CPred<"$0.hasOneUse()">, "has one use">;
def : Pattern<
    // Bind to source pattern op operand/attribute/result
    (OpSymbolBindingA:$res_a $operand, $attr), [
        // Bind to auxiliary op result
        (OpSymbolBindingC:$res_c (OpSymbolBindingB:$res_b $operand)),

        // Use bound symbols in resultant ops
        (OpSymbolBindingD $res_b, $res_c, $attr)],
    // Use bound symbols in additional constraints
    [(HasOneUse $res_a)]>;

// Test that we can bind to an op without results and reference it later.
def : Pat<(OpSymbolBindingNoResult:$op $operand),
          (NativeCodeCallVoid<"handleNoResultOp($_builder, $0)"> $op)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Attributes)
//===----------------------------------------------------------------------===//

def MoreConstraint : AttrConstraint<
    CPred<"::llvm::cast<IntegerAttr>($_self).getInt() == 4">, "more constraint">;
def : Pat<(OpAttrMatch1 $required, $optional, $default_valued,
                        MoreConstraint:$more),
          (OpAttrMatch2 $required, $optional, $default_valued, $more)>;
def : Pat<(OpAttrMatch3 $attr), (OpAttrMatch4 ConstUnitAttr, $attr)>;
def : Pat<(OpC $input), (OpB $input, ConstantAttr<I32Attr, "17">:$attr)>;

// Test integer enum attribute in rewrites.
def : Pat<(I32EnumAttrOp I32Case5), (I32EnumAttrOp I32Case10)>;
def : Pat<(I64EnumAttrOp I64Case5), (I64EnumAttrOp I64Case10)>;

// Test using multi-result op as a whole
def : Pat<(ThreeResultOp MultiResultOpKind1:$kind),
          (AnotherThreeResultOp $kind)>;

// Test using multi-result op as a whole for partial replacement
def : Pattern<(ThreeResultOp MultiResultOpKind2:$kind),
              [(TwoResultOp $kind),
               (OneResultOp1 $kind)]>;
def : Pattern<(ThreeResultOp MultiResultOpKind3:$kind),
              [(OneResultOp2 $kind),
               (AnotherTwoResultOp $kind)]>;

// Test using results separately in a multi-result op
def : Pattern<(ThreeResultOp MultiResultOpKind4:$kind),
              [(TwoResultOp:$res1__0 $kind),
               (OneResultOp1 $kind),
               (TwoResultOp:$res2__1 $kind)]>;

// Test referencing a single value in the value pack
// This rule only matches TwoResultOp if its second result has no use.
def : Pattern<(TwoResultOp:$res MultiResultOpKind5:$kind),
              [(OneResultOp2 $kind),
               (OneResultOp1 $kind)],
              [(HasNoUseOf:$res__1)]>;

// Test using auxiliary ops for replacing multi-result op
def : Pattern<
    (ThreeResultOp MultiResultOpKind6:$kind), [
        // Auxiliary op generated to help building the final result but not
        // directly used to replace the source op's results.
        (TwoResultOp:$interm $kind),

        (OneResultOp3 $interm__1),
        (AnotherTwoResultOp $kind)
    ]>;

//===----------------------------------------------------------------------===//
// Test Patterns (Variadic Ops)
//===----------------------------------------------------------------------===//

// Rewrite an op with one variadic operand and one variadic result to
// another similar op.
def : Pat<(OneVResOneVOperandOp1 $inputs), (OneVResOneVOperandOp2 $inputs)>;

// Rewrite an op with both variadic operands and normal operands.
def : Pat<(MixedVOperandOp1 $input1, $input2, $input3),
          (MixedVOperandOp2 $input1, $input2, $input3)>;

// Rewrite an op with both variadic results and normal results.
// Note that because we are generating the op with a top-level result pattern,
// we are able to deduce the correct result types for the generated op using
// the information from the matched root op.
def : Pat<(MixedVResultOp1), (MixedVResultOp2)>;

// Generates an op with variadic results using nested pattern.
def : Pat<(OneI32ResultOp),
          (MixedVOperandOp3
              (MixedVResultOp3:$results__0 ConstantAttr<I32Attr, "2">),
              (replaceWithValue $results__1),
              (replaceWithValue $results__2),
              ConstantAttr<I32Attr, "2">)>;

def : Pat<
  (MixedVOperandOp4 (variadic $input1a, $input1b), $input2,
                    ConstantAttr<I32Attr, "0">:$attr1),
  (MixedVOperandOp5 $input1a, $input1b, $input2, $attr1,
                    ConstantStrAttr<StrAttr, "MatchVariadic">)>;

def : Pat<
  (MixedVOperandOp5 $input1a, $input1b, $input2, $attr1,
                    ConstantStrAttr<StrAttr, "MatchInverseVariadic">),
  (MixedVOperandOp3 $input2, (variadic $input1b), (variadic $input1a),
                    ConstantAttr<I32Attr, "1">:$attr1)>;

def : Pat<
  (MixedVOperandOp4 (variadic (MixedVOperandInOutI32Op $input1a),
                              (MixedVOperandInOutI32Op $input1b)),
                    $input2, ConstantAttr<I32Attr, "1">:$attr1),
  (MixedVOperandOp5 $input1a, $input1b, $input2, $attr1,
                    ConstantStrAttr<StrAttr, "MatchVariadicSubDag">)>;

def : Pat<
  (MixedVOperandOp4 (variadic $input1, $input1), $input2,
                    ConstantAttr<I32Attr, "2">:$attr1),
  (MixedVOperandOp5 $input1, $input1, $input2, $attr1,
                    ConstantStrAttr<StrAttr, "MatchVariadicSameSymbol">)>;

def : Pat<
  (MixedVOperandOp6 (variadic:$input1 $input1a, $input1b),
                    (variadic:$input2 $input2a, $input2b),
                    ConstantAttr<I32Attr, "1">:$attr1),
  (MixedVOperandOp6 $input2, $input1, ConstantAttr<I32Attr, "-1">)>;

def : Pat<
  (MixedVOperandOp6 (variadic $input1a, $input1b),
                    (variadic $input2a, $input2b),
                    ConstantAttr<I32Attr, "2">:$attr1),
  (MixedVOperandOp5 $input2a, $input2b, $input1b, $attr1,
                    ConstantStrAttr<StrAttr, "MatchMultiVariadicSubSymbol">)>;

def : Pat<
  (MixedVOperandOp7 $input1, $input2, ConstantAttr<I32Attr, "2">:$attr1),
  (MixedVOperandOp6 $input1, (variadic $input2), $attr1),
  [(Constraint<CPred<"$0 != Value()">> $input2)]>;

//===----------------------------------------------------------------------===//
// Test Patterns (either)
//===----------------------------------------------------------------------===//

def : Pat<(TestEitherOpA (either I32:$arg1, I16:$arg2), $x),
          (TestEitherOpB $arg2, $x)>;

def : Pat<(TestEitherOpA (either (TestEitherOpB I32:$arg1, $_), I16:$arg2), $x),
          (TestEitherOpB $arg2, $x)>;

def : Pat<(TestEitherOpA (either (TestEitherOpB I32:$arg1, $_),
                                 (TestEitherOpB I16:$arg2, $_)),
                          $x),
          (TestEitherOpB $arg2, $x)>;

def : Pat<(TestEitherOpC ConstantAttr<I32Attr, "0">, (either $arg1, I32:$arg2)),
          (TestEitherOpB $arg1, $arg2)>;

// This test case ensures `emitOpMatch` doesn't redefine `castedOp{0}` local
// variables. To trigger this, we must ensure the matcher for
// `TestEitherHelperOpA` and `TestEitherHelperOpB` are not lifted as a static
// matcher.
def : Pat<(TestEitherOpB (either (TestEitherHelperOpA I32:$either_helper_0),
                                 (TestEitherHelperOpB I32:$either_helper_1))),
          (TestEitherOpB $either_helper_0, $either_helper_1)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Location)
//===----------------------------------------------------------------------===//

// Test that we can specify locations for generated ops.
def : Pat<(TestLocationSrcOp:$res1
           (TestLocationSrcOp:$res2
            (TestLocationSrcOp:$res3 $input))),
          (TestLocationDstOp
            (TestLocationDstOp
              (TestLocationDstOp $input, (location $res1)),
              (location "named")),
            (location "fused", $res2, $res3))>;

// Test that we can use the location of an op without results
def : Pat<(TestLocationSrcNoResOp:$loc
            (TestLocationSrcOp (TestLocationSrcOp $input))),
          (TestLocationDstNoResOp $input, (location $loc))>;

//===----------------------------------------------------------------------===//
// Test Patterns (Type Builders)
//===----------------------------------------------------------------------===//

// Test that ops without built-in type deduction can be created in the
// replacement DAG with an explicitly specified type.
def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "11">:$attr),
          (OpX (OpX $val, (returnType "$_builder.getI32Type()")))>;
// Test NativeCodeCall type builder can accept arguments.
def SameTypeAs : NativeCodeCall<"$0.getType()">;

def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "22">:$attr),
          (OpX (OpX $val, (returnType (SameTypeAs $val))))>;

// Test multiple return types.
def MakeI64Type : NativeCodeCall<"$_builder.getI64Type()">;
def MakeI32Type : NativeCodeCall<"$_builder.getI32Type()">;

def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "33">:$attr),
          (TwoToOne (OpX (OneToTwo:$res__0 $val, (returnType (MakeI64Type), (MakeI32Type))), (returnType (MakeI32Type))),
                    (OpX $res__1, (returnType (MakeI64Type))))>;

// Test copy value return type.
def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "44">:$attr),
          (OpX (OpX $val, (returnType $val)))>;

// Test create multiple return types with different methods.
def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "55">:$attr),
          (TwoToOne (OneToTwo:$res__0 $val, (returnType $val, "$_builder.getI64Type()")), $res__1)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Trailing Directives)
//===----------------------------------------------------------------------===//

// Test that we can specify both `location` and `returnType` directives.
def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "66">:$attr),
          (TwoToOne (OpX $val, (returnType $val), (location "loc1")),
                    (OpX $val, (location "loc2"), (returnType $val)))>;

//===----------------------------------------------------------------------===//
// Test Legalization
//===----------------------------------------------------------------------===//

def Test_LegalizerEnum_Success : ConstantStrAttr<StrAttr, "Success">;
def Test_LegalizerEnum_Failure : ConstantStrAttr<StrAttr, "Failure">;

// Check that smaller pattern depths are chosen, i.e. prioritize more direct
// mappings.
def : Pat<(ILLegalOpA), (LegalOpA Test_LegalizerEnum_Success)>;

def : Pat<(ILLegalOpA), (ILLegalOpB)>;
def : Pat<(ILLegalOpB), (LegalOpA Test_LegalizerEnum_Failure)>;

// Check that the higher benefit pattern is taken for multiple legalizations
// with the same depth.
def : Pat<(ILLegalOpC), (ILLegalOpD)>;
def : Pat<(ILLegalOpD), (LegalOpA Test_LegalizerEnum_Failure)>;

def : Pat<(ILLegalOpC), (ILLegalOpE), [], [], (addBenefit 10)>;
def : Pat<(ILLegalOpE), (LegalOpA Test_LegalizerEnum_Success)>;
def : Pat<(TestRewriteOp $input), (replaceWithValue $input)>;

//===----------------------------------------------------------------------===//
// Test Reducer Patterns
//===----------------------------------------------------------------------===//

def : Pat<(OpCrashLong $_, $_, $_), (OpCrashShort)>;

//===----------------------------------------------------------------------===//
// Test Ops with Default-Valued String Attributes
//===----------------------------------------------------------------------===//

def : Pat<(TestDefaultStrAttrNoValueOp $value),
          (TestDefaultStrAttrHasValueOp ConstantStrAttr<StrAttr, "foo">)>;

//===----------------------------------------------------------------------===//
// Test Ops with variadics
//===----------------------------------------------------------------------===//

def : Pat<(TestVariadicRewriteSrcOp $arg, $brg, $crg),
          (TestVariadicRewriteDstOp $brg, $crg, $arg)>;

//===----------------------------------------------------------------------===//
// Test Properties
//===----------------------------------------------------------------------===//

def : Pat<
  (TestPropPatternOp1 $tag, NonNegativeI64Prop:$val, ConstantProp<BoolProp, "false">),
  (TestPropPatternOp1 $tag, (NativeCodeCall<"$0 + 1"> $val), ConstantProp<BoolProp, "true">)>;

def : Pat<
  (TestPropPatternOp2 (TestPropPatternOp1 $tag1, $val, ConstantProp<BoolProp, "true">),
    PropConstraint<CPred<"!$_self.empty()">, "non-empty string">:$tag2),
  (TestPropPatternOp2
    (TestPropPatternOp1 $tag1,
      (NativeCodeCall<"-($0)"> $val),
      ConstantProp<BoolProp, "false">),
    (NativeCodeCall<"$0.str() + \".\" + $1.str()"> $tag1, $tag2))
>;

#endif // TEST_OPS
