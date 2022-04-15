//===-- TestOps.td - Test dialect operation definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS
#define TEST_OPS

include "TestDialect.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "TestInterfaces.td"


// Include the attribute definitions.
include "TestAttrDefs.td"
// Include the type definitions.
include "TestTypeDefs.td"


class TEST_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Test Types
//===----------------------------------------------------------------------===//

def IntTypesOp : TEST_Op<"int_types"> {
  let results = (outs
    AnyI16:$any_i16,
    SI32:$si32,
    UI64:$ui64,
    AnyInteger:$any_int
  );
}

def ComplexF64 : Complex<F64>;
def ComplexOp : TEST_Op<"complex_f64"> {
  let results = (outs ComplexF64);
}

def ComplexTensorOp : TEST_Op<"complex_f64_tensor"> {
  let results = (outs TensorOf<[ComplexF64]>);
}

def TupleOp : TEST_Op<"tuple_32_bit"> {
  let results = (outs TupleOf<[I32, F32]>);
}

def NestedTupleOp : TEST_Op<"nested_tuple_32_bit"> {
  let results = (outs NestedTupleOf<[I32, F32]>);
}

def TakesStaticMemRefOp : TEST_Op<"takes_static_memref"> {
  let arguments = (ins AnyStaticShapeMemRef:$x);
}

def RankLessThan2I8F32MemRefOp : TEST_Op<"rank_less_than_2_I8_F32_memref"> {
  let results = (outs MemRefRankOf<[I8, F32], [0, 1]>);
}

def NDTensorOfOp : TEST_Op<"nd_tensor_of"> {
  let arguments = (ins
    0DTensorOf<[F32]>:$arg0,
    1DTensorOf<[F32]>:$arg1,
    2DTensorOf<[I16]>:$arg2,
    3DTensorOf<[I16]>:$arg3,
    4DTensorOf<[I16]>:$arg4
  );
}

def RankedTensorOp : TEST_Op<"ranked_tensor_op"> {
  let arguments = (ins AnyRankedTensor:$input);
}

def MultiTensorRankOf : TEST_Op<"multi_tensor_rank_of"> {
  let arguments = (ins
    TensorRankOf<[I8, I32, F32], [0, 1]>:$arg0
  );
}

def TEST_TestType : DialectType<Test_Dialect,
    CPred<"$_self.isa<::test::TestType>()">, "test">,
    BuildableType<"$_builder.getType<::test::TestType>()">;

//===----------------------------------------------------------------------===//
// Test Symbols
//===----------------------------------------------------------------------===//

def SymbolOp : TEST_Op<"symbol", [Symbol]> {
  let summary =  "operation which defines a new symbol";
  let arguments = (ins StrAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
}

def SymbolScopeOp : TEST_Op<"symbol_scope",
    [SymbolTable, SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary =  "operation which defines a new symbol table";
  let regions = (region SizedRegion<1>:$region);
}

def SymbolTableRegionOp : TEST_Op<"symbol_table_region", [SymbolTable]> {
  let summary =  "operation which defines a new symbol table without a "
                 "restriction on a terminator";
  let regions = (region SizedRegion<1>:$region);
}

//===----------------------------------------------------------------------===//
// Test Operands
//===----------------------------------------------------------------------===//

def MixedNormalVariadicOperandOp : TEST_Op<
    "mixed_normal_variadic_operand", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<AnyTensor>:$input1,
    AnyTensor:$input2,
    Variadic<AnyTensor>:$input3
  );
}
def VariadicWithSameOperandsResult :
      TEST_Op<"variadic_with_same_operand_results",
              [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnySignlessInteger>);
  let results = (outs AnySignlessInteger:$result);
}

def SameOperandsResultType : TEST_Op<
    "same_operand_result_type", [SameOperandsAndResultType]> {
  let arguments = (ins AnyTensor:$operand);
  let results = (outs AnyTensor:$result);
}

//===----------------------------------------------------------------------===//
// Test Results
//===----------------------------------------------------------------------===//

def MixedNormalVariadicResults : TEST_Op<
    "mixed_normal_variadic_result", [SameVariadicResultSize]> {
  let results = (outs
    Variadic<AnyTensor>:$output1,
    AnyTensor:$output2,
    Variadic<AnyTensor>:$output3
  );
}

//===----------------------------------------------------------------------===//
// Test Attributes
//===----------------------------------------------------------------------===//

def AnyAttrOfOp : TEST_Op<"any_attr_of_i32_str"> {
  let arguments = (ins AnyAttrOf<[I32Attr, StrAttr]>:$attr);
}

def NonNegIntAttrOp : TEST_Op<"non_negative_int_attr"> {
  let arguments = (ins
      Confined<I32Attr, [IntNonNegative]>:$i32attr,
      Confined<I64Attr, [IntNonNegative]>:$i64attr
  );
}

def PositiveIntAttrOp : TEST_Op<"positive_int_attr"> {
  let arguments = (ins
      Confined<I32Attr, [IntPositive]>:$i32attr,
      Confined<I64Attr, [IntPositive]>:$i64attr
  );
}

def TypeArrayAttrOp : TEST_Op<"type_array_attr"> {
  let arguments = (ins TypeArrayAttr:$attr);
}
def TypeArrayAttrWithDefaultOp : TEST_Op<"type_array_attr_with_default"> {
  let arguments = (ins DefaultValuedAttr<TypeArrayAttr, "{}">:$attr);
}
def TypeStringAttrWithTypeOp : TEST_Op<"string_attr_with_type"> {
  let arguments = (ins TypedStrAttr<AnyType>:$attr);
  let assemblyFormat = "$attr attr-dict";
}

def I32Case5:  I32EnumAttrCase<"case5", 5>;
def I32Case10: I32EnumAttrCase<"case10", 10>;

def SomeI32Enum: I32EnumAttr<
  "SomeI32Enum", "", [I32Case5, I32Case10]>;

def I32EnumAttrOp : TEST_Op<"i32_enum_attr"> {
  let arguments = (ins SomeI32Enum:$attr);
  let results = (outs I32:$val);
}

def I64Case5:  I64EnumAttrCase<"case5", 5>;
def I64Case10: I64EnumAttrCase<"case10", 10>;

def SomeI64Enum: I64EnumAttr<
  "SomeI64Enum", "", [I64Case5, I64Case10]>;

def I64EnumAttrOp : TEST_Op<"i64_enum_attr"> {
  let arguments = (ins SomeI64Enum:$attr);
  let results = (outs I32:$val);
}

def SomeStructAttr : StructAttr<"SomeStructAttr", Test_Dialect, [
  StructFieldAttr<"some_field", I64Attr>,
  StructFieldAttr<"some_other_field", I64Attr>
]> {}

def StructAttrOp : TEST_Op<"struct_attr"> {
  let arguments = (ins SomeStructAttr:$the_struct_attr);
  let results = (outs);
}

def IntAttrOp : TEST_Op<"int_attrs"> {
  let arguments = (ins
    AnyI32Attr:$any_i32_attr,
    IndexAttr:$index_attr,
    UI32Attr:$ui32_attr,
    SI32Attr:$si32_attr
  );
}

def FloatElementsAttrOp : TEST_Op<"float_elements_attr"> {
  let arguments = (ins
      RankedF32ElementsAttr<[2]>:$scalar_f32_attr,
      RankedF64ElementsAttr<[4, 8]>:$tensor_f64_attr
  );
}

// A pattern that updates dense<[3.0, 4.0]> to dense<[5.0, 6.0]>.
// This tests both matching and generating float elements attributes.
def UpdateFloatElementsAttr : Pat<
  (FloatElementsAttrOp
    ConstantAttr<RankedF32ElementsAttr<[2]>, "{3.0f, 4.0f}">:$f32attr,
    $f64attr),
  (FloatElementsAttrOp
    ConstantAttr<RankedF32ElementsAttr<[2]>, "{5.0f, 6.0f}">:$f32attr,
    $f64attr)>;

def IntElementsAttrOp : TEST_Op<"int_elements_attr"> {
  let arguments = (ins
      AnyI32ElementsAttr:$any_i32_attr,
      I32ElementsAttr:$i32_attr
  );
}

def RankedIntElementsAttrOp : TEST_Op<"ranked_int_elements_attr"> {
  let arguments = (ins
      RankedI32ElementsAttr<[2]>:$vector_i32_attr,
      RankedI64ElementsAttr<[4, 8]>:$matrix_i64_attr
  );
}

def DerivedTypeAttrOp : TEST_Op<"derived_type_attr", []> {
  let results = (outs AnyTensor:$output);
  DerivedTypeAttr element_dtype =
    DerivedTypeAttr<"return getElementTypeOrSelf(getOutput().getType());">;
  DerivedAttr size = DerivedAttr<"int",
    "return getOutput().getType().cast<ShapedType>().getSizeInBits();",
    "$_builder.getI32IntegerAttr($_self)">;
}

def StringElementsAttrOp : TEST_Op<"string_elements_attr"> {
  let arguments = (ins
      StringElementsAttr:$scalar_string_attr
  );
}

//===----------------------------------------------------------------------===//
// Test Enum Attributes
//===----------------------------------------------------------------------===//

// Define the C++ enum.
def TestEnum
    : I32EnumAttr<"TestEnum", "a test enum", [
        I32EnumAttrCase<"First", 0, "first">,
        I32EnumAttrCase<"Second", 1, "second">,
        I32EnumAttrCase<"Third", 2, "third">,
      ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "test";
}

// Define the enum attribute.
def TestEnumAttr : EnumAttr<Test_Dialect, TestEnum, "enum">;

// Define an op that contains the enum attribute.
def OpWithEnum : TEST_Op<"op_with_enum"> {
  let arguments = (ins TestEnumAttr:$value, OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

// Define a pattern that matches and creates an enum attribute.
def : Pat<(OpWithEnum ConstantAttr<TestEnumAttr,
                                   "::test::TestEnum::First">:$value,
                      ConstantAttr<I32Attr, "0">:$tag),
          (OpWithEnum ConstantAttr<TestEnumAttr,
                                   "::test::TestEnum::Second">,
                      ConstantAttr<I32Attr, "1">)>;

//===----------------------------------------------------------------------===//
// Test Attribute Constraints
//===----------------------------------------------------------------------===//

def SymbolRefOp : TEST_Op<"symbol_ref_attr"> {
  let arguments = (ins
    Confined<FlatSymbolRefAttr, [ReferToOp<"FuncOp">]>:$symbol
  );
}

//===----------------------------------------------------------------------===//
// Test Regions
//===----------------------------------------------------------------------===//

def OneRegionOp : TEST_Op<"one_region_op", []> {
  let regions = (region AnyRegion);
}

def TwoRegionOp : TEST_Op<"two_region_op", []> {
  let regions = (region AnyRegion, AnyRegion);
}

def SizedRegionOp : TEST_Op<"sized_region_op", []> {
  let regions = (region SizedRegion<2>:$my_region, SizedRegion<1>);
}

def VariadicRegionInferredTypesOp : TEST_Op<"variadic_region_inferred",
                                            [InferTypeOpInterface]> {
  let regions = (region VariadicRegion<AnyRegion>:$bodies);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext *context,
          llvm::Optional<::mlir::Location> location, mlir::ValueRange operands,
          mlir::DictionaryAttr attributes, mlir::RegionRange regions,
          llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({mlir::IntegerType::get(context, 16)});
      return mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// NoTerminator Operation
//===----------------------------------------------------------------------===//

def SingleNoTerminatorOp : TEST_Op<"single_no_terminator_op",
                                   GraphRegionNoTerminator.traits> {
  let regions = (region SizedRegion<1>:$my_region);

  let assemblyFormat = "attr-dict `:` $my_region";
}

def SingleNoTerminatorCustomAsmOp : TEST_Op<"single_no_terminator_custom_asm_op",
                                            [SingleBlock, NoTerminator]> {
  let regions = (region SizedRegion<1>);
  let hasCustomAssemblyFormat = 1;
}

def VariadicNoTerminatorOp : TEST_Op<"variadic_no_terminator_op",
                                     GraphRegionNoTerminator.traits> {
  let regions = (region VariadicRegion<SizedRegion<1>>:$my_regions);

  let assemblyFormat = "attr-dict `:` $my_regions";
}

//===----------------------------------------------------------------------===//
// Test Call Interfaces
//===----------------------------------------------------------------------===//

def TestCallOp : TEST_Op<"call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def ConversionCallOp : TEST_Op<"conversion_call_op",
    [CallOpInterface]> {
  let arguments = (ins Variadic<AnyType>:$arg_operands, SymbolRefAttr:$callee);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee();
  }];
  let extraClassDefinition = [{
    ::mlir::CallInterfaceCallable $cppClass::getCallableForCallee() {
      return (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee");
    }
  }];
}

def FunctionalRegionOp : TEST_Op<"functional_region_op",
    [CallableOpInterface]> {
  let regions = (region AnyRegion:$body);
  let results = (outs FunctionType);

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getCallableResults() {
      return getType().cast<::mlir::FunctionType>().getResults();
    }
  }];
}


def FoldToCallOp : TEST_Op<"fold_to_call_op"> {
  let arguments = (ins FlatSymbolRefAttr:$callee);
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Test Traits
//===----------------------------------------------------------------------===//

def SameOperandElementTypeOp : TEST_Op<"same_operand_element_type",
    [SameOperandsElementType]> {
  let arguments = (ins AnyType, AnyType);
  let results = (outs AnyType);
}

def SameOperandAndResultElementTypeOp :
    TEST_Op<"same_operand_and_result_element_type",
    [SameOperandsAndResultElementType]> {
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
}

def SameOperandShapeOp : TEST_Op<"same_operand_shape", [SameOperandsShape]> {
  let arguments = (ins Variadic<AnyShaped>);
}

def SameOperandAndResultShapeOp : TEST_Op<"same_operand_and_result_shape",
    [SameOperandsAndResultShape]> {
  let arguments = (ins Variadic<AnyShaped>);
  let results = (outs Variadic<AnyShaped>);
}

def SameOperandAndResultTypeOp : TEST_Op<"same_operand_and_result_type",
    [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
}

def ElementwiseMappableOp : TEST_Op<"elementwise_mappable",
    ElementwiseMappable.traits> {
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
}

def ArgAndResHaveFixedElementTypesOp :
    TEST_Op<"arg_and_res_have_fixed_element_types",
      [PredOpTrait<"fixed type combination",
         And<[ElementTypeIsPred<"x", I32>,
              ElementTypeIsPred<"y", F32>]>>,
      ElementTypeIs<"res", I16>]> {
  let arguments = (ins
    AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def OperandsHaveSameElementType : TEST_Op<"operands_have_same_element_type", [
    AllElementTypesMatch<["x", "y"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
}

def OperandZeroAndResultHaveSameElementType : TEST_Op<
    "operand0_and_result_have_same_element_type",
    [AllElementTypesMatch<["x", "res"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
  let results = (outs AnyType:$res);
}

def OperandsHaveSameType :
    TEST_Op<"operands_have_same_type", [AllTypesMatch<["x", "y"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
}

def ResultHasSameTypeAsAttr :
    TEST_Op<"result_has_same_type_as_attr",
            [AllTypesMatch<["attr", "result"]>]> {
  let arguments = (ins AnyAttr:$attr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$attr `->` type($result) attr-dict";
}

def OperandZeroAndResultHaveSameType :
    TEST_Op<"operand0_and_result_have_same_type",
            [AllTypesMatch<["x", "res"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
  let results = (outs AnyType:$res);
}

def OperandsHaveSameRank :
    TEST_Op<"operands_have_same_rank", [AllRanksMatch<["x", "y"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
}

def OperandZeroAndResultHaveSameRank :
    TEST_Op<"operand0_and_result_have_same_rank",
            [AllRanksMatch<["x", "res"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def OperandZeroAndResultHaveSameShape :
    TEST_Op<"operand0_and_result_have_same_shape",
            [AllShapesMatch<["x", "res"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def OperandZeroAndResultHaveSameElementCount :
    TEST_Op<"operand0_and_result_have_same_element_count",
            [AllElementCountsMatch<["x", "res"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def FourEqualsFive :
    TEST_Op<"four_equals_five", [AllMatch<["5", "4"], "4 equals 5">]>;

def OperandRankEqualsResultSize :
    TEST_Op<"operand_rank_equals_result_size",
            [AllMatch<[Rank<"operand">.result, ElementCount<"result">.result],
                      "operand rank equals result size">]> {
  let arguments = (ins AnyShaped:$operand);
  let results = (outs AnyShaped:$result);
}

def IfFirstOperandIsNoneThenSoIsSecond :
    TEST_Op<"if_first_operand_is_none_then_so_is_second", [PredOpTrait<
    "has either both none type operands or first is not none",
     Or<[
        And<[TypeIsPred<"x", NoneType>, TypeIsPred<"y", NoneType>]>,
        Neg<TypeIsPred<"x", NoneType>>]>>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
}

def BroadcastableOp : TEST_Op<"broadcastable", [ResultsBroadcastableShape]> {
  let arguments = (ins Variadic<AnyTensor>);
  let results = (outs AnyTensor);
}

// HasParent trait
def ParentOp : TEST_Op<"parent"> {
    let regions = (region AnyRegion);
}
def ChildOp : TEST_Op<"child", [HasParent<"ParentOp">]>;

// ParentOneOf trait
def ParentOp1 : TEST_Op<"parent1"> {
  let regions = (region AnyRegion);
}
def ChildWithParentOneOf : TEST_Op<"child_with_parent_one_of",
                                [ParentOneOf<["ParentOp", "ParentOp1"]>]>;

def TerminatorOp : TEST_Op<"finish", [Terminator]>;
def SingleBlockImplicitTerminatorOp : TEST_Op<"SingleBlockImplicitTerminator",
    [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let regions = (region SizedRegion<1>:$region);
}

def I32ElementsAttrOp : TEST_Op<"i32ElementsAttr"> {
  let arguments = (ins I32ElementsAttr:$attr);
}

def IndexElementsAttrOp : TEST_Op<"indexElementsAttr"> {
  let arguments = (ins IndexElementsAttr:$attr);
}

def OpWithInferTypeInterfaceOp : TEST_Op<"op_with_infer_type_if", [
    DeclareOpInterfaceMethods<InferTypeOpInterface,
        ["inferReturnTypeComponents"]>]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

def OpWithShapedTypeInferTypeInterfaceOp : TEST_Op<"op_with_shaped_type_infer_type_if",
      [InferTensorTypeWithReify]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

def OpWithResultShapeInterfaceOp : TEST_Op<"op_with_result_shape_interface",
      [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
          ["reifyReturnTypeShapes"]>]> {
  let arguments = (ins AnyRankedTensor:$operand1, AnyRankedTensor:$operand2);
  let results = (outs AnyRankedTensor:$result1, AnyRankedTensor:$result2);
}

def OpWithResultShapePerDimInterfaceOp :
    TEST_Op<"op_with_result_shape_per_dim_interface",
        [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]> {
  let arguments = (ins AnyRankedTensor:$operand1, AnyRankedTensor:$operand2);
  let results = (outs AnyRankedTensor:$result1, AnyRankedTensor:$result2);
}

def IsNotScalar : Constraint<CPred<"$0.getType().getRank() != 0">>;

def UpdateAttr : Pat<(I32ElementsAttrOp $attr),
                     (I32ElementsAttrOp ConstantAttr<I32ElementsAttr, "0">),
                     [(IsNotScalar $attr)]>;

def TestBranchOp : TEST_Op<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$targetOperands);
  let successors = (successor AnySuccessor:$target);
}

def TestProducingBranchOp : TEST_Op<"producing_br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, Terminator,
     AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$firstOperands,
                       Variadic<AnyType>:$secondOperands);
  let results = (outs I32:$dummy);
  let successors = (successor AnySuccessor:$first,AnySuccessor:$second);
}

// Produces an error value on the error path
def TestInternalBranchOp : TEST_Op<"internal_br",
	[DeclareOpInterfaceMethods<BranchOpInterface>, Terminator,
	 AttrSizedOperandSegments]> {

  let arguments = (ins Variadic<AnyType>:$successOperands,
                       Variadic<AnyType>:$errorOperands);

  let successors = (successor AnySuccessor:$successPath, AnySuccessor:$errorPath);
}

def AttrSizedOperandOp : TEST_Op<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d,
    I32ElementsAttr:$operand_segment_sizes
  );
}

def AttrSizedResultOp : TEST_Op<"attr_sized_results",
                                [AttrSizedResultSegments]> {
  let arguments = (ins
    I32ElementsAttr:$result_segment_sizes
  );
  let results = (outs
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d
  );
}

// This is used to test that the fallback for a custom op's parser and printer
// is the dialect parser and printer hooks.
def CustomFormatFallbackOp : TEST_Op<"dialect_custom_format_fallback">;

// Ops related to OIList primitive
def OIListTrivial : TEST_Op<"oilist_with_keywords_only"> {
  let arguments = (ins UnitAttr:$keyword, UnitAttr:$otherKeyword,
                       UnitAttr:$diffNameUnitAttrKeyword);
  let assemblyFormat = [{
    oilist( `keyword` $keyword
          | `otherKeyword` $otherKeyword
          | `thirdKeyword` $diffNameUnitAttrKeyword) attr-dict
  }];
}

def OIListSimple : TEST_Op<"oilist_with_simple_args", [AttrSizedOperandSegments]> {
  let arguments = (ins Optional<AnyType>:$arg0,
                       Optional<AnyType>:$arg1,
                       Optional<AnyType>:$arg2);
  let assemblyFormat = [{
    oilist( `keyword` $arg0 `:` type($arg0)
          | `otherKeyword` $arg1 `:` type($arg1)
          | `thirdKeyword` $arg2 `:` type($arg2) ) attr-dict
  }];
}

def OIListVariadic : TEST_Op<"oilist_variadic_with_parens", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$arg0,
                       Variadic<AnyType>:$arg1,
                       Variadic<AnyType>:$arg2);
  let assemblyFormat = [{
    oilist( `keyword` `(` $arg0 `:` type($arg0) `)`
          | `otherKeyword` `(` $arg1 `:` type($arg1) `)`
          | `thirdKeyword` `(` $arg2 `:` type($arg2) `)`) attr-dict
  }];
}

def OIListCustom : TEST_Op<"oilist_custom", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$arg0,
                       Optional<I32>:$optOperand,
                       UnitAttr:$nowait);
  let assemblyFormat = [{
    oilist( `private` `(` $arg0 `:` type($arg0) `)`
          | `reduction` custom<CustomOptionalOperand>($optOperand)
          | `nowait` $nowait
    ) attr-dict
  }];
}

def OIListAllowedLiteral : TEST_Op<"oilist_allowed_literal"> {
  let assemblyFormat = [{
    oilist( `foo` | `bar` ) `buzz` attr-dict
  }];
}

// This is used to test encoding of a string attribute into an SSA name of a
// pretty printed value name.
def StringAttrPrettyNameOp
 : TEST_Op<"string_attr_pretty_name",
           [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins StrArrayAttr:$names);
  let results = (outs Variadic<I32>:$r);
  let hasCustomAssemblyFormat = 1;
}

// This is used to test the OpAsmOpInterface::getDefaultDialect() feature:
// operations nested in a region under this op will drop the "test." dialect
// prefix.
def DefaultDialectOp : TEST_Op<"default_dialect", [OpAsmOpInterface]> {
 let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    static ::llvm::StringRef getDefaultDialect() {
      return "test";
    }
    void getAsmResultNames(::llvm::function_ref<void(::mlir::Value, ::llvm::StringRef)> setNameFn) {}
  }];
  let assemblyFormat = "regions attr-dict-with-keyword";
}

// This is used to test the OpAsmOpInterface::getAsmBlockName() feature:
// blocks nested in a region under this op will have a name defined by the
// interface.
def AsmBlockNameOp : TEST_Op<"block_names", [OpAsmOpInterface]> {
 let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    void getAsmBlockNames(mlir::OpAsmSetBlockNameFn setNameFn) {
      std::string name;
      int count = 0;
      for (::mlir::Block &block : getRegion().getBlocks()) {
        name = "foo" + std::to_string(count++);
        setNameFn(&block, name);
      }
    }
  }];
  let assemblyFormat = "regions attr-dict-with-keyword";
}

// This operation requires its return type to have the trait 'TestTypeTrait'.
def ResultTypeWithTraitOp : TEST_Op<"result_type_with_trait", []> {
  let results = (outs AnyType);
  let hasVerifier = 1;
}

// This operation requires its "attr" attribute to have the
// trait 'TestAttrTrait'.
def AttrWithTraitOp : TEST_Op<"attr_with_trait", []> {
  let arguments = (ins AnyAttr:$attr);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Test Locations
//===----------------------------------------------------------------------===//

def TestLocationSrcOp : TEST_Op<"loc_src"> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$output);
}

def TestLocationDstOp : TEST_Op<"loc_dst", [SameOperandsAndResultType]> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$output);
}

def TestLocationSrcNoResOp : TEST_Op<"loc_src_no_res"> {
  let arguments = (ins I32:$input);
  let results = (outs);
}

def TestLocationDstNoResOp : TEST_Op<"loc_dst_no_res"> {
  let arguments = (ins I32:$input);
  let results = (outs);
}

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

// Test named pattern.
def TestNamedPatternRule : Pat<(OpA $input, $attr), (OpB $input, $attr)>;

// Test with fused location.
def : Pat<(OpA (OpA $input, $attr), $bttr), (OpB $input, $bttr)>;

// Test added benefit.
def OpD : TEST_Op<"op_d">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpE : TEST_Op<"op_e">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpF : TEST_Op<"op_f">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpG : TEST_Op<"op_g">, Arguments<(ins I32)>, Results<(outs I32)>;
// Verify that bumping benefit results in selecting different op.
def : Pat<(OpD $input), (OpE $input)>;
def : Pat<(OpD $input), (OpF $input), [], (addBenefit 10)>;
// Verify that patterns with more source nodes are selected before those with fewer.
def : Pat<(OpG $input), (OpB $input, ConstantAttr<I32Attr, "20">:$attr)>;
def : Pat<(OpG (OpG $input)), (OpB $input, ConstantAttr<I32Attr, "34">:$attr)>;

// Test patterns for zero-result op.
def OpH : TEST_Op<"op_h">, Arguments<(ins I32)>, Results<(outs)>;
def OpI : TEST_Op<"op_i">, Arguments<(ins I32)>, Results<(outs)>;
def : Pat<(OpH $input), (OpI $input)>;

// Test patterns for zero-input op.
def OpJ : TEST_Op<"op_j">, Arguments<(ins)>, Results<(outs I32)>;
def OpK : TEST_Op<"op_k">, Arguments<(ins)>, Results<(outs I32)>;
def : Pat<(OpJ), (OpK)>;

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
  let builders = [OpBuilder<(ins "::mlir::FuncOp":$function)>];
}

// Pattern add the argument plus a increasing static number hidden in
// OpMTest function. That value is set into the optional argument.
// That way, we will know if operations is called once or twice.
def OpMGetNullAttr : NativeCodeCall<"Attribute()">;
def OpMAttributeIsNull : Constraint<CPred<"! ($_self)">, "Attribute is null">;
def OpMVal : NativeCodeCall<"opMTest($_builder, $0)">;
def : Pat<(OpM $attr, $optAttr), (OpM $attr, (OpMVal $attr) ),
    [(OpMAttributeIsNull:$optAttr)]>;

// Test `$_` for ignoring op argument match.
def TestIgnoreArgMatchSrcOp : TEST_Op<"ignore_arg_match_src"> {
  let arguments = (ins
    AnyType:$a, AnyType:$b, AnyType:$c,
    AnyAttr:$d, AnyAttr:$e, AnyAttr:$f);
}
def TestIgnoreArgMatchDstOp : TEST_Op<"ignore_arg_match_dst"> {
  let arguments = (ins AnyType:$b, AnyAttr:$f);
}
def : Pat<(TestIgnoreArgMatchSrcOp $_, $b, I32, I64Attr:$_, $_, $f),
          (TestIgnoreArgMatchDstOp $b, $f)>;

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

def OpNativeCodeCall3 : TEST_Op<"native_code_call3"> {
  let arguments = (ins I32:$input);
  let results = (outs I32);
}
// Test that NativeCodeCall is not ignored if it is not used to directly
// replace the matched root op.
def : Pattern<(OpNativeCodeCall3 $input),
              [(NativeCodeCallVoid<"createOpI($_builder, $_loc, $0)"> $input),
               (OpK)]>;

def OpNativeCodeCall4 : TEST_Op<"native_code_call4"> {
  let arguments = (ins AnyType:$input1);
  let results = (outs I32:$output1, I32:$output2);
}
def OpNativeCodeCall5 : TEST_Op<"native_code_call5"> {
  let arguments = (ins I32:$input1, I32:$input2);
  let results = (outs I32:$output1, I32:$output2);
}

def GetFirstI32Result : NativeCodeCall<"success(getFirstI32Result($_self, $0))">;
def BindNativeCodeCallResult : NativeCodeCall<"bindNativeCodeCallResult($0)">;
def : Pat<(OpNativeCodeCall4 (GetFirstI32Result $ret)),
          (OpNativeCodeCall5 (BindNativeCodeCallResult:$native $ret), $native)>;

def OpNativeCodeCall6 : TEST_Op<"native_code_call6"> {
  let arguments = (ins I32:$input1, I32:$input2);
  let results = (outs I32:$output1, I32:$output2);
}
def OpNativeCodeCall7 : TEST_Op<"native_code_call7"> {
  let arguments = (ins I32:$input);
  let results = (outs I32);
}
def BindMultipleNativeCodeCallResult : NativeCodeCall<"bindMultipleNativeCodeCallResult($0, $1)", 2>;
def : Pattern<(OpNativeCodeCall6 $arg1, $arg2),
              [(OpNativeCodeCall7 (BindMultipleNativeCodeCallResult:$native__0 $arg1, $arg2)),
               (OpNativeCodeCall7 $native__1)]>;

// Test AllAttrConstraintsOf.
def OpAllAttrConstraint1 : TEST_Op<"all_attr_constraint_of1"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def OpAllAttrConstraint2 : TEST_Op<"all_attr_constraint_of2"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def Constraint0 : AttrConstraint<
    CPred<"$_self.cast<ArrayAttr>()[0]."
          "cast<::mlir::IntegerAttr>().getInt() == 0">,
    "[0] == 0">;
def Constraint1 : AttrConstraint<
    CPred<"$_self.cast<ArrayAttr>()[1].cast<::mlir::IntegerAttr>().getInt() == 1">,
    "[1] == 1">;
def : Pat<(OpAllAttrConstraint1
            AllAttrConstraintsOf<[Constraint0, Constraint1]>:$attr),
          (OpAllAttrConstraint2 $attr)>;

// Op for testing RewritePattern removing op with inner ops.
def TestOpWithRegionPattern : TEST_Op<"op_with_region_pattern"> {
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
}

def TestOpConstant : TEST_Op<"constant", [ConstantLike, NoSideEffect]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let hasFolder = 1;
}

def OpR : TEST_Op<"op_r">, Arguments<(ins AnyInteger, AnyInteger)>, Results<(outs AnyInteger)>;
def OpS : TEST_Op<"op_s">, Arguments<(ins AnyInteger, AnyAttr:$value)>, Results<(outs AnyInteger)>;

def : Pat<(OpR $input1, (ConstantLikeMatcher I32Attr:$input2)),
          (OpS:$unused $input1, $input2)>;

// Op for testing trivial removal via folding of op with inner ops and no uses.
def TestOpWithRegionFoldNoSideEffect : TEST_Op<
    "op_with_region_fold_no_side_effect", [NoSideEffect]> {
  let regions = (region SizedRegion<1>:$region);
}

// Op for testing folding of outer op with inner ops.
def TestOpWithRegionFold : TEST_Op<"op_with_region_fold"> {
  let arguments = (ins I32:$operand);
  let results = (outs I32);
  let regions = (region SizedRegion<1>:$region);
  let hasFolder = 1;
}

def TestOpWithVariadicResultsAndFolder: TEST_Op<"op_with_variadic_results_and_folder"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
  let hasFolder = 1;
}

def TestCommutativeOp : TEST_Op<"op_commutative", [Commutative]> {
  let arguments = (ins I32:$op1, I32:$op2, I32:$op3, I32:$op4);
  let results = (outs I32);
}

def TestCommutative2Op : TEST_Op<"op_commutative2", [Commutative]> {
  let arguments = (ins I32:$op1, I32:$op2);
  let results = (outs I32);
}

def TestIdempotentTraitOp
 : TEST_Op<"op_idempotent_trait",
           [SameOperandsAndResultType, NoSideEffect, Idempotent]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
}

def TestIdempotentTraitBinaryOp
    : TEST_Op<"op_idempotent_trait_binary",
              [SameOperandsAndResultType, NoSideEffect, Idempotent]> {
  let arguments = (ins I32:$op1, I32:$op2);
  let results = (outs I32);
}

def TestInvolutionTraitNoOperationFolderOp
 : TEST_Op<"op_involution_trait_no_operation_fold",
           [SameOperandsAndResultType, NoSideEffect, Involution]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
}

def TestInvolutionTraitFailingOperationFolderOp
 : TEST_Op<"op_involution_trait_failing_operation_fold",
           [SameOperandsAndResultType, NoSideEffect, Involution]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
  let hasFolder = 1;
}

def TestInvolutionTraitSuccesfulOperationFolderOp
 : TEST_Op<"op_involution_trait_succesful_operation_fold",
           [SameOperandsAndResultType, NoSideEffect, Involution]> {
  let arguments = (ins I32:$op1);
  let results = (outs I32);
  let hasFolder = 1;
}

def TestOpInPlaceFoldAnchor : TEST_Op<"op_in_place_fold_anchor"> {
  let arguments = (ins I32);
  let results = (outs I32);
}

def TestOpInPlaceFold : TEST_Op<"op_in_place_fold"> {
  let arguments = (ins I32:$op, I32Attr:$attr);
  let results = (outs I32);
  let hasFolder = 1;
}

// An op that always fold itself.
def TestPassthroughFold : TEST_Op<"passthrough_fold"> {
  let arguments = (ins AnyType:$op);
  let results = (outs AnyType);
  let hasFolder = 1;
}

def TestDialectCanonicalizerOp : TEST_Op<"dialect_canonicalizable"> {
  let arguments = (ins);
  let results = (outs I32);
}

//===----------------------------------------------------------------------===//
// Test Patterns (Symbol Binding)

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

def OpSymbolBindingNoResult : TEST_Op<"symbol_binding_no_result", []> {
  let arguments = (ins I32:$operand);
}

// Test that we can bind to an op without results and reference it later.
def : Pat<(OpSymbolBindingNoResult:$op $operand),
          (NativeCodeCallVoid<"handleNoResultOp($_builder, $0)"> $op)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Attributes)

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
def MoreConstraint : AttrConstraint<
    CPred<"$_self.cast<IntegerAttr>().getInt() == 4">, "more constraint">;
def : Pat<(OpAttrMatch1 $required, $optional, $default_valued,
                        MoreConstraint:$more),
          (OpAttrMatch2 $required, $optional, $default_valued, $more)>;

// Test unit attrs.
def OpAttrMatch3 : TEST_Op<"match_op_attribute3"> {
  let arguments = (ins UnitAttr:$attr);
  let results = (outs I32);
}
def OpAttrMatch4 : TEST_Op<"match_op_attribute4"> {
  let arguments = (ins UnitAttr:$attr1, UnitAttr:$attr2);
  let results = (outs I32);
}
def : Pat<(OpAttrMatch3 $attr), (OpAttrMatch4 ConstUnitAttr, $attr)>;

// Test with constant attr.
def OpC : TEST_Op<"op_c">, Arguments<(ins I32)>, Results<(outs I32)>;
def : Pat<(OpC $input), (OpB $input, ConstantAttr<I32Attr, "17">:$attr)>;

// Test integer enum attribute in rewrites.
def : Pat<(I32EnumAttrOp I32Case5), (I32EnumAttrOp I32Case10)>;
def : Pat<(I64EnumAttrOp I64Case5), (I64EnumAttrOp I64Case10)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Multi-result Ops)

def MultiResultOpKind1: I64EnumAttrCase<"kind1", 1>;
def MultiResultOpKind2: I64EnumAttrCase<"kind2", 2>;
def MultiResultOpKind3: I64EnumAttrCase<"kind3", 3>;
def MultiResultOpKind4: I64EnumAttrCase<"kind4", 4>;
def MultiResultOpKind5: I64EnumAttrCase<"kind5", 5>;
def MultiResultOpKind6: I64EnumAttrCase<"kind6", 6>;

def MultiResultOpEnum: I64EnumAttr<
  "MultiResultOpEnum", "Multi-result op kinds", [
    MultiResultOpKind1, MultiResultOpKind2, MultiResultOpKind3,
    MultiResultOpKind4, MultiResultOpKind5, MultiResultOpKind6
  ]>;

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

def OneVResOneVOperandOp1 : TEST_Op<"one_variadic_out_one_variadic_in1"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
}
def OneVResOneVOperandOp2 : TEST_Op<"one_variadic_out_one_variadic_in2"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
}

// Rewrite an op with one variadic operand and one variadic result to
// another similar op.
def : Pat<(OneVResOneVOperandOp1 $inputs), (OneVResOneVOperandOp2 $inputs)>;

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

// Rewrite an op with both variadic operands and normal operands.
def : Pat<(MixedVOperandOp1 $input1, $input2, $input3),
          (MixedVOperandOp2 $input1, $input2, $input3)>;

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

// Rewrite an op with both variadic results and normal results.
// Note that because we are generating the op with a top-level result pattern,
// we are able to deduce the correct result types for the generated op using
// the information from the matched root op.
def : Pat<(MixedVResultOp1), (MixedVResultOp2)>;

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

// Generates an op with variadic results using nested pattern.
def : Pat<(OneI32ResultOp),
          (MixedVOperandOp3
              (MixedVResultOp3:$results__0 ConstantAttr<I32Attr, "2">),
              (replaceWithValue $results__1),
              (replaceWithValue $results__2),
              ConstantAttr<I32Attr, "2">)>;

//===----------------------------------------------------------------------===//
// Test Patterns (either)

def TestEitherOpA : TEST_Op<"either_op_a"> {
  let arguments = (ins AnyInteger:$arg0, AnyInteger:$arg1, AnyInteger:$arg2);
  let results = (outs I32:$output);
}

def TestEitherOpB : TEST_Op<"either_op_b"> {
  let arguments = (ins AnyInteger:$arg0);
  let results = (outs I32:$output);
}

def : Pat<(TestEitherOpA (either I32:$arg1, I16:$arg2), $_),
          (TestEitherOpB $arg2)>;

def : Pat<(TestEitherOpA (either (TestEitherOpB I32:$arg1), I16:$arg2), $_),
          (TestEitherOpB $arg2)>;

def : Pat<(TestEitherOpA (either (TestEitherOpB I32:$arg1),
                                 (TestEitherOpB I16:$arg2)),
                          $_),
          (TestEitherOpB $arg2)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Location)

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

def SourceOp : TEST_Op<"source_op"> {
  let arguments = (ins AnyInteger:$arg, AnyI32Attr:$tag);
  let results = (outs AnyInteger);
}

// An op without return type deduction.
def OpX : TEST_Op<"op_x"> {
  let arguments = (ins AnyInteger:$input);
  let results = (outs AnyInteger);
}

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

def OneToTwo : TEST_Op<"one_to_two"> {
  let arguments = (ins AnyInteger);
  let results = (outs AnyInteger, AnyInteger);
}

def TwoToOne : TEST_Op<"two_to_one"> {
  let arguments = (ins AnyInteger, AnyInteger);
  let results = (outs AnyInteger);
}

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

// Test that we can specify both `location` and `returnType` directives.
def : Pat<(SourceOp $val, ConstantAttr<I32Attr, "66">:$attr),
          (TwoToOne (OpX $val, (returnType $val), (location "loc1")),
                    (OpX $val, (location "loc2"), (returnType $val)))>;

//===----------------------------------------------------------------------===//
// Test Legalization
//===----------------------------------------------------------------------===//

def Test_LegalizerEnum_Success : ConstantStrAttr<StrAttr, "Success">;
def Test_LegalizerEnum_Failure : ConstantStrAttr<StrAttr, "Failure">;

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
       $_builder.create<IllegalOpTerminator>($_state.location);
    }]>];
}
def IllegalOpWithRegionAnchor : TEST_Op<"illegal_op_with_region_anchor">;

// Check that smaller pattern depths are chosen, i.e. prioritize more direct
// mappings.
def : Pat<(ILLegalOpA), (LegalOpA Test_LegalizerEnum_Success)>;

def : Pat<(ILLegalOpA), (ILLegalOpB)>;
def : Pat<(ILLegalOpB), (LegalOpA Test_LegalizerEnum_Failure)>;

// Check that the higher benefit pattern is taken for multiple legalizations
// with the same depth.
def : Pat<(ILLegalOpC), (ILLegalOpD)>;
def : Pat<(ILLegalOpD), (LegalOpA Test_LegalizerEnum_Failure)>;

def : Pat<(ILLegalOpC), (ILLegalOpE), [], (addBenefit 10)>;
def : Pat<(ILLegalOpE), (LegalOpA Test_LegalizerEnum_Success)>;

// Check that patterns use the most up-to-date value when being replaced.
def TestRewriteOp : TEST_Op<"rewrite">,
  Arguments<(ins AnyType)>, Results<(outs AnyType)>;
def : Pat<(TestRewriteOp $input), (replaceWithValue $input)>;

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
def TestReturnOp : TEST_Op<"return", [ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>);
  let builders = [OpBuilder<(ins),
    [{ build($_builder, $_state, {}); }]>
  ];
}
def TestCastOp : TEST_Op<"cast">,
  Arguments<(ins Variadic<AnyType>)>, Results<(outs AnyType)>;
def TestInvalidOp : TEST_Op<"invalid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>)>;
def TestTypeProducerOp : TEST_Op<"type_producer">,
  Results<(outs AnyType)>;
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

//===----------------------------------------------------------------------===//
// Test parser.
//===----------------------------------------------------------------------===//

def ParseIntegerLiteralOp : TEST_Op<"parse_integer_literal"> {
  let results = (outs Variadic<Index>:$results);
  let hasCustomAssemblyFormat = 1;
}

def ParseWrappedKeywordOp : TEST_Op<"parse_wrapped_keyword"> {
  let arguments = (ins StrAttr:$keyword);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Test region argument list parsing.

def IsolatedRegionOp : TEST_Op<"isolated_region", [IsolatedFromAbove]> {
  let summary =  "isolated region operation";
  let description = [{
    Test op with an isolated region, to test passthrough region arguments. Each
    argument is of index type.
  }];

  let arguments = (ins Index);
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

def SSACFGRegionOp : TEST_Op<"ssacfg_region",  [
    DeclareOpInterfaceMethods<RegionKindInterface>]> {
  let summary =  "operation with an SSACFG region";
  let description = [{
    Test op that defines an SSACFG region.
  }];

  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
}

def GraphRegionOp : TEST_Op<"graph_region",  [
    DeclareOpInterfaceMethods<RegionKindInterface>]> {
  let summary =  "operation with a graph region";
  let description = [{
    Test op that defines a graph region.
  }];

  let regions = (region AnyRegion:$region);
  let hasCustomAssemblyFormat = 1;
}

def AffineScopeOp : TEST_Op<"affine_scope", [AffineScope]> {
  let summary =  "affine scope operation";
  let description = [{
    Test op that defines a new affine scope.
  }];

  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

def WrappingRegionOp : TEST_Op<"wrapping_region",
    [SingleBlockImplicitTerminator<"TestReturnOp">]> {
  let summary =  "wrapping region operation";
  let description = [{
    Test op wrapping another op in a region, to test calling
    parseGenericOperation from the custom parser.
  }];

  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

def PrettyPrintedRegionOp : TEST_Op<"pretty_printed_region",
    [SingleBlockImplicitTerminator<"TestReturnOp">]> {
  let summary =  "pretty_printed_region operation";
  let description = [{
    Test-op can be printed either in a "pretty" or "non-pretty" way based on
    some criteria. The custom parser parsers both the versions while testing
    APIs: parseCustomOperationName & parseGenericOperationAfterOpName.
  }];
  let arguments = (ins
    AnyType:$input1,
    AnyType:$input2
  );

  let results = (outs AnyType);
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

def PolyForOp : TEST_Op<"polyfor", [OpAsmOpInterface]> {
  let summary =  "polyfor operation";
  let description = [{
    Test op with multiple region arguments, each argument of index type.
  }];
  let extraClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Test OpAsmInterface.

def AsmInterfaceOp : TEST_Op<"asm_interface_op"> {
  let results = (outs AnyType:$first, Variadic<AnyType>:$middle_results,
                      AnyType);
}

def AsmDialectInterfaceOp : TEST_Op<"asm_dialect_interface_op"> {
  let results = (outs AnyType);
}

//===----------------------------------------------------------------------===//
// Test Op Asm Format
//===----------------------------------------------------------------------===//

def FormatLiteralOp : TEST_Op<"format_literal_op"> {
  let assemblyFormat = [{
    `keyword_$.` `->` `:` `,` `=` `<` `>` `(` `)` `[` `]` `` `(` ` ` `)`
    `?` `+` `*` `{` `\n` `}` attr-dict
  }];
}

// Test that we elide attributes that are within the syntax.
def FormatAttrOp : TEST_Op<"format_attr_op"> {
  let arguments = (ins I64Attr:$attr);
  let assemblyFormat = "$attr attr-dict";
}

// Test that we elide optional attributes that are within the syntax.
def FormatOptAttrAOp : TEST_Op<"format_opt_attr_op_a"> {
  let arguments = (ins OptionalAttr<I64Attr>:$opt_attr);
  let assemblyFormat = "(`(` $opt_attr^ `)` )? attr-dict";
}
def FormatOptAttrBOp : TEST_Op<"format_opt_attr_op_b"> {
  let arguments = (ins OptionalAttr<I64Attr>:$opt_attr);
  let assemblyFormat = "($opt_attr^)? attr-dict";
}

// Test that we format symbol name attributes properly.
def FormatSymbolNameAttrOp : TEST_Op<"format_symbol_name_attr_op"> {
  let arguments = (ins SymbolNameAttr:$attr);
  let assemblyFormat = "$attr attr-dict";
}

// Test that we format optional symbol name attributes properly.
def FormatOptSymbolNameAttrOp : TEST_Op<"format_opt_symbol_name_attr_op"> {
  let arguments = (ins OptionalAttr<SymbolNameAttr>:$opt_attr);
  let assemblyFormat = "($opt_attr^)? attr-dict";
}

// Test that we elide attributes that are within the syntax.
def FormatAttrDictWithKeywordOp : TEST_Op<"format_attr_dict_w_keyword"> {
  let arguments = (ins I64Attr:$attr, OptionalAttr<I64Attr>:$opt_attr);
  let assemblyFormat = "attr-dict-with-keyword";
}

// Test that we don't need to provide types in the format if they are buildable.
def FormatBuildableTypeOp : TEST_Op<"format_buildable_type_op"> {
  let arguments = (ins I64:$buildable);
  let results = (outs I64:$buildable_res);
  let assemblyFormat = "$buildable attr-dict";
}

// Test various mixings of region formatting.
class FormatRegionBase<string suffix, string fmt>
    : TEST_Op<"format_region_" # suffix # "_op"> {
  let regions = (region AnyRegion:$region);
  let assemblyFormat = fmt;
}
def FormatRegionAOp : FormatRegionBase<"a", [{
  regions attr-dict
}]>;
def FormatRegionBOp : FormatRegionBase<"b", [{
  $region attr-dict
}]>;
def FormatRegionCOp : FormatRegionBase<"c", [{
  (`region` $region^)? attr-dict
}]>;
class FormatVariadicRegionBase<string suffix, string fmt>
    : TEST_Op<"format_variadic_region_" # suffix # "_op"> {
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let assemblyFormat = fmt;
}
def FormatVariadicRegionAOp : FormatVariadicRegionBase<"a", [{
  $regions attr-dict
}]>;
def FormatVariadicRegionBOp : FormatVariadicRegionBase<"b", [{
  ($regions^ `found_regions`)? attr-dict
}]>;
class FormatRegionImplicitTerminatorBase<string suffix, string fmt>
    : TEST_Op<"format_implicit_terminator_region_" # suffix # "_op",
              [SingleBlockImplicitTerminator<"TestReturnOp">]> {
  let regions = (region AnyRegion:$region);
  let assemblyFormat = fmt;
}
def FormatFormatRegionImplicitTerminatorAOp
    : FormatRegionImplicitTerminatorBase<"a", [{
  $region attr-dict
}]>;

// Test various mixings of result type formatting.
class FormatResultBase<string suffix, string fmt>
    : TEST_Op<"format_result_" # suffix # "_op"> {
  let results = (outs I64:$buildable_res, AnyMemRef:$result);
  let assemblyFormat = fmt;
}
def FormatResultAOp : FormatResultBase<"a", [{
  type($result) attr-dict
}]>;
def FormatResultBOp : FormatResultBase<"b", [{
  type(results) attr-dict
}]>;
def FormatResultCOp : FormatResultBase<"c", [{
  functional-type($buildable_res, $result) attr-dict
}]>;

def FormatVariadicResult : TEST_Op<"format_variadic_result"> {
  let results = (outs Variadic<I64>:$result);
  let assemblyFormat = [{ `:` type($result) attr-dict}];
}

def FormatMultipleVariadicResults : TEST_Op<"format_multiple_variadic_results",
                                            [AttrSizedResultSegments]> {
  let results = (outs Variadic<I64>:$result0, Variadic<AnyType>:$result1);
  let assemblyFormat = [{
    `:` `(` type($result0) `)` `,` `(` type($result1) `)` attr-dict
  }];
}

// Test various mixings of operand type formatting.
class FormatOperandBase<string suffix, string fmt>
    : TEST_Op<"format_operand_" # suffix # "_op"> {
  let arguments = (ins I64:$buildable, AnyMemRef:$operand);
  let assemblyFormat = fmt;
}

def FormatOperandAOp : FormatOperandBase<"a", [{
  operands `:` type(operands) attr-dict
}]>;
def FormatOperandBOp : FormatOperandBase<"b", [{
  operands `:` type($operand) attr-dict
}]>;
def FormatOperandCOp : FormatOperandBase<"c", [{
  $buildable `,` $operand `:` type(operands) attr-dict
}]>;
def FormatOperandDOp : FormatOperandBase<"d", [{
  $buildable `,` $operand `:` type($operand) attr-dict
}]>;
def FormatOperandEOp : FormatOperandBase<"e", [{
  $buildable `,` $operand `:` type($buildable) `,` type($operand) attr-dict
}]>;

def FormatSuccessorAOp : TEST_Op<"format_successor_a_op", [Terminator]> {
  let successors = (successor VariadicSuccessor<AnySuccessor>:$targets);
  let assemblyFormat = "$targets attr-dict";
}

def FormatVariadicOperand : TEST_Op<"format_variadic_operand"> {
  let arguments = (ins Variadic<I64>:$operand);
  let assemblyFormat = [{ $operand `:` type($operand) attr-dict}];
}
def FormatVariadicOfVariadicOperand
   : TEST_Op<"format_variadic_of_variadic_operand"> {
  let arguments = (ins
    VariadicOfVariadic<I64, "operand_segments">:$operand,
    I32ElementsAttr:$operand_segments
  );
  let assemblyFormat = [{ $operand `:` type($operand) attr-dict}];
}

def FormatMultipleVariadicOperands :
    TEST_Op<"format_multiple_variadic_operands", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<I64>:$operand0, Variadic<AnyType>:$operand1);
  let assemblyFormat = [{
    ` ` `(` $operand0 `)` `,` `(` $operand1 `:` type($operand1) `)` attr-dict
  }];
}

// Test various mixings of optional operand and result type formatting.
class FormatOptionalOperandResultOpBase<string suffix, string fmt>
    : TEST_Op<"format_optional_operand_result_" # suffix # "_op",
              [AttrSizedOperandSegments]> {
  let arguments = (ins Optional<I64>:$optional, Variadic<I64>:$variadic);
  let results = (outs Optional<I64>:$optional_res);
  let assemblyFormat = fmt;
}

def FormatOptionalOperandResultAOp : FormatOptionalOperandResultOpBase<"a", [{
  `(` $optional `:` type($optional) `)` `:` type($optional_res)
  (`[` $variadic^ `]`)? attr-dict
}]>;

def FormatOptionalOperandResultBOp : FormatOptionalOperandResultOpBase<"b", [{
  (`(` $optional^ `:` type($optional) `)`)? `:` type($optional_res)
  (`[` $variadic^ `]`)? attr-dict
}]>;

// Test optional result type formatting.
class FormatOptionalResultOpBase<string suffix, string fmt>
    : TEST_Op<"format_optional_result_" # suffix # "_op",
              [AttrSizedResultSegments]> {
  let results = (outs Optional<I64>:$optional, Variadic<I64>:$variadic);
  let assemblyFormat = fmt;
}
def FormatOptionalResultAOp : FormatOptionalResultOpBase<"a", [{
  (`:` type($optional)^ `->` type($variadic))? attr-dict
}]>;

def FormatOptionalResultBOp : FormatOptionalResultOpBase<"b", [{
  (`:` type($optional) `->` type($variadic)^)? attr-dict
}]>;

def FormatOptionalResultCOp : FormatOptionalResultOpBase<"c", [{
  (`:` functional-type($optional, $variadic)^)? attr-dict
}]>;

def FormatOptionalResultDOp
  : TEST_Op<"format_optional_result_d_op" > {
  let results = (outs Optional<F80>:$optional);
  let assemblyFormat = "(`:` type($optional)^)? attr-dict";
}

def FormatTwoVariadicOperandsNoBuildableTypeOp
    : TEST_Op<"format_two_variadic_operands_no_buildable_type_op",
              [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$a,
                       Variadic<AnyType>:$b);
  let assemblyFormat = [{
    `(` $a `:` type($a) `)` `->` `(` $b `:` type($b) `)`  attr-dict
  }];
}

def FormatInferVariadicTypeFromNonVariadic
    : TEST_Op<"format_infer_variadic_type_from_non_variadic",
              [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$args attr-dict `:` type($result)";
}

def FormatOptionalUnitAttr : TEST_Op<"format_optional_unit_attribute"> {
  let arguments = (ins UnitAttr:$is_optional);
  let assemblyFormat = "(`is_optional` $is_optional^)? attr-dict";
}

def FormatOptionalUnitAttrNoElide
    : TEST_Op<"format_optional_unit_attribute_no_elide"> {
  let arguments = (ins UnitAttr:$is_optional);
  let assemblyFormat = "($is_optional^)? attr-dict";
}

def FormatOptionalEnumAttr : TEST_Op<"format_optional_enum_attr"> {
  let arguments = (ins OptionalAttr<SomeI64Enum>:$attr);
  let assemblyFormat = "($attr^)? attr-dict";
}

def FormatOptionalWithElse : TEST_Op<"format_optional_else"> {
  let arguments = (ins UnitAttr:$isFirstBranchPresent);
  let assemblyFormat = "(`then` $isFirstBranchPresent^):(`else`)? attr-dict";
}

def FormatCompoundAttr : TEST_Op<"format_compound_attr"> {
  let arguments = (ins CompoundAttrA:$compound);
  let assemblyFormat = "$compound attr-dict-with-keyword";
}

def FormatNestedAttr : TEST_Op<"format_nested_attr"> {
  let arguments = (ins CompoundAttrNested:$nested);
  let assemblyFormat = "$nested attr-dict-with-keyword";
}

def FormatNestedCompoundAttr : TEST_Op<"format_cpmd_nested_attr"> {
  let arguments = (ins CompoundNestedOuter:$nested);
  let assemblyFormat = "`nested` $nested attr-dict-with-keyword";
}

def FormatQualifiedCompoundAttr : TEST_Op<"format_qual_cpmd_nested_attr"> {
  let arguments = (ins CompoundNestedOuter:$nested);
  let assemblyFormat = "`nested` qualified($nested) attr-dict-with-keyword";
}

def FormatNestedType : TEST_Op<"format_cpmd_nested_type"> {
  let arguments = (ins CompoundNestedOuterType:$nested);
  let assemblyFormat = "$nested `nested` type($nested) attr-dict-with-keyword";
}

def FormatQualifiedNestedType : TEST_Op<"format_qual_cpmd_nested_type"> {
  let arguments = (ins CompoundNestedOuterType:$nested);
  let assemblyFormat = "$nested `nested` qualified(type($nested)) attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// Custom Directives

def FormatCustomDirectiveOperands
    : TEST_Op<"format_custom_directive_operands", [AttrSizedOperandSegments]> {
  let arguments = (ins I64:$operand, Optional<I64>:$optOperand,
                       Variadic<I64>:$varOperands);
  let assemblyFormat = [{
    custom<CustomDirectiveOperands>(
      $operand, $optOperand, $varOperands
    )
    attr-dict
  }];
}

def FormatCustomDirectiveOperandsAndTypes
    : TEST_Op<"format_custom_directive_operands_and_types",
              [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$operand, Optional<AnyType>:$optOperand,
                       Variadic<AnyType>:$varOperands);
  let assemblyFormat = [{
    custom<CustomDirectiveOperandsAndTypes>(
      $operand, $optOperand, $varOperands,
      type($operand), type($optOperand), type($varOperands)
    )
    attr-dict
  }];
}

def FormatCustomDirectiveRegions : TEST_Op<"format_custom_directive_regions"> {
  let regions = (region AnyRegion:$region, VariadicRegion<AnyRegion>:$other_regions);
  let assemblyFormat = [{
    custom<CustomDirectiveRegions>(
      $region, $other_regions
    )
    attr-dict
  }];
}

def FormatCustomDirectiveResults
    : TEST_Op<"format_custom_directive_results", [AttrSizedResultSegments]> {
  let results = (outs AnyType:$result, Optional<AnyType>:$optResult,
                      Variadic<AnyType>:$varResults);
  let assemblyFormat = [{
    custom<CustomDirectiveResults>(
      type($result), type($optResult), type($varResults)
    )
    attr-dict
  }];
}

def FormatCustomDirectiveResultsWithTypeRefs
    : TEST_Op<"format_custom_directive_results_with_type_refs",
              [AttrSizedResultSegments]> {
  let results = (outs AnyType:$result, Optional<AnyType>:$optResult,
                      Variadic<AnyType>:$varResults);
  let assemblyFormat = [{
    custom<CustomDirectiveResults>(
      type($result), type($optResult), type($varResults)
    )
    custom<CustomDirectiveWithTypeRefs>(
      ref(type($result)), ref(type($optResult)), ref(type($varResults))
    )
    attr-dict
  }];
}

def FormatCustomDirectiveWithOptionalOperandRef
    : TEST_Op<"format_custom_directive_with_optional_operand_ref"> {
  let arguments = (ins Optional<I64>:$optOperand);
  let assemblyFormat = [{
    ($optOperand^)? `:`
    custom<CustomDirectiveOptionalOperandRef>(ref($optOperand))
    attr-dict
  }];
}

def FormatCustomDirectiveSuccessors
    : TEST_Op<"format_custom_directive_successors", [Terminator]> {
  let successors = (successor AnySuccessor:$successor,
                              VariadicSuccessor<AnySuccessor>:$successors);
  let assemblyFormat = [{
    custom<CustomDirectiveSuccessors>(
      $successor, $successors
    )
    attr-dict
  }];
}

def FormatCustomDirectiveAttributes
    : TEST_Op<"format_custom_directive_attributes"> {
  let arguments = (ins I64Attr:$attr, OptionalAttr<I64Attr>:$optAttr);
  let assemblyFormat = [{
    custom<CustomDirectiveAttributes>(
      $attr, $optAttr
    )
    attr-dict
  }];
}

def FormatCustomDirectiveAttrDict
    : TEST_Op<"format_custom_directive_attrdict"> {
  let arguments = (ins I64Attr:$attr, OptionalAttr<I64Attr>:$optAttr);
  let assemblyFormat = [{
    custom<CustomDirectiveAttrDict>( attr-dict )
  }];
}

//===----------------------------------------------------------------------===//
// AllTypesMatch type inference

def FormatAllTypesMatchVarOp : TEST_Op<"format_all_types_match_var", [
    AllTypesMatch<["value1", "value2", "result"]>
  ]> {
  let arguments = (ins AnyType:$value1, AnyType:$value2);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value1 `,` $value2 `:` type($value1)";
}

def FormatAllTypesMatchAttrOp : TEST_Op<"format_all_types_match_attr", [
    AllTypesMatch<["value1", "value2", "result"]>
  ]> {
  let arguments = (ins AnyAttr:$value1, AnyType:$value2);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value1 `,` $value2";
}

//===----------------------------------------------------------------------===//
// TypesMatchWith type inference

def FormatTypesMatchVarOp : TEST_Op<"format_types_match_var", [
    TypesMatchWith<"result type matches operand", "value", "result", "$_self">
  ]> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value)";
}

def FormatTypesMatchVariadicOp : TEST_Op<"format_types_match_variadic", [
    RangedTypesMatchWith<"result type matches operand", "value", "result",
                         "llvm::make_range($_self.begin(), $_self.end())">
  ]> {
  let arguments = (ins Variadic<AnyType>:$value);
  let results = (outs Variadic<AnyType>:$result);
  let assemblyFormat = "attr-dict $value `:` type($value)";
}

def FormatTypesMatchAttrOp : TEST_Op<"format_types_match_attr", [
    TypesMatchWith<"result type matches constant", "value", "result", "$_self">
  ]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value";
}

def FormatTypesMatchContextOp : TEST_Op<"format_types_match_context", [
    TypesMatchWith<"tuple result type matches operand type", "value", "result",
        "::mlir::TupleType::get($_ctxt, $_self)">
  ]> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value `:` type($value)";
}

//===----------------------------------------------------------------------===//
// InferTypeOpInterface type inference in assembly format

def FormatInferTypeOp : TEST_Op<"format_infer_type", [InferTypeOpInterface]> {
  let results = (outs AnyType);
  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({::mlir::IntegerType::get(context, 16)});
      return ::mlir::success();
    }
   }];
}

// Check that formatget supports DeclareOpInterfaceMethods.
def FormatInferType2Op : TEST_Op<"format_infer_type2", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let results = (outs AnyType);
  let assemblyFormat = "attr-dict";
}

// Base class for testing mixing allOperandTypes, allOperands, and
// inferResultTypes.
class FormatInferAllTypesBaseOp<string mnemonic, list<Trait> traits = []>
    : TEST_Op<mnemonic, [InferTypeOpInterface] # traits> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$outs);
  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      ::mlir::TypeRange operandTypes = operands.getTypes();
      inferredReturnTypes.assign(operandTypes.begin(), operandTypes.end());
      return ::mlir::success();
    }
   }];
}

// Test inferReturnTypes is called when allOperandTypes and allOperands is true.
def FormatInferTypeAllOperandsAndTypesOp
    : FormatInferAllTypesBaseOp<"format_infer_type_all_operands_and_types"> {
  let assemblyFormat = "`(` operands `)` attr-dict `:` type(operands)";
}

// Test inferReturnTypes is called when allOperandTypes is true and there is one
// ODS operand.
def FormatInferTypeAllOperandsAndTypesOneOperandOp
    : FormatInferAllTypesBaseOp<"format_infer_type_all_types_one_operand"> {
  let assemblyFormat = "`(` $args `)` attr-dict `:` type(operands)";
}

// Test inferReturnTypes is called when allOperandTypes is true and there are
// more than one ODS operands.
def FormatInferTypeAllOperandsAndTypesTwoOperandsOp
    : FormatInferAllTypesBaseOp<"format_infer_type_all_types_two_operands",
                                [SameVariadicOperandSize]> {
  let arguments = (ins Variadic<AnyType>:$args0, Variadic<AnyType>:$args1);
  let assemblyFormat = "`(` $args0 `)` `(` $args1 `)` attr-dict `:` type(operands)";
}

// Test inferReturnTypes is called when allOperands is true and operand types
// are separately specified.
def FormatInferTypeAllTypesOp
    : FormatInferAllTypesBaseOp<"format_infer_type_all_types"> {
  let assemblyFormat = "`(` operands `)` attr-dict `:` type($args)";
}

// Test inferReturnTypes coupled with regions.
def FormatInferTypeRegionsOp
    : TEST_Op<"format_infer_type_regions", [InferTypeOpInterface]> {
  let results = (outs Variadic<AnyType>:$outs);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$region attr-dict";
  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      if (regions.empty())
        return ::mlir::failure();
      auto types = regions.front()->getArgumentTypes();
      inferredReturnTypes.assign(types.begin(), types.end());
      return ::mlir::success();
    }
  }];
}

// Test inferReturnTypes coupled with variadic operands (operand_segment_sizes).
def FormatInferTypeVariadicOperandsOp
    : TEST_Op<"format_infer_type_variadic_operands",
              [InferTypeOpInterface, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<I32>:$a, Variadic<I64>:$b);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = "`(` $a `:` type($a) `)` `(` $b `:` type($b) `)` attr-dict";
  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      FormatInferTypeVariadicOperandsOpAdaptor adaptor(operands, attributes);
      auto aTypes = adaptor.getA().getTypes();
      auto bTypes = adaptor.getB().getTypes();
      inferredReturnTypes.append(aTypes.begin(), aTypes.end());
      inferredReturnTypes.append(bTypes.begin(), bTypes.end());
      return ::mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Test SideEffects
//===----------------------------------------------------------------------===//

def SideEffectOp : TEST_Op<"side_effect_op",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TestEffectOpInterface>]> {
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Test CopyOpInterface
//===----------------------------------------------------------------------===//

def CopyOp : TEST_Op<"copy", [CopyOpInterface]> {
  let description = [{
    Represents a copy operation.
  }];
  let arguments = (ins Res<AnyRankedOrUnrankedMemRef, "", [MemRead]>:$source,
                   Res<AnyRankedOrUnrankedMemRef, "", [MemWrite]>:$target);
  let assemblyFormat = [{
    `(` $source `,` $target `)` `:` `(` type($source) `,` type($target) `)`
     attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Test Buffer/Tensor
//===----------------------------------------------------------------------===//

def RegionYieldOp : TEST_Op<"region_yield",
      [NoSideEffect, ReturnLike, Terminator]> {
  let description = [{
    This operation is used in a region and yields the corresponding type for
    that operation.
  }];
  let arguments = (ins AnyType:$result);
  let assemblyFormat = [{
    $result `:` type($result) attr-dict
  }];
  let builders = [OpBuilder<(ins),
    [{ build($_builder, $_state, {}); }]>
  ];
}

class BufferBasedOpBase<string mnemonic, list<Trait> traits>
    : TEST_Op<mnemonic, traits> {
  let description = [{
    A buffer based operation, that uses memRefs as input and output.
  }];
  let arguments = (ins AnyRankedOrUnrankedMemRef:$input,
                       AnyRankedOrUnrankedMemRef:$output);
}

def BufferBasedOp : BufferBasedOpBase<"buffer_based", []>{
  let assemblyFormat = [{
    `in` `(` $input`:` type($input) `)` `out` `(` $output`:` type($output) `)`
    attr-dict
  }];
}

def RegionBufferBasedOp : BufferBasedOpBase<"region_buffer_based",
      [SingleBlockImplicitTerminator<"RegionYieldOp">]> {
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    `in` `(` $input`:` type($input) `)` `out` `(` $output`:` type($output) `)`
    $region attr-dict
  }];
}

def TensorBasedOp : TEST_Op<"tensor_based", []> {
  let description = [{
    A tensor based operation, that uses a tensor as an input and results in a
    tensor again.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    `in` `(` $input`:` type($input) `)` `->` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Test RegionBranchOpInterface
//===----------------------------------------------------------------------===//

def RegionIfYieldOp : TEST_Op<"region_if_yield",
      [NoSideEffect, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $results `:` type($results) attr-dict
  }];
}

def RegionIfOp : TEST_Op<"region_if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                 ["getRegionInvocationBounds"]>,
       SingleBlockImplicitTerminator<"RegionIfYieldOp">,
       RecursiveSideEffects]> {
  let description =[{
    Represents an abstract if-then-else-join pattern. In this context, the then
    and else regions jump to the join region, which finally returns to its
    parent op.
  }];

  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        AnyRegion:$elseRegion,
                        AnyRegion:$joinRegion);
  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getThenArgs() {
      return getBody(0)->getArguments();
    }
    ::mlir::Block::BlockArgListType getElseArgs() {
      return getBody(1)->getArguments();
    }
    ::mlir::Block::BlockArgListType getJoinArgs() {
      return getBody(2)->getArguments();
    }
    ::mlir::OperandRange getSuccessorEntryOperands(unsigned index);
  }];
  let hasCustomAssemblyFormat = 1;
}

def AnyCondOp : TEST_Op<"any_cond",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                 ["getRegionInvocationBounds"]>,
       RecursiveSideEffects]> {
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// Test TableGen generated build() methods
//===----------------------------------------------------------------------===//

def TableGenConstant : TEST_Op<"tblgen_constant"> {
  let results = (outs AnyType);
}

// No variadic args or results.
def TableGenBuildOp0 : TEST_Op<"tblgen_build_0"> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
}

// Sigle variadic arg and single variadic results.
def TableGenBuildOp1 : TEST_Op<"tblgen_build_1"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
}

// Single variadic arg and non-variadic results.
def TableGenBuildOp2 : TEST_Op<"tblgen_build_2"> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
}

// Single variadic arg and multiple variadic results.
def TableGenBuildOp3 : TEST_Op<"tblgen_build_3", [SameVariadicResultSize]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$resultA, Variadic<AnyType>:$resultB);
}

// Single variadic arg, non variadic results, with SameOperandsAndResultType.
// Tests suppression of ambiguous build methods for operations with
// SameOperandsAndResultType trait.
def TableGenBuildOp4 : TEST_Op<"tblgen_build_4", [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
}

// Base class for testing `build` methods for ops with
// InferReturnTypeOpInterface.
class TableGenBuildInferReturnTypeBaseOp<string mnemonic,
                                         list<Trait> traits = []>
    : TEST_Op<mnemonic, [InferTypeOpInterface] # traits> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *,
          ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({operands[0].getType()});
      return ::mlir::success();
    }
   }];
}

// Single variadic arg with SameOperandsAndResultType and InferTypeOpInterface.
// Tests suppression of ambiguous build methods for operations with
// SameOperandsAndResultType and InferTypeOpInterface.
def TableGenBuildOp5 : TableGenBuildInferReturnTypeBaseOp<
    "tblgen_build_5", [SameOperandsAndResultType]>;

// Op with InferTypeOpInterface and regions.
def TableGenBuildOp6 : TableGenBuildInferReturnTypeBaseOp<
    "tblgen_build_6", [InferTypeOpInterface]> {
  let regions = (region AnyRegion:$body);
}

//===----------------------------------------------------------------------===//
// Test BufferPlacement
//===----------------------------------------------------------------------===//

def GetTupleElementOp: TEST_Op<"get_tuple_element"> {
  let description = [{
    Test op that returns a specified element of the tuple.
  }];

  let arguments = (ins
    TupleOf<[AnyType]>,
    I32Attr:$index
  );
  let results = (outs AnyType);
}

def MakeTupleOp: TEST_Op<"make_tuple"> {
  let description = [{
    Test op that creates a tuple value from a list of values.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs TupleOf<[AnyType]>);
}

//===----------------------------------------------------------------------===//
// Test Target DataLayout
//===----------------------------------------------------------------------===//

def OpWithDataLayoutOp : TEST_Op<"op_with_data_layout",
                                 [HasDefaultDLTIDataLayout, DataLayoutOpInterface]> {
  let summary =
      "An op that uses DataLayout implementation from the Target dialect";
  let regions = (region VariadicRegion<AnyRegion>:$regions);
}

def DataLayoutQueryOp : TEST_Op<"data_layout_query"> {
  let summary = "A token op recognized by data layout query test pass";
  let description = [{
    The data layout query pass pattern-matches this op and attaches to it an
    array attribute containing the result of data layout query of the result
    type of this op.
  }];

  let results = (outs AnyType:$res);
}

//===----------------------------------------------------------------------===//
// Test Reducer Patterns
//===----------------------------------------------------------------------===//

def OpCrashLong : TEST_Op<"op_crash_long"> {
  let arguments = (ins I32, I32, I32);
  let results = (outs I32);
}

def OpCrashShort : TEST_Op<"op_crash_short"> {
  let results = (outs I32);
}

def : Pat<(OpCrashLong $_, $_, $_), (OpCrashShort)>;

//===----------------------------------------------------------------------===//
// Test LinalgConvolutionOpInterface.
//===----------------------------------------------------------------------===//

def TestLinalgConvOpNotLinalgOp : TEST_Op<"conv_op_not_linalg_op", [
    LinalgConvolutionOpInterface]> {
  let arguments = (ins
    AnyType:$image, AnyType:$filter, AnyType:$output);
  let results = (outs AnyRankedTensor:$result);
}

def TestLinalgConvOp :
  TEST_Op<"linalg_conv_op", [AttrSizedOperandSegments, SingleBlock,
      LinalgStructuredInterface, LinalgConvolutionOpInterface]> {

  let arguments = (ins Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    bool hasIndexSemantics() { return false; }

    static void regionBuilder(mlir::ImplicitLocOpBuilder &b, mlir::Block &block,
                              mlir::ArrayRef<mlir::NamedAttribute> attrs) {
      b.create<mlir::linalg::YieldOp>(block.getArguments().back());
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder() {
      return &regionBuilder;
    }

    mlir::ArrayAttr iterator_types() {
      return getOperation()->getAttrOfType<mlir::ArrayAttr>("iterator_types");
    }

    mlir::ArrayAttr indexing_maps() {
      return getOperation()->getAttrOfType<mlir::ArrayAttr>("indexing_maps");
    }

    std::string getLibraryCallName() {
      return "";
    }

    // To conform with interface requirement on operand naming.
    mlir::ValueRange inputs() { return getInputs(); }
    mlir::ValueRange outputs() { return getOutputs(); }
  }];
}

//===----------------------------------------------------------------------===//
// Test LinalgFillOpInterface.
//===----------------------------------------------------------------------===//

def TestLinalgFillOpNotLinalgOp : TEST_Op<"fill_op_not_linalg_op", [
    LinalgFillOpInterface]> {
  let arguments = (ins
    AnyType:$value, AnyType:$output);
  let results = (outs AnyRankedTensor:$result);
}

def TestLinalgFillOp :
  TEST_Op<"linalg_fill_op", [AttrSizedOperandSegments, SingleBlock,
      LinalgStructuredInterface, LinalgFillOpInterface]> {

  let arguments = (ins Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    `outs` `(` $outputs `:` type($outputs) `)`
    $region (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    bool hasIndexSemantics() { return false; }

    static void regionBuilder(mlir::ImplicitLocOpBuilder &b, mlir::Block &block,
                              mlir::ArrayRef<mlir::NamedAttribute> attrs) {
      b.create<mlir::linalg::YieldOp>(block.getArguments().back());
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
    getRegionBuilder() {
      return &regionBuilder;
    }

    mlir::ArrayAttr iterator_types() {
      return getOperation()->getAttrOfType<mlir::ArrayAttr>("iterator_types");
    }

    mlir::ArrayAttr indexing_maps() {
      return getOperation()->getAttrOfType<mlir::ArrayAttr>("indexing_maps");
    }

    std::string getLibraryCallName() {
      return "";
    }

    // To conform with interface requirement on operand naming.
    mlir::ValueRange inputs() { return getInputs(); }
    mlir::ValueRange outputs() { return getOutputs(); }
  }];
}

//===----------------------------------------------------------------------===//
// Test Ops with Default-Valued String Attributes
//===----------------------------------------------------------------------===//

def TestDefaultStrAttrNoValueOp : TEST_Op<"no_str_value"> {
  let arguments = (ins DefaultValuedAttr<StrAttr, "">:$value);
  let assemblyFormat = "attr-dict";
}

def TestDefaultStrAttrHasValueOp : TEST_Op<"has_str_value"> {
  let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$value);
  let assemblyFormat = "attr-dict";
}

def : Pat<(TestDefaultStrAttrNoValueOp $value),
          (TestDefaultStrAttrHasValueOp ConstantStrAttr<StrAttr, "foo">)>;

//===----------------------------------------------------------------------===//
// Test Ops with effects
//===----------------------------------------------------------------------===//

def TestResource : Resource<"TestResource">;

def TestEffectsOpA : TEST_Op<"op_with_effects_a"> {
    let arguments = (ins
        Arg<Variadic<AnyMemRef>, "", [MemRead]>,
        Arg<FlatSymbolRefAttr, "", [MemRead]>:$first,
        Arg<SymbolRefAttr, "", [MemWrite]>:$second,
        Arg<OptionalAttr<SymbolRefAttr>, "", [MemRead]>:$optional_symbol
        );

    let results = (outs Res<AnyMemRef, "", [MemAlloc<TestResource>]>);
}

def TestEffectsOpB : TEST_Op<"op_with_effects_b",
    [MemoryEffects<[MemWrite<TestResource>]>]>;

def TestEffectsRead : TEST_Op<"op_with_memread",
    [MemoryEffects<[MemRead]>]> {
  let results = (outs AnyInteger);
}

def TestEffectsWrite : TEST_Op<"op_with_memwrite",
    [MemoryEffects<[MemWrite]>]>;

//===----------------------------------------------------------------------===//
// Test Ops with verifiers
//===----------------------------------------------------------------------===//

def TestVerifiersOp : TEST_Op<"verifiers",
                              [SingleBlock, NoTerminator, IsolatedFromAbove]> {
  let arguments = (ins I32:$input);
  let regions = (region SizedRegion<1>:$region);
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

#endif // TEST_OPS
