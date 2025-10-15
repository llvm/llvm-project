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
include "TestInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ValueBoundsOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

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
    CPred<"::llvm::isa<::test::TestType>($_self)">, "test">,
    BuildableType<"$_builder.getType<::test::TestType>()">;

def SignlessLikeVariadic : TEST_Op<"signless_like_variadic"> {
  let arguments = (ins Variadic<SignlessIntegerLike>:$x);
}

//===----------------------------------------------------------------------===//
// Test Symbols
//===----------------------------------------------------------------------===//

def SymbolOp : TEST_Op<"symbol", [NoMemoryEffect, Symbol]> {
  let summary =  "operation which defines a new symbol";
  let arguments = (ins StrAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
}

def OverriddenSymbolVisibilityOp : TEST_Op<"overridden_symbol_visibility", [
  DeclareOpInterfaceMethods<Symbol, ["getVisibility", "setVisibility"]>,
]> {
  let summary =  "operation overridden symbol visibility accessors";
  let arguments = (ins StrAttr:$sym_name);
}

def SymbolScopeOp : TEST_Op<"symbol_scope",
    [SymbolTable, SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary =  "operation which defines a new symbol table";
  let regions = (region SizedRegion<1>:$region);
}

def SymbolScopeIsolatedOp
    : TEST_Op<"symbol_scope_isolated", [IsolatedFromAbove, SymbolTable,
                                        SingleBlockImplicitTerminator<
                                            "TerminatorOp">]> {
  let summary =
      "operation which defines a new symbol table that is IsolatedFromAbove";
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
      ConfinedAttr<I32Attr, [IntNonNegative]>:$i32attr,
      ConfinedAttr<I64Attr, [IntNonNegative]>:$i64attr
  );
}

def PositiveIntAttrOp : TEST_Op<"positive_int_attr"> {
  let arguments = (ins
      ConfinedAttr<I32Attr, [IntPositive]>:$i32attr,
      ConfinedAttr<I64Attr, [IntPositive]>:$i64attr
  );
}

def TypeArrayAttrOp : TEST_Op<"type_array_attr"> {
  let arguments = (ins TypeArrayAttr:$attr);
}

def TypeArrayAttrWithDefaultOp : TEST_Op<"type_array_attr_with_default"> {
  let arguments = (ins DefaultValuedAttr<TypeArrayAttr, "{}">:$attr);
}

def TypedStringAttrWithTypeOp : TEST_Op<"string_attr_with_type"> {
  let arguments = (ins TypedStrAttr<AnyInteger>:$attr);
  let assemblyFormat = "$attr attr-dict";
}

def TypedStringAttrWithMixedTypeOp : TEST_Op<"string_attr_with_mixed_type"> {
  let arguments = (ins
    AnyAttrOf<[TypedStrAttr<AnyInteger>, I64Attr]>:$attr
  );
  let assemblyFormat = "$attr attr-dict";
}

def FloatAttrOp : TEST_Op<"float_attrs"> {
  // TODO: Clean up the OpBase float type and attribute selectors so they
  // can express all of the types.
  let arguments = (ins
    AnyAttr:$float_attr
  );
}

def I32EnumAttrOp : TEST_Op<"i32_enum_attr"> {
  let arguments = (ins SomeI32Enum:$attr);
  let results = (outs I32:$val);
}

def I64EnumAttrOp : TEST_Op<"i64_enum_attr"> {
  let arguments = (ins SomeI64Enum:$attr);
  let results = (outs I32:$val);
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
  DerivedAttr num_elements = DerivedAttr<"int",
    "return ::llvm::cast<ShapedType>(getOutput().getType()).getNumElements();",
    "$_builder.getI32IntegerAttr($_self)">;
}

def TestPropOp : TEST_Op<"prop">,
  Arguments<(ins Variadic<Index>:$upperInits,
      I32ElementsAttr:$transforms)>,
  Results<(outs Variadic<AnyType>:$results)> {
  DerivedAttr upperLen = DerivedAttr<"uint32_t", [{
    return getUpperInits().size() / getTransforms().size();
  }], [{ $_builder.getI32IntegerAttr($_self) }]>;
}


def StringElementsAttrOp : TEST_Op<"string_elements_attr"> {
  let arguments = (ins
      StringElementsAttr:$scalar_string_attr
  );
}

def TypedAttrOp : TEST_Op<"typed_attr"> {
  let arguments = (ins TypeAttr:$type, AnyAttr:$attr);
  let assemblyFormat = [{
    attr-dict $type `=` custom<AttrElideType>(ref($type), $attr)
  }];
}

def TypeAttrOfOp : TEST_Op<"type_attr_of"> {
  let arguments = (ins TypeAttrOf<I64>:$type);
  let assemblyFormat = [{
    attr-dict $type
  }];
}

def DenseArrayAttrOp : TEST_Op<"dense_array_attr"> {
  let arguments = (ins
    DenseBoolArrayAttr:$i1attr,
    DenseI8ArrayAttr:$i8attr,
    DenseI16ArrayAttr:$i16attr,
    DenseI32ArrayAttr:$i32attr,
    DenseI64ArrayAttr:$i64attr,
    DenseF32ArrayAttr:$f32attr,
    DenseF64ArrayAttr:$f64attr,
    DenseI32ArrayAttr:$emptyattr
  );
  let assemblyFormat = [{
   `i1attr` `=` $i1attr `i8attr` `=` $i8attr `i16attr` `=` $i16attr
   `i32attr` `=` $i32attr `i64attr` `=` $i64attr  `f32attr` `=` $f32attr
   `f64attr` `=` $f64attr `emptyattr` `=` $emptyattr attr-dict
  }];
}

def SlashAttrOp : TEST_Op<"slash_attr"> {
  let arguments = (ins SlashAttr:$attr);
}

//===----------------------------------------------------------------------===//
// Test Attributes Constraints
//===----------------------------------------------------------------------===//

def ConfinedDenseArrayAttrOp : TEST_Op<"confined_dense_array_attr"> {
  let arguments = (ins
      ConfinedAttr<DenseI16ArrayAttr,
                   [DenseArrayStrictlySorted<DenseI16ArrayAttr>]>:$emptyattr,
      ConfinedAttr<DenseI32ArrayAttr,
                   [DenseArraySorted<DenseI32ArrayAttr>]>:$i32attr,
      ConfinedAttr<DenseI64ArrayAttr,
                   [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$i64attr
  );
}

// It does not make sense to have this constraint on a DenseBoolArrayAttr.
def DenseArrayStrictlyPositiveAttrOp : TEST_Op<"confined_strictly_positive_attr"> {
  let arguments = (ins
      ConfinedAttr<DenseI8ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseI8ArrayAttr>]>:$i8attr,
      ConfinedAttr<DenseI16ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseI16ArrayAttr>]>:$i16attr,
      ConfinedAttr<DenseI32ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseI32ArrayAttr>]>:$i32attr,
      ConfinedAttr<DenseI64ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseI64ArrayAttr>]>:$i64attr,
      ConfinedAttr<DenseF32ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseF32ArrayAttr>]>:$f32attr,
      ConfinedAttr<DenseF64ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseF64ArrayAttr>]>:$f64attr,
      ConfinedAttr<DenseI16ArrayAttr,
                   [DenseArrayStrictlyPositive<DenseI16ArrayAttr>]>:$emptyattr
  );
}

// It does not make sense to have this constraint on a DenseBoolArrayAttr.
// It is always true.
def DenseArrayNonNegativeOp : TEST_Op<"confined_non_negative_attr"> {
  let arguments = (ins
      ConfinedAttr<DenseI8ArrayAttr,
                   [DenseArrayNonNegative<DenseI8ArrayAttr>]>:$i8attr,
      ConfinedAttr<DenseI16ArrayAttr,
                   [DenseArrayNonNegative<DenseI16ArrayAttr>]>:$i16attr,
      ConfinedAttr<DenseI32ArrayAttr,
                   [DenseArrayNonNegative<DenseI32ArrayAttr>]>:$i32attr,
      ConfinedAttr<DenseI64ArrayAttr,
                   [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$i64attr,
      ConfinedAttr<DenseF32ArrayAttr,
                   [DenseArrayNonNegative<DenseF32ArrayAttr>]>:$f32attr,
      ConfinedAttr<DenseF64ArrayAttr,
                   [DenseArrayNonNegative<DenseF64ArrayAttr>]>:$f64attr,
      ConfinedAttr<DenseI16ArrayAttr,
                   [DenseArrayNonNegative<DenseI16ArrayAttr>]>:$emptyattr
  );
}

//===----------------------------------------------------------------------===//
// Test Promised Interfaces Constraints
//===----------------------------------------------------------------------===//

def PromisedInterfacesOp : TEST_Op<"promised_interfaces"> {
  let arguments = (ins
      ConfinedAttr<AnyAttr,
          [PromisedAttrInterface<TestExternalAttrInterface>]>:$promisedAttr,
      ConfinedType<AnyType,
          [HasPromiseOrImplementsTypeInterface<TestExternalTypeInterface>]
        >:$promisedType
  );
}

//===----------------------------------------------------------------------===//
// Test Enum Attributes
//===----------------------------------------------------------------------===//

// Define the enum attribute.
def TestEnumAttr : EnumAttr<Test_Dialect, TestEnum, "enum">;

// Define an op that contains the enum attribute.
def OpWithEnum : TEST_Op<"op_with_enum"> {
  let arguments = (ins TestEnumAttr:$value, OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

// Define a pattern that matches and creates an enum attribute.
def : Pat<(OpWithEnum ConstantEnumCase<TestEnumAttr, "first">:$value,
                      ConstantAttr<I32Attr, "0">:$tag),
          (OpWithEnum ConstantEnumCase<TestEnumAttr, "second">,
                      ConstantAttr<I32Attr, "1">)>;

//===----------------------------------------------------------------------===//
// Test Enum Properties
//===----------------------------------------------------------------------===//

// Define the enum property.
def TestEnumProp : EnumProp<TestEnum>;
// Define an op that contains the enum property.
def OpWithEnumProp : TEST_Op<"op_with_enum_prop"> {
  let arguments = (ins TestEnumProp:$value);
  let assemblyFormat = "$value attr-dict";
}

def TestEnumPropAttrForm : EnumPropWithAttrForm<TestEnum, TestEnumAttr>;
def OpWithEnumPropAttrForm : TEST_Op<"op_with_enum_prop_attr_form"> {
  let arguments = (ins TestEnumPropAttrForm:$value);
  let assemblyFormat = "prop-dict attr-dict";
}

def TestEnumPropAttrFormAlways : EnumPropWithAttrForm<TestEnum, TestEnumAttr> {
  let storeInCustomAttribute = 1;
}
def OpWithEnumPropAttrFormAlways : TEST_Op<"op_with_enum_prop_attr_form_always"> {
  let arguments = (ins TestEnumPropAttrFormAlways:$value);
  let assemblyFormat = "prop-dict attr-dict";
}

def TestBitEnumProp : EnumProp<TestBitEnum> {
  let defaultValue = TestBitEnum.cppType # "::Read";
}
def OpWithTestBitEnum : TEST_Op<"op_with_bit_enum_prop"> {
  let arguments = (ins
    TestBitEnumProp:$value1,
    TestBitEnumProp:$value2);
  let assemblyFormat = "$value1 ($value2^)? attr-dict `:` `(``)`";
}

def TestBitEnumPropNamed : NamedEnumProp<TestBitEnum, "bit_enum"> {
  let defaultValue = TestBitEnum.cppType # "::Read";
}
def OpWithBitEnumPropNamed : TEST_Op<"op_with_bit_enum_prop_named"> {
  let arguments = (ins
    TestBitEnumPropNamed:$value1,
    TestBitEnumPropNamed:$value2);
  let assemblyFormat = "$value1 ($value2^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// Test Bit Enum Attributes
//===----------------------------------------------------------------------===//

// Define the enum attribute.
def TestBitEnumAttr : EnumAttr<Test_Dialect, TestBitEnum, "bit_enum"> {
  let assemblyFormat = "`<` $value `>`";
}

// Define an op that contains the enum attribute.
def OpWithBitEnum : TEST_Op<"op_with_bit_enum"> {
  let arguments = (ins TestBitEnumAttr:$value, OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

def TestBitEnumVerticalBarAttr
    : EnumAttr<Test_Dialect, TestBitEnumVerticalBar, "bit_enum_vbar"> {
  let assemblyFormat = "`<` $value `>`";
}

// Define an op that contains the enum attribute.
def OpWithBitEnumVerticalBar : TEST_Op<"op_with_bit_enum_vbar"> {
  let arguments = (ins TestBitEnumVerticalBarAttr:$value,
                   OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

// Define a pattern that matches and creates a bit enum attribute.
def : Pat<(OpWithBitEnum ConstantEnumCase<TestBitEnumAttr, "write|execute">,
                         ConstantAttr<I32Attr, "0">),
          (OpWithBitEnum ConstantEnumCase<TestBitEnumAttr, "execute|read">,
                         ConstantAttr<I32Attr, "1">)>;

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
    static llvm::LogicalResult inferReturnTypes(mlir::MLIRContext *context,
          std::optional<::mlir::Location> location, mlir::ValueRange operands,
          mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, mlir::RegionRange regions,
          llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({mlir::IntegerType::get(context, 16)});
      return mlir::success();
    }
  }];
}

def OneRegionWithOperandsOp : TEST_Op<"one_region_with_operands_op", []> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let regions = (region AnyRegion);
}

def IsolatedOneRegionOp : TEST_Op<"isolated_one_region_op", [IsolatedFromAbove]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$my_region);
  let assemblyFormat = [{
    attr-dict-with-keyword $operands $my_region `:` type($operands) `->` type($results)
  }];
}

def IsolatedRegionsOp : TEST_Op<"isolated_regions", [IsolatedFromAbove]> {
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let assemblyFormat = "attr-dict-with-keyword $regions";
}

def AllocaScopeRegionOp : TEST_Op<"alloca_scope_region",
                                  [AutomaticAllocationScope]> {
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "attr-dict-with-keyword $region";
}

def OneRegionWithRecursiveMemoryEffectsOp
    : TEST_Op<"one_region_with_recursive_memory_effects", [
        RecursiveMemoryEffects]> {
  let description = [{
    Op that has one region and recursive side effects. The
    RegionBranchOpInterface is not implemented on this op.
  }];
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);
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
  let arguments = (ins
    Variadic<AnyType>:$arg_operands,
    SymbolRefAttr:$callee,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee();

    /// Set the callee for this operation.
    void setCalleeFromCallable(::mlir::CallInterfaceCallable);
  }];
  let extraClassDefinition = [{
    ::mlir::CallInterfaceCallable $cppClass::getCallableForCallee() {
      return (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee");
    }

    void $cppClass::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];
}

def ConversionFuncOp : TEST_Op<"conversion_func_op", [FunctionOpInterface]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the argument types of this async function.
    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this async function.
    ::mlir::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns the number of results of this async function
    unsigned getNumResults() {return getResultTypes().size();}
  }];

  let hasCustomAssemblyFormat = 1;
}

def FunctionalRegionOp : TEST_Op<"functional_region_op",
    [CallableOpInterface]> {
  let regions = (region AnyRegion:$body);
  let arguments = (ins
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs FunctionType);

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return ::llvm::cast<::mlir::FunctionType>(getType()).getResults();
    }
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return ::llvm::cast<::mlir::FunctionType>(getType()).getInputs();
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
  let arguments = (ins TypedAttrInterface:$attr);
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

def OperandsAndResultHaveSameRank :
    TEST_Op<"operands_and_result_have_same_rank", [SameOperandsAndResultRank]> {
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
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

def OpWithInferTypeAdaptorInterfaceOp : TEST_Op<"op_with_infer_type_adaptor_if", [
    InferTypeOpAdaptor]> {
  let arguments = (ins AnyTensor:$x, AnyTensor:$y);
  let results = (outs AnyTensor);
}

def OpWithRefineTypeInterfaceOp : TEST_Op<"op_with_refine_type_if", [
    DeclareOpInterfaceMethods<InferTypeOpInterface,
        ["refineReturnTypes"]>]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

def OpWithShapedTypeInferTypeInterfaceOp : TEST_Op<"op_with_shaped_type_infer_type_if",
      [InferTensorTypeWithReify]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

def OpWithShapedTypeInferTypeAdaptorInterfaceOp :
      TEST_Op<"op_with_shaped_type_infer_type_adaptor_if",
              [InferTensorTypeAdaptorWithReify]> {
  let arguments = (ins AnyTensor:$operand1, AnyTensor:$operand2);
  let results = (outs AnyTensor:$result);
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
    Variadic<I32>:$d
  );
}

def AttrSizedResultOp : TEST_Op<"attr_sized_results",
                                [AttrSizedResultSegments]> {
  let results = (outs
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d
  );
}

def AttrSizedResultCompileTestOp : TEST_Op<"attr_sized_results_compile_test",
                                           [AttrSizedResultSegments]> {
  let results = (outs Variadic<I32>:$a, I32:$b, Optional<I32>:$c);
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


// This is used to test encoding of a string attribute into an SSA name of a
// pretty printed value name.
def CustomResultsNameOp
 : TEST_Op<"custom_result_name",
           [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let arguments = (ins
    Variadic<AnyInteger>:$optional,
    StrArrayAttr:$names
  );
  let results = (outs Variadic<AnyInteger>:$r);
}

// This is used to test OpAsmTypeInterface::getAsmName for op result name,
def ResultNameFromTypeOp
 : TEST_Op<"result_name_from_type",
           [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let results = (outs AnyType:$r);
}

// This is used to test OpAsmTypeInterface::getAsmName for block argument,
def BlockArgumentNameFromTypeOp
  : TEST_Op<"block_argument_name_from_type",
      [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "regions attr-dict-with-keyword";
}

// This is used to test OpAsmTypeInterface::getAsmName's integration with AsmPrinter
// for op result name when OpAsmOpInterface::getAsmResultNames is the default implementation
// i.e. does nothing.
def ResultNameFromTypeInterfaceOp
 : TEST_Op<"result_name_from_type_interface",
           [OpAsmOpInterface]> {
  let results = (outs Variadic<AnyType>:$r);
}

// This is used to test OpAsmTypeInterface::getAsmName's integration with AsmPrinter
// for block argument name when OpAsmOpInterface::getAsmBlockArgumentNames is the default implementation
// i.e. does nothing.
def BlockArgumentNameFromTypeInterfaceOp
  : TEST_Op<"block_argument_name_from_type_interface",
      [OpAsmOpInterface]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "regions attr-dict-with-keyword";
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
def : Pat<(OpD $input), (OpF $input), [], [], (addBenefit 10)>;
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

def OpQ : TEST_Op<"op_q"> {
  let arguments = (ins AnyType, AnyType);
  let results = (outs AnyType);
}

// Test constant-folding a pattern that maps `(F32) -> SI32`.
def SignOp : TEST_Op<"sign", [SameOperandsAndResultShape]> {
  let arguments = (ins RankedTensorOf<[F32]>:$operand);
  let results = (outs RankedTensorOf<[SI32]>:$result);

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

// Test AllAttrOf.
def OpAllAttrConstraint1 : TEST_Op<"all_attr_constraint_of1"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def OpAllAttrConstraint2 : TEST_Op<"all_attr_constraint_of2"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def Constraint0 : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<ArrayAttr>($_self)[0]).getInt() == 0">,
    "[0] == 0">;
def Constraint1 : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<ArrayAttr>($_self)[1]).getInt() == 1">,
    "[1] == 1">;
def : Pat<(OpAllAttrConstraint1
            AllAttrOf<[Constraint0, Constraint1]>:$attr),
          (OpAllAttrConstraint2 $attr)>;

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

def : Pat<(OpR $input1, (ConstantLikeMatcher I32Attr:$input2)),
          (OpS:$unused $input1, $input2)>;

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
def : Pat<(TestOpInPlaceSelfFold:$op $_),
          (TestOpConstant ConstantAttr<I32Attr, "97">)>;

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
def MoreConstraint : AttrConstraint<
    CPred<"::llvm::cast<IntegerAttr>($_self).getInt() == 4">, "more constraint">;
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
//===----------------------------------------------------------------------===//

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

def MixedVOperandOp6 : TEST_Op<"mixed_variadic_in6",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    Variadic<I32>:$input2,
    I32Attr:$attr1
  );
}

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

def MixedVOperandOp7 : TEST_Op<"mixed_variadic_optional_in7",
                               [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    Optional<I32>:$input2,
    I32Attr:$attr1
  );
}

def : Pat<
  (MixedVOperandOp7 $input1, $input2, ConstantAttr<I32Attr, "2">:$attr1),
  (MixedVOperandOp6 $input1, (variadic $input2), $attr1),
  [(Constraint<CPred<"$0 != Value()">> $input2)]>;

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

def TestEitherHelperOpA : TEST_Op<"either_helper_op_a"> {
  let arguments = (ins I32:$arg0);
  let results = (outs I32:$output);
}

def TestEitherHelperOpB : TEST_Op<"either_helper_op_b"> {
  let arguments = (ins I32:$arg0);
  let results = (outs I32:$output);
}

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
def TestReturnOp : TEST_Op<"return", [Pure, ReturnLike, Terminator]> {
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

def ParseB64BytesOp : TEST_Op<"parse_b64"> {
  let arguments = (ins StrAttr:$b64);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Test region argument list parsing.
//===----------------------------------------------------------------------===//

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
  let assemblyFormat = "attr-dict-with-keyword $region";
}

def IsolatedGraphRegionOp : TEST_Op<"isolated_graph_region",  [
    DeclareOpInterfaceMethods<RegionKindInterface>,
    IsolatedFromAbove]> {
  let summary =  "isolated from above operation with a graph region";
  let description = [{
    Test op that defines a graph region which is isolated from above.
  }];

  let regions = (region AnyRegion:$region);
  let assemblyFormat = "attr-dict-with-keyword $region";
}

def AffineScopeOp : TEST_Op<"affine_scope", [AffineScope]> {
  let summary =  "affine scope operation";
  let description = [{
    Test op that defines a new affine scope.
  }];

  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Custom printer/parser
//===----------------------------------------------------------------------===//

def CustomDimensionListAttrOp : TEST_Op<"custom_dimension_list_attr"> {
  let description = [{
    Test printing/parsing of dimension list attribute.
  }];
  let arguments = (ins DenseI64ArrayAttr:$dimension_list);
  let assemblyFormat = [{
    `dimension_list` `=` custom<DimensionList>($dimension_list)
    attr-dict
  }];
}

def OptionalCustomAttrOp : TEST_Op<"optional_custom_attr"> {
  let description = [{
    Test using a custom directive as the optional group anchor and the first
    element to parse. It is expected to return an `OptionalParseResult`.
  }];
  let arguments = (ins OptionalAttr<I1Attr>:$attr);
  let assemblyFormat = [{
    attr-dict (custom<OptionalCustomParser>($attr)^) : (`bar`)?
  }];
}

//===----------------------------------------------------------------------===//
// Test OpAsmInterface.
//===----------------------------------------------------------------------===//

def AsmInterfaceOp : TEST_Op<"asm_interface_op"> {
  let results = (outs AnyType:$first, Variadic<AnyType>:$middle_results,
                      AnyType);
}

def AsmDialectInterfaceOp : TEST_Op<"asm_dialect_interface_op"> {
  let results = (outs AnyType);
}

//===----------------------------------------------------------------------===//
// Test ArrayOfAttr
//===----------------------------------------------------------------------===//

// Embed the array attributes directly in the assembly format for a nice syntax.
def ArrayOfAttrOp : TEST_Op<"array_of_attr_op"> {
  let arguments = (ins TestArrayOfUglyAttrs:$a, TestArrayOfInts:$b,
                       TestArrayOfEnums:$c);
  let assemblyFormat = "`a` `=` $a `,` `b` `=` $b `,` `c` `=` $c attr-dict";
}

//===----------------------------------------------------------------------===//
// Test SideEffects
//===----------------------------------------------------------------------===//

def SideEffectOp : TEST_Op<"side_effect_op",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TestEffectOpInterface>]> {
  let results = (outs AnyType:$result);
}

def SideEffectWithRegionOp : TEST_Op<"side_effect_with_region_op",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TestEffectOpInterface>]> {
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    `(` $operand`)` $region attr-dict `:`  type($operand)  `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Copy Operation Test 
//===----------------------------------------------------------------------===//

def CopyOp : TEST_Op<"copy", []> {
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
      [Pure, ReturnLike, Terminator]> {
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
  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "reading",
                           [MemRead]>:$input,
                       Arg<AnyRankedOrUnrankedMemRef, "writing",
                           [MemWrite]>:$output);
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

def ReadBufferOp : TEST_Op<"read_buffer", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    An operation that reads the buffer operand and dumps its contents.
  }];
  let arguments = (ins AnyRankedOrUnrankedMemRef:$buffer);
}

def ForwardBufferOp : TEST_Op<"forward_buffer", [Pure]> {
  let description = [{
    A pure operation that takes a buffer and returns a buffer. This op does not
    have any side effects, so it cannot allocate or read a buffer from memory.
    It must return the input buffer (or a view thereof). This op purposely does
    does not implement any interface.
  }];
  let arguments = (ins AnyRankedOrUnrankedMemRef:$buffer);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
}

//===----------------------------------------------------------------------===//
// Test ValueBoundsOpInterface
//===----------------------------------------------------------------------===//

def TestValueWithBoundsOp : TEST_Op<"value_with_bounds", [
    DeclareOpInterfaceMethods<ValueBoundsOpInterface, ["populateBoundsForIndexValue"]>
  ]> {
  let description = [{
    Creates a value with specified [min, max] range for value bounds analysis.

    Example:

    ```mlir
    %0 = test.value_with_bounds { min = 4 : index, max = 5 : index}
    ```
  }];
  let arguments = (ins IndexAttr:$min, IndexAttr:$max);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}


def ReifyBoundOp : TEST_Op<"reify_bound", [Pure]> {
  let description = [{
    Reify a bound for the given index-typed value or dimension size of a shaped
    value. "LB", "EQ" and "UB" bounds are supported. If `scalable` is set,
    `vscale_min` and `vscale_max` must be provided, which allows computing
    a bound in terms of "vector.vscale" for a given range of vscale.
  }];

  let arguments = (ins AnyType:$var,
                       OptionalAttr<I64Attr>:$dim,
                       DefaultValuedAttr<StrAttr, "\"EQ\"">:$type,
                       UnitAttr:$constant,
                       UnitAttr:$scalable,
                       OptionalAttr<I64Attr>:$vscale_min,
                       OptionalAttr<I64Attr>:$vscale_max);
  let results = (outs Index:$result);

  let extraClassDeclaration = [{
    ::mlir::presburger::BoundType getBoundType();
    ::mlir::ValueBoundsConstraintSet::Variable getVariable();
  }];

  let hasVerifier = 1;
}

def CompareOp : TEST_Op<"compare"> {
  let description = [{
    Compare `lhs` and `rhs`. A remark is emitted which indicates whether the
    specified comparison operator was proven to hold. The remark also indicates
    whether the opposite comparison operator was proven to hold.

    `var_operands` must have exactly two operands: one for the LHS operand and
    one for the RHS operand. If `lhs_map` is specified, as many operands as
    `lhs_map` has inputs are expected instead of the first operand. If `rhs_map`
    is specified, as many operands as `rhs_map` has inputs are expected instead
    of the second operand.
  }];

  let arguments = (ins Variadic<Index>:$var_operands,
                       DefaultValuedAttr<StrAttr, "\"EQ\"">:$cmp,
                       OptionalAttr<AffineMapAttr>:$lhs_map,
                       OptionalAttr<AffineMapAttr>:$rhs_map,
                       UnitAttr:$compose);
  let results = (outs);

  let extraClassDeclaration = [{
    ::mlir::ValueBoundsConstraintSet::ComparisonOperator
        getComparisonOperator();
    ::mlir::ValueBoundsConstraintSet::Variable getLhs();
    ::mlir::ValueBoundsConstraintSet::Variable getRhs();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Test RegionBranchOpInterface
//===----------------------------------------------------------------------===//

def RegionIfYieldOp : TEST_Op<"region_if_yield",
      [NoMemoryEffect, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $results `:` type($results) attr-dict
  }];
}

def RegionIfOp : TEST_Op<"region_if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                 ["getRegionInvocationBounds",
                                  "getEntrySuccessorOperands"]>,
       SingleBlockImplicitTerminator<"RegionIfYieldOp">,
       RecursiveMemoryEffects]> {
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
  }];
  let hasCustomAssemblyFormat = 1;
}

def AnyCondOp : TEST_Op<"any_cond",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                 ["getRegionInvocationBounds"]>,
       RecursiveMemoryEffects]> {
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
}

def LoopBlockOp : TEST_Op<"loop_block",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands"]>, RecursiveMemoryEffects]> {

  let results = (outs F32:$floatResult);
  let arguments = (ins I32:$init);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $init `:` functional-type($init, $floatResult) $body
    attr-dict-with-keyword
  }];
}

def LoopBlockTerminatorOp : TEST_Op<"loop_block_term",
    [DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>, Pure,
     Terminator]> {
  let arguments = (ins I32:$nextIterArg, F32:$exitArg);

  let assemblyFormat = [{
    `iter` $nextIterArg `exit` $exitArg attr-dict
  }];
}

def TestNoTerminatorOp : TEST_Op<"switch_with_no_break", [
    NoTerminator,
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorRegions"]>
  ]> {
  let arguments = (ins Index:$arg, DenseI64ArrayAttr:$cases);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg attr-dict custom<SwitchCases>($cases, $caseRegions)
  }];
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
    static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *,
          ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({operands[0].getType()});
      return ::mlir::success();
    }
   }];
}

// Op with InferTypeOpInterface and regions.
def TableGenBuildOp5 : TableGenBuildInferReturnTypeBaseOp<
    "tblgen_build_5", [InferTypeOpInterface]> {
  let regions = (region AnyRegion:$body);
}

// Two variadic args, non variadic results, with AttrSizedOperandSegments
// Test build method generation for property conversion & type inference.
def TableGenBuildOp6 : TEST_Op<"tblgen_build_6", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$a, Variadic<AnyType>:$b);
  let results = (outs F32:$result);
}

// An inherent attribute. Test collective builders, both those that take properties as
// properties structs and those that take an attribute dictionary.
def TableGenBuildOp7 : TEST_Op<"tblgen_build_7", []> {
  let arguments = (ins BoolAttr:$attr0);
  let results = (outs);
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
                                 [IsolatedFromAbove, HasDefaultDLTIDataLayout, DataLayoutOpInterface]> {
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
// Test DestinationStyleOpInterface.
//===----------------------------------------------------------------------===//

def TestDestinationStyleOp :
    TEST_Op<"destination_style_op", [
      DestinationStyleOpInterface,
      AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    Variadic<AnyType>:$other_operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^  `:` type($outputs) `)`)?
    (`(` $other_operands^ `:` type($other_operands) `)`)?
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

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
      DestinationStyleOpInterface, LinalgStructuredInterface,
      LinalgConvolutionOpInterface]> {

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
                              mlir::ArrayRef<mlir::NamedAttribute> attrs,
                              llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
      mlir::linalg::YieldOp::create(b,block.getArguments().back());
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              llvm::function_ref<mlir::InFlightDiagnostic()>)>
    getRegionBuilder() {
      return &regionBuilder;
    }

    llvm::SmallVector<mlir::utils::IteratorType> getIteratorTypesArray() {
      auto attrs = getOperation()->getAttrOfType<mlir::ArrayAttr>("iterator_types");
      auto range = attrs.getAsValueRange<IteratorTypeAttr, mlir::utils::IteratorType>();
      return {range.begin(), range.end()};
    }

    mlir::ArrayAttr getIndexingMaps() {
      return getOperation()->getAttrOfType<mlir::ArrayAttr>("indexing_maps");
    }

    std::string getLibraryCallName() {
      return "";
    }

    mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
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
      DestinationStyleOpInterface, LinalgStructuredInterface,
      LinalgFillOpInterface]> {

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
                              mlir::ArrayRef<mlir::NamedAttribute> attrs,
                              llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
      mlir::linalg::YieldOp::create(b,block.getArguments().back());
    }

    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              llvm::function_ref<mlir::InFlightDiagnostic()>)>
    getRegionBuilder() {
      return &regionBuilder;
    }

    llvm::SmallVector<mlir::utils::IteratorType> getIteratorTypesArray() {
      auto attrs = getOperation()->getAttrOfType<mlir::ArrayAttr>("iterator_types");
      auto range = attrs.getAsValueRange<IteratorTypeAttr, mlir::utils::IteratorType>();
      return {range.begin(), range.end()};
    }

    mlir::ArrayAttr getIndexingMaps() {
      return getOperation()->getAttrOfType<mlir::ArrayAttr>("indexing_maps");
    }

    std::string getLibraryCallName() {
      return "";
    }

    mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Test NVVM RequiresSM trait.
//===----------------------------------------------------------------------===//

def TestNVVMRequiresSMOp :
    TEST_Op<"nvvm_requires_sm_80", [NVVMRequiresSM<80>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
}

def TestNVVMRequiresSMArchCondOp :
    TEST_Op<"nvvm_requires_sm_90a", [NVVMRequiresSMa<[90]>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
}

def TestNVVMRequirestSMArchCondMultiOp :
    TEST_Op<"nvvm_requires_sm_90a_or_sm_100a", [NVVMRequiresSMa<[90, 100]>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
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
// Test Ops with variadics
//===----------------------------------------------------------------------===//

def TestVariadicRewriteSrcOp : TEST_Op<"variadic_rewrite_src_op", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<AnyType>:$arg,
    AnyType:$brg,
    Variadic<AnyType>:$crg
  );
}

def TestVariadicRewriteDstOp : TEST_Op<"variadic_rewrite_dst_op", [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyType:$brg,
    Variadic<AnyType>:$crg,
    Variadic<AnyType>:$arg
  );
}

def : Pat<(TestVariadicRewriteSrcOp $arg, $brg, $crg),
          (TestVariadicRewriteDstOp $brg, $crg, $arg)>;

//===----------------------------------------------------------------------===//
// Test Ops with Default-Valued Attributes and Differing Print Settings
//===----------------------------------------------------------------------===//

def TestDefaultAttrPrintOp : TEST_Op<"default_value_print"> {
  let arguments = (ins DefaultValuedAttr<I32Attr, "0">:$value_with_default,
                   I32:$operand);
  let assemblyFormat = "attr-dict $operand";
}

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

  let results = (outs Res<AnyMemRef, "", [MemAlloc<TestResource, 0>]>);
}

def TestEffectsOpB : TEST_Op<"op_with_effects_b",
    [MemoryEffects<[MemWrite<TestResource, 0>]>]>;

def TestEffectsRead : TEST_Op<"op_with_memread",
    [MemoryEffects<[MemRead]>]> {
  let results = (outs AnyInteger);
}

def TestEffectsWrite : TEST_Op<"op_with_memwrite",
    [MemoryEffects<[MemWrite]>]>;

def TestEffectsResult : TEST_Op<"test_effects_result"> {
  let results = (outs Res<I32, "", [MemAlloc, MemWrite]>);
}

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

//===----------------------------------------------------------------------===//
// Test Loop Op with a graph region
//===----------------------------------------------------------------------===//

// Test loop op with a graph region.
def TestGraphLoopOp : TEST_Op<"graph_loop",
                         [LoopLikeOpInterface, NoMemoryEffect,
                          RecursivelySpeculatable, SingleBlock,
                          RegionKindInterface, HasOnlyGraphRegion]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$rets);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $args $body attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    llvm::SmallVector<mlir::Region *> getLoopRegions() { return {&getBody()}; }
  }];
}

//===----------------------------------------------------------------------===//
// Test InferIntRangeInterface
//===----------------------------------------------------------------------===//
def InferIntRangeType : AnyTypeOf<[AnyInteger, Index, VectorOfNonZeroRankOf<[AnyInteger, Index]>]>;

def TestWithBoundsOp : TEST_Op<"with_bounds",
                          [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
                           NoMemoryEffect]> {
  let description = [{
    Creates a value with specified [min, max] range for integer range analysis.

    Example:

    ```mlir
    %0 = test.with_bounds { umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index } : index
    ```
  }];

  let arguments = (ins APIntAttr:$umin,
                       APIntAttr:$umax,
                       APIntAttr:$smin,
                       APIntAttr:$smax);
  let results = (outs InferIntRangeType:$fakeVal);

  let assemblyFormat = "attr-dict `:` type($fakeVal)";
}

def TestWithBoundsRegionOp : TEST_Op<"with_bounds_region",
                          [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
                           SingleBlock, NoTerminator]> {
  let arguments = (ins APIntAttr:$umin,
                       APIntAttr:$umax,
                       APIntAttr:$smin,
                       APIntAttr:$smax);
  // The region has one argument of any integer type
  let regions = (region SizedRegion<1>:$region);
  let hasCustomAssemblyFormat = 1;
}

def TestIncrementOp : TEST_Op<"increment",
                         [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
                         NoMemoryEffect, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins InferIntRangeType:$value);
  let results = (outs InferIntRangeType:$result);

  let assemblyFormat = "attr-dict $value `:` type($result)";
}

def TestReflectBoundsOp : TEST_Op<"reflect_bounds",
                         [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
                          AllTypesMatch<["value", "result"]>]> {
  let description = [{
    Integer range analysis will update this op to reflect inferred integer range
    of the input, so it can be checked with FileCheck

    Example:

    ```mlir
    CHECK: test.reflect_bounds {smax = 7 : index, smin = 0 : index, umax = 7 : index, umin = 0 : index}
    %1 = test.reflect_bounds %0 : index
    ```
  }];

  let arguments = (ins InferIntRangeType:$value,
                       OptionalAttr<APIntAttr>:$umin,
                       OptionalAttr<APIntAttr>:$umax,
                       OptionalAttr<APIntAttr>:$smin,
                       OptionalAttr<APIntAttr>:$smax);
  let results = (outs InferIntRangeType:$result);

  let assemblyFormat = "attr-dict $value `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Test ConditionallySpeculatable
//===----------------------------------------------------------------------===//

def ConditionallySpeculatableOp : TEST_Op<"conditionally_speculatable_op",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let description = [{
    Op used to test conditional speculation.  This op can be speculatively
    executed if the input to it is an `arith.constant`.
  }];

  let arguments = (ins I32:$input);
  let results = (outs I32:$result);

  let extraClassDeclaration = [{
    ::mlir::Speculation::Speculatability getSpeculatability();
  }];

  let extraClassDefinition = [{
    ::mlir::Speculation::Speculatability
    ConditionallySpeculatableOp::getSpeculatability() {
      Operation* definingOp = getInput().getDefiningOp();
      return definingOp && isa<::mlir::arith::ConstantOp>(definingOp) ?
          ::mlir::Speculation::Speculatable : ::mlir::Speculation::NotSpeculatable;
    }
  }];
}

def PureOp : TEST_Op<"always_speculatable_op", [Pure]> {
  let description = [{
    Op used to test conditional speculation.  This op can always be
    speculatively executed.
  }];
  let results = (outs I32:$result);
}

def NeverSpeculatableOp : TEST_Op<"never_speculatable_op", [ConditionallySpeculatable]> {
  let description = [{
    Op used to test conditional speculation.  This op can never be
    speculatively executed.
  }];
  let results = (outs I32:$result);

  let extraClassDeclaration = [{
    ::mlir::Speculation::Speculatability getSpeculatability() {
      return ::mlir::Speculation::NotSpeculatable;
    }
  }];
}

def RecursivelySpeculatableOp : TEST_Op<"recursively_speculatable_op", [
    RecursivelySpeculatable, RecursiveMemoryEffects]> {
  let description = [{
    Op used to test conditional speculation.  This op can be speculatively
    executed only if all the ops in the attached region can be.
  }];
  let results = (outs I32:$result);
  let regions = (region SizedRegion<1>:$body);
}

//===---------------------------------------------------------------------===//
// Test CSE
//===---------------------------------------------------------------------===//

def TestCSEOfSingleBlockOp : TEST_Op<"cse_of_single_block_op",
    [SingleBlockImplicitTerminator<"RegionYieldOp">, Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    attr-dict `inputs` `(` $inputs `)`
    $region `:` type($inputs)  `->` type($outputs)
  }];
}

//===----------------------------------------------------------------------===//
// Test Ops to upgrade base on the dialect versions
//===----------------------------------------------------------------------===//

def TestVersionedOpA : TEST_Op<"versionedA"> {
  // A previous version of the dialect (let's say 1.*) supported an attribute
  // named "dimensions":
  // let arguments = (ins
  //   AnyI64Attr:$dimensions
  // );

  // In the current version (2.0) "dimensions" was renamed to "dims", and a new
  // boolean attribute "modifier" was added. The previous version of the op
  // corresponds to "modifier=false". We support loading old IR through
  // upgrading, see `upgradeFromVersion()` in `TestBytecodeDialectInterface`.
  let arguments = (ins
   AnyI64Attr:$dims,
   BoolAttr:$modifier
  );

  // Since we use properties to store attributes, we need a custom encoding
  // reader/writer to handle versioning.
  let useCustomPropertiesEncoding = 1;
}

def TestVersionedOpB : TEST_Op<"versionedB"> {
  // A previous version of the dialect (let's say 1.*) we encoded TestAttrParams
  // with a custom encoding:
  //
  //    #test.attr_params<X, Y> -> { varInt: Y, varInt: X }
  //
  // In the current version (2.0) the encoding changed and the two parameters of
  // the attribute are swapped:
  //
  //    #test.attr_params<X, Y> -> { varInt: X, varInt: Y }
  //
  // We support loading old IR through a custom readAttribute method, see
  // `readAttribute()` in `TestBytecodeDialectInterface`
  let arguments = (ins
    TestAttrParams:$attribute
  );
}

def TestVersionedOpC : TEST_Op<"versionedC"> {
  let arguments = (ins AnyAttrOf<[TestAttrParams,
                                  I32ElementsAttr]>:$attribute
  );
}

//===----------------------------------------------------------------------===//
// Test Properties
//===----------------------------------------------------------------------===//


// Op with a properties struct defined inline.
def TestOpWithProperties : TEST_Op<"with_properties"> {
  let assemblyFormat = [{
    `a` `=` $a `,`
    `b` `=` $b `,`
    `c` `=` $c `,`
    `flag` `=` $flag `,`
    `array` `=` $array `,`
    `array32` `=` $array32 attr-dict}];
  let arguments = (ins
    I64Prop:$a,
    StrAttr:$b, // Attributes can directly be used here.
    StringProp:$c,
    BoolProp:$flag,
    IntArrayProp<I64Prop>:$array, // Example of an array.
    IntArrayProp<I32Prop>:$array32 // Example of an array.
  );
}

def TestOpWithPropertiesAndAttr
  : TEST_Op<"with_properties_and_attr"> {
  let assemblyFormat = "$lhs prop-dict attr-dict";

  let arguments = (ins I32Attr:$lhs, IntProp<"int64_t">:$rhs);
}

def TestOpWithPropertiesAndInferredType
  : TEST_Op<"with_properties_and_inferred_type", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let assemblyFormat = "$lhs prop-dict attr-dict";

  let arguments = (ins I32Attr:$lhs, IntProp<"int64_t">:$rhs, OptionalAttr<UnitAttr>: $packed);
  let results = (outs AnyType:$result);
}

// Demonstrate how to wrap an existing C++ class named MyPropStruct.
def MyStructProperty : Property<"MyPropStruct"> {
  let convertToAttribute = "return $_storage.asAttribute($_ctxt);";
  let convertFromAttribute = "return MyPropStruct::setFromAttr($_storage, $_attr, $_diag);";
  let hashProperty = "$_storage.hash();";
}

def TestOpWithWrappedProperties : TEST_Op<"with_wrapped_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    MyStructProperty:$prop
  );
}

// Same as above, but without a custom `hashProperty` field, checking
// that ADL is correctly working.
def MyStructProperty2 : Property<"MyPropStruct"> {
  let convertToAttribute = "return $_storage.asAttribute($_ctxt);";
  let convertFromAttribute = "return MyPropStruct::setFromAttr($_storage, $_attr, $_diag);";
}

def TestOpWithWrappedProperties2 : TEST_Op<"with_wrapped_properties2"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
    MyStructProperty2:$prop
  );
}

def TestOpWithEmptyProperties : TEST_Op<"empty_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins);
}

def TestOpUsingPropertyInCustom : TEST_Op<"using_property_in_custom"> {
  let assemblyFormat = "custom<UsingPropertyInCustom>($prop) attr-dict";
  let arguments = (ins IntArrayProp<I64Prop>:$prop);
}

def TestOpUsingPropertyInCustomAndOther
  : TEST_Op<"using_property_in_custom_and_other"> {
  let assemblyFormat = "custom<UsingPropertyInCustom>($prop) prop-dict attr-dict";
  let arguments = (ins
    IntArrayProp<I64Prop>:$prop,
    I64Prop:$other
  );
}

def TestOpWithVariadicSegmentProperties : TEST_Op<"variadic_segment_prop",
    [AttrSizedOperandSegments, AttrSizedResultSegments]> {
  let arguments = (ins Variadic<I64>:$a1, Variadic<I64>:$a2);
  let results = (outs Variadic<I64>:$b1, Variadic<I64>:$b2);
  let assemblyFormat = [{
    $a1 `:` $a2 `:` type($b1) `:` type($b2) prop-dict attr-dict `end`
  }];
}

def TestOpUsingPropertyRefInCustom : TEST_Op<"using_property_ref_in_custom"> {
  let assemblyFormat = "custom<IntProperty>($first) `+` custom<SumProperty>($second, ref($first)) attr-dict";
  let arguments = (ins IntProp<"int64_t">:$first, IntProp<"int64_t">:$second);
}

def IntPropertyWithWorseBytecode : Property<"int64_t"> {
  let writeToMlirBytecode = writeMlirBytecodeWithConvertToAttribute;

  let readFromMlirBytecode = readMlirBytecodeUsingConvertFromAttribute;
}

def TestOpUsingIntPropertyWithWorseBytecode
    : TEST_Op<"using_int_property_with_worse_bytecode"> {
  let arguments = (ins IntPropertyWithWorseBytecode:$value);
}

// Op with a properties struct defined out-of-line. The struct has custom
// printer/parser.

def PropertiesWithCustomPrint : Property<"PropertiesWithCustomPrint"> {
  let convertToAttribute = [{
    return getPropertiesAsAttribute($_ctxt, $_storage);
  }];
  let convertFromAttribute = [{
    return setPropertiesFromAttribute($_storage, $_attr, $_diag);
  }];
  let hashProperty = [{
    computeHash($_storage);
  }];
}

def TestOpWithNiceProperties : TEST_Op<"with_nice_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
     PropertiesWithCustomPrint:$prop
  );
  let extraClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *ctx, ::mlir::OpAsmPrinter &p,
                         const Properties &prop,
                         ::mlir::ArrayRef<::llvm::StringRef> elidedProps);
    static ::mlir::ParseResult  parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
    static ::llvm::LogicalResult readFromMlirBytecode(
        ::mlir::DialectBytecodeReader &,
        test::PropertiesWithCustomPrint &prop);
    static void writeToMlirBytecode(
        ::mlir::DialectBytecodeWriter &,
        const test::PropertiesWithCustomPrint &prop);
  }];
  let extraClassDefinition = [{
    ::llvm::LogicalResult TestOpWithNiceProperties::readFromMlirBytecode(
        ::mlir::DialectBytecodeReader &reader,
        test::PropertiesWithCustomPrint &prop) {
      StringRef label;
      uint64_t value;
      if (failed(reader.readString(label)) || failed(reader.readVarInt(value)))
        return failure();
      prop.label = std::make_shared<std::string>(label.str());
      prop.value = value;
      return success();
    }
    void TestOpWithNiceProperties::writeToMlirBytecode(
        ::mlir::DialectBytecodeWriter &writer,
        const test::PropertiesWithCustomPrint &prop) {
      writer.writeOwnedString(*prop.label);
      writer.writeVarInt(prop.value);
    }
    void TestOpWithNiceProperties::printProperties(::mlir::MLIRContext *ctx,
            ::mlir::OpAsmPrinter &p, const Properties &prop,
            ::mlir::ArrayRef<::llvm::StringRef> elidedProps) {
      customPrintProperties(p, prop.prop);
    }
    ::mlir::ParseResult TestOpWithNiceProperties::parseProperties(
        ::mlir::OpAsmParser &parser,
        ::mlir::OperationState &result) {
      Properties &prop = result.getOrAddProperties<Properties>();
      if (customParseProperties(parser, prop.prop))
        return failure();
      return success();
    }
  }];
}

def VersionedProperties : Property<"VersionedProperties"> {
  let convertToAttribute = [{
    return getPropertiesAsAttribute($_ctxt, $_storage);
  }];
  let convertFromAttribute = [{
    return setPropertiesFromAttribute($_storage, $_attr, $_diag);
  }];
  let hashProperty = [{
    computeHash($_storage);
  }];
}

def TestOpWithVersionedProperties : TEST_Op<"with_versioned_properties"> {
  let assemblyFormat = "prop-dict attr-dict";
  let arguments = (ins
     VersionedProperties:$prop
  );
  let extraClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *ctx, ::mlir::OpAsmPrinter &p,
                         const Properties &prop,
                         ::mlir::ArrayRef<::llvm::StringRef> elidedProps);
    static ::mlir::ParseResult  parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
    static ::llvm::LogicalResult readFromMlirBytecode(
        ::mlir::DialectBytecodeReader &,
        test::VersionedProperties &prop);
    static void writeToMlirBytecode(
        ::mlir::DialectBytecodeWriter &,
        const test::VersionedProperties &prop);
  }];
  let extraClassDefinition = [{
    void TestOpWithVersionedProperties::printProperties(::mlir::MLIRContext *ctx,
            ::mlir::OpAsmPrinter &p, const Properties &prop,
            ::mlir::ArrayRef<::llvm::StringRef> elidedProps) {
      customPrintProperties(p, prop.prop);
    }
    ::mlir::ParseResult TestOpWithVersionedProperties::parseProperties(
        ::mlir::OpAsmParser &parser,
        ::mlir::OperationState &result) {
      Properties &prop = result.getOrAddProperties<Properties>();
      if (customParseProperties(parser, prop.prop))
        return failure();
      return success();
    }
  }];
}

def TestOpWithDefaultValuedProperties : TEST_Op<"with_default_valued_properties"> {
  let assemblyFormat = [{
    ($a^) : (`na`)?
    ($b^)?
    ($c^)?
    ($unit^)?
    attr-dict
  }];
  let arguments = (ins DefaultValuedAttr<I32Attr, "0">:$a,
    DefaultValuedProp<StringProp, "\"\"">:$b,
    DefaultValuedProp<IntProp<"int32_t">, "-1">:$c,
    UnitProp:$unit);
}

def TestOpWithOptionalProperties : TEST_Op<"with_optional_properties"> {
  let assemblyFormat = [{
    (`anAttr` `=` $anAttr^)?
    (`simple` `=` $simple^)?
    (`simplei8` `=` $simplei8^)?
    (`simpleui8` `=` $simpleui8^)?
    (`nonTrivialStorage` `=` $nonTrivialStorage^)?
    (`hasDefault` `=` $hasDefault^)?
    (`nested` `=` $nested^)?
    (`longSyntax` `=` $longSyntax^)?
    (`hasUnit` $hasUnit^)?
    (`maybeUnit` `=` $maybeUnit^)?
    attr-dict
  }];
  let arguments = (ins
    OptionalAttr<I32Attr>:$anAttr,
    OptionalProp<I64Prop>:$simple,
    OptionalProp<IntProp<"int8_t">>:$simplei8,
    OptionalProp<IntProp<"uint8_t">>:$simpleui8,
    OptionalProp<StringProp>:$nonTrivialStorage,
    // Confirm that properties with default values now default to nullopt and have
    // the long syntax.
    OptionalProp<DefaultValuedProp<I64Prop, "0">>:$hasDefault,
    OptionalProp<OptionalProp<I64Prop>>:$nested,
    OptionalProp<StringProp, 0>:$longSyntax,
    UnitProp:$hasUnit,
    OptionalProp<UnitProp>:$maybeUnit);
}

def TestOpWithArrayProperties : TEST_Op<"with_array_properties"> {
  let assemblyFormat = [{
    `ints` `=` $ints
    `strings` `=` $strings
    `nested` `=` $nested
    `opt` `=` $opt
    `explicitOptions` `=` $explicitOptions
    `explicitUnits` `=` $explicitUnits
    ($hasDefault^ `thats_has_default`)?
    attr-dict
  }];
  let arguments = (ins
    ArrayProp<I64Prop>:$ints,
    ArrayProp<StringProp>:$strings,
    ArrayProp<ArrayProp<I32Prop>>:$nested,
    OptionalProp<ArrayProp<I32Prop>>:$opt,
    ArrayProp<OptionalProp<I64Prop>>:$explicitOptions,
    ArrayProp<UnitProp>:$explicitUnits,
    DefaultValuedProp<ArrayProp<I64Prop>,
      "::llvm::ArrayRef<int64_t>{}", "::llvm::SmallVector<int64_t>{}">:$hasDefault
  );
}

def NonNegativeI64Prop : ConfinedProp<I64Prop,
  CPred<"$_self >= 0">, "non-negative int64_t">;

class NonEmptyArray<Property p> : ConfinedProp
    <ArrayProp<p>, Neg<CPred<"$_self.empty()">>,
    "non-empty array of " # p.summary>;

def OpWithPropertyPredicates : TEST_Op<"op_with_property_predicates"> {
  let arguments = (ins
    NonNegativeI64Prop:$scalar,
    OptionalProp<NonNegativeI64Prop>:$optional,
    DefaultValuedProp<NonNegativeI64Prop, "0">:$defaulted,
    ConfinedProp<NonNegativeI64Prop,
      CPred<"$_self <= 5">, "between 0 and 5">:$more_constrained,
    ArrayProp<NonNegativeI64Prop>:$array,
    NonEmptyArray<I64Prop>:$non_empty_unconstrained,
    NonEmptyArray<NonNegativeI64Prop>:$non_empty_constrained,
    // Test applying predicates when the fromStorage() on the optional<> isn't trivial.
    OptionalProp<NonEmptyArray<NonNegativeI64Prop>>:$non_empty_optional,
    I64Prop:$unconstrained
  );
  let assemblyFormat = "attr-dict prop-dict";
}

def TestPropPatternOp1 : TEST_Op<"prop_pattern_op_1"> {
  let arguments = (ins
    StringProp:$tag,
    I64Prop:$val,
    BoolProp:$cond
  );
  let results = (outs I32:$results);
  let assemblyFormat = "$tag $val $cond attr-dict";
}

def TestPropPatternOp2 : TEST_Op<"prop_pattern_op_2"> {
  let arguments = (ins
    I32:$input,
    StringProp:$tag
  );
  let assemblyFormat = "$input $tag attr-dict";
}

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

//===----------------------------------------------------------------------===//
// Test Dataflow
//===----------------------------------------------------------------------===//

def TestCallAndStoreOp : TEST_Op<"call_and_store",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let arguments = (ins
    SymbolRefAttr:$callee,
    Arg<AnyMemRef, "", [MemWrite]>:$address,
    Variadic<AnyType>:$callee_operands,
    BoolAttr:$store_before_call,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs
    Variadic<AnyType>:$results
  );
  let assemblyFormat =
    "$callee `(` $callee_operands `)` `,` $address attr-dict "
    "`:` functional-type(operands, results)";
}

def TestCallOnDeviceOp : TEST_Op<"call_on_device",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$forwarded_operands,
    AnyType:$non_forwarded_device_operand,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs
    Variadic<AnyType>:$results
  );
  let assemblyFormat =
    "$callee `(` $forwarded_operands `)` `,` $non_forwarded_device_operand "
    "attr-dict `:` functional-type(operands, results)";
}

def TestStoreWithARegion : TEST_Op<"store_with_a_region",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlock]> {
  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite]>:$address,
    BoolAttr:$store_before_region
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
    "$address attr-dict-with-keyword regions `:` type($address)";
}

def TestStoreWithALoopRegion : TEST_Op<"store_with_a_loop_region",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlock]> {
  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite]>:$address,
    BoolAttr:$store_before_region
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
    "$address attr-dict-with-keyword regions `:` type($address)";
}

def TestStoreWithARegionTerminator : TEST_Op<"store_with_a_region_terminator",
    [ReturnLike, Terminator, NoMemoryEffect]> {
  let assemblyFormat = "attr-dict";
}

def TestOpOptionallyImplementingInterface
    : TEST_Op<"op_optionally_implementing_interface",
        [TestOptionallyImplementedOpInterface]> {
  let arguments = (ins BoolAttr:$implementsInterface);
}

//===----------------------------------------------------------------------===//
// Test Mem2Reg & SROA
//===----------------------------------------------------------------------===//

def TestMultiSlotAlloca : TEST_Op<"multi_slot_alloca",
    [DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
     DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>]> {
  let results = (outs Variadic<MemRefOf<[I32]>>:$results);
  let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Test allocation Ops
//===----------------------------------------------------------------------===//

def TestAllocWithMultipleResults : TEST_Op<"alloc_with_multiple_results"> {
  let results = (outs Index:$index,
                      Res<AnyMemRef, "", [MemAlloc]>:$memref);
  let assemblyFormat = [{
     attr-dict `:` type($index) `,` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// Test Ops bufferization
//===----------------------------------------------------------------------===//

def TestDummyTensorOp : TEST_Op<"dummy_tensor_op",
    [DeclareOpInterfaceMethods<BufferizableOpInterface,
        ["bufferize", "bufferizesToMemoryRead",
         "bufferizesToMemoryWrite", "getAliasingValues"]>]> {
  let arguments = (ins
    Arg<TestTensorType>:$input
  );
  let results = (outs
    Arg<TestTensorType>:$output
  );

  let extraClassDefinition = [{
    bool test::TestDummyTensorOp::bufferizesToMemoryRead(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    bool test::TestDummyTensorOp::bufferizesToMemoryWrite(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    ::mlir::bufferization::AliasingValueList
    test::TestDummyTensorOp::getAliasingValues(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return {};
    }
  }];
}

def TestDummyMemrefOp : TEST_Op<"dummy_memref_op", []> {
  let arguments = (ins
    Arg<TestMemrefType>:$input
  );
  let results = (outs
    Arg<TestMemrefType>:$output
  );
}

def TestCreateTensorOp : TEST_Op<"create_tensor_op",
    [DeclareOpInterfaceMethods<BufferizableOpInterface,
        ["bufferize", "getBufferType", "bufferizesToMemoryRead",
         "bufferizesToMemoryWrite", "getAliasingValues",
         "bufferizesToAllocation"]>]> {
  let arguments = (ins);
  let results = (outs Arg<TestTensorType>:$output);
  let extraClassDefinition = [{
    bool test::TestCreateTensorOp::bufferizesToMemoryRead(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    bool test::TestCreateTensorOp::bufferizesToMemoryWrite(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    bool test::TestCreateTensorOp::bufferizesToAllocation(mlir::Value value) {
      return false;
    }

    ::mlir::bufferization::AliasingValueList
    test::TestCreateTensorOp::getAliasingValues(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return {};
    }
  }];
}

def TestCreateMemrefOp : TEST_Op<"create_memref_op"> {
  let arguments = (ins);
  let results = (outs Arg<TestMemrefType>:$output);
}

//===----------------------------------------------------------------------===//
// Test assembly format references
//===----------------------------------------------------------------------===//

def TestOpWithRegionRef : TEST_Op<"dummy_op_with_region_ref", [NoTerminator]> {
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $body attr-dict custom<DummyRegionRef>(ref($body))
  }];
}

def TestOpWithSuccessorRef : TEST_Op<"dummy_op_with_successor_ref"> {
  let successors = (successor AnySuccessor:$successor);
  let assemblyFormat = [{
    $successor attr-dict custom<DummySuccessorRef>(ref($successor))
  }];
}

def CallWithSegmentsOp : TEST_Op<"call_with_segments",
    [AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "test call op with segmented args";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$prefix,   // non-arg segment (e.g., 'in')
    Variadic<AnyType>:$args,     // <-- the call *arguments* segment
    Variadic<AnyType>:$suffix    // non-arg segment (e.g., 'out')
  );
  let results = (outs);
  let assemblyFormat = [{
    $callee `(` $prefix `:` type($prefix) `)`
            `(` $args `:` type($args) `)`
            `(` $suffix `:` type($suffix) `)` attr-dict
  }];

  // Provide stub implementations for the ArgAndResultAttrsOpInterface.
  let extraClassDeclaration = [{
    ::mlir::ArrayAttr getArgAttrsAttr() { return {}; }
    ::mlir::ArrayAttr getResAttrsAttr() { return {}; }
    void setArgAttrsAttr(::mlir::ArrayAttr) {}
    void setResAttrsAttr(::mlir::ArrayAttr) {}
    ::mlir::Attribute removeArgAttrsAttr() { return {}; }
    ::mlir::Attribute removeResAttrsAttr() { return {}; }
  }];

  let extraClassDefinition = [{
    ::mlir::CallInterfaceCallable $cppClass::getCallableForCallee() {
      if (auto sym = (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee"))
        return ::mlir::CallInterfaceCallable(sym);
      return ::mlir::CallInterfaceCallable();
    }
    void $cppClass::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      if (auto sym = callee.dyn_cast<::mlir::SymbolRefAttr>())
        (*this)->setAttr("callee", sym);
      else
        (*this)->removeAttr("callee");
    }
  }];
}


#endif // TEST_OPS
