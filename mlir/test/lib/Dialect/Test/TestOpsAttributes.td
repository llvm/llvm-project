//===- TestOpsAttributes.td - Test dialect attributes ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_ATTRIBUTES
#define TEST_OPS_ATTRIBUTES

include "TestOpBase.td"

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

// Define an op that contains the enum attribute.
def OpWithEnum : TEST_Op<"op_with_enum"> {
  let arguments = (ins TestEnumAttr:$value, OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "enum($value) (`tag` $tag^)? attr-dict";
}

// Define an op that prints just the symbolic value of an enum attribute whose
// own assembly format includes angle brackets.
def OpWithEnumDirective : TEST_Op<"op_with_enum_directive"> {
  let arguments = (ins TestPrettyEnumAttr:$value);
  let assemblyFormat = "enum($value) attr-dict";
}

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

def OpWithTestBitEnumPropDict
    : TEST_Op<"op_with_bit_enum_prop_dict"> {
  let arguments = (ins TestBitEnumProp:$flags, I64Prop:$next);
  let assemblyFormat = "prop-dict attr-dict";
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

def TestSimpleEnum8Prop : EnumProp<TestSimpleEnum8>;

def OpWithSimpleEnum8Prop : TEST_Op<"op_with_simple_enum_8_prop"> {
  let arguments = (ins TestSimpleEnum8Prop:$value);
  let assemblyFormat = "$value attr-dict";
}

def TestSimpleEnum16Prop : EnumProp<TestSimpleEnum16>;

def OpWithSimpleEnum16Prop : TEST_Op<"op_with_simple_enum_16_prop"> {
  let arguments = (ins TestSimpleEnum16Prop:$value);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// Test Bit Enum Attributes
//===----------------------------------------------------------------------===//

// Define an op that contains the enum attribute.
def OpWithBitEnum : TEST_Op<"op_with_bit_enum"> {
  let arguments = (ins TestBitEnumAttr:$value, OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

// Define an op that contains the enum attribute.
def OpWithBitEnumVerticalBar : TEST_Op<"op_with_bit_enum_vbar"> {
  let arguments = (ins TestBitEnumVerticalBarAttr:$value,
                   OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

def OpWithBitEnumDirective : TEST_Op<"op_with_bit_enum_directive"> {
  let arguments = (ins TestBitEnumAttr:$value);
  let assemblyFormat = "enum($value) attr-dict";
}

def OpWithBitEnumVerticalBarDirective
    : TEST_Op<"op_with_bit_enum_vbar_directive"> {
  let arguments = (ins TestBitEnumVerticalBarAttr:$value);
  let assemblyFormat = "enum($value) attr-dict";
}

def OpWithBitEnumSpace : TEST_Op<"op_with_bit_enum_space"> {
  let arguments = (ins TestBitEnumSpaceAttr:$value,
                   OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "$value (`tag` $tag^)? attr-dict";
}

def OpWithBitEnumSpaceDirective : TEST_Op<"op_with_bit_enum_space_directive"> {
  let arguments = (ins TestBitEnumSpaceAttr:$value,
                   OptionalAttr<AnyAttr>:$tag);
  let assemblyFormat = "enum($value) (`tag` $tag^)? attr-dict";
}

#endif // TEST_OPS_ATTRIBUTES
