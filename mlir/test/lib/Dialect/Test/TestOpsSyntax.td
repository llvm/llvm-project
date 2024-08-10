
//===-- TestOpsSyntax.td - Operations for testing syntax ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_SYNTAX
#define TEST_OPS_SYNTAX

include "TestAttrDefs.td"
include "TestDialect.td"
include "TestTypeDefs.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

class TEST_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

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

def TestAttrWithLoc : TEST_Op<"attr_with_loc"> {
  let summary = "op's attribute has a location";
  let arguments = (ins AnyAttr:$loc, AnyAttr:$value);
  let assemblyFormat = "`(` $value `` custom<OptionalLoc>($loc) `)` attr-dict";
}

// -----

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

// Ops related to OIList primitive
def OIListTrivialProperties : TEST_Op<"oilist_with_keywords_only_properties"> {
  let arguments = (ins UnitProperty:$keyword, UnitProperty:$otherKeyword,
                       UnitProperty:$diffNameUnitPropertyKeyword);
  let assemblyFormat = [{
    oilist( `keyword` $keyword
          | `otherKeyword` $otherKeyword
          | `thirdKeyword` $diffNameUnitPropertyKeyword) attr-dict
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

def TestEllipsisOp : TEST_Op<"ellipsis"> {
  let arguments = (ins Variadic<AnyType>:$operands, UnitAttr:$variadic);
  let assemblyFormat = [{
    `(` $operands (`...` $variadic^)? `)` attr-dict `:` type($operands) `...`
  }];
}

def ElseAnchorOp : TEST_Op<"else_anchor"> {
  let arguments = (ins Optional<AnyType>:$a);
  let assemblyFormat = "`(` (`?`) : (`` $a^ `:` type($a))? `)` attr-dict";
}

// This is used to test that the default dialect is not elided when printing an
// op with dots in the name to avoid parsing ambiguity.
def OpWithDotInNameOp : TEST_Op<"op.with_dot_in_name"> {
  let assemblyFormat = "attr-dict";
}

// --------------

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

// Test that we format optional symbol reference attributes properly.
def FormatOptSymbolRefAttrOp : TEST_Op<"format_opt_symbol_ref_attr_op"> {
  let arguments = (ins OptionalAttr<SymbolRefAttr>:$opt_attr);
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
    DenseI32ArrayAttr:$operand_segments
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
  let assemblyFormat = "operands attr-dict `:` type($result)";
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

def FormatOptionalUnitProperty : TEST_Op<"format_optional_unit_property"> {
  let arguments = (ins UnitProperty:$is_optional);
  let assemblyFormat = "(`is_optional` $is_optional^)? attr-dict";
}

def FormatOptionalUnitPropertyNoElide
    : TEST_Op<"format_optional_unit_property_no_elide"> {
  let arguments = (ins UnitProperty:$is_optional);
  let assemblyFormat = "($is_optional^)? attr-dict";
}

def FormatOptionalEnumAttr : TEST_Op<"format_optional_enum_attr"> {
  let arguments = (ins OptionalAttr<SomeI64Enum>:$attr);
  let assemblyFormat = "($attr^)? attr-dict";
}

def FormatOptionalDefaultAttrs : TEST_Op<"format_optional_default_attrs"> {
  let arguments = (ins DefaultValuedStrAttr<StrAttr, "default">:$str,
                       DefaultValuedStrAttr<SymbolNameAttr, "default">:$sym,
                       DefaultValuedAttr<SomeI64Enum, "SomeI64Enum::case5">:$e);
  let assemblyFormat = "($str^)? ($sym^)? ($e^)? attr-dict";
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

def FormatMaybeEmptyType : TEST_Op<"format_maybe_empty_type"> {
  let arguments = (ins TestTypeOptionalValueType:$in);
  let assemblyFormat = "$in `:` type($in) attr-dict";
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

def FormatCustomDirectiveSpacing
    : TEST_Op<"format_custom_directive_spacing"> {
  let arguments = (ins StrAttr:$attr1, StrAttr:$attr2);
  let assemblyFormat = [{
    custom<CustomDirectiveSpacing>($attr1)
    custom<CustomDirectiveSpacing>($attr2)
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

def FormatLiteralFollowingOptionalGroup
    : TEST_Op<"format_literal_following_optional_group"> {
  let arguments = (ins TypeAttr:$type, OptionalAttr<AnyAttr>:$value);
  let assemblyFormat = "(`(` $value^ `)`)? `:` $type attr-dict";
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
  let arguments = (ins TypedAttrInterface:$value1, AnyType:$value2);
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
  let arguments = (ins TypedAttrInterface:$value);
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
    static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
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
    static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
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
    static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      if (regions.empty())
        return ::mlir::failure();
      auto types = regions.front()->getArgumentTypes();
      inferredReturnTypes.assign(types.begin(), types.end());
      return ::mlir::success();
    }
  }];
}

// Test inferReturnTypes coupled with variadic operands (operandSegmentSizes).
def FormatInferTypeVariadicOperandsOp
    : TEST_Op<"format_infer_type_variadic_operands",
              [InferTypeOpInterface, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<I32>:$a, Variadic<I64>:$b);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = "`(` $a `:` type($a) `)` `(` $b `:` type($b) `)` attr-dict";
  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context,
          ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
          ::mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
          ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      FormatInferTypeVariadicOperandsOpAdaptor adaptor(
          operands, attributes, *properties.as<Properties *>(), {});
      auto aTypes = adaptor.getA().getTypes();
      auto bTypes = adaptor.getB().getTypes();
      inferredReturnTypes.append(aTypes.begin(), aTypes.end());
      inferredReturnTypes.append(bTypes.begin(), bTypes.end());
      return ::mlir::success();
    }
  }];
}

#endif  // TEST_OPS_SYNTAX
