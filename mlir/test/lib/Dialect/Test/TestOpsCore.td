//===- TestOpsCore.td - Test dialect core ops --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_CORE
#define TEST_OPS_CORE

include "TestOpBase.td"

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

def SignlessLikeVariadic : TEST_Op<"signless_like_variadic"> {
  let arguments = (ins Variadic<SignlessIntegerLike>:$x);
}

//===----------------------------------------------------------------------===//
// Test Token Type
//===----------------------------------------------------------------------===//

// Produce a builtin `!token` value.
def TestTokenProduceOp : TEST_Op<"token.produce"> {
  let results = (outs Token:$token);
  let assemblyFormat = "attr-dict";
}

// Produce a builtin `!token` value via an opt-in result type without the
// producer trait. ODS only auto-derives the trait when the declared type is
// statically `Token`, so this op (whose result type accepts any type
// including tokens) exercises the runtime trait check.
def TestTokenProduceWithoutTraitOp
    : TEST_Op<"token.produce_without_trait"> {
  let results = (outs Type<CPred<"true">>:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

// Consume a builtin `!token` value (token-only operand). The consumer trait
// is auto-derived by ODS from the statically-typed `Token` operand.
def TestTokenConsumeOp : TEST_Op<"token.consume"> {
  let arguments = (ins Token:$token);
  let assemblyFormat = "$token attr-dict";
}

// Consume a builtin `!token` value via an opt-in operand type without the
// consumer trait. ODS only auto-derives the trait when the declared type is
// statically `Token`, so this op (whose operand type accepts any type
// including tokens) exercises the runtime trait check.
def TestTokenConsumeWithoutTraitOp
    : TEST_Op<"token.consume_without_trait"> {
  let arguments = (ins Type<CPred<"true">>:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

// Op that uses the default `AnyType` predicate. Tokens are excluded by
// default and should be rejected by the verifier when passed here.
def TestTokenAnyTypeOp : TEST_Op<"token.any_type", [TokenConsumerTrait]> {
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

// Op whose region entry blocks may produce builtin `!token` values.
def TestTokenRegionOp : TEST_Op<"token.region", [TokenProducerTrait]> {
  let regions = (region AnyRegion:$body);
}

// Op whose regions do not opt in to producing builtin `!token` values.
def TestTokenRegionWithoutTraitOp
    : TEST_Op<"token.region_without_trait"> {
  let regions = (region AnyRegion:$body);
}

//===----------------------------------------------------------------------===//
// Test Symbols
//===----------------------------------------------------------------------===//

def SymbolOp
    : TEST_Op<"symbol", [NoMemoryEffect, SymbolName, SymbolVisibility, Symbol]> {
  let summary =  "operation which defines a new symbol";
  let arguments = (ins StrAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
}

def SymbolWithResultOp
    : TEST_Op<"symbol_with_result", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "invalid symbol operation that produces an SSA result";
  let arguments = (ins StrAttr:$sym_name,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let results = (outs AnyType:$result);
}

def OverriddenSymbolVisibilityOp : TEST_Op<"overridden_symbol_visibility", [
  SymbolName,
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
          mlir::DictionaryAttr attributes, mlir::PropertyRef properties, mlir::RegionRange regions,
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
      return getCalleeAttr();
    }

    void $cppClass::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      setCalleeAttr(cast<SymbolRefAttr>(callee));
    }
  }];
}

def ConversionFuncOp
    : TEST_Op<"conversion_func_op",
              [SymbolName, SymbolVisibility, FunctionOpInterface]> {
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

def TestLocationAttrOp : TEST_Op<"op_with_loc_attr"> {
  let arguments = (ins LocationAttr:$loc_attr);
  let results = (outs );
  let assemblyFormat = "$loc_attr attr-dict";
}

//===----------------------------------------------------------------------===//
// Test Legalization
//===----------------------------------------------------------------------===//

// Check that patterns use the most up-to-date value when being replaced.
def TestRewriteOp : TEST_Op<"rewrite">,
  Arguments<(ins AnyType)>, Results<(outs AnyType)>;

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

def ManyRegionsOp : TEST_Op<"many_regions", []> {
  let summary = "operation created with move-only objects";
  let description = [{
    Test op with multiple regions with a `create` function that
    takes parameters containing move-only objects.
  }];

  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let builders =
      [OpBuilder<(ins "::std::unique_ptr<::mlir::Region>":$singleRegion), [{
      $_state.addRegion(std::move(singleRegion));
      build($_builder, $_state, {}, /*regionsCount=*/1);
    }]>,
       // Define in TestOps.cpp.
       OpBuilder<(ins "::llvm::SmallVectorImpl<::std::unique_ptr<::mlir::"
                      "Region>>&&":$regions)>];
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
          ::mlir::DictionaryAttr attributes, mlir::PropertyRef properties, ::mlir::RegionRange regions,
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

// Properties of every supported kind on an inferred-result aggregate builder.
// The inference hook checks that the legacy mixed-attribute overload populated
// defaults and converted properties before invoking it.
def TableGenBuildOp8 : TEST_Op<"tblgen_build_8", [
    InferTypeOpInterface, AttrSizedOperandSegments, AttrSizedResultSegments
  ]> {
  let arguments = (ins
    Variadic<AnyType>:$a,
    Variadic<AnyType>:$b,
    BoolAttr:$attr0,
    DefaultValuedAttr<I32Attr, "7">:$defaultAttr,
    I64Prop:$nativeProp
  );
  let results = (outs Variadic<AnyType>:$resultA,
                      Variadic<AnyType>:$resultB);

  let extraClassDeclaration = [{
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *, ::std::optional<::mlir::Location>,
        ::mlir::ValueRange operands, ::mlir::DictionaryAttr,
        ::mlir::PropertyRef properties, ::mlir::RegionRange,
        ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
      if (!properties)
        return ::mlir::failure();
      const auto &prop = *properties.as<const Properties *>();
      if (!prop.attr0 || !prop.attr0.getValue() || !prop.defaultAttr ||
          prop.defaultAttr.getInt() != 7 || prop.nativeProp != 42 ||
          prop.operandSegmentSizes != std::array<int32_t, 2>{1, 1} ||
          prop.resultSegmentSizes != std::array<int32_t, 2>{1, 0})
        return ::mlir::failure();
      inferredReturnTypes.assign({operands.front().getType()});
      return ::mlir::success();
    }
  }];
}

// Exercise the aggregate builder that derives results from operand types.
def TableGenBuildOp9 : TEST_Op<"tblgen_build_9",
    [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$input, BoolAttr:$attr0);
  let results = (outs AnyType:$result);
}

// Exercise the aggregate builder that derives results from its first
// attribute after that attribute has been converted to properties.
def TableGenBuildOp10 : TEST_Op<"tblgen_build_10",
    [FirstAttrDerivedResultType]> {
  let arguments = (ins TypeAttr:$type, AnyType:$input);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Test Ops with Default-Valued String Attributes
//===----------------------------------------------------------------------===//

def TestDefaultStrAttrNoValueOp : TEST_Op<"no_str_value"> {
  let arguments = (ins DefaultValuedAttr<StrAttr, "">:$value);
  let assemblyFormat = "prop-dict attr-dict";
}

def TestDefaultStrAttrHasValueOp : TEST_Op<"has_str_value"> {
  let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$value);
  let assemblyFormat = "prop-dict attr-dict";
}

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

//===----------------------------------------------------------------------===//
// Test Ops with Default-Valued Attributes and Differing Print Settings
//===----------------------------------------------------------------------===//

def TestDefaultAttrPrintOp : TEST_Op<"default_value_print"> {
  let arguments = (ins DefaultValuedAttr<I32Attr, "0">:$value_with_default,
                   I32:$operand);
  let assemblyFormat = "prop-dict attr-dict $operand";
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
      if (auto sym = (*this)->getDiscardableAttrOfType<::mlir::SymbolRefAttr>(
              "callee"))
        return ::mlir::CallInterfaceCallable(sym);
      return ::mlir::CallInterfaceCallable();
    }
    void $cppClass::setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      if (auto sym = dyn_cast<::mlir::SymbolRefAttr>(callee))
        (*this)->setDiscardableAttr("callee", sym);
      else
        (*this)->removeDiscardableAttr("callee");
    }
  }];
}

#endif // TEST_OPS_CORE
