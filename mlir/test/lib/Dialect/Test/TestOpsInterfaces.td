//===- TestOpsInterfaces.td - Test dialect interfaces ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_INTERFACES
#define TEST_OPS_INTERFACES

include "TestOpBase.td"

//===----------------------------------------------------------------------===//
// Test SideEffects
//===----------------------------------------------------------------------===//

def SideEffectOp : TEST_Op<"side_effect_op",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TestEffectOpInterface>]> {
  let results = (outs AnyType:$result);
}

def ConditionalSideEffectOp
    : TEST_Op<"conditional_side_effect_op", [DeclareOpInterfaceMethods<
                                                MemoryEffectsOpInterface>]> {
  let arguments = (ins BoolAttr:$has_effects);
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
    %0 = test.value_with_bounds <min = 4, max = 5>
    ```
  }];
  let arguments = (ins IndexAttr:$min, IndexAttr:$max);
  let results = (outs Index:$result);
  let assemblyFormat = "prop-dict attr-dict";
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
                       UnitAttr:$allow_integer_type,
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

def SegmentedRegionBranchOp : TEST_Op<"segmented_region_branch", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorInputs"]>]> {
  let description = [{
    Select one of two regions and pass the corresponding operand segment to
    its block arguments. Each region can branch to either region or exit.
    The control operands are not successor operands.
  }];
  let arguments = (ins I1:$condition, Variadic<AnyType>:$first,
                       Variadic<AnyType>:$second, Optional<I1>:$exitCondition);
  let regions = (region SizedRegion<1>:$firstRegion,
                        SizedRegion<1>:$secondRegion);
  let hasCanonicalizer = 1;
}

def SegmentedRegionBranchTerminatorOp
    : TEST_Op<"segmented_region_branch_terminator", [
        AttrSizedOperandSegments, Terminator,
        HasParent<"SegmentedRegionBranchOp">,
        DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
            ["getMutableSuccessorOperands"]>]> {
  let arguments = (ins I1:$condition, Variadic<AnyType>:$first,
                       Variadic<AnyType>:$second, Optional<I1>:$exitCondition);
}

def IsolatedRegionBranchOp : TEST_Op<"isolated_region_branch", [
    IsolatedFromAbove, RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorInputs"]>]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let hasCanonicalizer = 1;
}

def IsolatedRegionYieldOp : TEST_Op<"isolated_region_yield", [
    Pure, ReturnLike, Terminator,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
        ["getMutableSuccessorOperands"]>]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def TypeChangingRegionBranchOp : TEST_Op<"type_changing_region_branch", [
    RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorInputs", "areTypesCompatible"]>]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let hasCanonicalizer = 1;
}

def TypeChangingYieldOp : TEST_Op<"type_changing_yield", [
    Pure, ReturnLike, Terminator,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
        ["getMutableSuccessorOperands"]>]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

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
                                  "getEntrySuccessorOperands",
                                  "getSuccessorInputs"]>,
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
                                 ["getRegionInvocationBounds",
                                  "getSuccessorInputs"]>,
       RecursiveMemoryEffects]> {
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
}

def LoopBlockOp : TEST_Op<"loop_block",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorInputs"]>,
     RecursiveMemoryEffects]> {

  let results = (outs F32:$floatResult);
  let arguments = (ins I32:$init);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $init `:` functional-type($init, $floatResult) $body
    attr-dict-with-keyword
  }];
}

def LoopBlockTerminatorOp : TEST_Op<"loop_block_term",
    [DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
        ["getMutableSuccessorOperands"]>,
     Pure, Terminator]> {
  let arguments = (ins I32:$nextIterArg, F32:$exitArg);

  let assemblyFormat = [{
    `iter` $nextIterArg `exit` $exitArg attr-dict
  }];
}

// An op that has a result and a block argument that are NOT successor inputs,
// used to test the fix for issue #175168 in getControlFlowPredecessors.
// - result #0 (extraResult): NOT a successor input to parent
// - result #1 (iterResult): IS a successor input to parent
// - body block arg #0: NOT a successor input to the body region
// - body block arg #1: IS a successor input to the body region
// This mirrors the scf.for induction variable / iter_arg structure.
def LoopWithExtraResultYieldOp : TEST_Op<"loop_with_extra_result_yield",
    [DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
        ["getMutableSuccessorOperands"]>,
     Pure, Terminator]> {
  let arguments = (ins I32:$iterArg);
  let assemblyFormat = "$iterArg `:` type($iterArg) attr-dict";
}

def LoopWithExtraResultOp : TEST_Op<"loop_with_extra_result",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getSuccessorInputs"]>,
     RecursiveMemoryEffects]> {
  let results = (outs I32:$extraResult, I32:$iterResult);
  let arguments = (ins I32:$init);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $init `:` type($init) `->` `(` type($extraResult) `,` type($iterResult) `)` $body attr-dict
  }];
}

def TestNoTerminatorOp : TEST_Op<"switch_with_no_break", [
    NoTerminator,
    DeclareOpInterfaceMethods<RegionBranchOpInterface>
  ]> {
  let arguments = (ins Index:$arg, DenseI64ArrayAttr:$cases);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg attr-dict custom<SwitchCases>($cases, $caseRegions)
  }];
}

//===----------------------------------------------------------------------===//
// Test areTypesCompatible for RegionBranchOpInterface / LoopLikeOpInterface
//===----------------------------------------------------------------------===//

def TestTypesCompatYieldOp : TEST_Op<"types_compat_yield",
    [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let assemblyFormat = "($args^ `:` type($args))? attr-dict";
}

def TestRegionTypesCompatOp : TEST_Op<"region_types_compat",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
       ["getEntrySuccessorOperands", "getSuccessorInputs",
        "areTypesCompatible"]>,
     SingleBlockImplicitTerminator<"TestTypesCompatYieldOp">,
     RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$entries);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
}

def TestLoopTypesCompatOp : TEST_Op<"loop_types_compat",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface,
       ["getInitsMutable", "getRegionIterArgs", "getYieldedValuesMutable",
        "getLoopResults"]>,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
       ["getEntrySuccessorOperands", "getSuccessorInputs",
        "areTypesCompatible"]>,
     SingleBlockImplicitTerminator<"TestTypesCompatYieldOp">,
     RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$init_args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
}

def TestRegionTypeChangerOp : TEST_Op<"region_type_changer",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
       ["getEntrySuccessorOperands", "getSuccessorInputs",
        "areTypesCompatible"]>,
     RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$entries);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`(` $entries `)` $body `:` functional-type($entries, $results) attr-dict";
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
                                 [IsolatedFromAbove, HasDefaultDLTIDataLayout,
                                  DataLayoutOpInterface,
                                  DLTIQueryOpInterface]> {
  let summary =
      "An op that uses DataLayout implementation from the Target dialect";
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let arguments = (ins OptionalAttr<DLTIQueryAttr>:$dlti);
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
      auto attrs = getOperation()->getDiscardableAttrOfType<mlir::ArrayAttr>(
          "iterator_types");
      auto range = attrs.getAsValueRange<IteratorTypeAttr, mlir::utils::IteratorType>();
      return {range.begin(), range.end()};
    }

    mlir::ArrayAttr getIndexingMaps() {
      return getOperation()->getDiscardableAttrOfType<mlir::ArrayAttr>(
          "indexing_maps");
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
      auto attrs = getOperation()->getDiscardableAttrOfType<mlir::ArrayAttr>(
          "iterator_types");
      auto range = attrs.getAsValueRange<IteratorTypeAttr, mlir::utils::IteratorType>();
      return {range.begin(), range.end()};
    }

    mlir::ArrayAttr getIndexingMaps() {
      return getOperation()->getDiscardableAttrOfType<mlir::ArrayAttr>(
          "indexing_maps");
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
// Test TilingInterface.
//===----------------------------------------------------------------------===//

def Test_TilingNoDpsOp : TEST_Op<"tiling_no_dps_op",
    [Pure, DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
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

def TestNVVMRequiresSMFamilyCondOp :
    TEST_Op<"nvvm_requires_sm_100f", [NVVMRequiresSMf<[100]>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
}

def TestNVVMRequiresSMFamilyCondHigherOp :
    TEST_Op<"nvvm_requires_sm_103f", [NVVMRequiresSMf<[103]>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
}

def TestNVVMRequiresSMFamilyCondMultiOp :
    TEST_Op<"nvvm_requires_sm_100f_or_sm_120f", [NVVMRequiresSMf<[100, 120]>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
}

def TestNVVMRequiresSMArchOrFamilyCondOp :
    TEST_Op<"nvvm_requires_sm_90a_or_sm_100f", [NVVMRequiresSMaOrSMf<[90], [100]>]> {
  let arguments = (ins );
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Test Ops with effects
//===----------------------------------------------------------------------===//

def TestResource : Resource<"TestResource">;
def ParameterizedTestResource : Resource<"TestResource",
    [{::mlir::StringAttr::get(getContext(), "test parameter")}]>;

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

def TestParameterizedEffectsOp : TEST_Op<"op_with_parameterized_effects"> {
  let arguments =
      (ins Arg<AnyMemRef, "", [MemRead<ParameterizedTestResource>]>:$input);
}

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
// Test InferIntRangeInterface
//===----------------------------------------------------------------------===//

def InferIntRangeType : AnyTypeOf<[AnyInteger, Index, VectorOfNonZeroRankOf<[AnyInteger, Index]>]>;

def TestWithBoundsOp : TEST_Op<"with_bounds",
                          [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
                           NoMemoryEffect]> {
  let hasVerifier = 1;
  let description = [{
    Creates a value with specified [min, max] range for integer range analysis.

    Example:

    ```mlir
    %0 = test.with_bounds <umin = 4 : index, umax = 5 : index, smin = 4 : index, smax = 5 : index> : index
    ```
  }];

  let arguments = (ins APIntAttr:$umin,
                       APIntAttr:$umax,
                       APIntAttr:$smin,
                       APIntAttr:$smax);
  let results = (outs InferIntRangeType:$fakeVal);

  let assemblyFormat = "prop-dict attr-dict `:` type($fakeVal)";
}

def TestWithoutBoundsOp : TEST_Op<"without_bounds",
                         [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRangesFromOptional"]>,
                          NoMemoryEffect]> {
  let description = [{
    Creates a value with uninitialized range for integer range analysis tests.

    Example:

    ```mlir
    %0 = test.without_bounds : index
    ```
  }];
  let results = (outs InferIntRangeType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
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
    CHECK: test.reflect_bounds <umin = 0 : index, umax = 7 : index, smin = 0 : index, smax = 7 : index>
    %1 = test.reflect_bounds %0 : index
    ```
  }];

  let arguments = (ins InferIntRangeType:$value,
                       OptionalAttr<APIntAttr>:$umin,
                       OptionalAttr<APIntAttr>:$umax,
                       OptionalAttr<APIntAttr>:$smin,
                       OptionalAttr<APIntAttr>:$smax);
  let results = (outs InferIntRangeType:$result);

  let assemblyFormat = "prop-dict attr-dict $value `:` type($result)";
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
    "$callee `(` $callee_operands `)` `,` $address prop-dict attr-dict "
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
    "prop-dict attr-dict `:` functional-type(operands, results)";
}

// A call-like operation whose first result is produced by the operation itself
// and is *not* forwarded from the callee. Only the trailing results are
// forwarded.
def TestCallAndProduceOp : TEST_Op<"call_and_produce",
    [DeclareOpInterfaceMethods<CallOpInterface, ["getForwardedResults"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$forwarded_operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs
    I1:$produced_status,
    Variadic<AnyType>:$forwarded
  );
  let assemblyFormat =
    "$callee `(` $forwarded_operands `)` prop-dict attr-dict "
    "`:` functional-type(operands, results)";
}

def TestStoreWithARegion : TEST_Op<"store_with_a_region",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
     SingleBlock]> {
  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite]>:$address,
    BoolAttr:$store_before_region
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
    "$address prop-dict attr-dict-with-keyword regions `:` type($address)";
}

def TestStoreWithALoopRegion : TEST_Op<"store_with_a_loop_region",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>,
     SingleBlock]> {
  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite]>:$address,
    BoolAttr:$store_before_region
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat =
    "$address prop-dict attr-dict-with-keyword regions `:` type($address)";
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

// Transparent alias of a memref slot with the same element type. Exercises
// alias-chain handling in mem2reg with identity projections.
def TestTransparentAlias : TEST_Op<"transparent_alias",
    [DeclareOpInterfaceMethods<PromotableOpInterface,
                               ["canUsesBeRemoved",
                                "removeBlockingUses"]>,
     DeclareOpInterfaceMethods<PromotableAliaserInterface,
                               ["getPromotableSlotAliases"]>]> {
  let arguments = (ins MemRefOf<[I32]>:$source);
  let results = (outs MemRefOf<[I32]>:$result);
  let assemblyFormat = "$source attr-dict `:` functional-type($source, $result)";
}

// Type-changing transparent alias of a memref slot. The projection methods
// bridge the element types using `builtin.unrealized_conversion_cast`.
def TestTransparentCastAlias : TEST_Op<"transparent_cast_alias",
    [DeclareOpInterfaceMethods<PromotableOpInterface,
                               ["canUsesBeRemoved",
                                "removeBlockingUses"]>,
     DeclareOpInterfaceMethods<PromotableAliaserInterface,
                               ["getPromotableSlotAliases",
                                "projectSlotValueToAliasValue",
                                "projectAliasValueToSlotValue"]>]> {
  let arguments = (ins MemRefOf<[I32, F32]>:$source);
  let results = (outs MemRefOf<[I32, F32]>:$result);
  let assemblyFormat = "$source attr-dict `:` functional-type($source, $result)";
}

// Records the value last stored to a memref slot to test
// `PromotableOpInterface::visitReplacedValues`.
def TestSlotTracker : TEST_Op<"slot_tracker",
    [DeclareOpInterfaceMethods<PromotableOpInterface,
                               ["canUsesBeRemoved",
                                "removeBlockingUses",
                                "requiresReplacedValues",
                                "visitReplacedValues"]>]> {
  let arguments = (ins AnyMemRef:$source, StrAttr:$name);
  let assemblyFormat =
      "$source `,` $name attr-dict `:` type($source)";
}

// Marker emitted by `test.slot_tracker` at each reaching-definition store.
def TestTrackedValue : TEST_Op<"tracked_value"> {
  let arguments = (ins AnyType:$value, StrAttr:$name);
  let assemblyFormat =
      "$value `,` $name attr-dict `:` type($value)";
}

// Transparent alias of a memref slot exposing two simultaneously-usable
// aliases of the same bit-width at different signednesses (signed and
// unsigned 32-bit integers, both aliasing the signless i32 storage).
// Exercises mem2reg's handling of an aliaser whose
// `getPromotableSlotAliases` returns more than one entry for a single
// aliased operand.
def TestTransparentDualAlias : TEST_Op<"transparent_dual_alias",
    [DeclareOpInterfaceMethods<PromotableOpInterface,
                               ["canUsesBeRemoved",
                                "removeBlockingUses"]>,
     DeclareOpInterfaceMethods<PromotableAliaserInterface,
                               ["getPromotableSlotAliases",
                                "projectSlotValueToAliasValue",
                                "projectAliasValueToSlotValue"]>]> {
  let arguments = (ins MemRefOf<[I32]>:$source);
  let results = (outs MemRefOf<[SI32]>:$result_signed,
                      MemRefOf<[UI32]>:$result_unsigned);
  let assemblyFormat = "$source attr-dict `:` functional-type($source, results)";
}

// Partial alias of a memref slot: exposes a sub-value of the parent slot.
// The slot-to-alias projection is a 1-input `unrealized_conversion_cast`
// (sub-value extraction). The alias-to-slot projection is a 2-input cast
// taking both the new alias value and the parent slot's reaching definition
// (sub-value insertion). This exercises mem2reg's `reachingDef` plumbing
// for partial sub-aliases.
def TestPartialAlias : TEST_Op<"partial_alias",
    [DeclareOpInterfaceMethods<PromotableOpInterface,
                               ["canUsesBeRemoved",
                                "removeBlockingUses"]>,
     DeclareOpInterfaceMethods<PromotableAliaserInterface,
                               ["getPromotableSlotAliases",
                                "projectSlotValueToAliasValue",
                                "projectAliasValueToSlotValue"]>]> {
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$source attr-dict `:` functional-type($source, $result)";
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

def TestDummyTensorOp
    : TEST_Op<"dummy_tensor_op",
              [DeclareOpInterfaceMethods<
                  BufferizableOpInterface,
                  ["bufferize", "getBufferType", "bufferizesToMemoryRead",
                   "bufferizesToMemoryWrite", "getAliasingValues",
                   "getAliasingOpOperands"]>]> {
  let arguments = (ins
    Arg<Bufferization_TensorLikeTypeInterface>:$input
  );
  let results = (outs
    Arg<Bufferization_TensorLikeTypeInterface>:$output
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
      auto relation = getInput().getType() == getOutput().getType()
                          ? ::mlir::bufferization::BufferRelation::Equivalent
                          : ::mlir::bufferization::BufferRelation::Unknown;
      return {{getOutput(), relation, /*isDefinite=*/true}};
    }

    ::mlir::bufferization::AliasingOpOperandList
    test::TestDummyTensorOp::getAliasingOpOperands(::mlir::Value value,
        const ::mlir::bufferization::AnalysisState&) {
      if (value != getOutput())
        return {};

      auto relation = getInput().getType() == getOutput().getType()
                          ? ::mlir::bufferization::BufferRelation::Equivalent
                          : ::mlir::bufferization::BufferRelation::Unknown;
      return {{&getOperation()->getOpOperand(0), relation,
               /*isDefinite=*/true}};
    }
  }];
}

def TestDummyMemrefOp : TEST_Op<"dummy_memref_op", []> {
  let arguments = (ins
    Arg<Bufferization_BufferLikeTypeInterface>:$input
  );
  let results = (outs
    Arg<Bufferization_BufferLikeTypeInterface>:$output
  );
}

def TestCreateTensorOp
    : TEST_Op<"create_tensor_op",
              [DeclareOpInterfaceMethods<
                  BufferizableOpInterface,
                  ["bufferize", "getBufferType", "bufferizesToMemoryRead",
                   "bufferizesToMemoryWrite", "getAliasingValues",
                   "getAliasingOpOperands", "bufferizesToAllocation"]>]> {
  let arguments = (ins);
  let results = (outs Arg<Bufferization_TensorLikeTypeInterface>:$output);
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

    ::mlir::bufferization::AliasingOpOperandList
    test::TestCreateTensorOp::getAliasingOpOperands(
        ::mlir::Value value,
        const ::mlir::bufferization::AnalysisState&) {
      (void)value;
      return {};
    }
  }];
}

def TestCreateMemrefOp : TEST_Op<"create_memref_op"> {
  let arguments = (ins);
  let results = (outs Arg<Bufferization_BufferLikeTypeInterface>:$output);
}

def TestTensorWithFutureLayoutOp
    : TEST_Op<"tensor_with_future_layout",
              [DeclareOpInterfaceMethods<
                  BufferizableOpInterface,
                  ["bufferize", "getBufferType", "bufferizesToMemoryRead",
                   "bufferizesToMemoryWrite", "getAliasingValues",
                   "getAliasingOpOperands", "bufferizesToAllocation"]>]> {
  let description = [{
    A tensor operation that creates a new value and then forces a specific
    `TestMemRefLayoutAttr` layout during bufferization into a memref. In tensor
    world, this op does nothing encoding/layout wise.
  }];

  let arguments = (ins TestMemRefLayoutAttr:$layout);
  let results = (outs Bufferization_TensorLikeTypeInterface:$output);
  let extraClassDefinition = [{
    bool test::TestTensorWithFutureLayoutOp::bufferizesToMemoryRead(
        ::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return false;
    }
    bool test::TestTensorWithFutureLayoutOp::bufferizesToMemoryWrite(
        ::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return false;
    }
    bool test::TestTensorWithFutureLayoutOp::bufferizesToAllocation(
        mlir::Value) {
      return true;
    }
    ::mlir::bufferization::AliasingValueList
    test::TestTensorWithFutureLayoutOp::getAliasingValues(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return {};
    }
    ::mlir::bufferization::AliasingOpOperandList
    test::TestTensorWithFutureLayoutOp::getAliasingOpOperands(::mlir::Value,
        const ::mlir::bufferization::AnalysisState&) {
      return {};
    }
  }];
}

def TestForceNewLayoutOp
    : TEST_Op<"force_new_layout",
              [DeclareOpInterfaceMethods<
                  BufferizableOpInterface,
                  ["bufferize", "getBufferType", "bufferizesToMemoryRead",
                   "bufferizesToMemoryWrite", "getAliasingValues",
                   "getAliasingOpOperands"]>]> {
  let description = [{
    A tensor operation that modifies a pre-existing value and then forces a
    specific `TestMemRefLayoutAttr` to be set during bufferization into a
    memref. In tensor world, this op does nothing encoding/layout wise. It is
    similar to TestTensorWithFutureLayoutOp, but it keeps the IR values
    "connected" which can be particularly useful in e.g. SCF loop's body (one
    can modify an initArg's layout and yield the new-layout value).
  }];

  let arguments = (ins
    Bufferization_TensorLikeTypeInterface:$input,
    TestMemRefLayoutAttr:$layout
  );
  let results = (outs Bufferization_TensorLikeTypeInterface:$output);
  let extraClassDefinition = [{
    bool test::TestForceNewLayoutOp::bufferizesToMemoryRead(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    bool test::TestForceNewLayoutOp::bufferizesToMemoryWrite(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    ::mlir::bufferization::AliasingValueList
    test::TestForceNewLayoutOp::getAliasingValues(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      auto relation = getInput().getType() == getOutput().getType()
                          ? ::mlir::bufferization::BufferRelation::Equivalent
                          : ::mlir::bufferization::BufferRelation::Unknown;
      return {{getOutput(), relation, /*isDefinite=*/true}};
    }

    ::mlir::bufferization::AliasingOpOperandList
    test::TestForceNewLayoutOp::getAliasingOpOperands(::mlir::Value value,
        const ::mlir::bufferization::AnalysisState&) {
      if (value != getOutput())
        return {};

      auto relation = getInput().getType() == getOutput().getType()
                          ? ::mlir::bufferization::BufferRelation::Equivalent
                          : ::mlir::bufferization::BufferRelation::Unknown;
      return {{&getOperation()->getOpOperand(0), relation,
               /*isDefinite=*/true}};
    }
  }];
}

def TestDummyCastOp : TEST_Op<"dummy_cast_op", []> {
  let arguments = (ins
    Arg<Bufferization_BufferLikeTypeInterface>:$source
  );
  let results = (outs
    Arg<Bufferization_BufferLikeTypeInterface>:$dest
  );
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

#endif // TEST_OPS_INTERFACES
