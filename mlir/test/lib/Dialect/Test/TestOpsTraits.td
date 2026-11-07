//===- TestOpsTraits.td - Test dialect traits ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OPS_TRAITS
#define TEST_OPS_TRAITS

include "TestOpBase.td"

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

def OperandAndResultHaveSameElementCountAnyType :
    TEST_Op<"operand_and_result_have_same_element_count_any_type",
            [AllElementCountsMatch<["x", "res"]>]> {
  let arguments = (ins AnyType:$x);
  let results = (outs AnyType:$res);
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

// HasAncestor trait
def ChildWithAncestor
    : TEST_Op<"child_with_ancestor", [HasAncestor<"ParentOp">]>;

// AncestorOneOf trait
def ChildWithAncestorOneOf
    : TEST_Op<"child_with_ancestor_one_of", [AncestorOneOf<["ParentOp",
                                                            "ParentOp1"]>]>;

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

def ReifyShapedTypeUsingReifyResultShapesOp :
    TEST_Op<"reify_shaped_type_using_reify_result_shapes",
        [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
            ["reifyResultShapes"]>]> {
  let description = [{
    Test that when resolving a single dimension of a result for an operation
    that doesnt implement `reifyShapeOfResult` nor implements `reifyDimOfResult`
    calls into the implementation of `reifyResultShapes` to get the required value.
    The op semantics is that the first result has the same shape as the second operand
    and the second result has the same shape as the first operand.
  }];
  let arguments = (ins AnyRankedTensor:$operand1, AnyRankedTensor:$operand2);
  let results = (outs AnyRankedTensor:$result1, AnyRankedTensor:$result2);
}

def ReifyShapedTypeUsingReifyShapeOfResultOp :
    TEST_Op<"reify_shaped_type_using_reify_shape_of_result",
        [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
            ["reifyResultShapes", "reifyShapeOfResult"]>]> {
  let description = [{
    Test that when resolving a single dimension of a result for an operation
    that doesnt implement `reifyDimOfResult` but implements `reifyShapeOfResult`, which
    is used to get the required value. `reifyResultShapes` is implemented as a failure
    (which is also the default implementation) to ensure it is not called.
    The op semantics is that the first result has the same shape as the second operand
    and the second result has the same shape as the first operand.
  }];
  let arguments = (ins AnyRankedTensor:$operand1, AnyRankedTensor:$operand2);
  let results = (outs AnyRankedTensor:$result1, AnyRankedTensor:$result2);
}

def ReifyShapedTypeUsingReifyDimOfResultOp :
    TEST_Op<"reify_shaped_type_using_reify_dim_of_result",
        [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
            ["reifyResultShapes", "reifyShapeOfResult", "reifyDimOfResult"]>]> {
  let description = [{
    Test that when resolving a single dimension of a result for an operation
    that implements `reifyDimOfResult`, which is used to get the required value.
    `reifyResultShapes` and `reifyShapeOfResult` are implemented as failures
    to ensure they are not called. The op semantics is that the first result has
    the same shape as the second operand and the second result has the same shape
    as the first operand.
  }];
  let arguments = (ins AnyRankedTensor:$operand1, AnyRankedTensor:$operand2);
  let results = (outs AnyRankedTensor:$result1, AnyRankedTensor:$result2);
}

def UnreifiableResultShapesOp : TEST_Op<"unreifiable_result_shapes",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
        ["reifyResultShapes"]>]> {
  let description = [{
    Test handling of case where some dimension of the result cannot be
    reified. This tests the path when `reifyResultShapes` is implemented.

    Expected that dim 0 of `result` is reifable as dim 0 of `operand`, but
    dim 1 of `result` is not reifiable.
  }];
  let arguments = (ins 2DTensorOf<[AnyType]>:$operand);
  let results = (outs 2DTensorOf<[AnyType]>:$result);
}

def UnreifiableResultShapeOp : TEST_Op<"unreifiable_result_shape",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
        ["reifyResultShapes", "reifyShapeOfResult"]>]> {
  let description = [{
    Test handling of case where some dimension of the result cannot be
    reified. This tests the path when `reifyShapeOfResult` is implemented,
    but not `reifyDimOfResult` with `reifyResultShapes` implemented as a failure.

    Expected that dim 0 of `result` is reifable as dim 0 of `operand`, but
    dim 1 of `result` is not reifiable.
  }];
  let arguments = (ins 2DTensorOf<[AnyType]>:$operand);
  let results = (outs 2DTensorOf<[AnyType]>:$result);
}

def UnreifiableDimOfResultShapeOp : TEST_Op<"unreifiable_dim_of_result_shape",
    [DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface,
        ["reifyResultShapes", "reifyShapeOfResult", "reifyDimOfResult"]>]> {
  let description = [{
    Test handling of case where some dimension of the result cannot be
    reified. This tests the path when `reifyDimOfResult` is implemented,
    and `reifyDimOfResult` with `reifyResultShapes` are implemented as a failure.

    Expected that dim 0 of `result` is reifable as dim 0 of `operand`, but
    dim 1 of `result` is not reifiable.
  }];
  let arguments = (ins 2DTensorOf<[AnyType]>:$operand);
  let results = (outs 2DTensorOf<[AnyType]>:$result);
}

def TestGreedyCfgRewriteOp : TEST_Op<"greedy_cfg_rewrite"> {
  let description = [{
    Exercises reachability changes while the greedy driver is processing its
    worklist. The matching test pattern performs the CFG change selected by
    `mode`, then erases the op:

      observe:           emit a remark; used to observe whether a block is
                         processed
      append_unreachable: append a disconnected block holding an observer
      append_reachable:  append a block holding an observer and branch to it
                         on `cond`
      insert_entry:      insert a new entry block, disconnecting the old one
      move_entry:        move the last block to the front
      move_entry_away:   insert a block after the entry and move the entry to
                         the end
      erase_entry:       erase the entry block; its successor becomes the entry
      redirect:          retarget the block's branch to the last block
      double_redirect:   retarget the block's branch to the last block and
                         replace the entry's branch with a return; for a
                         conditional branch, keep only the false destination
      reconnect:         replace the block's branch with a conditional branch
                         on `cond` to its destination and to the next block
      disconnect_parent: retarget the entry branch of the parent region
      move_parent:       move the grandparent operation into a new block
      move_erase:        retarget the block's branch, then move its old
                         destination to another region and erase it
      merge:             merge the branch destination into the block
      merge_disconnect:  merge, then fold the merged conditional branch
      redirect_retarget: retarget the block's branch to its destination's
                         destination, and the destination's branch to an
                         appended block holding an observer

    Modes that operate on the block's branch expect an unconditional branch
    without operands as the terminator, except that `double_redirect` also
    accepts a conditional branch without successor operands.
  }];
  let arguments = (ins EnumProp<TestGreedyCfgRewriteMode>:$mode,
                       Optional<I1>:$cond);
  let assemblyFormat = "$mode ($cond^)? attr-dict";
}

def TestBranchOp : TEST_Op<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$targetOperands,
                       UnitAttr:$fold_to_last);
  let successors = (successor AnySuccessor:$target);
  // With `fold_to_last`, the fold retargets the branch in place to the last
  // block of the region, to exercise CFG changes made by folding.
  let hasFolder = 1;
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

// A successor that must be the block immediately following the op's block.
def FallthroughSuccessor : Successor<
    CPred<"$_self == ($_op).getBlock()->getNextNode()">,
    "the fallthrough block (the block immediately following the op's block)">;

// A successor that must be a block listed after the op's block.
def ForwardSuccessor : Successor<
    CPred<[{
      [&] {
        for (::mlir::Block *block = ($_op).getBlock()->getNextNode(); block;
             block = block->getNextNode())
          if (block == $_self)
            return true;
        return false;
      }()
    }]>,
    "a forward block (a block listed after the op's block)">;

def TestFallthroughBranchOp : TEST_Op<"fallthrough_br", [Terminator]> {
  let summary = "branch whose successors are constrained by predicates";
  let successors = (successor FallthroughSuccessor:$target,
                              VariadicSuccessor<ForwardSuccessor>:$forwardTargets);
  let assemblyFormat = [{
    $target `forward` `[` $forwardTargets `]` attr-dict
  }];
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

#endif // TEST_OPS_TRAITS
