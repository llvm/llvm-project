//===-- SPIRVControlFlowOps.td - SPIR-V Control Flow Ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains control flow ops for the SPIR-V dialect. It corresponds
// to "3.32.17. Control-Flow Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_CONTROLFLOW_OPS
#define SPIRV_CONTROLFLOW_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPV_BranchOp : SPV_Op<"Branch", [
    DeclareOpInterfaceMethods<BranchOpInterface>, InFunctionScope, NoSideEffect,
    Terminator]> {
  let summary = "Unconditional branch to target block.";

  let description = [{
    This instruction must be the last instruction in a block.

    <!-- End of AutoGen section -->

    ```
    branch-op ::= `spv.Branch` successor
    successor ::= bb-id branch-use-list?
    branch-use-list ::= `(` ssa-use-list `:` type-list-no-parens `)`
    ```

    #### Example:

    ```mlir
    spv.Branch ^target
    spv.Branch ^target(%0, %1: i32, f32)
    ```
  }];

  let arguments = (ins Variadic<SPV_Type>:$targetOperands);

  let results = (outs);

  let successors = (successor AnySuccessor:$target);

  let verifier = [{ return success(); }];

  let builders = [
    OpBuilderDAG<(ins "Block *":$successor, CArg<"ValueRange", "{}">:$arguments),
    [{
      $_state.addSuccessors(successor);
      $_state.addOperands(arguments);
    }]>
  ];

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Returns the branch target block.
    Block *getTarget() { return target(); }

    /// Returns the block arguments.
    operand_range getBlockArguments() { return targetOperands(); }
  }];

  let autogenSerialization = 0;

  let assemblyFormat = [{
    $target (`(` $targetOperands^ `:` type($targetOperands) `)`)? attr-dict
  }];
}

// -----

def SPV_BranchConditionalOp : SPV_Op<"BranchConditional", [
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
    InFunctionScope, NoSideEffect, Terminator]> {
  let summary = [{
    If Condition is true, branch to true block, otherwise branch to false
    block.
  }];

  let description = [{
    Condition must be a Boolean type scalar.

    Branch weights are unsigned 32-bit integer literals. There must be
    either no Branch Weights or exactly two branch weights. If present, the
    first is the weight for branching to True Label, and the second is the
    weight for branching to False Label. The implied probability that a
    branch is taken is its weight divided by the sum of the two Branch
    weights. At least one weight must be non-zero. A weight of zero does not
    imply a branch is dead or permit its removal; branch weights are only
    hints. The two weights must not overflow a 32-bit unsigned integer when
    added together.

    This instruction must be the last instruction in a block.

    <!-- End of AutoGen section -->

    ```
    branch-conditional-op ::= `spv.BranchConditional` ssa-use
                              (`[` integer-literal, integer-literal `]`)?
                              `,` successor `,` successor
    successor ::= bb-id branch-use-list?
    branch-use-list ::= `(` ssa-use-list `:` type-list-no-parens `)`
    ```

    #### Example:

    ```mlir
    spv.BranchConditional %condition, ^true_branch, ^false_branch
    spv.BranchConditional %condition, ^true_branch(%0: i32), ^false_branch(%1: i32)
    ```
  }];

  let arguments = (ins
    SPV_Bool:$condition,
    Variadic<SPV_Type>:$trueTargetOperands,
    Variadic<SPV_Type>:$falseTargetOperands,
    OptionalAttr<I32ArrayAttr>:$branch_weights
  );

  let results = (outs);

  let successors = (successor AnySuccessor:$trueTarget,
                              AnySuccessor:$falseTarget);

  let builders = [
    OpBuilderDAG<(ins "Value":$condition, "Block *":$trueBlock,
      "ValueRange":$trueArguments, "Block *":$falseBlock,
      "ValueRange":$falseArguments,
      CArg<"Optional<std::pair<uint32_t, uint32_t>>", "{}">:$weights),
    [{
      ArrayAttr weightsAttr;
      if (weights) {
        weightsAttr =
            $_builder.getI32ArrayAttr({static_cast<int32_t>(weights->first),
                                     static_cast<int32_t>(weights->second)});
      }
      build($_builder, $_state, condition, trueArguments, falseArguments,
            weightsAttr, trueBlock, falseBlock);
    }]>
  ];

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
    /// Branch indices into the successor list.
    enum { kTrueIndex = 0, kFalseIndex = 1 };

    /// Returns the target block for the true branch.
    Block *getTrueBlock() { return getOperation()->getSuccessor(kTrueIndex); }

    /// Returns the target block for the false branch.
    Block *getFalseBlock() { return getOperation()->getSuccessor(kFalseIndex); }

    /// Returns the number of arguments to the true target block.
    unsigned getNumTrueBlockArguments() {
      return trueTargetOperands().size();
    }

    /// Returns the number of arguments to the false target block.
    unsigned getNumFalseBlockArguments() {
      return falseTargetOperands().size();
    }

    // Iterator and range support for true target block arguments.
    operand_range getTrueBlockArguments() {
      return trueTargetOperands();
    }

    // Iterator and range support for false target block arguments.
    operand_range getFalseBlockArguments() {
      return falseTargetOperands();
    }

  private:
    /// Gets the index of the first true block argument in the operand list.
    unsigned getTrueBlockArgumentIndex() {
      return 1; // Omit the first argument, which is the condition.
    }

    /// Gets the index of the first false block argument in the operand list.
    unsigned getFalseBlockArgumentIndex() {
      return getTrueBlockArgumentIndex() + getNumTrueBlockArguments();
    }
  }];
}

// -----

def SPV_FunctionCallOp : SPV_Op<"FunctionCall", [
    InFunctionScope, DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Call a function.";

  let description = [{
    Result Type is the type of the return value of the function. It must be
    the same as the Return Type operand of the Function Type operand of the
    Function operand.

    Function is an OpFunction instruction.  This could be a forward
    reference.

    Argument N is the object to copy to parameter N of Function.

    Note: A forward call is possible because there is no missing type
    information: Result Type must match the Return Type of the function, and
    the calling argument types must match the formal parameter types.

    <!-- End of AutoGen section -->

    ```
    function-call-op ::= `spv.FunctionCall` function-id `(` ssa-use-list `)`
                     `:` function-type
    ```

    #### Example:

    ```mlir
    spv.FunctionCall @f_void(%arg0) : (i32) ->  ()
    %0 = spv.FunctionCall @f_iadd(%arg0, %arg1) : (i32, i32) -> i32
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<SPV_Type>:$arguments
  );

  let results = (outs
    Optional<SPV_Type>:$result
  );

  let autogenSerialization = 0;

  let assemblyFormat = [{
    $callee `(` $arguments `)` attr-dict `:`
    functional-type($arguments, results)
  }];
}

// -----

def SPV_LoopOp : SPV_Op<"loop", [InFunctionScope]> {
  let summary = "Define a structured loop.";

  let description = [{
    SPIR-V can explicitly declare structured control-flow constructs using merge
    instructions. These explicitly declare a header block before the control
    flow diverges and a merge block where control flow subsequently converges.
    These blocks delimit constructs that must nest, and can only be entered
    and exited in structured ways. See "2.11. Structured Control Flow" of the
    SPIR-V spec for more details.

    Instead of having a `spv.LoopMerge` op to directly model loop merge
    instruction for indicating the merge and continue target, we use regions
    to delimit the boundary of the loop: the merge target is the next op
    following the `spv.loop` op and the continue target is the block that
    has a back-edge pointing to the entry block inside the `spv.loop`'s region.
    This way it's easier to discover all blocks belonging to a construct and
    it plays nicer with the MLIR system.

    The `spv.loop` region should contain at least four blocks: one entry block,
    one loop header block, one loop continue block, one loop merge block.
    The entry block should be the first block and it should jump to the loop
    header block, which is the second block. The loop merge block should be the
    last block. The merge block should only contain a `spv.mlir.merge` op.
    The continue block should be the second to last block and it should have a
    branch to the loop header block. The loop continue block should be the only
    block, except the entry block, branching to the header block.
  }];

  let arguments = (ins
    SPV_LoopControlAttr:$loop_control
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let builders = [OpBuilderDAG<(ins)>];

  let extraClassDeclaration = [{
    // Returns the entry block.
    Block *getEntryBlock();

    // Returns the loop header block.
    Block *getHeaderBlock();

    // Returns the loop continue block.
    Block *getContinueBlock();

    // Returns the loop merge block.
    Block *getMergeBlock();

    // Adds an empty entry block and loop merge block containing one
    // spv.mlir.merge op.
    void addEntryAndMergeBlock();
  }];

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

// -----

def SPV_MergeOp : SPV_Op<"mlir.merge", [NoSideEffect, Terminator]> {
  let summary = "A special terminator for merging a structured selection/loop.";

  let description = [{
    We use `spv.selection`/`spv.loop` for modelling structured selection/loop.
    This op is a terminator used inside their regions to mean jumping to the
    merge point, which is the next op following the `spv.selection` or
    `spv.loop` op. This op does not have a corresponding instruction in the
    SPIR-V binary format; it's solely for structural purpose.
  }];

  let arguments = (ins);

  let results = (outs);

  let assemblyFormat = "attr-dict";

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

// -----

def SPV_ReturnOp : SPV_Op<"Return", [InFunctionScope, NoSideEffect,
                                     Terminator]> {
  let summary = "Return with no value from a function with void return type.";

  let description = [{
    This instruction must be the last instruction in a block.

    <!-- End of AutoGen section -->

    ```
    return-op ::= `spv.Return`
    ```
  }];

  let arguments = (ins);

  let results = (outs);

  let assemblyFormat = "attr-dict";
}

// -----

def SPV_UnreachableOp : SPV_Op<"Unreachable", [InFunctionScope, Terminator]> {
  let summary = "Declares that this block is not reachable in the CFG.";

  let description = [{
    This instruction must be the last instruction in a block.

    <!-- End of AutoGen section -->

    ```
    unreachable-op ::= `spv.Unreachable`
    ```
  }];

  let arguments = (ins);

  let results = (outs);

  let assemblyFormat = "attr-dict";
}

// -----

def SPV_ReturnValueOp : SPV_Op<"ReturnValue", [InFunctionScope, NoSideEffect,
                                               Terminator]> {
  let summary = "Return a value from a function.";

  let description = [{
    Value is the value returned, by copy, and must match the Return Type
    operand of the OpTypeFunction type of the OpFunction body this return
    instruction is in.

    This instruction must be the last instruction in a block.

    <!-- End of AutoGen section -->

    ```
    return-value-op ::= `spv.ReturnValue` ssa-use `:` spirv-type
    ```

    #### Example:

    ```mlir
    spv.ReturnValue %0 : f32
    ```
  }];

  let arguments = (ins
    SPV_Type:$value
  );

  let results = (outs);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def SPV_SelectionOp : SPV_Op<"selection", [InFunctionScope]> {
  let summary = "Define a structured selection.";

  let description = [{
    SPIR-V can explicitly declare structured control-flow constructs using merge
    instructions. These explicitly declare a header block before the control
    flow diverges and a merge block where control flow subsequently converges.
    These blocks delimit constructs that must nest, and can only be entered
    and exited in structured ways. See "2.11. Structured Control Flow" of the
    SPIR-V spec for more details.

    Instead of having a `spv.SelectionMerge` op to directly model selection
    merge instruction for indicating the merge target, we use regions to delimit
    the boundary of the selection: the merge target is the next op following the
    `spv.selection` op. This way it's easier to discover all blocks belonging to
    the selection and it plays nicer with the MLIR system.

    The `spv.selection` region should contain at least two blocks: one selection
    header block, and one selection merge. The selection header block should be
    the first block. The selection merge block should be the last block.
    The merge block should only contain a `spv.mlir.merge` op.
  }];

  let arguments = (ins
    SPV_SelectionControlAttr:$selection_control
  );

  let results = (outs);

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the selection header block.
    Block *getHeaderBlock();

    /// Returns the selection merge block.
    Block *getMergeBlock();

    /// Adds a selection merge block containing one spv.mlir.merge op.
    void addMergeBlock();

    /// Creates a spv.selection op for `if (<condition>) then { <thenBody> }`
    /// with `builder`. `builder`'s insertion point will remain at after the
    /// newly inserted spv.selection op afterwards.
    static SelectionOp createIfThen(
        Location loc, Value condition,
        function_ref<void(OpBuilder &builder)> thenBody,
        OpBuilder &builder);
  }];

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let hasCanonicalizer = 1;
}

#endif // SPIRV_CONTROLFLOW_OPS
