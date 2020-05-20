//===- SCFOps.td - Structured Control Flow operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR structured control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_SCFOPS
#define MLIR_DIALECT_SCF_SCFOPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SCF_Dialect : Dialect {
  let name = "scf";
  let cppNamespace = "";
}

// Base class for SCF dialect ops.
class SCF_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<SCF_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ForOp : SCF_Op<"for",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"YieldOp">,
       RecursiveSideEffects]> {
  let summary = "for operation";
  let description = [{
    The "scf.for" operation represents a loop taking 3 SSA value as operands
    that represent the lower bound, upper bound and step respectively.  The
    operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "scf.yield". Calling ForOp::build will create such a region and insert
    the terminator implicitly if none is defined, so will the parsing even in
    cases when it is absent from the custom format. For example:

    ```mlir
    scf.for %iv = %lb to %ub step %step {
      ... // body
    }
    ```

    `scf.for` can also operate on loop-carried variables and returns the final
    values after loop termination. The initial values of the variables are
    passed as additional SSA operands to the "scf.for" following the 3 loop
    control SSA values mentioned above (lower bound, upper bound and step). The
    operation region has equivalent arguments for each variable representing
    the value of the variable at the current iteration.

    The region must terminate with a "scf.yield" that passes all the current
    iteration variables to the next iteration, or to the "scf.for" result, if
    at the last iteration. Note, that when the loop-carried variables are
    present, calling ForOp::build will not insert the terminator implicitly.
    The caller must insert "scf.yield" in that case.

    "scf.for" results hold the final values after the last iteration.
    For example, to sum-reduce a memref:

    ```mlir
    func @reduce(%buffer: memref<1024xf32>, %lb: index,
                 %ub: index, %step: index) -> (f32) {
      // Initial sum set to 0.
      %sum_0 = constant 0.0 : f32
      // iter_args binds initial values to the loop's region arguments.
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %sum_next = addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```

    If the "scf.for" defines any values, a yield must be explicitly present.
    The number and types of the "scf.for" results must match the initial
    values in the "iter_args" binding and the yield operands.

    Another example with a nested "scf.if" (see "scf.if" for details) to
    perform conditional reduction:

    ```mlir
    func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index,
                             %ub: index, %step: index) -> (f32) {
      %sum_0 = constant 0.0 : f32
      %c0 = constant 0.0 : f32
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %cond = cmpf "ugt", %t, %c0 : f32
        %sum_next = scf.if %cond -> (f32) {
          %new_sum = addf %sum_iter, %t : f32
          scf.yield %new_sum : f32
        } else {
          scf.yield %sum_iter : f32
        }
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```
  }];
  let arguments = (ins Index:$lowerBound,
                       Index:$upperBound,
                       Index:$step,
                       Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value lowerBound, Value upperBound, Value step, "
              "ValueRange iterArgs = llvm::None, "
              "function_ref<void(OpBuilder &, Location, Value, ValueRange)>"
              "    = nullptr">
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    Value getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }
    Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }

    /// Return operands used when entering the region at 'index'. These operands
    /// correspond to the loop iterator operands, i.e., those exclusing the
    /// induction variable. LoopOp only has one region, so 0 is the only valid
    /// value for `index`.
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];
}

def IfOp : SCF_Op<"if",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"YieldOp">, RecursiveSideEffects]> {
  let summary = "if-then-else operation";
  let description = [{
    The `scf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    scf.if %b  {
      ...
    } else {
      ...
    }
    ```

    `scf.if` may also return results that are defined in its regions. The
    values defined are determined by which execution path is taken.

    Example:

    ```mlir
    %x, %y = scf.if %b -> (f32, f32) {
      %x_true = ...
      %y_true = ...
      scf.yield %x_true, %y_true : f32, f32
    } else {
      %x_false = ...
      %y_false = ...
      scf.yield %x_false, %y_false : f32, f32
    }
    ```

    `scf.if` regions are always terminated with "scf.yield". If "scf.if"
    defines no values, the "scf.yield" can be left out, and will be inserted
    implicitly. Otherwise, it must be explicit.
    Also, if "scf.if" defines one or more values, the 'else' block cannot be
    omitted.

    Example:

    ```mlir
    scf.if %b  {
      ...
    }
    ```
  }];
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value cond, bool withElseRegion">,
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "TypeRange resultTypes, Value cond, bool withElseRegion">
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      Block* body = getBody(0);
      return results().empty() ? OpBuilder::atBlockTerminator(body)
                               : OpBuilder::atBlockEnd(body);
    }
    OpBuilder getElseBodyBuilder() {
      Block* body = getBody(1);
      return results().empty() ? OpBuilder::atBlockTerminator(body)
                               : OpBuilder::atBlockEnd(body);
    }
  }];
}

def ParallelOp : SCF_Op<"parallel",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveSideEffects,
     SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "parallel for operation";
  let description = [{
    The "scf.parallel" operation represents a loop nest taking 4 groups of SSA
    values as operands that represent the lower bounds, upper bounds, steps and
    initial values, respectively. The operation defines a variadic number of
    SSA values for its induction variables. It has one region capturing the
    loop body. The induction variables are represented as an argument of this
    region. These SSA values always have type index, which is the size of the
    machine word. The steps are values of type index, required to be positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound. The initial values
    have the same types as results of "scf.parallel". If there are no results,
    the keyword `init` can be omitted.

    Semantically we require that the iteration space can be iterated in any
    order, and the loop body can be executed in parallel. If there are data
    races, the behavior is undefined.

    The parallel loop operation supports reduction of values produced by
    individual iterations into a single result. This is modeled using the
    scf.reduce operation (see scf.reduce for details). Each result of a
    scf.parallel operation is associated with an initial value operand and
    reduce operation that is an immediate child. Reductions are matched to
    result and initial values in order of their appearance in the body.
    Consequently, we require that the body region has the same number of
    results and initial values as it has reduce operations.

    The body region must contain exactly one block that terminates with
    "scf.yield" without operands. Parsing ParallelOp will create such a region
    and insert the terminator when it is absent from the custom format.

    Example:

    ```mlir
    %init = constant 0.0 : f32
    scf.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init) -> f32 {
      %elem_to_reduce = load %buffer[%iv] : memref<100xf32>
      scf.reduce(%elem_to_reduce) : f32 {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = addf %lhs, %rhs : f32
          scf.reduce.return %res : f32
      }
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "ValueRange lowerBounds, ValueRange upperBounds, "
              "ValueRange steps, ValueRange initVals = {}">,
  ];

  let extraClassDeclaration = [{
    ValueRange getInductionVars() {
      return getBody()->getArguments();
    }
    unsigned getNumLoops() { return step().size(); }
    unsigned getNumReductions() { return initVals().size(); }
  }];
}

def ReduceOp : SCF_Op<"reduce", [HasParent<"ParallelOp">]> {
  let summary = "reduce operation for parallel for";
  let description = [{
    "scf.reduce" is an operation occurring inside "scf.parallel" operations.
    It consists of one block with two arguments which have the same type as the
    operand of "scf.reduce".

    "scf.reduce" is used to model the value for reduction computations of a
    "scf.parallel" operation. It has to appear as an immediate child of a
    "scf.parallel" and is associated with a result value of its parent
    operation.

    Association is in the order of appearance in the body where the first
    result of a parallel loop operation corresponds to the first "scf.reduce"
    in the operation's body region. The reduce operation takes a single
    operand, which is the value to be used in the reduction.

    The reduce operation contains a region whose entry block expects two
    arguments of the same type as the operand. As the iteration order of the
    parallel loop and hence reduction order is unspecified, the result of
    reduction may be non-deterministic unless the operation is associative and
    commutative.

    The result of the reduce operation's body must have the same type as the
    operands and associated result value of the parallel loop operation.
    Example:

    ```mlir
    %operand = constant 1.0 : f32
    scf.reduce(%operand) : f32 {
      ^bb0(%lhs : f32, %rhs: f32):
        %res = addf %lhs, %rhs : f32
        scf.reduce.return %res : f32
    }
    ```
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value operand">
  ];

  let arguments = (ins AnyType:$operand);
  let regions = (region SizedRegion<1>:$reductionOperator);
}

def ReduceReturnOp :
    SCF_Op<"reduce.return", [HasParent<"ReduceOp">, NoSideEffect,
                              Terminator]> {
  let summary = "terminator for reduce operation";
  let description = [{
    "scf.reduce.return" is a special terminator operation for the block inside
    "scf.reduce". It terminates the region. It should have the same type as
    the operand of "scf.reduce". Example for the custom format:

    ```mlir
    scf.reduce.return %res : f32
    ```
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

def YieldOp : SCF_Op<"yield", [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "scf.yield" yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "scf.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "scf.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result",
              [{ /* nothing to do */ }]>
  ];
}
#endif // MLIR_DIALECT_SCF_SCFOPS
