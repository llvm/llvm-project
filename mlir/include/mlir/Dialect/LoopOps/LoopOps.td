//===- Ops.td - Loop operation definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR loop operations.
//
//===----------------------------------------------------------------------===//

#ifndef LOOP_OPS
#define LOOP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Transforms/LoopLikeInterface.td"

def Loop_Dialect : Dialect {
  let name = "loop";
  let cppNamespace = "";
}

// Base class for Loop dialect ops.
class Loop_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Loop_Dialect, mnemonic, traits> {
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

def ForOp : Loop_Op<"for",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "for operation";
  let description = [{
    The "loop.for" operation represents a loop nest taking 3 SSA value as
    operands that represent the lower bound, upper bound and step respectively.
    The operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes the
    lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "loop.terminator".  Calling ForOp::build will create such region and insert
    the terminator, so will the parsing even in cases when it is absent from the
    custom format. For example:

    ```mlir
       loop.for %iv = %lb to %ub step %step {
         ... // body
       }
    ```
  }];
  let arguments = (ins Index:$lowerBound, Index:$upperBound, Index:$step);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value lowerBound, Value upperBound, Value step">
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Value getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }
  }];
}

def IfOp : Loop_Op<"if",
      [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "if-then-else operation";
  let description = [{
    The "loop.if" operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. The operation produces no results. For example:

    ```mlir
       loop.if %b  {
         ...
       } else {
         ...
       }
    ```

    The 'else' block is optional, and may be omitted. For
    example:

    ```mlir
       loop.if %b  {
         ...
       }
    ```
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value cond, bool withElseRegion">
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
    OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

def ParallelOp : Loop_Op<"parallel",
    [SameVariadicOperandSize, SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "parallel for operation";
  let description = [{
    The "loop.parallel" operation represents a loop nest taking 3 groups of SSA
    values as operands that represent the lower bounds, upper bounds and steps,
    respectively. The operation defines a variadic number of SSA values for its
    induction variables. It has one region capturing the loop body. The
    induction variables are represented as an argument of this region. These SSA
    values always have type index, which is the size of the machine word. The
    steps are values of type index, required to be positive.
    The lower and upper bounds specify a half-open range: the range includes the
    lower bound but does not include the upper bound.

    Semantically we require that the iteration space can be iterated in any
    order, and the loop body can be executed in parallel. If there are data
    races, the behavior is undefined.

    The parallel loop operation supports reduction of values produced by
    individual iterations into a single result. This is modeled using the
    loop.reduce operation (see loop.reduce for details). Each result of a
    loop.parallel operation is associated with a reduce operation that is an
    immediate child. Reduces are matched to result values in order of their
    appearance in the body. Consequently, we require that the body region has
    the same number of results as it has reduce operations.

    The body region must contain exactly one block that terminates with
    "loop.terminator". Parsing ParallelOp will create such region and insert the
    terminator when it is absent from the custom format. For example:

    ```mlir
       loop.parallel (%iv) = (%lb) to (%ub) step (%step) {
         %zero = constant 0.0 : f32
         loop.reduce(%zero) {
           ^bb0(%lhs : f32, %rhs: f32):
             %res = addf %lhs, %rhs : f32
             loop.reduce.return %res : f32
         } : f32
       }
    ```
   }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "ValueRange lowerBounds, ValueRange upperBounds, "
              "ValueRange steps">,
    OpBuilder<"Builder *builder, OperationState &result, "
              "ValueRange lowerBounds, ValueRange upperBounds, "
              "ValueRange steps, ArrayRef<Type> resultTypes">
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    iterator_range<Block::args_iterator> getInductionVars() {
      return {getBody()->args_begin(), getBody()->args_end()};
    }
    unsigned getNumLoops() { return step().size(); }
  }];
}

def ReduceOp : Loop_Op<"reduce", [HasParent<"ParallelOp">]> {
  let summary = "reduce operation for parallel for";
  let description = [{
    "loop.reduce" is an operation occurring inside "loop.parallel" operations.
    It consists of one block with two arguments which have the same type as the
    operand of "loop.reduce".

    "loop.reduce" is used to model the value for reduction computations of a
    "loop.parallel" operation. It has to appear as an immediate child of a
    "loop.parallel" and is associated with a result value of its parent
    operation.

    Association is in the order of appearance in the body where the first result
    of a parallel loop operation corresponds to the first "loop.reduce" in the
    operation's body region. The reduce operation takes a single operand, which
    is the value to be used in the reduction.

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
      loop.reduce(%operand) {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = addf %lhs, %rhs : f32
          loop.reduce.return %res : f32
      } : f32
    ```

  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value operand">
  ];

  let arguments = (ins AnyType:$operand);
  let regions = (region SizedRegion<1>:$reductionOperator);
}

def ReduceReturnOp :
    Loop_Op<"reduce.return", [HasParent<"ReduceOp">, Terminator]> {
  let summary = "terminator for reduce operation";
  let description = [{
    "loop.reduce.return" is a special terminator operation for the block inside
    "loop.reduce". It terminates the region. It should have the same type as the
    operand of "loop.reduce". Example for the custom format:

    ```mlir
      loop.reduce.return %res : f32
    ```
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

def TerminatorOp : Loop_Op<"terminator", [Terminator]> {
  let summary = "cf terminator operation";
  let description = [{
    "loop.terminator" is a special terminator operation for blocks inside
    loops. It terminates the region. This operation does _not_ have a custom
    syntax. However, `std` control operations omit the terminator in their
    custom syntax for brevity.

    ```mlir
       loop.terminator
    ```
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

#endif // LOOP_OPS
