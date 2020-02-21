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
       SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "for operation";
  let description = [{
    The "loop.for" operation represents a loop taking 3 SSA value as
    operands that represent the lower bound, upper bound and step respectively.
    The operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "loop.yield".  Calling ForOp::build will create such a region and insert
    the terminator implicitly if none is defined, so will the parsing even 
    in cases when it is absent from the custom format. For example:

    ```mlir
       loop.for %iv = %lb to %ub step %step {
         ... // body
       }
    ```

    "loop.for" can also operate on loop-carried variables and returns the final values
    after loop termination. The initial values of the variables are passed as additional SSA 
    operands to the "loop.for" following the 3 loop control SSA values mentioned above
    (lower bound, upper bound and step). The operation region has equivalent arguments 
    for each variable representing the value of the variable at the current iteration.
    
    The region must terminate with a "loop.yield" that passes all the current iteration 
    variables to the next iteration, or to the "loop.for" result, if at the last iteration.
    "loop.for" results hold the final values after the last iteration. 
    
    For example, to sum-reduce a memref:
    
    ```mlir
    func @reduce(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index) -> (f32) {
      // Initial sum set to 0.
      %sum_0 = constant 0.0 : f32
      // iter_args binds initial values to the loop's region arguments.
      %sum = loop.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (f32) {
	      %t = load %buffer[%iv] : memref<1024xf32>
        %sum_next = addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum if final iteration.
        loop.yield %sum_next : f32
      }
      return %sum : f32
    }   
    ```

    If the "loop.for" defines any values, a yield must be explicitly present. 
    The number and types of the "loop.for" results must match the initial values 
    in the "iter_args" binding and the yield operands.
    
    Another example with a nested "loop.if" (see "loop.if" for details) 
    to perform conditional reduction:

    ```mlir
    func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index) -> (f32) {
      %sum_0 = constant 0.0 : f32
      %c0 = constant 0.0 : f32
      %sum = loop.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (f32) {
    	  %t = load %buffer[%iv] : memref<1024xf32>
    	  %cond = cmpf "ugt", %t, %c0 : f32
    	  %sum_next = loop.if %cond -> (f32) {
	        %new_sum = addf %sum_iter, %t : f32
          loop.yield %new_sum : f32
	      } else {
      		loop.yield %sum_iter : f32
	      }
        loop.yield %sum_next : f32
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
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value lowerBound, Value upperBound, Value step">
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Value getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    iterator_range<Block::args_iterator> getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }
    iterator_range<Operation::operand_iterator> getIterOperands() {
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
    constexpr unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() { 
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }
  }];
}

def IfOp : Loop_Op<"if",
      [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "if-then-else operation";
  let description = [{
    The "loop.if" operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
       loop.if %b  {
         ...
       } else {
         ...
       }
    ```

    "loop.if" may also return results that are defined in its regions. The values
    defined are determined by which execution path is taken. 
    For example:
    ```mlir
       %x, %y = loop.if %b -> (f32, f32) {
         %x_true = ...
         %y_true = ...
         loop.yield %x_true, %y_true : f32, f32
       } else {
         %x_false = ... 
         %y_false = ... 
         loop.yield %x_false, %y_false : f32, f32
       }
    ```

    "loop.if" regions are always terminated with "loop.yield". If "loop.if"
    defines no values, the "loop.yield" can be left out, and will be
    inserted implicitly. Otherwise, it must be explicit.
    Also, if "loop.if" defines one or more values, the 'else' block cannot
    be omitted. 

    For example:
    ```mlir
       loop.if %b  {
         ...
       }
    ```
  }];
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
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
    [SameVariadicOperandSize, SingleBlockImplicitTerminator<"YieldOp">]> {
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
    "loop.yield" without operands. Parsing ParallelOp will create such a region
    and insert the terminator when it is absent from the custom format. For example:

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

def YieldOp : Loop_Op<"yield", [Terminator]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "loop.yield" yields an SSA value from a loop dialect op region and
    terminates the regions. The semantics of how the values are yielded
    is defined by the parent operation. 
    If "loop.yield" has any operands, the operands must match the parent
    operation's results. 
    If the parent operation defines no values, then the "loop.yield" may be 
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values
    are yielded. 
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result", [{ /* nothing to do */ }]>
  ];
}
#endif // LOOP_OPS
