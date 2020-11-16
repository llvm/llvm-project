//===-- OpenMPOps.td - OpenMP dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//


#ifndef OPENMP_OPS
#define OPENMP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/OpenMP/OmpCommon.td"

def OpenMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "::mlir::omp";
}

class OpenMP_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<OpenMP_Dialect, mnemonic, traits>;

// Type which can be constraint accepting standard integers, indices and
// LLVM integer types.
def IntLikeType : AnyTypeOf<[AnyInteger, Index, LLVM_AnyInteger]>;

//===----------------------------------------------------------------------===//
// 2.6 parallel Construct
//===----------------------------------------------------------------------===//

// Possible values for the default clause
def ClauseDefaultPrivate : StrEnumAttrCase<"defprivate">;
def ClauseDefaultFirstPrivate : StrEnumAttrCase<"deffirstprivate">;
def ClauseDefaultShared : StrEnumAttrCase<"defshared">;
def ClauseDefaultNone : StrEnumAttrCase<"defnone">;

def ClauseDefault : StrEnumAttr<
    "ClauseDefault",
    "default clause",
    [ClauseDefaultPrivate, ClauseDefaultFirstPrivate, ClauseDefaultShared,
     ClauseDefaultNone]> {
  let cppNamespace = "::mlir::omp";
}

def ParallelOp : OpenMP_Op<"parallel", [AttrSizedOperandSegments]> {
  let summary = "parallel construct";
  let description = [{
    The parallel construct includes a region of code which is to be executed
    by a team of threads.

    The optional $if_expr_var parameter specifies a boolean result of a
    conditional check. If this value is 1 or is not provided then the parallel
    region runs as normal, if it is 0 then the parallel region is executed with
    one thread.

    The optional $num_threads_var parameter specifies the number of threads which
    should be used to execute the parallel region.

    The optional $default_val attribute specifies the default data sharing attribute
    of values used in the parallel region that are not passed explicitly as parameters
    to the operation.

    The $private_vars, $firstprivate_vars, $shared_vars and $copyin_vars parameters
    are a variadic list of values that specify the data sharing attribute of
    those values.

    The $allocators_vars and $allocate_vars parameters are a variadic list of values
    that specify the memory allocator to be used to obtain storage for private values.

    The optional $proc_bind_val attribute controls the thread affinity for the execution
    of the parallel region.
  }];

  let arguments = (ins Optional<AnyType>:$if_expr_var,
             Optional<AnyType>:$num_threads_var,
             OptionalAttr<ClauseDefault>:$default_val,
             Variadic<AnyType>:$private_vars,
             Variadic<AnyType>:$firstprivate_vars,
             Variadic<AnyType>:$shared_vars,
             Variadic<AnyType>:$copyin_vars,
             Variadic<AnyType>:$allocate_vars,
             Variadic<AnyType>:$allocators_vars,
             OptionalAttr<ProcBindKind>:$proc_bind_val);

  let regions = (region AnyRegion:$region);

  let parser = [{ return parseParallelOp(parser, result); }];
  let printer = [{ return printParallelOp(p, *this); }];
  let verifier = [{ return ::verifyParallelOp(*this); }];
}

def TerminatorOp : OpenMP_Op<"terminator", [Terminator]> {
  let summary = "terminator for OpenMP regions";
  let description = [{
    A terminator operation for regions that appear in the body of OpenMP
    operation.  These regions are not expected to return any value so the
    terminator takes no operands. The terminator op returns control to the
    enclosing op.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.9.2 Workshare Loop Construct
//===----------------------------------------------------------------------===//

def WsLoopOp : OpenMP_Op<"wsloop", [AttrSizedOperandSegments]> {
  let summary = "workshare loop construct";
  let description = [{
    The workshare loop construct specifies that the iterations of the loop(s)
    will be executed in parallel by threads in the current context. These
    iterations are spread across threads that already exist in the enclosing
    parallel region.

    The body region can contain any number of blocks. The region is terminated
    by "omp.yield" instruction without operands.

    ```
      omp.wsloop (%i1, %i2) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
        %a = load %arrA[%i1, %i2] : memref<?x?xf32>
        %b = load %arrB[%i1, %i2] : memref<?x?xf32>
        %sum = addf %a, %b : f32
        store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
        omp.yield
      }
    ```

    `private_vars`, `firstprivate_vars`, `lastprivate_vars` and `linear_vars`
    arguments are variadic list of operands that specify the data sharing
    attributes of the list of values. The `linear_step_vars` operand
    additionally specifies the step for each associated linear operand. Note
    that the `linear_vars` and `linear_step_vars` variadic lists should contain
    the same number of elements.

    The optional `schedule_val` attribute specifies the loop schedule for this
    loop, determining how the loop is distributed across the parallel threads.
    The optional `schedule_chunk_var` associated with this determines further
    controls this distribution.

    The optional `collapse_val` attribute specifies the number of loops which
    are collapsed to form the worksharing loop.

    The `nowait` attribute, when present, signifies that there should be no
    implicit barrier at the end of the loop.

    The optional `ordered_val` attribute specifies how many loops are associated
    with the do loop construct.

    The optional `order` attribute specifies which order the iterations of the
    associate loops are executed in. Currently the only option for this
    attribute is "concurrent".
  }];

  let arguments = (ins Variadic<IntLikeType>:$lowerBound,
             Variadic<IntLikeType>:$upperBound,
             Variadic<IntLikeType>:$step,
             Variadic<AnyType>:$private_vars,
             Variadic<AnyType>:$firstprivate_vars,
             Variadic<AnyType>:$lastprivate_vars,
             Variadic<AnyType>:$linear_vars,
             Variadic<AnyType>:$linear_step_vars,
             OptionalAttr<ScheduleKind>:$schedule_val,
             Optional<AnyType>:$schedule_chunk_var,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$collapse_val,
             OptionalAttr<UnitAttr>:$nowait,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$ordered_val,
             OptionalAttr<OrderKind>:$order_val);

  let regions = (region AnyRegion:$region);
}

def YieldOp : OpenMP_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                                  HasParent<"WsLoopOp">]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "omp.yield" yields SSA values from the OpenMP dialect op region and
    terminates the region. The semantics of how the values are yielded is
    defined by the parent operation.
    If "omp.yield" has any operands, the operands must match the parent
    operation's results.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{ ( `(` $results^ `:` type($results) `)` )? attr-dict}];
}

//===----------------------------------------------------------------------===//
// 2.10.4 taskyield Construct
//===----------------------------------------------------------------------===//

def TaskyieldOp : OpenMP_Op<"taskyield"> {
  let summary = "taskyield construct";
  let description = [{
    The taskyield construct specifies that the current task can be suspended
    in favor of execution of a different task.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.13.7 flush Construct
//===----------------------------------------------------------------------===//
def FlushOp : OpenMP_Op<"flush"> {
  let summary = "flush construct";
  let description = [{
    The flush construct executes the OpenMP flush operation. This operation
    makes a thread’s temporary view of memory consistent with memory and
    enforces an order on the memory operations of the variables explicitly
    specified or implied.
  }];

  let arguments = (ins Variadic<AnyType>:$varList);

  let assemblyFormat = [{ ( `(` $varList^ `:` type($varList) `)` )? attr-dict}];
}

//===----------------------------------------------------------------------===//
// 2.16 master Construct
//===----------------------------------------------------------------------===//
def MasterOp : OpenMP_Op<"master"> {
  let summary = "master construct";
  let description = [{
    The master construct specifies a structured block that is executed by
    the master thread of the team.
  }];

  let regions = (region AnyRegion:$region);

  let assemblyFormat = "$region attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.17.2 barrier Construct
//===----------------------------------------------------------------------===//

def BarrierOp : OpenMP_Op<"barrier"> {
  let summary = "barrier construct";
  let description = [{
    The barrier construct specifies an explicit barrier at the point at which
    the construct appears.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.17.5 taskwait Construct
//===----------------------------------------------------------------------===//

def TaskwaitOp : OpenMP_Op<"taskwait"> {
  let summary = "taskwait construct";
  let description = [{
    The taskwait construct specifies a wait on the completion of child tasks
    of the current task.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // OPENMP_OPS
