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

def OpenMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "omp";
}

class OpenMP_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<OpenMP_Dialect, mnemonic, traits>;


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

// Possible values for the proc_bind clause
def ClauseProcMaster : StrEnumAttrCase<"master">;
def ClauseProcClose : StrEnumAttrCase<"close">;
def ClauseProcSpread : StrEnumAttrCase<"spread">;

def ClauseProcBind : StrEnumAttr<
    "ClauseProcBind",
    "procbind clause",
    [ClauseProcMaster, ClauseProcClose, ClauseProcSpread]> {
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
    of variables used in the parallel region that are not passed explicitly as parameters
    to the operation.

    The $private_vars, $firstprivate_vars, $shared_vars and $copyin_vars parameters
    are a variadic list of variables that specify the data sharing attribute of
    those variables.

    The optional $proc_bind_val attribute controls the thread affinity for the execution
    of the parallel region.
  }];

  let arguments = (ins Optional<I1>:$if_expr_var,
             Optional<AnyInteger>:$num_threads_var,
             OptionalAttr<ClauseDefault>:$default_val,
             Variadic<AnyType>:$private_vars,
             Variadic<AnyType>:$firstprivate_vars,
             Variadic<AnyType>:$shared_vars,
             Variadic<AnyType>:$copyin_vars,
             OptionalAttr<ClauseProcBind>:$proc_bind_val);

  let regions = (region AnyRegion:$region);
}

def TerminatorOp : OpenMP_Op<"terminator", [Terminator]> {
  let summary = "terminator for OpenMP regions.";
  let description = [{
    A terminator operation for regions that appear in the body of OpenMP
    operation.  These regions are not expected to return any value so the
    terminator takes no operands. The terminator op returns control to the
    enclosing op.
  }];

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
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

  let assemblyFormat = "attr-dict ($varList^ `:` type($varList))?";
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
