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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/OpenMP/OmpCommon.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.td"

def OpenMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "::mlir::omp";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

class OpenMP_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<OpenMP_Dialect, mnemonic, traits>;

// Type which can be constraint accepting standard integers and indices.
def IntLikeType : AnyTypeOf<[AnyInteger, Index]>;

def OpenMP_PointerLikeTypeInterface : TypeInterface<"PointerLikeType"> {
  let cppNamespace = "::mlir::omp";

  let description = [{
    An interface for pointer-like types suitable to contain a value that OpenMP
    specification refers to as variable.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the pointee type.",
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getElementType"
    >,
  ];
}

def OpenMP_PointerLikeType : Type<
  CPred<"$_self.isa<::mlir::omp::PointerLikeType>()">,
  "OpenMP-compatible variable type", "::mlir::omp::PointerLikeType">;

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

def ParallelOp : OpenMP_Op<"parallel", [AttrSizedOperandSegments,
                 DeclareOpInterfaceMethods<OutlineableOpenMPOpInterface>]> {
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

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];
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

def WsLoopOp : OpenMP_Op<"wsloop", [AttrSizedOperandSegments,
                         AllTypesMatch<["lowerBound", "upperBound", "step"]>]> {
  let summary = "workshare loop construct";
  let description = [{
    The workshare loop construct specifies that the iterations of the loop(s)
    will be executed in parallel by threads in the current context. These
    iterations are spread across threads that already exist in the enclosing
    parallel region. The lower and upper bounds specify a half-open range: the
    range includes the lower bound but does not include the upper bound. If the
    `inclusive` attribute is specified then the upper bound is also included.

    The body region can contain any number of blocks. The region is terminated
    by "omp.yield" instruction without operands.

    ```
    omp.wsloop (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %a = load %arrA[%i1, %i2] : memref<?x?xf32>
      %b = load %arrB[%i1, %i2] : memref<?x?xf32>
      %sum = arith.addf %a, %b : f32
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

    Reductions can be performed in a workshare loop by specifying reduction
    accumulator variables in `reduction_vars` and symbols referring to reduction
    declarations in the `reductions` attribute. Each reduction is identified
    by the accumulator it uses and accumulators must not be repeated in the same
    reduction. The `omp.reduction` operation accepts the accumulator and a
    partial value which is considered to be produced by the current loop
    iteration for the given reduction. If multiple values are produced for the
    same accumulator, i.e. there are multiple `omp.reduction`s, the last value
    is taken. The reduction declaration specifies how to combine the values from
    each iteration into the final value, which is available in the accumulator
    after the loop completes.

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
             Variadic<OpenMP_PointerLikeType>:$reduction_vars,
             OptionalAttr<TypedArrayAttrBase<SymbolRefAttr,
                            "array of symbol references">>:$reductions,
             OptionalAttr<ScheduleKind>:$schedule_val,
             Optional<AnyType>:$schedule_chunk_var,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$collapse_val,
             UnitAttr:$nowait,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$ordered_val,
             OptionalAttr<OrderKind>:$order_val,
             UnitAttr:$inclusive);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$lowerBound, "ValueRange":$upperBound,
               "ValueRange":$step,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$lowerBound,
               "ValueRange":$upperBound, "ValueRange":$step,
               "ValueRange":$privateVars, "ValueRange":$firstprivateVars,
               "ValueRange":$lastprivate_vars, "ValueRange":$linear_vars,
               "ValueRange":$linear_step_vars, "ValueRange":$reduction_vars,
               "StringAttr":$schedule_val, "Value":$schedule_chunk_var,
               "IntegerAttr":$collapse_val, "UnitAttr":$nowait,
               "IntegerAttr":$ordered_val, "StringAttr":$order_val,
               "UnitAttr":$inclusive, CArg<"bool", "true">:$buildBody)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// Returns the number of loops in the workshape loop nest.
    unsigned getNumLoops() { return lowerBound().size(); }

    /// Returns the number of reduction variables.
    unsigned getNumReductionVars() { return reduction_vars().size(); }
  }];
  let parser = [{ return parseWsLoopOp(parser, result); }];
  let printer = [{ return printWsLoopOp(p, *this); }];
  let verifier = [{ return ::verifyWsLoopOp(*this); }];
}

def YieldOp : OpenMP_Op<"yield",
    [NoSideEffect, ReturnLike, Terminator,
     ParentOneOf<["WsLoopOp", "ReductionDeclareOp"]>]> {
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
// 2.14.5 target construct
//===----------------------------------------------------------------------===//

def TargetOp : OpenMP_Op<"target",[AttrSizedOperandSegments]> {
  let summary = "target construct";
  let description = [{
    The target construct includes a region of code which is to be executed
    on a device.

    The optional $if_expr parameter specifies a boolean result of a
    conditional check. If this value is 1 or is not provided then the target
    region runs on a device, if it is 0 then the target region is executed on the
    host device.

    The optional $device parameter specifies the device number for the target region.

    The optional $thread_limit specifies the limit on the number of threads

    The optional $nowait elliminates the implicit barrier so the parent task can make progress
    even if the target task is not yet completed.

    TODO:  private, map, is_device_ptr, firstprivate, depend, defaultmap, in_reduction

  }];

  let arguments = (ins Optional<I1>:$if_expr,
                       Optional<AnyInteger>:$device,
                       Optional<AnyInteger>:$thread_limit,
                       UnitAttr:$nowait);

  let regions = (region AnyRegion:$region);
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
// 2.17.1 critical Construct
//===----------------------------------------------------------------------===//
def CriticalDeclareOp : OpenMP_Op<"critical.declare", [Symbol]> {
  let summary = "declares a named critical section.";

  let description = [{
    Declares a named critical section.

    The name can be used in critical constructs in the dialect.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = "$sym_name attr-dict";
}


def CriticalOp : OpenMP_Op<"critical"> {
  let summary = "critical construct";
  let description = [{
    The critical construct imposes a restriction on the associated structured
    block (region) to be executed by only a single thread at a time.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$name,
                       DefaultValuedAttr<I64Attr, "0">:$hint);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    (`(` $name^ `)`)? custom<SynchronizationHint>($hint) $region attr-dict
  }];

  let verifier = "return ::verifyCriticalOp(*this);";
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

//===----------------------------------------------------------------------===//
// 2.19.5.7 declare reduction Directive
//===----------------------------------------------------------------------===//

def ReductionDeclareOp : OpenMP_Op<"reduction.declare", [Symbol]> {
  let summary = "declares a reduction kind";

  let description = [{
    Declares an OpenMP reduction kind. This requires two mandatory and one
    optional region.

      1. The initializer region specifies how to initialize the thread-local
         reduction value. This is usually the neutral element of the reduction.
         For convenience, the region has an argument that contains the value
         of the reduction accumulator at the start of the reduction. It is
         expected to `omp.yield` the new value on all control flow paths.
      2. The reduction region specifies how to combine two values into one, i.e.
         the reduction operator. It accepts the two values as arguments and is
         expected to `omp.yield` the combined value on all control flow paths.
      3. The atomic reduction region is optional and specifies how two values
         can be combined atomically given local accumulator variables. It is
         expected to store the combined value in the first accumulator variable.

    Note that the MLIR type system does not allow for type-polymorphic
    reductions. Separate reduction declarations should be created for different
    element and accumulator types.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type);

  let regions = (region AnyRegion:$initializerRegion,
                        AnyRegion:$reductionRegion,
                        AnyRegion:$atomicReductionRegion);
  let verifier = "return ::verifyReductionDeclareOp(*this);";

  let assemblyFormat = "$sym_name `:` $type attr-dict-with-keyword "
                       "`init` $initializerRegion "
                       "`combiner` $reductionRegion "
                       "custom<AtomicReductionRegion>($atomicReductionRegion)";

  let extraClassDeclaration = [{
    PointerLikeType getAccumulatorType() {
      if (atomicReductionRegion().empty())
        return {};

      return atomicReductionRegion().front().getArgument(0).getType();
    }
  }];
}

//===----------------------------------------------------------------------===//
// 2.19.5.4 reduction clause
//===----------------------------------------------------------------------===//

def ReductionOp : OpenMP_Op<"reduction", [
    TypesMatchWith<"value types matches accumulator element type",
                   "accumulator", "operand",
                 "$_self.cast<::mlir::omp::PointerLikeType>().getElementType()">
  ]> {
  let summary = "reduction construct";
  let description = [{
    Indicates the value that is produced by the current reduction-participating
    entity for a reduction requested in some ancestor. The reduction is
    identified by the accumulator, but the value of the accumulator may not be
    updated immediately.
  }];

  let arguments= (ins AnyType:$operand, OpenMP_PointerLikeType:$accumulator);
  let assemblyFormat =
    "$operand `,` $accumulator attr-dict `:` type($accumulator)";
  let verifier = "return ::verifyReductionOp(*this);";
}

#endif // OPENMP_OPS
