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

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.td"

def OpenMP_Dialect : Dialect {
  let name = "omp";
  let cppNamespace = "::mlir::omp";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let useDefaultAttributePrinterParser = 1;
}

// OmpCommon requires definition of OpenACC_Dialect.
include "mlir/Dialect/OpenMP/OmpCommon.td"

class OpenMP_Op<string mnemonic, list<Trait> traits = []> :
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

def ParallelOp : OpenMP_Op<"parallel", [
                 AutomaticAllocationScope, AttrSizedOperandSegments,
                 DeclareOpInterfaceMethods<OutlineableOpenMPOpInterface>,
                 RecursiveSideEffects, ReductionClauseInterface]> {
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

    The $allocators_vars and $allocate_vars parameters are a variadic list of values
    that specify the memory allocator to be used to obtain storage for private values.

    Reductions can be performed in a parallel construct by specifying reduction
    accumulator variables in `reduction_vars` and symbols referring to reduction
    declarations in the `reductions` attribute. Each reduction is identified
    by the accumulator it uses and accumulators must not be repeated in the same
    reduction. The `omp.reduction` operation accepts the accumulator and a
    partial value which is considered to be produced by the thread for the
    given reduction. If multiple values are produced for the same accumulator,
    i.e. there are multiple `omp.reduction`s, the last value is taken. The
    reduction declaration specifies how to combine the values from each thread
    into the final value, which is available in the accumulator after all the
    threads complete.

    The optional $proc_bind_val attribute controls the thread affinity for the execution
    of the parallel region.
  }];

  let arguments = (ins Optional<AnyType>:$if_expr_var,
             Optional<AnyType>:$num_threads_var,
             Variadic<AnyType>:$allocate_vars,
             Variadic<AnyType>:$allocators_vars,
             Variadic<OpenMP_PointerLikeType>:$reduction_vars,
             OptionalAttr<SymbolRefArrayAttr>:$reductions,
             OptionalAttr<ProcBindKindAttr>:$proc_bind_val);

  let regions = (region AnyRegion:$region);

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];
  let assemblyFormat = [{
    oilist( `reduction` `(`
              custom<ReductionVarList>(
                $reduction_vars, type($reduction_vars), $reductions
              ) `)`
          | `if` `(` $if_expr_var `:` type($if_expr_var) `)`
          | `num_threads` `(` $num_threads_var `:` type($num_threads_var) `)`
          | `allocate` `(`
              custom<AllocateAndAllocator>(
                $allocate_vars, type($allocate_vars),
                $allocators_vars, type($allocators_vars)
              ) `)`
          | `proc_bind` `(` custom<ClauseAttr>($proc_bind_val) `)`
    ) $region attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // TODO: remove this once emitAccessorPrefix is set to
    // kEmitAccessorPrefix_Prefixed for the dialect.
    /// Returns the reduction variables
    operand_range getReductionVars() { return reduction_vars(); }
  }];
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

def OMP_ScheduleModNone         : I32EnumAttrCase<"none", 0>;
def OMP_ScheduleModMonotonic    : I32EnumAttrCase<"monotonic", 1>;
def OMP_ScheduleModNonmonotonic : I32EnumAttrCase<"nonmonotonic", 2>;
// FIXME: remove this value for the modifier because this is handled using a
// separate attribute
def OMP_ScheduleModSIMD         : I32EnumAttrCase<"simd", 3>;

def ScheduleModifier
    : I32EnumAttr<"ScheduleModifier", "OpenMP Schedule Modifier",
                  [OMP_ScheduleModNone, OMP_ScheduleModMonotonic,
                   OMP_ScheduleModNonmonotonic, OMP_ScheduleModSIMD]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::omp";
}
def ScheduleModifierAttr : EnumAttr<OpenMP_Dialect, ScheduleModifier,
                                    "sched_mod">;

//===----------------------------------------------------------------------===//
// 2.8.1 Sections Construct
//===----------------------------------------------------------------------===//

def SectionOp : OpenMP_Op<"section", [HasParent<"SectionsOp">]> {
  let summary = "section directive";
  let description = [{
    A section operation encloses a region which represents one section in a
    sections construct. A section op should always be surrounded by an
    `omp.sections` operation.
  }];
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$region attr-dict";
}

def SectionsOp : OpenMP_Op<"sections", [AttrSizedOperandSegments,
                           ReductionClauseInterface]> {
  let summary = "sections construct";
  let description = [{
    The sections construct is a non-iterative worksharing construct that
    contains `omp.section` operations. The `omp.section` operations are to be
    distributed among and executed by the threads in a team. Each `omp.section`
    is executed once by one of the threads in the team in the context of its
    implicit task.

    Reductions can be performed in a sections construct by specifying reduction
    accumulator variables in `reduction_vars` and symbols referring to reduction
    declarations in the `reductions` attribute. Each reduction is identified
    by the accumulator it uses and accumulators must not be repeated in the same
    reduction. The `omp.reduction` operation accepts the accumulator and a
    partial value which is considered to be produced by the section for the
    given reduction. If multiple values are produced for the same accumulator,
    i.e. there are multiple `omp.reduction`s, the last value is taken. The
    reduction declaration specifies how to combine the values from each section
    into the final value, which is available in the accumulator after all the
    sections complete.

    The $allocators_vars and $allocate_vars parameters are a variadic list of values
    that specify the memory allocator to be used to obtain storage for private values.

    The `nowait` attribute, when present, signifies that there should be no
    implicit barrier at the end of the construct.
  }];
  let arguments = (ins Variadic<OpenMP_PointerLikeType>:$reduction_vars,
                       OptionalAttr<SymbolRefArrayAttr>:$reductions,
                       Variadic<AnyType>:$allocate_vars,
                       Variadic<AnyType>:$allocators_vars,
                       UnitAttr:$nowait);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    oilist( `reduction` `(`
              custom<ReductionVarList>(
                $reduction_vars, type($reduction_vars), $reductions
              ) `)`
          | `allocate` `(`
              custom<AllocateAndAllocator>(
                $allocate_vars, type($allocate_vars),
                $allocators_vars, type($allocators_vars)
              ) `)`
          | `nowait` $nowait
    ) $region attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    // TODO: remove this once emitAccessorPrefix is set to
    // kEmitAccessorPrefix_Prefixed for the dialect.
    /// Returns the reduction variables
    operand_range getReductionVars() { return reduction_vars(); }
  }];
}

//===----------------------------------------------------------------------===//
// 2.8.2 Single Construct
//===----------------------------------------------------------------------===//

def SingleOp : OpenMP_Op<"single", [AttrSizedOperandSegments]> {
  let summary = "single directive";
  let description = [{
    The single construct specifies that the associated structured block is 
    executed by only one of the threads in the team (not necessarily the
    master thread), in the context of its implicit task. The other threads
    in the team, which do not execute the block, wait at an implicit barrier
    at the end of the single construct unless a nowait clause is specified.
  }];

  let arguments = (ins Variadic<AnyType>:$allocate_vars,
                       Variadic<AnyType>:$allocators_vars,
                       UnitAttr:$nowait);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    oilist(`allocate` `(`
              custom<AllocateAndAllocator>(
                $allocate_vars, type($allocate_vars),
                $allocators_vars, type($allocators_vars)
              ) `)`
          |`nowait` $nowait
    ) $region attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.9.2 Workshare Loop Construct
//===----------------------------------------------------------------------===//

def WsLoopOp : OpenMP_Op<"wsloop", [AttrSizedOperandSegments,
                         AllTypesMatch<["lowerBound", "upperBound", "step"]>,
                         RecursiveSideEffects, ReductionClauseInterface]> {
  let summary = "worksharing-loop construct";
  let description = [{
    The worksharing-loop construct specifies that the iterations of the loop(s)
    will be executed in parallel by threads in the current context. These
    iterations are spread across threads that already exist in the enclosing
    parallel region. The lower and upper bounds specify a half-open range: the
    range includes the lower bound but does not include the upper bound. If the
    `inclusive` attribute is specified then the upper bound is also included.

    The body region can contain any number of blocks. The region is terminated
    by "omp.yield" instruction without operands.

    ```
    omp.wsloop <clauses>
    for (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %a = load %arrA[%i1, %i2] : memref<?x?xf32>
      %b = load %arrB[%i1, %i2] : memref<?x?xf32>
      %sum = arith.addf %a, %b : f32
      store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
      omp.yield
    }
    ```

    The `linear_step_vars` operand additionally specifies the step for each
    associated linear operand. Note that the `linear_vars` and
    `linear_step_vars` variadic lists should contain the same number of
    elements.

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
    with the worksharing-loop construct. The value of zero refers to the ordered
    clause specified without parameter.

    The optional `order` attribute specifies which order the iterations of the
    associate loops are executed in. Currently the only option for this
    attribute is "concurrent".
  }];

  let arguments = (ins Variadic<IntLikeType>:$lowerBound,
             Variadic<IntLikeType>:$upperBound,
             Variadic<IntLikeType>:$step,
             Variadic<AnyType>:$linear_vars,
             Variadic<I32>:$linear_step_vars,
             Variadic<OpenMP_PointerLikeType>:$reduction_vars,
             OptionalAttr<SymbolRefArrayAttr>:$reductions,
             OptionalAttr<ScheduleKindAttr>:$schedule_val,
             Optional<AnyType>:$schedule_chunk_var,
             OptionalAttr<ScheduleModifierAttr>:$schedule_modifier,
             UnitAttr:$simd_modifier,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$collapse_val,
             UnitAttr:$nowait,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$ordered_val,
             OptionalAttr<OrderKindAttr>:$order_val,
             UnitAttr:$inclusive);

  let builders = [
    OpBuilder<(ins "ValueRange":$lowerBound, "ValueRange":$upperBound,
               "ValueRange":$step,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// Returns the number of loops in the workshape loop nest.
    unsigned getNumLoops() { return lowerBound().size(); }

    /// Returns the number of reduction variables.
    unsigned getNumReductionVars() { return reduction_vars().size(); }

    // TODO: remove this once emitAccessorPrefix is set to
    // kEmitAccessorPrefix_Prefixed for the dialect.
    /// Returns the reduction variables
    operand_range getReductionVars() { return reduction_vars(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    oilist(`linear` `(`
              custom<LinearClause>($linear_vars, type($linear_vars),
                                   $linear_step_vars) `)`
          |`schedule` `(`
              custom<ScheduleClause>(
                $schedule_val, $schedule_modifier, $simd_modifier,
                $schedule_chunk_var, type($schedule_chunk_var)) `)`
          |`collapse` `(` $collapse_val `)`
          |`nowait` $nowait
          |`ordered` `(` $ordered_val `)`
          |`order` `(` custom<ClauseAttr>($order_val) `)`
          |`reduction` `(`
              custom<ReductionVarList>(
                $reduction_vars, type($reduction_vars), $reductions
              ) `)`
    ) `for` custom<WsLoopControl>($region, $lowerBound, $upperBound, $step,
                                  type($step), $inclusive) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Simd construct [2.9.3.1] 
//===----------------------------------------------------------------------===//

def SimdLoopOp : OpenMP_Op<"simdloop", [AttrSizedOperandSegments,
                         AllTypesMatch<["lowerBound", "upperBound", "step"]>]> {
 let summary = "simd loop construct";
  let description = [{
    The simd construct can be applied to a loop to indicate that the loop can be 
    transformed into a SIMD loop (that is, multiple iterations of the loop can 
    be executed concurrently using SIMD instructions).. The lower and upper 
    bounds specify a half-open range: the range includes the lower bound but 
    does not include the upper bound.

    The body region can contain any number of blocks. The region is terminated
    by "omp.yield" instruction without operands.
    ```
    omp.simdloop (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) 
                                      step (%c1, %c1) {
      // block operations
      omp.yield
    }
    ``` 
  }];

  // TODO: Add other clauses
  let arguments = (ins Variadic<IntLikeType>:$lowerBound,
             Variadic<IntLikeType>:$upperBound,
             Variadic<IntLikeType>:$step);
 
  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// Returns the number of loops in the simd loop nest.
    unsigned getNumLoops() { return lowerBound().size(); }

  }];
 
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1; 
}


def YieldOp : OpenMP_Op<"yield",
    [NoSideEffect, ReturnLike, Terminator,
     ParentOneOf<["WsLoopOp", "ReductionDeclareOp", 
     "AtomicUpdateOp", "SimdLoopOp"]>]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "omp.yield" yields SSA values from the OpenMP dialect op region and
    terminates the region. The semantics of how the values are yielded is
    defined by the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  let assemblyFormat = [{ ( `(` $results^ `:` type($results) `)` )? attr-dict}];
}

//===----------------------------------------------------------------------===//
// 2.10.1 task Construct
//===----------------------------------------------------------------------===//

def TaskOp : OpenMP_Op<"task", [AttrSizedOperandSegments,
                       OutlineableOpenMPOpInterface, AutomaticAllocationScope,
                       ReductionClauseInterface]> {
  let summary = "task construct";
  let description = [{
    The task construct defines an explicit task.

    For definitions of "undeferred task", "included task", "final task" and
    "mergeable task", please check OpenMP Specification.

    When an `if` clause is present on a task construct, and the value of
    `if_expr` evaluates to `false`, an "undeferred task" is generated, and the
    encountering thread must suspend the current task region, for which
    execution cannot be resumed until execution of the structured block that is
    associated with the generated task is completed.

    When a `final` clause is present on a task construct and the `final_expr`
    evaluates to `true`, the generated task will be a "final task". All task
    constructs encountered during execution of a final task will generate final
    and included tasks.

    If the `untied` clause is present on a task construct, any thread in the
    team can resume the task region after a suspension. The `untied` clause is
    ignored if a `final` clause is present on the same task construct and the
    `final_expr` evaluates to `true`, or if a task is an included task.

    When the `mergeable` clause is present on a task construct, the generated
    task is a "mergeable task".

    The `in_reduction` clause specifies that this particular task (among all the
    tasks in current taskgroup, if any) participates in a reduction.

    The `priority` clause is a hint for the priority of the generated task.
    The `priority` is a non-negative integer expression that provides a hint for
    task execution order. Among all tasks ready to be executed, higher priority
    tasks (those with a higher numerical value in the priority clause
    expression) are recommended to execute before lower priority ones. The
    default priority-value when no priority clause is specified should be
    assumed to be zero (the lowest priority).

    The `allocators_vars` and `allocate_vars` parameters are a variadic list of
    values that specify the memory allocator to be used to obtain storage for
    private values.

  }];

  // TODO: depend, affinity and detach clauses
  let arguments = (ins Optional<I1>:$if_expr,
                       Optional<I1>:$final_expr,
                       UnitAttr:$untied,
                       UnitAttr:$mergeable,
                       Variadic<OpenMP_PointerLikeType>:$in_reduction_vars,
                       OptionalAttr<SymbolRefArrayAttr>:$in_reductions,
                       Optional<I32>:$priority,
                       Variadic<AnyType>:$allocate_vars,
                       Variadic<AnyType>:$allocators_vars);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    oilist(`if` `(` $if_expr `)`
          |`final` `(` $final_expr `)`
          |`untied` $untied
          |`mergeable` $mergeable
          |`in_reduction` `(`
              custom<ReductionVarList>(
                $in_reduction_vars, type($in_reduction_vars), $in_reductions
              ) `)`
          |`priority` `(` $priority `)`
          |`allocate` `(`
              custom<AllocateAndAllocator>(
                $allocate_vars, type($allocate_vars),
                $allocators_vars, type($allocators_vars)
              ) `)`
    ) $region attr-dict
  }];
  let extraClassDeclaration = [{
    /// Returns the reduction variables
    operand_range getReductionVars() { return in_reduction_vars(); }
  }];
  let hasVerifier = 1;
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

    TODO:  map, is_device_ptr, depend, defaultmap, in_reduction

  }];

  let arguments = (ins Optional<I1>:$if_expr,
                       Optional<AnyInteger>:$device,
                       Optional<AnyInteger>:$thread_limit,
                       UnitAttr:$nowait);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist( `if` `(` $if_expr `)`
          | `device` `(` $device `:` type($device) `)`
          | `thread_limit` `(` $thread_limit `:` type($thread_limit) `)`
          | `nowait` $nowait
    ) $region attr-dict
  }];
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

  let arguments = (ins SymbolNameAttr:$sym_name,
                       DefaultValuedAttr<I64Attr, "0">:$hint_val);

  let assemblyFormat = [{
    $sym_name oilist(`hint` `(` custom<SynchronizationHint>($hint_val) `)`)
    attr-dict
  }];
  let hasVerifier = 1;
}


def CriticalOp : OpenMP_Op<"critical",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "critical construct";
  let description = [{
    The critical construct imposes a restriction on the associated structured
    block (region) to be executed by only a single thread at a time.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$name);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    (`(` $name^ `)`)? $region attr-dict
  }];
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
// [5.1] 2.19.9 ordered Construct
//===----------------------------------------------------------------------===//

def ClauseDependSource : I32EnumAttrCase<"dependsource", 0>;
def ClauseDependSink   : I32EnumAttrCase<"dependsink",   1>;

def ClauseDepend : I32EnumAttr<
    "ClauseDepend",
    "depend clause",
    [ClauseDependSource, ClauseDependSink]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::omp";
}
def ClauseDependAttr : EnumAttr<OpenMP_Dialect, ClauseDepend, "clause_depend"> {
  let assemblyFormat = "`(` $value `)`";
}

def OrderedOp : OpenMP_Op<"ordered"> {
  let summary = "ordered construct without region";
  let description = [{
    The ordered construct without region is a stand-alone directive that
    specifies cross-iteration dependences in a doacross loop nest.

    The `depend_type_val` attribute refers to either the DEPEND(SOURCE) clause
    or the DEPEND(SINK: vec) clause.

    The `num_loops_val` attribute specifies the number of loops in the doacross
    nest.

    The `depend_vec_vars` is a variadic list of operands that specifies the index
    of the loop iterator in the doacross nest for the DEPEND(SOURCE) clause or
    the index of the element of "vec" for the DEPEND(SINK: vec) clause. It
    contains the operands in multiple "vec" when multiple DEPEND(SINK: vec)
    clauses exist in one ORDERED directive.
  }];

  let arguments = (ins OptionalAttr<ClauseDependAttr>:$depend_type_val,
             Confined<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$num_loops_val,
             Variadic<AnyType>:$depend_vec_vars);

  let assemblyFormat = [{
    ( `depend_type` `` $depend_type_val^ )?
    ( `depend_vec` `(` $depend_vec_vars^ `:` type($depend_vec_vars) `)` )?
    attr-dict
  }];
  let hasVerifier = 1;
}

def OrderedRegionOp : OpenMP_Op<"ordered_region"> {
  let summary = "ordered construct with region";
  let description = [{
    The ordered construct with region specifies a structured block in a
    worksharing-loop, SIMD, or worksharing-loop SIMD region that is executed in
    the order of the loop iterations.

    The `simd` attribute corresponds to the SIMD clause specified. If it is not
    present, it behaves as if the THREADS clause is specified or no clause is
    specified.
  }];

  let arguments = (ins UnitAttr:$simd);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{ ( `simd` $simd^ )? $region attr-dict}];
  let hasVerifier = 1;
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
// 2.17.7 atomic construct
//===----------------------------------------------------------------------===//

// In the OpenMP Specification, atomic construct has an `atomic-clause` which
// can take the values `read`, `write`, `update` and `capture`. These four
// kinds of atomic constructs are fundamentally independent and are handled
// separately while lowering. Having four separate operations (one for each
// value of the clause) here decomposes handling of this construct into a
// two-step process.

def AtomicReadOp : OpenMP_Op<"atomic.read", [AllTypesMatch<["x", "v"]>]> {

  let summary = "performs an atomic read";

  let description = [{
    This operation performs an atomic read.

    The operand `x` is the address from where the value is atomically read.
    The operand `v` is the address where the value is stored after reading.

    `hint` is the value of hint (as specified in the hint clause). It is a
    compile time constant. As the name suggests, this is just a hint for
    optimization.

    `memory_order` indicates the memory ordering behavior of the construct. It
    can be one of `seq_cst`, `acquire` or `relaxed`.
  }];

  let arguments = (ins OpenMP_PointerLikeType:$x,
                       OpenMP_PointerLikeType:$v,
                       DefaultValuedAttr<I64Attr, "0">:$hint_val,
                       OptionalAttr<MemoryOrderKindAttr>:$memory_order_val);
  let assemblyFormat = [{
    $v `=` $x 
    oilist( `memory_order` `(` custom<ClauseAttr>($memory_order_val) `)`
          | `hint` `(` custom<SynchronizationHint>($hint_val) `)`)
    `:` type($x) attr-dict
  }];
  let hasVerifier = 1;
}

def AtomicWriteOp : OpenMP_Op<"atomic.write"> {

  let summary = "performs an atomic write";

  let description = [{
    This operation performs an atomic write.

    The operand `address` is the address to where the `value` is atomically
    written w.r.t. multiple threads. The evaluation of `value` need not be
    atomic w.r.t. the write to address. In general, the type(address) must
    dereference to type(value).

    `hint` is the value of hint (as specified in the hint clause). It is a
    compile time constant. As the name suggests, this is just a hint for
    optimization.

    `memory_order` indicates the memory ordering behavior of the construct. It
    can be one of `seq_cst`, `release` or `relaxed`.
  }];

  let arguments = (ins OpenMP_PointerLikeType:$address,
                       AnyType:$value,
                       DefaultValuedAttr<I64Attr, "0">:$hint_val,
                       OptionalAttr<MemoryOrderKindAttr>:$memory_order_val);
  let assemblyFormat = [{
    $address `=` $value
    oilist( `hint` `(` custom<SynchronizationHint>($hint_val) `)`
          | `memory_order` `(` custom<ClauseAttr>($memory_order_val) `)`)
    `:` type($address) `,` type($value)
    attr-dict
  }];
  let hasVerifier = 1;
}

def AtomicUpdateOp : OpenMP_Op<"atomic.update",
                               [SingleBlockImplicitTerminator<"YieldOp">]> {

  let summary = "performs an atomic update";

  let description = [{
    This operation performs an atomic update.

    The operand `x` is exactly the same as the operand `x` in the OpenMP
    Standard (OpenMP 5.0, section 2.17.7). It is the address of the variable
    that is being updated. `x` is atomically read/written.

    `hint` is the value of hint (as used in the hint clause). It is a compile
    time constant. As the name suggests, this is just a hint for optimization.

    `memory_order` indicates the memory ordering behavior of the construct. It
    can be one of `seq_cst`, `release` or `relaxed`.

    The region describes how to update the value of `x`. It takes the value at
    `x` as an input and must yield the updated value. Only the update to `x` is
    atomic. Generally the region must have only one instruction, but can
    potentially have more than one instructions too. The update is sematically
    similar to a compare-exchange loop based atomic update.

    The syntax of atomic update operation is different from atomic read and
    atomic write operations. This is because only the host dialect knows how to
    appropriately update a value. For example, while generating LLVM IR, if
    there are no special `atomicrmw` instructions for the operation-type
    combination in atomic update, a compare-exchange loop is generated, where
    the core update operation is directly translated like regular operations by
    the host dialect. The front-end must handle semantic checks for allowed
    operations.
  }];

  let arguments = (ins OpenMP_PointerLikeType:$x,
                       DefaultValuedAttr<I64Attr, "0">:$hint_val,
                       OptionalAttr<MemoryOrderKindAttr>:$memory_order_val);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    oilist( `memory_order` `(` custom<ClauseAttr>($memory_order_val) `)`
          | `hint` `(` custom<SynchronizationHint>($hint_val) `)`)
    $x `:` type($x) $region attr-dict
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    Operation* getFirstOp() {
      return &getRegion().front().getOperations().front();
    }
  }];
}

def AtomicCaptureOp : OpenMP_Op<"atomic.capture",
    [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary = "performs an atomic capture";
  let description = [{
    This operation performs an atomic capture.

    `hint` is the value of hint (as used in the hint clause). It is a compile
    time constant. As the name suggests, this is just a hint for optimization.

    `memory_order` indicates the memory ordering behavior of the construct. It
    can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.

    The region has the following allowed forms:

    ```
      omp.atomic.capture {
        omp.atomic.update ...
        omp.atomic.read ...
        omp.terminator
      }

      omp.atomic.capture {
        omp.atomic.read ...
        omp.atomic.update ...
        omp.terminator
      }

      omp.atomic.capture {
        omp.atomic.read ...
        omp.atomic.write ...
        omp.terminator
      }
    ```

  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "0">:$hint_val,
                       OptionalAttr<MemoryOrderKindAttr>:$memory_order_val);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    oilist(`memory_order` `(` custom<ClauseAttr>($memory_order_val) `)`
          |`hint` `(` custom<SynchronizationHint>($hint_val) `)`)
    $region attr-dict
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Returns the first operation in atomic capture region
    Operation* getFirstOp();

    /// Returns the second operation in atomic capture region
    Operation* getSecondOp();

    /// Returns the `atomic.read` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicReadOp getAtomicReadOp();

    /// Returns the `atomic.write` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicWriteOp getAtomicWriteOp();

    /// Returns the `atomic.update` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicUpdateOp getAtomicUpdateOp();
  }];
}

//===----------------------------------------------------------------------===//
// [5.1] 2.21.2 threadprivate Directive
//===----------------------------------------------------------------------===//

def ThreadprivateOp : OpenMP_Op<"threadprivate"> {
  let summary = "threadprivate directive";
  let description = [{
    The threadprivate directive specifies that variables are replicated, with
    each thread having its own copy.

    The current implementation uses the OpenMP runtime to provide thread-local
    storage (TLS). Using the TLS feature of the LLVM IR will be supported in
    future.

    This operation takes in the address of a symbol that represents the original
    variable and returns the address of its TLS. All occurrences of
    threadprivate variables in a parallel region should use the TLS returned by
    this operation.

    The `sym_addr` refers to the address of the symbol, which is a pointer to
    the original variable.
  }];

  let arguments = (ins OpenMP_PointerLikeType:$sym_addr);
  let results = (outs OpenMP_PointerLikeType:$tls_addr);
  let assemblyFormat = [{
    $sym_addr `:` type($sym_addr) `->` type($tls_addr) attr-dict
  }];
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

    For initializer and reduction regions, the operand to `omp.yield` must
    match the parent operation's results.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type);

  let regions = (region AnyRegion:$initializerRegion,
                        AnyRegion:$reductionRegion,
                        AnyRegion:$atomicReductionRegion);

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
  let hasRegionVerifier = 1;
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
  let hasVerifier = 1;
}

#endif // OPENMP_OPS
