//=== OpenMPClauses.td - OpenMP dialect clause definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains clause definitions for the OpenMP dialect.
//
// For each "Xyz" clause, there is an "OpenMP_XyzClauseSkip" class and an
// "OpenMP_XyzClause" definition. The latter is an instantiation of the former
// where all "skip" template parameters are set to `false` and should be the
// preferred variant to used whenever possible when defining `OpenMP_Op`
// instances.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_CLAUSES
#define OPENMP_CLAUSES

include "mlir/Dialect/OpenMP/OpenMPOpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// V5.2: [6.3] `align` clause
//===----------------------------------------------------------------------===//

class OpenMP_AlignClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$align
  );

  let optAssemblyFormat = [{
    `align` `(` $align `)`
  }];

  let description = [{
    The `align` clause is used to specify the byte alignment to use for
    allocations associated with the construct on which the clause appears.
  }];
}

def OpenMP_AlignClause : OpenMP_AlignClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.11] `aligned` clause
//===----------------------------------------------------------------------===//

class OpenMP_AlignedClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$aligned_vars,
    OptionalAttr<I64ArrayAttr>:$alignments
  );

  let optAssemblyFormat = [{
    `aligned` `(` custom<AlignedClause>($aligned_vars, type($aligned_vars),
                                        $alignments) `)`
  }];

  let description = [{
    The `alignments` attribute additionally specifies alignment of each
    corresponding aligned operand. Note that `aligned_vars` and `alignments`
    must contain the same number of elements.
  }];
}

def OpenMP_AlignedClause : OpenMP_AlignedClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [6.6] `allocate` clause
//===----------------------------------------------------------------------===//

class OpenMP_AllocateClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<AnyType>:$allocate_vars,
    Variadic<AnyType>:$allocator_vars
  );

  let extraClassDeclaration = [{
    unsigned getNumAllocateVars() { return getAllocateVars().size(); }
    unsigned getNumAllocatorsVars() { return getAllocatorVars().size(); }
  }];

  let optAssemblyFormat = [{
    `allocate` `(`
      custom<AllocateAndAllocator>($allocate_vars, type($allocate_vars),
                                   $allocator_vars, type($allocator_vars)) `)`
  }];

  let description = [{
    The `allocator_vars` and `allocate_vars` parameters are a variadic list of
    values that specify the memory allocator to be used to obtain storage for
    private values.
  }];
}

def OpenMP_AllocateClause : OpenMP_AllocateClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [6.4] `allocator` clause
//===----------------------------------------------------------------------===//

class OpenMP_AllocatorClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {

  let arguments = (ins
    OptionalAttr<AllocatorHandleAttr>:$allocator
  );

  let optAssemblyFormat = [{
    `allocator` `(` custom<ClauseAttr>($allocator) `)`
  }];

  let description = [{
    `allocator` specifies the memory allocator to be used for allocations
    associated with the construct on which the clause appears.
  }];
}

def OpenMP_AllocatorClause : OpenMP_AllocatorClauseSkip<>;

//===----------------------------------------------------------------------===//
// LLVM OpenMP extension `ompx_bare` clause
//===----------------------------------------------------------------------===//

class OpenMP_BareClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$bare
  );

  let optAssemblyFormat = [{
    `ompx_bare` $bare
  }];

  let description = [{
    `ompx_bare` allows `omp target teams` to be executed on a GPU with an
    explicit number of teams and threads. This clause also allows the teams and
    threads sizes to have up to 3 dimensions.
  }];
}

def OpenMP_BareClause : OpenMP_BareClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [16.1, 16.2] `cancel-directive-name` clause set
//===----------------------------------------------------------------------===//

class OpenMP_CancelDirectiveNameClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    CancellationConstructTypeAttr:$cancel_directive
  );

  let reqAssemblyFormat = [{
    `cancellation_construct_type` `(`
      custom<ClauseAttr>($cancel_directive) `)`
  }];

  // TODO: Add description.
}

def OpenMP_CancelDirectiveNameClause : OpenMP_CancelDirectiveNameClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [11.7.1] `bind` clause
//===----------------------------------------------------------------------===//

class OpenMP_BindClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<BindKindAttr>:$bind_kind
  );

  let optAssemblyFormat = [{
    `bind` `(` custom<ClauseAttr>($bind_kind) `)`
  }];

  let description = [{
    The `bind` clause specifies the binding region of the construct on which it
    appears.
  }];
}

def OpenMP_BindClause : OpenMP_BindClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [4.4.3] `collapse` clause
//===----------------------------------------------------------------------===//

class OpenMP_CollapseClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
      ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "1">, [IntMinValue<1>]>
      :$collapse_num_loops
  );
}

def OpenMP_CollapseClause : OpenMP_CollapseClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.7.2] `copyprivate` clause
//===----------------------------------------------------------------------===//

class OpenMP_CopyprivateClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$copyprivate_vars,
    OptionalAttr<SymbolRefArrayAttr>:$copyprivate_syms
  );

  let optAssemblyFormat = [{
    `copyprivate` `(`
      custom<Copyprivate>($copyprivate_vars, type($copyprivate_vars),
                          $copyprivate_syms) `)`
  }];

  let description = [{
    If `copyprivate` variables and functions are specified, then each thread
    variable is updated with the variable value of the thread that executed
    the single region, using the specified copy functions.
  }];
}

def OpenMP_CopyprivateClause : OpenMP_CopyprivateClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.2] `critical` `name` argument
//===----------------------------------------------------------------------===//

class OpenMP_CriticalNameClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    Symbol
  ];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );

  let reqAssemblyFormat = "$sym_name";

  let description = [{
    The `sym_name` can be used in `omp.critical` constructs in the dialect.
  }];
}

def OpenMP_CriticalNameClause : OpenMP_CriticalNameClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.9.5] `depend` clause
//===----------------------------------------------------------------------===//

class OpenMP_DependClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<TaskDependArrayAttr>:$depend_kinds,
    Variadic<OpenMP_PointerLikeType>:$depend_vars
  );

  let optAssemblyFormat = [{
    `depend` `(`
      custom<DependVarList>($depend_vars, type($depend_vars), $depend_kinds) `)`
  }];

  let description = [{
    The `depend_kinds` and `depend_vars` arguments are variadic lists of values
    that specify the dependencies of this particular task in relation to other
    tasks.
  }];
}

def OpenMP_DependClause : OpenMP_DependClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [13.2] `device` clause
//===----------------------------------------------------------------------===//

class OpenMP_DeviceClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<AnyInteger>:$device
  );

  let optAssemblyFormat = [{
    `device` `(` $device `:` type($device) `)`
  }];

  let description = [{
    The optional `device` parameter specifies the device number for the target
    region.
  }];
}

def OpenMP_DeviceClause : OpenMP_DeviceClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [11.6.1] `dist_schedule` clause
//===----------------------------------------------------------------------===//

class OpenMP_DistScheduleClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$dist_schedule_static,
    Optional<IntLikeType>:$dist_schedule_chunk_size
  );

  let optAssemblyFormat = [{
    `dist_schedule_static` $dist_schedule_static
    | `dist_schedule_chunk_size` `(` $dist_schedule_chunk_size `:`
      type($dist_schedule_chunk_size) `)`
  }];

  let description = [{
    The `dist_schedule_static` attribute specifies the schedule for this loop,
    determining how the loop is distributed across the various teams. The
    optional `dist_schedule_chunk_size` associated with this determines further
    controls this distribution.
  }];
}

def OpenMP_DistScheduleClause : OpenMP_DistScheduleClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.9.6] `doacross` clause
//===----------------------------------------------------------------------===//

class OpenMP_DoacrossClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<ClauseDependAttr>:$doacross_depend_type,
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$doacross_num_loops,
    Variadic<AnyType>:$doacross_depend_vars
  );

  let reqAssemblyFormat = [{
    ( `depend_type` `` $doacross_depend_type^ )?
    ( `depend_vec` `(` $doacross_depend_vars^ `:` type($doacross_depend_vars)
                   `)` )?
  }];

  let description = [{
    The `doacross_depend_type` attribute refers to either the DEPEND(SOURCE)
    clause or the DEPEND(SINK: vec) clause.

    The `doacross_num_loops` attribute specifies the number of loops in the
    doacross nest.

    The `doacross_depend_vars` is a variadic list of operands that specifies the
    index of the loop iterator in the doacross nest for the DEPEND(SOURCE)
    clause or the index of the element of "vec" for the DEPEND(SINK: vec)
    clause. It contains the operands in multiple "vec" when multiple
    DEPEND(SINK: vec) clauses exist in one ORDERED directive.
  }];
}

def OpenMP_DoacrossClause : OpenMP_DoacrossClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.7] `exclusive` clause
//===----------------------------------------------------------------------===//

class OpenMP_ExclusiveClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<AnyType>:$exclusive_vars
  );

  let optAssemblyFormat = [{
    `exclusive` `(` $exclusive_vars `:` type($exclusive_vars) `)`
  }];

  let extraClassDeclaration = [{
    bool hasExclusiveVars() {
      return !getExclusiveVars().empty();
    }
  }];

  let description = [{
    The exclusive clause is used on a separating directive that separates a
    structured block into two structured block sequences. If it
    is specified, the input phase excludes the preceding structured block 
    sequence and instead includes the following structured block sequence, 
    while the scan phase includes the preceding structured block sequence.

    The `exclusive_vars` is a variadic list of operands that specifies the
    scan-reduction accumulator symbols.
  }];
}

def OpenMP_ExclusiveClause : OpenMP_ExclusiveClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.5.1] `filter` clause
//===----------------------------------------------------------------------===//

class OpenMP_FilterClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<IntLikeType>:$filtered_thread_id
  );

  let optAssemblyFormat = [{
    `filter` `(` $filtered_thread_id `:` type($filtered_thread_id) `)`
  }];

  let description = [{
    If `filter` is specified, the masked construct masks the execution of
    the region to only the thread id filtered. Other threads executing the
    parallel region are not expected to execute the region specified within
    the `masked` directive. If `filter` is not specified, master thread is
    expected to execute the region enclosed within `masked` directive.
  }];
}

def OpenMP_FilterClause : OpenMP_FilterClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [12.3] `final` clause
//===----------------------------------------------------------------------===//

class OpenMP_FinalClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<I1>:$final
  );

  let optAssemblyFormat = [{
    `final` `(` $final `)`
  }];

  let description = [{
    When a `final` clause is present and the `final` clause expression evaluates
    to `true`, the generated tasks will be final tasks. All task constructs
    encountered during execution of a final task will generate final and
    included tasks. The use of a variable in a `final` clause expression causes
    an implicit reference to the variable in all enclosing constructs.
  }];
}

def OpenMP_FinalClause : OpenMP_FinalClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [12.6.1] `grainsize` clause
//===----------------------------------------------------------------------===//

class OpenMP_GrainsizeClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins OptionalAttr<GrainsizeTypeAttr>:$grainsize_mod,
      Optional<IntLikeType>:$grainsize);

  let optAssemblyFormat = [{
    `grainsize` `(` custom<GrainsizeClause>($grainsize_mod , $grainsize, type($grainsize)) `)`
  }];

  let description = [{
    If a `grainsize` clause is present, the number of logical loop iterations
    assigned to each generated task is greater than or equal to the minimum of
    the value of the grain-size expression and the number of logical loop
    iterations, but less than two times the value of the grain-size expression.
  }];
}

def OpenMP_GrainsizeClause : OpenMP_GrainsizeClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.9] `has_device_addr` clause
//===----------------------------------------------------------------------===//

class OpenMP_HasDeviceAddrClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$has_device_addr_vars
  );

  let description = [{
    The optional `has_device_addr_vars` indicates that list items already have
    device addresses, so they may be directly accessed from the target device.
    This includes array sections.
  }];
}

def OpenMP_HasDeviceAddrClause : OpenMP_HasDeviceAddrClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.7] `inclusive` clause
//===----------------------------------------------------------------------===//

class OpenMP_InclusiveClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<AnyType>:$inclusive_vars
  );

  let optAssemblyFormat = [{
    `inclusive` `(` $inclusive_vars `:` type($inclusive_vars) `)`
  }];

  let extraClassDeclaration = [{
    bool hasInclusiveVars() {
      return !getInclusiveVars().empty();
    }
  }];

  let description = [{
    The inclusive clause is used on a separating directive that separates a
    structured block into two structured block sequences. If it is specified,
    the input phase includes the preceding structured block sequence and the
    scan phase includes the following structured block sequence.

    The `inclusive_vars` is a variadic list of operands that specifies the
    scan-reduction accumulator symbols.
  }];
}

def OpenMP_InclusiveClause : OpenMP_InclusiveClauseSkip<>;


//===----------------------------------------------------------------------===//
// V5.2: [15.1.2] `hint` clause
//===----------------------------------------------------------------------===//

class OpenMP_HintClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    DefaultValuedOptionalAttr<I64Attr, "0">:$hint
  );

  let optAssemblyFormat = [{
    `hint` `(` custom<SynchronizationHint>($hint) `)`
  }];

  let description = [{
    `hint` is the value of hint (as specified in the hint clause). It is a
    compile time constant. As the name suggests, this is just a hint for
    optimization.
  }];
}

def OpenMP_HintClause : OpenMP_HintClauseSkip<>;

//===----------------------------------------------------------------------===//
// Not in the spec: Clause-like structure to hold host-evaluated values.
//===----------------------------------------------------------------------===//

class OpenMP_HostEvalClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface, IsolatedFromAbove
  ];

  let arguments = (ins
    Variadic<AnyType>:$host_eval_vars
  );

  let description = [{
    The optional `host_eval_vars` holds values defined outside of the region of
    the `IsolatedFromAbove` operation for which a corresponding entry block
    argument is defined. The only legal uses for these captured values are the
    following:
      - `num_teams` or `thread_limit` clause of an immediately nested
      `omp.teams` operation.
      - If the operation is the top-level `omp.target` of a target SPMD kernel:
        - `num_threads` clause of the nested `omp.parallel` operation.
        - Bounds and steps of the nested `omp.loop_nest` operation.
  }];
}

def OpenMP_HostEvalClause : OpenMP_HostEvalClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [3.4] `if` clause
//===----------------------------------------------------------------------===//

class OpenMP_IfClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<I1>:$if_expr
  );

  let optAssemblyFormat = [{
    `if` `(` $if_expr `)`
  }];

  // Description varies depending on the operation.
}

def OpenMP_IfClause : OpenMP_IfClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.5.10] `in_reduction` clause
//===----------------------------------------------------------------------===//

class OpenMP_InReductionClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$in_reduction_vars,
    OptionalAttr<DenseBoolArrayAttr>:$in_reduction_byref,
    OptionalAttr<SymbolRefArrayAttr>:$in_reduction_syms
  );

  // Description varies depending on the operation. Assembly format not defined
  // because this clause must be processed together with the first region of the
  // operation, as it defines entry block arguments.
}

def OpenMP_InReductionClause : OpenMP_InReductionClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.7] `is_device_ptr` clause
//===----------------------------------------------------------------------===//

class OpenMP_IsDevicePtrClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$is_device_ptr_vars
  );

  let optAssemblyFormat = [{
    `is_device_ptr` `(` $is_device_ptr_vars `:` type($is_device_ptr_vars) `)`
  }];

  let description = [{
    The optional `is_device_ptr_vars` indicates list items are device pointers.
  }];
}

def OpenMP_IsDevicePtrClause : OpenMP_IsDevicePtrClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.6] `linear` clause
//===----------------------------------------------------------------------===//

class OpenMP_LinearClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<AnyType>:$linear_vars,
    Variadic<I32>:$linear_step_vars
  );

  let optAssemblyFormat = [{
    `linear` `(`
      custom<LinearClause>($linear_vars, type($linear_vars),
                           $linear_step_vars) `)`
  }];

  let description = [{
    The `linear_step_vars` operand additionally specifies the step for each
    associated linear operand. Note that the `linear_vars` and
    `linear_step_vars` variadic lists should contain the same number of
    elements.
  }];
}

def OpenMP_LinearClause : OpenMP_LinearClauseSkip<>;

//===----------------------------------------------------------------------===//
// Not in the spec: Clause-like structure to hold loop related information.
//===----------------------------------------------------------------------===//

class OpenMP_LoopRelatedClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    AllTypesMatch<
      ["loop_lower_bounds", "loop_upper_bounds", "loop_steps"]>
  ];
  
  let arguments = (ins
    Variadic<IntLikeType>:$loop_lower_bounds,
    Variadic<IntLikeType>:$loop_upper_bounds,
    Variadic<IntLikeType>:$loop_steps,
    UnitAttr:$loop_inclusive
  );

  let extraClassDeclaration = [{
    /// Returns the number of loops in the loop nest.
    unsigned getNumLoops() { return getLoopLowerBounds().size(); }
  }];

  // Description and formatting integrated in the `omp.loop_nest` operation,
  // which is the only one currently accepting this clause.
}

def OpenMP_LoopRelatedClause : OpenMP_LoopRelatedClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.8.3] `map` clause
//===----------------------------------------------------------------------===//

class OpenMP_MapClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    // Not adding the BlockArgOpenMPOpInterface here because omp.target is the
    // only operation defining block arguments for `map` clauses.
    MapClauseOwningOpInterface
  ];

  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$map_vars
  );

  // This assembly format should only be used by operations where `map` does not
  // define entry block arguments. Otherwise, it must be printed and parsed
  // together with the corresponding region.
  let optAssemblyFormat = [{
    `map_entries` `(` $map_vars `:` type($map_vars) `)`
  }];

  let description = [{
    The optional `map_vars` maps data from the current task's data environment
    to the device data environment.
  }];
}

def OpenMP_MapClause : OpenMP_MapClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.8.1] `memory-order` clause set
//===----------------------------------------------------------------------===//

class OpenMP_MemoryOrderClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<MemoryOrderKindAttr>:$memory_order
  );

  let optAssemblyFormat = [{
    `memory_order` `(` custom<ClauseAttr>($memory_order) `)`
  }];

  let description = [{
    `memory_order` indicates the memory ordering behavior of the construct. It
    can be one of `seq_cst`, `acq_rel`, `release`, `acquire` or `relaxed`.
  }];
}

def OpenMP_MemoryOrderClause : OpenMP_MemoryOrderClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [12.2] `mergeable` clause
//===----------------------------------------------------------------------===//

class OpenMP_MergeableClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$mergeable
  );

  let optAssemblyFormat = [{
    `mergeable` $mergeable
  }];

  let description = [{
    When the `mergeable` clause is present, the tasks generated by the construct
    are "mergeable tasks".
  }];
}

def OpenMP_MergeableClause : OpenMP_MergeableClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.7] `nogroup` clause
//===----------------------------------------------------------------------===//

class OpenMP_NogroupClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$nogroup
  );

  let optAssemblyFormat = [{
    `nogroup` $nogroup
  }];

  let description = [{
    By default, the taskloop construct executes as if it was enclosed in a
    taskgroup construct with no statements or directives outside of the taskloop
    construct. Thus, the taskloop construct creates an implicit taskgroup
    region. If the `nogroup` clause is present, no implicit taskgroup region is
    created.
  }];
}

def OpenMP_NogroupClause : OpenMP_NogroupClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.4.1] `nontemporal` clause
//===----------------------------------------------------------------------===//

class OpenMP_NontemporalClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$nontemporal_vars
  );

  let optAssemblyFormat = [{
    `nontemporal` `(`  $nontemporal_vars `:` type($nontemporal_vars) `)`
  }];

  let description = [{
    The optional `nontemporal` attribute specifies variables which have low
    temporal locality across the iterations where they are accessed.
  }];
}

def OpenMP_NontemporalClause : OpenMP_NontemporalClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.6] `nowait` clause
//===----------------------------------------------------------------------===//

class OpenMP_NowaitClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$nowait
  );

  let optAssemblyFormat = [{
    `nowait` $nowait
  }];

  let description = [{
    The optional `nowait` attribute, when present, eliminates the implicit
    barrier at the end of the construct, so the parent operation can make
    progress even if the child operation has not completed yet.
  }];
}

def OpenMP_NowaitClause : OpenMP_NowaitClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [12.6.2] `num_tasks` clause
//===----------------------------------------------------------------------===//

class OpenMP_NumTasksClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins OptionalAttr<NumTasksTypeAttr>:$num_tasks_mod,
      Optional<IntLikeType>:$num_tasks);

  let optAssemblyFormat = [{
    `num_tasks` `(` custom<NumTasksClause>($num_tasks_mod , $num_tasks, type($num_tasks)) `)`
  }];

  let description = [{
    If `num_tasks` is specified, the taskloop construct creates as many tasks as
    the minimum of the num-tasks expression and the number of logical loop
    iterations. Each task must have at least one logical loop iteration.
  }];
}

def OpenMP_NumTasksClause : OpenMP_NumTasksClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.2.1] `num_teams` clause
//===----------------------------------------------------------------------===//

class OpenMP_NumTeamsClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<AnyInteger>:$num_teams_lower,
    Optional<AnyInteger>:$num_teams_upper
  );

  let optAssemblyFormat = [{
    `num_teams` `(` ( $num_teams_lower^ `:` type($num_teams_lower) )? `to`
                      $num_teams_upper `:` type($num_teams_upper) `)`
  }];

  let description = [{
    The optional `num_teams_upper` and `num_teams_lower` arguments specify the
    limit on the number of teams to be created. If only the upper bound is
    specified, it acts as if the lower bound was set to the same value. It is
    not allowed to set `num_teams_lower` if `num_teams_upper` is not specified.
    They define a closed range, where both the lower and upper bounds are
    included.
  }];
}

def OpenMP_NumTeamsClause : OpenMP_NumTeamsClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.1.2] `num_threads` clause
//===----------------------------------------------------------------------===//

class OpenMP_NumThreadsClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<IntLikeType>:$num_threads
  );

  let optAssemblyFormat = [{
    `num_threads` `(` $num_threads `:` type($num_threads) `)`
  }];

  let description = [{
    The optional `num_threads` parameter specifies the number of threads which
    should be used to execute the parallel region.
  }];
}

def OpenMP_NumThreadsClause : OpenMP_NumThreadsClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.3] `order` clause
//===----------------------------------------------------------------------===//

class OpenMP_OrderClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<OrderKindAttr>:$order,
    OptionalAttr<OrderModifierAttr>:$order_mod
  );

  let optAssemblyFormat = [{
    `order` `(` custom<OrderClause>($order, $order_mod) `)`
  }];

  let description = [{
    The optional `order` attribute specifies which order the iterations of the
    associated loops are executed in. Currently the only option for this
    attribute is "concurrent".
  }];
}

def OpenMP_OrderClause : OpenMP_OrderClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [4.4.4] `ordered` clause
//===----------------------------------------------------------------------===//

class OpenMP_OrderedClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    ConfinedAttr<OptionalAttr<I64Attr>, [IntMinValue<0>]>:$ordered
  );

  let optAssemblyFormat = [{
    `ordered` `(` $ordered `)`
  }];

  let description = [{
    The optional `ordered` attribute specifies how many loops are associated
    with the worksharing-loop construct. The value of zero refers to the ordered
    clause specified without parameter.
  }];
}

def OpenMP_OrderedClause : OpenMP_OrderedClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [15.10.3] `parallelization-level` clause set
//===----------------------------------------------------------------------===//

class OpenMP_ParallelizationLevelClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$par_level_simd
  );

  let optAssemblyFormat = [{
    `par_level_simd` $par_level_simd
  }];

  let description = [{
    The `par_level_simd` attribute corresponds to the simd clause specified. If
    it is not present, it behaves as if the threads clause is specified or no
    clause is specified.
  }];
}

def OpenMP_ParallelizationLevelClause : OpenMP_ParallelizationLevelClauseSkip<>;

//===----------------------------------------------------------------------===//
// OpenMPV5.2: [12.5.2] `detach` clause
//===----------------------------------------------------------------------===//

class OpenMP_DetachClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false>
    : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {

  let arguments = (ins Optional<OpenMP_PointerLikeType>:$event_handle);

  let optAssemblyFormat = [{
		`detach` `(` $event_handle `:` type($event_handle) `)`
	}];

  let description = [{
		The detach clause specifies that the task generated by the construct on which it appears is a
	detachable task. A new allow-completion event is created and connected to the completion of the
	associated task region. The original event-handle is updated to represent that allow-completion
	event before the task data environment is created.
	}];
}

def OpenMP_DetachClause : OpenMP_DetachClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [12.4] `priority` clause
//===----------------------------------------------------------------------===//

class OpenMP_PriorityClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<AnyInteger>:$priority
  );

  let optAssemblyFormat = [{
    `priority` `(` $priority `:` type($priority) `)`
  }];

  let description = [{
    The `priority` clause is a hint for the priority of the generated tasks.
    The `priority` is a non-negative integer expression that provides a hint for
    task execution order. Among all tasks ready to be executed, higher priority
    tasks (those with a higher numerical value in the priority clause
    expression) are recommended to execute before lower priority ones. The
    default priority-value when no priority clause is specified should be
    assumed to be zero (the lowest priority).
  }];
}

def OpenMP_PriorityClause : OpenMP_PriorityClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.3, 5.4.4, 5.4.5] `private`, `firstprivate`, `lastprivate` clauses
//===----------------------------------------------------------------------===//

class OpenMP_PrivateClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    Variadic<AnyType>:$private_vars,
    OptionalAttr<SymbolRefArrayAttr>:$private_syms,
    // Set this attribute if a barrier is needed after initialization and
    // copying of lastprivate variables.
    UnitAttr:$private_needs_barrier
  );

  // TODO: Add description.
  // Assembly format not defined because this clause must be processed together
  // with the first region of the operation, as it defines entry block
  // arguments.
}

def OpenMP_PrivateClause : OpenMP_PrivateClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.1.4] `proc_bind` clause
//===----------------------------------------------------------------------===//

class OpenMP_ProcBindClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<ProcBindKindAttr>:$proc_bind_kind
  );

  let optAssemblyFormat = [{
    `proc_bind` `(` custom<ClauseAttr>($proc_bind_kind) `)`
  }];

  let description = [{
    The optional `proc_bind_kind` attribute controls the thread affinity for the
    execution of the parallel region.
  }];
}

def OpenMP_ProcBindClause : OpenMP_ProcBindClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.5.8] `reduction` clause
//===----------------------------------------------------------------------===//

class OpenMP_ReductionClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    OptionalAttr<ReductionModifierAttr>:$reduction_mod,
    Variadic<OpenMP_PointerLikeType>:$reduction_vars,
    OptionalAttr<DenseBoolArrayAttr>:$reduction_byref,
    OptionalAttr<SymbolRefArrayAttr>:$reduction_syms
  );

  let extraClassDeclaration = [{
    /// Returns the number of reduction variables.
    unsigned getNumReductionVars() { return getReductionVars().size(); }
  }];

  // Description varies depending on the operation.
  let description = [{
    Reductions can be performed by specifying the reduction modifer
    (`default`, `inscan` or `task`) in `reduction_mod`, reduction accumulator
    variables in `reduction_vars`, symbols referring to reduction declarations
    in the `reduction_syms` attribute, and whether the reduction variable
    should be passed into the reduction region by value or by reference in
    `reduction_byref`. Each reduction is identified by the accumulator it uses
    and accumulators must not be repeated in the same reduction. A private
    variable corresponding to the accumulator is used in place of the
    accumulator inside the body of the operation. The reduction declaration
    specifies how to combine the values from each iteration, section, team,
    thread or simd lane defined by the operation's region into the final value,
    which is available in the accumulator after they all complete.
  }];

  // Assembly format not defined because this clause must be processed together
  // with the first region of the operation, as it defines entry block
  // arguments.
}

def OpenMP_ReductionClause : OpenMP_ReductionClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.4.2] `safelen` clause
//===----------------------------------------------------------------------===//

class OpenMP_SafelenClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$safelen
  );

  let optAssemblyFormat = [{
    `safelen` `(` $safelen  `)`
  }];

  let description = [{
    The `safelen` clause specifies that no two concurrent iterations within a
    SIMD chunk can have a distance in the logical iteration space that is
    greater than or equal to the value given in the clause.
  }];
}

def OpenMP_SafelenClause : OpenMP_SafelenClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [11.5.3] `schedule` clause
//===----------------------------------------------------------------------===//

class OpenMP_ScheduleClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    OptionalAttr<ScheduleKindAttr>:$schedule_kind,
    Optional<AnyType>:$schedule_chunk,
    OptionalAttr<ScheduleModifierAttr>:$schedule_mod,
    UnitAttr:$schedule_simd
  );

  let optAssemblyFormat = [{
    `schedule` `(`
      custom<ScheduleClause>($schedule_kind, $schedule_mod, $schedule_simd,
                             $schedule_chunk, type($schedule_chunk)) `)`
  }];

  let description = [{
    The optional `schedule_kind` attribute specifies the loop schedule for this
    loop, determining how the loop is distributed across the parallel threads.
    The optional `schedule_chunk` associated with this determines further
    controls this distribution.
  }];
}

def OpenMP_ScheduleClause : OpenMP_ScheduleClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [10.4.3] `simdlen` clause
//===----------------------------------------------------------------------===//

class OpenMP_SimdlenClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    ConfinedAttr<OptionalAttr<I64Attr>, [IntPositive]>:$simdlen
  );

  let optAssemblyFormat = [{
    `simdlen` `(` $simdlen  `)`
  }];

  let description = [{
    When a `simdlen` clause is present, the preferred number of iterations to be
    executed concurrently is the value provided to the `simdlen` clause.
  }];
}

def OpenMP_SimdlenClause : OpenMP_SimdlenClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.5.9] `task_reduction` clause
//===----------------------------------------------------------------------===//

class OpenMP_TaskReductionClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$task_reduction_vars,
    OptionalAttr<DenseBoolArrayAttr>:$task_reduction_byref,
    OptionalAttr<SymbolRefArrayAttr>:$task_reduction_syms
  );

  let description = [{
    The `task_reduction` clause specifies a reduction among tasks. For each list
    item, the number of copies is unspecified. Any copies associated with the
    reduction are initialized before they are accessed by the tasks
    participating in the reduction. After the end of the region, the original
    list item contains the result of the reduction. Similarly to the `reduction`
    clause, accumulator variables must be passed in `task_reduction_vars`,
    symbols referring to reduction declarations in the `task_reduction_syms`
    attribute, and whether the reduction variable should be passed into the
    reduction region by value or by reference in `task_reduction_byref`.
  }];

  // Assembly format not defined because this clause must be processed together
  // with the first region of the operation, as it defines entry block
  // arguments.
}

def OpenMP_TaskReductionClause : OpenMP_TaskReductionClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [13.3] `thread_limit` clause
//===----------------------------------------------------------------------===//

class OpenMP_ThreadLimitClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    Optional<AnyInteger>:$thread_limit
  );

  let optAssemblyFormat = [{
    `thread_limit` `(` $thread_limit `:` type($thread_limit) `)`
  }];

  let description = [{
    The optional `thread_limit` specifies the limit on the number of threads.
  }];
}

def OpenMP_ThreadLimitClause : OpenMP_ThreadLimitClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [9.1.1] `sizes` clause
//===----------------------------------------------------------------------===//

class OpenMP_TileSizesClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
      OptionalAttr<DenseI64ArrayAttr>:$tile_sizes
  );
}

def OpenMP_TileSizesClause : OpenMP_TileSizesClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [12.1] `untied` clause
//===----------------------------------------------------------------------===//

class OpenMP_UntiedClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins
    UnitAttr:$untied
  );

  let optAssemblyFormat = [{
    `untied` $untied
  }];

  let description = [{
    If the `untied` clause is present on a task construct, any thread in the
    team can resume the task region after a suspension. The `untied` clause is
    ignored if a `final` clause is present on the same task construct and the
    `final` expression evaluates to `true`, or if a task is an included task.
  }];
}

def OpenMP_UntiedClause : OpenMP_UntiedClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.10] `use_device_addr` clause
//===----------------------------------------------------------------------===//

class OpenMP_UseDeviceAddrClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$use_device_addr_vars
  );

  let description = [{
    The optional `use_device_addr_vars` specifies the address of the objects in
    the device data environment.
  }];

  // Assembly format not defined because this clause must be processed together
  // with the first region of the operation, as it defines entry block
  // arguments.
}

def OpenMP_UseDeviceAddrClause : OpenMP_UseDeviceAddrClauseSkip<>;

//===----------------------------------------------------------------------===//
// V5.2: [5.4.8] `use_device_ptr` clause
//===----------------------------------------------------------------------===//

class OpenMP_UseDevicePtrClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let traits = [
    BlockArgOpenMPOpInterface
  ];

  let arguments = (ins
    Variadic<OpenMP_PointerLikeType>:$use_device_ptr_vars
  );

  let description = [{
    The optional `use_device_ptr_vars` specifies the device pointers to the
    corresponding list items in the device data environment.
  }];

  // Assembly format not defined because this clause must be processed together
  // with the first region of the operation, as it defines entry block
  // arguments.
}

def OpenMP_UseDevicePtrClause : OpenMP_UseDevicePtrClauseSkip<>;

#endif // OPENMP_CLAUSES
