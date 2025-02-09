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

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.td"
include "mlir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.td"
include "mlir/Dialect/OpenMP/OpenMPClauses.td"
include "mlir/Dialect/OpenMP/OpenMPOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// 2.19.4 Data-Sharing Attribute Clauses
//===----------------------------------------------------------------------===//

def PrivateClauseOp : OpenMP_Op<"private", [IsolatedFromAbove, RecipeInterface]> {
  let summary = "Provides declaration of [first]private logic.";
  let description = [{
    This operation provides a declaration of how to implement the
    [first]privatization of a variable. The dialect users should provide
    which type should be allocated for this variable. The allocated (usually by
    alloca) variable is passed to the initialization region which does everything
    else (e.g. initialization of Fortran runtime descriptors). Information about
    how to initialize the copy from the original item should be given in the
    copy region, and if needed, how to deallocate memory (allocated by the
    initialization region) in the dealloc region.

    Examples:

    * `private(x)` would not need any regions because no initialization is
      required by the standard for i32 variables and this is not firstprivate.
    ```mlir
    omp.private {type = private} @x.privatizer : i32
    ```

    * `firstprivate(x)` would be emitted as:
    ```mlir
    omp.private {type = firstprivate} @x.privatizer : i32 copy {
    ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
    // %arg0 is the original host variable.
    // %arg1 represents the memory allocated for this private variable.
    ... copy from host to the privatized clone ....
    omp.yield(%arg1 : !fir.ref<i32>)
    }
    ```

    * `private(x)` for "allocatables" would be emitted as:
    ```mlir
    omp.private {type = private} @x.privatizer : !some.type init {
    ^bb0(%arg0: !some.pointer<!some.type>, %arg1: !some.pointer<!some.type>):
    // initialize %arg1, using %arg0 as a mold for allocations.
    // For example if %arg0 is a heap allocated array with a runtime determined
    // length and !some.type is a runtime type descriptor, the init region
    // will read the array length from %arg0, and heap allocate an array of the
    // right length and initialize %arg1 to contain the array allocation and
    // length.
    omp.yield(%arg1 : !some.pointer<!some.type>)
    } dealloc {
    ^bb0(%arg0: !some.pointer<!some.type>):
    // ... deallocate memory allocated by the init region...
    // In the example above, this will free the heap allocated array data.
    omp.yield
    }
    ```

    There are no restrictions on the body except for:
    - The `dealloc` regions has a single argument.
    - The `init & `copy` regions have 2 arguments.
    - All three regions are terminated by `omp.yield` ops.
    The above restrictions and other obvious restrictions (e.g. verifying the
    type of yielded values) are verified by the custom op verifier. The actual
    contents of the blocks inside all regions are not verified.

    Instances of this op would then be used by ops that model directives that
    accept data-sharing attribute clauses.

    The $sym_name attribute provides a symbol by which the privatizer op can be
    referenced by other dialect ops.

    The $type attribute is the type of the value being privatized. This type
    will be implicitly allocated in MLIR->LLVMIR conversion and passed as the
    second argument to the init region. Therefore the type of arguments to
    the regions should be a type which represents a pointer to $type.

    The $data_sharing_type attribute specifies whether privatizer corresponds
    to a `private` or a `firstprivate` clause.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<AnyType>:$type,
                       DataSharingClauseTypeAttr:$data_sharing_type);

  let regions = (region AnyRegion:$init_region,
                        AnyRegion:$copy_region,
                        AnyRegion:$dealloc_region);

  let assemblyFormat = [{
    $data_sharing_type $sym_name `:` $type
      (`init` $init_region^)?
      (`copy` $copy_region^)?
      (`dealloc` $dealloc_region^)?
      attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"TypeRange">:$result,
                   CArg<"StringAttr">:$sym_name,
                   CArg<"TypeAttr">:$type)>
  ];

  let extraClassDeclaration = [{
    BlockArgument getInitMoldArg() {
      auto &region = getInitRegion();
      return region.empty() ? nullptr : region.getArgument(0);
    }
    BlockArgument getInitPrivateArg() {
      auto &region = getInitRegion();
      return region.empty() ? nullptr : region.getArgument(1);
    }
    BlockArgument getCopyMoldArg() {
      auto &region = getCopyRegion();
      return region.empty() ? nullptr : region.getArgument(0);
    }
    BlockArgument getCopyPrivateArg() {
      auto &region = getCopyRegion();
      return region.empty() ? nullptr : region.getArgument(1);
    }
    BlockArgument getDeallocMoldArg() {
      auto &region = getDeallocRegion();
      return region.empty() ? nullptr : region.getArgument(0);
    }

    /// needsMap returns true if the value being privatized should additionally
    /// be mapped to the target region using a MapInfoOp. This is most common
    /// when an allocatable is privatized. In such cases, the descriptor is used
    /// in privatization and needs to be mapped on to the device.
    bool needsMap() {
      BlockArgument moldArg = getInitMoldArg();
      return moldArg ? !moldArg.use_empty() : false;
    }

    /// Get the type for arguments to nested regions. This should
    /// generally be either the same as getType() or some pointer
    /// type (pointing to the type allocated by this op).
    /// This method will return Type{nullptr} if there are no nested
    /// regions.
    Type getArgType() {
      for (Region *region : getRegions())
        for (Type ty : region->getArgumentTypes())
          return ty;
      return nullptr;
    }
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.6 parallel Construct
//===----------------------------------------------------------------------===//

def ParallelOp : OpenMP_Op<"parallel", traits = [
    AttrSizedOperandSegments, AutomaticAllocationScope,
    DeclareOpInterfaceMethods<ComposableOpInterface>,
    DeclareOpInterfaceMethods<OutlineableOpenMPOpInterface>,
    RecursiveMemoryEffects
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_IfClause, OpenMP_NumThreadsClause,
    OpenMP_PrivateClause, OpenMP_ProcBindClause, OpenMP_ReductionClause
  ], singleRegion = true> {
  let summary = "parallel construct";
  let description = [{
    The parallel construct includes a region of code which is to be executed
    by a team of threads.

    The optional `if_expr` parameter specifies a boolean result of a conditional
    check. If this value is 1 or is not provided then the parallel region runs
    as normal, if it is 0 then the parallel region is executed with one thread.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins CArg<"const ParallelOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
        $private_syms, $reduction_mod, $reduction_vars, type($reduction_vars), $reduction_byref,
        $reduction_syms) attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def TerminatorOp : OpenMP_Op<"terminator", [Terminator, Pure]> {
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
// 2.7 teams Construct
//===----------------------------------------------------------------------===//
def TeamsOp : OpenMP_Op<"teams", traits = [
    AttrSizedOperandSegments, RecursiveMemoryEffects
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_IfClause, OpenMP_NumTeamsClause,
    OpenMP_PrivateClause, OpenMP_ReductionClause, OpenMP_ThreadLimitClause
  ], singleRegion = true> {
  let summary = "teams construct";
  let description = [{
    The teams construct defines a region of code that triggers the creation of a
    league of teams. Once created, the number of teams remains constant for the
    duration of its code region.

    If the `if_expr` is present and it evaluates to `false`, the number of teams
    created is one.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TeamsOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
        $private_syms, $reduction_mod, $reduction_vars, type($reduction_vars), $reduction_byref,
        $reduction_syms) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.8.1 Sections Construct
//===----------------------------------------------------------------------===//

def SectionOp : OpenMP_Op<"section", traits = [
    BlockArgOpenMPOpInterface, HasParent<"SectionsOp">
  ], singleRegion = true> {
  let summary = "section directive";
  let description = [{
    A section operation encloses a region which represents one section in a
    sections construct. A section op should always be surrounded by an
    `omp.sections` operation. The section operation may have block args
    which corespond to the block arguments of the surrounding `omp.sections`
    operation. This is done to reflect situations where these block arguments
    represent variables private to each section.
  }];
  let extraClassDeclaration = [{
    // Override BlockArgOpenMPOpInterface methods based on the parent
    // omp.sections operation. Only forward-declare here because SectionsOp is
    // not completely defined at this point.
    unsigned numPrivateBlockArgs();
    unsigned numReductionBlockArgs();
  }] # clausesExtraClassDeclaration;
  let assemblyFormat = "$region attr-dict";
}

def SectionsOp : OpenMP_Op<"sections", traits = [
    AttrSizedOperandSegments
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_NowaitClause, OpenMP_PrivateClause,
    OpenMP_ReductionClause
  ], singleRegion = true> {
  let summary = "sections construct";
  let description = [{
    The sections construct is a non-iterative worksharing construct that
    contains `omp.section` operations. The `omp.section` operations are to be
    distributed among and executed by the threads in a team. Each `omp.section`
    is executed once by one of the threads in the team in the context of its
    implicit task.
    Block arguments for reduction variables should be mirrored in enclosed
    `omp.section` operations.
  }] # clausesDescription;

  // Override region definition.
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins CArg<"const SectionsOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
        $private_syms, $reduction_mod, $reduction_vars, type($reduction_vars), $reduction_byref,
        $reduction_syms) attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.8.2 Single Construct
//===----------------------------------------------------------------------===//

def SingleOp : OpenMP_Op<"single", traits = [
    AttrSizedOperandSegments
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_CopyprivateClause, OpenMP_NowaitClause,
    OpenMP_PrivateClause
  ], singleRegion = true> {
  let summary = "single directive";
  let description = [{
    The single construct specifies that the associated structured block is
    executed by only one of the threads in the team (not necessarily the
    master thread), in the context of its implicit task. The other threads
    in the team, which do not execute the block, wait at an implicit barrier
    at the end of the single construct.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const SingleOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateRegion>($region, $private_vars, type($private_vars),
        $private_syms) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.8.3 Workshare Construct
//===----------------------------------------------------------------------===//

def WorkshareOp : OpenMP_Op<"workshare", traits = [
    RecursiveMemoryEffects,
  ], clauses = [
    OpenMP_NowaitClause,
  ], singleRegion = true> {
  let summary = "workshare directive";
  let description = [{
    The workshare construct divides the execution of the enclosed structured
    block into separate units of work, and causes the threads of the team to
    share the work such that each unit is executed only once by one thread, in
    the context of its implicit task

    This operation is used for the intermediate representation of the workshare
    block before the work gets divided between the threads. See the flang
    LowerWorkshare pass for details.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const WorkshareOperands &">:$clauses)>
  ];
}

def WorkshareLoopWrapperOp : OpenMP_Op<"workshare.loop_wrapper", traits = [
    DeclareOpInterfaceMethods<LoopWrapperInterface>, NoTerminator,
    RecursiveMemoryEffects, SingleBlock
  ], singleRegion = true> {
  let summary = "contains loop nests to be parallelized by workshare";
  let description = [{
    This operation wraps a loop nest that is marked for dividing into units of
    work by an encompassing omp.workshare operation.
  }];

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
  let assemblyFormat = "$region attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Loop Nest
//===----------------------------------------------------------------------===//

def LoopNestOp : OpenMP_Op<"loop_nest", traits = [
    RecursiveMemoryEffects, SameVariadicOperandSize
  ], clauses = [
    OpenMP_LoopRelatedClause
  ], singleRegion = true> {
  let summary = "rectangular loop nest";
  let description = [{
    This operation represents a collapsed rectangular loop nest. For each
    rectangular loop of the nest represented by an instance of this operation,
    lower and upper bounds, as well as a step variable, must be defined.

    The lower and upper bounds specify a half-open range: the range includes the
    lower bound but does not include the upper bound. If the `loop_inclusive`
    attribute is specified then the upper bound is also included.

    The body region can contain any number of blocks. The region is terminated
    by an `omp.yield` instruction without operands. The induction variables,
    represented as entry block arguments to the loop nest operation's single
    region, match the types of the `loop_lower_bounds`, `loop_upper_bounds` and
    `loop_steps` arguments.

    ```mlir
    omp.loop_nest (%i1, %i2) : i32 = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %a = load %arrA[%i1, %i2] : memref<?x?xf32>
      %b = load %arrB[%i1, %i2] : memref<?x?xf32>
      %sum = arith.addf %a, %b : f32
      store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
      omp.yield
    }
    ```

    This is a temporary simplified definition of a loop based on existing OpenMP
    loop operations intended to serve as a stopgap solution until the long-term
    representation of canonical loops is defined. Specifically, this operation
    is intended to serve as a unique source for loop information during the
    transition to making `omp.distribute`, `omp.simd`, `omp.taskloop` and
    `omp.wsloop` wrapper operations. It is not intended to help with the
    addition of support for loop transformations, non-rectangular loops and
    non-perfectly nested loops.
  }];

  let builders = [
    OpBuilder<(ins CArg<"const LoopNestOperands &">:$clauses)>
  ];

  let extraClassDeclaration = [{
    /// Returns the induction variables of the loop nest.
    ArrayRef<BlockArgument> getIVs() { return getRegion().getArguments(); }

    /// Fills a list of wrapper operations around this loop nest. Wrappers
    /// in the resulting vector will be sorted from innermost to outermost.
    void gatherWrappers(SmallVectorImpl<LoopWrapperInterface> &wrappers);
  }] # clausesExtraClassDeclaration;

  // Disable inherited clause-based declarative assembly format and instead
  // enable using the custom parser-printer implemented in C++.
  let assemblyFormat = ?;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.9.2 Workshare Loop Construct
//===----------------------------------------------------------------------===//

def LoopOp : OpenMP_Op<"loop", traits = [
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<LoopWrapperInterface>,
    NoTerminator, SingleBlock
  ], clauses = [
    OpenMP_BindClause, OpenMP_PrivateClause, OpenMP_OrderClause,
    OpenMP_ReductionClause
  ], singleRegion = true> {
  let summary = "loop construct";
  let description = [{
    A loop construct specifies that the logical iterations of the associated loops
    may execute concurrently and permits the encountering threads to execute the
    loop accordingly. A loop construct can have 3 different types of binding:
      1. teams: in which case the binding region is the innermost enclosing `teams`
         region.
      2. parallel: in which case the binding region is the innermost enclosing `parallel`
         region.
      3. thread: in which case the binding region is not defined.

    The body region can only contain a single block which must contain a single
    operation, this operation must be an `omp.loop_nest`.

    ```
    omp.loop <clauses> {
      omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
        %a = load %arrA[%i1, %i2] : memref<?x?xf32>
        %b = load %arrB[%i1, %i2] : memref<?x?xf32>
        %sum = arith.addf %a, %b : f32
        store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
        omp.yield
      }
    }
    ```
  }] # clausesDescription;

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
        $private_syms, $reduction_mod, $reduction_vars, type($reduction_vars), $reduction_byref,
        $reduction_syms) attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"const LoopOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def WsloopOp : OpenMP_Op<"wsloop", traits = [
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<ComposableOpInterface>,
    DeclareOpInterfaceMethods<LoopWrapperInterface>, NoTerminator,
    RecursiveMemoryEffects, SingleBlock
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_LinearClause, OpenMP_NowaitClause,
    OpenMP_OrderClause, OpenMP_OrderedClause, OpenMP_PrivateClause,
    OpenMP_ReductionClause, OpenMP_ScheduleClause
  ], singleRegion = true> {
  let summary = "worksharing-loop construct";
  let description = [{
    The worksharing-loop construct specifies that the iterations of the loop(s)
    will be executed in parallel by threads in the current context. These
    iterations are spread across threads that already exist in the enclosing
    parallel region.

    The body region can only contain a single block which must contain a single
    operation. This operation must be another compatible loop wrapper or an
    `omp.loop_nest`.

    ```
    omp.wsloop <clauses> {
      omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
        %a = load %arrA[%i1, %i2] : memref<?x?xf32>
        %b = load %arrB[%i1, %i2] : memref<?x?xf32>
        %sum = arith.addf %a, %b : f32
        store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
        omp.yield
      }
    }
    ```
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins CArg<"const WsloopOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
        $private_syms, $reduction_mod, $reduction_vars, type($reduction_vars), $reduction_byref,
        $reduction_syms) attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Simd construct [2.9.3.1]
//===----------------------------------------------------------------------===//

def SimdOp : OpenMP_Op<"simd", traits = [
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<ComposableOpInterface>,
    DeclareOpInterfaceMethods<LoopWrapperInterface>, NoTerminator,
    RecursiveMemoryEffects, SingleBlock
  ], clauses = [
    OpenMP_AlignedClause, OpenMP_IfClause, OpenMP_LinearClause,
    OpenMP_NontemporalClause, OpenMP_OrderClause, OpenMP_PrivateClause,
    OpenMP_ReductionClause, OpenMP_SafelenClause, OpenMP_SimdlenClause
  ], singleRegion = true> {
  let summary = "simd construct";
  let description = [{
    The simd construct can be applied to a loop to indicate that the loop can be
    transformed into a SIMD loop (that is, multiple iterations of the loop can
    be executed concurrently using SIMD instructions).

    The body region can only contain a single block which must contain a single
    operation. This operation must be another compatible loop wrapper or an
    `omp.loop_nest`.

    ```
    omp.simd <clauses> {
      omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
        %a = load %arrA[%i1, %i2] : memref<?x?xf32>
        %b = load %arrB[%i1, %i2] : memref<?x?xf32>
        %sum = arith.addf %a, %b : f32
        store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
        omp.yield
      }
    }
    ```

    When an if clause is present and evaluates to false, the preferred number of
    iterations to be executed concurrently is one, regardless of whether
    a simdlen clause is specified.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const SimdOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateReductionRegion>($region, $private_vars, type($private_vars),
        $private_syms, $reduction_mod, $reduction_vars, type($reduction_vars), $reduction_byref,
        $reduction_syms) attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}


def YieldOp : OpenMP_Op<"yield",
    [Pure, ReturnLike, Terminator,
     ParentOneOf<["AtomicUpdateOp", "DeclareReductionOp", "LoopNestOp",
                  "PrivateClauseOp"]>]> {
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

  let assemblyFormat = "( `(` $results^ `:` type($results) `)` )? attr-dict";
}

//===----------------------------------------------------------------------===//
// Distribute construct [2.9.4.1]
//===----------------------------------------------------------------------===//
def DistributeOp : OpenMP_Op<"distribute", traits = [
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<ComposableOpInterface>,
    DeclareOpInterfaceMethods<LoopWrapperInterface>, NoTerminator,
    RecursiveMemoryEffects, SingleBlock
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_DistScheduleClause, OpenMP_OrderClause,
    OpenMP_PrivateClause
  ], singleRegion = true> {
  let summary = "distribute construct";
  let description = [{
    The distribute construct specifies that the iterations of one or more loops
    (optionally specified using collapse clause) will be executed by the
    initial teams in the context of their implicit tasks. The loops that the
    distribute op is associated with starts with the outermost loop enclosed by
    the distribute op region and going down the loop nest toward the innermost
    loop. The iterations are distributed across the initial threads of all
    initial teams that execute the teams region to which the distribute region
    binds.

    The distribute loop construct specifies that the iterations of the loop(s)
    will be executed in parallel by threads in the current context. These
    iterations are spread across threads that already exist in the enclosing
    region.
    
    The body region can only contain a single block which must contain a single
    operation. This operation must be another compatible loop wrapper or an
    `omp.loop_nest`.

    ```mlir
    omp.distribute <clauses> {
      omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
        %a = load %arrA[%i1, %i2] : memref<?x?xf32>
        %b = load %arrB[%i1, %i2] : memref<?x?xf32>
        %sum = arith.addf %a, %b : f32
        store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
        omp.yield
      }
    }
    ```
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const DistributeOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<PrivateRegion>($region, $private_vars, type($private_vars),
        $private_syms) attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.10.1 task Construct
//===----------------------------------------------------------------------===//

def TaskOp
    : OpenMP_Op<"task",
                traits = [AttrSizedOperandSegments, AutomaticAllocationScope,
                          OutlineableOpenMPOpInterface],
                clauses = [
                    // TODO: Complete clause list (affinity, detach).
                    OpenMP_AllocateClause, OpenMP_DependClause,
                    OpenMP_FinalClause, OpenMP_IfClause,
                    OpenMP_InReductionClause, OpenMP_MergeableClause,
                    OpenMP_PriorityClause, OpenMP_PrivateClause,
                    OpenMP_UntiedClause, OpenMP_DetachClause],
                singleRegion = true> {
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

    The `in_reduction` clause specifies that this particular task (among all the
    tasks in current taskgroup, if any) participates in a reduction.
    `in_reduction_byref` indicates whether each reduction variable should
    be passed by value or by reference.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TaskOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<InReductionPrivateRegion>(
        $region, $in_reduction_vars, type($in_reduction_vars),
        $in_reduction_byref, $in_reduction_syms, $private_vars,
        type($private_vars), $private_syms) attr-dict
  }];

  let hasVerifier = 1;
}

def TaskloopOp : OpenMP_Op<"taskloop", traits = [
    AttrSizedOperandSegments, AutomaticAllocationScope,
    DeclareOpInterfaceMethods<ComposableOpInterface>,
    DeclareOpInterfaceMethods<LoopWrapperInterface>, NoTerminator,
    RecursiveMemoryEffects, SingleBlock
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_FinalClause, OpenMP_GrainsizeClause,
    OpenMP_IfClause, OpenMP_InReductionClauseSkip<extraClassDeclaration = true>,
    OpenMP_MergeableClause, OpenMP_NogroupClause, OpenMP_NumTasksClause,
    OpenMP_PriorityClause, OpenMP_PrivateClause,
    OpenMP_ReductionClauseSkip<extraClassDeclaration = true>,
    OpenMP_UntiedClause
  ], singleRegion = true> {
  let summary = "taskloop construct";
  let description = [{
    The taskloop construct specifies that the iterations of one or more
    associated loops will be executed in parallel using explicit tasks. The
    iterations are distributed across tasks generated by the construct and
    scheduled to be executed.

    The body region can only contain a single block which must contain a single
    operation. This operation must be another compatible loop wrapper or an
    `omp.loop_nest`.

    ```
    omp.taskloop <clauses> {
      omp.loop_nest (%i1, %i2) : index = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
        %a = load %arrA[%i1, %i2] : memref<?x?xf32>
        %b = load %arrB[%i1, %i2] : memref<?x?xf32>
        %sum = arith.addf %a, %b : f32
        store %sum, %arrC[%i1, %i2] : memref<?x?xf32>
        omp.yield
      }
    }
    ```

    For definitions of "undeferred task", "included task", "final task" and
    "mergeable task", please check OpenMP Specification.

    When an `if` clause is present on a taskloop construct, and if the `if`
    clause expression evaluates to `false`, undeferred tasks are generated. The
    use of a variable in an `if` clause expression of a taskloop construct
    causes an implicit reference to the variable in all enclosing constructs.
  }] # clausesDescription # [{
    If an `in_reduction` clause is present on the taskloop construct, the
    behavior is as if each generated task was defined by a task construct on
    which an `in_reduction` clause with the same reduction operator and list
    items is present. Thus, the generated tasks are participants of a reduction
    previously defined by a reduction scoping clause. In this case, accumulator
    variables are specified in `in_reduction_vars`, symbols referring to
    reduction declarations in `in_reduction_syms` and `in_reduction_byref`
    indicate for each reduction variable whether it should be passed by value or
    by reference.

    If a `reduction` clause is present on the taskloop construct, the behavior
    is as if a `task_reduction` clause with the same reduction operator and list
    items was applied to the implicit taskgroup construct enclosing the taskloop
    construct. The taskloop construct executes as if each generated task was
    defined by a task construct on which an `in_reduction` clause with the same
    reduction operator and list items is present. Thus, the generated tasks are
    participants of the reduction defined by the `task_reduction` clause that
    was applied to the implicit taskgroup construct.
  }];

  let builders = [
    OpBuilder<(ins CArg<"const TaskloopOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<InReductionPrivateReductionRegion>(
        $region, $in_reduction_vars, type($in_reduction_vars),
        $in_reduction_byref, $in_reduction_syms, $private_vars,
        type($private_vars), $private_syms, $reduction_mod, $reduction_vars,
        type($reduction_vars), $reduction_byref, $reduction_syms) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Returns the reduction variables
    SmallVector<Value> getAllReductionVars();

    // Define BlockArgOpenMPOpInterface methods here because they are not
    // inherited from the respective clauses.
    unsigned numInReductionBlockArgs() { return getInReductionVars().size(); }
    unsigned numReductionBlockArgs() { return getReductionVars().size(); }

    void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
  }] # clausesExtraClassDeclaration;

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def TaskgroupOp : OpenMP_Op<"taskgroup", traits = [
    AttrSizedOperandSegments, AutomaticAllocationScope
  ], clauses = [
    OpenMP_AllocateClause, OpenMP_TaskReductionClause
  ], singleRegion = true> {
  let summary = "taskgroup construct";
  let description = [{
    The taskgroup construct specifies a wait on completion of child tasks of the
    current task and their descendent tasks.

    When a thread encounters a taskgroup construct, it starts executing the
    region. All child tasks generated in the taskgroup region and all of their
    descendants that bind to the same parallel region as the taskgroup region
    are part of the taskgroup set associated with the taskgroup region. There is
    an implicit task scheduling point at the end of the taskgroup region. The
    current task is suspended at the task scheduling point until all tasks in
    the taskgroup set complete execution.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TaskgroupOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<TaskReductionRegion>(
        $region, $task_reduction_vars, type($task_reduction_vars),
        $task_reduction_byref, $task_reduction_syms) attr-dict
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
def FlushOp : OpenMP_Op<"flush", clauses = [
    // TODO: Complete clause list (memory_order).
  ]> {
  let summary = "flush construct";
  let description = [{
    The flush construct executes the OpenMP flush operation. This operation
    makes a thread's temporary view of memory consistent with memory and
    enforces an order on the memory operations of the variables explicitly
    specified or implied.
  }] # clausesDescription;

  let arguments = !con((ins Variadic<OpenMP_PointerLikeType>:$varList),
                       clausesArgs);

  // Override inherited assembly format to include `varList`.
  let assemblyFormat = "( `(` $varList^ `:` type($varList) `)` )? attr-dict";

  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      return getOperation()->getNumOperands();
    }
    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      return getOperand(i);
    }
  }] # clausesExtraClassDeclaration;
}

//===----------------------------------------------------------------------===//
// Map related constructs
//===----------------------------------------------------------------------===//

def MapBoundsOp : OpenMP_Op<"map.bounds",
    [AttrSizedOperandSegments, NoMemoryEffect]> {
  let summary = "Represents normalized bounds information for map clauses.";

  let description = [{
    This operation is a variation on the OpenACC dialects DataBoundsOp. Within
    the OpenMP dialect it stores the bounds/range of data to be mapped to a
    device specified by map clauses on target directives. Within
    the OpenMP dialect, the MapBoundsOp is associated with MapInfoOp,
    helping to store bounds information for the mapped variable.

    It is used to support OpenMP array sectioning, Fortran pointer and
    allocatable mapping and pointer/allocatable member of derived types.
    In all cases the MapBoundsOp holds information on the section of
    data to be mapped. Such as the upper bound and lower bound of the
    section of data to be mapped. This information is currently
    utilised by the LLVM-IR lowering to help generate instructions to
    copy data to and from the device when processing target operations.

    The example below copys a section of a 10-element array; all except the
    first element, utilising OpenMP array sectioning syntax where array
    subscripts are provided to specify the bounds to be mapped to device.
    To simplify the examples, the constants are used directly, in reality
    they will be MLIR SSA values.

    C++:
    ```
    int array[10];
    #pragma target map(array[1:9])
    ```
    =>
    ```mlir
    omp.map.bounds lower_bound(1) upper_bound(9) extent(9) start_idx(0)
    ```

    Fortran:
    ```
    integer :: array(1:10)
    !$target map(array(2:10))
    ```
    =>
    ```mlir
    omp.map.bounds lower_bound(1) upper_bound(9) extent(9) start_idx(1)
    ```

    For Fortran pointers and allocatables (as well as those that are
    members of derived types) the bounds information is provided by
    the Fortran compiler and runtime through descriptor information.

    A basic pointer example can be found below (constants again
    provided for simplicity, where in reality SSA values will be
    used, in this case that point to data yielded by Fortran's
    descriptors):

    Fortran:
    ```
    integer, pointer :: ptr(:)
    allocate(ptr(10))
    !$target map(ptr)
    ```
    =>
    ```mlir
    omp.map.bounds lower_bound(0) upper_bound(9) extent(10) start_idx(1)
    ```

    This operation records the bounds information in a normalized fashion
    (zero-based). This works well with the `PointerLikeType`
    requirement in data clauses - since a `lower_bound` of 0 means looking
    at data at the zero offset from pointer.

    This operation must have an `upper_bound` or `extent` (or both are allowed -
    but not checked for consistency). When the source language's arrays are
    not zero-based, the `start_idx` must specify the zero-position index.
  }];

  let arguments = (ins Optional<IntLikeType>:$lower_bound,
                       Optional<IntLikeType>:$upper_bound,
                       Optional<IntLikeType>:$extent,
                       Optional<IntLikeType>:$stride,
                       DefaultValuedAttr<BoolAttr, "false">:$stride_in_bytes,
                       Optional<IntLikeType>:$start_idx);
  let results = (outs OpenMP_MapBoundsType:$result);

  let assemblyFormat = [{
    oilist(
        `lower_bound` `(` $lower_bound `:` type($lower_bound) `)`
      | `upper_bound` `(` $upper_bound `:` type($upper_bound) `)`
      | `extent` `(` $extent `:` type($extent) `)`
      | `stride` `(` $stride `:` type($stride) `)`
      | `start_idx` `(` $start_idx `:` type($start_idx) `)`
    ) attr-dict
  }];

  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      return getNumOperands();
    }

    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      return getOperands()[i];
    }
  }];

  let hasVerifier = 1;
}

def MapInfoOp : OpenMP_Op<"map.info", [AttrSizedOperandSegments]> {
  let arguments = (ins OpenMP_PointerLikeType:$var_ptr,
                       TypeAttr:$var_type,
                       Optional<OpenMP_PointerLikeType>:$var_ptr_ptr,
                       Variadic<OpenMP_PointerLikeType>:$members,
                       OptionalAttr<IndexListArrayAttr>:$members_index,
                       Variadic<OpenMP_MapBoundsType>:$bounds, /* rank-0 to rank-{n-1} */
                       OptionalAttr<UI64Attr>:$map_type,
                       OptionalAttr<VariableCaptureKindAttr>:$map_capture_type,
                       OptionalAttr<StrAttr>:$name,
                       DefaultValuedAttr<BoolAttr, "false">:$partial_map);
  let results = (outs OpenMP_PointerLikeType:$omp_ptr);

  let description = [{
    The MapInfoOp captures information relating to individual OpenMP map clauses
    that are applied to certain OpenMP directives such as Target and Target Data.

    For example, the map type modifier; such as from, tofrom and to, the variable
    being captured or the bounds of an array section being mapped.

    It can be used to capture both implicit and explicit map information, where
    explicit is an argument directly specified to an OpenMP map clause or implicit
    where a variable is utilised in a target region but is defined externally to
    the target region.

    This map information is later used to aid the lowering of the target operations
    they are attached to providing argument input and output context for kernels
    generated or the target data mapping environment.

    Example (Fortran):

    ```
    integer :: index
    !$target map(to: index)
    ```
    =>
    ```mlir
    omp.map.info var_ptr(%index_ssa) map_type(to) map_capture_type(ByRef)
      name(index)
    ```

    Description of arguments:
    - `var_ptr`: The address of variable to copy.
    - `var_type`: The type of the variable to copy.
    - `var_ptr_ptr`: Used when the variable copied is a member of a class, structure
      or derived type and refers to the originating struct.
    - `members`: Used to indicate mapped child members for the current MapInfoOp,
       represented as other MapInfoOp's, utilised in cases where a parent structure
       type and members of the structure type are being mapped at the same time.
       For example: map(to: parent, parent->member, parent->member2[:10])
    - `members_index`: Used to indicate the ordering of members within the containing
       parent (generally a record type such as a structure, class or derived type),
       e.g. struct {int x, float y, double z}, x would be 0, y would be 1, and z
       would be 2. This aids the mapping.
    - `bounds`: Used when copying slices of array's, pointers or pointer members of
       objects (e.g. derived types or classes), indicates the bounds to be copied
       of the variable. When it's an array slice it is in rank order where rank 0
       is the inner-most dimension.
    - 'map_type': OpenMP map type for this map capture, for example: from, to and
       always. It's a bitfield composed of the OpenMP runtime flags stored in
       OpenMPOffloadMappingFlags.
    - 'map_capture_type': Capture type for the variable e.g. this, byref, byvalue, byvla
       this can affect how the variable is lowered.
    - `name`: Holds the name of variable as specified in user clause (including bounds).
    - `partial_map`: The record type being mapped will not be mapped in its entirety,
       it may be used however, in a mapping to bind it's mapped components together.
  }];

  let assemblyFormat = [{
    `var_ptr` `(` $var_ptr `:` type($var_ptr) `,` $var_type `)`
    oilist(
        `var_ptr_ptr` `(` $var_ptr_ptr `:` type($var_ptr_ptr) `)`
      | `map_clauses` `(` custom<MapClause>($map_type) `)`
      | `capture` `(` custom<CaptureType>($map_capture_type) `)`
      | `members` `(` $members `:` custom<MembersIndex>($members_index) `:` type($members) `)`
      | `bounds` `(` $bounds `)`
    ) `->` type($omp_ptr) attr-dict
  }];

  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      return getNumOperands();
    }

    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      return getOperands()[i];
    }
  }];
}

//===---------------------------------------------------------------------===//
// 2.14.2 target data Construct
//===---------------------------------------------------------------------===//

def TargetDataOp: OpenMP_Op<"target_data", traits = [
    AttrSizedOperandSegments
  ], clauses = [
    OpenMP_DeviceClause, OpenMP_IfClause, OpenMP_MapClause,
    OpenMP_UseDeviceAddrClause, OpenMP_UseDevicePtrClause
  ], singleRegion = true> {
  let summary = "target data construct";
  let description = [{
    Map variables to a device data environment for the extent of the region.

    The omp target data directive maps variables to a device data
    environment, and defines the lexical scope of the data environment
    that is created. The omp target data directive can reduce data copies
    to and from the offloading device when multiple target regions are using
    the same data.

    The optional `if_expr` parameter specifies a boolean result of a conditional
    check. If this value is 1 or is not provided then the target region runs on
    a device, if it is 0 then the target region is executed on the host device.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TargetDataOperands &">:$clauses)>
  ];

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<UseDeviceAddrUseDevicePtrRegion>(
        $region, $use_device_addr_vars, type($use_device_addr_vars),
        $use_device_ptr_vars, type($use_device_ptr_vars)) attr-dict
  }];

  let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// 2.14.3 target enter data Construct
//===---------------------------------------------------------------------===//

def TargetEnterDataOp: OpenMP_Op<"target_enter_data", traits = [
    AttrSizedOperandSegments
  ], clauses = [
    OpenMP_DependClause, OpenMP_DeviceClause, OpenMP_IfClause, OpenMP_MapClause,
    OpenMP_NowaitClause
  ]> {
  let summary = "target enter data construct";
  let description = [{
    The target enter data directive specifies that variables are mapped to
    a device data environment. The target enter data directive is a
    stand-alone directive.

    The optional `if_expr` parameter specifies a boolean result of a conditional
    check. If this value is 1 or is not provided then the target region runs on
    a device, if it is 0 then the target region is executed on the host device.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TargetEnterExitUpdateDataOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// 2.14.4 target exit data Construct
//===---------------------------------------------------------------------===//

def TargetExitDataOp: OpenMP_Op<"target_exit_data", traits = [
    AttrSizedOperandSegments
  ], clauses = [
    OpenMP_DependClause, OpenMP_DeviceClause, OpenMP_IfClause, OpenMP_MapClause,
    OpenMP_NowaitClause
  ]> {
  let summary = "target exit data construct";
  let description = [{
    The target exit data directive specifies that variables are mapped to a
    device data environment. The target exit data directive is
    a stand-alone directive.

    The optional `if_expr` parameter specifies a boolean result of a conditional
    check. If this value is 1 or is not provided then the target region runs on
    a device, if it is 0 then the target region is executed on the host device.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TargetEnterExitUpdateDataOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// 2.14.6 target update Construct
//===---------------------------------------------------------------------===//

def TargetUpdateOp: OpenMP_Op<"target_update", traits = [
    AttrSizedOperandSegments
  ], clauses = [
    OpenMP_DependClause, OpenMP_DeviceClause, OpenMP_IfClause, OpenMP_MapClause,
    OpenMP_NowaitClause
  ]> {
  let summary = "target update construct";
  let description = [{
    The target update directive makes the corresponding list items in the device
    data environment consistent with their original list items, according to the
    specified motion clauses. The target update construct is a stand-alone
    directive.

    The optional `if_expr` parameter specifies a boolean result of a conditional
    check. If this value is 1 or is not provided then the target region runs on
    a device, if it is 0 then the target region is executed on the host device.

    We use `MapInfoOp` to model the motion clauses and their modifiers. Even
    though the spec differentiates between map-types & map-type-modifiers vs.
    motion-clauses & motion-modifiers, the motion clauses and their modifiers
    are a subset of map types and their modifiers. The subset relation is
    handled in during verification to make sure the restrictions for target
    update are respected.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TargetEnterExitUpdateDataOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.5 target construct
//===----------------------------------------------------------------------===//

def TargetOp : OpenMP_Op<"target", traits = [
    AttrSizedOperandSegments, BlockArgOpenMPOpInterface, IsolatedFromAbove,
    OutlineableOpenMPOpInterface
  ], clauses = [
    // TODO: Complete clause list (defaultmap, uses_allocators).
    OpenMP_AllocateClause, OpenMP_BareClause, OpenMP_DependClause,
    OpenMP_DeviceClause, OpenMP_HasDeviceAddrClause, OpenMP_HostEvalClause,
    OpenMP_IfClause, OpenMP_InReductionClause, OpenMP_IsDevicePtrClause,
    OpenMP_MapClauseSkip<assemblyFormat = true>, OpenMP_NowaitClause,
    OpenMP_PrivateClause, OpenMP_ThreadLimitClause
  ], singleRegion = true> {
  let summary = "target construct";
  let description = [{
    The target construct includes a region of code which is to be executed
    on a device.

    The optional `if_expr` parameter specifies a boolean result of a conditional
    check. If this value is 1 or is not provided then the target region runs on
    a device, if it is 0 then the target region is executed on the host device.

    The `private_maps` attribute connects `private` operands to their corresponding
    `map` operands. For `private` operands that require a map, the value of the
    corresponding element in the attribute is the index of the `map` operand
    (relative to other `map` operands not the whole operands of the operation). For
    `private` opernads that do not require a map, this value is -1 (which is omitted
    from the assembly foramt printing).
  }] # clausesDescription;

  let arguments = !con(clausesArgs,
                       (ins OptionalAttr<DenseI64ArrayAttr>:$private_maps));

  let builders = [
    OpBuilder<(ins CArg<"const TargetOperands &">:$clauses)>
  ];

  let extraClassDeclaration = [{
    unsigned numMapBlockArgs() { return getMapVars().size(); }

    mlir::Value getMappedValueForPrivateVar(unsigned privVarIdx) {
      std::optional<DenseI64ArrayAttr> privateMapIdices = getPrivateMapsAttr();

      if (!privateMapIdices.has_value())
        return {};

      int64_t mapInfoOpIdx = (*privateMapIdices)[privVarIdx];

      if (mapInfoOpIdx == -1)
        return {};

      return getMapVars()[mapInfoOpIdx];
    }

    /// Returns the innermost OpenMP dialect operation captured by this target
    /// construct. For an operation to be detected as captured, it must be
    /// inside a (possibly multi-level) nest of OpenMP dialect operation's
    /// regions where none of these levels contain other operations considered
    /// not-allowed for these purposes (i.e. only terminator operations are
    /// allowed from the OpenMP dialect, and other dialect's operations are
    /// allowed as long as they don't have a memory write effect).
    ///
    /// If there are omp.loop_nest operations in the sequence of nested
    /// operations, the top level one will be the one captured.
    Operation *getInnermostCapturedOmpOp();

    /// Infers the kernel type (Generic, SPMD or Generic-SPMD) based on the
    /// contents of the target region.
    llvm::omp::OMPTgtExecModeFlags getKernelExecFlags();
  }] # clausesExtraClassDeclaration;

  let assemblyFormat = clausesAssemblyFormat # [{
    custom<HostEvalInReductionMapPrivateRegion>(
        $region, $host_eval_vars, type($host_eval_vars), $in_reduction_vars,
        type($in_reduction_vars), $in_reduction_byref, $in_reduction_syms,
        $map_vars, type($map_vars), $private_vars, type($private_vars),
        $private_syms, $private_maps) attr-dict
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}


//===----------------------------------------------------------------------===//
// 2.16 master Construct
//===----------------------------------------------------------------------===//
def MasterOp : OpenMP_Op<"master", singleRegion = true> {
  let summary = "master construct";
  let description = [{
    The master construct specifies a structured block that is executed by
    the master thread of the team.
  }];

  let assemblyFormat = "$region attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.17.1 critical Construct
//===----------------------------------------------------------------------===//
def CriticalDeclareOp : OpenMP_Op<"critical.declare", clauses = [
    OpenMP_CriticalNameClause, OpenMP_HintClause
  ]> {
  let summary = "declares a named critical section.";
  let description = [{
    Declares a named critical section.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const CriticalDeclareOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}


def CriticalOp : OpenMP_Op<"critical", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ], singleRegion = 1> {
  let summary = "critical construct";
  let description = [{
    The critical construct imposes a restriction on the associated structured
    block (region) to be executed by only a single thread at a time.

    The optional `name` argument of critical constructs is used to identify
    them. Unnamed critical constructs behave as though an identical name was
    specified.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$name);

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

def OrderedOp : OpenMP_Op<"ordered", clauses = [OpenMP_DoacrossClause]> {
  let summary = "ordered construct without region";
  let description = [{
    The ordered construct without region is a stand-alone directive that
    specifies cross-iteration dependencies in a doacross loop nest.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const OrderedOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

def OrderedRegionOp : OpenMP_Op<"ordered.region", clauses = [
    OpenMP_ParallelizationLevelClause
  ], singleRegion = true> {
  let summary = "ordered construct with region";
  let description = [{
    The ordered construct with region specifies a structured block in a
    worksharing-loop, SIMD, or worksharing-loop SIMD region that is executed in
    the order of the loop iterations.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const OrderedRegionOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.17.5 taskwait Construct
//===----------------------------------------------------------------------===//

def TaskwaitOp : OpenMP_Op<"taskwait", clauses = [
    OpenMP_DependClause, OpenMP_NowaitClause
  ]> {
  let summary = "taskwait construct";
  let description = [{
    The taskwait construct specifies a wait on the completion of child tasks
    of the current task.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const TaskwaitOperands &">:$clauses)>
  ];
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

def AtomicReadOp : OpenMP_Op<"atomic.read", traits = [
    AtomicReadOpInterface
  ], clauses = [
    OpenMP_HintClause, OpenMP_MemoryOrderClause
  ]> {
  let summary = "performs an atomic read";
  let description = [{
    This operation performs an atomic read.

    The operand `x` is the address from where the value is atomically read.
    The operand `v` is the address where the value is stored after reading.
  }] # clausesDescription;

  let arguments = !con((ins OpenMP_PointerLikeType:$x,
                            OpenMP_PointerLikeType:$v,
                            TypeAttr:$element_type), clausesArgs);

  // Override clause-based assemblyFormat.
  let assemblyFormat = "$v `=` $x" # clausesReqAssemblyFormat # " oilist(" #
    clausesOptAssemblyFormat #
    ") `:` type($v) `,` type($x) `,` $element_type attr-dict";

  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      assert(getX() && "expected 'x' operand");
      assert(getV() && "expected 'v' operand");
      return 2;
    }

    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      assert(i < 2 && "invalid index position for an operand");
      return i == 0 ? getX() : getV();
    }
  }] # clausesExtraClassDeclaration;

  let hasVerifier = 1;
}

def AtomicWriteOp : OpenMP_Op<"atomic.write", traits = [
    AtomicWriteOpInterface
  ], clauses = [
    OpenMP_HintClause, OpenMP_MemoryOrderClause
  ]> {
  let summary = "performs an atomic write";
  let description = [{
    This operation performs an atomic write.

    The operand `x` is the address to where the `expr` is atomically
    written w.r.t. multiple threads. The evaluation of `expr` need not be
    atomic w.r.t. the write to address. In general, the type(x) must
    dereference to type(expr).
  }] # clausesDescription;

  let arguments = !con((ins OpenMP_PointerLikeType:$x,
                            AnyType:$expr), clausesArgs);

  // Override clause-based assemblyFormat.
  let assemblyFormat = "$x `=` $expr" # clausesReqAssemblyFormat # " oilist(" #
    clausesOptAssemblyFormat # ") `:` type($x) `,` type($expr) attr-dict";

  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      assert(getX() && "expected address operand");
      assert(getExpr() && "expected value operand");
      return 2;
    }

    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      assert(i < 2 && "invalid index position for an operand");
      return i == 0 ? getX() : getExpr();
    }
  }] # clausesExtraClassDeclaration;

  let hasVerifier = 1;
}

def AtomicUpdateOp : OpenMP_Op<"atomic.update", traits = [
    AtomicUpdateOpInterface, RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">
  ], clauses = [
    OpenMP_HintClause, OpenMP_MemoryOrderClause
  ], singleRegion = 1> {
  let summary = "performs an atomic update";
  let description = [{
    This operation performs an atomic update.

    The operand `x` is exactly the same as the operand `x` in the OpenMP
    Standard (OpenMP 5.0, section 2.17.7). It is the address of the variable
    that is being updated. `x` is atomically read/written.

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
  }] # clausesDescription;

  let arguments = !con((ins Arg<OpenMP_PointerLikeType,
                                "Address of variable to be updated",
                                [MemRead, MemWrite]>:$x), clausesArgs);

  // Override region definition.
  let regions = (region SizedRegion<1>:$region);

  // Override clause-based assemblyFormat.
  let assemblyFormat = clausesAssemblyFormat #
    "$x `:` type($x) $region attr-dict";

  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      assert(getX() && "expected 'x' operand");
      return 1;
    }

    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      assert(i == 0 && "invalid index position for an operand");
      return getX();
    }
  }] # clausesExtraClassDeclaration;

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def AtomicCaptureOp : OpenMP_Op<"atomic.capture", traits = [
    AtomicCaptureOpInterface, RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TerminatorOp">
  ], clauses = [
    OpenMP_HintClause, OpenMP_MemoryOrderClause
  ], singleRegion = 1> {
  let summary = "performs an atomic capture";
  let description = [{
    This operation performs an atomic capture.

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
  }] # clausesDescription;

  // Override region definition.
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    /// Returns the `atomic.read` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicReadOp getAtomicReadOp();

    /// Returns the `atomic.write` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicWriteOp getAtomicWriteOp();

    /// Returns the `atomic.update` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicUpdateOp getAtomicUpdateOp();
  }] # clausesExtraClassDeclaration;

  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// [5.1] 2.21.2 threadprivate Directive
//===----------------------------------------------------------------------===//

def ThreadprivateOp : OpenMP_Op<"threadprivate",
                                [AllTypesMatch<["sym_addr", "tls_addr"]>]> {
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
  let extraClassDeclaration = [{
    /// The number of variable operands.
    unsigned getNumVariableOperands() {
      assert(getSymAddr() && "expected one variable operand");
      return 1;
    }

    /// The i-th variable operand passed.
    Value getVariableOperand(unsigned i) {
      assert(i == 0 && "invalid index position for an operand");
      return getSymAddr();
    }
  }];
}

//===----------------------------------------------------------------------===//
// 2.18.1 Cancel Construct
//===----------------------------------------------------------------------===//
def CancelOp : OpenMP_Op<"cancel", clauses = [
    OpenMP_CancelDirectiveNameClause, OpenMP_IfClause
  ]> {
  let summary = "cancel directive";
  let description = [{
    The cancel construct activates cancellation of the innermost enclosing
    region of the type specified.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const CancelOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.18.2 Cancellation Point Construct
//===----------------------------------------------------------------------===//
def CancellationPointOp : OpenMP_Op<"cancellation_point", clauses = [
    OpenMP_CancelDirectiveNameClause
  ]> {
  let summary = "cancellation point directive";
  let description = [{
    The cancellation point construct introduces a user-defined cancellation
    point at which implicit or explicit tasks check if cancellation of the
    innermost enclosing region of the type specified has been activated.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const CancellationPointOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

def ScanOp : OpenMP_Op<"scan", [
    AttrSizedOperandSegments, MemoryEffects<[MemWrite]>
  ], clauses = [
    OpenMP_InclusiveClause, OpenMP_ExclusiveClause]> {
  let summary = "scan directive";
  let description = [{
    The scan directive allows to specify scan reductions. It should be
    enclosed within a parent directive along with which a reduction clause
    with `inscan` modifier must be specified. The scan directive allows to
    split code blocks into input phase and scan phase in the region
    enclosed by the parent.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const ScanOperands &">:$clauses)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.19.5.7 declare reduction Directive
//===----------------------------------------------------------------------===//

def DeclareReductionOp : OpenMP_Op<"declare_reduction", [IsolatedFromAbove,
                                                         RecipeInterface,
                                                         Symbol]> {
  let summary = "declares a reduction kind";
  let description = [{
    Declares an OpenMP reduction kind. This requires two mandatory and three
    optional regions.

      1. The optional alloc region specifies how to allocate the thread-local
         reduction value. This region should not contain control flow and all
         IR should be suitable for inlining straight into an entry block. In
         the common case this is expected to contain only allocas. It is
         expected to `omp.yield` the allocated value on all control paths.
         If allocation is conditional (e.g. only allocate if the mold is
         allocated), this should be done in the initilizer region and this
         region not included. The alloc region is not used for by-value
         reductions (where allocation is implicit).
      2. The initializer region specifies how to initialize the thread-local
         reduction value. This is usually the neutral element of the reduction.
         For convenience, the region has an argument that contains the value
         of the reduction accumulator at the start of the reduction. If an alloc
         region is specified, there is a second block argument containing the
         address of the allocated memory. The initializer region is expected to
         `omp.yield` the new value on all control flow paths.
      3. The reduction region specifies how to combine two values into one, i.e.
         the reduction operator. It accepts the two values as arguments and is
         expected to `omp.yield` the combined value on all control flow paths.
      4. The atomic reduction region is optional and specifies how two values
         can be combined atomically given local accumulator variables. It is
         expected to store the combined value in the first accumulator variable.
      5. The cleanup region is optional and specifies how to clean up any memory
         allocated by the initializer region. The region has an argument that
         contains the value of the thread-local reduction accumulator. This will
         be executed after the reduction has completed.

    Note that the MLIR type system does not allow for type-polymorphic
    reductions. Separate reduction declarations should be created for different
    element and accumulator types.

    For initializer and reduction regions, the operand to `omp.yield` must
    match the parent operation's results.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type);

  let regions = (region MaxSizedRegion<1>:$allocRegion,
                        AnyRegion:$initializerRegion,
                        AnyRegion:$reductionRegion,
                        AnyRegion:$atomicReductionRegion,
                        AnyRegion:$cleanupRegion);

  let assemblyFormat = "$sym_name `:` $type attr-dict-with-keyword "
                       "( `alloc` $allocRegion^ )? "
                       "`init` $initializerRegion "
                       "`combiner` $reductionRegion "
                       "( `atomic` $atomicReductionRegion^ )? "
                       "( `cleanup` $cleanupRegion^ )? ";

  let extraClassDeclaration = [{
    BlockArgument getAllocMoldArg() {
      auto &region = getAllocRegion();
      return region.empty() ? nullptr : region.getArgument(0);
    }
    BlockArgument getInitializerMoldArg() {
      return getInitializerRegion().getArgument(0);
    }
    BlockArgument getInitializerAllocArg() {
      return getAllocRegion().empty() ?
          nullptr : getInitializerRegion().getArgument(1);
    }
    BlockArgument getReductionLhsArg() {
      return getReductionRegion().getArgument(0);
    }
    BlockArgument getReductionRhsArg() {
      return getReductionRegion().getArgument(1);
    }
    BlockArgument getAtomicReductionLhsArg() {
      auto &region = getAtomicReductionRegion();
      return region.empty() ? nullptr : region.getArgument(0);
    }
    BlockArgument getAtomicReductionRhsArg() {
      auto &region = getAtomicReductionRegion();
      return region.empty() ? nullptr : region.getArgument(1);
    }
    BlockArgument getCleanupAllocArg() {
      auto &region = getCleanupRegion();
      return region.empty() ? nullptr : region.getArgument(0);
    }

    PointerLikeType getAccumulatorType() {
      if (getAtomicReductionRegion().empty())
        return {};

      return cast<PointerLikeType>(getAtomicReductionLhsArg().getType());
    }
  }];
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// [Spec 5.2] 10.5 masked Construct
//===----------------------------------------------------------------------===//
def MaskedOp : OpenMP_Op<"masked", clauses = [
    OpenMP_FilterClause
  ], singleRegion = 1> {
  let summary = "masked construct";
  let description = [{
    Masked construct allows to specify a structured block to be executed by a subset of 
    threads of the current team.
  }] # clausesDescription;

  let builders = [
    OpBuilder<(ins CArg<"const MaskedOperands &">:$clauses)>
  ];
}

#endif // OPENMP_OPS
