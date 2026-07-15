//===- OpenACCCGOps.td - OpenACC intermediate operations ---*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR OpenACC intermediate operations used to represent semantics
// when decomposing and lowering `acc` dialect operations.
// These operations are placed in a separate file because they do not represent
// direct mappings of OpenACC language constructs.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_CG_OPS
#define OPENACC_CG_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/CommonAttrConstraints.td"

// This file is intended to be included from OpenACCOps.td, which provides
// the necessary includes and definitions. The operations defined here use
// types and definitions from that file.

//===----------------------------------------------------------------------===//
// acc.reduction_init
//===----------------------------------------------------------------------===//

def OpenACC_ReductionInitOp
    : OpenACC_Op<"reduction_init",
                 [SameOperandsAndResultType, RecursiveMemoryEffects,
                  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                            ["getRegionInvocationBounds",
                                             "getSuccessorInputs"]>,
                  SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Allocate and initialize a reduction variable from a recipe";
  let description = [{
    This operation provides materialized allocation and initialization for a
    private reduction variable from an OpenACC reduction recipe. The region
    contains the recipe's init code and must yield a single value (the private
    reduction storage) via `acc.yield`.

    The `var` operand is the original/shared reduction variable. The
    `reduction_operator` specifies the reduction kind (e.g. add, mul).
  }];
  let arguments = (ins OpenACC_AnyPointerOrMappableType:$var,
                       OpenACC_ReductionOperatorAttr:$reductionOperator);
  let results = (outs OpenACC_AnyPointerOrMappableType:$result);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    $var $reductionOperator `:` type($result) $region attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// acc.reduction_combine_region
//===----------------------------------------------------------------------===//

def OpenACC_ReductionCombineRegionOp
    : OpenACC_Op<"reduction_combine_region",
                 [SameTypeOperands, RecursiveMemoryEffects,
                  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                            ["getRegionInvocationBounds",
                                             "getSuccessorInputs"]>,
                  SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Combine a reduction private value with its original (recipe)";
  let description = [{
    This operation provides materialized reduction combine code from an
    OpenACC reduction recipe. The region takes the partially reduced value(s)
    from the private reduction variable and combines them with the current
    value(s) in the original/shared reduction variable. The region is
    terminated by `acc.yield` with no operands.

    The `destVar` operand is the original/shared reduction variable.
    The `srcVar` operand is typically the result of acc.reduction_init.
  }];
  let arguments = (ins OpenACC_AnyPointerOrMappableType:$destVar,
                       OpenACC_AnyPointerOrMappableType:$srcVar);
  let results = (outs);
  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    $srcVar `into` $destVar `:` type($srcVar) $region
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// acc.reduction_combine
//===----------------------------------------------------------------------===//

def OpenACC_ReductionCombineOp: OpenACC_Op<"reduction_combine", 
    [SameTypeOperands, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Combine a reduction partial sum with its original value";
  let description = [{
    This operation is a composite to do a typical update of a reduction
    variable. The intention of this operator is to facilitate codegen
    decisions (such as generate an atomic update). E.g.

    ```
      acc.reduction_combine %src into %dest <addi> : memref<i32>
    ```

    Might lower to something similar to

    ```
      %loadSrc = memref.load %src[] : memref<i32>
      %loadDest = memref.load %dest[] : memref<i32>
      %combine = arith.addi %loadSrc, %loadDest : i32
      memref.store %combine, %dest[] : memref<i32>
    ```
    
    The `destMemref` operand is a "pointer" to the original reduction 
    variable (typically shared). The `srcMemref` operand is a "pointer"
    to the partial sum of the reduction (typically private).

    The `kind` is the OpenACC reduction operator that determines how to 
    accumulate the two values.
  }];

  let arguments = (ins OpenACC_AnyPointerOrMappableType:$destMemref,
                       OpenACC_AnyPointerOrMappableType:$srcMemref,
                       OpenACC_ReductionOperatorAttr:$reductionOperator);
  
  let assemblyFormat = [{
    $srcMemref `into` $destMemref $reductionOperator `:` type($destMemref) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// acc.reduction_accumulate
//===----------------------------------------------------------------------===//

def OpenACC_ReductionAccumulateOp
    : OpenACC_Op<"reduction_accumulate", []> {
  let summary = "Accumulate an SSA value into a reduction variable";
  let description = [{
    Accumulates a scalar SSA value into a pointer-like reduction variable.

    Example:
    ```mlir
    %private = memref.alloca() {acc.par_dims = #acc<par_dims[thread_x]>} : memref<i32>
    memref.store %c0, %private[] : memref<i32>
    %partial = scf.parallel (%iv) = (%c0) to (%cN) step (%c1) init (%c0) -> i32 {
      %v = memref.load %data[%iv] : memref<Nxi32>
      scf.reduce(%v : i32) {
      ^bb0(%lhs: i32, %rhs: i32):
        %sum = arith.addi %lhs, %rhs : i32
        scf.reduce.return %sum : i32
      }
    } {acc.par_dims = #acc<par_dims[thread_x]>}
    acc.reduction_accumulate %partial to %private <add>
        : i32 -> memref<i32> {par_dims = #acc<par_dims[thread_x]>}
    acc.reduction_combine %private into %shared <add> : memref<i32>
        {acc.par_dims = #acc<par_dims[thread_x]>}
    ```
  }];
  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat, AnyComplex]>:$value,
                       Arg<OpenACC_PointerLikeType,
                           "Reduction variable to update",
                           [MemRead, MemWrite]>:$memref,
                       OpenACC_ReductionOperatorAttr:$reductionOperator,
                       OpenACC_GPUParallelDimsAttr:$par_dims);
  let assemblyFormat = [{
    $value `to` $memref $reductionOperator `:` type($value) `->` type($memref) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// acc.reduction_accumulate_array
//===----------------------------------------------------------------------===//

def OpenACC_ReductionAccumulateArrayOp
    : OpenACC_Op<"reduction_accumulate_array", []> {
  let summary = "Accumulates elements of an array";
  let description = [{
    Accumulates elements of an array across the specified parallel dimension 
    given an acc.bounds op.
  }];
  let arguments = (ins Arg<OpenACC_PointerLikeTypeInterface,
                           "Reduction variable to update",
                           [MemRead, MemWrite]>:$memref,
                       OpenACC_DataBoundsType:$bounds,
                       OpenACC_ReductionOperatorAttr:$reductionOperator,
                       OpenACC_GPUParallelDimsAttr:$par_dims);
  let assemblyFormat = [{
    $memref `bounds` `(` $bounds `)` $reductionOperator `:` type($memref) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// acc.kernel_environment
//===----------------------------------------------------------------------===//

def OpenACC_KernelEnvironmentOp
    : OpenACC_Op<"kernel_environment",
                 [AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock,
                  NoTerminator,
                  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                            ["getSuccessorInputs"]>,
                  MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                                 MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
  let summary = "Decomposition of compute constructs to capture data mapping "
                "and asynchronous behavior information";
  let description = [{
    The `acc.kernel_environment` operation represents a decomposition of
    any OpenACC compute construct (acc.kernels, acc.parallel, or
    acc.serial) that captures data mapping and asynchronous behavior:
    - data clause operands
    - async clause operands
    - wait clause operands

    This allows kernel execution parallelism and privatization to be
    handled separately, facilitating eventual lowering to GPU dialect where
    kernel launching and compute offloading are handled separately.
  }];

  let arguments = (ins
    Variadic<AnyType>:$dataClauseOperands,
    Optional<IntOrIndex>:$asyncOperand,
    UnitAttr:$asyncOnly,
    Optional<IntOrIndex>:$waitDevnum,
    Variadic<IntOrIndex>:$waitOperands,
    UnitAttr:$waitOnly);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `` custom<OperandWithKeywordOnly>($asyncOperand,
            type($asyncOperand), $asyncOnly)
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `` custom<OperandsWithKeywordOnly>($waitOperands,
            type($waitOperands), $waitOnly)
    )
    $region attr-dict
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Create a `KernelEnvironmentOp` populated with data mapping and the
    /// async/wait clauses for `deviceType` extracted from the given ACC compute
    /// construct. Emplaces a block in the region and sets the rewriter's
    /// insertion point to the start of that block so callers can create
    /// operations inside it (e.g., `acc.compute_region`).
    template <typename ComputeConstructT>
    static KernelEnvironmentOp createAndPopulate(
        ComputeConstructT computeConstruct, ::mlir::acc::DeviceType deviceType,
        ::mlir::OpBuilder &builder);
  }];
}

//===----------------------------------------------------------------------===//
// acc.firstprivate_map
//===----------------------------------------------------------------------===//

def OpenACC_FirstprivateMapInitialOp
    : OpenACC_DataEntryOp<
          "firstprivate_map", "mlir::acc::DataClause::acc_firstprivate", "", [],
          (ins Arg<OpenACC_AnyPointerOrMappableType, "Host variable">:$var)> {
  let summary = "Represents the mapping of the initial value for firstprivate "
                "semantics.";
  let description = [{
    The `acc.firstprivate_map` operation is an intermediate representation
    used during the decomposition of `acc.firstprivate` operations. It
    represents the mapping of the initial value from the host to the device,
    which is then used to initialize per-thread private copies.

    This operation is distinct from `acc.copyin` because:
    - `acc.copyin` includes present counter updates, but private variables
      do not impact reference counters
    - The mapped value is used to initialize private copies rather than
      being accessed directly
  }];
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable">:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// acc.privatize
//===----------------------------------------------------------------------===//

def OpenACC_PrivatizeOp : OpenACC_Op<"privatize", []> {
  let summary = "Create a handle for privatized storage along parallel dimensions";
  let description = [{
    Introduces a privatization handle for storage that varies across the active
    parallel dimensions (for example OpenACC `private` / `firstprivate` after
    recipe materialization). The handle type is `acc.private_type<T>` where
    `T` is the logical storage type (commonly a `memref`).

    Optional `index` operands supply dynamic sizes when the privatized shape
    depends on SSA values (for example `memref<?xi32>` row lengths).

    The optional `acc.par_dims` attribute records which GPU parallel dimensions
    participate in the privatization.
  }];
  let arguments = (ins Variadic<Index>:$dynamicSizes,
                       OptionalAttr<OpenACC_GPUParallelDimsAttr>:$par_dims);
  let results = (outs OpenACC_PrivateType:$result);
  let assemblyFormat = [{
    (`(` $dynamicSizes^ `)`)? (`[` qualified($par_dims)^ `]`)? attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Type":$resultType,
                   CArg<"::mlir::ValueRange", "{}">:$dynamicSizes), [{
      $_state.addTypes(resultType);
      $_state.addOperands(dynamicSizes);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// acc.unwrap_private
//===----------------------------------------------------------------------===//

def OpenACC_UnwrapPrivateOp
    : OpenACC_Op<"unwrap_private",
                 [NoMemoryEffect, AlwaysSpeculatable, ViewLikeOpInterface]> {
  let summary = "Unwrap acc.private_type handle to pointer-like storage";
  let description = [{
    Converts a privatization handle (`acc.private_type<T>`) to a pointer-like
    view of the underlying private storage.
  }];
  let arguments = (ins OpenACC_PrivateType:$handle);
  let results = (outs OpenACC_PointerLikeType:$result);
  let assemblyFormat = [{
    $handle attr-dict `:` qualified(type($handle)) `to` type($result)
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getHandle(); }
  }];
}

//===----------------------------------------------------------------------===//
// acc.private_local
//===----------------------------------------------------------------------===//

def OpenACC_PrivateLocalOp
    : OpenACC_Op<"private_local", [NoMemoryEffect, AlwaysSpeculatable]> {
  let summary = "Materialize privatized storage for the current parallelism context";
  let description = [{
    Given a value of type `acc.private_type<T>`, materializes the underlying
    storage (typically a `memref`) for the current thread / iteration. Which
    slice of the privatized allocation is selected is determined by surrounding
    parallelism assigned in the context.

    The result type is usually `T` (often a `memref` that matches the logical
    storage type). The result may instead use a different surface type that
    still aliases the same underlying storage as `T`; in that case passes that
    consume this operation must treat the handle type and the result type as
    describing the same storage layout.
  }];
  let arguments = (ins OpenACC_PrivateType:$privatized);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $privatized attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::Value":$privatized), [{
      $_state.addOperands(privatized);
      $_state.addTypes(resultType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// acc.par_width
//===----------------------------------------------------------------------===//

def OpenACC_ParWidthOp
    : OpenACC_Op<"par_width", [NoMemoryEffect, AlwaysSpeculatable]> {
  let summary = "Specify parallel width for a GPU dimension";
  let description = [{
    The `acc.par_width` operation specifies the parallel width for a
    given GPU parallel dimension. It is used as an input to
    `acc.compute_region` to define the launch configuration.

    The optional `launchArg` operand provides a known width value. When
    absent, the width is unknown and must be determined later (either at
    compile time by analysis or at runtime).

    Examples:

    ```mlir
    // Known width from SSA value
    %w1 = acc.par_width %vector_len {par_dim = #acc.par_dim<thread_x>}

    // Unknown width (to be computed later)
    %w2 = acc.par_width {par_dim = #acc.par_dim<block_x>}
    ```
  }];
  let arguments = (ins Optional<Index>:$launchArg,
                       OpenACC_GPUParallelDimAttr:$par_dim);
  let results = (outs Index:$output);
  let assemblyFormat = [{
    ($launchArg^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// acc.gpu_shared_memory
//===----------------------------------------------------------------------===//

def OpenACC_GPUSharedMemoryOp : OpenACC_Op<"gpu_shared_memory"> {
  let summary = "GPU workgroup (shared) memory allocation in a compute region";
  let description = [{
    Represents a GPU workgroup-memory allocation in a compute region.
    The result is a typed `memref` view into a byte slab which is later
    replaced by `memref.view` into a dynamic shared-memory blob at the
    byte offset within the workgroup allocation.

    Each operation occupies a distinct slot in that collective allocation.
    `static_upper_bound_bytes` is the conservative byte-size upper bound for
    the slot. `dynamic_sizes` supply values for dynamic memref result
    dimensions. The in-kernel layout of the slot is given by
    `static_upper_bound_bytes` and `dynamic_sizes`.

    Optional `dynamic_shared_memory_scaling_bytes` and
    `dynamic_shared_memory_fixed_bytes` parameterize
    `dynamic_shared_memory_size` when the slot footprint depends on launch
    geometry. They must be specified together. When present:

      dynamic_shared_memory_size =
          dynamic_shared_memory_scaling_bytes * W
          + dynamic_shared_memory_fixed_bytes

    where `W` is the launch width that scales the allocation.

    This linear model arises for `acc.cache` regions with dynamic bounds: one
    cache dimension grows with the thread-parallel launch width, while
    `dynamic_shared_memory_fixed_bytes` covers bytes that do not scale (for
    example overlap cells at cache tile boundaries so threads can read
    neighboring source elements without extra global memory traffic).
    For a 1D dynamic cache with stencil extent `E`:

      dynamic_shared_memory_fixed_bytes =
          (E - 1) * dynamic_shared_memory_scaling_bytes

    ```
    Global array:  ... | a | b | c | d | e | f | ...
                          [---- cached tile ----]
    Thread 0 primary:     a       (reads neighbor b)
    Thread 1 primary:         b   (reads neighbors a, c)
    ...
    Scaling portion:  dynamic_shared_memory_scaling_bytes * W
    Fixed portion:    dynamic_shared_memory_fixed_bytes  ((E - 1) cells)
    ```

    The scaling attributes affect only `dynamic_shared_memory_size`, not
    the slot layout. For purely static allocations both are omitted and
    `dynamic_shared_memory_size` is the sum of aligned
    `static_upper_bound_bytes` across all slots.

    Example:

    ```mlir
    %sz = arith.constant 128 : index
    %cache = acc.gpu_shared_memory(%sz)
        {num_copies = 1 : i64,
         static_upper_bound_bytes = 1560 : i64,
         dynamic_shared_memory_scaling_bytes = 12 : i64,
         dynamic_shared_memory_fixed_bytes = 24 : i64}
        : (index) -> memref<?xf32, #gpu.address_space<workgroup>>
    ```
  }];
  let arguments = (ins
    I64Attr:$num_copies,
    I64Attr:$static_upper_bound_bytes,
    Variadic<Index>:$dynamic_sizes,
    OptionalAttr<I64Attr>:$dynamic_shared_memory_scaling_bytes,
    OptionalAttr<I64Attr>:$dynamic_shared_memory_fixed_bytes
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc<DefaultResource, 0,
                                           FullEffect>]>:$result);
  let assemblyFormat = [{
    (`(` $dynamic_sizes^ `)`)? attr-dict `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Type":$resultType,
                   "::mlir::IntegerAttr":$numCopies,
                   "::mlir::IntegerAttr":$staticUpperBoundBytes,
                   CArg<"::mlir::ValueRange", "{}">:$dynamicSizes,
                   CArg<"::mlir::IntegerAttr", "{}">:$dynamicSharedMemoryScalingBytes,
                   CArg<"::mlir::IntegerAttr", "{}">:$dynamicSharedMemoryFixedBytes), [{
      $_state.addAttribute("num_copies", numCopies);
      $_state.addAttribute("static_upper_bound_bytes", staticUpperBoundBytes);
      $_state.addOperands(dynamicSizes);
      if (dynamicSharedMemoryScalingBytes)
        $_state.addAttribute("dynamic_shared_memory_scaling_bytes",
                            dynamicSharedMemoryScalingBytes);
      if (dynamicSharedMemoryFixedBytes)
        $_state.addAttribute("dynamic_shared_memory_fixed_bytes",
                            dynamicSharedMemoryFixedBytes);
      $_state.addTypes(resultType);
    }]>
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// acc.compute_region
//===----------------------------------------------------------------------===//

// Local type constraint for gpu::AsyncTokenType.
def OpenACC_GPUAsyncTokenType : Type<
  CPred<"::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)">,
  "GPU async token type">;

def OpenACC_ComputeRegionOp
    : OpenACC_Op<"compute_region",
                 [OffloadRegionOpInterface, AffineScope,
                  RecursiveMemoryEffects,
                  SingleBlockImplicitTerminator<"YieldOp">,
                  IsolatedFromAbove, AttrSizedOperandSegments]> {
  let summary = "Compute region for GPU execution";
  let description = [{
    The `acc.compute_region` operation wraps a region of code that will be
    compiled and executed on a GPU. It is typically produced by lowering
    OpenACC compute constructs (`acc.parallel`, `acc.kernels`, `acc.serial`)
    but can also be targeted directly by other frontends or lowered from
    other constructs that benefit from the automatic parallelization and data
    mapping facilities that the `acc` dialect provides. It serves as the
    bridge between the high-level representation and the `gpu.launch`
    operation.

    The operation is `IsolatedFromAbove`: all values used inside the
    region must be explicitly captured. Values are captured in two ways:

    - Launch arguments (`launch`): Results of `acc.par_width`
      operations that define the parallel launch configuration. These
      become `index`-typed block arguments representing the parallel
      width for each dimension.

    - Input arguments (`ins`): Arbitrary values captured from outside
      the region (data pointers, scalars, etc.). These become block
      arguments with their original types.

    The `origin` attribute records which construct produced this compute
    region (e.g., `"acc.parallel"`, `"acc.kernels"`). This is intended to
    be solely informational.

    Canonicalization may simplify `ins` captures: duplicate `ins` operands
    (same SSA value threaded more than once) are merged by reusing the first
    block argument, and unused `ins` operands (block arguments with no uses)
    are removed. `launch` operands are never merged or dropped.

    Example:

    ```mlir
    %w0 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    %w1 = acc.par_width %c8 {par_dim = #acc.par_dim<block_x>}
    acc.compute_region launch(%arg0 = %w0, %arg1 = %w1)
        ins(%arg2 = %data) : (memref<1024xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      scf.parallel (%iv) = (%c0) to (%c1024) step (%c1) {
        %v = memref.load %arg2[%iv] : memref<1024xf32>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      acc.yield
    } {origin = "acc.parallel"}
    ```
  }];

  let arguments = (ins Variadic<Index>:$launchArgs,
                       Variadic<AnyType>:$inputArgs,
                       Optional<OpenACC_GPUAsyncTokenType>:$stream,
                       StrAttr:$origin,
                       OptionalAttr<FlatSymbolRefAttr>:$kernel_func_name,
                       OptionalAttr<FlatSymbolRefAttr>:$kernel_module_name);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Look up the par_width op for the given dimension among launch args.
    std::optional<mlir::Value> getLaunchArg(
        ::mlir::acc::GPUParallelDimAttr parDim);

    /// Get the known (non-empty) launch value for a dimension.
    std::optional<mlir::Value> getKnownLaunchArg(
        ::mlir::acc::GPUParallelDimAttr parDim);

    /// Get the known constant launch value for a dimension.
    std::optional<uint64_t> getKnownConstantLaunchArg(
        ::mlir::acc::GPUParallelDimAttr parDim);

    /// Add a new input argument, appending to both the operand list and
    /// the region block arguments. Returns the new block argument.
    ::mlir::BlockArgument appendInputArg(::mlir::Value);

    /// After hoisting `value`'s defining op, wire it into this region: append it
    /// as an `ins` operand, add the matching body entry argument, and replace
    /// uses under this region with that argument (excluding this op's own `ins`
    /// operand uses).
    ///
    /// Requires that `value` is defined outside `getRegion()` and is still used
    /// inside the region. Otherwise returns `std::nullopt`.
    std::optional<::mlir::BlockArgument>
    wireHoistedValueThroughIns(::mlir::Value value);

    /// Check whether all parallel dimensions have width 1.
    bool isEffectivelySerial();

    /// Get the block argument representing the width for a given dimension.
    ::mlir::BlockArgument parDimToWidth(
        ::mlir::acc::GPUParallelDimAttr parDim);

    /// Get the block argument for a specific gpu::Processor.
    ::mlir::BlockArgument gpuParWidth(::mlir::gpu::Processor);

    /// Collect all GPU parallel dimensions present in the launch config.
    llvm::SmallVector<::mlir::acc::GPUParallelDimAttr> getLaunchParDims();

    /// Get the body block of the compute region.
    ::mlir::Block *getBody() { return &getRegion().front(); }

    /// Get the terminator of the compute region.
    ::mlir::Operation *getTerminator() {
      return &getRegion().back().back();
    }

    /// Return the `launch` or `ins` operand threaded to `blockArg`, or a null
    /// `Value` if `blockArg` is not an argument of `getBody()` or its index is
    /// out of range for this op's `launch` and `ins` operands.
    ::mlir::Value getOperand(::mlir::BlockArgument blockArg);

    /// If `value` is a launch or input operand, return the body block argument
    /// it is threaded through; otherwise `std::nullopt`. If `value` matches
    /// more than one `ins` operand, the first match is returned (canonicalization
    /// may merge duplicate `ins` values). Duplicate `launch` operands are not
    /// folded.
    std::optional<::mlir::BlockArgument> getBlockArg(::mlir::Value value);
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// acc.predicate_region
//===----------------------------------------------------------------------===//

def OpenACC_PredicateRegionOp
    : OpenACC_Op<"predicate_region", [RecursiveMemoryEffects, NoTerminator]> {
  let summary = "Groups operations at intermediate loop-nest points";
  let description = [{
    Groups statements within an `acc.compute_region` that sit at an
    intermediate point in a loop nest (outside a partitioned loop body,
    between or around nested loops). This grouping marks code whose execution
    scope differs from that of surrounding partitioned loops, so predication
    and synchronization can be applied correctly during lowering. Corresponds
    to OpenACC single or redundant execution at nest transitions.

    Example:
    ```mlir
    // !$acc parallel num_gangs(NG) vector_length(VL)
    // !$acc loop gang
    //   !$acc atomic update
    //   !$acc loop vector
    //     !$acc atomic update
    %w_gang = acc.par_width %cNG {par_dim = #acc.par_dim<block_x>}
    %w_vector = acc.par_width %cVL {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%ng = %w_gang, %vl = %w_vector)
        ins(%arg_c1 = %c1, %arg_c2 = %c2) : (memref<i32>, memref<i32>) {
      scf.parallel (%i) = (%c1) to (%cN) step (%c1) {
        acc.predicate_region {
          acc.atomic.update %arg_c1 : memref<i32> {
          ^bb0(%old: i32):
            %one = arith.constant 1 : i32
            %sum = arith.addi %old, %one : i32
            acc.yield %sum : i32
          }
        }
        scf.parallel (%j) = (%c1) to (%cN) step (%c1) {
          acc.atomic.update %arg_c2 : memref<i32> {
          ^bb0(%old: i32):
            %one = arith.constant 1 : i32
            %sum = arith.addi %old, %one : i32
            acc.yield %sum : i32
          }
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
    ```
  }];
  let regions = (region AnyRegion:$region);
  let assemblyFormat = "$region attr-dict";
  let hasVerifier = 1;
}

#endif // OPENACC_CG_OPS
