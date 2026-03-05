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

    The var operand is the original/shared reduction variable. The
    reduction_operator specifies the reduction kind (e.g. add, mul).
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

    The destVar operand is the original/shared reduction variable.
    The srcVar operand is typically the result of acc.reduction_init.
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
    
    The $destMemref operand is a "pointer" to the original reduction 
    variable (typically shared). The $srcMemref operand is a "pointer"
    to the partial sum of the reduction (typically private).

    The $kind is the OpenACC reduction operator that determines how to 
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
    Variadic<IntOrIndex>:$asyncOperands,
    OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
    OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
    Variadic<IntOrIndex>:$waitOperands,
    OptionalAttr<DenseI32ArrayAttr>:$waitOperandsSegments,
    OptionalAttr<DeviceTypeArrayAttr>:$waitOperandsDeviceType,
    OptionalAttr<BoolArrayAttr>:$hasWaitDevnum,
    OptionalAttr<DeviceTypeArrayAttr>:$waitOnly);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
      | `wait` `` custom<WaitClause>($waitOperands, type($waitOperands),
          $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,
          $waitOnly)
    )
    $region attr-dict
  }];

  let hasCanonicalizer = 1;
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
  let results = (outs OpenACC_ParWidthType:$output);
  let assemblyFormat = [{
    ($launchArg^)? attr-dict
  }];
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

    - Launch arguments (`launch`): Results of operations that define
      the parallel launch configuration. These are `!acc.par_width`-typed
      and become block arguments representing the parallel width for each
      dimension.

    - Input arguments (`ins`): Arbitrary values captured from outside
      the region (data pointers, scalars, etc.). These become block
      arguments with their original types.

    The `origin` attribute records which construct produced this compute
    region (e.g., `"acc.parallel"`, `"acc.kernels"`). This is intended to
    be solely informational.

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

  let arguments = (ins Variadic<OpenACC_ParWidthType>:$launchArgs,
                       Variadic<AnyType>:$inputArgs,
                       Optional<OpenACC_GPUAsyncTokenType>:$stream,
                       StrAttr:$origin,
                       OptionalAttr<FlatSymbolRefAttr>:$kernel_func_name,
                       OptionalAttr<FlatSymbolRefAttr>:$kernel_module_name);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);

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

    /// Map a block argument back to its corresponding operand
    /// ($launchArgs or $inputArgs).
    ::mlir::Value getOperand(::mlir::BlockArgument blockArg);
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // OPENACC_CG_OPS
