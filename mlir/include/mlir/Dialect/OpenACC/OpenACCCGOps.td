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

// This file is intended to be included from OpenACCOps.td, which provides
// the necessary includes and definitions. The operations defined here use
// types and definitions from that file.

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

#endif // OPENACC_CG_OPS
