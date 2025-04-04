//===- SPIRVIntelExtEmbargoOps.td - Intel SPIR-V extensions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of Intel-specific SPIR-V extensions
// These extensions are not part of Khronos specification and available under
// Embargo.
// Supported extensions
// * SPV_INTEL_region_group
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_SPIRV_IR_INTEL_EXT_EMBARGO_OPS
#define MLIR_DIALECT_SPIRV_IR_INTEL_EXT_EMBARGO_OPS

// -----

def SPIRV_INTELSubRegionControlBarrierOp : SPIRV_IntelVendorOp<"SubRegionControlBarrier", []> {
  let summary = "See extension SPV_INTEL_region_group";

  let description = [{
    Wait for all active invocations within the current sub-region to reach
    the current point of execution.

    All active invocations within the current sub-region reach this point of
    execution before any invocation proceeds beyond it.

    A sub-region is a subset of the workgroups in a region group. The region
    group is partitioned into groups of SubRegionSize workgroups, and
    the workgroups are ordered by their linearized ID. The first SubRegionSize
    workgroups in this sequence are the first sub-region, the next
    SubRegionSize workgroups are the next sub-region, etc. The total number of
    workgroups in the region-group must be evenly divisible by SubRegionSize,
    otherwise the behavior is undefined.

    Behavior is undefined unless all invocations within the current sub-region
    execute the same dynamic instance of this instruction. SubRegionSize value
    must be the same for all invocations within the current sub-region,
    or otherwise behavior is undefined.

    If Semantics is not None, this instruction also serves as an
    OpMemoryBarrier instruction, and also performs and adheres to the
    description and semantics of an OpMemoryBarrier instruction with the
    same Memory and Semantics operands. This allows atomically specifying
    both a control barrier and a memory barrier (that is, without needing
    two instructions). If Semantics is None, Memory is ignored.

    #### Example:

    ```mlir
    spirv.SubRegionControlBarrier %0, "RegionINTEL", "None"
    ```

  }];


  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_region_group]>,
    Capability<[SPIRV_C_RegionGroupINTEL]>
  ];

  let arguments = (ins
    SPIRV_Int32:$subregion_size,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$memory_semantics
  );

  let assemblyFormat = [{
    $subregion_size `,` $memory_scope `,` $memory_semantics attr-dict
  }];

  let results = (outs);

  let hasVerifier = 0;
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_INTEL_EXT_EMBARGO_OPS
