//===-- AMDGPUAttrs.td - AMDGPU dialect attributes *- tablegen -*----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_IR_AMDGPUATTRS_TD
#define MLIR_DIALECT_AMDGPU_IR_AMDGPUATTRS_TD

include "mlir/Dialect/AMDGPU/IR/AMDGPUBase.td"
include "mlir/Dialect/AMDGPU/IR/AMDGPUEnums.td"

def AMDGPU_AddressSpaceAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_AddressSpace,
    "address_space"> {
  let description = [{
    AMDGPU-specific memory spaces that may not have exact analogues on other
    GPU targets or backends.

    - `fat_raw_buffer` is the memory space used when a memref is stored as
    as a "buffer fat pointer" - that is, a buffer resource (that is set up to
    use raw byte-level indexing) along with its offset. The AMDGPU backend
    implements `ptr addrspace(7)` to represent these fat pointers so that
    buffer resources (which allow advanced features like bounds checking or
    cache swizzling) can be used like ordinary LLVM pointers or memrefs.
    See also the `fat_raw_buffer_cast` operation
    - `buffer_rsrc` is the memory space for `ptr addrspace(8)`, representing a
    buffer resource. It should not be used for memrefs, since it does not support
    indexing
    - `fat_structured_buffer` represents `ptr addrspace(9)`, a buffer resource
    that carries both an index and offset field, which are used for complex
    structured indexing that is primarily seen in graphics applications. This
    is also incompatible with the simple indexing model supported by memref.
  }];
  let assemblyFormat = "`<` $value `>`";
}

def AMDGPU_DPPPermAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_DPPPerm,
  "dpp_perm">;

def AMDGPU_SchedBarrierOpOptAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_SchedBarrierOpOpt,
  "sched_barrier_opt">{
   let assemblyFormat = "`<` $value `>`";
}

def AMDGPU_MFMAPermBAttr : EnumAttr<AMDGPU_Dialect, AMDGPU_MFMAPermB,
  "mfma_perm_b">;

#endif // MLIR_DIALECT_AMDGPU_IR_AMDGPUATTRS_TD
