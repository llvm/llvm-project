//===-- AMDGPUEnums.td - AMDGPU dialect enums *- tablegen -*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD
#define MLIR_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD

include "mlir/Dialect/AMDGPU/IR/AMDGPUBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/Properties.td"

//===----------------------------------------------------------------------===//
// AMDGPU general enum  definitions
//===----------------------------------------------------------------------===//

def AMDGPU_AddressSpace : I32Enum<"AddressSpace",
    "AMDGPU-specific address spaces",
    [
      I32EnumCase<"FatRawBuffer",        0, "fat_raw_buffer">,
      I32EnumCase<"BufferRsrc",          1, "buffer_rsrc">,
      I32EnumCase<"FatStructuredBuffer", 2, "fat_structured_buffer">,
    ]> {
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_DPPPerm : I32Enum<"DPPPerm",
    "The possible permutations for a DPP operation",
    [
      I32EnumAttrCase<"quad_perm",  0>,
      I32EnumAttrCase<"row_shl",    1>,
      I32EnumAttrCase<"row_shr",    2>,
      I32EnumAttrCase<"row_ror",    3>,
      I32EnumAttrCase<"wave_shl",   4>,
      I32EnumAttrCase<"wave_shr",   5>,
      I32EnumAttrCase<"wave_ror",   6>,
      I32EnumAttrCase<"wave_rol",   7>,
      I32EnumAttrCase<"row_mirror", 8>,
      I32EnumAttrCase<"row_half_mirror", 9>,
      I32EnumAttrCase<"row_bcast_15", 10>,
      I32EnumAttrCase<"row_bcast_31", 11>
    ]> {
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_SchedBarrierOpOpt : I32BitEnum<"sched_barrier_opt_enum",
    "The possible options for scheduling barriers",
    [
      I32BitEnumAttrCaseNone<"none">,
      I32BitEnumAttrCaseBit<"non_mem_non_sideffect", 0>,
      I32BitEnumAttrCaseBit<"valu", 1>,
      I32BitEnumAttrCaseBit<"salu", 2>,
      I32BitEnumAttrCaseBit<"mfma_wmma",  3>,
      I32BitEnumAttrCaseBit<"all_vmem",  4>,
      I32BitEnumAttrCaseBit<"vmem_read",  5>,
      I32BitEnumAttrCaseBit<"vmem_write", 6>,
      I32BitEnumAttrCaseBit<"all_ds", 7>,
      I32BitEnumAttrCaseBit<"ds_read", 8>,
      I32BitEnumAttrCaseBit<"ds_write", 9>,
      I32BitEnumAttrCaseBit<"transcendental", 10>
    ]> {
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_MFMAPermB : I32Enum<"MFMAPermB",
    "The possible permutations of the lanes storing B available in an MFMA",
    [
      I32EnumAttrCase<"none",            0>,
      I32EnumAttrCase<"bcast_first_32",  1>,
      I32EnumAttrCase<"bcast_second_32", 2>,
      I32EnumAttrCase<"rotate_16_right", 3>,
      I32EnumAttrCase<"bcast_first_16",  4>,
      I32EnumAttrCase<"bcast_second_16", 5>,
      I32EnumAttrCase<"bcast_third_16",  6>,
      I32EnumAttrCase<"bcast_fourth_16", 7>
    ]> {
  let cppNamespace = "::mlir::amdgpu";
}

#endif // MLIR_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD
