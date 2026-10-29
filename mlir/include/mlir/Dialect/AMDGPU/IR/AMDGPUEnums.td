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

def AMDGPU_LoadTemporalHint : I32Enum<"LoadTemporalHint",
    "AMDGPU-specific prefetch temporal hints for load instructions. "
    "RT - regular temporal for both near and far caches; "
    "NT - non-temporal for both near and far caches; "
    "HT - high-priority temporal for both near and far caches; "
    "LU - last-use; "
    "NT_RT - non-temporal for near cache(s) and regular for far caches; "
    "RT_NT - regular for near cache(s) and non-temporal for far caches; "
    "NT_HT - non-temporal for near cache(s) and high-priority temporal for far caches",
    [
      I32EnumCase<"RT",    0>,
      I32EnumCase<"NT",    1>,
      I32EnumCase<"HT",    2>,
      I32EnumCase<"LU",    3>,
      I32EnumCase<"NT_RT", 4>,
      I32EnumCase<"RT_NT", 5>,
      I32EnumCase<"NT_HT", 6>
    ]> {
  let cppNamespace = "::mlir::amdgpu";
}

def AMDGPU_CacheScope : I32Enum<"Scope",
    "AMDGPU-specific cache scopes. "
    "WGP - workgroup processor (CUs); "
    "SE - shader engine (GL2); "
    "DEV - device; "
    "SYS - system",
    [
      I32EnumCase<"WGP",    0>,
      I32EnumCase<"SE",     1>,
      I32EnumCase<"DEV",    2>,
      I32EnumCase<"SYS",    3>
    ]> {
  let cppNamespace = "::mlir::amdgpu";
}

#endif // MLIR_DIALECT_AMDGPU_IR_AMDGPUENUMS_TD
