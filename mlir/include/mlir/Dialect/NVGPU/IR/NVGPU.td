//===-- NVGPU.td - Attribute defs for NVGPU dialect *- tablegen -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NVGPU_IR_NVGPU_TD
#define MLIR_DIALECT_NVGPU_IR_NVGPU_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def NVGPU_Dialect : Dialect {
  let name = "nvgpu";
  let cppNamespace = "::mlir::nvgpu";
  let description = [{
    The `NVGPU` dialect provides a bridge between higher-level target-agnostic
    dialects (GPU and Vector) and the lower-level target-specific dialect
    (LLVM IR based NVVM dialect) for NVIDIA GPUs. This allow representing PTX
    specific operations while using MLIR high level dialects such as Memref
    and Vector for memory and target-specific register operands, respectively.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  
  let extraClassDeclaration = [{
    /// Return true if the given MemRefType has an integer address
    /// space that matches the NVVM shared memory address space or
    /// is a gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool hasSharedMemoryAddressSpace(MemRefType type);

    /// Return true if the given Attribute has an integer address
    /// space that matches the NVVM shared memory address space or
    /// is a gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool isSharedMemoryAddressSpace(Attribute type);

    /// Defines the MemRef memory space attribute numeric value that indicates
    /// a memref is located in global memory. This should correspond to the
    /// value used in NVVM.
    static constexpr unsigned kGlobaldMemoryAddressSpace = 1;

    /// Defines the MemRef memory space attribute numeric value that indicates
    /// a memref is located in shared memory. This should correspond to the
    /// value used in NVVM.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
  }];
}

//===----------------------------------------------------------------------===//
// NVGPU Attribute Definitions
//===----------------------------------------------------------------------===//

def TensorMapSwizzleNone : I32EnumAttrCase<"SWIZZLE_NONE", 0, "none">;
def TensorMapSwizzle32B  : I32EnumAttrCase<"SWIZZLE_32B", 1, "swizzle_32b">;
def TensorMapSwizzle64B  : I32EnumAttrCase<"SWIZZLE_64B", 2, "swizzle_64b">;
def TensorMapSwizzle128B : I32EnumAttrCase<"SWIZZLE_128B", 3, "swizzle_128b">;
def TensorMapSwizzleKind : I32EnumAttr<"TensorMapSwizzleKind", 
                                "Tensor map swizzling mode of shared memory banks",
  [ TensorMapSwizzleNone, TensorMapSwizzle32B, TensorMapSwizzle64B, 
    TensorMapSwizzle128B]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapL2PromoNone : I32EnumAttrCase<"L2PROMO_NONE", 0, "none">;
def TensorMapL2Promo64B  : I32EnumAttrCase<"L2PROMO_64B", 1, "l2promo_64b">;
def TensorMapL2Promo128B : I32EnumAttrCase<"L2PROMO_128B", 2, "l2promo_128b">;
def TensorMapL2Promo256B : I32EnumAttrCase<"L2PROMO_256B", 3, "l2promo_256b">;
def TensorMapL2PromoKind : I32EnumAttr<"TensorMapL2PromoKind", 
                                "Tensor map L2 promotion type",
  [ TensorMapL2PromoNone, TensorMapL2Promo64B, TensorMapL2Promo128B, 
    TensorMapL2Promo256B]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapOOBZero : I32EnumAttrCase<"OOB_ZERO", 0, "zero">;
def TensorMapOOBNaN  : I32EnumAttrCase<"OOB_NAN", 1, "nan">;
def TensorMapOOBKind : I32EnumAttr<"TensorMapOOBKind", 
                                "Tensor map out-of-bounds fill type",
  [ TensorMapOOBZero, TensorMapOOBNaN]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapInterleaveNone : I32EnumAttrCase<"INTERLEAVE_NONE", 0, "none">;
def TensorMapInterleave16B  : I32EnumAttrCase<"INTERLEAVE_16B", 1, "interleave_16b">;
def TensorMapInterleave32B  : I32EnumAttrCase<"INTERLEAVE_32B", 2, "interleave_32b">;
def TensorMapInterleaveKind : I32EnumAttr<"TensorMapInterleaveKind", 
                                "Tensor map interleave layout type",
  [ TensorMapInterleaveNone, TensorMapInterleave16B, TensorMapInterleave32B]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def RcpApprox : I32EnumAttrCase<"APPROX", 0, "approx">;
def RcpRN     : I32EnumAttrCase<"RN", 1, "rn">;
def RcpRZ     : I32EnumAttrCase<"RZ", 2, "rz">;
def RcpRM     : I32EnumAttrCase<"RM", 3, "rm">;
def RcpRP     : I32EnumAttrCase<"RP", 4, "rp">;
def RcpRoundingMode   : I32EnumAttr<"RcpRoundingMode", "Rounding mode of rcp",
  [RcpApprox, RcpRN, RcpRZ, RcpRM, RcpRP]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::nvgpu";
}

def TensorMapSwizzleAttr : EnumAttr<NVGPU_Dialect, TensorMapSwizzleKind, "swizzle">;
def TensorMapL2PromoAttr : EnumAttr<NVGPU_Dialect, TensorMapL2PromoKind, "l2promo">;
def TensorMapOOBAttr : EnumAttr<NVGPU_Dialect, TensorMapOOBKind, "oob">;
def TensorMapInterleaveAttr : EnumAttr<NVGPU_Dialect, TensorMapInterleaveKind, "interleave">;
def RcpRoundingModeAttr : EnumAttr<NVGPU_Dialect, RcpRoundingMode, "rcp_rounding_mode">;

#endif // MLIR_DIALECT_NVGPU_IR_NVGPU_TD
