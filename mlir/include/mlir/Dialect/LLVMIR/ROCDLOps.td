//===-- ROCDLOps.td - ROCDL IR dialect op definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the ROCDL IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef ROCDLIR_OPS
#define ROCDLIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ROCDL dialect definitions
//===----------------------------------------------------------------------===//

def ROCDL_Dialect : Dialect {
  let name = "rocdl";
  let cppNamespace = "::mlir::ROCDL";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "rocdl.kernel"; }
    static constexpr ::llvm::StringLiteral getFlatWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("rocdl.flat_work_group_size");
    }
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("rocdl.reqd_work_group_size");
    }

    /// The address space value that represents global memory.
    static constexpr unsigned kGlobalMemoryAddressSpace = 1;
    /// The address space value that represents shared memory.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
    /// The address space value that represents private memory.
    static constexpr unsigned kPrivateMemoryAddressSpace = 5;
  }];
}

//===----------------------------------------------------------------------===//
// ROCDL op definitions
//===----------------------------------------------------------------------===//

class ROCDL_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<ROCDL_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// ROCDL special register op definitions
//===----------------------------------------------------------------------===//

class ROCDL_SpecialRegisterOp<string mnemonic,
    list<Trait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [Pure])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createIntrinsicCallWithRange(builder,"
    # "llvm::Intrinsic::amdgcn_" # !subst(".","_", mnemonic)
    # ", op->getAttrOfType<::mlir::DenseI32ArrayAttr>(\"range\"));";
  let assemblyFormat = "attr-dict `:` type($res)";
}

class ROCDL_DeviceFunctionOp<string mnemonic, string device_function,
                             int parameter, list<Trait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [Pure])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createDeviceFunctionCall(builder, \""
  # device_function # "\", " # parameter # ");";
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Thread index and Block index

def ROCDL_ThreadIdXOp : ROCDL_SpecialRegisterOp<"workitem.id.x">;
def ROCDL_ThreadIdYOp : ROCDL_SpecialRegisterOp<"workitem.id.y">;
def ROCDL_ThreadIdZOp : ROCDL_SpecialRegisterOp<"workitem.id.z">;

def ROCDL_BlockIdXOp : ROCDL_SpecialRegisterOp<"workgroup.id.x">;
def ROCDL_BlockIdYOp : ROCDL_SpecialRegisterOp<"workgroup.id.y">;
def ROCDL_BlockIdZOp : ROCDL_SpecialRegisterOp<"workgroup.id.z">;

//===----------------------------------------------------------------------===//
// Thread range and Block range

def ROCDL_BlockDimXOp : ROCDL_DeviceFunctionOp<"workgroup.dim.x",
                                               "__ockl_get_local_size", 0>;

def ROCDL_BlockDimYOp : ROCDL_DeviceFunctionOp<"workgroup.dim.y",
                                               "__ockl_get_local_size", 1>;

def ROCDL_BlockDimZOp : ROCDL_DeviceFunctionOp<"workgroup.dim.z",
                                               "__ockl_get_local_size", 2>;

def ROCDL_GridDimXOp : ROCDL_DeviceFunctionOp<"grid.dim.x",
                                               "__ockl_get_global_size", 0>;

def ROCDL_GridDimYOp : ROCDL_DeviceFunctionOp<"grid.dim.y",
                                               "__ockl_get_global_size", 1>;

def ROCDL_GridDimZOp : ROCDL_DeviceFunctionOp<"grid.dim.z",
                                               "__ockl_get_global_size", 2>;

//===----------------------------------------------------------------------===//
// Synchronization primitives

def ROCDL_BarrierOp : ROCDL_Op<"barrier"> {
  string llvmBuilder = [{
    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(llvm::AtomicOrdering::Release,
                        llvmContext.getOrInsertSyncScopeID("workgroup"));
    createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_s_barrier);
    builder.CreateFence(llvm::AtomicOrdering::Acquire,
                        llvmContext.getOrInsertSyncScopeID("workgroup"));
  }];
  let assemblyFormat = "attr-dict";
}

//===---------------------------------------------------------------------===//
// Xdlops intrinsics

class ROCDL_Mfma_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "amdgcn_" # !subst(".","_", mnemonic),
                  [], [], traits, 1>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  let assemblyFormat =
    "$args attr-dict `:` functional-type($args, $res)";
}

// Available on all CDNA.
def ROCDL_mfma_f32_32x32x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x1f32">;
def ROCDL_mfma_f32_16x16x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x1f32">;
def ROCDL_mfma_f32_4x4x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x1f32">;
def ROCDL_mfma_f32_32x32x2f32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x2f32">;
def ROCDL_mfma_f32_16x16x4f32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4f32">;
def ROCDL_mfma_f32_32x32x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4f16">;
def ROCDL_mfma_f32_16x16x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4f16">;
def ROCDL_mfma_f32_4x4x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x4f16">;
def ROCDL_mfma_f32_32x32x8f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x8f16">;
def ROCDL_mfma_f32_16x16x16f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x16f16">;
def ROCDL_mfma_i32_32x32x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x4i8">;
def ROCDL_mfma_i32_16x16x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x4i8">;
def ROCDL_mfma_i32_4x4x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.4x4x4i8">;
def ROCDL_mfma_i32_32x32x8i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x8i8">;
def ROCDL_mfma_i32_16x16x16i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x16i8">;
def ROCDL_mfma_f32_32x32x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x2bf16">;
def ROCDL_mfma_f32_16x16x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x2bf16">;
def ROCDL_mfma_f32_4x4x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x2bf16">;
def ROCDL_mfma_f32_32x32x4bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4bf16">;
def ROCDL_mfma_f32_16x16x8bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x8bf16">;
// New in gfx90a.
def ROCDL_mfma_f32_32x32x4bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4bf16.1k">;
def ROCDL_mfma_f32_16x16x4bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4bf16.1k">;
def ROCDL_mfma_f32_4x4x4bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x4bf16.1k">;
def ROCDL_mfma_f32_32x32x8bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x8bf16.1k">;
def ROCDL_mfma_f32_16x16x16bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x16bf16.1k">;
// Note: in gfx940, unlike in gfx90a, the f64 xdlops use the "blgp" argument as a
// NEG bitfield. See IntrinsicsAMDGPU.td for more info.
def ROCDL_mfma_f64_16x16x4f64 : ROCDL_Mfma_IntrOp<"mfma.f64.16x16x4f64">;
def ROCDL_mfma_f64_4x4x4f64 : ROCDL_Mfma_IntrOp<"mfma.f64.4x4x4f64">;
// New in gfx940.
def ROCDL_mfma_i32_16x16x32_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x32.i8">;
def ROCDL_mfma_i32_32x32x16_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x16.i8">;
def ROCDL_mfma_f32_16x16x8_xf32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x8.xf32">;
def ROCDL_mfma_f32_32x32x4_xf32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4.xf32">;

//===---------------------------------------------------------------------===//
// Vector buffer load/store intrinsics

def ROCDL_MubufLoadOp :
  ROCDL_Op<"buffer.load">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$rsrc,
                 LLVM_Type:$vindex,
                 LLVM_Type:$offset,
                 LLVM_Type:$glc,
                 LLVM_Type:$slc)>{
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_buffer_load, {$rsrc, $vindex, $offset, $glc,
          $slc}, {$_resultType});
  }];
  let hasCustomAssemblyFormat = 1;
}

def ROCDL_MubufStoreOp :
  ROCDL_Op<"buffer.store">,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$vindex,
                 LLVM_Type:$offset,
                 LLVM_Type:$glc,
                 LLVM_Type:$slc)>{
  string llvmBuilder = [{
    auto vdataType = moduleTranslation.convertType(op.getVdata().getType());
    createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_buffer_store, {$vdata, $rsrc, $vindex,
          $offset, $glc, $slc}, {vdataType});
  }];
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Raw buffer load/store intrinsics

def ROCDL_RawBufferLoadOp :
  ROCDL_Op<"raw.buffer.load">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_load, {$rsrc, $offset,
          $soffset, $aux}, {$_resultType});
  }];
  let hasCustomAssemblyFormat = 1;
}

def ROCDL_RawBufferStoreOp :
  ROCDL_Op<"raw.buffer.store">,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  string llvmBuilder = [{
    auto vdataType = moduleTranslation.convertType(op.getVdata().getType());
    createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_store, {$vdata, $rsrc,
          $offset, $soffset, $aux}, {vdataType});
  }];
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// MI-100 and MI-200 buffer atomic floating point add intrinsic

def ROCDL_RawBufferAtomicFAddOp :
  ROCDL_Op<"raw.buffer.atomic.fadd">,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  string llvmBuilder = [{
      auto vdataType = moduleTranslation.convertType(op.getVdata().getType());
      createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_atomic_fadd, {$vdata, $rsrc,
            $offset, $soffset, $aux}, {vdataType});
  }];
  let hasCustomAssemblyFormat = 1;
}

#endif // ROCDLIR_OPS
