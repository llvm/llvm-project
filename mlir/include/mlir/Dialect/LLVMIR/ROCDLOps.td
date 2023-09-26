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

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
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

  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// ROCDL attribute definitions
//===----------------------------------------------------------------------===//

class ROCDL_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ROCDL_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}


//===----------------------------------------------------------------------===//
// ROCDL op definitions
//===----------------------------------------------------------------------===//

class ROCDL_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<ROCDL_Dialect, mnemonic, traits> {
}

class ROCDL_IntrPure1Op<string mnemonic> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
  "amdgcn_" # !subst(".", "_", mnemonic), [], [], [Pure], 1>;

class ROCDL_IntrOp<string mnemonic, list<int> overloadedResults,
  list<int> overloadedOperands, list<Trait> traits, int numResults,
  int requiresAccessGroup = 0, int requiresAliasAnalysis = 0> :
  LLVM_IntrOpBase<ROCDL_Dialect,  mnemonic,
    "amdgcn_" # !subst(".", "_", mnemonic), overloadedResults,
    overloadedOperands, traits, numResults, requiresAccessGroup,
    requiresAliasAnalysis>;

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
// Wave-level primitives

class ROCDL_MbcntOp<string mnemonic> :
    ROCDL_IntrPure1Op<"mbcnt." # mnemonic>,
  Arguments<(ins I32:$in0, I32:$in1)> {
  let assemblyFormat = [{
    $in0 `,` $in1  attr-dict `:` `(` type($in0) `,` type($in1) `)` `->` type($res)
   }];
}

def ROCDL_MbcntLoOp : ROCDL_MbcntOp<"lo">;
def ROCDL_MbcntHiOp : ROCDL_MbcntOp<"hi">;

def ROCDL_DsSwizzleOp : 
ROCDL_Op<"ds_swizzle">,
Results<(outs I32:$res)>,
Arguments<(ins I32:$src,
               I32:$offset)>
{
  string llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_ds_swizzle, {$src, $offset});
  }];
  let assemblyFormat = [{
    $src `,` $offset  attr-dict `:` `(` type($src) `,` type($offset) `)` `->` type($res)
   }];
}

def ROCDL_DsBpermuteOp : 
ROCDL_Op<"ds_bpermute">,
Results<(outs I32:$res)>,
Arguments<(ins I32:$index,
               I32:$src)>
{
  string llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_ds_bpermute, {$index, $src});
  }];
  let assemblyFormat = [{
    $index `,` $src  attr-dict `:` `(` type($index) `,` type($src) `)` `->` type($res)
   }];
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
                                               "__ockl_get_num_groups", 0>;

def ROCDL_GridDimYOp : ROCDL_DeviceFunctionOp<"grid.dim.y",
                                               "__ockl_get_num_groups", 1>;

def ROCDL_GridDimZOp : ROCDL_DeviceFunctionOp<"grid.dim.z",
                                               "__ockl_get_num_groups", 2>;

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
// fp8, only on gfx940
def ROCDL_mfma_f32_16x16x32_bf8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf8.bf8">;
def ROCDL_mfma_f32_16x16x32_bf8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf8.fp8">;
def ROCDL_mfma_f32_16x16x32_fp8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.fp8.bf8">;
def ROCDL_mfma_f32_16x16x32_fp8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.fp8.fp8">;
def ROCDL_mfma_f32_32x32x16_bf8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf8.bf8">;
def ROCDL_mfma_f32_32x32x16_bf8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf8.fp8">;
def ROCDL_mfma_f32_32x32x16_fp8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.fp8.bf8">;
def ROCDL_mfma_f32_32x32x16_fp8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.fp8.fp8">;

//===---------------------------------------------------------------------===//
// WMMA intrinsics
class ROCDL_Wmma_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "amdgcn_" # !subst(".","_", mnemonic),
                  [0], [], traits, 1>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  let assemblyFormat =
    "$args attr-dict `:` functional-type($args, $res)";
}

// Available on RDNA3
def ROCDL_wmma_f32_16x16x16_f16 : ROCDL_Wmma_IntrOp<"wmma.f32.16x16x16.f16">;
def ROCDL_wmma_f32_16x16x16_bf16 : ROCDL_Wmma_IntrOp<"wmma.f32.16x16x16.bf16">;
def ROCDL_wmma_f16_16x16x16_f16 : ROCDL_Wmma_IntrOp<"wmma.f16.16x16x16.f16">;
def ROCDL_wmma_bf16_16x16x16_bf16 : ROCDL_Wmma_IntrOp<"wmma.bf16.16x16x16.bf16">;
def ROCDL_wmma_i32_16x16x16_iu8 : ROCDL_Wmma_IntrOp<"wmma.i32.16x16x16.iu8">;
def ROCDL_wmma_i32_16x16x16_iu4 : ROCDL_Wmma_IntrOp<"wmma.i32.16x16x16.iu4">;

//===---------------------------------------------------------------------===//
// Operations on raw buffer resources (stride of 0, bounds checks either off or in
// raw buffer mode).
//===---------------------------------------------------------------------===//

def ROCDLBufferRsrc : LLVM_OpaquePointerInAddressSpace<8>;

def ROCDL_MakeBufferRsrcOp :
  ROCDL_IntrOp<"make.buffer.rsrc", [], [0], [Pure], 1>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 I16:$stride,
                 I32:$numRecords,
                 I32:$flags)> {
  let results = (outs ROCDLBufferRsrc:$res);
  let assemblyFormat = "operands attr-dict `:` type($base) `to` type($res)";
}

def ROCDL_RawPtrBufferLoadOp :
  ROCDL_IntrOp<"raw.ptr.buffer.load", [0], [], [], 1, 0, 1> {
  dag args = (ins Arg<ROCDLBufferRsrc, "", [MemRead]>:$rsrc,
                  I32:$offset,
                  I32:$soffset,
                  I32:$aux);
  let arguments = !con(args, aliasAttrs);
  let assemblyFormat = "operands attr-dict `:` type($res)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getRes()};
    }
  }];
}

def ROCDL_RawPtrBufferStoreOp :
  ROCDL_IntrOp<"raw.ptr.buffer.store", [], [0], [], 0, 0, 1> {
  dag args = (ins LLVM_Type:$vdata,
                  Arg<ROCDLBufferRsrc, "", [MemWrite]>:$rsrc,
                  I32:$offset,
                  I32:$soffset,
                  I32:$aux);
  let arguments = !con(args, aliasAttrs);
  let assemblyFormat = "operands attr-dict `:` type($vdata)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getRsrc()};
    }
  }];

}

def ROCDL_RawPtrBufferAtomicCmpSwap :
  ROCDL_IntrOp<"raw.ptr.buffer.atomic.cmpswap",
    [0], [], [AllTypesMatch<["res", "src", "cmp"]>], 1, 0, 1> {
  dag args = (ins LLVM_Type:$src,
                  LLVM_Type:$cmp,
                  Arg<ROCDLBufferRsrc, "", [MemRead, MemWrite]>:$rsrc,
                  I32:$offset,
                  I32:$soffset,
                  I32:$aux);
  let arguments = !con(args, aliasAttrs);
  let assemblyFormat = "operands attr-dict `:` type($res)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getRsrc()};
    }
  }];
}

class ROCDL_RawPtrBufferAtomicNoRet<string op> :
  ROCDL_IntrOp<"raw.ptr.buffer.atomic." # op, [], [0], [], 0, 0, 1> {
  dag args = (ins LLVM_Type:$vdata,
                  Arg<ROCDLBufferRsrc, "", [MemRead, MemWrite]>:$rsrc,
                  I32:$offset,
                  I32:$soffset,
                  I32:$aux);
  let arguments = !con(args, aliasAttrs);
  let assemblyFormat = "operands attr-dict `:` type($vdata)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getRsrc()};
    }
  }];
}

def ROCDL_RawPtrBufferAtomicFmaxOp : ROCDL_RawPtrBufferAtomicNoRet<"fmax">;
def ROCDL_RawPtrBufferAtomicSmaxOp : ROCDL_RawPtrBufferAtomicNoRet<"smax">;
def ROCDL_RawPtrBufferAtomicUminOp : ROCDL_RawPtrBufferAtomicNoRet<"umin">;
// Note: not supported on all architectures
def ROCDL_RawPtrBufferAtomicFaddOp : ROCDL_RawPtrBufferAtomicNoRet<"fadd">;

/// LEGACY BUFFER OPERATIONS. DO NOT USE IN NEW CODE. KEPT FOR IR COMPATIBILITY.
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

def ROCDL_RawBufferAtomicCmpSwap :
  ROCDL_Op<"raw.buffer.atomic.cmpswap", [AllTypesMatch<["res", "src", "cmp"]>]>,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$src,
                 LLVM_Type:$cmp,
                 LLVM_Type:$rsrc,
                 I32:$offset,
                 I32:$soffset,
                 I32:$aux)>{
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_atomic_cmpswap, {$src, $cmp, $rsrc,
            $offset, $soffset, $aux}, {$_resultType});
  }];
  let assemblyFormat = [{
    attr-dict `(` operands `)` `:` type($res) `,` type($rsrc)
  }];
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

//===---------------------------------------------------------------------===//
// Buffer atomic floating point max intrinsic. GFX9 does not support fp32.

def ROCDL_RawBufferAtomicFMaxOp :
  ROCDL_Op<"raw.buffer.atomic.fmax">,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  string llvmBuilder = [{
      auto vdataType = moduleTranslation.convertType(op.getVdata().getType());
      createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_atomic_fmax, {$vdata, $rsrc,
            $offset, $soffset, $aux}, {vdataType});
  }];
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Buffer atomic signed integer max intrinsic.

def ROCDL_RawBufferAtomicSMaxOp :
  ROCDL_Op<"raw.buffer.atomic.smax">,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  string llvmBuilder = [{
      auto vdataType = moduleTranslation.convertType(op.getVdata().getType());
      createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_atomic_smax, {$vdata, $rsrc,
            $offset, $soffset, $aux}, {vdataType});
  }];
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Buffer atomic unsigned integer min intrinsic.

def ROCDL_RawBufferAtomicUMinOp :
  ROCDL_Op<"raw.buffer.atomic.umin">,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  string llvmBuilder = [{
      auto vdataType = moduleTranslation.convertType(op.getVdata().getType());
      createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_raw_buffer_atomic_umin, {$vdata, $rsrc,
            $offset, $soffset, $aux}, {vdataType});
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ROCDL target attribute.
//===----------------------------------------------------------------------===//

def ROCDL_TargettAttr :
    ROCDL_Attr<"ROCDLTarget", "target"> {
  let description = [{
    ROCDL target attribute for controlling compilation of AMDGPU targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#rocdl.target] attributes {...} {
        ...
      }
    ```

    2. Target with `gfx90a` chip and fast math.
    ```
      gpu.module @mymodule [#rocdl.target<chip = "gfx90a", flags = {fast, no_wave64}>] {
        ...
      }
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"amdgcn-amd-amdhsa\"">:$triple,
    StringRefParameter<"Target chip.", "\"gfx900\"">:$chip,
    StringRefParameter<"Target chip features.", "\"\"">:$features,
    StringRefParameter<"ABI version.", "\"400\"">:$abi,
    OptionalParameter<"DictionaryAttr", "Target specific flags.">:$flags,
    OptionalParameter<"ArrayAttr", "Files to link to the LLVM module.">:$link
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features, $abi, $flags, $link)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"amdgcn-amd-amdhsa\"">:$triple,
                     CArg<"StringRef", "\"gfx900\"">:$chip,
                     CArg<"StringRef", "\"\"">:$features,
                     CArg<"StringRef", "\"400\"">:$abiVersion,
                     CArg<"DictionaryAttr", "nullptr">:$targetFlags,
                     CArg<"ArrayAttr", "nullptr">:$linkFiles), [{
      return Base::get($_ctxt, optLevel, triple, chip, features, abiVersion,
                       targetFlags, linkFiles);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool hasFlag(StringRef flag) const;
    bool hasWave64() const;
    bool hasFastMath() const;
    bool hasDaz() const;
    bool hasFiniteOnly() const;
    bool hasUnsafeMath() const;
    bool hasCorrectSqrt() const;
  }];
  let extraClassDefinition = [{
    bool $cppClass::hasFlag(StringRef flag) const {
      if (DictionaryAttr flags = getFlags())
        return flags.get(flag) != nullptr;
      return false;
    }
    bool $cppClass::hasWave64() const {
      return hasFlag("wave64") || !hasFlag("no_wave64");
    }
    bool $cppClass::hasFastMath() const {
      return hasFlag("fast");
    }
    bool $cppClass::hasDaz() const {
      return hasFlag("daz");
    }
    bool $cppClass::hasFiniteOnly() const {
      return hasFlag("finite_only");
    }
    bool $cppClass::hasUnsafeMath() const {
      return hasFlag("unsafe_math");
    }
    bool $cppClass::hasCorrectSqrt() const {
      return !hasFlag("unsafe_sqrt");
    }
  }];
}

#endif // ROCDLIR_OPS
