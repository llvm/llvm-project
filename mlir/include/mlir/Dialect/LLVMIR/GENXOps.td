//===-- GENXOps.td - GENX IR dialect op definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the GENX IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef GENXIR_OPS
#define GENXIR_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// GENX dialect definitions
//===----------------------------------------------------------------------===//

def GENX_Dialect : Dialect {
  let name = "genx";
  let cppNamespace = "::mlir::GENX";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "genx.kernel"; }
    /// Get the name of the attribute used to annotate max work group size
    /// required for kernel functions
    static constexpr ::llvm::StringLiteral getMaxWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.max_work_group_size");
    }
    /// Get the name of the attribute used to annotate exact work group size
    /// required for kernel functions.
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.reqd_work_group_size");
    }

    /// The address space value that represents global memory.
    static constexpr unsigned kGlobalMemoryAddressSpace = 1;
    /// The address space value that represents shared memory.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
    /// The address space value that represents private memory.
    static constexpr unsigned kPrivateMemoryAddressSpace = 0;
  }];

  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// GENX op definitions
//===----------------------------------------------------------------------===//

class GENX_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<GENX_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// GENX intrinsic operations
//===----------------------------------------------------------------------===//
class GENX_IntrOp<string mnem, list<Trait> traits,
                  int numResults>
  : LLVM_IntrOpBase<GENX_Dialect, mnem, "genx_" # !subst(".", "_", mnem),
                    /*list<int> overloadedResults=*/[],
                    /*list<int> overloadedOperands=*/[],
                    traits, numResults>;


//===----------------------------------------------------------------------===//
// GENX special register op definitions
//===----------------------------------------------------------------------===//

class GENX_SpecialRegisterOp<string mnemonic, list<Trait> traits = []> :
  GENX_IntrOp<mnemonic, !listconcat(traits, [Pure], 1)> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($res)";
}

class GENX_DeviceFunctionOp<string mnemonic, string device_function,
                            int parameter, list<Trait> traits = []> :
  GENX_Op<mnemonic, !listconcat(traits, [Pure])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createDeviceFunctionCall(builder, \""
  # device_function # "\", " # parameter # ");";
  let assemblyFormat = "attr-dict `:` type($res)";
}

#if 0
// Let the SPIRV dialect to deal with the SPIRV builtin for now

//===----------------------------------------------------------------------===//
// Thread index and Block index

def GENX_ThreadIdXOp : GENX_SpecialRegisterOp<"workitem.id.x">;
def GENX_ThreadIdYOp : GENX_SpecialRegisterOp<"workitem.id.y">;
def GENX_ThreadIdZOp : GENX_SpecialRegisterOp<"workitem.id.z">;

def GENX_BlockIdXOp : GENX_SpecialRegisterOp<"workgroup.id.x">;
def GENX_BlockIdYOp : GENX_SpecialRegisterOp<"workgroup.id.y">;
def GENX_BlockIdZOp : GENX_SpecialRegisterOp<"workgroup.id.z">;

//===----------------------------------------------------------------------===//
// Thread range and Block range

def GENX_BlockDimXOp : GENX_DeviceFunctionOp<"workgroup.dim.x",
                                             "__ockl_get_local_size", 0>;

def GENX_BlockDimYOp : GENX_DeviceFunctionOp<"workgroup.dim.y",
                                             "__ockl_get_local_size", 1>;

def GENX_BlockDimZOp : GENX_DeviceFunctionOp<"workgroup.dim.z",
                                             "__ockl_get_local_size", 2>;

def GENX_GridDimXOp : GENX_DeviceFunctionOp<"grid.dim.x",
                                            "__ockl_get_global_size", 0>;

def GENX_GridDimYOp : GENX_DeviceFunctionOp<"grid.dim.y",
                                            "__ockl_get_global_size", 1>;

def GENX_GridDimZOp : GENX_DeviceFunctionOp<"grid.dim.z",
                                            "__ockl_get_global_size", 2>;
#endif

//===----------------------------------------------------------------------===//
// Attributes for load intrinsics
//===----------------------------------------------------------------------===//

def CacheNone           : I32EnumAttrCase<"NONE", 0, "none">;
def CacheUncached       : I32EnumAttrCase<"UNCACHED", 1, "uncached">;
def CacheCached         : I32EnumAttrCase<"CACHED", 2, "cached">;
def CacheWriteBack      : I32EnumAttrCase<"WRITE_BACK", 3, "write_back">;
def CacheWriteThrough   : I32EnumAttrCase<"WRITE_THROUGH", 4, "write_through">;
def CacheStreaming      : I32EnumAttrCase<"STREAMING", 5, "streaming">;
def CacheReadInvalidate : I32EnumAttrCase<"READ_INVALIDATE", 6, "read_invalidate">;

def CacheHint : I32EnumAttr<"CacheHint", "GENX cache hint",
    [CacheNone, CacheUncached, CacheCached, CacheWriteBack, CacheWriteThrough,
     CacheStreaming, CacheReadInvalidate]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::GENX";
}

def CacheHintAttr : EnumAttr<GENX_Dialect, CacheHint, "cache_hint">;

def LSCDataSizeAttr : I32EnumAttr<
    "LSCDataSize", "",
    [
      I32EnumAttrCase<"NONE", 0, "none">,
      I32EnumAttrCase<"U8", 1, "u8">,
      I32EnumAttrCase<"U16", 2, "u16">,
      I32EnumAttrCase<"U32", 3, "u32">,
      I32EnumAttrCase<"U64", 4, "u64">,
      I32EnumAttrCase<"U8U32", 5, "u2u32">, // load u8, zero extend to u32
      I32EnumAttrCase<"U16U32", 6, "u16u32">, // load u16, zero extend to u32
      I32EnumAttrCase<"U16U32H", 7 "u16u32h">,
    ]> {
    let cppNamespace = "::mlir::GENX";
}

def LSCDataOrderAttr : I32EnumAttr<
    "LSCDataOrder", "",
    [
      I32EnumAttrCase<"NONTRANSPOSE", 1, "nontranspose">,
      I32EnumAttrCase<"TRANSPOSE", 2, "transpose">,
    ]> {
    let cppNamespace = "::mlir::GENX";
}

def LSCVectorSizeAttr : I32EnumAttr<
    "LSCVectorSize", "",
    [
      I32EnumAttrCase<"N1", 1, "n1">,    // vlen = 1
      I32EnumAttrCase<"N2", 2, "n2">,    // vlen = 2
      I32EnumAttrCase<"N3", 3, "n3">,    // vlen = 3
      I32EnumAttrCase<"N4", 4, "n4">,    // vlen = 4
      I32EnumAttrCase<"N8", 5, "n8">,    // vlen = 8
      I32EnumAttrCase<"N16", 6, "n16">,  // vlen = 16
      I32EnumAttrCase<"N32", 7, "n32">,  // vlen = 32
      I32EnumAttrCase<"N64", 8, "n64">.  // vlen = 64
    ]> {
    let cppNamespace = "::mlir::GENX";
}

//===----------------------------------------------------------------------===//
// masked load intrinsics
//===----------------------------------------------------------------------===//
#if 0
def GENX_Masked4LoadOp : 
    GENX_Op<"lsc.load.stateless">,
    Results<(outs LLVM_Type:$res)>,
    Arguments<(ins LLVM_Type:$addr,
                   LLVM_Type:$mask,
                   CacheHintAttr:$l1h,
                   CacheHintAttr:$l3h)> {
    string llvmBuilder = [{
        auto elemType = LLVM::getVectorElementType($_resultType);
        auto numElems = LLVM::getVectorNumElements($_resultType);
        auto arg0 = true;
        auto arg1 = Subopcode;
        auto arg2 = llvm::ConstantInt::get(Int8Ty, $l1h);
        auto arg3 = llvm::ConstantInt::get(Int8Ty, $l3h);
        uint16_t addressScale = 1;
        auto arg4 = llvm::ConstantInt::get(Int16Ty, addressScale);
        int32_t immOffset = 0;
        auto arg5 = llvm::ConstantInt::get(Int32Ty, immOffset);
        uint8_t dataSize = getPrimitiveTypeSizeInBits(elemType)/8;
        auto arg6 = llvm::ConstantInt::get(int8Ty, dataSize);
        auto arg7 = llvm::ConstantInt::get(int8Ty, numElems);
        auto arg7 = getLSC
        auto arg8 = llvm::ConstantInt::get(int8Ty, LSCDataOrder::TRANSPOSE);
        auto arg9 = $mask
        auto arg10 = $addr
        auto arg11 = llvm::ConstantInt::get(int32Ty, 0 /*surface*/);
        auto IntId = 
        $res = createIntrinsicCall(builder, intId, {pred, subOpcode,})
    }];

    }
#else
def GENX_MaskedLoadOp : GENX_Op<"lsc.load.stateless", [MemoryEffects<[MemRead]>]> {
  let summary = "Load of a vector of length 1, 2, 3, 4, 8, 16, 32, 64 with mask";

  let arguments = (ins LLVM_AnyPointer:$ptr, 
                       FixedVectorOfLengthAndType<[1, 2, 3, 4, 8, 16, 32, 64], [I1]>:$mask
                       CacheHintAttr:$l1h, CacheHintAttr:$l3h);

  let results = (outs FixedVectorOfLength<[1, 2, 3, 4, 8, 16, 32, 64]>);
}
#endif
             ,



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

#endif // GENXIR_OPS