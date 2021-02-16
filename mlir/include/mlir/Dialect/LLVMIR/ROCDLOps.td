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

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "rocdl.kernel"; }
  }];
}

//===----------------------------------------------------------------------===//
// ROCDL op definitions
//===----------------------------------------------------------------------===//

class ROCDL_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<ROCDL_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// ROCDL special register op definitions
//===----------------------------------------------------------------------===//

class ROCDL_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = "$res = createIntrinsicCall(builder,"
    # "llvm::Intrinsic::amdgcn_" # !subst(".","_", mnemonic) # ");";
  let assemblyFormat = "attr-dict `:` type($res)";
}

class ROCDL_DeviceFunctionOp<string mnemonic, string device_function,
                             int parameter, list<OpTrait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>,
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

class ROCDL_Mfma_IntrOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "amdgcn_" # !subst(".","_", mnemonic),
                  [], [], traits, 1>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  let assemblyFormat =
    "$args attr-dict `:` functional-type($args, $res)";
}

def ROCDL_mfma_f32_32x32x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x1f32">;
def ROCDL_mfma_f32_32x32x2f32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x2f32">;
def ROCDL_mfma_f32_16x16x4f32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4f32">;
def ROCDL_mfma_f32_16x16x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x1f32">;
def ROCDL_mfma_f32_32x32x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4f16">;
def ROCDL_mfma_f32_32x32x8f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x8f16">;
def ROCDL_mfma_f32_16x16x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4f16">;
def ROCDL_mfma_f32_16x16x16f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x16f16">;
def ROCDL_mfma_f32_32x32x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x2bf16">;
def ROCDL_mfma_f32_32x32x4bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4bf16">;
def ROCDL_mfma_f32_16x16x8bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x8bf16">;
def ROCDL_mfma_f32_16x16x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x2bf16">;
def ROCDL_mfma_f32_4x4x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x2bf16">;
def ROCDL_mfma_f32_4x4x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x1f32">;
def ROCDL_mfma_f32_4x4x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x4f16">;
def ROCDL_mfma_i32_32x32x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x4i8">;
def ROCDL_mfma_i32_16x16x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x4i8">;
def ROCDL_mfma_i32_4x4x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.4x4x4i8">;
def ROCDL_mfma_i32_32x32x8i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x8i8">;
def ROCDL_mfma_i32_16x16x16i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x16i8">;

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
  let parser = [{ return parseROCDLMubufLoadOp(parser, result); }];
  let printer = [{
    Operation *op = this->getOperation();
    p << op->getName() << " " << op->getOperands()
      << " : " << op->getResultTypes();
  }];
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
    auto vdataType = moduleTranslation.convertType(op.vdata().getType());
    createIntrinsicCall(builder,
          llvm::Intrinsic::amdgcn_buffer_store, {$vdata, $rsrc, $vindex,
          $offset, $glc, $slc}, {vdataType});
  }];
  let parser = [{ return parseROCDLMubufStoreOp(parser, result); }];
  let printer = [{
    Operation *op = this->getOperation();
    p << op->getName() << " " << op->getOperands()
      << " : " << vdata().getType();
  }];
}

#endif // ROCDLIR_OPS
