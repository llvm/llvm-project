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
    /// MLIR's gpu-related infrastructure effectively assume uniform workgroup
    /// sizes, so this attribute defaults to "true" on `rocdl.kernel` functions.
    /// It is provided here to allow overriding this assumption.
    static constexpr ::llvm::StringLiteral getUniformWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("rocdl.uniform_work_group_size");
    }

    /// The address space value that represents global memory.
    static constexpr unsigned kGlobalMemoryAddressSpace = 1;
    /// The address space value that represents shared memory.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
    /// The address space value that represents private memory.
    static constexpr unsigned kPrivateMemoryAddressSpace = 5;
  }];

  let discardableAttrs = (ins
     "::mlir::UnitAttr":$kernel,
     "::mlir::DenseI32ArrayAttr":$reqd_work_group_size,
     "::mlir::StringAttr":$flat_work_group_size,
     "::mlir::IntegerAttr":$max_flat_work_group_size,
     "::mlir::IntegerAttr":$waves_per_eu,
     "::mlir::BoolAttr":$unsafe_fp_atomics,
     // Correspond to LLVM metadata of the same name
     "::mlir::UnitAttr":$last_use,
     "::mlir::UnitAttr":$no_remote_memory,
     "::mlir::UnitAttr":$no_fine_grained_memory,
     "::mlir::UnitAttr":$ignore_denormal_mode
  );

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
  int requiresAccessGroup = 0, int requiresAliasAnalysis = 0, list<int> immArgPositions = [],
  list<string> immArgAttrNames = []> :
  LLVM_IntrOpBase<ROCDL_Dialect,  mnemonic,
    "amdgcn_" # !subst(".", "_", mnemonic), overloadedResults,
    overloadedOperands, traits, numResults, requiresAccessGroup,
    requiresAliasAnalysis, 0, 0, immArgPositions, immArgAttrNames>;

//===----------------------------------------------------------------------===//
// ROCDL special register op definitions
//===----------------------------------------------------------------------===//

class ROCDL_SpecialIdRegisterOp<string mnemonic> :
    ROCDL_IntrPure1Op<mnemonic>,
    Arguments<(ins OptionalAttr<LLVM_ConstantRangeAttr>:$range)> {
  string llvmBuilder = baseLlvmBuilder # setRangeRetAttrCode # baseLlvmBuilderCoda;
  string mlirBuilder = baseMlirBuilder # importRangeRetAttrCode # baseMlirBuilderCoda;

  let assemblyFormat = "(`range` $range^)? attr-dict `:` type($res)";

    // Temporaly builder until Nvidia ops also support range attributes.
  let builders = [
    OpBuilder<(ins "Type":$resultType), [{
      build($_builder, $_state, resultType, ::mlir::LLVM::ConstantRangeAttr{});
    }]>
  ];
}

class ROCDL_DimGetterFunctionOp<string mnemonic, string device_function,
                             int parameter, list<Trait> traits = []> :
  ROCDL_Op<mnemonic, !listconcat(traits, [Pure])>,
  Results<(outs LLVM_Type:$res)>, Arguments<(ins OptionalAttr<LLVM_ConstantRangeAttr>:$range)> {
  string llvmBuilder = "$res = createDimGetterFunctionCall(builder, op, \""
  # device_function # "\", " # parameter # ");";
  let assemblyFormat = "(`range` $range^)? attr-dict `:` type($res)";

  // Temporaly builder until Nvidia ops also support range attributes.
  let builders = [
    OpBuilder<(ins "Type":$resultType), [{
      build($_builder, $_state, resultType, ::mlir::LLVM::ConstantRangeAttr{});
    }]>
  ];
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

def ROCDL_BallotOp :
  ROCDL_Op<"ballot">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins I1:$pred)> {
  let summary = "Vote across thread group";

  let description = [{
      Ballot provides a bit mask containing the 1-bit predicate value from each lane.
      The nth bit of the result contains the 1 bit contributed by the nth warp lane.
  }];

  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::amdgcn_ballot, {$pred}, {$_resultType});
  }];

  let assemblyFormat = "$pred attr-dict `:` type($res)";
}

def ROCDL_ReadlaneOp : ROCDL_IntrOp<"readlane", [], [0], [AllTypesMatch<["res", "src0"]>], 1>,
  Arguments<(ins LLVM_Type:$src0,
                 I32:$src1)> {
  let results = (outs LLVM_Type:$res);
  let summary = "Get the value in the specific lane.";

  let description = [{
    Get the value in lane `src1` from input `src0`.
  }];

  let assemblyFormat = [{
    $src0 `,` $src1  attr-dict `:` `(` type($src0) `,` type($src1) `)` `->` type($res)
   }];
}

//===----------------------------------------------------------------------===//
// Thread index and Block index

def ROCDL_ThreadIdXOp : ROCDL_SpecialIdRegisterOp<"workitem.id.x">;
def ROCDL_ThreadIdYOp : ROCDL_SpecialIdRegisterOp<"workitem.id.y">;
def ROCDL_ThreadIdZOp : ROCDL_SpecialIdRegisterOp<"workitem.id.z">;

def ROCDL_BlockIdXOp : ROCDL_SpecialIdRegisterOp<"workgroup.id.x">;
def ROCDL_BlockIdYOp : ROCDL_SpecialIdRegisterOp<"workgroup.id.y">;
def ROCDL_BlockIdZOp : ROCDL_SpecialIdRegisterOp<"workgroup.id.z">;

//===----------------------------------------------------------------------===//
// Thread range and Block range

def ROCDL_BlockDimXOp : ROCDL_DimGetterFunctionOp<"workgroup.dim.x",
                                               "__ockl_get_local_size", 0>;

def ROCDL_BlockDimYOp : ROCDL_DimGetterFunctionOp<"workgroup.dim.y",
                                               "__ockl_get_local_size", 1>;

def ROCDL_BlockDimZOp : ROCDL_DimGetterFunctionOp<"workgroup.dim.z",
                                               "__ockl_get_local_size", 2>;

def ROCDL_GridDimXOp : ROCDL_DimGetterFunctionOp<"grid.dim.x",
                                               "__ockl_get_num_groups", 0>;

def ROCDL_GridDimYOp : ROCDL_DimGetterFunctionOp<"grid.dim.y",
                                               "__ockl_get_num_groups", 1>;

def ROCDL_GridDimZOp : ROCDL_DimGetterFunctionOp<"grid.dim.z",
                                               "__ockl_get_num_groups", 2>;

//===----------------------------------------------------------------------===//
// Synchronization primitives

// Emits the waintcnt instruction. The bitfield's semantics depend
// on the target chipset
def ROCDL_WaitcntOp : ROCDL_Op<"waitcnt">, Arguments<(ins I32Attr:$bitfield)> {
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_s_waitcnt,
      {builder.getInt32($bitfield)});
  }];
  let assemblyFormat = "attr-dict $bitfield";
}

def ROCDL_SBarrierOp : ROCDL_Op<"s.barrier"> {
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_s_barrier);
  }];
  let assemblyFormat = "attr-dict";
}

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

def ROCDL_BarrierSignalOp : ROCDL_IntrOp<"s.barrier.signal", [], [], [], 0, 0, 0, [0], ["id"]>,
  Arguments<(ins I32Attr:$id)> {
  let results = (outs);
  let assemblyFormat = "$id attr-dict";
}

def ROCDL_BarrierWaitOp : ROCDL_IntrOp<"s.barrier.wait", [], [], [], 0, 0, 0, [0], ["id"]>,
  Arguments<(ins I16Attr:$id)> {
  let results = (outs);
  let assemblyFormat = "$id attr-dict";
  string llvmBuilder =
    "createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_s_barrier_wait,builder.getInt16(op.getId()));";
}

def ROCDL_WaitDscntOp: ROCDL_IntrOp<"s.wait.dscnt", [], [], [], 0, 0, 0, [0], ["id"]>,
  Arguments<(ins I16Attr:$id)> {
  let results = (outs);
  let assemblyFormat = "$id attr-dict";
}

def ROCDL_SetPrioOp : ROCDL_IntrOp<"s.setprio", [], [], [], 0>,
  Arguments<(ins I16Attr:$priority)> {
  let results = (outs);
  let assemblyFormat = "$priority attr-dict";
  string llvmBuilder =
    "createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_s_setprio,builder.getInt16(op.getPriority()));";
}

def ROCDL_SchedBarrier : ROCDL_IntrOp<"sched.barrier", [], [], [], 0>,
  Arguments<(ins I32Attr:$mask)> {
  let results = (outs);
  let assemblyFormat = "$mask attr-dict";
  string llvmBuilder =
    "createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_sched_barrier,builder.getInt32(op.getMask()));";
}

def ROCDL_SchedGroupBarrier : ROCDL_IntrOp<"sched.group.barrier", [], [], [], 0>,
  Arguments<(ins I32Attr:$mask, I32Attr:$size, I32Attr:$groupId)> {
  let results = (outs);
  let assemblyFormat = "$mask `,` $size `,` $groupId attr-dict";
  string llvmBuilder = [{
    createIntrinsicCall(builder,
      llvm::Intrinsic::amdgcn_sched_group_barrier,
      {builder.getInt32(op.getMask()), builder.getInt32(op.getSize()), builder.getInt32(op.getGroupId())});
  }];
}

def ROCDL_IglpOpt : ROCDL_IntrOp<"iglp.opt", [], [], [], 0>,
  Arguments<(ins I32Attr:$variant)> {
  let results = (outs);
  let assemblyFormat = "$variant attr-dict";
  string llvmBuilder =
    "createIntrinsicCall(builder, llvm::Intrinsic::amdgcn_iglp_opt,builder.getInt32(op.getVariant()));";
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

//===---------------------------------------------------------------------===//
// MFMA intrinsics with overloaded operands
class ROCDL_Mfma_OO_IntrOp<string mnemonic, list<int> overloadedOperands,
                        list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "amdgcn_" # !subst(".","_", mnemonic),
                  [], overloadedOperands, traits, 1>,
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
def ROCDL_mfma_f32_16x16x32_bf8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf8.bf8">;
def ROCDL_mfma_f32_16x16x32_bf8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf8.fp8">;
def ROCDL_mfma_f32_16x16x32_fp8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.fp8.bf8">;
def ROCDL_mfma_f32_16x16x32_fp8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.fp8.fp8">;
def ROCDL_mfma_f32_32x32x16_bf8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf8.bf8">;
def ROCDL_mfma_f32_32x32x16_bf8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf8.fp8">;
def ROCDL_mfma_f32_32x32x16_fp8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.fp8.bf8">;
def ROCDL_mfma_f32_32x32x16_fp8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.fp8.fp8">;
// New in gfx950.
def ROCDL_mfma_f32_16x16x32_bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf16">;
def ROCDL_mfma_i32_16x16x64_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x64.i8">;
def ROCDL_mfma_f32_16x16x32_f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.f16">;
def ROCDL_mfma_f32_32x32x16_bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf16">;
def ROCDL_mfma_i32_32x32x32_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x32.i8">;
def ROCDL_mfma_f32_32x32x16_f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.f16">;
def ROCDL_mfma_scale_f32_16x16x128_f8f6f4 : ROCDL_Mfma_OO_IntrOp<"mfma.scale.f32.16x16x128.f8f6f4", [0,1]>;
def ROCDL_mfma_scale_f32_32x32x64_f8f6f4 : ROCDL_Mfma_OO_IntrOp<"mfma.scale.f32.32x32x64.f8f6f4", [0,1]>;

// 2:4 Sparsity ops (GFX940)
def ROCDL_smfmac_f32_16x16x32_f16 : ROCDL_Mfma_IntrOp<"smfmac.f32.16x16x32.f16">;
def ROCDL_smfmac_f32_32x32x16_f16 : ROCDL_Mfma_IntrOp<"smfmac.f32.32x32x16.f16">;
def ROCDL_smfmac_f32_16x16x32_bf16 : ROCDL_Mfma_IntrOp<"smfmac.f32.16x16x32.bf16">;
def ROCDL_smfmac_f32_32x32x16_bf16 : ROCDL_Mfma_IntrOp<"smfmac.f32.32x32x16.bf16">;
def ROCDL_smfmac_i32_16x16x64_i8 : ROCDL_Mfma_IntrOp<"smfmac.i32.16x16x64.i8">;
def ROCDL_smfmac_i32_32x32x32_i8 : ROCDL_Mfma_IntrOp<"smfmac.i32.32x32x32.i8">;
def ROCDL_smfmac_f32_16x16x64_bf8_bf8 : ROCDL_Mfma_IntrOp<"smfmac.f32.16x16x64.bf8.bf8">;
def ROCDL_smfmac_f32_16x16x64_bf8_fp8 : ROCDL_Mfma_IntrOp<"smfmac.f32.16x16x64.bf8.fp8">;
def ROCDL_smfmac_f32_16x16x64_fp8_bf8 : ROCDL_Mfma_IntrOp<"smfmac.f32.16x16x64.fp8.bf8">;
def ROCDL_smfmac_f32_16x16x64_fp8_fp8 : ROCDL_Mfma_IntrOp<"smfmac.f32.16x16x64.fp8.fp8">;
def ROCDL_smfmac_f32_32x32x32_bf8_bf8 : ROCDL_Mfma_IntrOp<"smfmac.f32.32x32x32.bf8.bf8">;
def ROCDL_smfmac_f32_32x32x32_bf8_fp8 : ROCDL_Mfma_IntrOp<"smfmac.f32.32x32x32.bf8.fp8">;
def ROCDL_smfmac_f32_32x32x32_fp8_bf8 : ROCDL_Mfma_IntrOp<"smfmac.f32.32x32x32.fp8.bf8">;
def ROCDL_smfmac_f32_32x32x32_fp8_fp8 : ROCDL_Mfma_IntrOp<"smfmac.f32.32x32x32.fp8.fp8">;


//===---------------------------------------------------------------------===//
// WMMA intrinsics
class ROCDL_Wmma_IntrOp<string mnemonic, list<int> overloadedOperands,
                        list<Trait> traits = []> :
  LLVM_IntrOpBase<ROCDL_Dialect, mnemonic,
                  "amdgcn_" # !subst(".","_", mnemonic),
                  [0], overloadedOperands, traits, 1>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  let assemblyFormat =
    "$args attr-dict `:` functional-type($args, $res)";
}

// Available from gfx11
def ROCDL_wmma_f32_16x16x16_f16 : ROCDL_Wmma_IntrOp<"wmma.f32.16x16x16.f16", [0]>;
def ROCDL_wmma_f32_16x16x16_bf16 : ROCDL_Wmma_IntrOp<"wmma.f32.16x16x16.bf16", [0]>;
def ROCDL_wmma_f16_16x16x16_f16 : ROCDL_Wmma_IntrOp<"wmma.f16.16x16x16.f16", [0]>;
def ROCDL_wmma_bf16_16x16x16_bf16 : ROCDL_Wmma_IntrOp<"wmma.bf16.16x16x16.bf16", [0]>;
def ROCDL_wmma_i32_16x16x16_iu8 : ROCDL_Wmma_IntrOp<"wmma.i32.16x16x16.iu8", [1]>;
def ROCDL_wmma_i32_16x16x16_iu4 : ROCDL_Wmma_IntrOp<"wmma.i32.16x16x16.iu4", [1]>;
// Available from gfx12
def ROCDL_wmma_f32_16x16x16_fp8 : ROCDL_Wmma_IntrOp<"wmma.f32.16x16x16.fp8_fp8", [1]>;
def ROCDL_wmma_f32_16x16x16_bf8 : ROCDL_Wmma_IntrOp<"wmma.f32.16x16x16.bf8_bf8", [1]>;

//===---------------------------------------------------------------------===//
// LDS transpose intrinsics (available in GFX950)

def ROCDLGlobalBuffer : LLVM_PointerInAddressSpace<1>;
def ROCDLBufferLDS : LLVM_PointerInAddressSpace<3>;

class ROCDL_LDS_Read_Tr_IntrOp<string mnemonic> :
  ROCDL_IntrOp<mnemonic, [1], [], [], 1>,
  Arguments<(ins Arg<ROCDLBufferLDS, "", [MemRead]>:$ptr)>{
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($res)";
}

def ROCDL_ds_read_tr4_b64 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr4.b64">;
def ROCDL_ds_read_tr8_b64 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr8.b64">;
def ROCDL_ds_read_tr6_b96 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr6.b96">;
def ROCDL_ds_read_tr16_b64 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr16.b64">;

//===---------------------------------------------------------------------===//
// Global load to LDS intrinsic (available in GFX950)

def ROCDL_GlobalLoadLDSOp :
  ROCDL_IntrOp<"global.load.lds", [], [], [], 0>,
  Arguments<(ins Arg<ROCDLGlobalBuffer, "", [MemRead]>:$globalPtr,
                 Arg<ROCDLBufferLDS, "", [MemWrite]>:$ldsPtr,
                 I32:$size,
                 I32:$offset,
                 I32:$aux)> {
  let assemblyFormat = "operands attr-dict";
}

//===---------------------------------------------------------------------===//
// Operations on raw buffer resources (stride of 0, bounds checks either off or in
// raw buffer mode).
//===---------------------------------------------------------------------===//

def ROCDLBufferRsrc : LLVM_PointerInAddressSpace<8>;

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

// DPP Update intrinsic
def ROCDL_DPPUpdateOp : ROCDL_IntrOp<"update.dpp", [], [0],
    [AllTypesMatch<["res", "src", "old"]>], 1>,
  Arguments<(ins LLVM_Type:$old, LLVM_Type:$src, I32Attr:$dppCtrl, I32Attr:$rowMask,
      I32Attr:$bankMask, I1Attr:$boundCtrl)> {
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = [{
    attr-dict $old `,` $src `with` $dppCtrl `,` $rowMask `,` $bankMask `,` $boundCtrl `:` type($src)
  }];
  string llvmBuilder = [{
      auto vdataType = moduleTranslation.convertType(op.getSrc().getType());
      llvm::Value *args[] = {
        moduleTranslation.lookupValue(op.getOld()),
        moduleTranslation.lookupValue(op.getSrc()),
          builder.getInt32(op.getDppCtrl()),
          builder.getInt32(op.getRowMask()),
          builder.getInt32(op.getBankMask()),
          builder.getInt1(op.getBoundCtrl())
      };
      $res = createIntrinsicCall(builder,
        llvm::Intrinsic::amdgcn_update_dpp, args, {vdataType});
  }];
}

//===---------------------------------------------------------------------===//
// 16-bit float intrinsics
//===---------------------------------------------------------------------===//
def ROCDL_CvtPkRtz:
    ROCDL_IntrOp<"cvt.pkrtz", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, F32:$srcB)> {
  let summary = "Convert two f32 input into a vector<2xf16>";
  let description = [{
    Convert two f32 values into a packed vector<2xf16>.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `:` type($res)
  }];
}

//===---------------------------------------------------------------------===//
// 32-bit float intrinsics
//===---------------------------------------------------------------------===//
def ROCDL_CvtScalePkF32Fp8 :
    ROCDL_IntrOp<"cvt.scalef32.pk.f32.fp8", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, F32: $scale, I1:$wordSel)> {
  let summary = "Scale and convert packed fp8 to packed f32";
  let description = [{
    Scale `src` by the exponent in `scale` then convert to packed fp32.
    Store the result in low/high word based on $wordSel, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $wordSel `]` `,` $scale `:` type($res)
  }];
}
def ROCDL_CvtScalePkF32Bf8 :
    ROCDL_IntrOp<"cvt.scalef32.pk.f32.bf8", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, F32: $scale, I1:$wordSel)> {
  let summary = "Scale and convert packed bf8 to packed f32";
  let description = [{
    Scale `src` by the exponent in `scale` then convert to packed fp32.
    Store the result in low/high word based on $wordSel, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $wordSel `]` `,` $scale `:` type($res)
  }];
}
//===---------------------------------------------------------------------===//
// 8-bit float scale intrinsics
//===---------------------------------------------------------------------===//
def ROCDL_V2I16Type : FixedVectorOfLengthAndType<[2], [I16]>,
                        BuildableType<"::mlir::VectorType::get("
                          "{2},$_builder.getI16Type())">;

def ROCDL_CvtScaleF32PkFp8F32:
    ROCDL_IntrOp<"cvt.scalef32.pk.fp8.f32", [], [], [Pure], 1>,
    Arguments<(ins ROCDL_V2I16Type:$old, F32:$srcA, F32:$srcB, F32:$scale, I1:$wordSel)> {
  let summary = "Scale and convert two f32's to packed fp8";
  let description = [{
    Scale `srcA` and `srcB` by the exponent in `scale` then convert to packed fp8
    and store into the low/high word of `old`, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `,` $scale `->` $old `[` $wordSel `]` `:` type($res)
  }];
}
    
def ROCDL_CvtScaleF32PkBf8F32:
    ROCDL_IntrOp<"cvt.scalef32.pk.bf8.f32", [], [], [Pure], 1>,
    Arguments<(ins ROCDL_V2I16Type:$old, F32:$srcA, F32:$srcB, F32: $scale, I1:$wordSel)> {
  let summary = "Scale and convert two f32's to packed bf8";
  let description = [{
    Scale `srcA` and `srcB` by the exponent in `scale` then convert to packed bf8
    and store into the low/high word of `old`, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `,` $scale `->` $old `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_CvtScaleF32SrFp8F32:
    ROCDL_IntrOp<"cvt.scalef32.sr.fp8.f32", [], [], [Pure], 1>,
    Arguments<(ins I32:$old, F32:$src, I32:$seed, F32: $scale, I32:$byteSel)> {
    let summary = "Scale and convert f32 to fp8 using stochastic rounding";
    let description = [{
       Scale `src` by the exponent in `scale` then convert to fp8 with stochastic rounding
       using seed data in `seed`. store into the `byteSel`th byte of `old`, preserving the others.
    }];
  let assemblyFormat = [{
    attr-dict $src `,` $seed `,` $scale `->` $old `[` $byteSel `]` `:` type($res)
  }];
}


def ROCDL_CvtScaleF32SrBf8F32:
    ROCDL_IntrOp<"cvt.scalef32.sr.bf8.f32", [], [], [Pure], 1>,
    Arguments<(ins I32:$old, F32:$src, I32:$seed, F32: $scale, I32:$byteSel)> {
    let summary = "Scale and convert f32 to bf8 using stochastic rounding";
    let description = [{
       Scale `src` by the exponent in `scale` then convert to bf8 with stochastic rounding
       using seed data in `seed`. store into the `byteSel`th byte of `old`, preserving the others.
    }];
  let assemblyFormat = [{
    attr-dict $src `,` $seed `,` $scale `->` $old `[` $byteSel `]` `:` type($res)
  }];
}

//===---------------------------------------------------------------------===//
// 8-bit float intrinsics
//===---------------------------------------------------------------------===//
def ROCDL_CvtF32Bf8Op :
    ROCDL_IntrOp<"cvt.f32.bf8", [], [], [Pure], 1>,
    Arguments<(ins I32:$srcA, I32:$byteSel)> {
  let summary = "Convert bf8 to f32";
  let description = [{
    Convert 8-bit bf8 value from the `byteSel`th bit of `srcA` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_CvtScaleF32Bf8Op :
    ROCDL_IntrOp<"cvt.scalef32.f32.bf8", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, F32: $scale, I32:$byteSel)> {
  let summary = "Scale and convert bf8 to f32";
  let description = [{
    Scale `src` by the exponent in `scale` then convert 8-bit bf8 value 
    from the `byteSel`th bit of `src` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $byteSel `]` `,` $scale `:` type($res)
  }];
}


def ROCDL_CvtF32Fp8Op :
    ROCDL_IntrOp<"cvt.f32.fp8", [], [], [Pure], 1>,
    Arguments<(ins I32:$srcA, I32:$byteSel)> {
  let summary = "Convert fp8 to f32";
  let description = [{
    Convert 8-bit fp8 value from the `byteSel`th bit of `srcA` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `[` $byteSel `]` `:` type($res)
  }];
}


def ROCDL_CvtScaleF32Fp8Op :
    ROCDL_IntrOp<"cvt.scalef32.f32.fp8", [], [], [Pure], 1>,
    Arguments<(ins I32:$src, F32: $scale, I32:$byteSel)> {
  let summary = "Scale and convert fp8 to f32";
  let description = [{
    Scale `src` by the exponent in `scale` then convert 8-bit fp8 value
    from the `byteSel`th bit of `src` to fp32.

  }];
  let assemblyFormat = [{
    attr-dict $src `[` $byteSel `]` `,` $scale `:` type($res)
  }];
}


def ROCDL_CvtPkBf8F32Op :
    ROCDL_IntrOp<"cvt.pk.bf8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, F32:$srcB, I32:$old, I1:$wordSel)> {
  let summary = "Convert two f32's to bf8";
  let description = [{
    Convert `srcA` and `srcB` to bf8 and store into the low/high word of
    `old`, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_CvtPkFp8F32Op :
    ROCDL_IntrOp<"cvt.pk.fp8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, F32:$srcB, I32:$old, I1:$wordSel)> {
  let summary = "Convert two f32's to fp8";
  let description = [{
    Convert `srcA` and `srcB` to fp8 and store into the low/high word of
    `old`, preserving the other word.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_CvtSrBf8F32Op :
    ROCDL_IntrOp<"cvt.sr.bf8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, I32:$srcB, I32:$old, I32:$byteSel)> {
  let summary = "Convert f32 to bf8, stochiastic rounding";
  let description = [{
    Convert `srcA` to bf8, adding the rounding factor from `srcB`,
    and store into the `byteSel`th byte of `old`, preserving the others.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_CvtSrFp8F32Op :
    ROCDL_IntrOp<"cvt.sr.fp8.f32", [], [], [Pure], 1>,
    Arguments<(ins F32:$srcA, I32:$srcB, I32:$old, I32:$byteSel)> {
  let summary = "Convert f32 to fp8, stochiastic rounding";
  let description = [{
    Convert `srcA` to fp8, adding the rounding factor from `srcB`,
    and store into the `byteSel`th byte of `old`, preserving the others.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// ROCDL target attribute.
//===----------------------------------------------------------------------===//

def ROCDL_TargetAttr :
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
    // Also update the default builder below and rocdl-attach-target in
    // Dialect/GPU/Transforms/Passes.td .
    StringRefParameter<"ABI version.", "\"500\"">:$abi,
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
                     CArg<"StringRef", "\"500\"">:$abiVersion,
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
