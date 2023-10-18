//===-- NVVMOps.td - NVVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the NVVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_OPS
#define NVVMIR_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.td"

def LLVM_i8Ptr_global : LLVM_IntPtrBase<8, 1>;
def LLVM_i8Ptr_shared : LLVM_IntPtrBase<8, 3>;
def LLVM_i64ptr_any : LLVM_IntPtrBase<64>;
def LLVM_i64ptr_shared : LLVM_IntPtrBase<64, 3>;

//===----------------------------------------------------------------------===//
// NVVM dialect definitions
//===----------------------------------------------------------------------===//

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "::mlir::NVVM";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "nvvm.kernel"; }
    /// Get the name of the attribute used to annotate max threads required
    /// per CTA for kernel functions.
    static StringRef getMaxntidAttrName() { return "nvvm.maxntid"; }
    /// Get the name of the metadata names for each dimension
    static StringRef getMaxntidXName() { return "maxntidx"; }
    static StringRef getMaxntidYName() { return "maxntidy"; }
    static StringRef getMaxntidZName() { return "maxntidz"; }

    /// Get the name of the attribute used to annotate exact threads required
    /// per CTA for kernel functions.
    static StringRef getReqntidAttrName() { return "nvvm.reqntid"; }
    /// Get the name of the metadata names for each dimension
    static StringRef getReqntidXName() { return "reqntidx"; }
    static StringRef getReqntidYName() { return "reqntidy"; }
    static StringRef getReqntidZName() { return "reqntidz"; }

    /// Get the name of the attribute used to annotate min CTA required
    /// per SM for kernel functions.
    static StringRef getMinctasmAttrName() { return "nvvm.minctasm"; }

    /// Get the name of the attribute used to annotate max number of
    /// registers that can be allocated per thread.
    static StringRef getMaxnregAttrName() { return "nvvm.maxnreg"; }
  }];

  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// NVVM op definitions
//===----------------------------------------------------------------------===//

class NVVM_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

/// Base class that defines BasicPtxBuilderOpInterface. 
class NVVM_PTXBuilder_Op<string mnemonic, 
  list<Trait> traits = [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>]> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// NVVM attribute definitions
//===----------------------------------------------------------------------===//

class NVVM_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<NVVM_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// NVVM intrinsic operations
//===----------------------------------------------------------------------===//

class NVVM_IntrOp<string mnem, list<Trait> traits,
                  int numResults>
  : LLVM_IntrOpBase<NVVM_Dialect, mnem, "nvvm_" # !subst(".", "_", mnem),
                    /*list<int> overloadedResults=*/[],
                    /*list<int> overloadedOperands=*/[],
                    traits, numResults>;


//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_SpecialRegisterOp<string mnemonic, list<Trait> traits = []> :
  NVVM_IntrOp<mnemonic, !listconcat(traits, [Pure]), 1> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Lane index and range
def NVVM_LaneIdOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.laneid">;
def NVVM_WarpSizeOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.warpsize">;

//===----------------------------------------------------------------------===//
// Thread index and range
def NVVM_ThreadIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_BlockDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_BlockDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_BlockDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.z">;

//===----------------------------------------------------------------------===//
// Block index and range
def NVVM_BlockIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_GridDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_GridDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_GridDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.z">;

//===----------------------------------------------------------------------===//
// CTA Cluster index and range
def NVVM_ClusterIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.clusterid.x">;
def NVVM_ClusterIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.clusterid.y">;
def NVVM_ClusterIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.clusterid.z">;
def NVVM_ClusterDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nclusterid.x">;
def NVVM_ClusterDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nclusterid.y">;
def NVVM_ClusterDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nclusterid.z">;


//===----------------------------------------------------------------------===//
// CTA index and range within Cluster
def NVVM_BlockInClusterIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.ctaid.x">;
def NVVM_BlockInClusterIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.ctaid.y">;
def NVVM_BlockInClusterIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.ctaid.z">;
def NVVM_GridInClusterDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.nctaid.x">;
def NVVM_GridInClusterDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.nctaid.y">;
def NVVM_GridInClusterDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.nctaid.z">;

//===----------------------------------------------------------------------===//
// CTA index and across Cluster dimensions
def NVVM_ClusterId : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.ctarank">;
def NVVM_ClusterDim : NVVM_SpecialRegisterOp<"read.ptx.sreg.cluster.nctarank">;

//===----------------------------------------------------------------------===//
// NVVM approximate op definitions
//===----------------------------------------------------------------------===//

def NVVM_RcpApproxFtzF32Op : NVVM_IntrOp<"rcp.approx.ftz.f", [Pure], 1> {
  let arguments = (ins F32:$arg);
  let results = (outs F32:$res);
  let assemblyFormat = "$arg attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// NVVM redux op definitions
//===----------------------------------------------------------------------===//

def ReduxKindNone : I32EnumAttrCase<"NONE", 0, "none">;
def ReduxKindAdd  : I32EnumAttrCase<"ADD", 1, "add">;
def ReduxKindAnd  : I32EnumAttrCase<"AND", 2, "and">;
def ReduxKindMax  : I32EnumAttrCase<"MAX", 3, "max">;
def ReduxKindMin  : I32EnumAttrCase<"MIN", 4, "min">;
def ReduxKindOr   : I32EnumAttrCase<"OR", 5, "or">;
def ReduxKindUmax : I32EnumAttrCase<"UMAX", 6, "umax">;
def ReduxKindUmin : I32EnumAttrCase<"UMIN", 7, "umin">;
def ReduxKindXor  : I32EnumAttrCase<"XOR", 8, "xor">; 

/// Enum attribute of the different kinds.
def ReduxKind : I32EnumAttr<"ReduxKind", "NVVM redux kind",
  [ReduxKindAdd, ReduxKindAnd, ReduxKindMax, ReduxKindMin, ReduxKindOr, 
    ReduxKindUmax, ReduxKindUmin, ReduxKindXor]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}

def ReduxKindAttr : EnumAttr<NVVM_Dialect, ReduxKind, "redux_kind">;

def NVVM_ReduxOp :
  NVVM_Op<"redux.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$val,
                 ReduxKindAttr:$kind,
                 I32:$mask_and_clamp)> {
  string llvmBuilder = [{
      auto intId = getReduxIntrinsicId($_resultType, $kind);
      $res = createIntrinsicCall(builder, intId, {$val, $mask_and_clamp});
  }];
  let assemblyFormat = [{
    $kind $val `,` $mask_and_clamp  attr-dict `:` type($val) `->` type($res)
   }];   
}

//===----------------------------------------------------------------------===//
// NVVM Split arrive/wait barrier
//===----------------------------------------------------------------------===//

/// mbarrier.init instruction with generic pointer type
def NVVM_MBarrierInitOp : NVVM_PTXBuilder_Op<"mbarrier.init">,
  Arguments<(ins LLVM_i64ptr_any:$addr, I32:$count, PtxPredicate:$predicate)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_init, {$addr, $count});
  }];
  let assemblyFormat = "$addr `,` $count (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDeclaration = [{
    bool hasIntrinsic() { if(getPredicate()) return false; return true; }
  }];
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("mbarrier.init.b64 [%0], %1;"); }
  }];
}

/// mbarrier.init instruction with shared pointer type
def NVVM_MBarrierInitSharedOp : NVVM_PTXBuilder_Op<"mbarrier.init.shared">,
  Arguments<(ins LLVM_i64ptr_shared:$addr, I32:$count, PtxPredicate:$predicate)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_init_shared, {$addr, $count});
  }];
  let assemblyFormat = "$addr `,` $count (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDeclaration = "bool hasIntrinsic() { return !getPredicate(); }";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("mbarrier.init.shared.b64 [%0], %1;"); }
  }];
}

def NVVM_MBarrierInvalOp : NVVM_Op<"mbarrier.inval">,
  Arguments<(ins LLVM_i64ptr_any:$addr)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_inval, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type(operands)";
}

def NVVM_MBarrierInvalSharedOp : NVVM_Op<"mbarrier.inval.shared">,
  Arguments<(ins LLVM_i64ptr_shared:$addr)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_inval_shared, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type(operands)";
}

def NVVM_MBarrierArriveOp : NVVM_Op<"mbarrier.arrive">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_i64ptr_any:$addr)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($res)";
}

def NVVM_MBarrierArriveSharedOp : NVVM_Op<"mbarrier.arrive.shared">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_i64ptr_shared:$addr)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive_shared, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($res)";
}

def NVVM_MBarrierArriveNocompleteOp : NVVM_Op<"mbarrier.arrive.nocomplete">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_i64ptr_any:$addr, I32:$count)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive_noComplete, {$addr, $count});
  }];
  let assemblyFormat = "$addr `,` $count attr-dict `:` type(operands) `->` type($res)";
}

def NVVM_MBarrierArriveNocompleteSharedOp : NVVM_Op<"mbarrier.arrive.nocomplete.shared">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_i64ptr_shared:$addr, I32:$count)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive_noComplete_shared, {$addr, $count});
  }];
  let assemblyFormat = "$addr `,` $count attr-dict `:` type(operands) `->` type($res)";
}

def NVVM_MBarrierArriveExpectTxOp : NVVM_PTXBuilder_Op<"mbarrier.arrive.expect_tx">,  
  Arguments<(ins LLVM_i64ptr_any:$addr, I32:$txcount, PtxPredicate:$predicate)> {
  let assemblyFormat = "$addr `,` $txcount (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("mbarrier.arrive.expect_tx.b64 _, [%0], %1;"); }
  }];
}

def NVVM_MBarrierArriveExpectTxSharedOp : NVVM_PTXBuilder_Op<"mbarrier.arrive.expect_tx.shared">,  
  Arguments<(ins LLVM_i64ptr_shared:$addr, I32:$txcount, PtxPredicate:$predicate)> {    
  let assemblyFormat = "$addr `,` $txcount (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"); }
  }];
}

def NVVM_MBarrierTryWaitParityOp : NVVM_PTXBuilder_Op<"mbarrier.try_wait.parity">,  
  Arguments<(ins LLVM_i64ptr_any:$addr, I32:$phase, I32:$ticks)> {  
  let assemblyFormat = "$addr `,` $phase `,` $ticks attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      return std::string(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra.uni DONE; \n\t"
        "bra.uni     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
      ); 
    }
  }];
}

def NVVM_MBarrierTryWaitParitySharedOp : NVVM_PTXBuilder_Op<"mbarrier.try_wait.parity.shared">,  
  Arguments<(ins LLVM_i64ptr_shared:$addr, I32:$phase, I32:$ticks)> {  
  let assemblyFormat = "$addr `,` $phase `,` $ticks attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      return std::string(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra.uni DONE; \n\t"
        "bra.uni     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
      ); 
    }
  }];
}

def NVVM_MBarrierTestWaitOp : NVVM_Op<"mbarrier.test.wait">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_i64ptr_any:$addr, LLVM_Type:$state)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_test_wait, {$addr, $state});
  }];
  let assemblyFormat = "$addr `,` $state attr-dict `:` type(operands) `->` type($res)";
}

def NVVM_MBarrierTestWaitSharedOp : NVVM_Op<"mbarrier.test.wait.shared">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_i64ptr_shared:$addr, LLVM_Type:$state)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_test_wait_shared, {$addr, $state});
  }];
  let assemblyFormat = "$addr `,` $state attr-dict `:` type(operands) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ClusterArriveOp : NVVM_Op<"cluster.arrive"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_arrive);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ClusterArriveRelaxedOp : NVVM_Op<"cluster.arrive.relaxed"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_arrive_relaxed);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ClusterWaitOp : NVVM_Op<"cluster.wait"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_wait);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_FenceScClusterOp : NVVM_Op<"fence.sc.cluster"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_fence_sc_cluster);
  }];
  let assemblyFormat = "attr-dict";
}

def ShflKindBfly : I32EnumAttrCase<"bfly", 0>;
def ShflKindUp   : I32EnumAttrCase<"up", 1>;
def ShflKindDown : I32EnumAttrCase<"down", 2>;
def ShflKindIdx  : I32EnumAttrCase<"idx", 3>;

/// Enum attribute of the different shuffle kinds.
def ShflKind : I32EnumAttr<"ShflKind", "NVVM shuffle kind",
  [ShflKindBfly, ShflKindUp, ShflKindDown, ShflKindIdx]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ShflKindAttr : EnumAttr<NVVM_Dialect, ShflKind, "shfl_kind">;

def NVVM_ShflOp :
  NVVM_Op<"shfl.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins I32:$dst,
                 LLVM_Type:$val,
                 I32:$offset,
                 I32:$mask_and_clamp,
                 ShflKindAttr:$kind,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  string llvmBuilder = [{
      auto intId = getShflIntrinsicId(
          $_resultType, $kind, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$dst, $val, $offset, $mask_and_clamp});
  }];
  let assemblyFormat = [{
    $kind $dst `,` $val `,` $offset `,` $mask_and_clamp  attr-dict
     `:` type($val) `->` type($res)
   }];
   let hasVerifier = 1;
}

def NVVM_VoteBallotOp :
  NVVM_Op<"vote.ballot.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$mask, LLVM_Type:$pred)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {$mask, $pred});
  }];
  let hasCustomAssemblyFormat = 1;
}

def NVVM_SyncWarpOp :
  NVVM_Op<"bar.warp.sync">,
  Arguments<(ins LLVM_Type:$mask)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_bar_warp_sync, {$mask});
  }];
  let assemblyFormat = "$mask attr-dict `:` type($mask)";
}


def NVVM_ElectSyncOp : NVVM_Op<"elect.sync", 
                  [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>]>
{  
  let results = (outs I1:$pred);
  let assemblyFormat = "attr-dict `->` type(results)";  
  let extraClassDefinition = [{        
    std::string $cppClass::getPtx() { 
      return std::string(
        "{                                  \n"
        ".reg .u32 rx;                      \n"
        ".reg .pred px;                     \n"
        " mov.u32 %0, 0;                    \n"
        "    elect.sync rx | px, 0xFFFFFFFF;\n"
        "@px mov.u32 %0, 1;                 \n"
        "}\n"
      ); 
    }
  }];
}

def LoadCacheModifierCA : I32EnumAttrCase<"CA", 0, "ca">;
def LoadCacheModifierCG : I32EnumAttrCase<"CG", 1, "cg">;
def LoadCacheModifierCS : I32EnumAttrCase<"CS", 2, "cs">;
def LoadCacheModifierLU : I32EnumAttrCase<"LU", 3, "lu">;
def LoadCacheModifierCV : I32EnumAttrCase<"CV", 4, "cv">;

/// Enum attribute of the different kinds.
def LoadCacheModifierKind : I32EnumAttr<"LoadCacheModifierKind", 
                                "NVVM load cache modifier kind",
  [LoadCacheModifierCA, LoadCacheModifierCG, LoadCacheModifierCS, 
    LoadCacheModifierLU, LoadCacheModifierCV]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
  let description = [{
    Enum attribute of the different kinds of cache operators for load instructions.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#id62)    
  }];
}

def LoadCacheModifierAttr : EnumAttr<NVVM_Dialect, LoadCacheModifierKind, "load_cache_modifier">;

def NVVM_CpAsyncOp : NVVM_PTXBuilder_Op<"cp.async.shared.global">,
  Arguments<(ins LLVM_i8Ptr_shared:$dst,
                 LLVM_i8Ptr_global:$src,
                 I32Attr:$size,
                 LoadCacheModifierAttr:$modifier,
                 Optional<LLVM_Type>:$cpSize)> {
  string llvmBuilder = [{
      llvm::Intrinsic::ID id;
      switch ($size) {
        case 4:
          id = llvm::Intrinsic::nvvm_cp_async_ca_shared_global_4;
          break;
        case 8:
          id = llvm::Intrinsic::nvvm_cp_async_ca_shared_global_8;
          break;
        case 16:
          if($modifier == NVVM::LoadCacheModifierKind::CG)
            id = llvm::Intrinsic::nvvm_cp_async_cg_shared_global_16;
          else if($modifier == NVVM::LoadCacheModifierKind::CA)
            id = llvm::Intrinsic::nvvm_cp_async_ca_shared_global_16;
          else 
            llvm_unreachable("unsupported cache modifier");
          break;
        default:
          llvm_unreachable("unsupported async copy size");
      }
      createIntrinsicCall(builder, id, {$dst, $src});
  }];
  let assemblyFormat = "$dst `,` $src `,` $size `,` `cache` `=` $modifier (`,` $cpSize^)? attr-dict `:` type(operands)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool hasIntrinsic() { if(getCpSize()) return false; return true; }

    void getAsmValues(RewriterBase &rewriter, 
        llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>> &asmValues) {
      asmValues.push_back({getDst(), PTXRegisterMod::Read});
      asmValues.push_back({getSrc(), PTXRegisterMod::Read});
      asmValues.push_back({makeConstantI32(rewriter, getSize()), PTXRegisterMod::Read});
      asmValues.push_back({getCpSize(), PTXRegisterMod::Read});
    }        
  }];
  let extraClassDefinition = [{        
    std::string $cppClass::getPtx() { 
      if(getModifier() == NVVM::LoadCacheModifierKind::CG)
        return std::string("cp.async.cg.shared.global [%0], [%1], %2, %3;\n");
      if(getModifier() == NVVM::LoadCacheModifierKind::CA)
        return std::string("cp.async.ca.shared.global [%0], [%1], %2, %3;\n");
      llvm_unreachable("unsupported cache modifier");      
    }
  }];
}

def NVVM_CpAsyncCommitGroupOp : NVVM_Op<"cp.async.commit.group"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_cp_async_commit_group);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_CpAsyncWaitGroupOp : NVVM_Op<"cp.async.wait.group">,
  Arguments<(ins I32Attr:$n)> {
  string llvmBuilder = [{
      createIntrinsicCall(
        builder,
        llvm::Intrinsic::nvvm_cp_async_wait_group,
        llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(moduleTranslation.getLLVMContext()),
          $n));
  }];
  let assemblyFormat = "$n attr-dict";
}

/// Helpers to instantiate different version of wmma intrinsics.
/// This matches the hierarchy used in IntrinsicsNVVM.td to define all the
/// combinations of the intrinsics.
class GEOM<int M, int N, int K> {
  int m = M;
  int n = N;
  int k = K;
}

/// Class containing information about valid mma matrix types.
class WMMA_REGS<GEOM Geom, string Frag, string PtxEltType> {
  int m = Geom.m;
  int n = Geom.n;
  int k = Geom.k;
  string geom = "m"#Geom.m#"n"#Geom.n#"k"#Geom.k;
  string frag = Frag;
  string ptx_elt_type = PtxEltType;
  string gft = geom#":"#Frag#":"#ptx_elt_type;
}

//// Generate enum value of the mma.load/mma.store intrinsic.
class WMMA_NAME_LDST<string Op, WMMA_REGS Frag, string Layout, int WithStride> {
  string id =   "llvm::Intrinsic::nvvm_wmma"
                # "_" # Frag.geom
                # "_" # Op
                # "_" # Frag.frag
                # "_" # Frag.ptx_elt_type
                # "_" # Layout
                # !if(WithStride, "_stride", "");
}

/// Generate the signature part of the mma intrinsic name.
class MMA_SIGNATURE<WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  list<WMMA_REGS> id_frags = !cond(
     // FP16 ops are identified by accumulator & result type.
     !eq(A.ptx_elt_type, "f16") : [D, C],
     // other ops are identified by input types.
     !ne(A.ptx_elt_type, B.ptx_elt_type): [A, B],
     true: [A]
     );
   string ret = !foldl("", id_frags, a, b, !strconcat(a, "_", b.ptx_elt_type));
}

/// Generate enum value of the wmma.mma intrinsic.
class WMMA_NAME<string Op, string ALayout, string BLayout, WMMA_REGS A,
  WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  string signature = MMA_SIGNATURE<A, B, C, D>.ret;
  string id =   "llvm::Intrinsic::nvvm_wmma"
                # "_" # A.geom
                # "_" # Op
                # "_" # ALayout
                # "_" # BLayout
                # signature;
}

// Generates list of 4-tuples of WMMA_REGS representing a valid MMA op.
//   Geom: list of supported geometries.
//   TypeN: PTX type of the corresponding fragment's element.
//   TypeB and TypeD may be empty if it must match that of TypeA or TypeC.
class MMA_OPS<list<GEOM> Geom, list<string> TypeA, list<string> TypeB,
            list<string> TypeC, list<string> TypeD> {
  list<list<WMMA_REGS>> ret =
     !foldl([]<list<WMMA_REGS>>, Geom, t1, geom, !listconcat(t1,
     !foldl([]<list<WMMA_REGS>>, TypeA, t2, type_a, !listconcat(t2,
     !foldl([]<list<WMMA_REGS>>, !if(!size(TypeB), TypeB, [type_a]), t3, type_b, !listconcat(t3,
     !foldl([]<list<WMMA_REGS>>, TypeC, t4, type_c, !listconcat(t4,
     !foldl([]<list<WMMA_REGS>>, !if(!size(TypeD), TypeD, [type_c]), t5, type_d, !listconcat(t5,
            [[WMMA_REGS<geom, "a", type_a>,
              WMMA_REGS<geom, "b", type_b>,
              WMMA_REGS<geom, "c", type_c>,
              WMMA_REGS<geom, "d", type_d>]]))))))))));
   // Debugging aid for readable representation of the list above.
   list<list<string>> ops = !foreach(x, ret, [x[0].gft, x[1].gft, x[2].gft, x[3].gft]);
}

/// Creates a list of combinations of load/store operations supported.
class MMA_LDST_OPS<list<GEOM> Geom, list<string> Frags, list<string> Types> {
  list<WMMA_REGS> ret =
     !foldl([]<WMMA_REGS>, Geom, t1, geom, !listconcat(t1,
     !foldl([]<WMMA_REGS>, Frags, t2, frag, !listconcat(t2,
     !foldl([]<WMMA_REGS>, Types, t3, type, !listconcat(t3,
            [WMMA_REGS<geom, frag, type>]))))));
   // Debugging aid for readable representation of the list above.
   list<string> ops = !foreach(x, ret, x.gft);
}

// Creates list of valid combinations of fragments. This is a subset of what
// llvm supports and can be extended as needed.
class NVVM_MMA_OPS {
  // "wmma" operations
  list<list<WMMA_REGS>> tf32_wmma_ops = MMA_OPS<
            [GEOM<16, 16, 8>],
            ["tf32"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> fp_wmma_ops = MMA_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["f16"], [], ["f16", "f32"], []>.ret;
  list<list<WMMA_REGS>> i8_wmma_ops = MMA_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["s8","u8"], [], ["s32"], []>.ret;
  list<list<WMMA_REGS>> all_wmma_ops = !listconcat(
            tf32_wmma_ops,
            fp_wmma_ops,
            i8_wmma_ops);

  list<WMMA_REGS> ldst_ab_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["a", "b"], ["f16","s8","u8"]>.ret;
  list<WMMA_REGS> ldst_cd_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 16>, GEOM<32, 8, 16>, GEOM<8, 32, 16>],
            ["c", "d"], ["f16", "f32","s32"]>.ret;
  list<WMMA_REGS> ldst_tf32_ab_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 8>],
            ["a", "b"], ["tf32"]>.ret;
  list<WMMA_REGS> ldst_tf32_cd_ops = MMA_LDST_OPS<
            [GEOM<16, 16, 8>],
            ["c", "d"], ["f32"]>.ret;
  list<WMMA_REGS> all_ldst_ops = !listconcat(ldst_ab_ops, ldst_cd_ops,
                                             ldst_tf32_ab_ops,
                                             ldst_tf32_cd_ops);
  // Separate A/B/C fragments (loads) from D (stores).
  list<WMMA_REGS> all_ld_ops = !filter(op, all_ldst_ops, !ne(op.frag, "d"));
  list<WMMA_REGS> all_st_ops = !filter(op, all_ldst_ops, !eq(op.frag, "d"));

  // "mma_sync" operations
  list<list<WMMA_REGS>> tf32_mma_ops = MMA_OPS<
            [GEOM<16,8,4>, GEOM<16,8,8>],
            ["tf32"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> bf16_mma_ops = MMA_OPS<
            [GEOM<16,8,16>, GEOM<16,8,8>],
            ["bf16"], [], ["f32"], []>.ret;
  list<list<WMMA_REGS>> f64_mma_ops = MMA_OPS<
            [GEOM<8,8,4>],
            ["f64"], [], ["f64"], []>.ret;
  list<list<WMMA_REGS>> fp_mma_ops = MMA_OPS<
            [GEOM<8,8,4>, GEOM<16,8,8>, GEOM<16,8,16>],
            ["f16"], [], ["f16", "f32"], ["f16", "f32"]>.ret;
  list<list<WMMA_REGS>> int_mma_ops = MMA_OPS<
            [GEOM<8,8,16>, GEOM<16,8,16>, GEOM<16,8,32>],
            ["s8", "u8"], ["s8", "u8"], ["s32"], []>.ret;
  list<list<WMMA_REGS>> subint_mma_ops = MMA_OPS<
            [GEOM<8,8,32>, GEOM<16,8,32>, GEOM<16,8,64>],
            ["s4", "u4"], ["s4", "u4"], ["s32"], []>.ret;
  list<list<WMMA_REGS>> bit_mma_ops = MMA_OPS<
            [GEOM<8,8,128>, GEOM<16,8,128>, GEOM<16,8,256>],
            ["b1"], [], ["s32"], []>.ret;
  list<list<WMMA_REGS>> all_mma_sync_ops = !listconcat(
            tf32_mma_ops, bf16_mma_ops, f64_mma_ops,
            fp_mma_ops, int_mma_ops, subint_mma_ops, bit_mma_ops);
}

def NVVM_MMA_OPS : NVVM_MMA_OPS;

/// Helper to create the mapping between the configuration and the store
/// intrinsic enum value.
class MMA_ST_INTR<string op> {
  list<list<string>> cond0 = !foreach(frag, NVVM_MMA_OPS.all_st_ops,
                                !foreach(layout, ["row", "col"],
  "if (layout == \"" # layout #  "\" && m == " # frag.m # " &&"
  "    n == " #frag.n # " && k == " # frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return " #WMMA_NAME_LDST<op, frag, layout, 1>.id #";"));
  string id = !foldl("",
                !foldl([""], cond0, acc, el, !listconcat(acc, el)),
                acc1, el1, acc1 # "\n" # el1);
}

/// Helper to map a mxk shape to a supported mxnxk matrix type. This will return
/// the n value of the supported configuration.
class MMA_ST_INFER_N<list<WMMA_REGS> ldst> {
  list<string> cond = !foreach(frag, ldst,
  "if (m == " # frag.m # " && k == " #frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return "# frag.n #";");
  string id = !foldl("", cond, acc, el, acc # "\n" # el);
}

/// Helper to map a kxn shape to a supported mxnxk matrix type. This will return
/// the m value of the supported configuration.
class MMA_ST_INFER_M<list<WMMA_REGS> ldst> {
  list<string> cond = !foreach(frag, ldst,
  "if (n == " # frag.n # " && k == " #frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return "# frag.m #";");
  string id = !foldl("", cond, acc, el, acc # "\n" # el);
}

/// Helper to map a mxn shape to a supported mxnxk matrix type. This will return
/// the k value of the supported configuration.
class MMA_ST_INFER_K<list<WMMA_REGS> ldst> {
  list<string> cond = !foreach(frag, ldst,
  "if (m == " # frag.m # " && n == " #frag.n # " && \"" #
       frag.ptx_elt_type # "\" == eltype)"
  "  return "# frag.k #";");
  string id = !foldl("", cond, acc, el, acc # "\n" # el);
}

/// Helper to create the mapping between the configuration and the load
/// intrinsic enum value.
class MMA_LD_INTR<string op> {
  list<list<string>> cond0 = !foreach(frag, NVVM_MMA_OPS.all_ld_ops,
                                !foreach(layout, ["row", "col"],
  "if (layout == \"" # layout #  "\" && m == " # frag.m # " &&"
  "    n == " #frag.n # " && k == " # frag.k # " && \"" #
       frag.ptx_elt_type # "\" == eltype && frag == \""#frag.frag#"\")"
  "  return "# WMMA_NAME_LDST<op, frag, layout, 1>.id #";"));
  string id = !foldl("",
                !foldl([""], cond0, acc, el, !listconcat(acc, el)),
                acc1, el1, acc1 # "\n" # el1);
}

/// Helper to create the mapping between the configuration and the wmma.mma
/// intrinsic enum value.
class MMA_MMA_INTR<string opName> {
  list<list<list<string>>> cond0 =
    !foreach(op, NVVM_MMA_OPS.all_wmma_ops,
      !foreach(layoutA, ["row", "col"],
        !foreach(layoutB, ["row", "col"],
  "if (layoutA == \"" # layoutA #  "\" && layoutB == \"" # layoutB #  "\" && "
  "    m == " # op[0].m # " && n == " #op[0].n # " && k == " # op[0].k #
  "    && \"" # op[0].ptx_elt_type # "\" == eltypeA && \""
   # op[3].ptx_elt_type # "\" == eltypeB)"
  "  return " #
       WMMA_NAME<opName, layoutA, layoutB, op[0], op[1], op[2], op[3]>.id # ";")));
  list<string> f = !foldl([""],
                     !foldl([[""]], cond0, acc, el, !listconcat(acc, el)),
                          acc1, el1, !listconcat(acc1, el1));
  string id = !foldl("", f, acc, el, acc # "\n" # el);
}

/// Enum attribute for binary (b1) MMA operation type
def MMAB1OpNone : I32EnumAttrCase<"none", 0>;
def MMAB1OpXorPopc : I32EnumAttrCase<"xor_popc", 1>;
def MMAB1OpAndPopc : I32EnumAttrCase<"and_popc", 2>;
def MMAB1Op : I32EnumAttr<"MMAB1Op", "MMA binary operations",
  [MMAB1OpNone, MMAB1OpXorPopc, MMAB1OpAndPopc]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAB1OpAttr : EnumAttr<NVVM_Dialect, MMAB1Op, "mma_b1op"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Enum attribute type for the overflow behavior of MMA integer operations
def MMAIntOverflowWrap : I32EnumAttrCase<"wrapped", 0>;
def MMAIntOverflowSat : I32EnumAttrCase<"satfinite", 1>;
def MMAIntOverflow : I32EnumAttr<"MMAIntOverflow", "MMA overflow options",
  [MMAIntOverflowSat, MMAIntOverflowWrap]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAIntOverflowAttr : EnumAttr<NVVM_Dialect, MMAIntOverflow, "mma_int_overflow"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Attribute to hold the MMA shape
def NVVM_MMAShapeAttr : NVVM_Attr<"MMAShape", "shape"> {
  let summary = "Attribute for MMA operation shape.";
  let parameters = (ins "int":$m, "int":$n, "int":$k);
  let assemblyFormat = "`<` struct(params) `>`";
}

// Returns true if this combination of layout/satf for MMA ops is supported;
// false otherwise.
// E.g.
// if NVVM_MMA_SUPPORTED<...>.ret then
//   def : FOO<>; // The record will only be defined for supported ops.
//
class NVVM_MMA_SUPPORTED<list<WMMA_REGS> frags, string layout_a, string layout_b, int satf> {
  // MMA ops check both layouts.
  string layout = layout_a # ":" # layout_b;
  string a_type = frags[0].ptx_elt_type;
  string b_type = frags[1].ptx_elt_type;
  string c_type = frags[2].ptx_elt_type;
  string d_type = frags[3].ptx_elt_type;
  string geom = frags[0].geom;

  // gcd is a shortcut used to identify instructions that depend on
  // geom+frag_c+frag_d.
  string gcd = geom # ":" # c_type # d_type;
  bit ret = !cond(

    // Limit satf to valid types
    !and(!eq(satf, 1),
         !ne(a_type, "s8"),
         !ne(a_type, "u8"),
         !ne(a_type, "s4"),
         !ne(a_type, "u4")): false,

    // m8n8k4 has no C=f32 D=f16 variant.
    !eq(gcd, "m8n8k4:f32f16"): false,

    // only m8n8k4 for f16 does not require row:col layout
    !and(!ne(layout, "row:col"),
         !or(!ne(geom, "m8n8k4"),
             !ne(a_type, "f16"))) : false,

    // m16n8k8 requires A and B to be the same type and C and D to be the same
    // type.
    !and(!eq(geom, "m16n8k8"),
         !or(!ne(a_type, b_type),
             !ne(c_type, d_type))): false,

    // m16n8k8 requires C and D to be the same type.
    !and(!eq(geom, "m16n8k8"),
         !ne(c_type, d_type)): false,

    // All other are OK.
    true: true
  );
}

// Returns a list of operation suffixes corresponding to possible b1
// multiply-and-accumulate operations for all fragments which have a
// b1 type. For all other fragments, the list returned holds a list
// containing the empty string.
class NVVM_MMA_B1OPS<list<WMMA_REGS> frags> {
  list<string> ret = !cond(
    !eq(frags[0].ptx_elt_type, "b1") : ["xor_popc", "and_popc"],
    true: [""]
  );
}

/// Generate enum value of the mma.sync intrinsic.
class MMA_SYNC_NAME<string ALayout, string BLayout, string b1op, int Satfinite,
               WMMA_REGS A, WMMA_REGS B, WMMA_REGS C, WMMA_REGS D> {
  string signature = MMA_SIGNATURE<A, B, C, D>.ret;
  string id = "llvm::Intrinsic::nvvm_mma"
                # !if(!ne(b1op, ""), "_" # b1op, "")
                # "_" # A.geom
                # "_" # ALayout
                # "_" # BLayout
                # !if(Satfinite, "_satfinite", "")
                # signature;
}

/// Helper to create the mapping between the configuration and the mma.sync
/// intrinsic enum value.
class MMA_SYNC_INTR {
  list<list<list<list<list<string>>>>> cond0 =
    !foreach(op, NVVM_MMA_OPS.all_mma_sync_ops,
      !foreach(layoutA, ["row", "col"],
        !foreach(layoutB, ["row", "col"],
          !foreach (sat, [0, 1],
            !foreach (b1op, NVVM_MMA_B1OPS<op>.ret,
              !if(NVVM_MMA_SUPPORTED<[op[0], op[1], op[2], op[3]],
                                     layoutA, layoutB, sat>.ret,
      "if (layoutA == \"" # layoutA #  "\" && layoutB == \"" # layoutB #  "\" && "
      "    m == " # op[0].m # " && n == " # op[0].n # " && k == " # op[0].k #
      "    && \"" # op[0].ptx_elt_type # "\" == eltypeA && \""
       # op[1].ptx_elt_type # "\" == eltypeB && "
       # " \"" # op[2].ptx_elt_type # "\" == eltypeC && "
       # " \"" # op[3].ptx_elt_type # "\" == eltypeD "
       # " && (sat.has_value()  ? " # sat # " == static_cast<int>(*sat) : true)"
       # !if(!ne(b1op, ""), " && (b1Op.has_value() ? MMAB1Op::" # b1op # " == *b1Op : true)", "") # ")\n"
       # "  return " #
       MMA_SYNC_NAME<layoutA, layoutB, b1op, sat, op[0], op[1], op[2], op[3]>.id # ";",
          "") // if supported
          ) // b1op
        ) // sat
      ) // layoutB
    ) // layoutA
  ); // all_mma_sync_ops
  list<list<list<string>>> f1 = !foldl([[[""]]],
                                  !foldl([[[[""]]]], cond0, acc, el,
                                      !listconcat(acc, el)),
                                    acc1, el1, !listconcat(acc1, el1));
  list<list<string>> f2 = !foldl([[""]], f1, acc1, el1, !listconcat(acc1, el1));
  list<string> f3 = !foldl([""], f2, acc, el, !listconcat(acc, el));
  string id = !foldl("", f3, acc, el, acc # "\n" # el);
}

def MMALayoutRow : I32EnumAttrCase<"row", 0>;
def MMALayoutCol : I32EnumAttrCase<"col", 1>;

/// Enum attribute of the different matrix layout.
def MMALayout : I32EnumAttr<"MMALayout", "NVVM MMA layout",
  [MMALayoutRow, MMALayoutCol]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMALayoutAttr : EnumAttr<NVVM_Dialect, MMALayout, "mma_layout"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Enum attribute of the different PTX element types used for MMA operands.
def MMATypeF16  : I32EnumAttrCase<"f16", 0>;
def MMATypeF32  : I32EnumAttrCase<"f32", 1>;
def MMATypeTF32 : I32EnumAttrCase<"tf32", 2>;
def MMATypeU8 : I32EnumAttrCase<"u8", 3>;
def MMATypeS8 : I32EnumAttrCase<"s8", 4>;
def MMATypeS32 : I32EnumAttrCase<"s32", 5>;
def MMATypeB1 : I32EnumAttrCase<"b1", 6>;
def MMATypeU4 : I32EnumAttrCase<"u4", 7>;
def MMATypeS4 : I32EnumAttrCase<"s4", 8>;
def MMATypeBF16 : I32EnumAttrCase<"bf16", 9>;
def MMATypeF64 : I32EnumAttrCase<"f64", 10>;

def MMATypes : I32EnumAttr<"MMATypes", "NVVM MMA types",
  [MMATypeF16, MMATypeF32, MMATypeTF32,
  MMATypeBF16, MMATypeS8, MMATypeU8,
  MMATypeS32, MMATypeS4, MMATypeU4,
  MMATypeB1, MMATypeF64]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMATypesAttr : EnumAttr<NVVM_Dialect, MMATypes, "mma_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def MMAFragA : I32EnumAttrCase<"a", 0>;
def MMAFragB : I32EnumAttrCase<"b", 1>;
def MMAFragC : I32EnumAttrCase<"c", 2>;

/// Enum attribute of the different frag types.
def MMAFrag: I32EnumAttr<"MMAFrag", "NVVM MMA frag type",
  [MMAFragA, MMAFragB, MMAFragC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MMAFragAttr : EnumAttr<NVVM_Dialect, MMAFrag, "mma_frag"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_WMMALoadOp: NVVM_Op<"wmma.load">,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins LLVM_AnyPointer: $ptr, I32: $stride, I32Attr:$m,
             I32Attr:$n, I32Attr:$k, MMALayoutAttr:$layout,
             MMATypesAttr:$eltype, MMAFragAttr:$frag)> {

  let summary = "Warp synchronous matrix load";

  // Since LLVM intrinsic IDs are enum that cannot be dynamically generated in
  // C++ we instanciate a function in tablegen to map the valide configuration
  // to the corresponsding intrinsic ID.
  // Because we want a single source of truth, this mean the source of truth
  // about valid combinations needs to be in tablgen, therefore we generate
  // extra helpers to query valid configurations based on the shapes of
  // load/store operations.
  let extraClassDeclaration =
    "static llvm::Intrinsic::ID getIntrinsicID("
    "int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,"
    "mlir::NVVM::MMATypes eltypeEnum,mlir::NVVM::MMAFrag fragEnum) {"
    "llvm::StringRef layout = stringifyEnum(layoutEnum);"
    "llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    "llvm::StringRef frag = stringifyEnum(fragEnum);"
    #MMA_LD_INTR<"load">.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid n dimension based on mxk load shape.\n"
    "static int inferNDimension(int m, int k, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_N<!filter(op, NVVM_MMA_OPS.all_ld_ops, !eq(op.frag, "a"))>.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid m dimension based on kxn load shape.\n"
    "static int inferMDimension(int k, int n, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_M<!filter(op, NVVM_MMA_OPS.all_ld_ops, !eq(op.frag, "b"))>.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid k dimension based on mxn load shape.\n"
    "static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_K<!filter(op, NVVM_MMA_OPS.all_ld_ops, !eq(op.frag, "c"))>.id# "\n"
    "return 0;"
    "}\n";


  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = mlir::NVVM::WMMALoadOp::getIntrinsicID(
        $m, $n, $k, $layout, $eltype, $frag);
      $res = createIntrinsicCall(builder, intId, operands, {operands[0]->getType()});
  }];

  string baseDescription = [{
    The `nvvm.wmma.load` operation loads a matrix collectively using all the
    threads in a warp.

    The operation takes two arguments, the address from where the matrix
    elements are to be loaded from and a stride. The stride argument
    represents the leading dimension of the source matrix. The address and
    the stride are required to be the same across all threads in the warp.
    Each thread in a warp holds a certain number of elements. The Op returns
    a LLVMStruct which holds the elements of the matrix held by this thread.

    This op is meant to be used along with `nvvm.wmma.store` and
    `nvvm.wmma.mma`.

    Example:

    ```mlir
    %2 = nvvm.wmma.load %0, %1
      {eltype = "f16", frag = "a", k = 16 : i32, layout = "row", m = 16 : i32, n = 16 : i32}
      : (!llvm.ptr<i32, 3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    ```
    }];

  let assemblyFormat = "$ptr `,` $stride attr-dict `:` functional-type($ptr, $res)";
  let hasVerifier = 1;
}

def NVVM_WMMAStoreOp : NVVM_Op<"wmma.store">,
  Arguments<(ins LLVM_AnyPointer: $ptr,
             I32Attr:$m, I32Attr:$n, I32Attr:$k, MMALayoutAttr:$layout,
             MMATypesAttr:$eltype, Variadic<LLVM_Type>:$args, I32: $stride)>{
  let summary = "Warp synchronous matrix store";

  let extraClassDeclaration =
    "static llvm::Intrinsic::ID getIntrinsicID("
    "int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,"
    "mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef layout = stringifyEnum(layoutEnum);"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INTR<"store">.id# "\n"
    "return 0;"
    "}\n"
    "/// Helpers to find valid k dimension based on mxn store shape.\n"
    "static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {"
    "  llvm::StringRef eltype = stringifyEnum(eltypeEnum);"
    #MMA_ST_INFER_K<NVVM_MMA_OPS.all_st_ops>.id#  "\n"
    "return 0;"
    "}";

  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId =
        mlir::NVVM::WMMAStoreOp::getIntrinsicID($m, $n, $k, $layout, $eltype);
      createIntrinsicCall(builder, intId, operands, {operands[0]->getType()});
  }];

  string baseDescription = [{
    The `nvvm.wmma.store` operation stores a matrix collectively using
    all the threads in a warp.

    The operation takes as arguments the address to where the matrix elements are
    to be stored, a stride and the elements to store, held by the current thread.
    The stride argument represents the leading dimension of the destination matrix.
    The address and the stride are required to be the same across all threads in the
    warp.

    This op is meant to be used along with `nvvm.wmma.m16n16k16.load` and
    `nvvm.wmma.m16n16k16.mma`.

    Example:

    ```mlir
    nvvm.wmma.store %0, %1, %2, %3, %4, %5
      {eltype = "f16", k = 16 : i32, layout = "row", m = 16 : i32, n = 16 : i32}
      : !llvm.ptr<i32, 3>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
    ```
  }];

  let assemblyFormat = [{
    $ptr `,` $stride `,` $args attr-dict `:` qualified(type($ptr)) `,`
    type($args)
  }];
  let hasVerifier = 1;
}

// Base class for all the variants of WMMA mmaOps that may be defined.
def NVVM_WMMAMmaOp : NVVM_Op<"wmma.mma">,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins I32Attr:$m, I32Attr:$n, I32Attr:$k, MMALayoutAttr:$layoutA,
             MMALayoutAttr:$layoutB, MMATypesAttr:$eltypeA,
             MMATypesAttr:$eltypeB, Variadic<LLVM_Type>:$args)>{
  let summary = "Warp synchronous matrix-multiply accumulate using tensor cores.";

  let extraClassDeclaration =
    "static llvm::Intrinsic::ID getIntrinsicID("
    "int m, int n, int k, mlir::NVVM::MMALayout layoutAEnum,"
    "mlir::NVVM::MMALayout layoutBEnum, mlir::NVVM::MMATypes eltypeAEnum,"
    "mlir::NVVM::MMATypes eltypeBEnum) {"
    "llvm::StringRef layoutA = stringifyEnum(layoutAEnum);"
    "llvm::StringRef layoutB = stringifyEnum(layoutBEnum);"
    "llvm::StringRef eltypeA = stringifyEnum(eltypeAEnum);"
    "llvm::StringRef eltypeB = stringifyEnum(eltypeBEnum);"
    #MMA_MMA_INTR<"mma">.id# "\n"
    "return 0;"
    "}";

  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = mlir::NVVM::WMMAMmaOp::getIntrinsicID(
        $m, $n, $k, $layoutA, $layoutB, $eltypeA, $eltypeB);
      $res = createIntrinsicCall(builder, intId, operands);
  }];

  string baseDescription = [{
    The `nvvm.wmma.mma` operation performs a matrix-multiply accumulate
    (mma) operation using all the threads in a warp.

    The operation performed is represented as `D = A * B + C`. The operation takes
    as arguments the elements of the matrices `A`, `B`, `C` and `D`, held by the
    current thread. The op returns a LLVM struct which holds a part of the result
    held by the current thread.

    This op is meant to be used along with `nvvm.wmma.load` and
    `nvvm.wmma.store`.

    Example:

    ```mlir
    %16 = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
      {eltypeA = "tf32", eltypeB = "f32", k = 8 : i32, layoutA = "row", layoutB = "row", m = 16 : i32, n = 16 : i32}
      : (i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32, f32, f32, f32)
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    ```
  }];

  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
  let hasVerifier = 1;
}

def NVVM_LdMatrixOp: NVVM_Op<"ldmatrix">,
  Results<(outs AnyType:$res)>,
  Arguments<(ins LLVM_AnyPointer: $ptr, I32Attr:$num, MMALayoutAttr:$layout)> {

  let summary = "cooperative matrix load";

  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = getLdMatrixIntrinsicId($layout, $num);
      $res = createIntrinsicCall(builder, intId, operands, {operands[0]->getType()});
  }];

  string baseDescription = [{
    The `nvvm.ldmatrix` operation collectively loads one or more matrices across
    all threads in a warp from the location indicated by the address operand
    `ptr` from shared memory.

    The attribute `num` indicates how many 8x8 16-bit matrices are to be loaded.

    All the threads in the warp must execute the same ldmatrix operations.

    Each row of 8 elements needs to be consecutive in memory. Each lane of the
    warp contains the start address of a row of 8 elements laid out as below:

    ```
    num | lane 0--7    | Threads 8--15  | Threads 16--31
    1   | addr0--addr7 |                |
    2   | addr0--addr7 | addr8--addr15  |
    4   | addr0--addr7 | addr8--addr15  | addr16--addr31
    ```

    Example:
    ```mlir
    %l1 = nvvm.ldmatrix %ptr {num = 1 : i32, layout = #nvvm.mma_layout<row>} :
      (!llvm.ptr<i32, 3>) -> i32
    %l2 = nvvm.ldmatrix %ptr {num = 4 : i32, layout = #nvvm.mma_layout<row>} :
      (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
    ```
  }];

  let assemblyFormat = "$ptr attr-dict `:` functional-type($ptr, $res)";
  let hasVerifier = 1;
}

def NVVM_MmaOp : NVVM_Op<"mma.sync", [AttrSizedOperandSegments]> {

  let summary = "cooperative matrix-multiply and accumulate";

  let description = [{
    The `nvvm.mma.sync` operation collectively performs the operation
    `D = matmul(A, B) + C` using all threads in a warp.

    All the threads in the warp must execute the same `mma.sync` operation.

    For each possible multiplicand PTX data type, there are one or more possible
    instruction shapes given as "mMnNkK". The below table describes the posssibilities
    as well as the types required for the operands. Note that the data type for
    C (the accumulator) and D (the result) can vary independently when there are
    multiple possibilities in the "C/D Type" column.

    When an optional attribute cannot be immediately inferred from the types of
    the operands and the result during parsing or validation, an error will be
    raised.

    `b1Op` is only relevant when the binary (b1) type is given to
    `multiplicandDataType`. It specifies how the multiply-and-acumulate is
    performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

    `intOverflowBehavior` is only relevant when the `multiplicandType` attribute
    is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
    in the accumulator. When the attribute is `satfinite`, the accumulator values
    are clamped in the int32 range on overflow. This is the default behavior.
    Alternatively, accumulator behavior `wrapped` can also be specified, in
    which case overflow wraps from one end of the range to the other.

    `layoutA` and `layoutB` are required and should generally be set to
    `#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
    combinations are possible for certain layouts according to the table below.

    ```
    | A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
    |----------|-----------|---------|---------|----------|----------|-------------------|
    | f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
    | f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
    |          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
    |          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
    | bf16     | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
    |          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
    | tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
    |          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
    | u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    |          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
    | u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    |          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
    | b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    ```


    Example:
    ```mlir

    %128 = nvvm.mma.sync A[%120, %121, %122, %123]
                         B[%124, %125]
                         C[%126, %127]
                         {layoutA = #nvvm.mma_layout<row>,
                          layoutB = #nvvm.mma_layout<col>,
                          shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
           -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    ```
  }];

  let results = (outs LLVM_AnyStruct:$res);
  let arguments = (ins NVVM_MMAShapeAttr:$shape,
             OptionalAttr<MMAB1OpAttr>:$b1Op,
             OptionalAttr<MMAIntOverflowAttr>:$intOverflowBehavior,
             MMALayoutAttr:$layoutA,
             MMALayoutAttr:$layoutB,
             OptionalAttr<MMATypesAttr>:$multiplicandAPtxType,
             OptionalAttr<MMATypesAttr>:$multiplicandBPtxType,
             Variadic<LLVM_Type>:$operandA,
             Variadic<LLVM_Type>:$operandB,
             Variadic<LLVM_Type>:$operandC);

  let extraClassDeclaration = !strconcat([{
      static llvm::Intrinsic::ID getIntrinsicID(
            int64_t m, int64_t n, uint64_t k,
            std::optional<MMAB1Op> b1Op,
            std::optional<MMAIntOverflow> sat,
            mlir::NVVM::MMALayout layoutAEnum, mlir::NVVM::MMALayout layoutBEnum,
            mlir::NVVM::MMATypes eltypeAEnum, mlir::NVVM::MMATypes eltypeBEnum,
            mlir::NVVM::MMATypes eltypeCEnum, mlir::NVVM::MMATypes eltypeDEnum) {
        llvm::StringRef layoutA = stringifyEnum(layoutAEnum);
        llvm::StringRef layoutB = stringifyEnum(layoutBEnum);
        llvm::StringRef eltypeA = stringifyEnum(eltypeAEnum);
        llvm::StringRef eltypeB = stringifyEnum(eltypeBEnum);
        llvm::StringRef eltypeC = stringifyEnum(eltypeCEnum);
        llvm::StringRef eltypeD = stringifyEnum(eltypeDEnum);
        }],
        MMA_SYNC_INTR<>.id, [{
        return 0;
      }

      static std::optional<mlir::NVVM::MMATypes> inferOperandMMAType(Type operandElType,
        bool isAccumulator);

      MMATypes accumPtxType();
      MMATypes resultPtxType();
    }]);

  let builders = [
      OpBuilder<(ins  "Type":$resultType, "ValueRange":$operandA,
        "ValueRange":$operandB, "ValueRange":$operandC,
        "ArrayRef<int64_t>":$shape, "std::optional<MMAB1Op>":$b1Op,
        "std::optional<MMAIntOverflow>":$intOverflow,
        "std::optional<std::array<MMATypes, 2>>":$multiplicandPtxTypes,
        "std::optional<std::array<MMALayout, 2>>":$multiplicandLayouts)>
    ];

  string llvmBuilder = [{
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    auto intId = mlir::NVVM::MmaOp::getIntrinsicID(
        $shape.getM(), $shape.getN(), $shape.getK(),
        $b1Op, $intOverflowBehavior,
        $layoutA, $layoutB,
        *$multiplicandAPtxType,
        *$multiplicandBPtxType,
        op.accumPtxType(),
        op.resultPtxType());

    $res = createIntrinsicCall(
      builder, intId, operands);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NVVM TMA Ops
//===----------------------------------------------------------------------===//

def NVVM_CpAsyncBulkTensorGlobalToSharedClusterOp : 
  NVVM_Op<"cp.async.bulk.tensor.shared.cluster.global", 
  [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>, 
  AttrSizedOperandSegments]>,
  Arguments<(ins  LLVM_i64ptr_shared:$dstMem,
                  LLVM_i64ptr_any:$tmaDescriptor,
                  LLVM_i64ptr_shared:$mbar,
                  Variadic<I32>:$coordinates,
                  PtxPredicate:$predicate)> {
  let assemblyFormat = [{ 
    $dstMem `,` 
    $tmaDescriptor `,` 
    $mbar `,` 
    `box` `[`$coordinates `]` 
    (`,` `predicate` `=` $predicate^)? 
    attr-dict  `:` type(operands)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      int dim = getCoordinates().size();
      std::string ptx = "cp.async.bulk.tensor.";
      ptx += std::to_string(dim) + "d.";
      ptx += "shared::cluster.global.mbarrier::complete_tx::bytes";
      if(dim == 1) ptx += " [%0], [%1, {%3} ], [%2];";
      if(dim == 2) ptx += " [%0], [%1, {%3, %4} ], [%2];";
      if(dim == 3) ptx += " [%0], [%1, {%3, %4, %5} ], [%2];";
      if(dim == 4) ptx += " [%0], [%1, {%3, %4, %5, %6} ], [%2];";
      if(dim == 5) ptx += " [%0], [%1, {%3, %4, %5, %6, %7} ], [%2];";
      return ptx;
    }
  }];
  let hasVerifier = 1;
}

def NVVM_CpAsyncBulkTensorSharedCTAToGlobalOp : 
  NVVM_Op<"cp.async.bulk.tensor.global.shared.cta", 
  [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>, 
  AttrSizedOperandSegments]>,
  Arguments<(ins  LLVM_i64ptr_any:$tmaDescriptor,
                  LLVM_i64ptr_shared:$srcMem,
                  Variadic<I32>:$coordinates,
                  PtxPredicate:$predicate)> {
  let assemblyFormat = [{ 
    $tmaDescriptor `,` 
    $srcMem `,` 
    `box` `[`$coordinates `]` 
    (`,` `predicate` `=` $predicate^)?  
    attr-dict  `:` type(operands)
  }];
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      int dim = getCoordinates().size();
      std::string ptx = "cp.async.bulk.tensor.";
      ptx += std::to_string(dim) + "d.";
      ptx += "global.shared::cta.bulk_group";
      if(dim == 1) ptx += " [%0, {%2} ], [%1];";
      if(dim == 2) ptx += " [%0, {%2, %3} ], [%1];";
      if(dim == 3) ptx += " [%0, {%2, %3, %4} ], [%1];";
      if(dim == 4) ptx += " [%0, {%2, %3, %4, %5} ], [%1];";
      if(dim == 5) ptx += " [%0, {%2, %3, %4, %5, %6} ], [%1];";
      return ptx;
    }
  }];
  let hasVerifier = 1;
}

def NVVM_PrefetchTensorMapOp : NVVM_Op<"prefetch.tensormap",
                    [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>]>,
  Arguments<(ins LLVM_i64ptr_any:$tmaDescriptor, PtxPredicate:$predicate)> {
  let assemblyFormat = "$tmaDescriptor (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { 
      return std::string("prefetch.tensormap [%0];");
    }
  }];
}

//===----------------------------------------------------------------------===//
// NVVM Wgmma Ops
//===----------------------------------------------------------------------===//

def NVVM_WgmmaFenceAlignedOp : NVVM_PTXBuilder_Op<"wgmma.fence.aligned"> {
  let arguments = (ins);
  let description = [{
    Enforce an ordering of register accesses between warpgroup level matrix 
    multiplication and other operations. 
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
  }];
  let assemblyFormat = "attr-dict";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("wgmma.fence.sync.aligned;"); }
  }];
}

def NVVM_WgmmaGroupSyncAlignedOp : NVVM_PTXBuilder_Op<"wgmma.commit.group.sync.aligned">,
  Arguments<(ins )> {
  let assemblyFormat = "attr-dict";
  let description = [{
    Commits all prior uncommitted warpgroup level matrix multiplication operations.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)
  }];
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("wgmma.commit_group.sync.aligned;"); }
  }];
}

def NVVM_WgmmaWaitGroupSyncOp : NVVM_PTXBuilder_Op<"wgmma.wait.group.sync.aligned">{
  let arguments = (ins I32Attr:$group);
  let assemblyFormat = "attr-dict $group";
  let description = [{
    Signal the completion of a preceding warpgroup operation.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
  }];
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("wgmma.wait_group.sync.aligned %0;"); }
  }];
}

/// Enum attribute type for the negating of input operands
def WGMMAScaleInNeg : I32EnumAttrCase<"neg", -1>;
def WGMMAScaleInOne : I32EnumAttrCase<"one", 1>;
def WGMMAScaleIn : I32EnumAttr<"WGMMAScaleIn", "WGMMA overflow options",
  [WGMMAScaleInOne, WGMMAScaleInNeg]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def WGMMAScaleInAttr : EnumAttr<NVVM_Dialect, WGMMAScaleIn, "wgmma_scale_in"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Enum attribute type for the output operand
def WGMMAScaleOutZero : I32EnumAttrCase<"zero", 0>;
def WGMMAScaleOutOne : I32EnumAttrCase<"one", 1>;
def WGMMAScaleOut : I32EnumAttr<"WGMMAScaleOut", "WGMMA input predicate",
  [WGMMAScaleOutZero, WGMMAScaleOutOne]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def WGMMAScaleOutAttr : EnumAttr<NVVM_Dialect, WGMMAScaleOut, "wgmma_scale_out"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Enum attribute of the different PTX element types used for WGMMA operands.
def WGMMATypeF16  : I32EnumAttrCase<"f16", 0>;
def WGMMATypeTF32 : I32EnumAttrCase<"tf32", 1>;
def WGMMATypeU8 : I32EnumAttrCase<"u8", 2>;
def WGMMATypeS8 : I32EnumAttrCase<"s8", 3>;
def WGMMATypeB1 : I32EnumAttrCase<"b1", 4>;
def WGMMATypeBF16 : I32EnumAttrCase<"bf16", 5>;
def WGMMATypeF8E4M3 : I32EnumAttrCase<"e4m3", 6>;
def WGMMATypeF8E5M2 : I32EnumAttrCase<"e5m2", 7>;
def WGMMATypes : I32EnumAttr<"WGMMATypes", "NVVM WGMMA types",
  [WGMMATypeF16, WGMMATypeTF32,
    WGMMATypeU8, WGMMATypeS8,
    WGMMATypeB1, WGMMATypeBF16, WGMMATypeF8E4M3, 
    WGMMATypeF8E5M2]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def WGMMATypesAttr : EnumAttr<NVVM_Dialect, WGMMATypes, "wgmma_type"> {
  let assemblyFormat = "`<` $value `>`";
}


def NVVM_WgmmaMmaAsyncOp : NVVM_Op<"wgmma.mma_async", 
              [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>,
                PredOpTrait<"input struct and result struct must be the same type",
                  TCresIsSameAsOpBase<0, 0>>,]> 
{
  let results = (outs LLVM_AnyStruct:$results);
  let arguments = (ins 
    LLVM_AnyStruct:$inouts,
    I64:$descriptorA, 
    I64:$descriptorB, 
    NVVM_MMAShapeAttr:$shape,
    WGMMATypesAttr:$typeA,
    WGMMATypesAttr:$typeB,
    WGMMAScaleOutAttr:$scaleD,
    WGMMAScaleInAttr:$scaleA,
    WGMMAScaleInAttr:$scaleB, 
    MMALayoutAttr:$layoutA,
    MMALayoutAttr:$layoutB,
    OptionalAttr<MMAIntOverflowAttr>:$satfinite
  );  
  
   let assemblyFormat = [{ 
      $descriptorA `,` $descriptorB `,` $shape `,` 
      `D` `[` $inouts `,` $scaleD (`,` $satfinite^)? `]` `,`
      `A` `[` $typeA `,` $scaleA `,` $layoutA `]` `,` 
      `B` `[` $typeB `,` $scaleB `,` $layoutB `]`
      attr-dict `:` 
      type($inouts) `->` type($results)
    }];
  
  let description = [{
    The warpgroup (128 threads) level matrix multiply and accumulate operation 
    has either of the following forms, where matrix D is called accumulator:
      D = A * B + D
      D = A * B, where the input from accumulator D is disabled.

    Supported shapes:  
    ```
    |--------------|--------------|------------|--------------|---------------|
    |              |              |            |              |f16+=e4m3*e4m3 |
    |              |              |            |              |f16+=e5m2*e5m2 |
    |f32+=tf32*tf32|f16+=f16 *f16 | s32+=s8*s8 |s32 += b1 * b1|f16+=e5m2*e4m3 |
    |              |f32+=f16 *f16 | s32+=u8*u8 |              |f16+=e4m3*e5m2 |
    |              |f32+=bf16*bf16| s32+=u8*u8 |              |f16+=e4m3*e5m2 |
    |              |f32+=bf16*bf16| s32+=s8*u8 |              |f32+=e4m3*e4m3 |
    |              |              | s32+=u8*s8 |              |f32+=e5m2*e5m2 |
    |              |              |            |              |f32+=e4m3*e5m2 |
    |              |              |            |              |f32+=e4m3*e5m2 |
    |--------------|--------------|------------|--------------|---------------|
    |   .m64n8k8   |  .m64n8k16   | .m64n8k32  | .m64n8k256   | .m64n8k32     |
    |   .m64n16k8  |  .m64n16k16  | .m64n16k32 | .m64n16k256  | .m64n16k32    |
    |   .m64n24k8  |  .m64n24k16  | .m64n24k32 | .m64n24k256  | .m64n24k32    |
    |   .m64n32k8  |  .m64n32k16  | .m64n32k32 | .m64n32k256  | .m64n32k32    |
    |   .m64n40k8  |  .m64n40k16  | .m64n48k32 | .m64n48k256  | .m64n40k32    |
    |   .m64n48k8  |  .m64n48k16  | .m64n64k32 | .m64n64k256  | .m64n48k32    |
    |   .m64n56k8  |  .m64n56k16  | .m64n80k32 | .m64n80k256  | .m64n56k32    |
    |   .m64n64k8  |  .m64n64k16  | .m64n96k32 | .m64n96k256  | .m64n64k32    |
    |   .m64n72k8  |  .m64n72k16  | .m64n112k32| .m64n112k256 | .m64n72k32    |
    |   .m64n80k8  |  .m64n80k16  | .m64n128k32| .m64n128k256 | .m64n80k32    |
    |   .m64n88k8  |  .m64n88k16  | .m64n144k32| .m64n144k256 | .m64n88k32    |
    |   .m64n96k8  |  .m64n96k16  | .m64n160k32| .m64n160k256 | .m64n96k32    |
    |   .m64n104k8 |  .m64n104k16 | .m64n176k32| .m64n176k256 | .m64n104k32   |
    |   .m64n112k8 |  .m64n112k16 | .m64n192k32| .m64n192k256 | .m64n112k32   |
    |   .m64n120k8 |  .m64n120k16 | .m64n208k32| .m64n208k256 | .m64n120k32   |
    |   .m64n128k8 |  .m64n128k16 | .m64n224k32| .m64n224k256 | .m64n128k32   |
    |   .m64n136k8 |  .m64n136k16 | .m64n240k32| .m64n240k256 | .m64n136k32   |
    |   .m64n144k8 |  .m64n144k16 | .m64n256k32| .m64n256k256 | .m64n144k32   |
    |   .m64n152k8 |  .m64n152k16 |            |              | .m64n152k32   |
    |   .m64n160k8 |  .m64n160k16 |            |              | .m64n160k32   |
    |   .m64n168k8 |  .m64n168k16 |            |              | .m64n168k32   |
    |   .m64n176k8 |  .m64n176k16 |            |              | .m64n176k32   |
    |   .m64n184k8 |  .m64n184k16 |            |              | .m64n184k32   |
    |   .m64n192k8 |  .m64n192k16 |            |              | .m64n192k32   |
    |   .m64n200k8 |  .m64n200k16 |            |              | .m64n200k32   |
    |   .m64n208k8 |  .m64n208k16 |            |              | .m64n208k32   |
    |   .m64n216k8 |  .m64n216k16 |            |              | .m64n216k32   |
    |   .m64n224k8 |  .m64n224k16 |            |              | .m64n224k32   |
    |   .m64n232k8 |  .m64n232k16 |            |              | .m64n232k32   |
    |   .m64n240k8 |  .m64n240k16 |            |              | .m64n240k32   |
    |   .m64n248k8 |  .m64n248k16 |            |              | .m64n248k32   |
    |   .m64n256k8 |  .m64n256k16 |            |              | .m64n256k32   |
    |--------------|--------------|------------|--------------|---------------|
    ```

    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)
  }];
  
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void getAsmValues(RewriterBase &rewriter, 
        llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>> &asmValues);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM target attribute.
//===----------------------------------------------------------------------===//

def NVVM_TargettAttr : NVVM_Attr<"NVVMTarget", "target"> {
  let description = [{
    GPU target attribute for controlling compilation of NVIDIA targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#nvvm.target] attributes {...} {
        ...
      }
    ```

    2. Target with `sm_90` chip and fast math.
    ```
      gpu.module @mymodule [#nvvm.target<chip = "sm_90", flags = {fast}>] {
        ...
      }
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"nvptx64-nvidia-cuda\"">:$triple,
    StringRefParameter<"Target chip.", "\"sm_50\"">:$chip,
    StringRefParameter<"Target chip features.", "\"+ptx60\"">:$features,
    OptionalParameter<"DictionaryAttr", "Target specific flags.">:$flags,
    OptionalParameter<"ArrayAttr", "Files to link to the LLVM module.">:$link
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features, $flags, $link)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"nvptx64-nvidia-cuda\"">:$triple,
                     CArg<"StringRef", "\"sm_50\"">:$chip,
                     CArg<"StringRef", "\"+ptx60\"">:$features,
                     CArg<"DictionaryAttr", "nullptr">:$targetFlags,
                     CArg<"ArrayAttr", "nullptr">:$linkFiles), [{
      return Base::get($_ctxt, optLevel, triple, chip, features, targetFlags, linkFiles);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool hasFlag(StringRef flag) const;
    bool hasFastMath() const;
    bool hasFtz() const;
  }];
  let extraClassDefinition = [{
    bool $cppClass::hasFlag(StringRef flag) const {
      if (DictionaryAttr flags = getFlags())
        return flags.get(flag) != nullptr;
      return false;
    }
    bool $cppClass::hasFastMath() const {
      return hasFlag("fast");
    }
    bool $cppClass::hasFtz() const {
      return hasFlag("ftz");
    }
  }];
}

#endif // NVVMIR_OPS
