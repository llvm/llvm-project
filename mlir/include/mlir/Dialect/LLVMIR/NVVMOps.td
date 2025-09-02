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
include "mlir/Dialect/LLVMIR/NVVMRequiresSMTraits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

def LLVM_PointerGeneric : LLVM_PointerInAddressSpace<0>;
def LLVM_PointerGlobal : LLVM_PointerInAddressSpace<1>;
def LLVM_PointerShared : LLVM_PointerInAddressSpace<3>;
def LLVM_PointerConst : LLVM_PointerInAddressSpace<4>;
def LLVM_PointerLocal : LLVM_PointerInAddressSpace<5>;
def LLVM_PointerTensor : LLVM_PointerInAddressSpace<6>;
def LLVM_PointerSharedCluster : LLVM_PointerInAddressSpace<7>;

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

    /// Get the name of the attribute used to annotate exact CTAs required
    /// per cluster for kernel functions.
    static StringRef getClusterDimAttrName() { return "nvvm.cluster_dim"; }
    /// Get the name of the metadata names for each dimension
    static StringRef getClusterDimXName() { return "cluster_dim_x"; }
    static StringRef getClusterDimYName() { return "cluster_dim_y"; }
    static StringRef getClusterDimZName() { return "cluster_dim_z"; }

    /// Get the name of the attribute used to annotate maximum number of
    /// CTAs per cluster for kernel functions.
    static StringRef getClusterMaxBlocksAttrName() {  return "nvvm.cluster_max_blocks"; }

    /// Get the name of the attribute used to annotate min CTA required
    /// per SM for kernel functions.
    static StringRef getMinctasmAttrName() { return "nvvm.minctasm"; }

    /// Get the name of the attribute used to annotate max number of
    /// registers that can be allocated per thread.
    static StringRef getMaxnregAttrName() { return "nvvm.maxnreg"; }

    /// Get the name of the attribute used to annotate kernel arguments that
    /// are grid constants.
    static StringRef getGridConstantAttrName() { return "nvvm.grid_constant"; }

    /// Get the name of the attribute used to annotate the `.blocksareclusters`
    /// PTX directive for kernel functions.
    /// This attribute implies that the grid launch configuration for the
    /// corresponding kernel function is specifying the number of clusters
    /// instead of the number of thread blocks. This attribute is only
    /// allowed for kernel functions and requires nvvm.reqntid and
    /// nvvm.cluster_dim attributes.
    static StringRef getBlocksAreClustersAttrName() { return "nvvm.blocksareclusters"; }

    /// Verify an attribute from this dialect on the argument at 'argIndex' for
    /// the region at 'regionIndex' on the given operation. Returns failure if
    /// the verification failed, success otherwise. This hook may optionally be
    /// invoked from any operation containing a region.
    LogicalResult verifyRegionArgAttribute(Operation *op,
                                           unsigned regionIndex,
                                           unsigned argIndex,
                                           NamedAttribute argAttr) override;
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

// Cache Eviction Priority enum definitions
def EvictNormal : I32EnumCase<"EvictNormal", 0, "evict_normal">;
def EvictFirst : I32EnumCase<"EvictFirst", 1, "evict_first">;
def EvictLast : I32EnumCase<"EvictLast", 2, "evict_last">;
def EvictUnchanged : I32EnumCase<"EvictUnchanged", 3, "evict_unchanged">;
def NoAllocate : I32EnumCase<"NoAllocate", 4, "no_allocate">;

def CacheEvictionPriority : I32Enum<"CacheEvictionPriority",
                                    "NVVM Cache Eviction Priority",
                                    [EvictNormal, EvictFirst, EvictLast,   
                                     EvictUnchanged, NoAllocate]> {
  let cppNamespace = "::mlir::NVVM";
}

def CacheEvictionPriorityAttr : EnumAttr<NVVM_Dialect, CacheEvictionPriority, 
                                         "cache_eviction_priority"> {
  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// NVVM intrinsic operations
//===----------------------------------------------------------------------===//

class NVVM_IntrOp<string mnem, list<Trait> traits = [],
                  int numResults = 0>
  : LLVM_IntrOpBase<NVVM_Dialect, mnem, "nvvm_" # !subst(".", "_", mnem),
                    /*list<int> overloadedResults=*/[],
                    /*list<int> overloadedOperands=*/[],
                    traits, numResults>;

//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_PureSpecialRegisterOp<string mnemonic, list<Trait> traits = []> :
  NVVM_IntrOp<mnemonic, !listconcat(traits, [Pure]), 1> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($res)";
}

class NVVM_SpecialRegisterOp<string mnemonic, list<Trait> traits = []> :
  NVVM_IntrOp<mnemonic, traits, 1> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($res)";
}

class NVVM_PureSpecialRangeableRegisterOp<string mnemonic, list<Trait> traits = []> :
  NVVM_PureSpecialRegisterOp<mnemonic,
    !listconcat(traits,
      [DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>])> {
  let arguments = (ins OptionalAttr<LLVM_ConstantRangeAttr>:$range);
  let assemblyFormat = "(`range` $range^)? attr-dict `:` type($res)";
  let llvmBuilder = baseLlvmBuilder # setRangeRetAttrCode # baseLlvmBuilderCoda;
  let mlirBuilder = baseMlirBuilder # importRangeRetAttrCode # baseMlirBuilderCoda;

  // Backwards-compatibility builder for an unspecified range.
  let builders = [
    OpBuilder<(ins "Type":$resultType), [{
      build($_builder, $_state, resultType, ::mlir::LLVM::ConstantRangeAttr{});
    }]>
  ];

  // Define this method for the InferIntRangeInterface.
  let extraClassDefinition = [{
    // Infer the result ranges based on the range attribute.
    void $cppClass::inferResultRanges(
        ArrayRef<::mlir::ConstantIntRanges> argRanges,
        SetIntRangeFn setResultRanges) {
        nvvmInferResultRanges(getOperation(), getResult(), argRanges, setResultRanges);
    }
  }];

}

//===----------------------------------------------------------------------===//
// Lane, Warp, SM, Grid index and range
def NVVM_LaneIdOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.laneid">;
def NVVM_WarpSizeOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.warpsize">;
def NVVM_WarpIdOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.warpid">;
def NVVM_WarpDimOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nwarpid">;
def NVVM_SmIdOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.smid">;
def NVVM_SmDimOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nsmid">;
def NVVM_GridIdOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.gridid">;

//===----------------------------------------------------------------------===//
// Lane Mask Comparison Ops
def NVVM_LaneMaskEqOp : NVVM_PureSpecialRegisterOp<"read.ptx.sreg.lanemask.eq">;
def NVVM_LaneMaskLeOp : NVVM_PureSpecialRegisterOp<"read.ptx.sreg.lanemask.le">;
def NVVM_LaneMaskLtOp : NVVM_PureSpecialRegisterOp<"read.ptx.sreg.lanemask.lt">;
def NVVM_LaneMaskGeOp : NVVM_PureSpecialRegisterOp<"read.ptx.sreg.lanemask.ge">;
def NVVM_LaneMaskGtOp : NVVM_PureSpecialRegisterOp<"read.ptx.sreg.lanemask.gt">;

//===----------------------------------------------------------------------===//
// Thread index and range
def NVVM_ThreadIdXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_BlockDimXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_BlockDimYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_BlockDimZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.ntid.z">;

//===----------------------------------------------------------------------===//
// Block index and range
def NVVM_BlockIdXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_GridDimXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_GridDimYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_GridDimZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nctaid.z">;

//===----------------------------------------------------------------------===//
// CTA Cluster index and range
def NVVM_ClusterIdXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.clusterid.x", [NVVMRequiresSM<90>]>;
def NVVM_ClusterIdYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.clusterid.y">;
def NVVM_ClusterIdZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.clusterid.z">;
def NVVM_ClusterDimXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nclusterid.x">;
def NVVM_ClusterDimYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nclusterid.y">;
def NVVM_ClusterDimZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.nclusterid.z">;


//===----------------------------------------------------------------------===//
// CTA index and range within Cluster
def NVVM_BlockInClusterIdXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.ctaid.x", [NVVMRequiresSM<90>]>;
def NVVM_BlockInClusterIdYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.ctaid.y", [NVVMRequiresSM<90>]>;
def NVVM_BlockInClusterIdZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.ctaid.z", [NVVMRequiresSM<90>]>;
def NVVM_ClusterDimBlocksXOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.nctaid.x", [NVVMRequiresSM<90>]>;
def NVVM_ClusterDimBlocksYOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.nctaid.y", [NVVMRequiresSM<90>]>;
def NVVM_ClusterDimBlocksZOp : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.nctaid.z">;

//===----------------------------------------------------------------------===//
// CTA index and across Cluster dimensions
def NVVM_ClusterId : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.ctarank", [NVVMRequiresSM<90>]>;
def NVVM_ClusterDim : NVVM_PureSpecialRangeableRegisterOp<"read.ptx.sreg.cluster.nctarank">;

//===----------------------------------------------------------------------===//
// Clock registers
def NVVM_ClockOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.clock">;
def NVVM_Clock64Op : NVVM_SpecialRegisterOp<"read.ptx.sreg.clock64">;
def NVVM_GlobalTimerOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.globaltimer">;
def NVVM_GlobalTimerLoOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.globaltimer.lo">;

//===----------------------------------------------------------------------===//
// envreg registers
foreach index = !range(0, 32) in {
  def NVVM_EnvReg # index # Op : NVVM_PureSpecialRegisterOp<"read.ptx.sreg.envreg" # index>;
}

//===----------------------------------------------------------------------===//
// Inline PTX op definition
//===----------------------------------------------------------------------===//

def NVVM_InlinePtxOp : NVVM_Op<"inline_ptx", 
  [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>, 
    AttrSizedOperandSegments]>
{
  let summary = "Inline PTX Op";
  let description = [{This op allows using PTX directly within the NVVM 
    dialect, while greatly simplifying llvm.inline_asm generation. It 
    automatically handles register size selection and sets the correct 
    read/write access for each operand. The operation leverages the 
    `BasicPtxBuilderInterface` to abstract away low-level details of 
    PTX assembly formatting.

    The `predicate` attribute is used to specify a predicate for the 
    PTX instruction.

    Example 1: Read-only Parameters
    ```mlir
    nvvm.inline_ptx "mbarrier.init.b64 [$0], $1;" (%barrier_gen, %count) : !llvm.ptr, i32

    // Lowers to:
    llvm.inline_asm has_side_effects asm_dialect = att 
      "mbarrier.init.b64 [$0], $1;", "l,r" %arg0, %arg2 : (!llvm.ptr, i32) -> ()
    ```

    Example 2: Read-only and Write-only Parameters
    ```mlir
    %0 = nvvm.inline_ptx "ex2.approx.ftz.f32 $0, $1;" (%input) : f32 -> f32

    // Lowers to:
    %0 = llvm.inline_asm has_side_effects asm_dialect = att 
      "ex2.approx.ftz.f32 $0, $1;", "=f,f" %arg0 : (f32) -> f32
    ```

    Example 3: Predicate Usage
    ```mlir
    nvvm.inline_ptx "mbarrier.init.b64 [$0], $1;" (%barrier_gen, %count), 
      predicate = %pred : !llvm.ptr, i32, i1

    // Lowers to:
    llvm.inline_asm has_side_effects asm_dialect = att 
      "@$2 mbarrier.init.b64 [$0], $1;", "l,r,b" %arg0, %arg2, %arg3 
      : (!llvm.ptr, i32, i1) -> ()
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$readOnlyArgs, 
                       Variadic<AnyType>:$readWriteArgs, 
                       StrAttr:$ptxCode,
                       PtxPredicate:$predicate);
                      
  let results = (outs Variadic<AnyType>:$writeOnlyArgs);
  
  let assemblyFormat = [{
    $ptxCode
    ( `ro` `(` $readOnlyArgs^ `:` type($readOnlyArgs) `)` )?
    ( `rw` `(` $readWriteArgs^ `:` type($readWriteArgs) `)` )?
    (`,` `predicate` `=` $predicate^)? 
    attr-dict
    ( `->` type($writeOnlyArgs)^ )?
  }];
  
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      StringRef ptxInstStr = getPtxCode();
      return std::string(ptxInstStr.data());
    }
  }];

  let extraClassDeclaration = [{
    bool getAsmValues(RewriterBase &, llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>> &);
  }];
}

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
def ReduxKindFmin : I32EnumAttrCase<"FMIN", 9, "fmin">;
def ReduxKindFmax : I32EnumAttrCase<"FMAX", 10, "fmax">;

/// Enum attribute of the different kinds.
def ReduxKind : I32EnumAttr<"ReduxKind", "NVVM redux kind",
  [ReduxKindAdd, ReduxKindAnd, ReduxKindMax, ReduxKindMin, ReduxKindOr, 
   ReduxKindUmax, ReduxKindUmin, ReduxKindXor, ReduxKindFmin, ReduxKindFmax]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}

def ReduxKindAttr : EnumAttr<NVVM_Dialect, ReduxKind, "redux_kind">;

def NVVM_ReduxOp :
  NVVM_Op<"redux.sync", [NVVMRequiresSM<80>]>,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$val,
                 ReduxKindAttr:$kind,
                 I32:$mask_and_clamp,
                 DefaultValuedAttr<BoolAttr, "false">:$abs,
                 DefaultValuedAttr<BoolAttr, "false">:$nan)> {
  let summary = "Redux Sync Op";
  let description = [{
    `redux.sync` performs a reduction operation `kind` of the 32 bit source 
    register across all non-exited threads in the membermask.

    The `abs` and `nan` attributes can be used in the case of f32 input type, 
    where the `abs` attribute causes the absolute value of the input to be used 
    in the reduction operation, and the `nan` attribute causes the reduction 
    operation to return NaN if any of the inputs to participating threads are 
    NaN.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync)
  }];
  string llvmBuilder = [{
      auto intId = getReduxIntrinsicId($_resultType, $kind, $abs, $nan);
      $res = createIntrinsicCall(builder, intId, {$val, $mask_and_clamp});
  }];
  let assemblyFormat = [{
    $kind $val `,` $mask_and_clamp  attr-dict `:` type($val) `->` type($res)
   }];   
}

//===----------------------------------------------------------------------===//
// NVVM nanosleep
//===----------------------------------------------------------------------===//

def NVVM_NanosleepOp : NVVM_Op<"nanosleep">,
  Arguments<(ins 
  ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<1000000>]>:$duration)> 
{
  let summary = "Suspends the thread for a specified duration.";

  let description = [{
    The op suspends the thread for a sleep duration approximately close to the 
    delay `$duration`, specified in nanoseconds. 

    The sleep duration is approximated, but guaranteed to be in the 
    interval [0, 2*t]. The maximum sleep duration is 1 millisecond. 
    The implementation may reduce the sleep duration for individual threads 
    within a warp such that all sleeping threads in the warp wake up together.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep)
  }];
  
  string llvmBuilder = [{
      createIntrinsicCall(builder, 
                          llvm::Intrinsic::nvvm_nanosleep, 
                          {builder.getInt32($duration)});
  }];
  let assemblyFormat = "attr-dict $duration";
}

//===----------------------------------------------------------------------===//
// NVVM Performance Monitor events
//===----------------------------------------------------------------------===//

def NVVM_PMEventOp : NVVM_PTXBuilder_Op<"pmevent">,
  Arguments<(ins OptionalAttr<I16Attr>:$maskedEventId, 
                 OptionalAttr<I32Attr>:$eventId)> {
  let summary = "Trigger one or more Performance Monitor events.";

  let description = [{
    Triggers one or more of a fixed number of performance monitor events, with
    event index or mask specified by immediate operand.

    Without `mask` it triggers a single performance monitor event indexed by
    immediate operand a, in the range 0..15.

    With `mask` it triggers one or more of the performance monitor events. Each
    bit in the 16-bit immediate operand controls an event.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent)
  }];
  
  string llvmBuilder = [{
      llvm::Value *mId = builder.getInt16(* $maskedEventId);
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_pm_event_mask, {mId});
  }];

  let assemblyFormat = "attr-dict (`id` `=` $eventId^)? (`mask` `=` $maskedEventId^)?";

  let extraClassDeclaration = [{
    bool hasIntrinsic() { return !getEventId(); }
  }];
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("pmevent %0;"); }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NVVM Split arrive/wait barrier
//===----------------------------------------------------------------------===//

/// mbarrier.init instruction with generic pointer type
def NVVM_MBarrierInitOp : NVVM_PTXBuilder_Op<"mbarrier.init">,
  Arguments<(ins LLVM_AnyPointer:$addr, I32:$count, PtxPredicate:$predicate)> {
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
def NVVM_MBarrierInitSharedOp : NVVM_PTXBuilder_Op<"mbarrier.init.shared", [NVVMRequiresSM<80>, DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>]>,
  Arguments<(ins LLVM_PointerShared:$addr, I32:$count, PtxPredicate:$predicate)> {
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
  Arguments<(ins LLVM_AnyPointer:$addr)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_inval, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type(operands)";
}

def NVVM_MBarrierInvalSharedOp : NVVM_Op<"mbarrier.inval.shared">,
  Arguments<(ins LLVM_PointerShared:$addr)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_inval_shared, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type(operands)";
}

def NVVM_MBarrierArriveOp : NVVM_Op<"mbarrier.arrive">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_AnyPointer:$addr)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($res)";
}

def NVVM_MBarrierArriveSharedOp : NVVM_Op<"mbarrier.arrive.shared">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_PointerShared:$addr)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive_shared, {$addr});
  }];
  let assemblyFormat = "$addr attr-dict `:` qualified(type($addr)) `->` type($res)";
}

def NVVM_MBarrierArriveNocompleteOp : NVVM_Op<"mbarrier.arrive.nocomplete">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_AnyPointer:$addr, I32:$count)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive_noComplete, {$addr, $count});
  }];
  let assemblyFormat = "$addr `,` $count attr-dict `:` type(operands) `->` type($res)";
}

def NVVM_MBarrierArriveNocompleteSharedOp : NVVM_Op<"mbarrier.arrive.nocomplete.shared">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_PointerShared:$addr, I32:$count)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_arrive_noComplete_shared, {$addr, $count});
  }];
  let assemblyFormat = "$addr `,` $count attr-dict `:` type(operands) `->` type($res)";
}

def NVVM_MBarrierArriveExpectTxOp : NVVM_PTXBuilder_Op<"mbarrier.arrive.expect_tx">,  
  Arguments<(ins LLVM_AnyPointer:$addr, I32:$txcount, PtxPredicate:$predicate)> {
  let assemblyFormat = "$addr `,` $txcount (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("mbarrier.arrive.expect_tx.b64 _, [%0], %1;"); }
  }];
}

def NVVM_MBarrierArriveExpectTxSharedOp : NVVM_PTXBuilder_Op<"mbarrier.arrive.expect_tx.shared">,  
  Arguments<(ins LLVM_PointerShared:$addr, I32:$txcount, PtxPredicate:$predicate)> {    
  let assemblyFormat = "$addr `,` $txcount (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() { return std::string("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"); }
  }];
}

def NVVM_MBarrierTryWaitParityOp : NVVM_PTXBuilder_Op<"mbarrier.try_wait.parity">,  
  Arguments<(ins LLVM_AnyPointer:$addr, I32:$phase, I32:$ticks)> {  
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
  Arguments<(ins LLVM_PointerShared:$addr, I32:$phase, I32:$ticks)> {  
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
  Arguments<(ins LLVM_AnyPointer:$addr, LLVM_Type:$state)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_test_wait, {$addr, $state});
  }];
  let assemblyFormat = "$addr `,` $state attr-dict `:` type(operands) `->` type($res)";
}

def NVVM_MBarrierTestWaitSharedOp : NVVM_Op<"mbarrier.test.wait.shared">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_PointerShared:$addr, LLVM_Type:$state)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder, llvm::Intrinsic::nvvm_mbarrier_test_wait_shared, {$addr, $state});
  }];
  let assemblyFormat = "$addr `,` $state attr-dict `:` type(operands) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  let assemblyFormat = "attr-dict";
  string llvmBuilder = [{
      createIntrinsicCall(
          builder, llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all,
          {builder.getInt32(0)});
  }];
}

def NVVM_BarrierOp : NVVM_Op<"barrier", [AttrSizedOperandSegments]> {
  let arguments = (ins     
    Optional<I32>:$barrierId,
    Optional<I32>:$numberOfThreads);
  string llvmBuilder = [{
    llvm::Value *id = $barrierId ? $barrierId : builder.getInt32(0);
    if ($numberOfThreads)
      createIntrinsicCall(
          builder, llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_count,
          {id, $numberOfThreads});
    else
      createIntrinsicCall(
          builder, llvm::Intrinsic::nvvm_barrier_cta_sync_aligned_all, {id});
  }];
  let hasVerifier = 1;

  let assemblyFormat = "(`id` `=` $barrierId^)? (`number_of_threads` `=` $numberOfThreads^)? attr-dict";

  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, Value{}, Value{});
    }]>,
    OpBuilder<(ins "Value":$barrierId), [{
      return build($_builder, $_state, barrierId, Value{});
    }]>
  ];
}

def NVVM_BarrierArriveOp : NVVM_PTXBuilder_Op<"barrier.arrive"> 
{
  let arguments = (ins Optional<I32>:$barrierId, I32:$numberOfThreads);

  let description = [{
    Thread that executes this op announces their arrival at the barrier with 
    given id and continue their execution.

    The default barrier id is 0 that is similar to `nvvm.barrier` Op. When 
    `barrierId` is not present, the default barrier id is used. 

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
  }];
  
  let assemblyFormat = "(`id` `=` $barrierId^)? `number_of_threads` `=` $numberOfThreads attr-dict";

  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      std::string ptx = "bar.arrive ";
      if (getBarrierId()) { ptx += "%0, %1;"; } 
      else { ptx += "0, %0;"; }
      return ptx;
    }
  }];
}

def NVVM_ClusterArriveOp : NVVM_Op<"cluster.arrive"> {
  let arguments = (ins OptionalAttr<UnitAttr>:$aligned);

  let summary = "Cluster Barrier Arrive Op";
  let description = [{
    The `cluster.arrive` can be used by the threads within the cluster for synchronization and
    communication. The `cluster.arrive` instruction marks the warps' arrival at the barrier
    without causing the executing thread to wait for other participating threads.

    The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
  }];

  string llvmBuilder = [{
      if ($aligned)
        createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_arrive_aligned);
      else
        createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_arrive);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ClusterArriveRelaxedOp : NVVM_Op<"cluster.arrive.relaxed", [NVVMRequiresSM<90>]> {
  let arguments = (ins OptionalAttr<UnitAttr>:$aligned);

  let summary = "Cluster Barrier Relaxed Arrive Op";
  let description = [{
    The `cluster.arrive` can be used by the threads within the cluster for synchronization and
    communication. The `cluster.arrive` instruction marks the warps' arrival at the barrier
    without causing the executing thread to wait for other participating threads.

    The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.
    The .relaxed qualifier on `cluster.arrive` specifies that there are no memory
    ordering and visibility guarantees provided for the memory accesses performed prior to
    `cluster.arrive`.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
  }];

  string llvmBuilder = [{
      if ($aligned)
        createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_arrive_relaxed_aligned);
      else
        createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_arrive_relaxed);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ClusterWaitOp : NVVM_Op<"cluster.wait", [NVVMRequiresSM<90>]> {
  let arguments = (ins OptionalAttr<UnitAttr>:$aligned);

  let summary = "Cluster Barrier Wait Op";
  let description = [{
    The `cluster.wait` causes the executing thread to wait for all non-exited threads
    of the cluster to perform `cluster.arrive`. The `aligned` attribute, when provided,
    generates the .aligned version of the PTX instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
  }];

  string llvmBuilder = [{
      if ($aligned)
        createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier_cluster_wait_aligned);
      else
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

def SharedSpaceCTA : I32EnumAttrCase<"shared_cta", 0, "cta">;
def SharedSpaceCluster   : I32EnumAttrCase<"shared_cluster", 1, "cluster">;
def SharedSpace : I32EnumAttr<"SharedSpace", "Shared memory space",
  [SharedSpaceCTA, SharedSpaceCluster]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SharedSpaceAttr : EnumAttr<NVVM_Dialect, SharedSpace, "shared_space"> {
  let assemblyFormat = "`<` $value `>`";
}

def ProxyAlias : I32EnumAttrCase<"alias", 0, "alias">;
def ProxyAsync   : I32EnumAttrCase<"async", 1, "async">;
def ProxyAsyncGlobal   : I32EnumAttrCase<"async_global", 2, "async.global">;
def ProxyAsyncShared   : I32EnumAttrCase<"async_shared", 3, "async.shared">;
def ProxyTensorMap : I32EnumAttrCase<"TENSORMAP", 4, "tensormap">;
def ProxyGeneric : I32EnumAttrCase<"GENERIC", 5, "generic">;
def ProxyKind : I32EnumAttr<"ProxyKind", "Proxy kind",
  [ProxyAlias, ProxyAsync, ProxyAsyncGlobal, ProxyAsyncShared, ProxyTensorMap, ProxyGeneric]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}

def ProxyKindAttr : EnumAttr<NVVM_Dialect, ProxyKind, "proxy_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_FenceProxyOp : NVVM_PTXBuilder_Op<"fence.proxy">,
  Arguments<(ins ProxyKindAttr:$kind,
                 OptionalAttr<SharedSpaceAttr>:$space)> {
  let description = [{
    Fence operation with proxy to establish an ordering between memory accesses
    that may happen through different proxies.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  
  let assemblyFormat = "attr-dict";
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      std::string ptx = "fence.proxy.";
      ptx += stringifyProxyKind(getKind());
      if(getKind() == NVVM::ProxyKind::async_shared)
        { ptx += "::"; ptx += stringifySharedSpace(getSpace().value()); }
      ptx += ";";
      return ptx;
    }
  }];
  let hasVerifier = 1;
}

// Attrs describing the scope of the Memory Operation
def MemScopeKindCTA      : I32EnumAttrCase<"CTA", 0, "cta">;
def MemScopeKindCluster  : I32EnumAttrCase<"CLUSTER", 1, "cluster">;
def MemScopeKindGPU      : I32EnumAttrCase<"GPU", 2, "gpu">;
def MemScopeKindSYS      : I32EnumAttrCase<"SYS", 3, "sys">;

def MemScopeKind : I32EnumAttr<"MemScopeKind", "NVVM Memory Scope kind",
  [MemScopeKindCTA, MemScopeKindCluster, MemScopeKindGPU, MemScopeKindSYS]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def MemScopeKindAttr : EnumAttr<NVVM_Dialect, MemScopeKind, "mem_scope"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_FenceProxyAcquireOp : NVVM_Op<"fence.proxy.acquire">,
      Arguments<(ins MemScopeKindAttr:$scope, LLVM_PointerGeneric:$addr, I32:$size,
                     DefaultValuedAttr<ProxyKindAttr,
                                       "ProxyKind::GENERIC">:$fromProxy,
                     DefaultValuedAttr<ProxyKindAttr,
                                       "ProxyKind::TENSORMAP">:$toProxy)> {
  let summary = "Uni-directional proxy fence operation with acquire semantics";
  let description = [{
    `fence.proxy.acquire` is a uni-directional fence used to establish ordering
    between a prior memory access performed via the generic proxy and a
    subsequent memory access performed via the tensormap proxy

    The address operand `addr` and the operand `size` together specify the
    memory range `[addr, addr+size)` on which the ordering guarantees on the
    memory accesses across the proxies is to be provided. The only supported
    value for the `size` operand is 128 and must be an immediate. Generic Addressing
    is used unconditionally, and the address specified by the operand `addr` must
    fall within the `.global` state space. Otherwise, the behavior is undefined
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];

  let assemblyFormat = "$scope $addr `,` $size (`from_proxy` `=` $fromProxy^)? (`to_proxy` `=` $toProxy^)? attr-dict";
  let llvmBuilder = [{
    createIntrinsicCall(
        builder,
        getUnidirectionalFenceProxyID($fromProxy, $toProxy, $scope, false),
        {$addr, $size});
  }];

  let hasVerifier = 1;
}

def NVVM_FenceProxyReleaseOp : NVVM_Op<"fence.proxy.release">,
      Arguments<(ins MemScopeKindAttr:$scope,
                     DefaultValuedAttr<ProxyKindAttr,
                                       "ProxyKind::GENERIC">:$fromProxy,
                     DefaultValuedAttr<ProxyKindAttr,
                                       "ProxyKind::TENSORMAP">:$toProxy)> {
  let summary = "Uni-directional proxy fence operation with release semantics";
  let description = [{
    `fence.proxy.release` is a uni-directional fence used to establish ordering
    between a prior memory access performed via the generic proxy and a
    subsequent memory access performed via the tensormap proxy. `fence.proxy.release`
    operation can form a release sequence that synchronizes with an acquire
    sequence that contains the fence.proxy.acquire proxy fence operation
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];

  let assemblyFormat = "$scope (`from_proxy` `=` $fromProxy^)? (`to_proxy` `=` $toProxy^)? attr-dict";
  let llvmBuilder = [{
    createIntrinsicCall(builder, getUnidirectionalFenceProxyID(
                                     $fromProxy, $toProxy, $scope, true));
  }];

  let hasVerifier = 1;
}

def SetMaxRegisterActionIncrease : I32EnumAttrCase<"increase", 0>;
def SetMaxRegisterActionDecrease   : I32EnumAttrCase<"decrease", 1>;
def SetMaxRegisterAction : I32EnumAttr<"SetMaxRegisterAction", "NVVM set max register action",
  [SetMaxRegisterActionDecrease, SetMaxRegisterActionIncrease]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SetMaxRegisterActionAttr : EnumAttr<NVVM_Dialect, SetMaxRegisterAction, "action">;

def NVVM_SetMaxRegisterOp : NVVM_Op<"setmaxregister"> {
  let arguments = (ins I32Attr:$regCount, SetMaxRegisterActionAttr:$action);
  let assemblyFormat = "$action $regCount attr-dict";
  let hasVerifier = 1;
  string llvmBuilder = [{
    auto intId = (op.getAction() == NVVM::SetMaxRegisterAction::increase) ?
      llvm::Intrinsic::nvvm_setmaxnreg_inc_sync_aligned_u32 :
      llvm::Intrinsic::nvvm_setmaxnreg_dec_sync_aligned_u32;

    createIntrinsicCall(builder, intId, builder.getInt32($regCount));
  }];
}

def NVVM_FenceMbarrierInitOp : NVVM_PTXBuilder_Op<"fence.mbarrier.init"> {
  let arguments = (ins );
    let description = [{
    Fence operation that applies on the prior nvvm.mbarrier.init
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  
  let assemblyFormat = "attr-dict";
  let extraClassDefinition = [{        
    std::string $cppClass::getPtx() {
      return std::string("fence.mbarrier_init.release.cluster;");
    }
  }];
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
  NVVM_Op<"shfl.sync", [NVVMRequiresSM<30>]>,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins I32:$thread_mask,
                 LLVM_Type:$val,
                 I32:$offset,
                 I32:$mask_and_clamp,
                 ShflKindAttr:$kind,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  let summary = "NVVM Dialect Op for shfl.sync";
  let description = [{
    The `shfl.sync` Op implements data shuffle within threads of a warp.
    The `thread_mask` denotes the threads participating in the Op where
    the bit position corresponds to a particular thread’s laneid.
    The `offset` specifies a source lane or source lane offset
    (depending on `kind`). The `val` is the input value to be copied from
    the source. The `mask_and_clamp` contains two packed values specifying
    a mask for logically splitting warps into sub-segments and an upper bound
    for clamping the source lane index.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync)
  }];
  string llvmBuilder = [{
      auto intId = getShflIntrinsicId(
          $_resultType, $kind, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$thread_mask, $val, $offset, $mask_and_clamp});
  }];
  let assemblyFormat = [{
    $kind $thread_mask `,` $val `,` $offset `,` $mask_and_clamp  attr-dict
     `:` type($val) `->` type($res)
   }];
   let hasVerifier = 1;
}

def VoteSyncKindAny : I32EnumAttrCase<"any", 0>;
def VoteSyncKindAll : I32EnumAttrCase<"all", 1>;
def VoteSyncKindBallot : I32EnumAttrCase<"ballot", 2>;
def VoteSyncKindUni : I32EnumAttrCase<"uni", 3>;

def VoteSyncKind : I32EnumAttr<"VoteSyncKind", "NVVM vote sync kind",
                               [VoteSyncKindAny, VoteSyncKindAll,
                                VoteSyncKindBallot, VoteSyncKindUni]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}

def VoteSyncKindAttr : EnumAttr<NVVM_Dialect, VoteSyncKind, "vote_sync_kind">;

def NVVM_VoteSyncOp
    : NVVM_Op<"vote.sync">,
      Results<(outs AnyTypeOf<[I32, I1]>:$res)>,
      Arguments<(ins I32:$mask, I1:$pred, VoteSyncKindAttr:$kind)> {
  let summary = "Vote across thread group";
  let description = [{
    The `vote.sync` op will cause executing thread to wait until all non-exited
    threads corresponding to membermask have executed `vote.sync` with the same
    qualifiers and same membermask value before resuming execution.

    The vote operation kinds are:
    - `any`: True if source predicate is True for some thread in membermask.
    - `all`: True if source predicate is True for all non-exited threads in
      membermask. 
    - `uni`: True if source predicate has the same value in all non-exited
      threads in membermask.
    - `ballot`: In the ballot form, the destination result is a 32 bit integer.
      In this form, the predicate from each thread in membermask are copied into
      the corresponding bit position of the result, where the bit position
      corresponds to the thread’s lane id.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync)
  }];
  string llvmBuilder = [{
    auto intId = getVoteSyncIntrinsicId($kind);
    $res = createIntrinsicCall(builder, intId, {$mask, $pred});
  }];
  let assemblyFormat = "$kind $mask `,` $pred attr-dict `->` type($res)";
  let hasVerifier = 1;
}

def NVVM_SyncWarpOp :
  NVVM_Op<"bar.warp.sync">,
  Arguments<(ins LLVM_Type:$mask)> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_bar_warp_sync, {$mask});
  }];
  let assemblyFormat = "$mask attr-dict `:` type($mask)";
}

def NVVM_ElectSyncOp : NVVM_Op<"elect.sync">
{  
  let summary = "Elect one leader thread";
  let description = [{
    The `elect.sync` instruction elects one predicated active leader
    thread from among a set of threads specified in the `membermask`.
    When the `membermask` is not provided explicitly, a default value
    of `0xFFFFFFFF` is used. The predicate result is set to `True` for
    the leader thread, and `False` for all other threads.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)
  }];

  let arguments = (ins Optional<I32>:$membermask);
  let results = (outs I1:$pred);
  let assemblyFormat = "($membermask^)? attr-dict `->` type(results)";
  string llvmBuilder = [{
    auto *resultTuple = createIntrinsicCall(builder,
        llvm::Intrinsic::nvvm_elect_sync,
        {$membermask ? $membermask : builder.getInt32(0xFFFFFFFF)});
    // Extract the second value into $pred
    $pred = builder.CreateExtractValue(resultTuple, 1);
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

def NVVM_CpAsyncOp : NVVM_Op<"cp.async.shared.global">,
  Arguments<(ins LLVM_PointerShared:$dst,
                 LLVM_PointerGlobal:$src,
                 I32Attr:$size,
                 LoadCacheModifierAttr:$modifier,
                 Optional<LLVM_Type>:$cpSize)> {
  let assemblyFormat = "$dst `,` $src `,` $size `,` `cache` `=` $modifier (`,` $cpSize^)? attr-dict `:` type(operands)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID
      getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                            llvm::SmallVector<llvm::Value *> &args);
  }];
  string llvmBuilder = [{
    llvm::SmallVector<llvm::Value *> translatedOperands;
    auto id = NVVM::CpAsyncOp::getIntrinsicIDAndArgs(
      *op, moduleTranslation, translatedOperands);
    createIntrinsicCall(builder, id, translatedOperands);
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

def NVVM_CpAsyncMBarrierArriveOp : NVVM_Op<"cp.async.mbarrier.arrive"> {
  let summary = "NVVM Dialect Op for cp.async.mbarrier.arrive";
  let description = [{
    The `cp.async.mbarrier.arrive` Op makes the mbarrier object track
    all prior cp.async operations initiated by the executing thread.
    The `addr` operand specifies the address of the mbarrier object
    in generic address space. The `noinc` attr impacts how the
    mbarrier's state is updated.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
  }];
  let assemblyFormat = "$addr attr-dict `:` type(operands)";

  let arguments = (ins
    LLVM_AnyPointer:$addr, DefaultValuedAttr<I1Attr, "0">:$noinc);

  string llvmBuilder = [{
    auto intId = $noinc ?
      llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc :
      llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive;

    createIntrinsicCall(builder, intId, {$addr});
  }];
}

def NVVM_CpAsyncMBarrierArriveSharedOp : NVVM_Op<"cp.async.mbarrier.arrive.shared"> {
  let summary = "NVVM Dialect Op for cp.async.mbarrier.arrive.shared";
  let description = [{
    The `cp.async.mbarrier.arrive.shared` Op makes the mbarrier object
    track all prior cp.async operations initiated by the executing thread.
    The `addr` operand specifies the address of the mbarrier object in
    shared memory. The `noinc` attr impacts how the mbarrier's state
    is updated. 
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
  }];
  let assemblyFormat = "$addr attr-dict `:` type(operands)";

  let arguments = (ins
    LLVM_PointerShared:$addr, DefaultValuedAttr<I1Attr, "0">:$noinc);

  string llvmBuilder = [{
    auto intId = $noinc ?
      llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc_shared :
      llvm::Intrinsic::nvvm_cp_async_mbarrier_arrive_shared;

    createIntrinsicCall(builder, intId, {$addr});
  }];
}

//===----------------------------------------------------------------------===//
// NVVM Conversion Ops (for "cvt.*" family of PTX instructions)
//===----------------------------------------------------------------------===//

// Attributes for the floating point rounding modes supported by PTX
def FPRoundingModeNone : I32EnumAttrCase<"NONE", 0, "none">;
def FPRoundingModeRN   : I32EnumAttrCase<"RN",   1, "rn">;
def FPRoundingModeRM   : I32EnumAttrCase<"RM",   2, "rm">;
def FPRoundingModeRP   : I32EnumAttrCase<"RP",   3, "rp">;
def FPRoundingModeRZ   : I32EnumAttrCase<"RZ",   4, "rz">;
def FPRoundingModeRNA  : I32EnumAttrCase<"RNA",  5, "rna">;

def FPRoundingMode : I32EnumAttr<"FPRoundingMode", "NVVM FPRoundingMode kind",
  [FPRoundingModeNone, FPRoundingModeRN, FPRoundingModeRM,
    FPRoundingModeRP, FPRoundingModeRZ, FPRoundingModeRNA]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def FPRoundingModeAttr : EnumAttr<NVVM_Dialect, FPRoundingMode, "fp_rnd_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def SaturationModeNone   : I32EnumAttrCase<"NONE", 0, "none">;
def SaturationModeFinite : I32EnumAttrCase<"SATFINITE", 1, "satfinite">;

def SaturationMode : I32EnumAttr<"SaturationMode", "NVVM SaturationMode kind",
  [SaturationModeNone, SaturationModeFinite]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SaturationModeAttr : EnumAttr<NVVM_Dialect, SaturationMode, "sat_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_ConvertFloatToTF32Op : NVVM_Op<"convert.float.to.tf32"> {
  let summary = "Convert the given float input to TF32";
  let description = [{
    This Op converts the given f32 input to tf32.
    The result `res` is represented as an i32 type.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction. The `rnd` and `sat` attributes specify the
    the rounding and saturation modes respectively.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];

  let hasVerifier = 1;
  let results = (outs I32:$res);
  let arguments = (ins
    F32:$src,
    DefaultValuedAttr<FPRoundingModeAttr, "FPRoundingMode::NONE">:$rnd,
    DefaultValuedAttr<SaturationModeAttr, "SaturationMode::NONE">:$sat,
    DefaultValuedAttr<BoolAttr, "false">:$relu);

  let assemblyFormat = "$src attr-dict";

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(NVVM::FPRoundingMode,
                                              NVVM::SaturationMode,
                                              bool hasRelu);
  }];

  string llvmBuilder = [{
    auto intId = NVVM::ConvertFloatToTF32Op::getIntrinsicID($rnd, $sat, $relu);
    $res = createIntrinsicCall(builder, intId, {$src});
  }];
}

def ConvertFP6E2M3 : I32EnumAttrCase<"E2M3", 0, "e2m3">;
def ConvertFP6E3M2 : I32EnumAttrCase<"E3M2", 1, "e3m2">;

def ConvertFP6Type : I32EnumAttr<"ConvertFP6Type", "NVVM ConvertFP6Type kind",
  [ConvertFP6E2M3, ConvertFP6E3M2]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ConvertFP6TypeAttr : EnumAttr<NVVM_Dialect, ConvertFP6Type, "convert_fp6_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_ConvertF32x2ToF6x2Op : NVVM_Op<"convert.f32x2.to.f6x2"> {
  let summary = "Convert a pair of float inputs to f6x2";
  let description = [{
    This Op converts each of the given float inputs to the specified fp6 type.
    The result `dst` is represented either as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such 
    that the value converted from `a` is stored in the upper 8 bits of `dst` 
    with 2 MSB bits padded with zeros and the value converted from `b` is 
    stored in the lower 8 bits of `dst` with 2 MSB bits padded with zeros.
    If `dst` is returned as a vector type, each converted value is stored as an 
    i8 element in the vector.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];

  let results = (outs AnyTypeOf<[I16, VectorOfLengthAndType<[2], [I8]>]>:$dst);
  let arguments = (ins 
    ConvertFP6TypeAttr:$type,
    F32:$a,
    F32:$b,
    DefaultValuedAttr<BoolAttr, "false">:$relu);
  let assemblyFormat = "$type $a `,` $b attr-dict `:` type($dst)";
  
  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(NVVM::ConvertFP6Type,
                                              bool hasRelu);
  }];

  string llvmBuilder = [{
    auto intId = NVVM::ConvertF32x2ToF6x2Op::getIntrinsicID($type, $relu);
    llvm::Value *packedI16 = createIntrinsicCall(builder, intId, {$a, $b});
    if(op.getDst().getType().isInteger(16))
      $dst = packedI16;
    else
      $dst = builder.CreateBitCast(packedI16,
                      llvm::FixedVectorType::get(llvm::Type::getInt8Ty(builder.getContext()), 2));
  }];
}

def ConvertFP8E4M3 : I32EnumAttrCase<"E4M3", 0, "e4m3">;
def ConvertFP8E5M2 : I32EnumAttrCase<"E5M2", 1, "e5m2">;
def ConvertFP8UE8M0 : I32EnumAttrCase<"UE8M0", 2, "ue8m0">;

def ConvertFP8Type : I32EnumAttr<"ConvertFP8Type", "NVVM ConvertFP8Type kind",
  [ConvertFP8E4M3, ConvertFP8E5M2, ConvertFP8UE8M0]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def ConvertFP8TypeAttr : EnumAttr<NVVM_Dialect, ConvertFP8Type, "convert_fp8_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_ConvertF32x2ToF8x2Op : NVVM_Op<"convert.f32x2.to.f8x2"> {
  let summary = "Convert a pair of float inputs to f8x2";
  let description = [{
    This Op converts each of the given float inputs to the specified fp8 type.
    The result `dst` is represented as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such 
    that the value converted from `a` is stored in the upper 8 bits of `dst` 
    and the value converted from `b` is stored in the lower 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an 
    i8 element in the vector.
    The `rnd` and `sat` attributes specify the rounding and saturation modes respectively.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];

  let hasVerifier = 1;
  let results = (outs AnyTypeOf<[I16, VectorOfLengthAndType<[2], [I8]>]>:$dst);
  let arguments = (ins
    ConvertFP8TypeAttr:$type,
    F32:$a,
    F32:$b,
    DefaultValuedAttr<FPRoundingModeAttr, "FPRoundingMode::NONE">:$rnd,
    DefaultValuedAttr<SaturationModeAttr, "SaturationMode::NONE">:$sat,
    DefaultValuedAttr<BoolAttr, "false">:$relu);
  let assemblyFormat = "$type $a `,` $b attr-dict `:` type($dst)";

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(NVVM::ConvertFP8Type to,
                                              NVVM::FPRoundingMode rnd,
                                              NVVM::SaturationMode sat,
                                              bool hasRelu);
  }];
  
  string llvmBuilder = [{
    auto intId = NVVM::ConvertF32x2ToF8x2Op::getIntrinsicID($type, $rnd, $sat, $relu);
    llvm::Value *packedI16 = createIntrinsicCall(builder, intId, {$a, $b});
    if(op.getDst().getType().isInteger(16))
      $dst = packedI16;
    else
      $dst = builder.CreateBitCast(packedI16,
                      llvm::FixedVectorType::get(llvm::Type::getInt8Ty(builder.getContext()), 2));
  }];
}

def NVVM_ConvertF16x2ToF8x2Op : NVVM_Op<"convert.f16x2.to.f8x2"> {
  let summary = "Convert an f16x2 input to f8x2";
  let description = [{
    This Op converts the given f16 inputs in an f16x2 vector to the specified 
    f8 type.
    The result `dst` is represented as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values from `a`
    are packed such that the value converted from the first element of `a`
    is stored in the upper 8 bits of `dst` and the value converted from the
    second element of `a` is stored in the lower 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an 
    i8 element in the vector.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];

  let hasVerifier = 1;
  let results = (outs AnyTypeOf<[I16, VectorOfLengthAndType<[2], [I8]>]>:$dst);
  let arguments = (ins
    ConvertFP8TypeAttr:$type,
    VectorOfLengthAndType<[2], [F16]>:$a,
    DefaultValuedAttr<BoolAttr, "false">:$relu);
  let assemblyFormat = "$type $a attr-dict `:` type($a) `->` type($dst)";

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(NVVM::ConvertFP8Type to,
                                              bool hasRelu);
  }];

  string llvmBuilder = [{
    auto intId = NVVM::ConvertF16x2ToF8x2Op::getIntrinsicID($type, $relu);
    llvm::Value *packedI16 = createIntrinsicCall(builder, intId, {$a});
    if(op.getDst().getType().isInteger(16))
      $dst = packedI16;
    else
      $dst = builder.CreateBitCast(packedI16,
                      llvm::FixedVectorType::get(llvm::Type::getInt8Ty(builder.getContext()), 2));
  }];
}

def NVVM_ConvertBF16x2ToF8x2Op : NVVM_Op<"convert.bf16x2.to.f8x2"> {
  let summary = "Convert a pair of bf16 inputs to f8x2";
  let description = [{
    This Op converts the given bf16 inputs in a bf16x2 vector to the specified 
    f8 type.
    The result `dst` is represented as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values from `a`
    are packed such that the value converted from the first element of `a`
    is stored in the upper 8 bits of `dst` and the value converted from the
    second element of `a` is stored in the lower 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an 
    i8 element in the vector.
    The `rnd` and `sat` attributes specify the rounding and saturation modes 
    respectively.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];

  let hasVerifier = 1;
  let results = (outs AnyTypeOf<[I16, VectorOfLengthAndType<[2], [I8]>]>:$dst);
  let arguments = (ins
    ConvertFP8TypeAttr:$type,
    VectorOfLengthAndType<[2], [BF16]>:$a,
    DefaultValuedAttr<FPRoundingModeAttr, "FPRoundingMode::NONE">:$rnd,
    DefaultValuedAttr<SaturationModeAttr, "SaturationMode::NONE">:$sat);
  let assemblyFormat = "$type $a attr-dict `:` type($a) `->` type($dst)";
  
  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(NVVM::FPRoundingMode rnd,
                                              NVVM::SaturationMode sat);
  }];

  string llvmBuilder = [{
    auto intId = NVVM::ConvertBF16x2ToF8x2Op::getIntrinsicID($rnd, $sat);
    llvm::Value *packedI16 = createIntrinsicCall(builder, intId, {$a});
    if(op.getDst().getType().isInteger(16))
      $dst = packedI16;
    else
      $dst = builder.CreateBitCast(packedI16,
                      llvm::FixedVectorType::get(llvm::Type::getInt8Ty(builder.getContext()), 2));
  }];
}

//===----------------------------------------------------------------------===//
// NVVM MMA Ops
//===----------------------------------------------------------------------===//
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
      : (!llvm.ptr<3>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
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
      : !llvm.ptr<3>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>, vector<2 x f16>
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

def LdStMatrixShapeAttr : NVVM_Attr<"LdStMatrixShape", "ld_st_matrix_shape"> {
  let summary = "Matrix shape for ldmatrix and stmatrix";
  let parameters = (ins "int":$m, "int":$n);
  let assemblyFormat = "`<` struct(params) `>`";
}

def LdStMatrixEltTypeB16 : I32EnumAttrCase<"B16", 0, "b16">;
def LdStMatrixEltTypeB8 : I32EnumAttrCase<"B8", 1, "b8">;
def LdStMatrixEltTypeB8X16_B6X16_P32 : I32EnumAttrCase<"B8X16_B6X16_P32", 2, "b8x16.b6x16_p32">;
def LdStMatrixEltTypeB8X16_B4X16_P64 : I32EnumAttrCase<"B8X16_B4X16_P64", 3, "b8x16.b4x16_p64">;

def LdStMatrixEltType : I32EnumAttr<"LdStMatrixEltType", "Element type for ldmatrix and stmatrix",
  [LdStMatrixEltTypeB16, LdStMatrixEltTypeB8,
   LdStMatrixEltTypeB8X16_B6X16_P32, LdStMatrixEltTypeB8X16_B4X16_P64]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def LdStMatrixEltTypeAttr : EnumAttr<NVVM_Dialect, LdStMatrixEltType, "ld_st_matrix_elt_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_StMatrixOp: NVVM_Op<"stmatrix">,
  Arguments<(ins LLVM_PointerShared: $ptr, Variadic<I32>:$sources, MMALayoutAttr:$layout,
             LdStMatrixShapeAttr:$shape, LdStMatrixEltTypeAttr:$eltType)> {
  let summary = "cooperative matrix store";
  let description = [{
    Collectively store one or more matrices across all threads in a warp to the
    location indicated by the address operand $ptr in shared memory.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)
  }];
  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = getStMatrixIntrinsicId($layout, $sources.size(), $shape, $eltType);
      createIntrinsicCall(builder, intId, operands, operands[0]->getType());
  }];
  let assemblyFormat = "$ptr `,` $sources attr-dict `:` type(operands)";
  let hasVerifier = 1;
}

def NVVM_LdMatrixOp: NVVM_Op<"ldmatrix">,
  Results<(outs AnyType:$res)>,
  Arguments<(ins LLVM_PointerShared:$ptr, I32Attr:$num,
                 MMALayoutAttr:$layout,
                 LdStMatrixShapeAttr:$shape,
                 LdStMatrixEltTypeAttr:$eltType)> {

  let summary = "cooperative matrix load";

  string llvmBuilder = [{
      auto operands = moduleTranslation.lookupValues(opInst.getOperands());
      auto intId = getLdMatrixIntrinsicId($layout, $num, $shape, $eltType);
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
      (!llvm.ptr<3>) -> i32
    %l2 = nvvm.ldmatrix %ptr {num = 4 : i32, layout = #nvvm.mma_layout<row>} :
      (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
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
    | bf16     | .m16n8k8  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
    |          | .m16n8k16 | row     | col     | 4x i32   | 2x i32   | 4x f32            |
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

// List of modes supported for TMA Load and Prefetch Ops
def TMALoadModeTile   : I32EnumAttrCase<"TILE", 0, "tile">;
def TMALoadModeIm2Col : I32EnumAttrCase<"IM2COL", 1, "im2col">;
def TMALoadModeIm2ColW : I32EnumAttrCase<"IM2COL_W", 2, "im2col_w">;
def TMALoadModeIm2ColW128 : I32EnumAttrCase<"IM2COL_W_128", 3, "im2col_w_128">;
def TMALoadModeTileGather4 : I32EnumAttrCase<"TILE_GATHER4", 4, "tile_gather4">;

def TMALoadMode : I32EnumAttr<"TMALoadMode", "NVVM TMA Load Mode",
    [TMALoadModeTile, TMALoadModeIm2Col,
     TMALoadModeIm2ColW, TMALoadModeIm2ColW128,
     TMALoadModeTileGather4]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def TMALoadModeAttr : EnumAttr<NVVM_Dialect, TMALoadMode, "tma_load_mode"> {
  let summary = "List of Load-Modes supported for TMA Tensor Ops";
  let description = [{
    TMA Tensor Ops support the following modes, when copying data from
    global memory to shared memory (i.e. load):

    Tile Mode: It's the default mode. The source multi-dimensional tensor
    layout is preserved at the destination.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode)

    Im2col Mode: This mode is used when `im2colOffsets` operands are present.
    The elements in the Bounding Box of the source tensor are rearranged into
    columns at the destination. In this mode, the tensor has to be at least
    3-dimensional. The number of `im2colOffsets` is `dims - 2` where `dims`
    is the dimension of the tensor.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-mode)

    Im2col_W Mode: This mode is similar to Im2Col mode with the restriction that
    elements are accessed across the W dimension only. The number of `im2colOffsets`
    are always two, referred as `wHalo` and `wOffset`.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes)

    Im2col_W_128 Mode: This mode is similar to Im2Col_W mode with the number of
    elements accessed across the W dimension is always 128 only.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-im2col-w-w128-modes)

    Tile_Gather4 Mode: This mode is similar to Tile mode but works only on 2D tensor.
    In gather4 mode, four rows in the source 2D tensor are combined to form a single
    2D tensor at the destination. This mode requires five co-ordinates. The first one
    represents the column-index followed by four row indices.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-scatter4-gather4-modes)
  }];

  let assemblyFormat = "`<` $value `>`";
}

// List of modes supported for TMA Store and Reduction Ops
def TMAStoreModeTile   : I32EnumAttrCase<"TILE", 0, "tile">;
def TMAStoreModeIm2Col : I32EnumAttrCase<"IM2COL", 1, "im2col">;
def TMAStoreModeTileScatter4 : I32EnumAttrCase<"TILE_SCATTER4", 2, "tile_scatter4">;

def TMAStoreMode : I32EnumAttr<"TMAStoreMode", "NVVM TMA Store Mode",
    [TMAStoreModeTile, TMAStoreModeIm2Col, TMAStoreModeTileScatter4]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def TMAStoreModeAttr : EnumAttr<NVVM_Dialect, TMAStoreMode, "tma_store_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_CpAsyncBulkCommitGroupOp : NVVM_Op<"cp.async.bulk.commit.group">,
  Arguments<(ins )> {
  let assemblyFormat = "attr-dict";
  let description = [{
    This Op commits all prior initiated but uncommitted cp.async.bulk
    instructions into a cp.async.bulk-group.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)
  }];

  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::nvvm_cp_async_bulk_commit_group);
  }];
}

def NVVM_CpAsyncBulkWaitGroupOp : NVVM_Op<"cp.async.bulk.wait_group", [NVVMRequiresSM<90>]>,
  Arguments<(ins 
    ConfinedAttr<I32Attr, [IntMinValue<0>]>:$group, 
    OptionalAttr<UnitAttr>:$read)> {
  let assemblyFormat = "$group attr-dict";
  let description = [{
    Op waits for completion of the most recent bulk async-groups.

    The `$group` operand tells waiting has to be done until for $group or fewer
    of the most recent bulk async-groups. If `$group` is 0, the op wait until 
    all the most recent bulk async-groups have completed.

    The `$read` indicates that the waiting has to be done until all the bulk 
    async operations in the specified bulk async-group have completed reading 
    from their source locations.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
  }];
  
  string llvmBuilder = [{
    auto intId = op.getRead() ?
      llvm::Intrinsic::nvvm_cp_async_bulk_wait_group_read :
      llvm::Intrinsic::nvvm_cp_async_bulk_wait_group;
    createIntrinsicCall(builder, intId, builder.getInt32($group));
  }];
}

def NVVM_CpAsyncBulkTensorGlobalToSharedClusterOp : 
  NVVM_Op<"cp.async.bulk.tensor.shared.cluster.global", 
  [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>, 
  AttrSizedOperandSegments, NVVMRequiresSM<90>]>,
  Arguments<(ins  LLVM_PointerShared:$dstMem,
                  LLVM_AnyPointer:$tmaDescriptor,
                  Variadic<I32>:$coordinates,
                  LLVM_PointerShared:$mbar,                  
                  Variadic<I16>:$im2colOffsets,
                  Optional<I16>:$multicastMask,
                  Optional<I64>:$l2CacheHint,
                  PtxPredicate:$predicate)> {
  let description = [{
    Initiates an asynchronous copy operation on the tensor data from global 
    memory to shared memory. 

    The Op operates has two load modes:
    1) Tiled Mode: It's the default mode. The source multi-dimensional tensor 
    layout is preserved at the destination. 

    2) Im2col Mode: This mode is used when `im2colOffsets` operands are present.
    the elements in the Bounding Box of the source tensor are rearranged into
    columns at the destination. In this mode, the tensor has to be at least 
    3-dimensional. 

    The `multicastMask` operand is optional. When it is present, the Op copies
    data from global memory to shared memory of multiple CTAs in the cluster.
    Operand `multicastMask` specifies the destination CTAs in the cluster such 
    that each bit position in the 16-bit `multicastMask` operand corresponds to
    the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.     

    The `l2CacheHint` operand is optional, and it is used to specify cache 
    eviction policy that may be used during the memory access.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
  }];

  let assemblyFormat = [{ 
    $dstMem `,` 
    $tmaDescriptor `,` 
    $mbar `,` 
    `box` `[`$coordinates `]` 
    (`im2col` `[` $im2colOffsets^ `]` )?
    (`multicast_mask` `=` $multicastMask^ )?
    (`l2_cache_hint` `=` $l2CacheHint^ )?
    (`predicate` `=` $predicate^)? 
    attr-dict  `:` type($dstMem) `,` type($tmaDescriptor)
  }];

  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      int im2colDim = getIm2colOffsets().size();
      int dim = getCoordinates().size();
      std::string ptx = "cp.async.bulk.tensor.";
      ptx += std::to_string(dim) + "d.";
      ptx += "shared::cluster.global.mbarrier::complete_tx::bytes";      
      if(im2colDim) ptx += ".im2col";
      if(getMulticastMask()) ptx += ".multicast::cluster";      
      if(getL2CacheHint()) ptx += ".L2::cache_hint";
      
      auto preg = [](int r) { return "%" + std::to_string(r); };

      // Build Registers
      ptx += " [%0], [%1, {";
      int r = 2;      
      for(int i = 0; i < dim; i++) ptx += preg(r+i) + ",";
      ptx.pop_back(); r += dim;
      ptx += "} ], [%" + std::to_string(r++) + "]";
      if(im2colDim) {
        ptx += ",{";
        for(int i = 0; i < im2colDim; i++) ptx += preg(r+i) + ",";
        ptx.pop_back(); r += im2colDim;
        ptx += "}";
      }
      if(getMulticastMask()) ptx += ", " + preg(r++);
      if(getL2CacheHint()) ptx += ", " + preg(r++);
      ptx += ";";
      return ptx;
    }
  }];
  let hasVerifier = 1;
}

def NVVM_CpAsyncBulkTensorSharedCTAToGlobalOp : 
  NVVM_PTXBuilder_Op<"cp.async.bulk.tensor.global.shared.cta",
  [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>,
  AttrSizedOperandSegments]>,
  Arguments<(ins LLVM_PointerGeneric:$tmaDescriptor,
                 LLVM_PointerShared:$srcMem,
                 Variadic<I32>:$coordinates,
                 Optional<I64>:$l2CacheHint,
                 DefaultValuedAttr<TMAStoreModeAttr, "TMAStoreMode::TILE">:$mode,
                 PtxPredicate:$predicate)> {
  let description = [{
    Initiates an asynchronous copy of the tensor data from shared::cta
    memory to global memory. This Op supports all the store modes specified in
    `TMAStoreMode`.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
  }];

  let assemblyFormat = [{ 
    $tmaDescriptor `,` 
    $srcMem `,` 
    `box` `[`$coordinates `]` 
    (`l2_cache_hint` `=` $l2CacheHint^ )?
    (`,` `predicate` `=` $predicate^)?
    attr-dict `:` type($tmaDescriptor) `,` type($srcMem)
  }];

  let extraClassDeclaration = [{
    bool hasIntrinsic() { return !getPredicate(); }

    static mlir::NVVM::IDArgPair
    getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase& builder);
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

  string llvmBuilder = [{
    auto [id, args] = NVVM::CpAsyncBulkTensorSharedCTAToGlobalOp::getIntrinsicIDAndArgs(
                      *op, moduleTranslation, builder);
    createIntrinsicCall(builder, id, args);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM Prefetch Op
//===----------------------------------------------------------------------===//

def PrefetchCacheLevelL1 : I32EnumCase<"L1", 0, "L1">;
def PrefetchCacheLevelL2 : I32EnumCase<"L2", 1, "L2">;

def PrefetchCacheLevel : I32Enum<"PrefetchCacheLevel",
                                 "NVVM Prefetch Cache Level",
                                 [PrefetchCacheLevelL1, PrefetchCacheLevelL2]> {
  let cppNamespace = "::mlir::NVVM";
}

def PrefetchCacheLevelAttr : EnumAttr<NVVM_Dialect, PrefetchCacheLevel, "prefetch_cache_level"> {
  let assemblyFormat = "$value";
}

def NVVM_PrefetchOp : NVVM_Op<"prefetch",
    [DeclareOpInterfaceMethods<BasicPtxBuilderOpInterface>]> {
  let summary = "Brings the cache line containing an address into the specified cache level";
  let description = [{
    Prefetches the cache line containing the address given by `addr`. The 
    operand may be a global, local, or generic pointer. When `tensormap` is 
    specified, the operand may instead be a constant or generic pointer. If the 
    address maps to shared memory, the operation has no effect.

    At most one of `cacheLevel` or `tensormap` may be present. The `cacheLevel` 
    attribute selects the target cache level. When combined with `uniform`, the 
    prefetch is performed to the uniform cache, in which case `addr` must be a 
    generic pointer.

    When `tensormap` is used, the line containing `addr` is brought from the 
    constant or parameter state space for later use by `cp.async.bulk.tensor`. 
    If `in_param_space` is specified, the generic pointer is interpreted as 
    referring to the parameter state space.

    `uniform` can be specified after the `cacheLevel` to indicate that the 
    prefetch is performed to the specified uniform cache level. If `uniform` is 
    specified, `addr` must be a generic address pointer and no operation is 
    performed if `addr` maps to a `const`, `local`, or `shared` memory location.

    The `evictPriority` attribute is optional and specifies the cache eviction
    priority when `cacheLevel` is L2.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu)
  }];
  let arguments = (ins OptionalAttr<PrefetchCacheLevelAttr>:$cacheLevel,
                       OptionalAttr<CacheEvictionPriorityAttr>:$evictPriority,
                       AnyTypeOf<[LLVM_PointerGlobal,
                                  LLVM_PointerLocal,
                                  LLVM_PointerGeneric,
                                  LLVM_PointerConst]>:$addr,
                       PtxPredicate:$predicate,
                       UnitAttr:$tensormap,
                       UnitAttr:$uniform,
                       UnitAttr:$in_param_space);
  let assemblyFormat = "(`level` `=` $cacheLevel^ (`uniform` $uniform^)? `,`)? (`tensormap` $tensormap^ (`in_param_space` $in_param_space^)? `,`)? (`evict_priority` `=` $evictPriority^ `,`)? $addr (`,` `predicate` `=` $predicate^)? attr-dict `:` type(operands)";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static NVVM::IDArgPair
    getIntrinsicIDAndArgs(NVVM::PrefetchOp &op,LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase &builder);
    bool hasIntrinsic() { return !getPredicate() || !getTensormap(); }
  }];
  let extraClassDefinition = [{
    std::string $cppClass::getPtx() {
      // Inline PTX is only supported for prefetch tensormap
      return std::string("prefetch.tensormap [%0];");
    }
  }];
  let llvmBuilder = [{
    auto [id, args] = NVVM::PrefetchOp::getIntrinsicIDAndArgs(op,
                                          moduleTranslation, builder);

    if(op.getTensormap())
      // Overloaded intrinsic
      createIntrinsicCall(builder, id, args, {args[0]->getType()});
    else
      createIntrinsicCall(builder, id, args);
  }];
}

def NVVM_CpAsyncBulkPrefetchOp : NVVM_Op<"cp.async.bulk.prefetch"> {
  let summary = "Async bulk prefetch from global memory to L2 cache";
  let description = [{
    Initiates an asynchronous prefetch of data from the location
    specified by `srcMem` to the L2 cache.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    Example:
    ```mlir
      nvvm.cp.async.bulk.prefetch %src, %size : !llvm.ptr<1>

      // with l2_cache_hint
      nvvm.cp.async.bulk.prefetch %src, %size l2_cache_hint = %ch : !llvm.ptr<1>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch)
  }];

  let arguments = (ins
    LLVM_PointerGlobal:$srcMem,
    I32:$size,
    Optional<I64>:$l2CacheHint);

  let assemblyFormat = [{
    $srcMem `,` $size (`l2_cache_hint` `=` $l2CacheHint^ )?
    attr-dict  `:` type($srcMem)
  }];

  let extraClassDeclaration = [{
    static mlir::NVVM::IDArgPair
    getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase& builder);
  }];

  string llvmBuilder = [{
    auto [id, args] = NVVM::CpAsyncBulkPrefetchOp::getIntrinsicIDAndArgs(
                      *op, moduleTranslation, builder);
    createIntrinsicCall(builder, id, args);
  }];
}

def NVVM_CpAsyncBulkTensorPrefetchOp :
  NVVM_Op<"cp.async.bulk.tensor.prefetch", [AttrSizedOperandSegments]> {
  let arguments = (ins
    LLVM_PointerGeneric:$tmaDescriptor,
    Variadic<I32>:$coordinates,
    Variadic<I16>:$im2colOffsets,
    DefaultValuedAttr<TMALoadModeAttr, "TMALoadMode::TILE">:$mode,
    Optional<I64>:$l2CacheHint);

  let description = [{
    Initiates an asynchronous prefetch operation on the tensor data from global
    memory to L2 cache. This Op supports all the load modes specified in
    `TMALoadMode`.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
  }];

  let assemblyFormat = [{
    $tmaDescriptor `,`
    `box` `[`$coordinates `]`
    (`im2col` `[` $im2colOffsets^ `]` )?
    (`l2_cache_hint` `=` $l2CacheHint^ )?
    attr-dict  `:` type($tmaDescriptor)
  }];

  let extraClassDeclaration = [{
    static mlir::NVVM::IDArgPair
    getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase& builder);
  }];

  let hasVerifier = 1;

  string llvmBuilder = [{
    auto [id, args] = NVVM::CpAsyncBulkTensorPrefetchOp::getIntrinsicIDAndArgs(
                      *op, moduleTranslation, builder);
    createIntrinsicCall(builder, id, args);
  }];
}

// List of Reduction Ops supported with TMA Store
def TMAReduxKindAdd : I32EnumAttrCase<"ADD", 0, "add">;
def TMAReduxKindMin : I32EnumAttrCase<"MIN", 1, "min">;
def TMAReduxKindMax : I32EnumAttrCase<"MAX", 2, "max">;
def TMAReduxKindInc : I32EnumAttrCase<"INC", 3, "inc">;
def TMAReduxKindDec : I32EnumAttrCase<"DEC", 4, "dec">;
def TMAReduxKindAnd : I32EnumAttrCase<"AND", 5, "and">;
def TMAReduxKindOr  : I32EnumAttrCase<"OR",  6, "or">;
def TMAReduxKindXor : I32EnumAttrCase<"XOR", 7, "xor">;

def TMAReduxKind : I32EnumAttr<"TMAReduxKind", "NVVM TMA redux kind",
    [TMAReduxKindAdd, TMAReduxKindMax, TMAReduxKindMin,
     TMAReduxKindInc, TMAReduxKindDec, TMAReduxKindAnd,
     TMAReduxKindOr,  TMAReduxKindXor]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def TMAReduxKindAttr : EnumAttr<NVVM_Dialect, TMAReduxKind, "tma_redux_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_CpAsyncBulkTensorReduceOp :
  NVVM_Op<"cp.async.bulk.tensor.reduce", [AttrSizedOperandSegments]> {
  let arguments = (ins
    LLVM_AnyPointer:$tmaDescriptor,
    LLVM_PointerShared:$srcMem,
    TMAReduxKindAttr:$redKind,
    DefaultValuedAttr<TMAStoreModeAttr, "TMAStoreMode::TILE">:$mode,
    Variadic<I32>:$coordinates,
    Optional<I64>:$l2CacheHint);

  let description = [{
    Initiates an asynchronous reduction operation of tensor data in
    global memory with tensor data in shared memory.

    The `mode` attribute indicates whether the copy mode is tile or im2col.
    The `redOp` attribute specifies the reduction operations applied.
    The supported reduction operations are:
    {add, min, max, inc, dec, and, or, xor}

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)
  }];

  let assemblyFormat = [{
    $tmaDescriptor `,`
    $srcMem `,`
    `box` `[`$coordinates `]`
    (`l2_cache_hint` `=` $l2CacheHint^ )?
    attr-dict  `:` type($tmaDescriptor) `,` type($srcMem)
  }];

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(int tensorDims,
                                              NVVM::TMAReduxKind kind,
                                              bool isIm2Col);
  }];

  let hasVerifier = 1;

  string llvmBuilder = [{
    // Arguments to the intrinsic:
    // shared_mem_ptr, tmaDesc, tensorDims
    // cache_hint(if applicable) and flag(boolean)
    llvm::SmallVector<llvm::Value *> translatedOperands;
    translatedOperands.push_back($srcMem);
    translatedOperands.push_back($tmaDescriptor);

    for (auto v : op.getCoordinates())
      translatedOperands.push_back(moduleTranslation.lookupValue(v));

    llvm::LLVMContext &ctx = moduleTranslation.getLLVMContext();
    auto *i64Undef = llvm::UndefValue::get(llvm::IntegerType::get(ctx, 64));

    bool isCacheHint = op.getL2CacheHint() ? true : false;
    translatedOperands.push_back(isCacheHint ? $l2CacheHint : i64Undef);
    translatedOperands.push_back(builder.getInt1(isCacheHint));

    auto intId = NVVM::CpAsyncBulkTensorReduceOp::getIntrinsicID(
                 op.getCoordinates().size(), $redKind,
                 (op.getMode() == NVVM::TMAStoreMode::IM2COL));
    createIntrinsicCall(builder, intId, translatedOperands);
  }];
}

def NVVM_CpAsyncBulkGlobalToSharedClusterOp :
  NVVM_Op<"cp.async.bulk.shared.cluster.global", [AttrSizedOperandSegments]> {
  let summary = "Async bulk copy from global memory to Shared cluster memory";
  let description = [{
    Initiates an asynchronous copy operation from global memory to cluster's
    shared memory.

    The `multicastMask` operand is optional. When it is present, the Op copies
    data from global memory to shared memory of multiple CTAs in the cluster.
    Operand `multicastMask` specifies the destination CTAs in the cluster such
    that each bit position in the 16-bit `multicastMask` operand corresponds to
    the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
  }];

  let arguments = (ins
    LLVM_PointerSharedCluster:$dstMem,
    LLVM_PointerGlobal:$srcMem,
    LLVM_PointerShared:$mbar,
    I32:$size,
    Optional<I16>:$multicastMask,
    Optional<I64>:$l2CacheHint);

  let assemblyFormat = [{
    $dstMem `,` $srcMem `,` $mbar `,` $size
    (`multicast_mask` `=` $multicastMask^ )?
    (`l2_cache_hint` `=` $l2CacheHint^ )?
    attr-dict  `:` type($dstMem) `,` type($srcMem)
  }];

  string llvmBuilder = [{
    // Arguments to the intrinsic:
    // dst, mbar, src, size
    // multicast_mask, cache_hint,
    // flag for multicast_mask,
    // flag for cache_hint
    llvm::SmallVector<llvm::Value *> translatedOperands;
    translatedOperands.push_back($dstMem);
    translatedOperands.push_back($mbar);
    translatedOperands.push_back($srcMem);
    translatedOperands.push_back($size);

    // Multicast, if available
    llvm::LLVMContext &ctx = moduleTranslation.getLLVMContext();
    auto *i16Unused = llvm::ConstantInt::get(llvm::Type::getInt16Ty(ctx), 0);
    bool isMulticast = op.getMulticastMask() ? true : false;
    translatedOperands.push_back(isMulticast ? $multicastMask : i16Unused);

    // Cachehint, if available
    auto *i64Unused = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0);
    bool isCacheHint = op.getL2CacheHint() ? true : false;
    translatedOperands.push_back(isCacheHint ? $l2CacheHint : i64Unused);

    // Flag arguments for multicast and cachehint
    translatedOperands.push_back(builder.getInt1(isMulticast));
    translatedOperands.push_back(builder.getInt1(isCacheHint));

    createIntrinsicCall(builder,
      llvm::Intrinsic::nvvm_cp_async_bulk_global_to_shared_cluster, translatedOperands);
  }];
}

def NVVM_CpAsyncBulkSharedCTAToSharedClusterOp :
  NVVM_Op<"cp.async.bulk.shared.cluster.shared.cta"> {
  let summary = "Async bulk copy from Shared CTA memory to Shared cluster memory";
  let description = [{
    Initiates an asynchronous copy operation from Shared CTA memory to Shared
    cluster memory.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
  }];

  let arguments = (ins
    LLVM_PointerSharedCluster:$dstMem,
    LLVM_PointerShared:$srcMem,
    LLVM_PointerShared:$mbar,
    I32:$size);

  let assemblyFormat = [{
    $dstMem `,` $srcMem `,` $mbar `,` $size
    attr-dict  `:` type($dstMem) `,` type($srcMem)
  }];

  string llvmBuilder = [{
    createIntrinsicCall(builder,
      llvm::Intrinsic::nvvm_cp_async_bulk_shared_cta_to_cluster,
      {$dstMem, $mbar, $srcMem, $size});
  }];
}

def NVVM_CpAsyncBulkSharedCTAToGlobalOp :
  NVVM_Op<"cp.async.bulk.global.shared.cta", [AttrSizedOperandSegments]> {
  let summary = "Async bulk copy from Shared CTA memory to Global memory";
  let description = [{
    Initiates an asynchronous copy operation from Shared CTA memory to
    global memory. The 32-bit operand `size` specifies the amount of
    memory to be copied, in terms of number of bytes. `size` must be a
    multiple of 16. The `l2CacheHint` operand is optional, and it is used
    to specify cache eviction policy that may be used during the memory
    access. The `byteMask` operand is optional. The i-th bit in the 16-bit
    wide `byteMask` specifies whether the i-th byte of each 16-byte wide
    chunk of source data is copied to the destination. If the bit is set,
    the byte is copied.

    Example:
    ```mlir
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size
          : !llvm.ptr<1>, !llvm.ptr<3>

      // with l2_cache_hint
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch
          : !llvm.ptr<1>, !llvm.ptr<3>

      // with byte_mask
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size byte_mask = %mask
          : !llvm.ptr<1>, !llvm.ptr<3>

      // with both l2_cache_hint and byte_mask
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch byte_mask = %mask
          : !llvm.ptr<1>, !llvm.ptr<3>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
  }];

  let arguments = (ins
    LLVM_PointerGlobal:$dstMem,
    LLVM_PointerShared:$srcMem,
    I32:$size,
    Optional<I64>:$l2CacheHint,
    Optional<I16>:$byteMask);

  let assemblyFormat = [{
    $dstMem `,` $srcMem `,` $size
    (`l2_cache_hint` `=` $l2CacheHint^ )?
    (`byte_mask` `=` $byteMask^ )?
    attr-dict `:` type($dstMem) `,` type($srcMem)
  }];

  let extraClassDeclaration = [{
    static mlir::NVVM::IDArgPair
    getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase& builder);
  }];
  string llvmBuilder = [{
    auto [id, args] = NVVM::CpAsyncBulkSharedCTAToGlobalOp::getIntrinsicIDAndArgs(
                      *op, moduleTranslation, builder);
    createIntrinsicCall(builder, id, args);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM Wgmma Ops
//===----------------------------------------------------------------------===//

def NVVM_WgmmaFenceAlignedOp : NVVM_Op<"wgmma.fence.aligned", [NVVMRequiresSMa<[90]>]> {
  let arguments = (ins);
  let description = [{
    Enforce an ordering of register accesses between warpgroup level matrix 
    multiplication and other operations. 
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
  }];
  let assemblyFormat = "attr-dict";
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::nvvm_wgmma_fence_sync_aligned);
  }];
}

def NVVM_WgmmaGroupSyncAlignedOp : NVVM_Op<"wgmma.commit.group.sync.aligned", [NVVMRequiresSMa<[90]>]> {
  let assemblyFormat = "attr-dict";
  let description = [{
    Commits all prior uncommitted warpgroup level matrix multiplication operations.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)
  }];
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::nvvm_wgmma_commit_group_sync_aligned);
  }];
}

def NVVM_WgmmaWaitGroupSyncOp : NVVM_Op<"wgmma.wait.group.sync.aligned", [NVVMRequiresSMa<[90]>]> {
  let arguments = (ins I64Attr:$group);
  let assemblyFormat = "attr-dict $group";
  let description = [{
    Signal the completion of a preceding warpgroup operation.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
  }];
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::nvvm_wgmma_wait_group_sync_aligned, builder.getInt64($group));
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
def WGMMATypeF32 : I32EnumAttrCase<"f32", 8>;
def WGMMATypeS32 : I32EnumAttrCase<"s32", 9>;

def WGMMATypes : I32EnumAttr<"WGMMATypes", "NVVM WGMMA types",
  [WGMMATypeF16, WGMMATypeTF32,
    WGMMATypeU8, WGMMATypeS8,
    WGMMATypeB1, WGMMATypeBF16, WGMMATypeF8E4M3, 
    WGMMATypeF8E5M2, WGMMATypeF32, WGMMATypeS32]> {
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
    WGMMATypesAttr:$typeD,
    WGMMAScaleOutAttr:$scaleD,
    WGMMAScaleInAttr:$scaleA,
    WGMMAScaleInAttr:$scaleB, 
    MMALayoutAttr:$layoutA,
    MMALayoutAttr:$layoutB,
    OptionalAttr<MMAIntOverflowAttr>:$satfinite
  );  
  
   let assemblyFormat = [{ 
      $descriptorA `,` $descriptorB `,` $inouts `,` $shape `,`
      `D` `[` $typeD `,` $scaleD (`,` $satfinite^)? `]` `,`
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
    bool getAsmValues(RewriterBase &, llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>> &);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM Griddepcontrol Ops
//===----------------------------------------------------------------------===//

def GridDepActionWait : I32EnumCase<"wait", 0>;
def GridDepActionLaunchDependent : I32EnumCase<"launch_dependents", 1>;

def GridDepActionKind : I32Enum<"GridDepActionKind", "Action kind for grid dependency control",
  [GridDepActionWait, GridDepActionLaunchDependent]> {
  let cppNamespace = "::mlir::NVVM";
}

def GridDepActionAttr : EnumAttr<NVVM_Dialect, GridDepActionKind, "grid_dep_action">;

def NVVM_GriddepcontrolOp : NVVM_Op<"griddepcontrol", []> {
  let description = [{
    If the $kind attribute is set to `wait`, it causes the 
    executing thread to wait until all prerequisite grids in flight 
    have completed and all the memory operations from the prerequisite grids 
    are performed and made visible to the current grid.

    When the $kind is launch_dependents, it signals that specific dependents 
    the runtime system designated to react to this instruction can be scheduled 
    as soon as all other CTAs in the grid issue the same instruction or have 
    completed.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)
  }];

  let arguments = (ins GridDepActionAttr:$kind);

  let assemblyFormat = "$kind attr-dict";

  string llvmBuilder = [{
    llvm::Intrinsic::ID id;
      switch ($kind) {
        case NVVM::GridDepActionKind::wait:
          id = llvm::Intrinsic::nvvm_griddepcontrol_wait;
          break;
        case NVVM::GridDepActionKind::launch_dependents:
          id = llvm::Intrinsic::nvvm_griddepcontrol_launch_dependents;
          break;
      }
      createIntrinsicCall(builder, id);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM Mapa Op
//===----------------------------------------------------------------------===//

def NVVM_MapaOp: NVVM_Op<"mapa",
  [InputAddressIsCombinationOf<["a", "res"],
    [[LLVM_PointerShared, LLVM_PointerSharedCluster], [LLVM_PointerGeneric, LLVM_PointerGeneric]],
    "Valid address-space check(or mapping) for mapa Op">, NVVMRequiresSM<90>]> {
  let results = (outs AnyTypeOf<[LLVM_PointerGeneric, LLVM_PointerSharedCluster]>:$res);
  let arguments = (ins AnyTypeOf<[LLVM_PointerGeneric, LLVM_PointerShared]>:$a, I32:$b);

  string llvmBuilder = [{
    int addrSpace = llvm::cast<LLVMPointerType>(op.getA().getType()).getAddressSpace();
    
    bool isSharedMemory = addrSpace == NVVM::NVVMMemorySpace::kSharedMemorySpace;

    auto intId = isSharedMemory? llvm::Intrinsic::nvvm_mapa_shared_cluster : llvm::Intrinsic::nvvm_mapa;
    $res = createIntrinsicCall(builder, intId, {$a, $b});
  }];
  
  let assemblyFormat = "$a`,` $b attr-dict `:` type($a) `->` type($res)";
}

//===----------------------------------------------------------------------===//
// NVVM match.sync Op
//===----------------------------------------------------------------------===//

def MatchSyncKindAny : I32EnumAttrCase<"any", 0>;
def MatchSyncKindAll : I32EnumAttrCase<"all", 1>;

def MatchSyncKind : I32EnumAttr<"MatchSyncKind", "NVVM match sync kind",
  [MatchSyncKindAny, MatchSyncKindAll]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}

def MatchSyncKindAttr : EnumAttr<NVVM_Dialect, MatchSyncKind, "match_sync_kind">;

def NVVM_MatchSyncOp : NVVM_Op<"match.sync">,
  Results<(outs AnyTypeOf<[I32, LLVMStructType]>:$res)>,
  Arguments<(ins I32:$thread_mask,
                 AnyTypeOf<[I32, I64]>:$val,
                 MatchSyncKindAttr:$kind)> {
  let summary = "Broadcast and compare a value across threads in warp";
  let description = [{
    The `match.sync` op performs broadcast and compare of operand `val` across 
    all non-exited threads in `thread_mask` and returns a mask depending on the 
    kind and an optional predicate.

    The matching operation kinds are:
    - `any`: Returns a mask corresponding to the non-exited threads in the 
    `thread_mask` that have the same value of operand `val`.
    - `all`: Returns a mask and a predicate. If all non-exited threads in the 
    `thread_mask` have the same value of operand `val`, the predicate is set to 
    true and the mask corresponds to the non-exited threads in the 
    `thread_mask`. Otherwise, the predicate is set to false and the mask is 0.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync)
  }];
  string llvmBuilder = [{
    auto intId = getMatchSyncIntrinsicId(
        op.getVal().getType(), $kind);
    $res = createIntrinsicCall(builder,
        intId, {$thread_mask, $val});
  }];
  let assemblyFormat = "$kind $thread_mask `,` $val attr-dict `:` type($val) `->` type($res)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NVVM Bulk Store Op
//===----------------------------------------------------------------------===//

def NVVM_BulkStoreOp: NVVM_Op<"st.bulk"> {
  let arguments = (ins AnyTypeOf<[LLVM_PointerGeneric, LLVM_PointerShared]>:$addr, I64:$size, DefaultValuedAttr<I64Attr, "0">:$initVal);

  let summary = "Bulk Store Op";
  let description = [{
    Initializes a region of shared memory at the address given by `addr`.
    The `size` operand specifies the number of bytes to initialize and must be 
    a multiple of 8.
    The `initVal` operand specifies the value to initialize the memory to. The 
    only supported value is 0.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
  }];

  string llvmBuilder = [{
    auto intId = getStBulkIntrinsicId(
          llvm::cast<LLVM::LLVMPointerType>(op.getAddr().getType()));
    createIntrinsicCall(builder, intId,
                      {$addr, $size, builder.getInt64($initVal)});
  }];
  
  let assemblyFormat = "$addr `,` `size` `=` $size (`,` `init` `=` $initVal^)? attr-dict `:` type($addr)";
  
  let hasVerifier = 1;
}

def NVVM_Exit : NVVM_Op<"exit"> {
  let summary = "Exit Op";
  let description = [{
    Ends execution of a thread.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit)
  }];
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::nvvm_exit);
  }];

  let assemblyFormat = "attr-dict";
}


//===----------------------------------------------------------------------===//
// NVVM breakpoint Op
//===----------------------------------------------------------------------===//

def NVVM_Breakpoint : NVVM_Op<"breakpoint"> {
  let summary = "Breakpoint Op";
  let description = [{
    Breakpoint suspends execution of the program for debugging.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt)
  }];
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::debugtrap);
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// NVVM TCGEN05 Ops
//===----------------------------------------------------------------------===//
// Num CTAs in a group participating in the TCGEN05 operation.
// This corresponds to the "cta_group::1", "cta_group::2"
// modifiers in the PTX instructions.
def Tcgen05GroupCTA_1 : I32EnumAttrCase<"CTA_1", 0, "cta_1">;
def Tcgen05GroupCTA_2 : I32EnumAttrCase<"CTA_2", 1, "cta_2">;

def Tcgen05GroupKind : I32EnumAttr<"Tcgen05GroupKind",
                            "NVVM Tcgen05 group kind",
  [Tcgen05GroupCTA_1, Tcgen05GroupCTA_2]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def Tcgen05GroupKindAttr :
  EnumAttr<NVVM_Dialect, Tcgen05GroupKind, "tcgen05_group"> {
  let assemblyFormat = "`<` $value `>`";
}

def Tcgen05FenceBefore : I32EnumAttrCase<"BEFORE_THREAD_SYNC", 0, "before">;
def Tcgen05FenceAfter  : I32EnumAttrCase<"AFTER_THREAD_SYNC",  1, "after">;
def Tcgen05FenceKind : I32EnumAttr<"Tcgen05FenceKind", "NVVM Tcgen05 fence kind",
  [Tcgen05FenceBefore, Tcgen05FenceAfter]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def Tcgen05FenceKindAttr :
  EnumAttr<NVVM_Dialect, Tcgen05FenceKind, "tcgen05_fence"> {
  let assemblyFormat = "`<` $value `>`";
}

def Tcgen05WaitLoad  : I32EnumAttrCase<"LOAD",  0, "load">;
def Tcgen05WaitStore : I32EnumAttrCase<"STORE", 1, "store">;
def Tcgen05WaitKind : I32EnumAttr<"Tcgen05WaitKind", "NVVM Tcgen05 wait kind",
  [Tcgen05WaitLoad, Tcgen05WaitStore]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def Tcgen05WaitKindAttr :
  EnumAttr<NVVM_Dialect, Tcgen05WaitKind, "tcgen05_wait"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_Tcgen05AllocOp : NVVM_Op<"tcgen05.alloc", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 alloc operation";
  let description = [{
    The `tcgen05.alloc` Op allocates tensor core memory for
    the amount specified by `nCols` and writes the destination
    address to the `addr` argument. The `nCols` operand specifies the
    number of columns to be allocated and it must be a power-of-two.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
  }];

  let arguments = (ins
    AnyTypeOf<[LLVM_AnyPointer, LLVM_PointerShared]>:$addr,
    I32:$nCols,
    DefaultValuedAttr<Tcgen05GroupKindAttr, "Tcgen05GroupKind::CTA_1">:$group);

  let assemblyFormat = "$addr `,` $nCols attr-dict `:` type(operands)";

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID
      getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                            llvm::SmallVector<llvm::Value *> &args);
  }];
  string llvmBuilder = [{
    llvm::SmallVector<llvm::Value *> args;
    auto id = NVVM::Tcgen05AllocOp::getIntrinsicIDAndArgs(
      *op, moduleTranslation, args);
    createIntrinsicCall(builder, id, args);
  }];
}

def NVVM_Tcgen05DeallocOp : NVVM_Op<"tcgen05.dealloc", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 dealloc operation";
  let description = [{
    The `tcgen05.dealloc` Op de-allocates the tensor core memory
    specified by `tmemAddr`, which must be from a previous tensor
    memory allocation. The `nCols` operand specifies the number
    of columns to be de-allocated, and it must be a power-of-two.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
  }];

  let arguments = (ins LLVM_PointerTensor:$taddr, I32:$nCols,
    DefaultValuedAttr<Tcgen05GroupKindAttr, "Tcgen05GroupKind::CTA_1">:$group);

  let assemblyFormat = "$taddr `,` $nCols attr-dict `:` type(operands)";

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID
      getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                            llvm::SmallVector<llvm::Value *> &args);
  }];
  string llvmBuilder = [{
    llvm::SmallVector<llvm::Value *> args;
    auto id = NVVM::Tcgen05DeallocOp::getIntrinsicIDAndArgs(
      *op, moduleTranslation, args);
    createIntrinsicCall(builder, id, args);
  }];
}

def NVVM_Tcgen05RelinquishAllocPermitOp : NVVM_Op<"tcgen05.relinquish_alloc_permit", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 Op to relinquish the right to allocate";
  let description = [{
    The `tcgen05.relinquish_alloc_permit` Op specifies that the CTA
    of the executing thread is relinquishing the right to allocate
    Tensor Memory. So, it is illegal for a CTA to perform `tcgen05.alloc`
    after any of its constituent threads execute `tcgen05.relinquish_alloc_permit`.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
  }];

  let arguments = (ins
    DefaultValuedAttr<Tcgen05GroupKindAttr, "Tcgen05GroupKind::CTA_1">:$group);

  let assemblyFormat = "attr-dict";

  string llvmBuilder = [{
    auto id = ($group == NVVM::Tcgen05GroupKind::CTA_1) ?
      llvm::Intrinsic::nvvm_tcgen05_relinq_alloc_permit_cg1 :
      llvm::Intrinsic::nvvm_tcgen05_relinq_alloc_permit_cg2;
    createIntrinsicCall(builder, id);
  }];
}

def NVVM_Tcgen05FenceOp : NVVM_Op<"tcgen05.fence", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 fence operations";
  let description = [{
    The `tcgen05.fence<before>` orders all prior async tcgen05 operations
    with respect to the subsequent tcgen05 and execution ordering operations.
    The `tcgen05.fence<after>` orders all subsequent async tcgen05 operations
    with respect to the prior tcgen05 and execution ordering operations.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-fence)
  }];

  let arguments = (ins Tcgen05FenceKindAttr:$kind);
  let assemblyFormat = "$kind attr-dict";

  string llvmBuilder = [{
    auto id = ($kind == NVVM::Tcgen05FenceKind::BEFORE_THREAD_SYNC)
      ? llvm::Intrinsic::nvvm_tcgen05_fence_before_thread_sync
      : llvm::Intrinsic::nvvm_tcgen05_fence_after_thread_sync;
    createIntrinsicCall(builder, id);
  }];
}

def NVVM_Tcgen05WaitOp : NVVM_Op<"tcgen05.wait", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 wait operations";
  let description = [{
    The `tcgen05.wait<load>` causes the executing thread to block until
    all prior `tcgen05.ld` operations issued by the executing thread
    have completed. Similarly, the `tcgen05.wait<store>` causes the executing
    thread to block until all prior `tcgen05.st` operations issued by the
    executing thread have completed.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait)
  }];

  let arguments = (ins Tcgen05WaitKindAttr:$kind);
  let assemblyFormat = "$kind attr-dict";

  string llvmBuilder = [{
    auto id = ($kind == NVVM::Tcgen05WaitKind::LOAD)
      ? llvm::Intrinsic::nvvm_tcgen05_wait_ld
      : llvm::Intrinsic::nvvm_tcgen05_wait_st;
    createIntrinsicCall(builder, id);
  }];
}

def NVVM_Tcgen05CommitOp : NVVM_Op<"tcgen05.commit", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 commit operations";
  let description = [{
    The `tcgen05.commit` makes the mbarrier object, specified by
    the operand `addr`, track the completion of all the prior
    async-tcgen05 operations initiated by the executing thread.
    The multicast variants allow signaling on the mbarrier objects
    of multiple CTAs within the cluster. Operand `multicastMask`,
    when present, specifies the destination CTAs in the cluster such
    that each bit position in the 16-bit `multicastMask` operand
    corresponds to the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
  }];

  let arguments = (ins
    AnyTypeOf<[LLVM_AnyPointer, LLVM_PointerShared]>:$addr,
    Optional<I16>:$multicastMask,
    DefaultValuedAttr<Tcgen05GroupKindAttr, "Tcgen05GroupKind::CTA_1">:$group);

  let assemblyFormat = [{
    $addr (`,` `multicast_mask` `=` $multicastMask^)?
    attr-dict `:` type(operands)
  }];

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID
      getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                            llvm::SmallVector<llvm::Value *> &args);
  }];

  string llvmBuilder = [{
    llvm::SmallVector<llvm::Value *> args;
    auto id = NVVM::Tcgen05CommitOp::getIntrinsicIDAndArgs(
      *op, moduleTranslation, args);
    createIntrinsicCall(builder, id, args);
  }];
}

def NVVM_Tcgen05ShiftOp : NVVM_Op<"tcgen05.shift", [NVVMRequiresSMa<[100, 101, 103]>]> {
  let summary = "Tcgen05 shift operation";
  let description = [{
    The `tcgen05.shift` is an asynchronous instruction which initiates
    the shifting of 32-byte elements downwards across all the rows,
    except the last, by one row. The operand `taddr` specifies the base
    address of the matrix in Tensor Memory whose rows must be down shifted.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift)
  }];

  let arguments = (ins LLVM_PointerTensor:$taddr,
    DefaultValuedAttr<Tcgen05GroupKindAttr, "Tcgen05GroupKind::CTA_1">:$group);

  let assemblyFormat = "$taddr attr-dict `:` type(operands)";

  string llvmBuilder = [{
    auto id = ($group == NVVM::Tcgen05GroupKind::CTA_1) ?
      llvm::Intrinsic::nvvm_tcgen05_shift_down_cg1 :
      llvm::Intrinsic::nvvm_tcgen05_shift_down_cg2;
    createIntrinsicCall(builder, id, {$taddr});
  }];
}

def Shape128x256b : I32EnumAttrCase<"SHAPE_128x256b", 0, "shape_128x256b">;
def Shape4x256b   : I32EnumAttrCase<"SHAPE_4x256b",   1, "shape_4x256b">;
def Shape128x128b : I32EnumAttrCase<"SHAPE_128x128b", 2, "shape_128x128b">;
def Shape64x128b  : I32EnumAttrCase<"SHAPE_64x128b",  3, "shape_64x128b">;
def Shape32x128b  : I32EnumAttrCase<"SHAPE_32x128b",  4, "shape_32x128b">;

def Tcgen05CpShape : I32EnumAttr<"Tcgen05CpShape", "tcgen05 cp shapes",
  [Shape128x256b, Shape4x256b, Shape128x128b, Shape64x128b, Shape32x128b]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05CpShapeAttr : EnumAttr<NVVM_Dialect, Tcgen05CpShape, "tcgen05_cp_shape"> {
  let assemblyFormat = "`<` $value `>`";
}

def Tcgen05CpMulticastNone: I32EnumAttrCase<"NONE", 0, "none">;
def Tcgen05CpMulticastWarpx2_02_13: I32EnumAttrCase<"WARPX2_02_13", 1, "warpx2_02_13">;
def Tcgen05CpMulticastWarpx2_01_23: I32EnumAttrCase<"WARPX2_01_23", 2, "warpx2_01_23">;
def Tcgen05CpMulticastWarpx4: I32EnumAttrCase<"WARPX4", 3, "warpx4">;

def Tcgen05CpMulticast : I32EnumAttr<"Tcgen05CpMulticast", "tcgen05 cp multicast",
  [Tcgen05CpMulticastNone, Tcgen05CpMulticastWarpx2_02_13,
   Tcgen05CpMulticastWarpx2_01_23, Tcgen05CpMulticastWarpx4]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05CpMulticastAttr : EnumAttr<NVVM_Dialect, Tcgen05CpMulticast, "tcgen05_cp_multicast"> {
  let assemblyFormat = "`<` $value `>`";
}

def FormatB6x16_P32: I32EnumAttrCase<"B6x16_P32", 0, "b6x16_p32">;
def FormatB4x16_P64: I32EnumAttrCase<"B4x16_P64", 1, "b4x16_p64">;

def Tcgen05CpSrcFormat : I32EnumAttr<"Tcgen05CpSrcFormat", "tcgen05 cp source format",
  [FormatB6x16_P32, FormatB4x16_P64]> {
    let cppNamespace = "::mlir::NVVM";
    let genSpecializedAttr = 0;
}
def Tcgen05CpSrcFormatAttr : EnumAttr<NVVM_Dialect, Tcgen05CpSrcFormat, "tcgen05_cp_src_fmt"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_Tcgen05CpOp : NVVM_Op<"tcgen05.cp", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "Tcgen05 copy operation";
  let description = [{
    Instruction tcgen05.cp initiates an asynchronous copy operation from
    shared memory to the location specified by the address operand `taddr`
    in the Tensor Memory. The 64-bit register operand `smem_desc` specifies
    the matrix descriptor representing the source matrix in the shared memory
    that needs to be copied.

    Example:
    ```mlir
      nvvm.tcgen05.cp %taddr, %smem_desc {
        group = #nvvm.tcgen05_group<cta_2>,
        shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
        multicast = #nvvm.tcgen05_cp_multicast<warpx2_01_23>,
        srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
      }
    ```
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-cp)
  }];

  let arguments = (ins
    Tcgen05CpShapeAttr:$shape,
    DefaultValuedAttr<Tcgen05GroupKindAttr, "Tcgen05GroupKind::CTA_1">:$group,
    DefaultValuedAttr<Tcgen05CpMulticastAttr, "Tcgen05CpMulticast::NONE">:$multicast,
    OptionalAttr<Tcgen05CpSrcFormatAttr>:$srcFormat,
    LLVM_PointerTensor:$taddr,
    I64:$smem_desc);

  let assemblyFormat = "$taddr`,` $smem_desc attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static llvm::Intrinsic::ID getIntrinsicID(Operation &op);
  }];

  string llvmBuilder = [{
    auto id = NVVM::Tcgen05CpOp::getIntrinsicID(*op);
    createIntrinsicCall(builder, id, {$taddr, $smem_desc});
  }];
}

def NVVM_Tcgen05MmaSmemDescOp : NVVM_Op<"tcgen05.mma_smem_desc", []> {
  let summary = "Constructs a Shared Memory descriptor for MMA Operands A or B";
  let description = [{
    The `nvvm.tcgen05_mma_smem_desc` constructs a Shared Memory descriptor
    for tcgen05.mma. This descriptor is a 64-bit value which describes the
    properties of multiplicand matrix in shared memory including its location
    in the shared memory of the current CTA.

    ```
    +-----------+------+------------------------------------------------------+
    | Bit-field | Size | Description                                          |
    +-----------+------+------------------------------------------------------+
    | 0-13      | 14   | Matrix start address                                 |
    | 14-15     | 2    | Reserved                                             |
    | 16-29     | 14   | Leading dim relative-offset (or) absolute-address    |
    | 30-31     | 2    | Reserved                                             |
    | 32-45     | 14   | Stride dimension byte offset                         |
    | 46-48     | 3    | Fixed constant value of 0b001                        |
    | 49-51     | 3    | Matrix base offset                                   |
    | 52        | 1    | Leading dimension stride mode:                       |
    |           |      |   0: byte offset relative                            |
    |           |      |   1: byte address absolute                           |
    | 53-60     | 8    | Fixed constant value of 0xb00000000                  |
    | 61-63     | 3    | Swizzling mode:                                      |
    |           |      |   0: No swizzling                                    |
    |           |      |   1: 128-Byte with 32B atomic swizzling              |
    |           |      |   2: 128-Byte swizzling                              |
    |           |      |   4: 64-Byte swizzling                               |
    |           |      |   6: 32-Byte swizzling                               |
    |           |      |   (Values 3, 5 and 7 are invalid)                    |
    +-----------+------+------------------------------------------------------+    
    ```

    Example:
    ```mlir
      %desc = nvvm.tcgen05.mma_smem_desc (%startAddr, %leadingDimOffset, %strideDimOffset,
                                          %baseOffset, %leadingDimMode, %swizzleMode) : (i32, i32, i32, i8, i1, i8) -> i64
    ```
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)
  }];

  let arguments = (ins
      I32:$startAddr,        // Matrix A or B start address (bits 13-0)
      I32:$leadingDimOffset, // Matrix A or B leading dim byte offset (bits 29-16)
      I32:$strideDimOffset,  // Matrix A or B stride  dim byte offset (bits 45-32)
      I8:$baseOffset,        // Matrix A or B base offset (bits 51-49)
      I1:$leadingDimMode,    // Matrix A or B leading dim mode (bit 52)
      I8:$swizzleMode        // Swizzle mode (bits 63-61)
  );

  let results = (outs I64:$res);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type($res)
  }];

  let extraClassDeclaration = [{
    static void createSmemDescriptor(Operation &op, LLVM::ModuleTranslation &mt,
                                     llvm::IRBuilderBase& builder);
  }];

  string llvmBuilder = [{
    NVVM::Tcgen05MmaSmemDescOp::createSmemDescriptor(*op, moduleTranslation, builder);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05 LdSt Shape Attr
//===----------------------------------------------------------------------===//

def Tcgen05LdStShape16x64b: I32EnumAttrCase<"SHAPE_16X64B", 0, "shape_16x64b">;
def Tcgen05LdStShape16x128b: I32EnumAttrCase<"SHAPE_16X128B", 1, "shape_16x128b">;
def Tcgen05LdStShape16x256b: I32EnumAttrCase<"SHAPE_16X256B", 2, "shape_16x256b">;
def Tcgen05LdStShape32x32b: I32EnumAttrCase<"SHAPE_32X32B", 3, "shape_32x32b">;
def Tcgen05LdStShape16x32bx2: I32EnumAttrCase<"SHAPE_16X32BX2", 4, "shape_16x32bx2">;

def Tcgen05LdStShape: I32EnumAttr<
  "Tcgen05LdStShape",
  "",
  [Tcgen05LdStShape16x64b, Tcgen05LdStShape16x128b, Tcgen05LdStShape16x256b,
   Tcgen05LdStShape32x32b, Tcgen05LdStShape16x32bx2]
> {
  let cppNamespace = "::mlir::NVVM";
  let genSpecializedAttr = 0;
}

def Tcgen05LdStShapeAttr: EnumAttr<NVVM_Dialect, Tcgen05LdStShape, "tcgen05_ldst_shape"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.ld Op
//===----------------------------------------------------------------------===//

def NVVM_Tcgen05LdOp : NVVM_Op<"tcgen05.ld", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "tensor memory load instructions";
  let arguments = (ins
    // Attributes
    UnitAttr:$pack,
    Tcgen05LdStShapeAttr:$shape,
    // Arguments
    LLVM_PointerTensor:$tmemAddr,
    Optional<I64>:$offset
  );

  let results = (outs AnyTypeOf<[I32, VectorOfLengthAndType<
                                  [2, 4, 8, 16, 32, 64, 128], [I32]>]>:$res);

  let assemblyFormat = [{
    $tmemAddr (`,` $offset^)? (`pack` $pack^)? attr-dict `:` type($res)
  }];

  let description = [{
    Instruction `tcgen05.ld` asynchronously loads data from the Tensor Memory at
    the location specified by the 32-bit address operand `tmemAddr` into the
    destination register `res`, collectively across all threads of the warps.

    The `shape` and the `num` attribute together determines the total
    dimension of the data which is loaded from the Tensor Memory. The `shape`
    attribute indicates the base dimension of data to be accessed as described
    in the Data Movement Shape. The `num` attribute indicates the repeat
    factor on the base dimension resulting in the total dimension of the data
    that is accessed.

    The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
    `16x32b`. The base address of the first access is specified by `tmemAddr`
    and the base address of the second access is specified by
    `tmemAddr + offset`, where `offset` is an immediate argument.

    The unit attribute `pack` can be used to pack two 16-bit
    elements from adjacent columns into a single 32-bit element during the load.

    The following table describes the size of the vector for various combinations
    of `num` and `shape` attributes:
    ```
    |=====================================================================|
    | num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  |
    |=====================================================================|
    | x1             |          1                 |    2       |    4     |
    | x2             |          2                 |    4       |    8     |
    | x4             |          4                 |    8       |    16    |
    | x8             |          8                 |    16      |    32    |
    | x16            |          16                |    32      |    64    |
    | x32            |          32                |    64      |    128   |
    | x64            |          64                |    128     |    NA    |
    | x128           |          128               |    NA      |    NA    |
    |=====================================================================|
    ```

    Example:
    ```mlir
      nvvm.tcgen05.ld %tmemAddr, %offset pack {
        shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
      } : <2xi32>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
  }];

  let hasVerifier = 1;

  string llvmBuilder = [{
    llvm::LLVMContext &Context = moduleTranslation.getLLVMContext();
    auto Pack = llvm::ConstantInt::get(Context, llvm::APInt(1, $pack));

    unsigned num = $_resultType->isVectorTy()
                       ? llvm::cast<llvm::VectorType>($_resultType)
                             ->getElementCount()
                             .getFixedValue()
                       : 1;

    auto ID = getTcgen05LdIntrinsicID($shape, num);
    if (ID == llvm::Intrinsic::not_intrinsic)
      llvm::report_fatal_error("unknow intrinsic signature for tcgen05.ld");

    if ($offset)
      $res = createIntrinsicCall(builder, ID, {$tmemAddr, $offset, Pack});
    else
      $res = createIntrinsicCall(builder, ID, {$tmemAddr, Pack});
  }];
}

//===----------------------------------------------------------------------===//
// NVVM tcgen05.st Op
//===----------------------------------------------------------------------===//

def NVVM_Tcgen05StOp : NVVM_Op<"tcgen05.st", [NVVMRequiresSMa<[100, 101]>]> {
  let summary = "tensor memory store instructions";
  let arguments = (ins
    // Attributes
    UnitAttr:$unpack,
    Tcgen05LdStShapeAttr:$shape,
    // Arguments
    LLVM_PointerTensor:$tmemAddr,
    AnyTypeOf<[I32, VectorOfLengthAndType<
                      [2, 4, 8, 16, 32, 64, 128], [I32]>]>:$val,
    Optional<I64>:$offset
  );

  let assemblyFormat = [{
    $tmemAddr `,` $val (`,` $offset^)? (`unpack` $unpack^)? attr-dict `:` type($val)
  }];

  let description = [{
    Instruction `tcgen05.st` asynchronously stores data from the source register `r`
    into the Tensor Memory at the location specified by the 32-bit address operand
    `tmemAddr`, collectively across all threads of the warps.

    The `shape` and the `num` attribute together determines the total dimension of
    the data which is stored to the Tensor Memory. The `shape` indicates the base
    dimension of data to be accessed. The `num` attribute indicates the repeat
    factor on the base dimension resulting in the total dimension of the data that
    is accessed.

    The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
    `16x32b`. The base address of the first access is specified by `tmemAddr`
    and the base address of the second access is specified by
    `tmemAddr + offset`, where `offset` is an immediate argument.

    The unit attribute `unpack` can be used to unpack a 32-bit element
    in the register into two 16-bit elements and store them in adjacent columns.

    The following table describes the size of the vector for various combinations
    of `num` and `shape` attributes:
    ```
    |=====================================================================|
    | num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  |
    |=====================================================================|
    | x1             |          1                 |    2       |    4     |
    | x2             |          2                 |    4       |    8     |
    | x4             |          4                 |    8       |    16    |
    | x8             |          8                 |    16      |    32    |
    | x16            |          16                |    32      |    64    |
    | x32            |          32                |    64      |    128   |
    | x64            |          64                |    128     |    NA    |
    | x128           |          128               |    NA      |    NA    |
    |=====================================================================|
    ```

    Example:
    ```mlir
      nvvm.tcgen05.st %tmemAddr, %val, %offset unpack {
        shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
      } : <2xi32>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
  }];

  string llvmBuilder = [{
    llvm::LLVMContext &Context = moduleTranslation.getLLVMContext();
    auto Unpack = llvm::ConstantInt::get(Context, llvm::APInt(1, $unpack));

    auto valTy = $val->getType();
    uint32_t num = valTy->isVectorTy() ? llvm::cast<llvm::VectorType>(valTy)
                                             ->getElementCount()
                                             .getFixedValue()
                                       : 1;

    auto ID = getTcgen05StIntrinsicID($shape, num);
    if (ID == llvm::Intrinsic::not_intrinsic)
      llvm::report_fatal_error("unknow intrinsic signature for tcgen05.st");

    if ($offset)
      createIntrinsicCall(builder, ID, {$tmemAddr, $offset, $val, Unpack});
    else
      createIntrinsicCall(builder, ID, {$tmemAddr, $val, Unpack});
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NVVM dot.accumulate Ops
//===----------------------------------------------------------------------===//

def DotAccumulateUnsigned : I32EnumAttrCase<"UNSIGNED", 0, "unsigned">;
def DotAccumulateSigned : I32EnumAttrCase<"SIGNED", 1, "signed">;

def DotAccumulateType : I32EnumAttr<"DotAccumulateType",
                              "NVVM DotAccumulateType",
                              [DotAccumulateSigned, DotAccumulateUnsigned]> {
  let cppNamespace = "::mlir::NVVM";
  let genSpecializedAttr = 0;
}

def DotAccumulateTypeAttr : EnumAttr<NVVM_Dialect, DotAccumulateType, "dot_accumulate_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def NVVM_DotAccumulate4WayOp : NVVM_Op<"dot.accumulate.4way"> {
  let summary = "Four-way byte dot product-accumulate instruction";
  let description = [{
    Performs a four-way byte dot-product which is accumulated in a 32-bit
    result.
    Operand `a` and `b` are vectors of 4 bytes between which the dot product is 
    computed.

    The `a_type` and `b_type` attributes specify the type of the elements in `a`
    and `b` respectively.
    If `a_type` or `b_type` is `signed`, then the elements in the corresponding 
    vector are sign-extended to 32-bit before the dot product is computed.
    If `a_type` or `b_type` is `unsigned`, then the elements in the 
    corresponding vector are zero-extended to 32-bit instead.

    Operand `c` is a 32-bit integer to which the result is accumulated. It is
    treated as holding a signed integer if any of `a_type` or `b_type` is `s8`.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp4a)
  }];
  
  let arguments = (ins
    VectorOfLengthAndType<[4], [I8]>:$a,
    DotAccumulateTypeAttr:$a_type,
    VectorOfLengthAndType<[4], [I8]>:$b,
    DotAccumulateTypeAttr:$b_type,
    I32:$c
  );

  let results = (outs I32:$res);

  let assemblyFormat = "$a $a_type `,` $b $b_type `,` $c attr-dict `:` type($a) `,` type($b)";
  
  let extraClassDeclaration = [{
    static mlir::NVVM::IDArgPair
    getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase &builder);
  }];

  string llvmBuilder = [{
    auto [id, args] = NVVM::DotAccumulate4WayOp::getIntrinsicIDAndArgs(
                        *op, moduleTranslation, builder);
    $res = createIntrinsicCall(builder, id, args);
  }];
}

def NVVM_DotAccumulate2WayOp : NVVM_Op<"dot.accumulate.2way"> {
  let summary = "Two-way 16-bit to 8-bit dot product-accumulate instruction";
  let description = [{
    Performs a two-way 16-bit to 8-bit dot-product which is accumulated in a 
    32-bit result.
    Operand `a` is a vector of two 16-bit elements and operand `b` a vector 
    of four 8-bit elements between which the dot product is computed.

    The `a_type` and `b_type` attributes specify the type of the elements in `a`
    and `b` respectively.
    If `a_type` or `b_type` is `s`, then the elements in the corresponding 
    vector are sign-extended to 32-bit before the dot product is computed.
    If `a_type` or `b_type` is `u`, then the elements in the corresponding 
    vector are zero-extended to 32-bit instead.

    The `b_hi` boolean attribute specifies which two bytes of `b` are used for 
    the dot product. If `b_hi` is true, then the dot product is computed 
    between  `a` and elements at indices 2 and 3 of `b`. If `b_hi` is false, 
    then the dot product is computed between `a` and elements at indices 0 and 
    1 of `b`.

    Operand `c` is a 32-bit integer to which the result is accumulated. It is
    treated as holding a signed integer if any of `a_type` or `b_type` is 
    signed.
    
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp2a)
  }];

  let arguments = (ins
    VectorOfLengthAndType<[2], [I16]>:$a,
    DotAccumulateTypeAttr:$a_type,
    VectorOfLengthAndType<[4], [I8]>:$b,
    DotAccumulateTypeAttr:$b_type,
    I32:$c,
    BoolAttr:$b_hi
  );

  let results = (outs I32:$res);

  let assemblyFormat = "$a $a_type `,` $b $b_type `,` $c attr-dict `:` type($a) `,` type($b)";
  
  let extraClassDeclaration = [{
    static mlir::NVVM::IDArgPair
    getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                          llvm::IRBuilderBase &builder);
  }];
  
  string llvmBuilder = [{
    auto [id, args] = NVVM::DotAccumulate2WayOp::getIntrinsicIDAndArgs(
                        *op, moduleTranslation, builder);
    $res = createIntrinsicCall(builder, id, args);
  }];
}

//===----------------------------------------------------------------------===//
// NVVM target attribute.
//===----------------------------------------------------------------------===//

def NVVM_TargetAttr : NVVM_Attr<"NVVMTarget", "target", 
  [DeclareAttrInterfaceMethods<GPUTargetAttrVerifyInterface>]> {
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
    OptionalParameter<"ArrayAttr", "Files to link to the LLVM module.">:$link,
    DefaultValuedParameter<"bool", "true", "Perform SM version check on Ops.">:$verifyTarget
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features, $flags, $link, $verifyTarget)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"nvptx64-nvidia-cuda\"">:$triple,
                     CArg<"StringRef", "\"sm_50\"">:$chip,
                     CArg<"StringRef", "\"+ptx60\"">:$features,
                     CArg<"DictionaryAttr", "nullptr">:$targetFlags,
                     CArg<"ArrayAttr", "nullptr">:$linkFiles,
                     CArg<"bool", "true">:$verifyTarget), [{
      return Base::get($_ctxt, optLevel, triple, chip, features, targetFlags, linkFiles, verifyTarget);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool hasFlag(StringRef flag) const;
    bool hasFastMath() const;
    bool hasFtz() const;
    bool hasCmdOptions() const;
    std::optional<mlir::NamedAttribute> getCmdOptions() const;
    LogicalResult verifyTarget(Operation *gpuModule);
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
    bool $cppClass::hasCmdOptions() const {
      return hasFlag("ptxas-cmd-options");
    }
    std::optional<mlir::NamedAttribute> $cppClass::getCmdOptions() const {
      return getFlags().getNamed("ptxas-cmd-options");
    }
  }];
}

#endif // NVVMIR_OPS
