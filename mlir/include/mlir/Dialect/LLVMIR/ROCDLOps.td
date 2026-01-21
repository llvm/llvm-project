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
  let summary = "Dialect for wrapping LLVM AMDGPU backend intrinsics and attributes";
  let hasOperationAttrVerify = 1;

  let description = [{
    The ROCDL dialect, like the other platform-specific LLVM dialects, serves
    as the location of wrappers around the AMD-specific intrinsics and attributes
    in LLVM.

    This dialect, like other GPU lowering targets, also contains the infrastructure
    used by the built-in compilation/offloading framework to compile AMD-specific
    LLVM IR into binaries.

    # Dialect inclusion criteria and guidelines

    The operations in this dialect are 1:1 wrappers around their corresponding
    LLVM intrinsics. Operations that do not correspond to intrinsics should not
    be placed in this dialect.

    The definition of a ROCDL op should match its LLVM counterpart. If the
    argument and result types are fixed, they should be specified as type
    constraints, including by overriding the default variadic type on LLVM
    intrinsics by doing a `let results` in the operation definition.

    LLVM attributes do not need to be replicated exactly if it wouldn't be
    easy to do so, but pure operations and ones that read/write memory should
    be annotated as such.

    While LLVM intrinsics currently don't allow constraining the values an
    `any_type` can take, it is acceptable (but not required) to impose such
    constraints if they are known.

    When an LLVM intrinsic uses an `immarg`, this corresponds to an attribute
    in MLIR.

    Human-readable assembly formats (those that, for example, explicitly indicate
    parameter names) may be used, and are encouraged for intrinsics that have
    complex argument schemes and don't have any higher-level wrapper (such as
    in the `amdgpu` dialect).

    While not all existing operations follow this convention, new operations should
    generally provide argument and result types except in cases where they are
    clearly redundant (such as with operations like `rocdl.fmed3`, which doesn't
    need to reiterate the single type at issue multiple times). This convention
    enhances the readability of low-level IR and prevents programmers from needing
    to find non-local type information.

    Dialect-defined discardable attributes (any attribute starting with `rocdl.`
    that has special handling) need to correspond to AMD-specific attributes, metadata,
    or other entities (such as calling conventions) in LLVM, or be needed for
    GPU compilation management. Outside of the compilation infrastructure,
    dialect-specific enums or attributes are extmelely unlikely to be needed
    and should be avoided.

    Operation documentation should specify when the operation was introduced
    (if relevant) and include usage examples. Operations should have
    parser/printer tests in `mlir/test/Dialect/LLVMIR/rocdl.mlir` and
    lowering tests in `mlir/test/Target/LLVMIR/rocdl.mlir`.

    # General documentation (What does this op do?)

    While rocdl ops sometimes carry their own documentation, there is no
    expectation that such documentation will exist (or be kept up to date).

    Since ROCDL operations correspond to LLVM intrinsics, the semantics and
    behavior of these operations can be determined by investigating the
    documentation for the corresponding intrinsic. This documentation
    can be found in
    - `llvm/docs/AMDGPUUsage.rst` and
    - The comments of `llvm/include/llvm/IR/IntrinsicsAMDGPU.td`, which
      is where details of the meaning of certain bitfields or of how an
      intrinsic corresponds to hardware instructions are most likely to
      be found.

    Since many intrinsics are themselves minimal wrappers around hardware
    instructions, these documentation sources often do not repeat hardware
    documentation. If an intrinsic appears undocumented, information about
    its behavior will often be available in published ISA descriptions or
    (sometimes known as shader programming guides).

    If an operation doesn't provide usage examples, it is likely that they
    can be found in `mlir/test/Dialect/LLVMIR/rocdl.td`.
  }];

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
  int requiresAccessGroup = 0, int requiresAliasAnalysis = 0,
  int requiresArgAndResultAttrs = 0,
  list<int> immArgPositions = [],
  list<string> immArgAttrNames = []> :
  LLVM_IntrOpBase<ROCDL_Dialect,  mnemonic,
    "amdgcn_" # !subst(".", "_", mnemonic), overloadedResults,
    overloadedOperands, traits, numResults, requiresAccessGroup,
    requiresAliasAnalysis, 0, requiresArgAndResultAttrs, 0,
    immArgPositions, immArgAttrNames>;

// Subclass to save typing and ease readibility when there aren't overloaded
// operands or memory accesses.
class ROCDL_ConcreteNonMemIntrOp<string mnemonic, list<Trait> traits,
    int numResults, list<int> immArgPositions = [],
    list<string> immArgNames = []>
  : ROCDL_IntrOp<mnemonic, [], [], traits, numResults, 0, 0, 0,
      immArgPositions, immArgNames>;
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

// TODO(krzysz00): This should be a lowering pattern, not an op.
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
// ROCDL vector types definitions
//===----------------------------------------------------------------------===//

class ROCDL_ConcreteVector<Type elem, int length> :
  FixedVectorOfLengthAndType<[length], [elem]>,
  BuildableType<
    "::mlir::VectorType::get({" # length # "} ,"
      # elem.builderCall # ")">;

def ROCDL_V2I16Type : ROCDL_ConcreteVector<I16, 2>;
def ROCDL_V2F16Type : ROCDL_ConcreteVector<F16, 2>;
def ROCDL_V2I32Type : ROCDL_ConcreteVector<I32, 2>;
def ROCDL_V2BF16Type : ROCDL_ConcreteVector<BF16, 2>;
def ROCDL_V2F32Type : ROCDL_ConcreteVector<F32, 2>;
def ROCDL_V3I32Type : ROCDL_ConcreteVector<I32, 3>;
def ROCDL_V4I32Type : ROCDL_ConcreteVector<I32, 4>;
def ROCDL_V6I32Type : ROCDL_ConcreteVector<I32, 6>;
def ROCDL_V8I32Type : ROCDL_ConcreteVector<I32, 8>;
def ROCDL_V8BF16Type : ROCDL_ConcreteVector<BF16, 8>;
def ROCDL_V8F16Type : ROCDL_ConcreteVector<F16, 8>;
def ROCDL_V8F32Type : ROCDL_ConcreteVector<F32, 8>;
def ROCDL_V16BF16Type : ROCDL_ConcreteVector<BF16, 16>;
def ROCDL_V16F16Type : ROCDL_ConcreteVector<F16, 16>;
def ROCDL_V16F32Type : ROCDL_ConcreteVector<F32, 16>;
def ROCDL_V32F16Type : ROCDL_ConcreteVector<F16, 32>;
def ROCDL_V32BF16Type : ROCDL_ConcreteVector<BF16, 32>;
def ROCDL_V32F32Type : ROCDL_ConcreteVector<F32, 32>;

//===----------------------------------------------------------------------===//
// Wave-level primitives
//===----------------------------------------------------------------------===//

class ROCDL_MbcntOp<string mnemonic> :
    ROCDL_IntrOp<"mbcnt." # mnemonic, [], [], [Pure], 1,
    0, 0, /*requiresArgAndResultAttrs=*/1> {
  dag args = (ins I32:$in0, I32:$in1);
  let arguments = !con(args, baseArgs);
  let results = (outs I32:$res);
  let assemblyFormat = [{
    $in0 `,` $in1  attr-dict `:` `(` type($in0) `,` type($in1) `)` `->` type($res)
   }];
}

def ROCDL_MbcntLoOp : ROCDL_MbcntOp<"lo">;
def ROCDL_MbcntHiOp : ROCDL_MbcntOp<"hi">;

def ROCDL_DsSwizzleOp : ROCDL_ConcreteNonMemIntrOp<"ds_swizzle", [], 1>,
  Arguments<(ins I32:$src,
                 I32:$offset)> {
  let results = (outs I32:$res);
  let assemblyFormat = [{
    $src `,` $offset  attr-dict `:` `(` type($src) `,` type($offset) `)` `->` type($res)
   }];
}

def ROCDL_DsBpermuteOp : ROCDL_ConcreteNonMemIntrOp<"ds_bpermute", [], 1>,
  Arguments<(ins I32:$index,
                 I32:$src)> {
  let results = (outs I32:$res);
  let assemblyFormat = [{
    $index `,` $src  attr-dict `:` `(` type($index) `,` type($src) `)` `->` type($res)
   }];
}

def ROCDL_BallotOp :
  ROCDL_IntrOp<"ballot", [0], [], [], 1>,
  Arguments<(ins I1:$pred)> {
  let summary = "Vote across thread group";

  let description = [{
      Ballot provides a bit mask containing the 1-bit predicate value from each lane.
      The nth bit of the result contains the 1 bit contributed by the nth warp lane.
  }];

  let assemblyFormat = "$pred attr-dict `:` type($res)";
}

def ROCDL_ReadfirstlaneOp : ROCDL_IntrOp<"readfirstlane", [], [0], [AllTypesMatch<["res", "src"]>], 1>,
  Arguments<(ins LLVM_Type:$src)> {
  let results = (outs LLVM_Type:$res);
  let summary = "Get the value in first active lane.";

  let description = [{
    Returns the value in the lowest active lane of the input operand.
  }];

  let assemblyFormat = [{
    $src attr-dict `:` type($res)
  }];
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
// Thread, Block and Cluster index
//===----------------------------------------------------------------------===//

def ROCDL_ThreadIdXOp : ROCDL_SpecialIdRegisterOp<"workitem.id.x">;
def ROCDL_ThreadIdYOp : ROCDL_SpecialIdRegisterOp<"workitem.id.y">;
def ROCDL_ThreadIdZOp : ROCDL_SpecialIdRegisterOp<"workitem.id.z">;

def ROCDL_BlockIdXOp : ROCDL_SpecialIdRegisterOp<"workgroup.id.x">;
def ROCDL_BlockIdYOp : ROCDL_SpecialIdRegisterOp<"workgroup.id.y">;
def ROCDL_BlockIdZOp : ROCDL_SpecialIdRegisterOp<"workgroup.id.z">;

def ROCDL_ClusterIdXOp : ROCDL_SpecialIdRegisterOp<"cluster.id.x">;
def ROCDL_ClusterIdYOp : ROCDL_SpecialIdRegisterOp<"cluster.id.y">;
def ROCDL_ClusterIdZOp : ROCDL_SpecialIdRegisterOp<"cluster.id.z">;

def ROCDL_WaveId : ROCDL_SpecialIdRegisterOp<"wave.id">;
def ROCDL_WavefrontSizeOp : ROCDL_SpecialIdRegisterOp<"wavefrontsize">;

//===----------------------------------------------------------------------===//
// Thread range and Block range
//===----------------------------------------------------------------------===//

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
//===----------------------------------------------------------------------===//

// Emits the waintcnt instruction. The bitfield's semantics depend
// on the target chipset
def ROCDL_SWaitcntOp : ROCDL_ConcreteNonMemIntrOp<"s.waitcnt", [], 0, [0], ["bitfield"]>,
  Arguments<(ins I32Attr:$bitfield)> {
  let assemblyFormat = "attr-dict $bitfield";
}

def ROCDL_SSleepOp : ROCDL_ConcreteNonMemIntrOp<"s.sleep", [], 0, [0], ["count"]>,
  Arguments<(ins I32Attr:$count)> {
  let assemblyFormat = "attr-dict $count";
}

def ROCDL_SNopOp : ROCDL_ConcreteNonMemIntrOp<"s.nop", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let assemblyFormat = "attr-dict $count";
}

def ROCDL_SBarrierOp : ROCDL_ConcreteNonMemIntrOp<"s.barrier", [], 0> {
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
  let description = [{
    An operation with the same expansion as HIP's __synchthreads();

    **DEPRECATION NOTICE**: Use `gpu.barrier`, which will expand to these
    operations, instead.
  }];
  let assemblyFormat = "attr-dict";
}

def ROCDLGlobalBuffer : LLVM_PointerInAddressSpace<1>;
def ROCDLBufferLDS : LLVM_PointerInAddressSpace<3>;

def ROCDL_BarrierInitOp : ROCDL_IntrOp<"s.barrier.init", [], [], [], 0, 0, 0, 0, [1], ["memberCnt"]>,
  Arguments<(ins Arg<ROCDLBufferLDS, "", []>:$ptr, I32Attr:$memberCnt)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$ptr `member_cnt` `=` $memberCnt attr-dict `:` qualified(type($ptr))";
}

def ROCDL_BarrierSignalOp : ROCDL_ConcreteNonMemIntrOp<"s.barrier.signal", [], 0, [0], ["id"]>,
  Arguments<(ins I32Attr:$id)> {
  let results = (outs);
  let assemblyFormat = "`id` `=` $id attr-dict";
}

def ROCDL_BarrierSignalVarOp : ROCDL_IntrOp<"s.barrier.signal.var", [], [], [], 0, 0, 0, 0, [1], ["memberCnt"]>,
  Arguments<(ins Arg<ROCDLBufferLDS, "", []>:$ptr, I32Attr:$memberCnt)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$ptr `member_cnt` `=` $memberCnt attr-dict `:` qualified(type($ptr))";
}

def ROCDL_BarrierJoinOp : ROCDL_IntrOp<"s.barrier.join", [], [], [], 0>,
  Arguments<(ins Arg<ROCDLBufferLDS, "", []>:$ptr)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
}

def ROCDL_BarrierLeaveOp : ROCDL_ConcreteNonMemIntrOp<"s.barrier.leave", [], 0, [0], ["id"]>,
  Arguments<(ins I16Attr:$id)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "`id` `=` $id attr-dict";
}

def ROCDL_BarrierWaitOp : ROCDL_ConcreteNonMemIntrOp<"s.barrier.wait", [], 0, [0], ["id"]>,
  Arguments<(ins I16Attr:$id)> {
  let results = (outs);
  let assemblyFormat = "`id` `=` $id attr-dict";
}

def ROCDL_BarrierSignalIsfirstOp : ROCDL_ConcreteNonMemIntrOp<"s.barrier.signal.isfirst", [], 1, [0], ["id"]>,
  Arguments<(ins I32Attr:$id)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs I1:$res);
  let assemblyFormat = "`id` `=` $id attr-dict `->` type($res)";
}

def ROCDL_GetBarrierStateOp : ROCDL_ConcreteNonMemIntrOp<"s.get.barrier.state", [], 1, [0], ["id"]>,
  Arguments<(ins I32Attr:$id)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs I32:$res);
  let assemblyFormat = "`id` `=` $id attr-dict `->` type($res)";
}

def ROCDL_GetNamedBarrierStateOp : ROCDL_ConcreteNonMemIntrOp<"s.get.named.barrier.state", [], 1, [], []>,
  Arguments<(ins Arg<ROCDLBufferLDS, "", []>:$ptr)> {
  let description = [{
    Available on gfx1250+.
  }];
  let results = (outs I32:$res);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr)) `->` type($res)";
}

def ROCDL_WakeupBarrierOp : ROCDL_ConcreteNonMemIntrOp<"s.wakeup.barrier", [], 0, [], []>,
  Arguments<(ins Arg<ROCDLBufferLDS, "", []>:$ptr)> {
  let description = [{
    Wakes up waves associated with a given named barrier. Note, This op does not release waves waiting
    at the barrier. It just signal other waves in the same work-group waiting on the indicated named barrier
    to wake up.
    Available on gfx1250+.
  }];
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
}

def ROCDL_WaitDscntOp: ROCDL_ConcreteNonMemIntrOp<"s.wait.dscnt", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let summary = "Wait until DSCNT is less than or equal to `count`";
  let description = [{
      Wait for the counter specified to be less-than or equal-to the `count`
      before continuing.

      Available on gfx12+.
  }];
  let results = (outs);
  let assemblyFormat = "$count attr-dict";
}

def ROCDL_WaitLoadcntOp: ROCDL_ConcreteNonMemIntrOp<"s.wait.loadcnt", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let summary = "Wait until LOADCNT is less than or equal to `count`";
  let description = [{
      Wait for the counter specified to be less-than or equal-to the `count`
      before continuing.

      Available on gfx12+.
  }];
  let results = (outs);
  let assemblyFormat = "$count attr-dict";
}

def ROCDL_WaitStorecntOp: ROCDL_ConcreteNonMemIntrOp<"s.wait.storecnt", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let summary = "Wait until STORECNT is less than or equal to `count`";
  let description = [{
      Wait for the counter specified to be less-than or equal-to the `count`
      before continuing.

      Available on gfx12+.
  }];
  let results = (outs);
  let assemblyFormat = "$count attr-dict";
}

def ROCDL_WaitExpcntOp: ROCDL_ConcreteNonMemIntrOp<"s.wait.expcnt", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let summary = "Wait until EXPCNT is less than or equal to `count`";
  let description = [{
      Wait for the counter specified to be less-than or equal-to the `count`
      before continuing.

      Available on gfx12+.
  }];
  let results = (outs);
  let assemblyFormat = "$count attr-dict";
}

def ROCDL_WaitAsynccntOp: ROCDL_ConcreteNonMemIntrOp<"s.wait.asynccnt", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let summary = "Wait until ASYNCCNT is less than or equal to `count`";
  let description = [{
      Wait for the counter specified to be less-than or equal-to the `count`
      before continuing.

      Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$count attr-dict";
}

def ROCDL_WaitTensorcntOp: ROCDL_ConcreteNonMemIntrOp<"s.wait.tensorcnt", [], 0, [0], ["count"]>,
  Arguments<(ins I16Attr:$count)> {
  let summary = "Wait until TENSORCNT is less than or equal to `count`";
  let description = [{
      Wait for the counter specified to be less-than or equal-to the `count`
      before continuing.

      Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$count attr-dict";
}

def ROCDL_SetPrioOp : ROCDL_ConcreteNonMemIntrOp<"s.setprio", [], 0, [0], ["priority"]>,
  Arguments<(ins I16Attr:$priority)> {
  let assemblyFormat = "$priority attr-dict";
}

def ROCDL_SchedBarrier : ROCDL_ConcreteNonMemIntrOp<"sched.barrier", [], 0, [0],["mask"]>,
  Arguments<(ins I32Attr:$mask)> {
  let assemblyFormat = "$mask attr-dict";
}

def ROCDL_SchedGroupBarrier
  : ROCDL_ConcreteNonMemIntrOp<"sched.group.barrier", [], 0,
      [0, 1, 2], ["mask", "size", "groupId"]>,
    Arguments<(ins I32Attr:$mask, I32Attr:$size, I32Attr:$groupId)> {
  let assemblyFormat = "$mask `,` $size `,` $groupId attr-dict";
}

def ROCDL_IglpOpt : ROCDL_ConcreteNonMemIntrOp<"iglp.opt", [], 0, [0], ["variant"]>,
  Arguments<(ins I32Attr:$variant)> {
  let assemblyFormat = "$variant attr-dict";
}

//===---------------------------------------------------------------------===//
// Xdlops intrinsics

class ROCDL_Mfma_IntrOp<string mnemonic, Type ABType, Type CDType> :
  ROCDL_IntrOp<mnemonic, [], [], [], 1, 0, 0, 0, [3, 4, 5], ["cbsz", "abid", "blgp"]>,
  Arguments<(ins
             ABType:$a,
             ABType:$b,
             CDType:$c,
             I32Attr:$cbsz,
             I32Attr:$abid,
             I32Attr:$blgp)> {
  let results = (outs CDType:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $cbsz `,` $abid `,` $blgp attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_Mfma_Scale_IntrOp<string mnemonic, Type AB, Type CD> :
  ROCDL_IntrOp<mnemonic, [], [0, 1], [], 1, 0, 0, 0, [3, 4, 5, 7], ["cbsz", "blgp", "opselA", "opselB"]>,
  Arguments<(ins
             LLVM_VectorOf<AB>:$a,
             LLVM_VectorOf<AB>:$b,
             LLVM_VectorOf<CD>:$c,
             I32Attr:$cbsz,
             I32Attr:$blgp,
             I32Attr:$opselA,
             I32:$scaleA,
             I32Attr:$opselB,
             I32:$scaleB)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $cbsz `,` $blgp `,` $opselA `,` $scaleA `,` $opselB `,` $scaleB attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_Smfmac_IntrOp<string mnemonic, Type AType, Type BType, Type CDType> :
  ROCDL_IntrOp<mnemonic, [], [], [], 1, 0, 0, 0, [4, 5], ["cbsz", "abid"]>,
  Arguments<(ins
             AType:$a,
             BType:$b,
             CDType:$c,
             I32:$index,
             I32Attr:$cbsz,
             I32Attr:$abid)> {
  let results = (outs CDType:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $index `,` $cbsz `,` $abid attr-dict `:` functional-type(operands, $res)
  }];
}

// Available on all CDNA.
def ROCDL_mfma_f32_32x32x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x1f32", /*Type AB=*/F32, /*Type CD=*/ROCDL_ConcreteVector<F32, 32>>;
def ROCDL_mfma_f32_16x16x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x1f32", F32, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_4x4x1f32 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x1f32", F32, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x2f32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x2f32", F32, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_16x16x4f32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4f32", F32, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F32, 32>>;
def ROCDL_mfma_f32_16x16x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_4x4x4f16 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x4f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x8f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x8f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_16x16x16f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x16f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_i32_32x32x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x4i8", I32, ROCDL_ConcreteVector<I32, 32>>;
def ROCDL_mfma_i32_16x16x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x4i8", I32, ROCDL_ConcreteVector<I32, 16>>;
def ROCDL_mfma_i32_4x4x4i8 : ROCDL_Mfma_IntrOp<"mfma.i32.4x4x4i8", I32, ROCDL_ConcreteVector<I32, 4>>;
def ROCDL_mfma_i32_32x32x8i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x8i8", I32, ROCDL_ConcreteVector<I32, 16>>;
def ROCDL_mfma_i32_16x16x16i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x16i8", I32, ROCDL_ConcreteVector<I32, 4>>;
def ROCDL_mfma_f32_32x32x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x2bf16", ROCDL_ConcreteVector<I16, 2>, ROCDL_ConcreteVector<F32, 32>>;
def ROCDL_mfma_f32_16x16x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x2bf16", ROCDL_ConcreteVector<I16, 2>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_4x4x2bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x2bf16", ROCDL_ConcreteVector<I16, 2>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x4bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4bf16", ROCDL_ConcreteVector<I16, 2>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_16x16x8bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x8bf16", ROCDL_ConcreteVector<I16, 2>, ROCDL_ConcreteVector<F32, 4>>;
// New in gfx90a.
def ROCDL_mfma_f32_32x32x4bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4bf16.1k", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<F32, 32>>;
def ROCDL_mfma_f32_16x16x4bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x4bf16.1k", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_4x4x4bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.4x4x4bf16.1k", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x8bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x8bf16.1k", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_16x16x16bf16_1k : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x16bf16.1k", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<F32, 4>>;
// Note: in gfx94x, unlike in gfx90a, the f64 xdlops use the "blgp" argument as
// a NEG bitfield. See IntrinsicsAMDGPU.td for more info.
def ROCDL_mfma_f64_16x16x4f64 : ROCDL_Mfma_IntrOp<"mfma.f64.16x16x4f64", F64, ROCDL_ConcreteVector<F64, 4>>;
def ROCDL_mfma_f64_4x4x4f64 : ROCDL_Mfma_IntrOp<"mfma.f64.4x4x4f64", F64, F64>;
// New in gfx94x.
def ROCDL_mfma_i32_16x16x32_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x32.i8", I64, ROCDL_ConcreteVector<I32, 4>>;
def ROCDL_mfma_i32_32x32x16_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x16.i8", I64, ROCDL_ConcreteVector<I32, 16>>;
def ROCDL_mfma_f32_16x16x8_xf32 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x8.xf32", ROCDL_ConcreteVector<F32, 2>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x4_xf32 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x4.xf32", ROCDL_ConcreteVector<F32, 2>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_16x16x32_bf8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf8.bf8", I64, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_16x16x32_bf8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf8.fp8", I64, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_16x16x32_fp8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.fp8.bf8", I64, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_16x16x32_fp8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.fp8.fp8", I64, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x16_bf8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf8.bf8", I64, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_32x32x16_bf8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf8.fp8", I64, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_32x32x16_fp8_bf8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.fp8.bf8", I64, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_f32_32x32x16_fp8_fp8 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.fp8.fp8", I64, ROCDL_ConcreteVector<F32, 16>>;
// New in gfx950.
def ROCDL_mfma_f32_16x16x32_bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.bf16", ROCDL_ConcreteVector<BF16, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_i32_16x16x64_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.16x16x64.i8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 4>>;
def ROCDL_mfma_f32_16x16x32_f16 : ROCDL_Mfma_IntrOp<"mfma.f32.16x16x32.f16", ROCDL_ConcreteVector<F16, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_mfma_f32_32x32x16_bf16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.bf16", ROCDL_ConcreteVector<BF16, 8>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_mfma_i32_32x32x32_i8 : ROCDL_Mfma_IntrOp<"mfma.i32.32x32x32.i8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 16>>;
def ROCDL_mfma_f32_32x32x16_f16 : ROCDL_Mfma_IntrOp<"mfma.f32.32x32x16.f16", ROCDL_ConcreteVector<F16, 8>, ROCDL_ConcreteVector<F32, 16>>;

def ROCDL_mfma_scale_f32_16x16x128_f8f6f4 : ROCDL_Mfma_Scale_IntrOp<"mfma.scale.f32.16x16x128.f8f6f4", I32, F32>;
def ROCDL_mfma_scale_f32_32x32x64_f8f6f4 : ROCDL_Mfma_Scale_IntrOp<"mfma.scale.f32.32x32x64.f8f6f4", I32, F32>;

// 2:4 Sparsity ops (GFX94x)
def ROCDL_smfmac_f32_16x16x32_f16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x32.f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F16, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_32x32x16_f16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x16.f16", ROCDL_ConcreteVector<F16, 4>, ROCDL_ConcreteVector<F16, 8>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_16x16x32_bf16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x32.bf16", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<I16, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_32x32x16_bf16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x16.bf16", ROCDL_ConcreteVector<I16, 4>, ROCDL_ConcreteVector<I16, 8>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_i32_16x16x64_i8 : ROCDL_Smfmac_IntrOp<"smfmac.i32.16x16x64.i8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 4>>;
def ROCDL_smfmac_i32_32x32x32_i8 : ROCDL_Smfmac_IntrOp<"smfmac.i32.32x32x32.i8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 16>>;
def ROCDL_smfmac_f32_16x16x64_bf8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x64.bf8.bf8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x64_bf8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x64.bf8.fp8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x64_fp8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x64.fp8.bf8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x64_fp8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x64.fp8.fp8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_32x32x32_bf8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x32.bf8.bf8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x32_bf8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x32.bf8.fp8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x32_fp8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x32.fp8.bf8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x32_fp8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x32.fp8.fp8", ROCDL_ConcreteVector<I32, 2>, ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<F32, 16>>;
// New in gfx950.
def ROCDL_smfmac_f32_16x16x64_bf16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x64.bf16", ROCDL_ConcreteVector<BF16, 8>, ROCDL_ConcreteVector<BF16, 16>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x64_f16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x64.f16", ROCDL_ConcreteVector<F16, 8>, ROCDL_ConcreteVector<F16, 16>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_i32_16x16x128_i8 : ROCDL_Smfmac_IntrOp<"smfmac.i32.16x16x128.i8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<I32, 4>>;
def ROCDL_smfmac_f32_16x16x128_bf8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x128.bf8.bf8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x128_bf8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x128.bf8.fp8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x128_fp8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x128.fp8.bf8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_16x16x128_fp8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.16x16x128.fp8.fp8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 4>>;
def ROCDL_smfmac_f32_32x32x32_bf16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x32.bf16", ROCDL_ConcreteVector<BF16, 8>, ROCDL_ConcreteVector<BF16, 16>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x32_f16 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x32.f16", ROCDL_ConcreteVector<F16, 8>, ROCDL_ConcreteVector<F16, 16>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_i32_32x32x64_i8 : ROCDL_Smfmac_IntrOp<"smfmac.i32.32x32x64.i8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<I32, 16>>;
def ROCDL_smfmac_f32_32x32x64_bf8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x64.bf8.bf8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x64_bf8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x64.bf8.fp8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x64_fp8_bf8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x64.fp8.bf8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 16>>;
def ROCDL_smfmac_f32_32x32x64_fp8_fp8 : ROCDL_Smfmac_IntrOp<"smfmac.f32.32x32x64.fp8.fp8", ROCDL_ConcreteVector<I32, 4>, ROCDL_ConcreteVector<I32, 8>, ROCDL_ConcreteVector<F32, 16>>;


//===---------------------------------------------------------------------===//
// WMMA intrinsics
class ROCDL_WMMA_IntrOp<string mnemonic, Type AB, Type CD> : ROCDL_IntrOp<mnemonic,
    [0], [0], [], 1, 0, 0, 0, [], []>,
  Arguments<(ins
             LLVM_ScalarOrVectorOf<AB>:$a,
             LLVM_ScalarOrVectorOf<AB>:$b,
             LLVM_ScalarOrVectorOf<CD>:$c)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_Opsel_IntrOp<string mnemonic, Type AB, Type CD> : ROCDL_IntrOp<mnemonic,
    [0], [1], [], 1, 0, 0, 0, [3], ["opsel"]>,
  Arguments<(ins
             LLVM_ScalarOrVectorOf<AB>:$a,
             LLVM_ScalarOrVectorOf<AB>:$b,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I1Attr, "0">:$opsel)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_IU_IntrOp<string mnemonic, Type AB, Type CD> : ROCDL_IntrOp<mnemonic,
    [0], [1], [], 1, 0, 0, 0, [0, 2, 5], ["signA", "signB", "clamp"]>,
  Arguments<(ins
             DefaultValuedAttr<I1Attr, "0">:$signA,
             LLVM_ScalarOrVectorOf<AB>:$a,
             DefaultValuedAttr<I1Attr, "0">:$signB,
             LLVM_ScalarOrVectorOf<AB>:$b,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I1Attr, "0">:$clamp)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_ModsAll_Reuse_IntrOp<string mnemonic, Type AB, Type CD> : ROCDL_IntrOp<mnemonic,
    [0], [1], [], 1, 0, 0, 0, [0, 2, 4, 6, 7], ["signA", "signB","modC","reuseA","reuseB"]>,
  Arguments<(ins
             DefaultValuedAttr<I1Attr, "0">:$signA,
             LLVM_ScalarOrVectorOf<AB>:$a,
             DefaultValuedAttr<I1Attr, "0">:$signB,
             LLVM_ScalarOrVectorOf<AB>:$b,
             DefaultValuedAttr<I16Attr, "0">:$modC,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I1Attr, "0">:$reuseA,
             DefaultValuedAttr<I1Attr, "0">:$reuseB)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_ModsC_IntrOp<string mnemonic, Type AB, Type CD> : ROCDL_IntrOp<mnemonic,
    [0], [0], [], 1, 0, 0, 0, [2, 4, 5], ["modC","reuseA","reuseB"]>,
  Arguments<(ins
             LLVM_ScalarOrVectorOf<AB>:$a,
             LLVM_ScalarOrVectorOf<AB>:$b,
             DefaultValuedAttr<I16Attr, "0">:$modC,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I1Attr, "0">:$reuseA,
             DefaultValuedAttr<I1Attr, "0">:$reuseB)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_ModsAll_Diff_IntrOp<string mnemonic, Type AB, Type C, Type D> : ROCDL_IntrOp<mnemonic,
    [0], [1, 5], [], 1, 0, 0, 0, [0, 2, 4, 6, 7], ["signA", "signB","modC","reuseA","reuseB"]>,
  Arguments<(ins
             DefaultValuedAttr<I1Attr, "0">:$signA,
             LLVM_ScalarOrVectorOf<AB>:$a,
             DefaultValuedAttr<I1Attr, "0">:$signB,
             LLVM_ScalarOrVectorOf<AB>:$b,
             DefaultValuedAttr<I16Attr, "0">:$modC,
             LLVM_ScalarOrVectorOf<C>:$c,
             DefaultValuedAttr<I1Attr, "0">:$reuseA,
             DefaultValuedAttr<I1Attr, "0">:$reuseB)> {
  let results = (outs LLVM_ScalarOrVectorOf<D>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_ModsABClamp_IntrOp<string mnemonic, Type AB, Type CD> : ROCDL_IntrOp<mnemonic,
    [0], [1], [], 1, 0, 0, 0, [0, 2, 5, 6, 7], ["signA", "signB", "reuseA","reuseB", "clamp"]>,
  Arguments<(ins
             DefaultValuedAttr<I1Attr, "0">:$signA,
             LLVM_ScalarOrVectorOf<AB>:$a,
             DefaultValuedAttr<I1Attr, "0">:$signB,
             LLVM_ScalarOrVectorOf<AB>:$b,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I1Attr, "0">:$reuseA,
             DefaultValuedAttr<I1Attr, "0">:$reuseB,
             DefaultValuedAttr<I1Attr, "0">:$clamp)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` functional-type(operands, $res)
  }];
}

// Overloaded operands: [1, 3] refers to LLVM intrinsic parameter positions where
// A is at position 1 and B is at position 3 (after format parameters).
class ROCDL_WMMA_Scale_IntrOp<string mnemonic, Type AB, Type CD, Type ScaleExpTy> : ROCDL_IntrOp<mnemonic,
    [0], [1, 3], [], 1, 0, 0, 0, [0, 2, 4, 6, 7, 9, 10, 12, 13],
    ["fmtA", "fmtB", "modC", "scaleAType", "fmtScaleA",
     "scaleBType", "fmtScaleB", "reuseA", "reuseB"]>,
  Arguments<(ins
             DefaultValuedAttr<I32Attr, "0">:$fmtA,
             LLVM_ScalarOrVectorOf<AB>:$a,
             DefaultValuedAttr<I32Attr, "0">:$fmtB,
             LLVM_ScalarOrVectorOf<AB>:$b,
             DefaultValuedAttr<I16Attr, "0">:$modC,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I32Attr, "0">:$scaleAType,
             DefaultValuedAttr<I32Attr, "0">:$fmtScaleA,
             ScaleExpTy:$scaleA,
             DefaultValuedAttr<I32Attr, "0">:$scaleBType,
             DefaultValuedAttr<I32Attr, "0">:$fmtScaleB,
             ScaleExpTy:$scaleB,
             DefaultValuedAttr<I1Attr, "0">:$reuseA,
             DefaultValuedAttr<I1Attr, "0">:$reuseB)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $scaleA `,` $scaleB attr-dict `:` functional-type(operands, $res)
  }];
}

class ROCDL_WMMA_Scale_F4_IntrOp<string mnemonic, Type AB, Type CD, Type ScaleExpTy> : ROCDL_IntrOp<mnemonic,
    [0], [0, 1], [], 1, 0, 0, 0, [2, 4, 5, 7, 8, 10, 11],
    ["modC", "scaleAType", "fmtScaleA",
     "scaleBType", "fmtScaleB", "reuseA", "reuseB"]>,
  Arguments<(ins
             LLVM_ScalarOrVectorOf<AB>:$a,
             LLVM_ScalarOrVectorOf<AB>:$b,
             DefaultValuedAttr<I16Attr, "0">:$modC,
             LLVM_ScalarOrVectorOf<CD>:$c,
             DefaultValuedAttr<I32Attr, "0">:$scaleAType,
             DefaultValuedAttr<I32Attr, "0">:$fmtScaleA,
             ScaleExpTy:$scaleA,
             DefaultValuedAttr<I32Attr, "0">:$scaleBType,
             DefaultValuedAttr<I32Attr, "0">:$fmtScaleB,
             ScaleExpTy:$scaleB,
             DefaultValuedAttr<I1Attr, "0">:$reuseA,
             DefaultValuedAttr<I1Attr, "0">:$reuseB)> {
  let results = (outs LLVM_ScalarOrVectorOf<CD>:$res);
  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $scaleA `,` $scaleB attr-dict `:` functional-type(operands, $res)
  }];
}

// Available from gfx11
def ROCDL_wmma_f32_16x16x16_f16 : ROCDL_WMMA_IntrOp<"wmma.f32.16x16x16.f16", /*Type AB=*/F16, /*Type CD=*/F32>;
def ROCDL_wmma_f32_16x16x16_bf16 : ROCDL_WMMA_IntrOp<"wmma.f32.16x16x16.bf16", AnyInteger, F32>;
def ROCDL_wmma_f16_16x16x16_f16 : ROCDL_WMMA_Opsel_IntrOp<"wmma.f16.16x16x16.f16", F16, F16>;
def ROCDL_wmma_bf16_16x16x16_bf16 : ROCDL_WMMA_Opsel_IntrOp<"wmma.bf16.16x16x16.bf16", AnyInteger, AnyInteger>;
def ROCDL_wmma_i32_16x16x16_iu8 : ROCDL_WMMA_IU_IntrOp<"wmma.i32.16x16x16.iu8", AnyInteger, AnyInteger>;
def ROCDL_wmma_i32_16x16x16_iu4 : ROCDL_WMMA_IU_IntrOp<"wmma.i32.16x16x16.iu4", AnyInteger, AnyInteger>;
// Available from gfx12
def ROCDL_wmma_f32_16x16x16_fp8_fp8 : ROCDL_WMMA_IntrOp<"wmma.f32.16x16x16.fp8_fp8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x16_fp8_bf8 : ROCDL_WMMA_IntrOp<"wmma.f32.16x16x16.fp8_bf8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x16_bf8_bf8 : ROCDL_WMMA_IntrOp<"wmma.f32.16x16x16.bf8_bf8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x16_bf8_fp8 : ROCDL_WMMA_IntrOp<"wmma.f32.16x16x16.bf8_fp8", AnyInteger, F32>;
def ROCDL_wmma_i32_16x16x32_iu4 : ROCDL_WMMA_IU_IntrOp<"wmma.i32.16x16x32.iu4", AnyInteger, AnyInteger>;
// Available from gfx1250
def ROCDL_wmma_f32_16x16x4_f32 : ROCDL_WMMA_ModsAll_Reuse_IntrOp<"wmma.f32.16x16x4.f32", F32, F32>;
def ROCDL_wmma_f32_16x16x32_bf16 : ROCDL_WMMA_ModsAll_Reuse_IntrOp<"wmma.f32.16x16x32.bf16", BF16, F32>;
def ROCDL_wmma_f32_16x16x32_f16 : ROCDL_WMMA_ModsAll_Reuse_IntrOp<"wmma.f32.16x16x32.f16", F16, F32>;
def ROCDL_wmma_f16_16x16x32_f16 : ROCDL_WMMA_ModsAll_Reuse_IntrOp<"wmma.f16.16x16x32.f16", F16, F16>;
def ROCDL_wmma_bf16_16x16x32_bf16 : ROCDL_WMMA_ModsAll_Reuse_IntrOp<"wmma.bf16.16x16x32.bf16", BF16, BF16>;
def ROCDL_wmma_bf16f32_16x16x32_bf16 : ROCDL_WMMA_ModsAll_Diff_IntrOp<"wmma.bf16f32.16x16x32.bf16", BF16, /*Type C=*/F32, /*Type D=*/BF16>;
def ROCDL_wmma_f32_16x16x64_fp8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x64.fp8_fp8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x64_fp8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x64.fp8_bf8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x64_bf8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x64.bf8_fp8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x64_bf8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x64.bf8_bf8", AnyInteger, F32>;
def ROCDL_wmma_f16_16x16x64_fp8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x64.fp8_fp8", AnyInteger, F16>;
def ROCDL_wmma_f16_16x16x64_fp8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x64.fp8_bf8", AnyInteger, F16>;
def ROCDL_wmma_f16_16x16x64_bf8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x64.bf8_fp8", AnyInteger, F16>;
def ROCDL_wmma_f16_16x16x64_bf8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x64.bf8_bf8", AnyInteger, F16>;
def ROCDL_wmma_f32_16x16x128_fp8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x128.fp8_fp8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x128_fp8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x128.fp8_bf8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x128_bf8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x128.bf8_fp8", AnyInteger, F32>;
def ROCDL_wmma_f32_16x16x128_bf8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f32.16x16x128.bf8_bf8", AnyInteger, F32>;
def ROCDL_wmma_f16_16x16x128_fp8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x128.fp8_fp8", AnyInteger, F16>;
def ROCDL_wmma_f16_16x16x128_fp8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x128.fp8_bf8", AnyInteger, F16>;
def ROCDL_wmma_f16_16x16x128_bf8_fp8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x128.bf8_fp8", AnyInteger, F16>;
def ROCDL_wmma_f16_16x16x128_bf8_bf8 : ROCDL_WMMA_ModsC_IntrOp<"wmma.f16.16x16x128.bf8_bf8", AnyInteger, F16>;
def ROCDL_wmma_i32_16x16x64_iu8 : ROCDL_WMMA_ModsABClamp_IntrOp<"wmma.i32.16x16x64.iu8", AnyInteger, AnyInteger>;

// Scaled wmma intrinsics (available from gfx1250)
def ROCDL_wmma_scale_f32_16x16x128_f8f6f4   : ROCDL_WMMA_Scale_IntrOp<"wmma.scale.f32.16x16x128.f8f6f4", AnyInteger, F32, I32>;
def ROCDL_wmma_scale16_f32_16x16x128_f8f6f4 : ROCDL_WMMA_Scale_IntrOp<"wmma.scale16.f32.16x16x128.f8f6f4", AnyInteger, F32, I64>;
def ROCDL_wmma_scale_f32_32x16x128_f4       : ROCDL_WMMA_Scale_F4_IntrOp<"wmma.scale.f32.32x16x128.f4", AnyInteger, F32, I32>;
def ROCDL_wmma_scale16_f32_32x16x128_f4     : ROCDL_WMMA_Scale_F4_IntrOp<"wmma.scale16.f32.32x16x128.f4", AnyInteger, F32, I64>;

//===---------------------------------------------------------------------===//
// LDS transpose intrinsics (available in GFX950)

class ROCDL_LDS_Read_Tr_IntrOp<string mnemonic> :
  ROCDL_IntrOp<mnemonic, [1], [], [], 1, 0, 1> {
  dag args = (ins Arg<ROCDLBufferLDS, "", [MemRead]>:$ptr);
  let arguments = !con(args, baseArgs);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($res)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getPtr()};
    }
  }];
}

def ROCDL_ds_read_tr4_b64 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr4.b64">;
def ROCDL_ds_read_tr8_b64 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr8.b64">;
def ROCDL_ds_read_tr6_b96 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr6.b96">;
def ROCDL_ds_read_tr16_b64 : ROCDL_LDS_Read_Tr_IntrOp<"ds.read.tr16.b64">;



//===---------------------------------------------------------------------===//
// Glb/DS load-transpose intrinsics (available in GFX1250+)

class AddrKind<string n, int s> {
  string name = n;
  int space = s;
}
def GlobalAddrKind : AddrKind<"global", 1>;
def DSAddrKind : AddrKind<"ds", 3>;

class ROCDL_TrLoadOpMeta<AddrKind kind, int inElemBits, int outElemBits> {
  AddrKind addrKind = kind;
  string inBits = !cast<string>(inElemBits);
  string outBits = !cast<string>(outElemBits);
  string inBitsEnc = !if(!eq(addrKind.space, 1),
                     !if(!or(!eq(inElemBits, 8), !eq(inElemBits, 16)), "", inBits), inBits);
  string mnemonic = addrKind.name # ".load.tr" # inBitsEnc # ".b" # outBits;
}

class ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta meta> :
  ROCDL_IntrOp<meta.mnemonic, [1], [], [], 1, 0, 1> {

  dag args = (ins Arg<LLVM_PointerInAddressSpace<meta.addrKind.space>, "", [MemRead]>:$ptr);
  let arguments = !con(args, baseArgs);
  let summary = "Loads and transposes a matrix from " # meta.addrKind.name # " memory to registers (available in gfx1250+).";
  let description = [{
    Load a matrix of }] # meta.inBits # [{-bit data from the }] # meta.addrKind.name # [{ memory,
    transpose data between row-major and column-major order,
    and store the result into a }] # meta.outBits # [{-bit vector register.

    Available in gfx1250+.
  }];
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr)) `->` type($res)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getPtr()};
    }
  }];
}

def ROCDL_GlobalLoadTr4_B64 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<GlobalAddrKind, 4, 64>>;
def ROCDL_GlobalLoadTr8_B64 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<GlobalAddrKind, 8, 64>>;
def ROCDL_GlobalLoadTr6_B96 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<GlobalAddrKind, 6, 96>>;
def ROCDL_GlobalLoadTr8_B128 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<GlobalAddrKind, 16, 128>>;

def ROCDL_DsLoadTr4_B64 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<DSAddrKind, 4, 64>>;
def ROCDL_DsLoadTr8_B64 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<DSAddrKind, 8, 64>>;
def ROCDL_DsLoadTr6_B96 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<DSAddrKind, 6, 96>>;
def ROCDL_DsLoadTr16_B128 : ROCDL_TrLoadOp<ROCDL_TrLoadOpMeta<DSAddrKind, 16, 128>>;

//===---------------------------------------------------------------------===//
// Load to LDS intrinsic (available in GFX9 and GFX10)
//===---------------------------------------------------------------------===//

def ROCDL_LoadToLDSOp :
  ROCDL_IntrOp<"load.to.lds", [], [0], [], 0, 0, 1, 0, [2, 3, 4], ["size", "offset", "aux"]> {
  dag args = (ins Arg<LLVM_AnyPointer, "", [MemRead]>:$globalPtr,
                 Arg<ROCDLBufferLDS, "", [MemWrite]>:$ldsPtr,
                 I32Attr:$size,
                 I32Attr:$offset,
                 I32Attr:$aux);
  let arguments = !con(args, baseArgs);
  let assemblyFormat = [{
    $globalPtr `,`  $ldsPtr `,` $size `,` $offset `,` $aux
    attr-dict `:` type($globalPtr)
  }];
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getGlobalPtr(), getLdsPtr()};
    }
  }];
}

def ROCDL_GlobalLoadLDSOp :
  ROCDL_IntrOp<"global.load.lds", [], [], [], 0, 0, 1, 0, [2, 3, 4], ["size", "offset", "aux"]> {
  dag args = (ins Arg<ROCDLGlobalBuffer, "", [MemRead]>:$globalPtr,
                 Arg<ROCDLBufferLDS, "", [MemWrite]>:$ldsPtr,
                 I32Attr:$size,
                 I32Attr:$offset,
                 I32Attr:$aux);
  let arguments = !con(args, baseArgs);
  let assemblyFormat = [{
    $globalPtr `,`  $ldsPtr `,` $size `,` $offset `,` $aux
    attr-dict
  }];
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getGlobalPtr(), getLdsPtr()};
    }
  }];
}

//===---------------------------------------------------------------------===//
// Async load to LDS intrinsic (available in GFX1250)
//===---------------------------------------------------------------------===//

foreach bitsVal = [8, 32, 64, 128] in {
  defvar bitsStr = "b" # !cast<string>(bitsVal);
  def ROCDL_GlobalLoadAsyncToLDS # !toupper(bitsStr) # Op :
    ROCDL_IntrOp<"global.load.async.to.lds." # bitsStr, [], [], [], 0, 0, 1, 0, [2, 3], ["offset", "aux"]> {
    dag args = (ins Arg<ROCDLGlobalBuffer, "", [MemRead]>:$globalPtr,
                   Arg<ROCDLBufferLDS, "", [MemWrite]>:$ldsPtr,
                   I32Attr:$offset,
                   I32Attr:$aux);
    let arguments = !con(args, baseArgs);
    let assemblyFormat = [{
      $globalPtr `,`  $ldsPtr `,` $offset `,` $aux
      attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
    }];
    let description = [{
      Asynchronously loads }] # !cast<string>(bitsVal) # [{ bits of data from a global memory pointer
      to a Local Data Share (LDS) pointer.

      Available on gfx1250+.
    }];

    let extraClassDefinition = [{
      ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
        return {getGlobalPtr(), getLdsPtr()};
      }
    }];
  }
}

foreach bitsVal = [8, 32, 64, 128] in {
  defvar bitsStr = "b" # !cast<string>(bitsVal);
  def ROCDL_ClusterLoadAsyncToLDS # !toupper(bitsStr) # Op :
    ROCDL_IntrOp<"cluster.load.async.to.lds." # bitsStr, [], [], [], 0, 0, 1, 0, [2, 3], ["offset", "cpol"]> {
    dag args = (ins Arg<ROCDLGlobalBuffer, "", [MemRead]>:$globalPtr,
                   Arg<ROCDLBufferLDS, "", [MemWrite]>:$ldsPtr,
                   I32Attr:$offset,
                   I32Attr:$cpol,
                   I32:$mask);
    let arguments = !con(args, baseArgs);
    let assemblyFormat = [{
      $globalPtr `,`  $ldsPtr `,` $offset `,` $cpol `,` $mask
      attr-dict `:` qualified(type($globalPtr)) `,` qualified(type($ldsPtr))
    }];
    let description = [{
      Broadcasts memory load of }] # !cast<string>(bitsVal) # [{ bits of data for a cluster of workgroups.

      Available on gfx1250+.
    }];

    let extraClassDefinition = [{
      ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
        return {getGlobalPtr(), getLdsPtr()};
      }
    }];
  }
}

//===---------------------------------------------------------------------===//
// Tensor load/store intrinsics (available in GFX1250)
//===---------------------------------------------------------------------===//

// Base class for tensor load/store operations with 4 descriptor groups.
class ROCDL_TensorLDSIntrOp<string mnemonic> :
  ROCDL_IntrOp<mnemonic, [], [], [], 0, 0, 1, 0, [4], ["cachePolicy"]> {
  dag args = (ins ROCDL_V4I32Type:$dgroup0, ROCDL_V8I32Type:$dgroup1,
                  ROCDL_V4I32Type:$dgroup2, ROCDL_V4I32Type:$dgroup3,
                  I32Attr:$cachePolicy);
  let arguments = !con(args, baseArgs);
  let summary = "Base class for ROCDL tensor load/store to/from LDS.";
  let description = [{
    Moves tiles of tensor data between global memory and LDS. The tile is
    described by the $dgroup descriptors. 4 $dgroup descriptors allows for
    movement of up to 5D tensors. $cachePolicy describes the memory scope and an
    indicator of expected data re-use.

    This op is for gfx1250+ architectures.
  }];
  let assemblyFormat = [{
    attr-dict operands `cachepolicy` $cachePolicy `:` type($dgroup0) `,` type($dgroup1)
  }];
  let extraClassDefinition = [{
    SmallVector<Value> $cppClass::getAccessedOperands() {
      return {getDgroup0(), getDgroup1(), getDgroup2(), getDgroup3()};
    }
  }];
}

// Base class for tensor load/store operations with 2 descriptor groups
// (D2 variant).
class ROCDL_TensorLDSIntrD2Op<string mnemonic> :
  ROCDL_IntrOp<mnemonic, [], [], [], 0, 0, 1, 0, [2], ["cachePolicy"]> {
  dag args = (ins ROCDL_V4I32Type:$dgroup0, ROCDL_V8I32Type:$dgroup1,
                  I32Attr:$cachePolicy);
  let arguments = !con(args, baseArgs);
  let summary = "Base class for ROCDL tensor load/store to/from LDS (D2 variant).";
  let description = [{
    Moves tiles of tensor data between global memory and LDS. The tile is
    described by the $dgroup descriptors. 2 $dgroup descriptors allows for
    movement of up to 2D tensors. $cachePolicy describes the memory scope and an
    indicator of expected data re-use.

    This op is for gfx1250+ architectures.
  }];
  let assemblyFormat = [{
    attr-dict operands `cachepolicy` $cachePolicy `:` type($dgroup0) `,` type($dgroup1)
  }];
  let extraClassDefinition = [{
    SmallVector<Value> $cppClass::getAccessedOperands() {
      return {getDgroup0(), getDgroup1()};
    }
  }];
}

// Tensor load and store operations
def ROCDL_TensorLoadToLDSOp : ROCDL_TensorLDSIntrOp<"tensor.load.to.lds">;
def ROCDL_TensorStoreFromLDSOp : ROCDL_TensorLDSIntrOp<"tensor.store.from.lds">;
def ROCDL_TensorLoadToLDSD2Op : ROCDL_TensorLDSIntrD2Op<"tensor.load.to.lds.d2">;
def ROCDL_TensorStoreFromLDSD2Op : ROCDL_TensorLDSIntrD2Op<"tensor.store.from.lds.d2">;

//===---------------------------------------------------------------------===//
// Operations on raw buffer resources (stride of 0, bounds checks either off or in
// raw buffer mode).
//===---------------------------------------------------------------------===//

def ROCDLBufferRsrc : LLVM_PointerInAddressSpace<8>;

def ROCDL_MakeBufferRsrcOp :
  ROCDL_IntrOp<"make.buffer.rsrc", [0], [0], [Pure], 1>,
  Arguments<(ins LLVM_AnyPointer:$base,
                 I16:$stride,
                 I64:$numRecords,
                 I32:$flags)> {
  let results = (outs LLVM_AnyPointer:$res);
  let assemblyFormat = "operands attr-dict `:` type($base) `to` type($res)";
}

def ROCDL_RawPtrBufferLoadOp :
  ROCDL_IntrOp<"raw.ptr.buffer.load", [0], [], [], 1, 0, 1> {
  dag args = (ins Arg<ROCDLBufferRsrc, "", [MemRead]>:$rsrc,
                  I32:$offset,
                  I32:$soffset,
                  I32:$aux);
  let arguments = !con(args, baseArgs);
  let assemblyFormat = "operands attr-dict `:` type($res)";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getRsrc()};
    }
  }];
}

def ROCDL_RawPtrBufferLoadLdsOp :
  ROCDL_IntrOp<"raw.ptr.buffer.load.lds", [], [], [], 0, 0, 1> {
  dag args = (ins Arg<ROCDLBufferRsrc, "", [MemRead]>:$rsrc,
                  Arg<ROCDLBufferLDS, "", [MemWrite]>:$ldsPtr,
                  I32:$size,
                  I32:$voffset,
                  I32:$soffset,
                  I32:$offset,
                  I32:$aux);
  let arguments = !con(args, baseArgs);
  let assemblyFormat = "operands attr-dict";
  let extraClassDefinition = [{
    ::llvm::SmallVector<::mlir::Value> $cppClass::getAccessedOperands() {
      return {getRsrc(), getLdsPtr()};
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
  let arguments = !con(args, baseArgs);
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
  let arguments = !con(args, baseArgs);
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
  let arguments = !con(args, baseArgs);
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
  ROCDL_IntrOp<"raw.buffer.load", [0], [], [], 1>,
  Arguments<(ins LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)> {
  let hasCustomAssemblyFormat = 1;
}

def ROCDL_RawBufferStoreOp :
  ROCDL_IntrOp<"raw.buffer.store", [], [0], [], 0>,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  let hasCustomAssemblyFormat = 1;
}

def ROCDL_RawBufferAtomicCmpSwap :
  ROCDL_IntrOp<"raw.buffer.atomic.cmpswap", [], [0], [AllTypesMatch<["res", "src", "cmp"]>], 1>,
  Arguments<(ins LLVM_Type:$src,
                 LLVM_Type:$cmp,
                 LLVM_Type:$rsrc,
                 I32:$offset,
                 I32:$soffset,
                 I32:$aux)>{
  let assemblyFormat = [{
    attr-dict `(` operands `)` `:` type($res) `,` type($rsrc)
  }];
}

//===---------------------------------------------------------------------===//
// Memory prefetch intrinsics

def ROCDL_GlobalPrefetchOp :
  ROCDL_IntrOp<"global.prefetch", [], [], [], 0, 0, 1, 0, [1], ["scope"]> {
  dag args = (ins Arg<LLVM_PointerInAddressSpace<1>, "", [MemRead, MemRead]>:$ptr,
                  I32Attr:$scope);
  let arguments = !con(args, baseArgs);
  let description = [{
    Prefetches 1 byte of data per lane from global memory into the WGP-cache or L2-cache.
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$ptr `,` `scope` $scope attr-dict `:` qualified(type($ptr))";
  let extraClassDefinition = [{
    SmallVector<Value> $cppClass::getAccessedOperands() {
      return {getPtr()};
    }
  }];
}

def ROCDL_FlatPrefetchOp :
  ROCDL_IntrOp<"flat.prefetch", [], [], [], 0, 0, 1, 0, [1], ["scope"]> {
  dag args = (ins Arg<LLVM_PointerInAddressSpace<0>, "", [MemRead, MemRead]>:$ptr,
                  I32Attr:$scope);
  let arguments = !con(args, baseArgs);
  let description = [{
    Prefetches 1 byte of data per lane using flat-memory addresses into the WGP-cache or L2-cache.
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$ptr `,` `scope` $scope attr-dict `:` qualified(type($ptr))";
  let extraClassDefinition = [{
    SmallVector<Value> $cppClass::getAccessedOperands() {
      return {getPtr()};
    }
  }];
}

//===---------------------------------------------------------------------===//
// Atomic barrier intrinsic (LDS memory barriers).

def ROCDL_DsAtomicBarrierArriveRtnOp :
  ROCDL_IntrOp<"ds.atomic.barrier.arrive.rtn.b64", [], [], [], 1, 0, 1, 0, [], []> {
  dag args = (ins Arg<ROCDLBufferLDS, "", [MemRead, MemWrite]>:$barrierPtr,
                  I64:$val);
  let arguments = !con(args, baseArgs);
  let description = [{
    Waits on a given DS barrier and decrements its pending count by a given value. Note, the barrier state
    is given as a 64-bit structure containing pending count, phase and init count. The op returns the old
    barrier state. The op is executed as an ordinary LDS operations and it is ordered with other LDS operations.
    Thus, check DSCNT to determine when this instruction has executed.
    Available on gfx1250+.
  }];
  let results = (outs I64:$res);
  let assemblyFormat = "$barrierPtr `,` $val attr-dict `:` qualified(type($barrierPtr)) `,` type($val) `->` type($res)";
  let extraClassDefinition = [{
    SmallVector<Value> $cppClass::getAccessedOperands() {
      return {getBarrierPtr()};
    }
  }];
}

def ROCDL_DsAtomicAsyncBarrierArriveOp :
  ROCDL_IntrOp<"ds.atomic.async.barrier.arrive.b64", [], [], [], 0, 0, 1, 0, [], []> {
  dag args = (ins Arg<ROCDLBufferLDS, "", [MemWrite]>:$barrierPtr);
  let arguments = !con(args, baseArgs);
  let description = [{
    Waits on a given DS barrier and decrements pending count by -1.
    Stays in order with ASYNC loads to LDS, and uses ASYNCcnt to track its completion.
    Available on gfx1250+.
  }];
  let results = (outs);
  let assemblyFormat = "$barrierPtr attr-dict `:` qualified(type($barrierPtr))";
  let extraClassDefinition = [{
    SmallVector<Value> $cppClass::getAccessedOperands() {
      return {getBarrierPtr()};
    }
  }];
}

//===---------------------------------------------------------------------===//
// MI-100 and MI-200 buffer atomic floating point add intrinsic

def ROCDL_RawBufferAtomicFAddOp :
  ROCDL_IntrOp<"raw.buffer.atomic.fadd", [], [0], [], 0>,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Buffer atomic floating point max intrinsic. GFX9 does not support fp32.

def ROCDL_RawBufferAtomicFMaxOp :
  ROCDL_IntrOp<"raw.buffer.atomic.fmax", [], [0], [], 0>,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Buffer atomic signed integer max intrinsic.

def ROCDL_RawBufferAtomicSMaxOp :
  ROCDL_IntrOp<"raw.buffer.atomic.smax", [], [0], [], 0>,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Buffer atomic unsigned integer min intrinsic.

def ROCDL_RawBufferAtomicUMinOp :
  ROCDL_IntrOp<"raw.buffer.atomic.umin", [], [0], [], 0>,
  Arguments<(ins LLVM_Type:$vdata,
                 LLVM_Type:$rsrc,
                 LLVM_Type:$offset,
                 LLVM_Type:$soffset,
                 LLVM_Type:$aux)>{
  let hasCustomAssemblyFormat = 1;
}

// DPP Update intrinsic
def ROCDL_DPPUpdateOp : ROCDL_IntrOp<"update.dpp", [], [0],
    [AllTypesMatch<["res", "src", "old"]>], 1, 0, 0, 0,
      [2, 3, 4, 5], ["dppCtrl", "rowMask", "bankMask", "boundCtrl"]>,
  Arguments<(ins LLVM_Type:$old, LLVM_Type:$src, I32Attr:$dppCtrl, I32Attr:$rowMask,
      I32Attr:$bankMask, I1Attr:$boundCtrl)> {
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = [{
    attr-dict $old `,` $src `with` $dppCtrl `,` $rowMask `,` $bankMask `,` $boundCtrl `:` type($src)
  }];
}

// PermLaneX16 intrinsic operation
def ROCDL_PermlaneX16Op : ROCDL_IntrOp<"permlanex16", [], [0],
    [AllTypesMatch<["res", "old", "src0"]>, AllTypesMatch<["src1", "src2"]>], 1, 0, 0, 0,
    [4, 5], ["fi", "boundControl"]>,
  Arguments<(ins LLVM_Type:$old, LLVM_Type:$src0, LLVM_Type:$src1, LLVM_Type:$src2,
             I1Attr:$fi, I1Attr:$boundControl)> {
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = [{
    attr-dict $old `,` $src0 `,` $src1 `,` $src2 `,` $fi `,` $boundControl `:` type($src0) `,` type($src1)
  }];
  let description = [{
    Performs a `permlanex16` operation with the given operands, applying the
    permutation specified by $fi to the provided inputs.
  }];
}

class ROCDL_ConcretePair<Type elem0, Type elem1> :
  Type<And<[
    LLVM_AnyStruct.predicate,
    SubstLeaves<
         "$_self",
         "::llvm::cast<::mlir::LLVM::LLVMStructType>($_self).getBody()[0]",
         elem0.predicate>,
      SubstLeaves<
         "$_self",
         "::llvm::cast<::mlir::LLVM::LLVMStructType>($_self).getBody()[1]",
         elem1.predicate>
  ]>,
  "LLVM dialect-compatible struct of " # elem0.summary # "and" # elem1.summary,
  "::mlir::LLVM::LLVMStructType">,
  BuildableType<"::mlir::LLVM::LLVMStructType::getLiteral($_builder.getContext(), "
  "{" # elem0.builderCall # ", " # elem1.builderCall # "})">;

// Permlane16 swap intrinsic operation
def ROCDL_Permlane16SwapOp : ROCDL_IntrOp<"permlane16.swap", [], [],
    [], 1, 0, 0, 0,
    [2, 3], ["fi", "boundControl"]>,
  Arguments<(ins I32:$old, I32:$src, I1Attr:$fi, I1Attr:$boundControl)> {
  let results = (outs ROCDL_ConcretePair<I32, I32>:$res);
  let assemblyFormat = [{
    attr-dict $old `,` $src `,` $fi `,` $boundControl `:` `(` type($old) `,` type($src) `)` `->` type($res)
  }];
  let description = [{
    Performs a `permlane16.swap` operation with the given operands, applying the
    permutation specified by $fi to the provided inputs.
  }];
}

// Permlane32 swap intrinsic operation
def ROCDL_Permlane32SwapOp : ROCDL_IntrOp<"permlane32.swap", [], [],
    [], 1, 0, 0, 0,
    [2, 3], ["fi", "boundControl"]>,
  Arguments<(ins I32:$old, I32:$src, I1Attr:$fi, I1Attr:$boundControl)> {
  let results = (outs ROCDL_ConcretePair<I32, I32>:$res);
  let assemblyFormat = [{
    attr-dict $old `,` $src `,` $fi `,` $boundControl `:` `(` type($old) `,` type($src) `)` `->` type($res)
  }];
  let description = [{
    Performs a `permlane32.swap` operation with the given operands, applying the
    permutation specified by $fi to the provided inputs.
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
// 8-bit float intrinsics
//===---------------------------------------------------------------------===//
def ROCDL_CvtF32Bf8Op :
    ROCDL_ConcreteNonMemIntrOp<"cvt.f32.bf8", [Pure], 1, [1], ["byteSel"]>,
    Arguments<(ins I32:$srcA, I32Attr:$byteSel)> {
  let summary = "Convert bf8 to f32";
  let description = [{
    Convert 8-bit bf8 value from the `byteSel`th bit of `srcA` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_CvtF32Fp8Op :
    ROCDL_ConcreteNonMemIntrOp<"cvt.f32.fp8", [Pure], 1, [1], ["byteSel"]>,
    Arguments<(ins I32:$srcA, I32Attr:$byteSel)> {
  let summary = "Convert fp8 to f32";
  let description = [{
    Convert 8-bit fp8 value from the `byteSel`th bit of `srcA` to fp32.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `[` $byteSel `]` `:` type($res)
  }];
}

def ROCDL_CvtPkF32Fp8Op :
    ROCDL_ConcreteNonMemIntrOp<"cvt.pk.f32.fp8", [Pure], 1, [1], ["wordSel"]>,
    Arguments<(ins I32:$src, I1Attr:$wordSel)> {
  let summary = "Convert packed fp8 to packed f32";
  let description = [{
    Convert `src` based on $wordSel to packed fp32.
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_CvtPkF32Bf8Op :
    ROCDL_ConcreteNonMemIntrOp<"cvt.pk.f32.bf8", [Pure], 1, [1], ["wordSel"]>,
    Arguments<(ins I32:$src, I1Attr:$wordSel)> {
  let summary = "Convert packed bf8 to packed f32";
  let description = [{
    Convert `src` based on $wordSel to packed fp32,
  }];
  let assemblyFormat = [{
    attr-dict $src `[` $wordSel `]` `:` type($res)
  }];
}

def ROCDL_CvtPkBf8F32Op :
    ROCDL_ConcreteNonMemIntrOp<"cvt.pk.bf8.f32", [Pure], 1, [3], ["wordSel"]>,
    Arguments<(ins F32:$srcA, F32:$srcB, I32:$old, I1Attr:$wordSel)> {
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
    ROCDL_ConcreteNonMemIntrOp<"cvt.pk.fp8.f32", [Pure], 1, [3], ["wordSel"]>,
    Arguments<(ins F32:$srcA, F32:$srcB, I32:$old, I1Attr:$wordSel)> {
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
    ROCDL_ConcreteNonMemIntrOp<"cvt.sr.bf8.f32", [Pure], 1, [3], ["byteSel"]>,
    Arguments<(ins F32:$srcA, I32:$srcB, I32:$old, I32Attr:$byteSel)> {
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
    ROCDL_ConcreteNonMemIntrOp<"cvt.sr.fp8.f32", [Pure], 1, [3], ["byteSel"]>,
    Arguments<(ins F32:$srcA, I32:$srcB, I32:$old, I32Attr:$byteSel)> {
  let summary = "Convert f32 to fp8, stochiastic rounding";
  let description = [{
    Convert `srcA` to fp8, adding the rounding factor from `srcB`,
    and store into the `byteSel`th byte of `old`, preserving the others.
  }];
  let assemblyFormat = [{
    attr-dict $srcA `,` $srcB `->` $old `[` $byteSel `]` `:` type($res)
  }];
}

//===---------------------------------------------------------------------===//
// Scaled float conversion intrinsics
//
// These are using some tablegen trickery to avoid repetitive documentation
//===---------------------------------------------------------------------===//

// Pair used so we can iterate over types..
class ScaleArgInfo<TypeConstraint argTyVal, string typeName> {
  TypeConstraint type = argTyVal;
  string name = !tolower(typeName);
  string nameForOp = typeName;
}

//===---------------------------------------------------------------------===//
// Scaled {fp4,bf8,fp8} to {bf16,f16,f32} conversion intrinsics
//===---------------------------------------------------------------------===//
foreach smallT = [
  ScaleArgInfo<I32, "Fp4">,
  ScaleArgInfo<ROCDL_V2I32Type, "Fp8">,
  ScaleArgInfo<ROCDL_V2I32Type, "Bf8">
] in {
  foreach largeT = [
    ScaleArgInfo<ROCDL_V8F16Type, "F16">,
    ScaleArgInfo<ROCDL_V8BF16Type, "Bf16">,
    ScaleArgInfo<ROCDL_V8F32Type, "F32">,
  ] in {

    // Up-scaling
    def ROCDL_CvtPkScalePk8 # largeT.nameForOp # smallT.nameForOp # Op :
          ROCDL_ConcreteNonMemIntrOp<"cvt.scale.pk8." # largeT.name # "." # smallT.name,
          [Pure], 1, [2], ["scaleSel"]>,
        Arguments<(ins smallT.type:$src, I32:$scale, I32Attr:$scaleSel)> {

      let summary = "Scales 8 " # smallT.name # " and converts them to 8 " # largeT.name # ".";
      let description = [{
        Available on gfx1250+.
      }];
      let results = (outs largeT.type:$res);
      let assemblyFormat = [{
        attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
      }];
    }

    // Down-scaling
    def ROCDL_CvtScaleF32Pk8 # smallT.nameForOp # largeT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk8." # smallT.name # "." # largeT.name,
          [Pure], 1>,
        Arguments<(ins largeT.type:$src, F32:$scale)> {
      let results = (outs smallT.type:$res);
      let summary = "Scale and convert packed "
        # largeT.name # " to packed " # smallT.name ;
     let description = [{
        Convert 8 packed }] # largeT.name # [{ values to packed }]
        # smallT.name # [{, multiplying by the exponent part of `scale`
        before doing so. This op is for gfx1250+ arch.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $scale `:` type($res)
      }];
    }


    def ROCDL_CvtScaleF32SrPk8 # smallT.nameForOp # largeT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.sr.pk8." # smallT.name # "." # largeT.name,
          [Pure], 1>,
        Arguments<(ins largeT.type:$src, I32:$seed, F32:$scale)> {
      let results = (outs smallT.type:$res);
      let summary = "Scale and convert packed "
        # largeT.name # " to packed " # smallT.name # " with stochastic rounding";
     let description = [{
        Convert 8 packed }] # largeT.name # [{ values to packed }]
        # smallT.name # [{, multiplying by the exponent part of `scale`
        before doing so and apply stochastic rounding. This op is for gfx1250+ arch.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $seed `,` $scale `:` type($res)
      }];
    }
  } // foreach largeT
} // foreach smallTOp

//===---------------------------------------------------------------------===//
// Scaled {bf6,fp6} to {bf16,f16,f32} conversion intrinsics
//===---------------------------------------------------------------------===//
foreach smallT = [
  ScaleArgInfo<ROCDL_V3I32Type, "Fp6">,
  ScaleArgInfo<ROCDL_V3I32Type, "Bf6">
] in {
  foreach largeT = [
    ScaleArgInfo<ROCDL_V16F16Type, "F16">,
    ScaleArgInfo<ROCDL_V16BF16Type, "Bf16">,
    ScaleArgInfo<ROCDL_V16F32Type, "F32">,
  ] in {
    // Up-scaling
    def ROCDL_CvtPkScalePk16 # largeT.nameForOp # smallT.nameForOp # Op :
          ROCDL_ConcreteNonMemIntrOp<"cvt.scale.pk16." # largeT.name # "." # smallT.name,
          [Pure], 1, [2], ["scaleSel"]>,
        Arguments<(ins smallT.type:$src, I32:$scale, I32Attr:$scaleSel)> {

      let summary = "Scales 16 " # smallT.name # " and converts them to 16 " # largeT.name # ".";
      let description = [{
        Available on gfx1250+.
      }];
      let results = (outs largeT.type:$res);
      let assemblyFormat = [{
        attr-dict $src `,` $scale `[` $scaleSel `]` `:` type($res)
      }];

    }

    // Down-scaling
    def ROCDL_CvtScaleF32Pk16 # smallT.nameForOp # largeT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk16." # smallT.name # "." # largeT.name,
          [Pure], 1>,
        Arguments<(ins largeT.type:$src, F32:$scale)> {
      let results = (outs smallT.type:$res);
      let summary = "Scale and convert packed "
        # largeT.name # " to packed " # smallT.name ;
     let description = [{
        Convert 8 packed }] # largeT.name # [{ values to packed }]
        # smallT.name # [{, multiplying by the exponent part of `scale`
        before doing so. This op is for gfx1250+ arch.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $scale `:` type($res)
      }];
    }

    def ROCDL_CvtScaleF32SrPk16 # smallT.nameForOp # largeT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.sr.pk16." # smallT.name # "." # largeT.name,
          [Pure], 1>,
        Arguments<(ins largeT.type:$src, I32:$seed, F32:$scale)> {
      let results = (outs smallT.type:$res);
      let summary = "Scale and convert packed "
        # largeT.name # " to packed " # smallT.name # " with stochastic rounding";
     let description = [{
        Convert 8 packed }] # largeT.name # [{ values to packed }]
        # smallT.name # [{, multiplying by the exponent part of `scale`
        before doing so and apply stochastic rounding. This op is for gfx1250+ arch.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $seed `,` $scale `:` type($res)
      }];
    }

  } // foreach largeT
} // foreach smallTOp

//===---------------------------------------------------------------------===//
// Scaled 32x6-bit float float conversion intrinsics
//===---------------------------------------------------------------------===//
foreach smallT = [
   // MLIR f6E2M3FN
  ScaleArgInfo<ROCDL_V6I32Type, "Fp6">,
   // MLIR f8E3M2FN
  ScaleArgInfo<ROCDL_V6I32Type, "Bf6">
] in {
  foreach largeT = [
    ScaleArgInfo<ROCDL_V32F16Type, "F16">,
    ScaleArgInfo<ROCDL_V32BF16Type, "Bf16">,
    ScaleArgInfo<ROCDL_V32F32Type, "F32">,
  ] in {
    // Note: rouding down f32 values has a special case where
    // we have to use 2 16xf32 arguments.
    if !ne(largeT.name, "f32") then {
      def ROCDL_CvtScaleF32Pk32 # smallT.nameForOp # largeT.nameForOp # Op :
          ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk32." # smallT.name # "." # largeT.name,
            [Pure], 1>,
          Arguments<(ins largeT.type:$src, F32:$scale)> {
        let results = (outs smallT.type:$res);
        let summary = "Scale and convert packed "
          # largeT.name # " to packed " # smallT.name;
        let description = [{
          Convert 32 packed }] # largeT.name # [{ values to packed }]
          # smallT.name # [{, dividing by the exponent part of `scale`
          before doing so.
        }];
        let assemblyFormat = [{
          attr-dict $src `,` $scale `:` type($res)
        }];
      }
    } // if

    def ROCDL_CvtScaleF32SrPk32 # smallT.nameForOp # largeT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.sr.pk32." # smallT.name # "." # largeT.name,
          [Pure], 1>,
        Arguments<(ins largeT.type:$src, I32:$seed, F32:$scale)> {
      let results = (outs smallT.type:$res);
      let summary = "Scale and convert packed "
        # largeT.name # " to packed " # smallT.name
        # " with stochiastic rounding";
      let description = [{
        Convert 32 packed }] # largeT.name # [{ values to packed }]
        # smallT.name # [{, dividing by the exponent part of `scale`
        before doing so and applying random rounding derived from
        `seed`.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $seed `,` $scale `:` type($res)
      }];
    }

    def ROCDL_CvtScaleF32Pk32 # largeT.nameForOp # smallT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk32." # largeT.name # "." # smallT.name,
          [Pure], 1>,
        Arguments<(ins smallT.type:$src, F32:$scale)> {
      let results = (outs largeT.type:$res);
      let summary = "Scale and convert packed "
        # smallT.name # " to packed " # largeT.name;
      let description = [{
        Convert 32 packed }] # smallT.name # [{ values to packed }]
        # largeT.name # [{, multiplying by the exponent part of `scale`
        before doing so.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $scale `:` type($res)
      }];
    }
  } // foreach largeT

  def ROCDL_CvtScaleF322xPk16 # smallT.nameForOp # F32Op :
      ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.2xpk16." # smallT.name # ".f32",
        [Pure], 1>,
      Arguments<(ins ROCDL_V16F32Type:$src0, ROCDL_V16F32Type:$src1, F32:$scale)> {
    let results = (outs smallT.type:$res);
    let summary = "Scale and convert two vector<16xf32> to 32 packed " # smallT.name;
    let description = [{
      Convert 32 single-precision float values, packed into two length-16
      vectors that will be logically concanenated, to packed }]
      # smallT.name # [{, dividing by the exponent part of `scale`
      before doing so.
    }];
   let assemblyFormat = [{
      attr-dict $src0 `,` $src1 `,` $scale `:` type($res)
    }];
  }
} // forach smallT

//===---------------------------------------------------------------------===//
// Scaled conversions to/from fp8/bf8 (f8E4M3FN / f8E5M2)
//===---------------------------------------------------------------------===//
foreach smallTOp = ["Fp8", "Bf8"] in {
  defvar smallT = !tolower(smallTOp);

  def ROCDL_CvtScaleF32F16 # smallTOp # Op :
      ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.f16." # smallT,
      [Pure], 1, [3, 4], ["srcSelIndex", "dstLoHiSel"]>,
    Arguments<(ins ROCDL_V2F16Type:$oldVdst, I32:$src, F32:$scale, I32Attr:$srcSelIndex, I1Attr:$dstLoHiSel)> {
    let results = (outs ROCDL_V2F16Type:$res);
    let summary = "Scaled convert " # smallT # " from packed vector to f16, updating tied result";
    let description = [{
      Convert a }] # smallT # [{ byte from `src`, selected by
      `srcSelIndex`, to f16 while multiplying it by the expontent of `scale`,
      and place it into the `dstLoHiSel`th bit
      of `oldVdst` preserving the other element of that vector in
      the return value.

      The bytes are stored as an `i32` and not a `<4 x i8>`.
    }];
    let assemblyFormat = [{
      attr-dict $src `[` $srcSelIndex `]` `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
    }];
  }

  def ROCDL_CvtScaleF32F32 # smallTOp # Op :
      ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.f32." # smallT,
      [Pure], 1, [2], ["srcSelIndex"]>,
    Arguments<(ins I32:$src, F32:$scale, I32Attr:$srcSelIndex)> {
    let results = (outs F32:$res);
    let summary = "Scaled convert " # smallT # " from packed vector to f32";
    let description = [{
      Convert a }] # smallT # [{ byte from `src`, selected by
      `srcSelIndex`, to f32, multiplying it by the exponent of `scale`.

      The bytes are stored in an `i32`, not a `<4 x i8>`.
    }];
    let assemblyFormat = [{
      attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
    }];
  }

  def ROCDL_CvtScaleF32Pk # smallTOp # F32Op :
      ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk." # smallT # ".f32",
      [Pure], 1, [4], ["dstLoHiSel"]>,
    Arguments<(ins ROCDL_V2I16Type:$oldVdst, F32:$src0, F32:$src1, F32:$scale, I1Attr:$dstLoHiSel)> {
    let results = (outs ROCDL_V2I16Type:$res);
    let summary = "Scaled convert two f32 to two " # smallT # ", updating packed vector";
    let description = [{
      Convert two f32 values in `src0` and `src1` to two }] # smallT # [{ bytes,
      dividing by the exponent in `scale`. The bytes are packed into
      a 16-bit value which is inserted into `oldVdst` at the `dstLoHiSel`
      position, with the entire updated vector being returned.
    }];
    let assemblyFormat = [{
      attr-dict  $src0 `,` $src1 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
    }];
  }

  foreach largeT = [
    ScaleArgInfo<ROCDL_V2F16Type, "F16">,
    ScaleArgInfo<ROCDL_V2BF16Type, "Bf16">,
  ] in {
    def ROCDL_CvtScaleF32Pk # smallTOp # largeT.nameForOp # Op :
          ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk." # smallT # "." # largeT.name,
          [Pure], 1, [3], ["dstLoHiSel"]>,
        Arguments<(ins ROCDL_V2I16Type:$oldVdst, largeT.type:$src0, F32:$scale, I1Attr:$dstLoHiSel)> {
      let results = (outs ROCDL_V2I16Type:$res);
      let summary = "Scaled convert two " # largeT.name # "to two " # smallT # ", updating packed vector";
      let description = [{
        Convert two }] # largeT.name # [{ values in `src0` to two }]
        # smallT # [{ bytes, dividing by the exponent in `scale`. The bytes are
        packed into a 16-bit value which is inserted into `oldVdst` at the
        `dstLoHiSel` position, with the entire updated vector being returned.
      }];
      let assemblyFormat = [{
        attr-dict $src0 `,` $scale `->` $oldVdst `[` $dstLoHiSel `]` `:` type($res)
      }];
    }
  } // foreach largeT

  foreach largeT = [
    ScaleArgInfo<ROCDL_V2F16Type, "F16">,
    ScaleArgInfo<ROCDL_V2BF16Type, "Bf16">,
    ScaleArgInfo<ROCDL_V2F32Type, "F32">
  ] in {
    def ROCDL_CvtScaleF32Pk # largeT.nameForOp # smallTOp # Op :
          ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk." # largeT.name # "." # smallT,
          [Pure], 1, [2], ["srcLoHiSel"]>,
        Arguments<(ins I32:$src, F32:$scale, I1Attr:$srcLoHiSel)> {
      let results = (outs largeT.type:$res);
      let summary = "Scaled convert two " # smallT # "to two " # largeT.name #;
      let description = [{
        Convert two packed }] # smallT # [{ values in `src0` to two }]
        # largeT.name # [{ values, multiplying by the exponent in `scale`.
        The two values to be converted are selected from the low or high half
        of `src` (a packed vector represented as an `i32`)
        on the basis of `srcLoHiSel`.
      }];
      let assemblyFormat = [{
        attr-dict $src `[` $srcLoHiSel `]` `,` $scale `:` type($res)
      }];
    }
  } // foreach largeT

  foreach largeT = [
    ScaleArgInfo<F32, "F32">,
    ScaleArgInfo<F16, "F16">,
    ScaleArgInfo<BF16, "BF16">
  ] in {
    def ROCDL_CvtScaleF32Sr # smallTOp # largeT.nameForOp # Op :
          ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.sr." # smallT # "." # largeT.name,
          [Pure], 1, [4], ["dstSelIndex"]>,
        Arguments<(ins I32:$oldVdst, largeT.type:$src0, I32:$seed, F32:$scale, I32Attr:$dstSelIndex)> {
      let results = (outs I32:$res);
      let summary = "Scaled convert " # largeT.name # "to " # smallT # " with stochiastic rounding, updating packed vector";
      let description = [{
        Convert a }] # largeT.name # [{ value in `src0` to a }]
        # smallT # [{ bytes, dividing by the exponent in `scale` and using `seed`
        for stochiastic rounding. Place the resulting byte in the
        `dstSelIndex`th bit of `oldVdst` and return the entire packed vector,
        which is stored as an `i32`.
      }];
      let assemblyFormat = [{
        attr-dict $src0 `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
      }];
    }
  } // foreach largeT
} // foreach smallTOp

//===---------------------------------------------------------------------===//
// Scaled conversions to/from fp4 (f4E2M1FN)
//===---------------------------------------------------------------------===//

foreach largeT = [
  ScaleArgInfo<ROCDL_V2F16Type, "F16">,
  ScaleArgInfo<ROCDL_V2BF16Type, "Bf16">,
  ScaleArgInfo<ROCDL_V2F32Type, "F32">,
] in {
  // Note: rouding down f32 values has a special case where
  // we have to use 2 float arguments.
  if !ne(largeT.name, "f32") then {
    def ROCDL_CvtScaleF32PkFp4 # largeT.nameForOp # Op :
        ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk.fp4." # largeT.name,
          [Pure], 1, [3], ["dstSelIndex"]>,
        Arguments<(ins I32:$oldVdst, largeT.type:$src, F32:$scale, I32Attr:$dstSelIndex)> {
      let results = (outs I32:$res);
      let summary = "Scale and convert two "
        # largeT.name # " to packed fp4, updating tied vector";
      let description = [{
        Convert two packed }] # largeT.name # [{ values to packed
        fp4, dividing by the exponent part of `scale`
        before doing so.

        The two scaled values are packed  into a byte.
        That byte is used to update the `dstSelIndex`th
        byte of `oldVdst`, which is returned in its entirity.
      }];
      let assemblyFormat = [{
        attr-dict $src `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
      }];
    }
  } // if

  def ROCDL_CvtScaleF32SrPkFp4 # largeT.nameForOp # Op :
      ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.sr.pk.fp4." # largeT.name,
        [Pure], 1, [4], ["dstSelIndex"]>,
      Arguments<(ins I32:$oldVdst, largeT.type:$src, I32:$seed, F32:$scale, I32Attr:$dstSelIndex)> {
    let results = (outs I32:$res);
    let summary = "Scale and convert two "
      # largeT.name # " to packed fp4 with stochiastic rounding, updating tied vector";
    let description = [{
      Convert two packed }] # largeT.name # [{ values to packed
      fp4, dividing by the exponent part of `scale`
      before doing so and using `seed` as the random seed for
      stochiastic rounding.

      The two scaled values are packed (little-endian)
      into a byte. That byte is used to update the `dstSelIndex`th
      byte of `oldVdst`, which is returned in its entirity.
    }];
    let assemblyFormat = [{
      attr-dict $src `,` $seed `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
    }];
  }

  def ROCDL_CvtScaleF32Pk # largeT.nameForOp # Fp4Op :
      ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk." # largeT.name # ".fp4",
        [Pure], 1, [2], ["srcSelIndex"]>,
      Arguments<(ins I32:$src, F32:$scale, I32Attr:$srcSelIndex)> {
    let results = (outs largeT.type:$res);
    let summary = "Scale and convert two packed fp4 to packed " # largeT.name;
    let description = [{
      Convert two packed fp4 (f4E2M1) values  stored as one byte of a 32-bit integer
      to packed }] # largeT.name # [{, multiplying by the exponent part of `scale`
      before doing so.

      The byte to convert is chosen by `srcSelIndex`.
    }];
    let assemblyFormat = [{
      attr-dict $src `[` $srcSelIndex `]` `,` $scale `:` type($res)
    }];
  }
} // foreach largeT

def ROCDL_CvtScaleF32PkFp4F32Op :
    ROCDL_ConcreteNonMemIntrOp<"cvt.scalef32.pk.fp4.f32",
      [Pure], 1, [4], ["dstSelIndex"]>,
    Arguments<(ins I32:$oldVdst, F32:$src0, F32:$src1, F32:$scale, I32Attr:$dstSelIndex)> {
  let results = (outs I32:$res);
  let summary = "Scale and convert two f32 values to two packed fp4, updating tied vector";
  let description = [{
    Convert two single-precision float values, passed in `src0` and `src1`
    into two fp4 values, dividing them by the expontent part of `scale`
    before doing so.

    The two scaled values are packed  into a byte.
    That byte is used to update the `dstSelIndex`th
    byte of `oldVdst`, which is returned in its entirity.
  }];
 let assemblyFormat = [{
    attr-dict $src0 `,` $src1 `,` $scale `->` $oldVdst `[` $dstSelIndex `]` `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// FMED3 operations
//===----------------------------------------------------------------------===//

def ROCDL_FMed3Op : ROCDL_IntrOp<"fmed3", [0], [], [Pure, AllTypesMatch<["res", "src0", "src1", "src2"]>], 1>,
  Arguments<(ins LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$src0,
                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$src1,
                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$src2)> {
  let results = (outs LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$res);
  let summary = "Median of three float/half values";
  let description = [{
    Computes the median of three floating-point values using the AMDGPU fmed3 intrinsic.
    This operation is equivalent to `max(min(a, b), min(max(a, b), c))` but uses the
    hardware-accelerated V_MED3_F16/V_MED3_F32 instruction for better performance.

    The operation supports both scalar and vector floating-point types (f16, f32).

    Example:
    ```mlir
    // Scalar f32 median
    %result = rocdl.fmed3 %a, %b, %c : f32

    // Vector f16 median
    %result = rocdl.fmed3 %va, %vb, %vc : vector<4xf16>
    ```
  }];
  let assemblyFormat = [{
    $src0 `,` $src1 `,` $src2 attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Math operations
//===----------------------------------------------------------------------===//

class ROCDL_Math_IntrOp<string mnemonic, list<Trait> traits = [Pure]> :
  ROCDL_IntrOp<mnemonic, [0], [], traits, 1>,
  Arguments<(ins LLVM_AnyFloat:$arg)> {
  let results = (outs LLVM_AnyFloat:$res);
  let description = [{
    Note: In the general case, prefer the conventional `arith`, `math`, or `llvm` ops over this.
    Use this ROCDL-specific operation only when you fully understand its implication and
    when it is strictly necessary. This op is usually chosen when a small loss in precision is
    acceptable in exchange for higher execution speed.
  }];
  let assemblyFormat =
    "$arg qualified(type($arg)) attr-dict `->` qualified(type($res))";
}

def ROCDLTanh : ROCDL_Math_IntrOp<"tanh">;
def ROCDLSin : ROCDL_Math_IntrOp<"sin">;
def ROCDLCos : ROCDL_Math_IntrOp<"cos">;
def ROCDLRcp : ROCDL_Math_IntrOp<"rcp">;
def ROCDLExp : ROCDL_Math_IntrOp<"exp">;
def ROCDLExp2 : ROCDL_Math_IntrOp<"exp2">;
def ROCDLLog : ROCDL_Math_IntrOp<"log">;
def ROCDLSqrt : ROCDL_Math_IntrOp<"sqrt">;
def ROCDLRsq : ROCDL_Math_IntrOp<"rsq">;

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
    StringRefParameter<"ABI version.", "\"600\"">:$abi,
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
                     CArg<"StringRef", "\"600\"">:$abiVersion,
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
