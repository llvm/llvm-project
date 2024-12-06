//===-- GPUBase.td - GPU dialect definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the GPU dialect
//
//===----------------------------------------------------------------------===//

#ifndef GPU_BASE
#define GPU_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU Dialect.
//===----------------------------------------------------------------------===//

def GPU_Dialect : Dialect {
  let name = "gpu";
  let cppNamespace = "::mlir::gpu";
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "gpu.container_module";
    }
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "gpu.kernel"; }

    /// Returns whether the given function is a kernel function, i.e., has the
    /// 'gpu.kernel' attribute.
    static bool isKernel(Operation *op);

    /// Returns the number of workgroup (thread, block) dimensions supported in
    /// the GPU dialect.
    // TODO: consider generalizing this.
    static unsigned getNumWorkgroupDimensions() { return 3; }

    /// Returns the numeric value used to identify the workgroup memory address
    /// space.
    static AddressSpace getWorkgroupAddressSpace() { return AddressSpace::Workgroup; }

    /// Returns the numeric value used to identify the private memory address
    /// space.
    static AddressSpace getPrivateAddressSpace() { return AddressSpace::Private; }

    /// Return true if the given MemRefType has an address space that matches
    /// with the gpu::AddressSpaceAttr attribute with value 'workgroup`.
    static bool hasWorkgroupMemoryAddressSpace(MemRefType type);

    /// Return true if the given Attribute is an gpu::AddressSpaceAttr
    /// attribute with value 'workgroup`.
    static bool isWorkgroupMemoryAddressSpace(Attribute memorySpace);
  }];

  let discardableAttrs = (ins
    "::mlir::DenseI32ArrayAttr":$known_block_size,
    "::mlir::DenseI32ArrayAttr":$known_grid_size
  );

  let dependentDialects = ["arith::ArithDialect"];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// GPU Enums.
//===----------------------------------------------------------------------===//

class GPU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
class GPU_I32EnumAttr<string mnemonic, GPU_I32Enum enumInfo> :
    EnumAttr<GPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def GPU_AddressSpaceGlobal : I32EnumAttrCase<"Global", 1, "global">;
def GPU_AddressSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 2, "workgroup">;
def GPU_AddressSpacePrivate : I32EnumAttrCase<"Private", 3, "private">;
def GPU_AddressSpaceEnum : GPU_I32Enum<
  "AddressSpace", "GPU address space", [
    GPU_AddressSpaceGlobal,
    GPU_AddressSpaceWorkgroup,
    GPU_AddressSpacePrivate
  ]>;

def GPU_AddressSpaceAttr :
  GPU_I32EnumAttr<"address_space", GPU_AddressSpaceEnum>;

def GPU_Dimension : I32EnumAttr<"Dimension",
    "a dimension, either 'x', 'y', or 'z'",
    [
      I32EnumAttrCase<"x", 0>,
      I32EnumAttrCase<"y", 1>,
      I32EnumAttrCase<"z", 2>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def GPU_DimensionAttr : EnumAttr<GPU_Dialect, GPU_Dimension, "dim">;

// These mirror the reduction combining kinds from the vector dialect.
def GPU_AllReduceOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
def GPU_AllReduceOpMul : I32EnumAttrCase<"MUL", 1, "mul">;
def GPU_AllReduceOpMinUI : I32EnumAttrCase<"MINUI", 2, "minui">;
def GPU_AllReduceOpMinSI : I32EnumAttrCase<"MINSI", 3, "minsi">;
// Follows the `arith.minnumf` semantics.
def GPU_AllReduceOpMinnumF : I32EnumAttrCase<"MINNUMF", 4, "minnumf">;
def GPU_AllReduceOpMaxUI : I32EnumAttrCase<"MAXUI", 5, "maxui">;
def GPU_AllReduceOpMaxSI : I32EnumAttrCase<"MAXSI", 6, "maxsi">;
// Follows the `arith.maxnumf` semantics.
def GPU_AllReduceOpMaxnumF : I32EnumAttrCase<"MAXNUMF", 7, "maxnumf">;
def GPU_AllReduceOpAnd : I32EnumAttrCase<"AND", 8, "and">;
def GPU_AllReduceOpOr  : I32EnumAttrCase<"OR",  9, "or">;
def GPU_AllReduceOpXor : I32EnumAttrCase<"XOR", 10, "xor">;
// Follows the `arith.minimumf` semantics.
def GPU_AllReduceOpMinimumF : I32EnumAttrCase<"MINIMUMF", 11, "minimumf">;
// Follows the `arith.maximumf` semantics.
def GPU_AllReduceOpMaximumF : I32EnumAttrCase<"MAXIMUMF", 12, "maximumf">;

def GPU_AllReduceOperation : I32EnumAttr<"AllReduceOperation",
    "built-in reduction operations supported by gpu.allreduce.",
    [
      GPU_AllReduceOpAdd,
      GPU_AllReduceOpMul,
      GPU_AllReduceOpMinUI,
      GPU_AllReduceOpMinSI,
      GPU_AllReduceOpMinnumF,
      GPU_AllReduceOpMaxUI,
      GPU_AllReduceOpMaxSI,
      GPU_AllReduceOpMaxnumF,
      GPU_AllReduceOpAnd,
      GPU_AllReduceOpOr,
      GPU_AllReduceOpXor,
      GPU_AllReduceOpMinimumF,
      GPU_AllReduceOpMaximumF
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def GPU_AllReduceOperationAttr : EnumAttr<GPU_Dialect, GPU_AllReduceOperation,
                                          "all_reduce_op">;

def GPU_ShuffleOpXor  : I32EnumAttrCase<"XOR",  0, "xor">;
def GPU_ShuffleOpDown : I32EnumAttrCase<"DOWN", 1, "down">;
def GPU_ShuffleOpUp   : I32EnumAttrCase<"UP",   2, "up">;
def GPU_ShuffleOpIdx  : I32EnumAttrCase<"IDX",  3, "idx">;

def GPU_ShuffleMode : I32EnumAttr<"ShuffleMode",
    "Indexing modes supported by gpu.shuffle.",
    [
      GPU_ShuffleOpXor, GPU_ShuffleOpUp, GPU_ShuffleOpDown, GPU_ShuffleOpIdx,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def GPU_ShuffleModeAttr : EnumAttr<GPU_Dialect, GPU_ShuffleMode,
                                   "shuffle_mode">;

def GPU_ElementwiseOpAddF  : I32EnumAttrCase<"ADDF", 0, "addf">;
def GPU_ElementwiseOpMulF  : I32EnumAttrCase<"MULF", 1, "mulf">;
def GPU_ElementwiseOpSUBF  : I32EnumAttrCase<"SUBF", 2, "subf">;
def GPU_ElementwiseOpMaxF : I32EnumAttrCase<"MAXF", 3, "maxf">;
def GPU_ElementwiseOpMinF : I32EnumAttrCase<"MINF", 4, "minf">;
def GPU_ElementwiseOpDivF : I32EnumAttrCase<"DIVF", 5, "divf">;
def GPU_ElementwiseOpAddI  : I32EnumAttrCase<"ADDI", 6, "addi">;
def GPU_ElementwiseOpMulI  : I32EnumAttrCase<"MULI", 7, "muli">;
def GPU_ElementwiseOpSUBI  : I32EnumAttrCase<"SUBI", 8, "subi">;
def GPU_ElementwiseOpDivS : I32EnumAttrCase<"DIVS", 9, "divs">;
def GPU_ElementwiseOpDivU : I32EnumAttrCase<"DIVU", 10, "divu">;
def GPU_ElementwiseOpNEGF : I32EnumAttrCase<"NEGATEF", 11, "negatef">;
def GPU_ElementwiseOpNEGS : I32EnumAttrCase<"NEGATES", 12, "negates">;
def GPU_ElementwiseOpEXTF : I32EnumAttrCase<"EXTF", 13, "extf">;

def MMAElementWise : I32EnumAttr<"MMAElementwiseOp",
  "elementwise operation to apply to mma matrix", [
    GPU_ElementwiseOpAddF,
    GPU_ElementwiseOpMulF,
    GPU_ElementwiseOpSUBF,
    GPU_ElementwiseOpMaxF,
    GPU_ElementwiseOpMinF,
    GPU_ElementwiseOpDivF,
    GPU_ElementwiseOpAddI,
    GPU_ElementwiseOpMulI,
    GPU_ElementwiseOpSUBI,
    GPU_ElementwiseOpDivS,
    GPU_ElementwiseOpDivU,
    GPU_ElementwiseOpNEGF,
    GPU_ElementwiseOpNEGS,
    GPU_ElementwiseOpEXTF
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def MMAElementWiseAttr : EnumAttr<GPU_Dialect, MMAElementWise,
                                  "mma_element_wise">;

def GPU_Prune2To4SpMatFlag : I32EnumAttr<"Prune2To4SpMatFlag",
  "pruning strategy for 2:4 sparse matrix",
  [
    I32EnumAttrCase<"NONE", 0>,
    I32EnumAttrCase<"PRUNE_ONLY", 1>,
    I32EnumAttrCase<"PRUNE_AND_CHECK", 2>,
  ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = GPU_Dialect.cppNamespace;
}

def GPU_Prune2To4SpMatFlagAttr : EnumAttr<GPU_Dialect, GPU_Prune2To4SpMatFlag,
                                   "prune_2to4_spmat_flag">{
  let defaultValue = "Prune2To4SpMatFlag::PRUNE_AND_CHECK";
}

// To avoid coupling this dialect with cusparse.h specifics, we hardcoded magic
// literals in this enum. Note that this should be kept in sync with
// cusparseOperation_t in cusparse.h:
// typedef enum {
// CUSPARSE_OPERATION_NON_TRANSPOSE       = 0,
// CUSPARSE_OPERATION_TRANSPOSE           = 1,
// CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
// } cusparseOperation_t;
// TODO: find a proper way to keep them in sync?
def GPU_TransposeMode : I32EnumAttr<"TransposeMode",
    "transpose mode of sparse matrix supported by sparse tensor ops",
    [
      I32EnumAttrCase<"NON_TRANSPOSE", 0>,
      I32EnumAttrCase<"TRANSPOSE", 1>,
      I32EnumAttrCase<"CONJUGATE_TRANSPOSE", 2>,
    ]> {
      let genSpecializedAttr = 0;
      let cppNamespace = GPU_Dialect.cppNamespace;
}

def GPU_TransposeModeAttr : EnumAttr<GPU_Dialect, GPU_TransposeMode,
                                   "mat_transpose_mode">{
  let defaultValue = "TransposeMode::NON_TRANSPOSE";
}

def GPU_SpGEMMWorkEstimationOrComputeKind : I32EnumAttr<"SpGEMMWorkEstimationOrComputeKind",
    "choose whether spgemm_work_estimation_or_compute does work estimation or compute",
    [
      I32EnumAttrCase<"WORK_ESTIMATION", 0>,
      I32EnumAttrCase<"COMPUTE", 1>,
    ]> {
      let genSpecializedAttr = 0;
      let cppNamespace = GPU_Dialect.cppNamespace;
}

def GPU_SpGEMMWorkEstimationOrComputeKindAttr : EnumAttr<GPU_Dialect,
    GPU_SpGEMMWorkEstimationOrComputeKind,
    "spgemm_work_estimation_or_compute_kind"> {}


//===----------------------------------------------------------------------===//
// GPU Types.
//===----------------------------------------------------------------------===//

def GPU_AsyncToken : DialectType<
  GPU_Dialect, CPred<"::llvm::isa<::mlir::gpu::AsyncTokenType>($_self)">, "async token type">,
             BuildableType<"mlir::gpu::AsyncTokenType::get($_builder.getContext())">;

// Predicat to check if type is gpu::MMAMatrixType.
def IsMMAMatrixTypePred : CPred<"::llvm::isa<::mlir::gpu::MMAMatrixType>($_self)">;

def GPU_MMAMatrix : DialectType<
  GPU_Dialect, IsMMAMatrixTypePred, "MMAMatrix type">;

// Memref type acceptable to gpu.subgroup_mma_{load|store}_matrix ops.
def GPU_MMAMemRef : MemRefOf<[I8, I32, F16, F32, VectorOfRankAndType<[1], [I8, I32, F16, F32]>]>;

class MMAMatrixOf<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, IsMMAMatrixTypePred,
  "::llvm::cast<::mlir::gpu::MMAMatrixType>($_self).getElementType()",
  "gpu.mma_matrix", "::mlir::gpu::MMAMatrixType">;

// Types for all sparse handles.
class GPU_SparseHandle<string typeStr, string description> :
    DialectType<GPU_Dialect,
    CPred<"llvm::isa<::mlir::gpu::"#typeStr#">($_self)">,
    description#" handle type">,
  BuildableType<"mlir::gpu::"#typeStr#"::get($_builder.getContext())">;

def GPU_SparseDnTensorHandle : GPU_SparseHandle<"SparseDnTensorHandleType", "dense tensor">;
def GPU_SparseSpGEMMOpHandle : GPU_SparseHandle<"SparseSpGEMMOpHandleType", "SpGEMM operation">;
def GPU_SparseSpMatHandle : GPU_SparseHandle<"SparseSpMatHandleType", "sparse matrix">;


//===----------------------------------------------------------------------===//
// GPU Interfaces.
//===----------------------------------------------------------------------===//

def GPU_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for GPU operations that execute asynchronously on the device.

    GPU operations implementing this interface take a list of dependencies
    as `gpu.async.token` arguments and optionally return a `gpu.async.token`.

    The op doesn't start executing until all depent ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op merely schedules the execution on the
    device and returns immediately, without waiting for the execution to
    complete. On the hand, if the op does not return a token, the op will wait
    for the execution to complete.
  }];
  let cppNamespace = "::mlir::gpu";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAsyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies if it is not already there.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        if (!::llvm::is_contained(this->getAsyncDependencies(), token))
          ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "Value", "getAsyncToken"
    >
  ];
}

//===----------------------------------------------------------------------===//
// GPU Attributes.
//===----------------------------------------------------------------------===//

class GPU_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GPU_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // GPU_BASE
