//===-- GENXOps.td - GENX IR dialect op definition file ----*- tablegen -*-===//
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

// spirv::CrossWorkGroup: 1
def LLVM_i8Ptr_cross_workgroup : LLVM_IntPtrBase<8, 1>;
// spirv::WorkGroup: 3
def LLVM_i8Ptr_workgroup : LLVM_IntPtrBase<8, 3>;
// spirv::Function: 0
def LLVM_i8Ptr_function : LLVM_IntPtrBase<8, 0>;
// spirv::UniformConst: 2
def LLVM_i8Ptr_uniform_constant : LLVM_IntPtrBase<8, 2>;
// spirv::Generic: 4
def LLVM_i8Ptr_generic : LLVM_IntPtrBase<8, 4>;

def IntegerOrFloatType : AnyTypeOf<[AnyInteger, AnyFloat]>;

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
    /// required for kernel functions.
    static constexpr ::llvm::StringLiteral getMaxWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.max_work_group_size");
    }
    /// Get the name of the attribute used to annotate exact work group size
    /// required for kernel functions.
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.reqd_work_group_size");
    }
    /// Get the name for the attribute used to annotate the exact sub group 
    /// size required for kernel functions.
    static constexpr ::llvm::StringLiteral getReqdSubGroupSizeAttrName() {
      return ::llvm::StringLiteral("genx.intel_reqd_sub_group_size");
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

class GENX_DeviceFunctionOp<string mnemonic, string fnName, 
                            int arg, list<Trait> traits = []> 
  : GENX_Op<mnemonic, !listconcat(traits, [Pure])>, 
    Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = ""
    # "  llvm::Type *retType = builder.getInt64Ty();\n"
    # "  llvm::Type *argType = builder.getInt32Ty();\n"
    # "  llvm::Value *arg = llvm::ConstantInt::get(argType, " # arg # ");\n"
    # "  $res = createDeviceFunctionCall(builder, \"" # fnName # "\",\n"
    # "                             retType, { argType }, { arg });";

  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Thread index and Block index

def GENX_ThreadIdXOp : GENX_DeviceFunctionOp<"workitem.id.x",
                        "_Z12get_local_idj", 0>;
def GENX_ThreadIdYOp : GENX_DeviceFunctionOp<"workitem.id.y",
                        "_Z12get_local_idj", 1>;
def GENX_ThreadIdZOp : GENX_DeviceFunctionOp<"workitem.id.z",
                        "_Z12get_local_idj", 2>;
def GENX_BlockIdXOp : GENX_DeviceFunctionOp<"workgroup.id.x",
                        "_Z12get_group_idj", 0>;
def GENX_BlockIdYOp : GENX_DeviceFunctionOp<"workgroup.id.y",
                        "_Z12get_group_idj", 1>;
def GENX_BlockIdZOp : GENX_DeviceFunctionOp<"workgroup.id.z",
                        "_Z12get_group_idj", 2>;

//===----------------------------------------------------------------------===//
// Block and Grid range

def GENX_BlockDimXOp : GENX_DeviceFunctionOp<"workgroup.dim.x",
                        "_Z12get_local_sizej", 0>;
def GENX_BlockDimYOp : GENX_DeviceFunctionOp<"workgroup.dim.y",
                        "_Z12get_local_sizej", 1>;
def GENX_BlockDimZOp : GENX_DeviceFunctionOp<"workgroup.dim.z",
                        "_Z12get_local_sizej", 2>;
def GENX_GridDimXOp : GENX_DeviceFunctionOp<"grid.dim.x",
                        "_Z12get_global_sizej", 0>;
def GENX_GridDimYOp : GENX_DeviceFunctionOp<"grid.dim.y",
                        "_Z12get_global_sizej", 1>;
def GENX_GridDimZOp : GENX_DeviceFunctionOp<"grid.dim.z",
                        "_Z12get_global_sizej", 2>;

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def GENX_BarrierOp : GENX_Op<"barrier"> {
  let summary = "Workgroup barrier";

  string baseDescription = [{
    The `genx.barrier` operation performs a workgroup barrier and ensures all 
    outstanding memory transaction using local or global memory are complete.
  }];

  string llvmBuilder = [{
    llvm::Type *retType = builder.getVoidTy();
    llvm::Type *argType = builder.getInt32Ty();
    int memFence = 3; // local + global memory fence
    llvm::Value *arg = llvm::ConstantInt::get(argType, memFence);
    createDeviceFunctionCall(builder, "_Z7barrierj", retType, {argType}, {arg});
  }];

  let assemblyFormat = "attr-dict";
}

def ShflKindXor : I32EnumAttrCase<"XOR", 0>;
def ShflKindUp   : I32EnumAttrCase<"UP", 1>;
def ShflKindDown : I32EnumAttrCase<"DOWN", 2>;
def ShflKindIdx  : I32EnumAttrCase<"IDX", 3>;

/// Enum attribute of the different shuffle kinds.
def ShflKind : I32EnumAttr<"ShflKind", "GENX shuffle kind",
  [ShflKindXor, ShflKindUp, ShflKindDown, ShflKindIdx]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def ShflKindAttr : EnumAttr<GENX_Dialect, ShflKind, "shfl_kind">;

def GENX_SubGroupShuffleOp : GENX_Op<"sub_group_shuffle", [
      TypesMatchWith<"result and value have the same type",
                     "res", "value", "$_self">]>,
  Results<(outs IntegerOrFloatType:$res)>,
  Arguments<(ins IntegerOrFloatType:$value,
                 I32:$mask,
                 ShflKindAttr:$kind)> {
  let summary = "Subgroup shuffle";
  string baseDescription = [{
    The `genx.sub_group_shuffle` operation is invoked by different work items
    with different values, given by $value. Different work items have different
    subgroup local IDs. The shuffle kind, $kind, is given to determine how to
    calculate the associated subgroup local ID. It returns the associated
    $value for the work item with subgroup local ID equal to:
    - $kind == XOR, the current invocation’s subgroup local ID xor’ed with $mask.
    - $kind == UP, the current invocation’s subgroup local ID - $mask.
    - $kind == DOWN, the current invocation’s subgroup local ID + $mask.
    - $kind == IDX, the subgroup local ID $mask.
  }];
  string llvmBuilder = [{
    $res = createSubGroupShuffle(builder, $value, $mask, $kind);
  }];
  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($value) `->` type($res)
  }];
}

#endif // GENXIR_OPS
