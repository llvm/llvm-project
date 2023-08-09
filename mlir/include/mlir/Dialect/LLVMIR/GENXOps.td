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
    static StringRef getKernelFuncAttrName() { return "opencl.kernels"; }
    /// Get the name of the attribute used to annotate max work group size
    /// required for kernel functions
    static constexpr ::llvm::StringLiteral getMaxWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("max_work_group_size");
    }
    /// Get the name of the attribute used to annotate exact work group size
    /// required for kernel functions.
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("reqd_work_group_size");
    }

    /// The address space value that represents global memory.
    static constexpr unsigned kGlobalMemoryAddressSpace = 1;
    /// The address space value that represents shared memory.
    static constexpr unsigned kSharedMemoryAddressSpace = 3;
    /// The address space value that represents private memory.
    static constexpr unsigned kPrivateMemoryAddressSpace = 0;

    /// private:
    ///  Attribute parseAttribute(DialectAsmParser& parser, Type type) const override;
    ///  void printAttribute(Attribute attr, DialectAsmPrinter& p) const override;
  }];

  /// let useDefaultAttributePrinterParser = 1;
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
  GENX_IntrOp<mnemonic, !listconcat(traits, [Pure]), 1> {
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

//===----------------------------------------------------------------------===//
// Thread index and Block index

def GENX_ThreadIdXOp : GENX_DeviceFunctionOp<"workitem.id.x",
                        "get_local_id", 0>;
def GENX_ThreadIdYOp : GENX_DeviceFunctionOp<"workitem.id.y",
                        "get_local_id", 1>;
def GENX_ThreadIdZOp : GENX_DeviceFunctionOp<"workitem.id.z",
                        "get_local_id", 2>;
def GENX_BlockIdXOp : GENX_DeviceFunctionOp<"workgroup.id.x",
                        "get_group_id", 0>;
def GENX_BlockIdYOp : GENX_DeviceFunctionOp<"workgroup.id.y",
                        "get_group_id", 1>;
def GENX_BlockIdZOp : GENX_DeviceFunctionOp<"workgroup.id.z",
                        "get_group_id", 2>;

//===----------------------------------------------------------------------===//
// Thread range and Block range

def GENX_BlockDimXOp : GENX_DeviceFunctionOp<"workgroup.dim.x",
                        "get_local_size", 0>;

def GENX_BlockDimYOp : GENX_DeviceFunctionOp<"workgroup.dim.y",
                        "get_local_size", 1>;

def GENX_BlockDimZOp : GENX_DeviceFunctionOp<"workgroup.dim.z",
                        "get_local_size", 2>;

def GENX_GridDimXOp : GENX_DeviceFunctionOp<"grid.dim.x",
                        "get_global_size", 0>;

def GENX_GridDimYOp : GENX_DeviceFunctionOp<"grid.dim.y",
                        "get_global_size", 1>;

def GENX_GridDimZOp : GENX_DeviceFunctionOp<"grid.dim.z",
                        "get_global_size", 2>;

#endif // GENXIR_OPS