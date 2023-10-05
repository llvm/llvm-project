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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

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
  let summary = "The GENX dialect in MLIR.";
  let description = [{
    GENX is a dialect for representing operations on Intel GPUs and extensions 
    to the LLVM type system.
  }];
  let cppNamespace = "::mlir::GENX";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;  

  let extraClassDeclaration = [{
    void registerAttributes();    
    void registerTypes();

    /// Get the name of the attribute used to annotate external kernel functions.
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

/// Enum attribute of the different memory fence flags.
def MemoryFenceFlag : I32BitEnumAttr<"MemoryFenceFlag", "GENX memory fence flag",
  [
    I32BitEnumAttrCaseBit<"LOCAL_MEM_FENCE",  0>,
    I32BitEnumAttrCaseBit<"GLOBAL_MEM_FENCE", 1>,
    I32BitEnumAttrCaseBit<"IMAGE_MEM_FENCE",  2>
  ]> {
  let separator = ", ";
  let cppNamespace = "::mlir::GENX";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}
def MemoryFenceFlagAttr : EnumAttr<GENX_Dialect, MemoryFenceFlag, "memory_fence_flag"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Enum attribute of the different memory order.
def MemoryOrder : I32EnumAttr<"MemoryOrder", "GENX memory order",
  [
    I32EnumAttrCase<"Relaxed",                 0>,
    I32EnumAttrCase<"Acquire",                 2>,
    I32EnumAttrCase<"Release",                 3>,
    I32EnumAttrCase<"AcquireRelease",          4>,
    I32EnumAttrCase<"SequentiallyConsistent",  5>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def MemoryOrderAttr : EnumAttr<GENX_Dialect, MemoryOrder, "memory_order"> {
  let assemblyFormat = "`<` $value `>`";
}

/// Enum attribute of the different memory scope.
def MemoryScope : I32EnumAttr<"MemoryScope", "GENX memory scope",
  [
    I32EnumAttrCase<"work_item",        0>,
    I32EnumAttrCase<"work_group",       1>,
    I32EnumAttrCase<"device",           2>,
    I32EnumAttrCase<"all_svm_devices",  3>,
    I32EnumAttrCase<"sub_group",        4>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def MemoryScopeAttr : EnumAttr<GENX_Dialect, MemoryScope, "memory_scope"> {
  let assemblyFormat = "`<` $value `>`";
}

def GENX_FenceOp : GENX_Op<"atomic_work_item_fence">,
  Arguments<(ins MemoryFenceFlagAttr:$flags,
                 MemoryOrderAttr:$order,
                 MemoryScopeAttr:$scope)> {
  let summary = "Fence of memory accesses";
  string baseDescription = [{
    Ensures that memory accesses issued before this instruction will be observed
    before memory accesses issued after this instruction. Function arguments
    determine what kind of memory is being controlled and what kind of control
    to apply.
    $flags must be set to MemoryFenceFlag or a combination of them.
    $order determines the type of fence (e.g., acquire, release, etc.).
    $scope control the extent a fence is visible with respect to the memory model.
  }];
  string llvmBuilder = [{
    createFence(builder, $flags, $order, $scope);
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}

/// Enum attribute of the different shuffle kinds.
def ShflKind : I32EnumAttr<"ShflKind", "GENX shuffle kind",
  [
    I32EnumAttrCase<"XOR",  0>,
    I32EnumAttrCase<"UP",   1>,
    I32EnumAttrCase<"DOWN", 2>,
    I32EnumAttrCase<"IDX",  3>
  ]> {
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

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def GENX_AtomicCmpXchgType : AnyTypeOf<[AnyI32, AnyI64]>;

def GENX_AtomicCmpXchgOp : GENX_Op<"atomic.cmpxchg", [
      TypesMatchWith<"operand #1 and operand #2 have the same type",
                     "cmp", "val", "$_self">,
      TypesMatchWith<"result #0 and operand #1 have the same type",
                     "res", "cmp", "$_self">]>,
  Results<(outs GENX_AtomicCmpXchgType:$res)>,
  Arguments<(ins Arg<LLVM_PointerTo<GENX_AtomicCmpXchgType>,"", [MemRead]>:$ptr,
             GENX_AtomicCmpXchgType:$cmp, GENX_AtomicCmpXchgType:$val)> {
  let summary = "Atomic compare and exchange";

  string baseDescription = [{
    The 'genx.atomic.cmpxchg' operation performs an atomic compare-and-swap at 
    the memory location specified by the pointer argument. 
    The semantics are:
      atomic {
        old = *ptr; 
        *ptr = (old == cmp) ? cmp : val 
        return old
      }  
  }];

  string llvmBuilder = [{
    $res = createAtomicCmpXchg(builder, $ptr, $cmp, $val);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

/// Enum attribute of the different read-modify-write operation kinds.
def RMWOpKind : I32EnumAttr<"RMWOpKind", "GENX RMW operation kind",
  [
    I32EnumAttrCase<"AND",  0>,
    I32EnumAttrCase<"OR",   1>,
    I32EnumAttrCase<"XOR",  2>,
    I32EnumAttrCase<"ADD",  3>,
    I32EnumAttrCase<"MIN",  4>,    
    I32EnumAttrCase<"MAX",  5>,
    I32EnumAttrCase<"XCHG", 6>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def RMWOpKindAttr : EnumAttr<GENX_Dialect, RMWOpKind, "RMW_op_kind">;

def GENX_AtomicRMWType : AnyTypeOf<[AnyI32, AnyI64]>;

def GENX_AtomicRMWOp : GENX_Op<"atomic.rmw", [
      TypesMatchWith<"result #0 and operand #1 have the same type",
                     "res", "val", "$_self">]>,
  Results<(outs GENX_AtomicRMWType:$res)>,
  Arguments<(ins Arg<LLVM_PointerTo<GENX_AtomicRMWType>,"", [MemRead, MemWrite]>:$ptr,
             GENX_AtomicRMWType:$val, RMWOpKindAttr:$bin_op)> {
  let summary = "Atomic read-modify-write";

  string baseDescription = [{
    The 'genx.atomic.rmw' operation performs an atomic read-modify-write at 
    the memory location specified by the pointer argument. 
  }];

  string llvmBuilder = [{
    $res = createAtomicRMW(builder, $ptr, $val, $bin_op);
  }];

  let assemblyFormat = [{
    $bin_op $ptr `,` $val attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Joint Matrix operations
//===----------------------------------------------------------------------===//

def GENX_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

def GENX_MatrixDPASOp : GENX_Op<"matrix.dpas">,
  Results<(outs FixedVectorOf<[GENX_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOf<[GENX_MatrixElemType]>:$c,
    FixedVectorOf<[GENX_MatrixElemType]>:$a,
    FixedVectorOf<[GENX_MatrixElemType]>:$b,
    I32Attr:$pa,
    I32Attr:$pb,
    I32Attr:$sd,
    I32Attr:$rc
  )> {

  let summary = "GENX matrix multiply-add";

  string baseDescription = [{
    The 'genx.matrix.dpas' operation is a matrix multiply-add operation as follows:

       D = C + A x B

       where
          D : MxN
          C : MxN
          A : MxK
          B : KxN

          M : repeat count ($rc)
          N : fixed execution size, must be 16 for PVC
          K : depth * OPS_PER_CHAN
              OPS_PER_CHAN
                 1 : for TF32
                 2 : for 16-bit precision(BF, HF)
                 4 : for 8-bit precision (FP8, UB, B)
                 8 : for less-then 8 bit precision (U4/S4, U2/S2).

              If depth is 8, K would be 8, 16, 32, or 64 (based on OPS_PER_CHAN).

    $a, $b, $c, $d - matrix A, B, C, D, respectively
    $pa, $pb - precision of matrix A and B resepectively
    $sd - systolic depth
    $rc - repect count
  }];

  string llvmBuilder = [{
    $d = createGenISADPAS(op, builder, moduleTranslation);
  }];
 
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GENX_Matrix2DBlockLoadOp : GENX_Op<"matrix.2Dblockload">,
  Results<(outs FixedVectorOf<[GENX_MatrixElemType]>:$res)>,
  Arguments<(ins
    Arg<LLVM_PointerTo<GENX_MatrixElemType>, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform
  )> {

  let summary = "GENX 2D block load";

  string baseDescription = [{
    The 'genx.matrix.2Dblockload' operation loads a submatrix from an array in memory.
    $ptr - the base address of the memory array
    $base_width, $base_height, $base_pitch - the shape of the memory array
    $x, $y, $tile_width, $tile_height - the starting offets and shape of the submatrix to load
    $elem_size_in_bits - 32 for f32, bf32; 16 for f16, int16, bf16; 8 for int8, int4, int2 and etc
    $v_blocks - number of blocks to load
    $transpose - transpose the submatrix in vector register (useful for 32 bit element types)
    $vnni_transform - transpose and pack the submatrix in register (useful for < 32 bit element types)

    The $transpose and $vnni_transform are mutual exclusive. These transformations are used for
    the B matrix in MMA (DPAS) operations D = C + A * B, where A should have row-major layout in register
    and B should have column-major layout.

    If the submatrix contains out of bound elements of the memory array, they are filled with 0.
  }];

  string llvmBuilder = [{
    $res = createGenISA2DBlockRead(op, builder, moduleTranslation);
  }];
 
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GENX_Matrix2DBlockStoreOp : GENX_Op<"matrix.2Dblockstore">,
  Arguments<(ins
    Arg<LLVM_PointerTo<GENX_MatrixElemType>, "", [MemWrite]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    FixedVectorOf<[GENX_MatrixElemType]>:$stored_val
  )> {

  let summary = "GENX 2D block store";

  string baseDescription = [{
    The 'genx.matrix.2Dblockstore' operation stores to a submatrix from an array in memory.
    $ptr - the base address of the memory array
    $base_width, $base_height, $base_pitch - the shape of the memory array
    $x, $y, $tile_width, $tile_height - the starting offets and shape of the submatrix to load
    $elem_size_in_bits - 32 for f32, bf32; 16 for f16, int16, bf16; 8 for int8, int4, int2 and etc
    $v_blocks - number of blocks to store
    $transpose - transpose the submatrix in vector register (useful for 32 bit element types)
    $vnni_transform - transpose and pack the submatrix in register (useful for < 32 bit element types)
    $stored_val - block to store

    The $transpose and $vnni_transform are mutual exclusive. These transformations are used for
    the B matrix in MMA (DPAS) operations D = C + A * B, where A should have row-major layout in register
    and B should have column-major layout.

    If the submatrix contains out of bound elements of the memory array, they are filled with 0.
  }];

  string llvmBuilder = [{
    createGenISA2DBlockWrite(op, builder, moduleTranslation);
  }];
 
  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

def GENX_IsJointMatrixType :
  CPred<"$_self.isa<::mlir::GENX::JointMatrixType>()">;

def GENX_AnyJointMatrix : DialectType<GENX_Dialect, GENX_IsJointMatrixType,
                                "any GENX joint matrix type">;

def ScopeKind : I32EnumAttr<"Scope", "GENX Scope", 
  [
    I32EnumAttrCase<"Device", 0>,
    I32EnumAttrCase<"Workgroup", 1>,
    I32EnumAttrCase<"Subgroup", 2>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def ScopeAttr : EnumAttr<GENX_Dialect, ScopeKind, "scope"> {
  let assemblyFormat = "`<` $value `>`";
}

def MatrixLayoutKind : I32EnumAttr<"MatrixLayout", "GENX Matrix Layout", 
  [
    I32EnumAttrCase<"RowMajor", 0>,    
    I32EnumAttrCase<"ColumnMajor", 1>,
    I32EnumAttrCase<"Packed", 2>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def MatrixLayoutAttr : EnumAttr<GENX_Dialect, MatrixLayoutKind, "layout"> {
  let assemblyFormat = "`<` $value `>`";
}

def MemoryAccessKind : I32BitEnumAttr<"MemoryAccess", "GENX Memory Access", 
  [
    I32BitEnumAttrCaseBit<"None", 0>,
    I32BitEnumAttrCaseBit<"Volatile", 1>,
    I32BitEnumAttrCaseBit<"Aligned", 2>,
    I32BitEnumAttrCaseBit<"NonTemporal", 3>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::GENX";
}
def MemoryAccessAttr : EnumAttr<GENX_Dialect, MemoryAccessKind, "memory_access"> {
  let assemblyFormat = "`<` $value `>`";
}

def GENX_MatrixLoadOp : GENX_Op<"matrix.load">,
  Results<(outs GENX_AnyJointMatrix:$res)>,
  Arguments<(ins Arg<LLVM_AnyPointer,"", [MemRead]>:$ptr,
             Index:$stride, MatrixLayoutAttr:$layout, ScopeAttr:$scope,
             OptionalAttr<MemoryAccessAttr>:$memory_access)> {
  let summary = "Load a joint matrix through a pointer";

  string baseDescription = [{
    The `genx.matrix.load` operation performs a load of the matrix at the memory 
    location specified by the pointer argument $ptr. The $stride argument specifies 
    the number of elements in memory between consecutive rows (for the row major layout) 
    and columns (for the column major layout) in the result. The matrix elements loaded 
    from memory are arranged according to the specified $layout. The $scope indicates 
    the synchronization scope for operations on the matrix.
  }];

  string llvmBuilder = [{
    createMatrixLoad(builder, $res, $ptr, $stride, $layout, $scope, $memory_access);
  }];

  let assemblyFormat = [{
    $scope $layout operands attr-dict `:` `(` type(operands) `)` `->` type($res)
  }];

  let hasVerifier = 1;
}

def GENX_MatrixStoreOp : GENX_Op<"matrix.store">,
  Arguments<(ins Arg<LLVM_AnyPointer,"", [MemWrite]>:$ptr, GENX_AnyJointMatrix:$val, 
             Index:$stride, MatrixLayoutAttr:$layout, ScopeAttr:$scope,
             OptionalAttr<MemoryAccessAttr>:$memory_access)> {
  let summary = "Store a matrix through a pointer";

  string baseDescription = [{
    The `genx.matrix.store` operation stores the joint matrix value $val at the memory
    location specified by the pointer argument $ptr. The $stride argument specifies 
    the number of elements in memory between consecutive rows (for the row major layout) 
    and columns (for the column major layout) in the result. The matrix elements stored
    to memory are arranged according to the specified $layout. The $scope indicates the 
    synchronization scope for operations on the matrix.
  }];

  string llvmBuilder = [{
    createMatrixStore(builder, $ptr, $val, $stride, $layout, $scope, $memory_access);
  }];

  let assemblyFormat = [{
    $scope $layout operands attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;  
}

def GENX_MatrixMadOp : GENX_Op<"matrix.mad", [Pure, AllTypesMatch<["c", "res"]>]>,
  Results<(outs GENX_AnyJointMatrix:$res)>,
  Arguments<(ins GENX_AnyJointMatrix:$a, GENX_AnyJointMatrix:$b, GENX_AnyJointMatrix:$c, 
            ScopeAttr:$scope)> {
  let summary = "Multiply and add operation on joint matrices";

  string baseDescription = [{
    The `genx.matrix.mad` operation multiplies matrices $a and $b and adds matrix $c to 
    the result. Matrix $a is a M x K matrix, $b is a K x N matrix and $c is a M x N matrix. 
    The scope operand indicates the synchronization scope for operations on the matrix.
  }];

  string llvmBuilder = [{
    $res = createMatrixMad(builder, $a, $b, $c, $scope);
  }];

  let assemblyFormat = [{
    $scope operands attr-dict`:` type(operands) `->` type($res)
  }];

  let hasVerifier = 1;  
}

def GENX_MatrixInitOp : GENX_Op<"matrix.init">,
  Arguments<(ins GENX_AnyJointMatrix:$mat, IntegerOrFloatType:$val, ScopeAttr:$scope)> {
  let summary = "Initialize a joint matrix";

  string baseDescription = [{
    The `genx.matrix.init` operation initializes the matrix $mat by filling it with 
    the value $val. The scope operand indicates the synchronization scope for operations 
    on the matrix.
  }];

  string llvmBuilder = [{
    createMatrixInit(builder, $mat, $val, $scope);
  }];

  let assemblyFormat = [{
    $scope operands attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

def GENX_MatrixCopyOp : GENX_Op<"matrix.copy">,
  Results<(outs GENX_AnyJointMatrix:$res)>,
  Arguments<(ins GENX_AnyJointMatrix:$src, ScopeAttr:$scope)> {
  let summary = "Copy a joint matrix";

  string baseDescription = [{
    The `genx.matrix.copy` operation copies the input matrix $src and returns the 
    new matrix. The result type must be a joint matrix with the same shape, and layout 
    as the source matrix. The element type might be different.
  }];

  string llvmBuilder = [{
    $res = createMatrixCopy(builder, $res, $src, $scope);
  }];

  let assemblyFormat = [{
    $scope operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GENX_YieldOp : GENX_Op<"yield", [Pure, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Terminates and yields values from GENX regions";

  string baseDescription = [{
    The `genx.yield` operation yields one or more SSA value from a GENX dialect op 
    region and terminates the region. The operands of this operation must correspond 
    to the parent operation results.
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

def GENX_MatrixMapOp : GENX_Op<"matrix.map", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        RecursiveMemoryEffects,
        SingleBlockImplicitTerminator<"YieldOp">,
        TypesMatchWith<"result #0 and operand #0 have the same type",
                       "res", "mat", "$_self">]>,
    Results<(outs GENX_AnyJointMatrix:$res)>,    
    Arguments<(ins GENX_AnyJointMatrix:$mat, Variadic<IntegerOrFloatType>:$inputs, ScopeAttr:$scope)> {
  let summary = "Elementwise operation on a joint matrix";

  string baseDescription = [{
    The `genx.matrix.map` operation performs an element-wise operation on the joint 
    matrix $mat. The region attached to the operation should contain the workload and 
    it is applied to each element of the matrix.

    Example:

    %res = genx.matrix.map <Subgroup>
    ins(%mat, %val : !genx.jointmatrix<8x32xf32, RowMajor>, f32)
    (%elem: f32, %v: f32) {
       %0 = arith.addf %elem, %v : f32
       genx.yield %0 : f32
    } : !genx.jointmatrix<8x32xf32, RowMajor>
  }];

  string llvmBuilder = [{
    assert(false); // TODO
  }];

  let builders = [
    OpBuilder<(ins "Value":$mat, "ValueRange":$inputs, 
      "function_ref<void(OpBuilder &, Location, ValueRange)>",
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let regions = (region SizedRegion<1>:$mapper);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // GENXIR_OPS
