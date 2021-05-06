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

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
  }];
}

//===----------------------------------------------------------------------===//
// NVVM op definitions
//===----------------------------------------------------------------------===//

class NVVM_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// NVVM intrinsic operations
//===----------------------------------------------------------------------===//

class NVVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<OpTrait> traits,
                  int numResults>
  : LLVM_IntrOpBase<NVVM_Dialect, mnem, "nvvm_" # !subst(".", "_", mnem),
                    overloadedResults, overloadedOperands, traits, numResults>;


//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  NVVM_IntrOp<mnemonic, [], [], !listconcat(traits, [NoSideEffect]), 1>,
  Arguments<(ins)> {
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
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ShflBflyOp :
  NVVM_Op<"shfl.sync.bfly">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$dst,
                 LLVM_Type:$val,
                 LLVM_Type:$offset,
                 LLVM_Type:$mask_and_clamp,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  string llvmBuilder = [{
      auto intId = getShflBflyIntrinsicId(
          $_resultType, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$dst, $val, $offset, $mask_and_clamp});
  }];
  let parser = [{ return parseNVVMShflSyncBflyOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
  let verifier = [{
    if (!(*this)->getAttrOfType<UnitAttr>("return_value_and_is_valid"))
      return success();
    auto type = getType().dyn_cast<LLVM::LLVMStructType>();
    auto elementType = (type && type.getBody().size() == 2)
                     ? type.getBody()[1].dyn_cast<IntegerType>()
                     : nullptr;
    if (!elementType || elementType.getWidth() != 1)
      return emitError("expected return type to be a two-element struct with "
                       "i1 as the second element");
    return success();
  }];
}

def NVVM_VoteBallotOp :
  NVVM_Op<"vote.ballot.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$mask, LLVM_Type:$pred)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {$mask, $pred});
  }];
  let parser = [{ return parseNVVMVoteBallotOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
}

def NVVM_MmaOp :
  NVVM_Op<"mma.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  string llvmBuilder = [{
    $res = createIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f32, $args);
  }];
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
  let verifier = [{ return ::verify(*this); }];
}

// Base class for all the variants of WMMA loadOps that may be defined.
class NVVM_WMMALoadOp<string mnemonic> : NVVM_Op<mnemonic>,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {

  let summary = "Warp synchronous matrix load";

  string baseDescription = [{"The `nvvm.wmma.m*n*k*.load.[a, b, c]` operation"
    "loads a matrix collectively using all the threads in a warp."

    "The operation takes two arguments, the address from where the matrix"
    "elements are to be loaded from and a stride. The stride argument"
    "represents the leading dimension of the source matrix. The address and"
    "the stride are required to be the same across all threads in the warp."
    "Each thread in a warp holds a certain number of elements. The Op returns"
    "a LLVMStruct which holds the elements of the matrix held by this thread."

    "This op is meant to be used along with `nvvm.wmma.m*n*k*.store` and"
    "`nvvm.wmma.m*n*k*.mma`."}];

  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
}

def NVVM_WMMALoadAM16N16K16Op :
  NVVM_WMMALoadOp<"wmma.m16n16k16.load.a.f16.row.stride">{

  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    %2 = nvvm.wmma.m16n16k16.load.a %0, %1 : !llvm.ptr<i32, 3>, !llvm.i32 ->
    !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>,
    vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    ```
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMALoadBM16N16K16Op :
  NVVM_WMMALoadOp<"wmma.m16n16k16.load.b.f16.row.stride">{

  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    %2 = nvvm.wmma.m16n16k16.load.b %0, %1 : !llvm.ptr<i32, 3>, !llvm.i32 ->
    !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>,
    vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    ```
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMALoadCF16M16N16K16Op :
  NVVM_WMMALoadOp<"wmma.m16n16k16.load.c.f16.row.stride">{
  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    %2 = nvvm.wmma.m16n16k16.load.c.f16.row.stride %0, %1 : !llvm.ptr<i32, 3>, !llvm.i32 ->
    !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>
    ```
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMALoadCF32M16N16K16Op :
  NVVM_WMMALoadOp<"wmma.m16n16k16.load.c.f32.row.stride">{
  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
      builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row_stride, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    %2 = nvvm.wmma.m16n16k16.load.c.f32.row.stride %0, %1 : !llvm.ptr<i32, 3>, !llvm.i32 ->
    !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    ```
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

// Base class for all the variants of WMMA storeOps that may be defined.
class NVVM_WMMAStoreOp<string mnemonic> : NVVM_Op<mnemonic>,
  Arguments<(ins Variadic<LLVM_Type>:$args)>{
  let summary = "Warp synchronous matrix store";

  string baseDescription = [{
    The `nvvm.wmma.m*n*k*.store` operation stores a matrix collectively using
    all the threads in a warp.

    The operation takes as arguments the address to where the matrix elements are
    to be stored, a stride and the elements to store, held by the current thread.
    The stride argument represents the leading dimension of the destination matrix.
    The address and the stride are required to be the same across all threads in the
    warp.

    This op is meant to be used along with `nvvm.wmma.m16n16k16.load` and
    `nvvm.wmma.m16n16k16.mma`.
  }];

  let assemblyFormat = "$args attr-dict `:` type($args)";
}

def NVVM_WMMAStoreF16M16N16K16Op : NVVM_WMMAStoreOp<"wmma.m16n16k16.store.d.f16.row.stride"> {
  string llvmBuilder = [{
        createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    nvvm.wmma.m16n16k16.stored.f16.row.stride %0, %1, %2, %3, %4, %5, %6 : !llvm.ptr<i32, 3>,
    !llvm.struct<(vec<2 x half>, vec<2 x half>, vec<2 x half>, vec<2 x half>)>, !llvm.i32
    ```
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMAStoreF32M16N16K16Op : NVVM_WMMAStoreOp<"wmma.m16n16k16.store.d.f32.row.stride"> {
  string llvmBuilder = [{
        createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row_stride, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    nvvm.wmma.m16n16k16.store.d.f32.row.stride %0, %1, %2, %3, %4, %5, %6, %7, %8, %9,
    %10 : !llvm.ptr<i32, 3>, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>,
    !llvm.i32
    ```
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

// Base class for all the variants of WMMA mmaOps that may be defined.
class NVVM_WMMAMmaOp<string mnemonic> : NVVM_Op<mnemonic>,
  Results<(outs LLVM_AnyStruct:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)>{
  let summary = "Warp synchronous matrix-multiply accumulate using tensor cores.";

  string baseDescription = [{
    The `nvvm.wmma.m*n*k*.mma` operation performs a matrix-multiply accumulate
    (mma) operation using all the threads in a warp.

    The operation performed is represented as `D = A * B + C`. The operation takes
    as arguments the elements of the matrices `A`, `B`, `C` and `D`, held by the
    current thread. The op returns a LLVM struct which holds a part of the result
    held by the current thread.

    This op is meant to be used along with `nvvm.wmma.m16n16k16.load` and `nvvm.wmma.
    m16n16k16.store`.
  }];
}

def NVVM_WMMAMmaF16F16M16N16K16Op : NVVM_WMMAMmaOp<"wmma.m16n16k16.mma.row.row.f16.f16">{
  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f16, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    %20 = nvvm.wmma.m16n16k16.mma.row.row.f16.f16 %0, %1, %2, %3, %4, %5, %6, %7, %8,
    %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19 : vector<2xf16> -> !llvm.struct
    <(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
    ```
  }];

  let parser = [{
    return parseWMMAMmaF16F16M16N16K16Op(parser, result);
  }];

  let printer = [{
    printWMMAMmaF16F16M16N16K16Op(p, *this);
  }];

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

def NVVM_WMMAMmaF32F32M16N16K16Op : NVVM_WMMAMmaOp<"wmma.m16n16k16.mma.row.row.f32.f32">{
  string llvmBuilder = [{
    $res = createNvvmIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f32_f32, $args);
  }];

  string opDescription = [{
    Example:

    ```mlir
    %24 = nvvm.wmma.m16n16k16.mma.row.row.f32.f32 %0, %1, %2, %3, %4, %5, %6, %7, %8
    %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23 :
    (vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>,
    vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>,
    vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>,
    vector<2xf16>, f32, f32, f32, f32, f32, f32, f32, f32) -> !llvm.struct<(f32,
    f32, f32, f32, f32, f32, f32, f32)>
    ```
  }];

  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";

  let description = !strconcat(baseDescription, opDescription);

  let verifier = [{ return ::verify(*this); }];
}

#endif // NVVMIR_OPS
