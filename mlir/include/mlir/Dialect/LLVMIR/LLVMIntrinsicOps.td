#ifndef LLVM_INTRINSIC_OPS
#define LLVM_INTRINSIC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Operations that correspond to LLVM intrinsics. With MLIR operation set being
// extendable, there is no reason to introduce a hard boundary between "core"
// operations and intrinsics. However, we systematically prefix them with
// "intr." to avoid potential name clashes.

class LLVM_UnaryIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$in);
}

class LLVM_BinarySameArgsIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b);
}

class LLVM_BinaryIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0,1],
           !listconcat([NoSideEffect], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b);
}

class LLVM_TernarySameArgsIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b, LLVM_Type:$c);
}

class LLVM_CountZerosIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect], traits)> {
  let arguments = (ins LLVM_Type:$in, I<1>:$zero_undefined);
}

def LLVM_CopySignOp : LLVM_BinarySameArgsIntrinsicOp<"copysign">;
def LLVM_CosOp : LLVM_UnaryIntrinsicOp<"cos">;
def LLVM_ExpOp : LLVM_UnaryIntrinsicOp<"exp">;
def LLVM_Exp2Op : LLVM_UnaryIntrinsicOp<"exp2">;
def LLVM_FAbsOp : LLVM_UnaryIntrinsicOp<"fabs">;
def LLVM_FCeilOp : LLVM_UnaryIntrinsicOp<"ceil">;
def LLVM_FFloorOp : LLVM_UnaryIntrinsicOp<"floor">;
def LLVM_FMAOp : LLVM_TernarySameArgsIntrinsicOp<"fma">;
def LLVM_FMulAddOp : LLVM_TernarySameArgsIntrinsicOp<"fmuladd">;
def LLVM_Log10Op : LLVM_UnaryIntrinsicOp<"log10">;
def LLVM_Log2Op : LLVM_UnaryIntrinsicOp<"log2">;
def LLVM_LogOp : LLVM_UnaryIntrinsicOp<"log">;
def LLVM_Prefetch : LLVM_ZeroResultIntrOp<"prefetch", [0]> {
  let arguments = (ins LLVM_Type:$addr, LLVM_Type:$rw, LLVM_Type:$hint,
                   LLVM_Type:$cache);
}
def LLVM_SinOp : LLVM_UnaryIntrinsicOp<"sin">;
def LLVM_SqrtOp : LLVM_UnaryIntrinsicOp<"sqrt">;
def LLVM_PowOp : LLVM_BinarySameArgsIntrinsicOp<"pow">;
def LLVM_PowIOp : LLVM_BinaryIntrinsicOp<"powi">;
def LLVM_BitReverseOp : LLVM_UnaryIntrinsicOp<"bitreverse">;
def LLVM_CountLeadingZerosOp : LLVM_CountZerosIntrinsicOp<"ctlz">;
def LLVM_CountTrailingZerosOp : LLVM_CountZerosIntrinsicOp<"cttz">;
def LLVM_CtPopOp : LLVM_UnaryIntrinsicOp<"ctpop">;
def LLVM_MaxNumOp : LLVM_BinarySameArgsIntrinsicOp<"maxnum">;
def LLVM_MinNumOp : LLVM_BinarySameArgsIntrinsicOp<"minnum">;
def LLVM_MaximumOp : LLVM_BinarySameArgsIntrinsicOp<"maximum">;
def LLVM_MinimumOp : LLVM_BinarySameArgsIntrinsicOp<"minimum">;
def LLVM_SMaxOp : LLVM_BinarySameArgsIntrinsicOp<"smax">;
def LLVM_SMinOp : LLVM_BinarySameArgsIntrinsicOp<"smin">;
def LLVM_UMaxOp : LLVM_BinarySameArgsIntrinsicOp<"umax">;
def LLVM_UMinOp : LLVM_BinarySameArgsIntrinsicOp<"umin">;

def LLVM_MemcpyOp : LLVM_ZeroResultIntrOp<"memcpy", [0, 1, 2]> {
  let arguments = (ins Arg<LLVM_Type,"",[MemWrite]>:$dst, Arg<LLVM_Type,"",[MemRead]>:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}
def LLVM_MemcpyInlineOp : LLVM_ZeroResultIntrOp<"memcpy.inline", [0, 1, 2]> {
  let arguments = (ins Arg<LLVM_Type,"",[MemWrite]>:$dst, Arg<LLVM_Type,"",[MemRead]>:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}
def LLVM_MemmoveOp : LLVM_ZeroResultIntrOp<"memmove", [0, 1, 2]> {
  let arguments = (ins Arg<LLVM_Type,"",[MemWrite]>:$dst, Arg<LLVM_Type,"",[MemRead]>:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}

def LLVM_MemsetOp : LLVM_ZeroResultIntrOp<"memset", [0, 2]> {
  let arguments = (ins Arg<LLVM_Type,"",[MemWrite]>:$dst, LLVM_Type:$val, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}

// Intrinsics with multiple returns.

def LLVM_SAddWithOverflowOp
    : LLVM_IntrOp<"sadd.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_UAddWithOverflowOp
    : LLVM_IntrOp<"uadd.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_SSubWithOverflowOp
    : LLVM_IntrOp<"ssub.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_USubWithOverflowOp
    : LLVM_IntrOp<"usub.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_SMulWithOverflowOp
    : LLVM_IntrOp<"smul.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_UMulWithOverflowOp
    : LLVM_IntrOp<"umul.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}


def LLVM_AssumeOp : LLVM_Op<"intr.assume", []> {
  let arguments = (ins LLVM_Type:$cond);
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn =
        llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::assume, {});
    builder.CreateCall(fn, {$cond});
  }];
}


//
// Coroutine intrinsics.
//

def LLVM_CoroIdOp : LLVM_IntrOp<"coro.id", [], [], [], 1> {
  let arguments = (ins I32:$align,
                       LLVM_i8Ptr:$promise,
                       LLVM_i8Ptr:$coroaddr,
                       LLVM_i8Ptr:$fnaddrs);
  let assemblyFormat = "$align `,` $promise `,` $coroaddr `,` $fnaddrs"
    " attr-dict `:` type($res)";
}

def LLVM_CoroBeginOp : LLVM_IntrOp<"coro.begin", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$token,
                       LLVM_i8Ptr:$mem);
  let assemblyFormat = "$token `,` $mem attr-dict `:` type($res)";
}

def LLVM_CoroSizeOp : LLVM_IntrOp<"coro.size", [0], [], [], 1> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_CoroAlignOp : LLVM_IntrOp<"coro.align", [0], [], [], 1> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_CoroSaveOp : LLVM_IntrOp<"coro.save", [], [], [], 1> {
  let arguments = (ins LLVM_i8Ptr:$handle);
  let assemblyFormat = "$handle attr-dict `:` type($res)";
}

def LLVM_CoroSuspendOp : LLVM_IntrOp<"coro.suspend", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$save,
                       I1:$final);
  let assemblyFormat = "$save `,` $final attr-dict `:` type($res)";
}

def LLVM_CoroEndOp : LLVM_IntrOp<"coro.end", [], [], [], 1> {
  let arguments = (ins LLVM_i8Ptr:$handle,
                       I1:$unwind);
  let assemblyFormat = "$handle `,` $unwind attr-dict `:` type($res)";
}

def LLVM_CoroFreeOp : LLVM_IntrOp<"coro.free", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$id,
                       LLVM_i8Ptr:$handle);
  let assemblyFormat = "$id `,` $handle attr-dict `:` type($res)";
}

def LLVM_CoroResumeOp : LLVM_IntrOp<"coro.resume", [], [], [], 0> {
  let arguments = (ins LLVM_i8Ptr:$handle);
  let assemblyFormat = "$handle attr-dict";
}

//
// Exception handling intrinsics.
//

def LLVM_EhTypeidForOp : LLVM_OneResultIntrOp<"eh.typeid.for"> {
    let arguments = (ins LLVM_i8Ptr:$type_info);
    let assemblyFormat = "$type_info attr-dict `:` type($res)";
}

//
// Stack save/restore intrinsics.
//

def LLVM_StackSaveOp : LLVM_OneResultIntrOp<"stacksave"> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_StackRestoreOp : LLVM_ZeroResultIntrOp<"stackrestore"> {
  let arguments = (ins LLVM_i8Ptr:$ptr);
  let assemblyFormat = "$ptr attr-dict";
}

//
// Vector Reductions.
//

def LLVM_vector_reduce_add : LLVM_VectorReduction<"add">;
def LLVM_vector_reduce_and : LLVM_VectorReduction<"and">;
def LLVM_vector_reduce_mul : LLVM_VectorReduction<"mul">;
def LLVM_vector_reduce_fmax : LLVM_VectorReduction<"fmax">;
def LLVM_vector_reduce_fmin : LLVM_VectorReduction<"fmin">;
def LLVM_vector_reduce_or : LLVM_VectorReduction<"or">;
def LLVM_vector_reduce_smax : LLVM_VectorReduction<"smax">;
def LLVM_vector_reduce_smin : LLVM_VectorReduction<"smin">;
def LLVM_vector_reduce_umax : LLVM_VectorReduction<"umax">;
def LLVM_vector_reduce_umin : LLVM_VectorReduction<"umin">;
def LLVM_vector_reduce_xor : LLVM_VectorReduction<"xor">;

def LLVM_vector_reduce_fadd : LLVM_VectorReductionAcc<"fadd">;
def LLVM_vector_reduce_fmul : LLVM_VectorReductionAcc<"fmul">;

//
// LLVM Matrix operations.
//

/// Create a column major, strided 2-D matrix load, as specified in the LLVM
/// MatrixBuilder.
/// data       - Start address of the matrix read
/// rows       - Number of rows in matrix (must be a constant)
/// isVolatile - True if the load operation is marked as volatile.
/// columns    - Number of columns in matrix (must be a constant)
/// stride     - Space between columns
def LLVM_MatrixColumnMajorLoadOp : LLVM_Op<"intr.matrix.column.major.load"> {
  let arguments = (ins LLVM_Type:$data, LLVM_Type:$stride, I1Attr:$isVolatile,
                   I32Attr:$rows, I32Attr:$columns);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Type *ElemTy = moduleTranslation.convertType(
        getVectorElementType(op.getType()));
    llvm::Align align = dl.getABITypeAlign(ElemTy);
    $res = mb.CreateColumnMajorLoad(
      ElemTy, $data, align, $stride, $isVolatile, $rows,
      $columns);
  }];
  let assemblyFormat = "$data `,` `<` `stride` `=` $stride `>` attr-dict"
    "`:` type($res) `from` type($data) `stride` type($stride)";
}

/// Create a column major, strided 2-D matrix store, as specified in the LLVM
/// MatrixBuilder.
/// matrix     - Matrix to store
/// ptr        - Pointer to write back to
/// isVolatile - True if the load operation is marked as volatile.
/// rows       - Number of rows in matrix (must be a constant)
/// columns    - Number of columns in matrix (must be a constant)
/// stride     - Space between columns
def LLVM_MatrixColumnMajorStoreOp : LLVM_Op<"intr.matrix.column.major.store"> {
  let arguments = (ins LLVM_AnyVector:$matrix, LLVM_Type:$data,
                   LLVM_Type:$stride, I1Attr:$isVolatile, I32Attr:$rows,
                   I32Attr:$columns);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    Type elementType = getVectorElementType(op.getMatrix().getType());
    llvm::Align align = dl.getABITypeAlign(
      moduleTranslation.convertType(elementType));
    mb.CreateColumnMajorStore(
      $matrix, $data, align, $stride, $isVolatile,
      $rows, $columns);
  }];
  let assemblyFormat = "$matrix `,` $data `,` `<` `stride` `=` $stride `>` "
    "attr-dict`:` type($matrix) `to` type($data) `stride` type($stride)";
}

/// Create a llvm.matrix.multiply call, multiplying 2-D matrices LHS and RHS, as
/// specified in the LLVM MatrixBuilder.
def LLVM_MatrixMultiplyOp : LLVM_Op<"intr.matrix.multiply"> {
  let arguments = (ins LLVM_Type:$lhs, LLVM_Type:$rhs, I32Attr:$lhs_rows,
                   I32Attr:$lhs_columns, I32Attr:$rhs_columns);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    $res = mb.CreateMatrixMultiply(
      $lhs, $rhs, $lhs_rows, $lhs_columns,
      $rhs_columns);
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict "
    "`:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

/// Create a llvm.matrix.transpose call, transposing a `rows` x `columns` 2-D
/// `matrix`, as specified in the LLVM MatrixBuilder.
def LLVM_MatrixTransposeOp : LLVM_Op<"intr.matrix.transpose"> {
  let arguments = (ins LLVM_Type:$matrix, I32Attr:$rows, I32Attr:$columns);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    $res = mb.CreateMatrixTranspose(
      $matrix, $rows, $columns);
  }];
  let assemblyFormat = "$matrix attr-dict `:` type($matrix) `into` type($res)";
}

//
// LLVM masked operations.
//

/// Create a llvm.get.active.lane.mask to set a mask up to a given position.
def LLVM_GetActiveLaneMaskOp
    : LLVM_OneResultIntrOp<"get.active.lane.mask", [0], [0], [NoSideEffect]> {
  let arguments = (ins LLVM_Type:$base, LLVM_Type:$n);
  let assemblyFormat = "$base `,` $n attr-dict `:` "
    "type($base) `,` type($n) `to` type($res)";
}

/// Create a call to Masked Load intrinsic.
def LLVM_MaskedLoadOp : LLVM_Op<"intr.masked.load"> {
  let arguments = (ins LLVM_Type:$data, LLVM_Type:$mask,
                   Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment);
  let results = (outs LLVM_AnyVector:$res);
  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedLoad(
        $_resultType, $data, llvm::Align($alignment), $mask) :
      builder.CreateMaskedLoad(
        $_resultType, $data, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Store intrinsic.
def LLVM_MaskedStoreOp : LLVM_Op<"intr.masked.store"> {
  let arguments = (ins LLVM_Type:$value, LLVM_Type:$data, LLVM_Type:$mask,
                   I32Attr:$alignment);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmBuilder = [{
    builder.CreateMaskedStore(
      $value, $data, llvm::Align($alignment), $mask);
  }];
  let assemblyFormat = "$value `,` $data `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($data)";
}

/// Create a call to Masked Gather intrinsic.
def LLVM_masked_gather : LLVM_Op<"intr.masked.gather"> {
  let arguments = (ins LLVM_AnyVector:$ptrs, LLVM_Type:$mask,
                   Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedGather(
        $_resultType, $ptrs, llvm::Align($alignment), $mask) :
      builder.CreateMaskedGather(
        $_resultType, $ptrs, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Scatter intrinsic.
def LLVM_masked_scatter : LLVM_Op<"intr.masked.scatter"> {
  let arguments = (ins LLVM_Type:$value, LLVM_Type:$ptrs, LLVM_Type:$mask,
                   I32Attr:$alignment);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmBuilder = [{
    builder.CreateMaskedScatter(
      $value, $ptrs, llvm::Align($alignment), $mask);
  }];
  let assemblyFormat = "$value `,` $ptrs `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($ptrs)";
}

/// Create a call to Masked Expand Load intrinsic.
def LLVM_masked_expandload : LLVM_IntrOp<"masked.expandload", [0], [], [], 1> {
  let arguments = (ins LLVM_Type, LLVM_Type, LLVM_Type);
}

/// Create a call to Masked Compress Store intrinsic.
def LLVM_masked_compressstore
    : LLVM_IntrOp<"masked.compressstore", [], [0], [], 0> {
  let arguments = (ins LLVM_Type, LLVM_Type, LLVM_Type);
}

/// Create a call to vscale intrinsic.
def LLVM_vscale : LLVM_IntrOp<"vscale", [0], [], [], 1>;

/// Create a call to stepvector intrinsic.
def LLVM_StepVectorOp
    : LLVM_IntrOp<"experimental.stepvector", [0], [], [NoSideEffect], 1> {
  let arguments = (ins);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

//
// LLVM Vector Predication operations.
//

class LLVM_VPBinaryBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [], [NoSideEffect]>,
      Arguments<(ins LLVM_VectorOf<element>:$lhs, LLVM_VectorOf<element>:$rhs,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPBinaryI<string mnem> : LLVM_VPBinaryBase<mnem, AnyInteger>;

class LLVM_VPBinaryF<string mnem> : LLVM_VPBinaryBase<mnem, AnyFloat>;

class LLVM_VPUnaryBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [], [NoSideEffect]>,
      Arguments<(ins LLVM_VectorOf<element>:$op,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPUnaryF<string mnem> : LLVM_VPUnaryBase<mnem, AnyFloat>;

class LLVM_VPTernaryBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [], [NoSideEffect]>,
      Arguments<(ins LLVM_VectorOf<element>:$op1, LLVM_VectorOf<element>:$op2,
                     LLVM_VectorOf<element>:$op3, LLVM_VectorOf<I1>:$mask,
                     I32:$evl)>;

class LLVM_VPTernaryF<string mnem> : LLVM_VPTernaryBase<mnem, AnyFloat>;

class LLVM_VPReductionBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp.reduce." # mnem, [], [1], [NoSideEffect]>,
      Arguments<(ins element:$satrt_value, LLVM_VectorOf<element>:$val,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPReductionI<string mnem> : LLVM_VPReductionBase<mnem, AnyInteger>;

class LLVM_VPReductionF<string mnem> : LLVM_VPReductionBase<mnem, AnyFloat>;

class LLVM_VPSelectBase<string mnem>
    : LLVM_OneResultIntrOp<"vp." # mnem, [], [1], [NoSideEffect]>,
      Arguments<(ins LLVM_VectorOf<I1>:$cond, LLVM_AnyVector:$true_val,
                     LLVM_AnyVector:$false_val, I32:$evl)>;

class LLVM_VPCastBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [0], [NoSideEffect]>,
      Arguments<(ins LLVM_VectorOf<element>:$src,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPCastI<string mnem>   : LLVM_VPCastBase<mnem, AnyInteger>;

class LLVM_VPCastF<string mnem>   : LLVM_VPCastBase<mnem, AnyFloat>;

class LLVM_VPCastPtr<string mnem> : LLVM_VPCastBase<mnem, LLVM_AnyPointer>;

// Integer Binary
def LLVM_VPAddOp  : LLVM_VPBinaryI<"add">;
def LLVM_VPSubOp  : LLVM_VPBinaryI<"sub">;
def LLVM_VPMulOp  : LLVM_VPBinaryI<"mul">;
def LLVM_VPSDivOp : LLVM_VPBinaryI<"sdiv">;
def LLVM_VPUDivOp : LLVM_VPBinaryI<"udiv">;
def LLVM_VPSRemOp : LLVM_VPBinaryI<"srem">;
def LLVM_VPURemOp : LLVM_VPBinaryI<"urem">;
def LLVM_VPAShrOp : LLVM_VPBinaryI<"ashr">;
def LLVM_VPLShrOp : LLVM_VPBinaryI<"lshr">;
def LLVM_VPShlOp  : LLVM_VPBinaryI<"shl">;
def LLVM_VPOrOp   : LLVM_VPBinaryI<"or">;
def LLVM_VPAndOp  : LLVM_VPBinaryI<"and">;
def LLVM_VPXorOp  : LLVM_VPBinaryI<"xor">;

// Float Binary
def LLVM_VPFAddOp : LLVM_VPBinaryF<"fadd">;
def LLVM_VPFSubOp : LLVM_VPBinaryF<"fsub">;
def LLVM_VPFMulOp : LLVM_VPBinaryF<"fmul">;
def LLVM_VPFDivOp : LLVM_VPBinaryF<"fdiv">;
def LLVM_VPFRemOp : LLVM_VPBinaryF<"frem">;

// Float Unary
def LLVM_VPFNegOp : LLVM_VPUnaryF<"fneg">;

// Float Ternary
def LLVM_VPFmaOp  : LLVM_VPTernaryF<"fma">;

// Integer Reduction
def LLVM_VPReduceAddOp  : LLVM_VPReductionI<"add">;
def LLVM_VPReduceMulOp  : LLVM_VPReductionI<"mul">;
def LLVM_VPReduceAndOp  : LLVM_VPReductionI<"and">;
def LLVM_VPReduceOrOp   : LLVM_VPReductionI<"or">;
def LLVM_VPReduceXorOp  : LLVM_VPReductionI<"xor">;
def LLVM_VPReduceSMaxOp : LLVM_VPReductionI<"smax">;
def LLVM_VPReduceSMinOp : LLVM_VPReductionI<"smin">;
def LLVM_VPReduceUMaxOp : LLVM_VPReductionI<"umax">;
def LLVM_VPReduceUMinOp : LLVM_VPReductionI<"umin">;

// Float Reduction
def LLVM_VPReduceFAddOp : LLVM_VPReductionF<"fadd">;
def LLVM_VPReduceFMulOp : LLVM_VPReductionF<"fmul">;
def LLVM_VPReduceFMaxOp : LLVM_VPReductionF<"fmax">;
def LLVM_VPReduceFMinOp : LLVM_VPReductionF<"fmin">;

def LLVM_VPSelectMinOp : LLVM_VPSelectBase<"select">;
def LLVM_VPMergeMinOp  : LLVM_VPSelectBase<"merge">;

// Load/store
def LLVM_VPLoadOp
    : LLVM_OneResultIntrOp<"vp.load", [0], [0], []>,
      Arguments<(ins LLVM_AnyPointer:$ptr,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

def LLVM_VPStoreOp
    : LLVM_ZeroResultIntrOp<"vp.store", [0, 1], []>,
      Arguments<(ins LLVM_AnyVector:$val,
                     LLVM_AnyPointer:$ptr,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

// Strided load/store
def LLVM_VPStridedLoadOp
    : LLVM_OneResultIntrOp<"experimental.vp.strided.load", [0], [0, 1], []>,
      Arguments<(ins LLVM_AnyPointer:$ptr, AnyInteger:$stride,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

def LLVM_VPStridedStoreOp
    : LLVM_ZeroResultIntrOp<"experimental.vp.strided.store",[0, 1, 2], []>,
      Arguments<(ins LLVM_AnyVector:$val, LLVM_AnyPointer:$ptr,
                     AnyInteger:$stride, LLVM_VectorOf<I1>:$mask, I32:$evl)>;

def LLVM_VPTruncOp : LLVM_VPCastI<"trunc">;
def LLVM_VPZExtOp  : LLVM_VPCastI<"zext">;
def LLVM_VPSExtOp  : LLVM_VPCastI<"sext">;

def LLVM_VPFPTruncOp : LLVM_VPCastF<"fptrunc">;
def LLVM_VPFPExtOp   : LLVM_VPCastF<"fpext">;

def LLVM_VPFPToUIOp : LLVM_VPCastF<"fptoui">;
def LLVM_VPFPToSIOp : LLVM_VPCastF<"fptosi">;

def LLVM_VPUIToFPOp : LLVM_VPCastI<"uitofp">;
def LLVM_VPSIToFPOp : LLVM_VPCastI<"sitofp">;

def LLVM_VPPtrToIntOp : LLVM_VPCastPtr<"ptrtoint">;
def LLVM_VPIntToPtrOp : LLVM_VPCastI<"inttoptr">;

#endif // LLVM_INTRINSIC_OPS