/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* MIR dialect generic (G_*) operations                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: mir_gen.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

include "mlir/Dialect/MIR/IR/MIRTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MIR_G_ABDS : MIR_GenericOp<"g_abds", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ABDU : MIR_GenericOp<"g_abdu", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ABS : MIR_GenericOp<"g_abs", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ADD : MIR_GenericOp<"g_add", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ADDRSPACE_CAST : MIR_GenericOp<"g_addrspace_cast", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_AND : MIR_GenericOp<"g_and", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ANYEXT : MIR_GenericOp<"g_anyext", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ASHR : MIR_GenericOp<"g_ashr", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ASSERT_ALIGN : MIR_GenericOp<"g_assert_align", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$align);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ASSERT_SEXT : MIR_GenericOp<"g_assert_sext", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$sz);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ASSERT_ZEXT : MIR_GenericOp<"g_assert_zext", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$sz);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ATOMICRMW_ADD : MIR_GenericOp<"g_atomicrmw_add", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_AND : MIR_GenericOp<"g_atomicrmw_and", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FADD : MIR_GenericOp<"g_atomicrmw_fadd", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FMAX : MIR_GenericOp<"g_atomicrmw_fmax", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FMAXIMUM : MIR_GenericOp<"g_atomicrmw_fmaximum", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FMAXIMUMNUM : MIR_GenericOp<"g_atomicrmw_fmaximumnum", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FMIN : MIR_GenericOp<"g_atomicrmw_fmin", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FMINIMUM : MIR_GenericOp<"g_atomicrmw_fminimum", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FMINIMUMNUM : MIR_GenericOp<"g_atomicrmw_fminimumnum", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_FSUB : MIR_GenericOp<"g_atomicrmw_fsub", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_MAX : MIR_GenericOp<"g_atomicrmw_max", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_MIN : MIR_GenericOp<"g_atomicrmw_min", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_NAND : MIR_GenericOp<"g_atomicrmw_nand", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_OR : MIR_GenericOp<"g_atomicrmw_or", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_SUB : MIR_GenericOp<"g_atomicrmw_sub", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_UDEC_WRAP : MIR_GenericOp<"g_atomicrmw_udec_wrap", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_UINC_WRAP : MIR_GenericOp<"g_atomicrmw_uinc_wrap", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_UMAX : MIR_GenericOp<"g_atomicrmw_umax", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_UMIN : MIR_GenericOp<"g_atomicrmw_umin", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_USUB_COND : MIR_GenericOp<"g_atomicrmw_usub_cond", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_USUB_SAT : MIR_GenericOp<"g_atomicrmw_usub_sat", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_XCHG : MIR_GenericOp<"g_atomicrmw_xchg", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMICRMW_XOR : MIR_GenericOp<"g_atomicrmw_xor", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMIC_CMPXCHG : MIR_GenericOp<"g_atomic_cmpxchg", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$cmpval, MIR_AnyLLT:$newval);
  let results = (outs MIR_AnyLLT:$oldval);
}

def MIR_G_ATOMIC_CMPXCHG_WITH_SUCCESS : MIR_GenericOp<"g_atomic_cmpxchg_with_success", []> {
  let arguments = (ins MIR_AnyLLT:$addr, MIR_AnyLLT:$cmpval, MIR_AnyLLT:$newval);
  let results = (outs MIR_AnyLLT:$oldval, MIR_AnyLLT:$success);
}

def MIR_G_BITCAST : MIR_GenericOp<"g_bitcast", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_BITREVERSE : MIR_GenericOp<"g_bitreverse", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_BLOCK_ADDR : MIR_GenericOp<"g_block_addr", [Pure]> {
  let arguments = (ins AnyAttr:$ba);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_BR : MIR_GenericOp<"g_br", [Pure]> {
  let arguments = (ins AnyAttr:$src1);
  let results = (outs);
}

def MIR_G_BRCOND : MIR_GenericOp<"g_brcond", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$tst, AnyAttr:$truebb);
  let results = (outs);
}

def MIR_G_BRINDIRECT : MIR_GenericOp<"g_brindirect", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs);
}

def MIR_G_BRJT : MIR_GenericOp<"g_brjt", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$tbl, MIR_AnyLLT:$idx, AnyAttr:$jti);
  let results = (outs);
}

def MIR_G_BSWAP : MIR_GenericOp<"g_bswap", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_BUILD_VECTOR : MIR_GenericOp<"g_build_vector", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, Variadic<MIR_AnyLLT>:$varargs);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_BUILD_VECTOR_TRUNC : MIR_GenericOp<"g_build_vector_trunc", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, Variadic<MIR_AnyLLT>:$varargs);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_BZERO : MIR_GenericOp<"g_bzero", []> {
  let arguments = (ins MIR_AnyLLT:$dst_addr, MIR_AnyLLT:$size, AnyAttr:$tailcall);
  let results = (outs);
}

def MIR_G_CLMUL : MIR_GenericOp<"g_clmul", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CONCAT_VECTORS : MIR_GenericOp<"g_concat_vectors", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, Variadic<MIR_AnyLLT>:$varargs);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CONSTANT : MIR_GenericOp<"g_constant", [Pure]> {
  let arguments = (ins AnyAttr:$imm);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CONSTANT_FOLD_BARRIER : MIR_GenericOp<"g_constant_fold_barrier", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CONSTANT_POOL : MIR_GenericOp<"g_constant_pool", [Pure]> {
  let arguments = (ins AnyAttr:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CTLS : MIR_GenericOp<"g_ctls", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CTLZ : MIR_GenericOp<"g_ctlz", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CTLZ_ZERO_POISON : MIR_GenericOp<"g_ctlz_zero_poison", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CTPOP : MIR_GenericOp<"g_ctpop", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CTTZ : MIR_GenericOp<"g_cttz", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_CTTZ_ZERO_POISON : MIR_GenericOp<"g_cttz_zero_poison", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_DEBUGTRAP : MIR_GenericOp<"g_debugtrap", []> {
  let arguments = (ins);
  let results = (outs);
}

def MIR_G_DYN_STACKALLOC : MIR_GenericOp<"g_dyn_stackalloc", []> {
  let arguments = (ins MIR_AnyLLT:$size, AnyAttr:$align);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_EXTRACT : MIR_GenericOp<"g_extract", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$offset);
  let results = (outs MIR_AnyLLT:$res);
}

def MIR_G_EXTRACT_SUBVECTOR : MIR_GenericOp<"g_extract_subvector", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$idx);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_EXTRACT_VECTOR_ELT : MIR_GenericOp<"g_extract_vector_elt", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$idx);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FABS : MIR_GenericOp<"g_fabs", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FACOS : MIR_GenericOp<"g_facos", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FADD : MIR_GenericOp<"g_fadd", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FASIN : MIR_GenericOp<"g_fasin", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FATAN : MIR_GenericOp<"g_fatan", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FATAN2 : MIR_GenericOp<"g_fatan2", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCANONICALIZE : MIR_GenericOp<"g_fcanonicalize", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCEIL : MIR_GenericOp<"g_fceil", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCMP : MIR_GenericOp<"g_fcmp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, AnyAttr:$tst);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCONSTANT : MIR_GenericOp<"g_fconstant", [Pure]> {
  let arguments = (ins AnyAttr:$imm);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCOPYSIGN : MIR_GenericOp<"g_fcopysign", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCOS : MIR_GenericOp<"g_fcos", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FCOSH : MIR_GenericOp<"g_fcosh", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FDIV : MIR_GenericOp<"g_fdiv", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FENCE : MIR_GenericOp<"g_fence", []> {
  let arguments = (ins AnyAttr:$ordering, AnyAttr:$scope);
  let results = (outs);
}

def MIR_G_FEXP : MIR_GenericOp<"g_fexp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FEXP2 : MIR_GenericOp<"g_fexp2", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FEXP10 : MIR_GenericOp<"g_fexp10", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FFLOOR : MIR_GenericOp<"g_ffloor", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FFREXP : MIR_GenericOp<"g_ffrexp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0);
  let results = (outs MIR_AnyLLT:$dst0, MIR_AnyLLT:$dst1);
}

def MIR_G_FLDEXP : MIR_GenericOp<"g_fldexp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FLOG : MIR_GenericOp<"g_flog", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FLOG2 : MIR_GenericOp<"g_flog2", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FLOG10 : MIR_GenericOp<"g_flog10", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMA : MIR_GenericOp<"g_fma", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$src3);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMAD : MIR_GenericOp<"g_fmad", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$src3);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMAXIMUM : MIR_GenericOp<"g_fmaximum", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMAXIMUMNUM : MIR_GenericOp<"g_fmaximumnum", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMAXNUM : MIR_GenericOp<"g_fmaxnum", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMAXNUM_IEEE : MIR_GenericOp<"g_fmaxnum_ieee", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMINIMUM : MIR_GenericOp<"g_fminimum", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMINIMUMNUM : MIR_GenericOp<"g_fminimumnum", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMINNUM : MIR_GenericOp<"g_fminnum", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMINNUM_IEEE : MIR_GenericOp<"g_fminnum_ieee", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FMODF : MIR_GenericOp<"g_fmodf", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst1, MIR_AnyLLT:$dst2);
}

def MIR_G_FMUL : MIR_GenericOp<"g_fmul", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FNEARBYINT : MIR_GenericOp<"g_fnearbyint", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FNEG : MIR_GenericOp<"g_fneg", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPEXT : MIR_GenericOp<"g_fpext", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPEXTLOAD : MIR_GenericOp<"g_fpextload", []> {
  let arguments = (ins MIR_AnyLLT:$addr);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPOW : MIR_GenericOp<"g_fpow", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPOWI : MIR_GenericOp<"g_fpowi", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPTOSI : MIR_GenericOp<"g_fptosi", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPTOSI_SAT : MIR_GenericOp<"g_fptosi_sat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPTOUI : MIR_GenericOp<"g_fptoui", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPTOUI_SAT : MIR_GenericOp<"g_fptoui_sat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPTRUNC : MIR_GenericOp<"g_fptrunc", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FPTRUNCSTORE : MIR_GenericOp<"g_fptruncstore", []> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$addr);
  let results = (outs);
}

def MIR_G_FRAME_INDEX : MIR_GenericOp<"g_frame_index", [Pure]> {
  let arguments = (ins AnyAttr:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FREEZE : MIR_GenericOp<"g_freeze", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FREM : MIR_GenericOp<"g_frem", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FRINT : MIR_GenericOp<"g_frint", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FSHL : MIR_GenericOp<"g_fshl", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$src3);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FSHR : MIR_GenericOp<"g_fshr", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$src3);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FSIN : MIR_GenericOp<"g_fsin", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FSINCOS : MIR_GenericOp<"g_fsincos", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst1, MIR_AnyLLT:$dst2);
}

def MIR_G_FSINH : MIR_GenericOp<"g_fsinh", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FSQRT : MIR_GenericOp<"g_fsqrt", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FSUB : MIR_GenericOp<"g_fsub", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FTAN : MIR_GenericOp<"g_ftan", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_FTANH : MIR_GenericOp<"g_ftanh", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_GET_FPENV : MIR_GenericOp<"g_get_fpenv", []> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_GET_FPMODE : MIR_GenericOp<"g_get_fpmode", []> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_GET_ROUNDING : MIR_GenericOp<"g_get_rounding", []> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_GLOBAL_VALUE : MIR_GenericOp<"g_global_value", [Pure]> {
  let arguments = (ins AnyAttr:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ICMP : MIR_GenericOp<"g_icmp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, AnyAttr:$tst);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_IMPLICIT_DEF : MIR_GenericOp<"g_implicit_def", [Pure]> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INDEXED_LOAD : MIR_GenericOp<"g_indexed_load", []> {
  let arguments = (ins MIR_AnyLLT:$base, MIR_AnyLLT:$offset, AnyAttr:$am);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$newaddr);
}

def MIR_G_INDEXED_SEXTLOAD : MIR_GenericOp<"g_indexed_sextload", []> {
  let arguments = (ins MIR_AnyLLT:$base, MIR_AnyLLT:$offset, AnyAttr:$am);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$newaddr);
}

def MIR_G_INDEXED_STORE : MIR_GenericOp<"g_indexed_store", []> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$base, MIR_AnyLLT:$offset, AnyAttr:$am);
  let results = (outs MIR_AnyLLT:$newaddr);
}

def MIR_G_INDEXED_ZEXTLOAD : MIR_GenericOp<"g_indexed_zextload", []> {
  let arguments = (ins MIR_AnyLLT:$base, MIR_AnyLLT:$offset, AnyAttr:$am);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$newaddr);
}

def MIR_G_INSERT : MIR_GenericOp<"g_insert", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$op, AnyAttr:$offset);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INSERT_SUBVECTOR : MIR_GenericOp<"g_insert_subvector", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$idx);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INSERT_VECTOR_ELT : MIR_GenericOp<"g_insert_vector_elt", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$elt, MIR_AnyLLT:$idx);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC : MIR_GenericOp<"g_intrinsic", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$varargs, AnyAttr:$intrin);
  let results = (outs);
}

def MIR_G_INTRINSIC_CONVERGENT : MIR_GenericOp<"g_intrinsic_convergent", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$varargs, AnyAttr:$intrin);
  let results = (outs);
}

def MIR_G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS : MIR_GenericOp<"g_intrinsic_convergent_w_side_effects", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$varargs, AnyAttr:$intrin);
  let results = (outs);
}

def MIR_G_INTRINSIC_FPTRUNC_ROUND : MIR_GenericOp<"g_intrinsic_fptrunc_round", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, AnyAttr:$round_mode);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC_LLRINT : MIR_GenericOp<"g_intrinsic_llrint", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC_LRINT : MIR_GenericOp<"g_intrinsic_lrint", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC_ROUND : MIR_GenericOp<"g_intrinsic_round", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC_ROUNDEVEN : MIR_GenericOp<"g_intrinsic_roundeven", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC_TRUNC : MIR_GenericOp<"g_intrinsic_trunc", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INTRINSIC_W_SIDE_EFFECTS : MIR_GenericOp<"g_intrinsic_w_side_effects", []> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$varargs, AnyAttr:$intrin);
  let results = (outs);
}

def MIR_G_INTTOPTR : MIR_GenericOp<"g_inttoptr", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_INVOKE_REGION_START : MIR_GenericOp<"g_invoke_region_start", [Pure]> {
  let arguments = (ins);
  let results = (outs);
}

def MIR_G_IS_FPCLASS : MIR_GenericOp<"g_is_fpclass", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$test);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_JUMP_TABLE : MIR_GenericOp<"g_jump_table", [Pure]> {
  let arguments = (ins AnyAttr:$jti);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_LLROUND : MIR_GenericOp<"g_llround", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_LOAD : MIR_GenericOp<"g_load", []> {
  let arguments = (ins MIR_AnyLLT:$addr);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_LROUND : MIR_GenericOp<"g_lround", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_LSHR : MIR_GenericOp<"g_lshr", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_MEMCPY : MIR_GenericOp<"g_memcpy", []> {
  let arguments = (ins MIR_AnyLLT:$dst_addr, MIR_AnyLLT:$src_addr, MIR_AnyLLT:$size, AnyAttr:$tailcall);
  let results = (outs);
}

def MIR_G_MEMCPY_INLINE : MIR_GenericOp<"g_memcpy_inline", []> {
  let arguments = (ins MIR_AnyLLT:$dst_addr, MIR_AnyLLT:$src_addr, MIR_AnyLLT:$size);
  let results = (outs);
}

def MIR_G_MEMMOVE : MIR_GenericOp<"g_memmove", []> {
  let arguments = (ins MIR_AnyLLT:$dst_addr, MIR_AnyLLT:$src_addr, MIR_AnyLLT:$size, AnyAttr:$tailcall);
  let results = (outs);
}

def MIR_G_MEMSET : MIR_GenericOp<"g_memset", []> {
  let arguments = (ins MIR_AnyLLT:$dst_addr, MIR_AnyLLT:$value, MIR_AnyLLT:$size, AnyAttr:$tailcall);
  let results = (outs);
}

def MIR_G_MEMSET_INLINE : MIR_GenericOp<"g_memset_inline", []> {
  let arguments = (ins MIR_AnyLLT:$dst_addr, MIR_AnyLLT:$value, MIR_AnyLLT:$size);
  let results = (outs);
}

def MIR_G_MERGE_VALUES : MIR_GenericOp<"g_merge_values", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, Variadic<MIR_AnyLLT>:$varargs);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_MUL : MIR_GenericOp<"g_mul", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_OR : MIR_GenericOp<"g_or", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_PHI : MIR_GenericOp<"g_phi", [Pure]> {
  let arguments = (ins Variadic<MIR_AnyLLT>:$varargs);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_PREFETCH : MIR_GenericOp<"g_prefetch", []> {
  let arguments = (ins MIR_AnyLLT:$address, AnyAttr:$rw, AnyAttr:$locality, AnyAttr:$cachetype);
  let results = (outs);
}

def MIR_G_PTRAUTH_GLOBAL_VALUE : MIR_GenericOp<"g_ptrauth_global_value", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$addrdisc, AnyAttr:$addr, AnyAttr:$key, AnyAttr:$disc);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_PTRMASK : MIR_GenericOp<"g_ptrmask", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$bits);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_PTRTOINT : MIR_GenericOp<"g_ptrtoint", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_PTR_ADD : MIR_GenericOp<"g_ptr_add", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_READCYCLECOUNTER : MIR_GenericOp<"g_readcyclecounter", []> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_READSTEADYCOUNTER : MIR_GenericOp<"g_readsteadycounter", []> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_READ_REGISTER : MIR_GenericOp<"g_read_register", []> {
  let arguments = (ins AnyAttr:$register_);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_RESET_FPENV : MIR_GenericOp<"g_reset_fpenv", []> {
  let arguments = (ins);
  let results = (outs);
}

def MIR_G_RESET_FPMODE : MIR_GenericOp<"g_reset_fpmode", []> {
  let arguments = (ins);
  let results = (outs);
}

def MIR_G_ROTL : MIR_GenericOp<"g_rotl", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ROTR : MIR_GenericOp<"g_rotr", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SADDE : MIR_GenericOp<"g_sadde", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$carry_in);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_SADDO : MIR_GenericOp<"g_saddo", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_SADDSAT : MIR_GenericOp<"g_saddsat", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SAVGCEIL : MIR_GenericOp<"g_savgceil", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SAVGFLOOR : MIR_GenericOp<"g_savgfloor", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SBFX : MIR_GenericOp<"g_sbfx", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$lsb, MIR_AnyLLT:$width);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SCMP : MIR_GenericOp<"g_scmp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SDIV : MIR_GenericOp<"g_sdiv", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SDIVFIX : MIR_GenericOp<"g_sdivfix", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SDIVFIXSAT : MIR_GenericOp<"g_sdivfixsat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SDIVREM : MIR_GenericOp<"g_sdivrem", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$div, MIR_AnyLLT:$rem);
}

def MIR_G_SELECT : MIR_GenericOp<"g_select", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$tst, MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SET_FPENV : MIR_GenericOp<"g_set_fpenv", []> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs);
}

def MIR_G_SET_FPMODE : MIR_GenericOp<"g_set_fpmode", []> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs);
}

def MIR_G_SET_ROUNDING : MIR_GenericOp<"g_set_rounding", []> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs);
}

def MIR_G_SEXT : MIR_GenericOp<"g_sext", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SEXTLOAD : MIR_GenericOp<"g_sextload", []> {
  let arguments = (ins MIR_AnyLLT:$addr);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SEXT_INREG : MIR_GenericOp<"g_sext_inreg", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, AnyAttr:$sz);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SHL : MIR_GenericOp<"g_shl", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SHUFFLE_VECTOR : MIR_GenericOp<"g_shuffle_vector", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v1, MIR_AnyLLT:$v2, AnyAttr:$mask);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SITOFP : MIR_GenericOp<"g_sitofp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SMAX : MIR_GenericOp<"g_smax", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SMIN : MIR_GenericOp<"g_smin", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SMULFIX : MIR_GenericOp<"g_smulfix", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SMULFIXSAT : MIR_GenericOp<"g_smulfixsat", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SMULH : MIR_GenericOp<"g_smulh", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SMULO : MIR_GenericOp<"g_smulo", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_SPLAT_VECTOR : MIR_GenericOp<"g_splat_vector", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$val);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SREM : MIR_GenericOp<"g_srem", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SSHLSAT : MIR_GenericOp<"g_sshlsat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SSUBE : MIR_GenericOp<"g_ssube", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$carry_in);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_SSUBO : MIR_GenericOp<"g_ssubo", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_SSUBSAT : MIR_GenericOp<"g_ssubsat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STACKRESTORE : MIR_GenericOp<"g_stackrestore", []> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs);
}

def MIR_G_STACKSAVE : MIR_GenericOp<"g_stacksave", []> {
  let arguments = (ins);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STEP_VECTOR : MIR_GenericOp<"g_step_vector", [Pure]> {
  let arguments = (ins AnyAttr:$step);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STORE : MIR_GenericOp<"g_store", []> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$addr);
  let results = (outs);
}

def MIR_G_STRICT_FADD : MIR_GenericOp<"g_strict_fadd", [Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FCMP : MIR_GenericOp<"g_strict_fcmp", []> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, AnyAttr:$tst);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FCMPS : MIR_GenericOp<"g_strict_fcmps", []> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, AnyAttr:$tst);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FDIV : MIR_GenericOp<"g_strict_fdiv", []> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FLDEXP : MIR_GenericOp<"g_strict_fldexp", []> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FMA : MIR_GenericOp<"g_strict_fma", []> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$src3);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FMUL : MIR_GenericOp<"g_strict_fmul", [Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FREM : MIR_GenericOp<"g_strict_frem", []> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FSQRT : MIR_GenericOp<"g_strict_fsqrt", []> {
  let arguments = (ins MIR_AnyLLT:$src1);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_STRICT_FSUB : MIR_GenericOp<"g_strict_fsub", []> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_SUB : MIR_GenericOp<"g_sub", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_TRAP : MIR_GenericOp<"g_trap", []> {
  let arguments = (ins);
  let results = (outs);
}

def MIR_G_TRUNC : MIR_GenericOp<"g_trunc", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_TRUNC_SSAT_S : MIR_GenericOp<"g_trunc_ssat_s", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_TRUNC_SSAT_U : MIR_GenericOp<"g_trunc_ssat_u", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_TRUNC_USAT_U : MIR_GenericOp<"g_trunc_usat_u", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UADDE : MIR_GenericOp<"g_uadde", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$carry_in);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_UADDO : MIR_GenericOp<"g_uaddo", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_UADDSAT : MIR_GenericOp<"g_uaddsat", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UAVGCEIL : MIR_GenericOp<"g_uavgceil", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UAVGFLOOR : MIR_GenericOp<"g_uavgfloor", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UBFX : MIR_GenericOp<"g_ubfx", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src, MIR_AnyLLT:$lsb, MIR_AnyLLT:$width);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UBSANTRAP : MIR_GenericOp<"g_ubsantrap", []> {
  let arguments = (ins AnyAttr:$kind);
  let results = (outs);
}

def MIR_G_UCMP : MIR_GenericOp<"g_ucmp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UDIV : MIR_GenericOp<"g_udiv", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UDIVFIX : MIR_GenericOp<"g_udivfix", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UDIVFIXSAT : MIR_GenericOp<"g_udivfixsat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UDIVREM : MIR_GenericOp<"g_udivrem", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$div, MIR_AnyLLT:$rem);
}

def MIR_G_UITOFP : MIR_GenericOp<"g_uitofp", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UMAX : MIR_GenericOp<"g_umax", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UMIN : MIR_GenericOp<"g_umin", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UMULFIX : MIR_GenericOp<"g_umulfix", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UMULFIXSAT : MIR_GenericOp<"g_umulfixsat", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src0, MIR_AnyLLT:$src1, AnyAttr:$scale);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UMULH : MIR_GenericOp<"g_umulh", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_UMULO : MIR_GenericOp<"g_umulo", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_UNMERGE_VALUES : MIR_GenericOp<"g_unmerge_values", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst0, Variadic<MIR_AnyLLT>:$outs);
}

def MIR_G_UREM : MIR_GenericOp<"g_urem", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_USHLSAT : MIR_GenericOp<"g_ushlsat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_USUBE : MIR_GenericOp<"g_usube", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2, MIR_AnyLLT:$carry_in);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_USUBO : MIR_GenericOp<"g_usubo", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst, MIR_AnyLLT:$carry_out);
}

def MIR_G_USUBSAT : MIR_GenericOp<"g_usubsat", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VAARG : MIR_GenericOp<"g_vaarg", []> {
  let arguments = (ins MIR_AnyLLT:$list, AnyAttr:$align);
  let results = (outs MIR_AnyLLT:$val);
}

def MIR_G_VASTART : MIR_GenericOp<"g_vastart", []> {
  let arguments = (ins MIR_AnyLLT:$list);
  let results = (outs);
}

def MIR_G_VECREDUCE_ADD : MIR_GenericOp<"g_vecreduce_add", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_AND : MIR_GenericOp<"g_vecreduce_and", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_FADD : MIR_GenericOp<"g_vecreduce_fadd", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_FMAX : MIR_GenericOp<"g_vecreduce_fmax", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_FMAXIMUM : MIR_GenericOp<"g_vecreduce_fmaximum", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_FMIN : MIR_GenericOp<"g_vecreduce_fmin", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_FMINIMUM : MIR_GenericOp<"g_vecreduce_fminimum", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_FMUL : MIR_GenericOp<"g_vecreduce_fmul", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_MUL : MIR_GenericOp<"g_vecreduce_mul", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_OR : MIR_GenericOp<"g_vecreduce_or", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_SEQ_FADD : MIR_GenericOp<"g_vecreduce_seq_fadd", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$acc, MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_SEQ_FMUL : MIR_GenericOp<"g_vecreduce_seq_fmul", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$acc, MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_SMAX : MIR_GenericOp<"g_vecreduce_smax", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_SMIN : MIR_GenericOp<"g_vecreduce_smin", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_UMAX : MIR_GenericOp<"g_vecreduce_umax", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_UMIN : MIR_GenericOp<"g_vecreduce_umin", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECREDUCE_XOR : MIR_GenericOp<"g_vecreduce_xor", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$v);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VECTOR_COMPRESS : MIR_GenericOp<"g_vector_compress", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$vec, MIR_AnyLLT:$mask, MIR_AnyLLT:$passthru);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_VSCALE : MIR_GenericOp<"g_vscale", [Pure]> {
  let arguments = (ins AnyAttr:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_WRITE_REGISTER : MIR_GenericOp<"g_write_register", []> {
  let arguments = (ins MIR_AnyLLT:$value, AnyAttr:$register_);
  let results = (outs);
}

def MIR_G_XOR : MIR_GenericOp<"g_xor", [Pure, Commutative]> {
  let arguments = (ins MIR_AnyLLT:$src1, MIR_AnyLLT:$src2);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ZEXT : MIR_GenericOp<"g_zext", [Pure]> {
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
}

def MIR_G_ZEXTLOAD : MIR_GenericOp<"g_zextload", []> {
  let arguments = (ins MIR_AnyLLT:$addr);
  let results = (outs MIR_AnyLLT:$dst);
}

// Emitted 279 ops, skipped 0.
