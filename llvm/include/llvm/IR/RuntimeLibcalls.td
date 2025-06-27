//===-- llvm/RuntimeLibcallList.td - Describe libcalls --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/IR/RuntimeLibcallsImpl.td"

//--------------------------------------------------------------------
// Declare all kinds of used libcalls
//--------------------------------------------------------------------

// Integer
foreach IntTy = ["I16", "I32", "I64", "I128"] in {
  def SHL_#IntTy : RuntimeLibcall;
  def SRL_#IntTy : RuntimeLibcall;
  def SRA_#IntTy : RuntimeLibcall;
}

foreach IntTy = ["I8", "I16", "I32", "I64", "I128"] in {
  def MUL_#IntTy : RuntimeLibcall;
}

foreach IntTy = ["I32", "I64", "I128" ] in {
  def MULO_#IntTy : RuntimeLibcall;
}

foreach IntTy = ["I8", "I16", "I32", "I64", "I128"] in {
  def SDIV_#IntTy : RuntimeLibcall;
  def UDIV_#IntTy : RuntimeLibcall;
  def SREM_#IntTy : RuntimeLibcall;
  def UREM_#IntTy : RuntimeLibcall;
  def SDIVREM_#IntTy : RuntimeLibcall;
  def UDIVREM_#IntTy : RuntimeLibcall;
}

foreach IntTy = ["I32", "I64" ] in {
  def NEG_#IntTy : RuntimeLibcall;
}

foreach IntTy = ["I32", "I64", "I128"] in {
  def CTLZ_#IntTy : RuntimeLibcall;
  def CTPOP_#IntTy : RuntimeLibcall;
}

foreach FPTy = ["F32", "F64", "F80", "F128", "PPCF128"] in {
  def ADD_#FPTy : RuntimeLibcall;
  def SUB_#FPTy : RuntimeLibcall;
  def MUL_#FPTy : RuntimeLibcall;
  def DIV_#FPTy : RuntimeLibcall;
  def REM_#FPTy : RuntimeLibcall;
  def FMA_#FPTy : RuntimeLibcall;
  def POWI_#FPTy : RuntimeLibcall;
  def SQRT_#FPTy : RuntimeLibcall;
  def CBRT_#FPTy : RuntimeLibcall;
  def LOG_#FPTy : RuntimeLibcall;
  def LOG_FINITE_#FPTy : RuntimeLibcall;
  def LOG2_#FPTy : RuntimeLibcall;
  def LOG2_FINITE_#FPTy : RuntimeLibcall;
  def LOG10_#FPTy : RuntimeLibcall;
  def LOG10_FINITE_#FPTy : RuntimeLibcall;
  def EXP_#FPTy : RuntimeLibcall;
  def EXP_FINITE_#FPTy : RuntimeLibcall;
  def EXP2_#FPTy : RuntimeLibcall;
  def EXP2_FINITE_#FPTy : RuntimeLibcall;
  def EXP10_#FPTy : RuntimeLibcall;
  def EXP10_FINITE_#FPTy : RuntimeLibcall;
  def SIN_#FPTy : RuntimeLibcall;
  def COS_#FPTy : RuntimeLibcall;
  def TAN_#FPTy : RuntimeLibcall;
  def SINH_#FPTy : RuntimeLibcall;
  def COSH_#FPTy : RuntimeLibcall;
  def TANH_#FPTy : RuntimeLibcall;
  def ASIN_#FPTy : RuntimeLibcall;
  def ACOS_#FPTy : RuntimeLibcall;
  def ATAN_#FPTy : RuntimeLibcall;
  def ATAN2_#FPTy : RuntimeLibcall;
  def SINCOS_#FPTy : RuntimeLibcall;
}

foreach FPTy = [ "F32", "F64" ] in {
  def SINCOS_STRET_#FPTy : RuntimeLibcall;
}

foreach FPTy = ["F32", "F64", "F80", "F128", "PPCF128"] in {
  def POW_#FPTy : RuntimeLibcall;
  def POW_FINITE_#FPTy : RuntimeLibcall;
  def CEIL_#FPTy : RuntimeLibcall;
  def TRUNC_#FPTy : RuntimeLibcall;
  def RINT_#FPTy : RuntimeLibcall;
  def NEARBYINT_#FPTy : RuntimeLibcall;
  def ROUND_#FPTy : RuntimeLibcall;
  def ROUNDEVEN_#FPTy : RuntimeLibcall;
  def FLOOR_#FPTy : RuntimeLibcall;
  def COPYSIGN_#FPTy : RuntimeLibcall;
  def FMIN_#FPTy : RuntimeLibcall;
  def FMAX_#FPTy : RuntimeLibcall;
  def FMINIMUM_#FPTy : RuntimeLibcall;
  def FMAXIMUM_#FPTy : RuntimeLibcall;
  def FMINIMUM_NUM_#FPTy : RuntimeLibcall;
  def FMAXIMUM_NUM_#FPTy : RuntimeLibcall;
  def LROUND_#FPTy : RuntimeLibcall;
  def LLROUND_#FPTy : RuntimeLibcall;
  def LRINT_#FPTy : RuntimeLibcall;
  def LLRINT_#FPTy : RuntimeLibcall;
  def LDEXP_#FPTy : RuntimeLibcall;
  def FREXP_#FPTy : RuntimeLibcall;
  def SINCOSPI_#FPTy : RuntimeLibcall;
  def MODF_#FPTy : RuntimeLibcall;
}

def FEGETENV : RuntimeLibcall;
def FESETENV : RuntimeLibcall;

def FEGETMODE : RuntimeLibcall;
def FESETMODE : RuntimeLibcall;

def FPEXT_BF16_F32 : RuntimeLibcall;
def FPEXT_F32_PPCF128 : RuntimeLibcall;
def FPEXT_F64_PPCF128 : RuntimeLibcall;
def FPEXT_F80_F128 : RuntimeLibcall;
def FPEXT_F64_F128 : RuntimeLibcall;
def FPEXT_F32_F128 : RuntimeLibcall;
def FPEXT_F16_F128 : RuntimeLibcall;
def FPEXT_F16_F80 : RuntimeLibcall;
def FPEXT_F32_F64 : RuntimeLibcall;
def FPEXT_F16_F64 : RuntimeLibcall;
def FPEXT_F16_F32 : RuntimeLibcall;
def FPROUND_F32_F16 : RuntimeLibcall;
def FPROUND_F64_F16 : RuntimeLibcall;
def FPROUND_F80_F16 : RuntimeLibcall;
def FPROUND_F128_F16 : RuntimeLibcall;
def FPROUND_PPCF128_F16 : RuntimeLibcall;
def FPROUND_F32_BF16 : RuntimeLibcall;
def FPROUND_F64_BF16 : RuntimeLibcall;
def FPROUND_F80_BF16 : RuntimeLibcall;
def FPROUND_F128_BF16 : RuntimeLibcall;
def FPROUND_F64_F32 : RuntimeLibcall;
def FPROUND_F80_F32 : RuntimeLibcall;
def FPROUND_F128_F32 : RuntimeLibcall;
def FPROUND_PPCF128_F32 : RuntimeLibcall;
def FPROUND_F80_F64 : RuntimeLibcall;
def FPROUND_F128_F64 : RuntimeLibcall;
def FPROUND_PPCF128_F64 : RuntimeLibcall;
def FPROUND_F128_F80 : RuntimeLibcall;
def FPTOSINT_F16_I32 : RuntimeLibcall;
def FPTOSINT_F16_I64 : RuntimeLibcall;
def FPTOSINT_F16_I128 : RuntimeLibcall;
def FPTOSINT_F32_I32 : RuntimeLibcall;
def FPTOSINT_F32_I64 : RuntimeLibcall;
def FPTOSINT_F32_I128 : RuntimeLibcall;
def FPTOSINT_F64_I32 : RuntimeLibcall;
def FPTOSINT_F64_I64 : RuntimeLibcall;
def FPTOSINT_F64_I128 : RuntimeLibcall;
def FPTOSINT_F80_I32 : RuntimeLibcall;
def FPTOSINT_F80_I64 : RuntimeLibcall;
def FPTOSINT_F80_I128 : RuntimeLibcall;
def FPTOSINT_F128_I32 : RuntimeLibcall;
def FPTOSINT_F128_I64 : RuntimeLibcall;
def FPTOSINT_F128_I128 : RuntimeLibcall;
def FPTOSINT_PPCF128_I32 : RuntimeLibcall;
def FPTOSINT_PPCF128_I64 : RuntimeLibcall;
def FPTOSINT_PPCF128_I128 : RuntimeLibcall;
def FPTOUINT_F16_I32 : RuntimeLibcall;
def FPTOUINT_F16_I64 : RuntimeLibcall;
def FPTOUINT_F16_I128 : RuntimeLibcall;
def FPTOUINT_F32_I32 : RuntimeLibcall;
def FPTOUINT_F32_I64 : RuntimeLibcall;
def FPTOUINT_F32_I128 : RuntimeLibcall;
def FPTOUINT_F64_I32 : RuntimeLibcall;
def FPTOUINT_F64_I64 : RuntimeLibcall;
def FPTOUINT_F64_I128 : RuntimeLibcall;
def FPTOUINT_F80_I32 : RuntimeLibcall;
def FPTOUINT_F80_I64 : RuntimeLibcall;
def FPTOUINT_F80_I128 : RuntimeLibcall;
def FPTOUINT_F128_I32 : RuntimeLibcall;
def FPTOUINT_F128_I64 : RuntimeLibcall;
def FPTOUINT_F128_I128 : RuntimeLibcall;
def FPTOUINT_PPCF128_I32 : RuntimeLibcall;
def FPTOUINT_PPCF128_I64 : RuntimeLibcall;
def FPTOUINT_PPCF128_I128 : RuntimeLibcall;
def SINTTOFP_I32_F16 : RuntimeLibcall;
def SINTTOFP_I32_F32 : RuntimeLibcall;
def SINTTOFP_I32_F64 : RuntimeLibcall;
def SINTTOFP_I32_F80 : RuntimeLibcall;
def SINTTOFP_I32_F128 : RuntimeLibcall;
def SINTTOFP_I32_PPCF128 : RuntimeLibcall;
def SINTTOFP_I64_BF16 : RuntimeLibcall;
def SINTTOFP_I64_F16 : RuntimeLibcall;
def SINTTOFP_I64_F32 : RuntimeLibcall;
def SINTTOFP_I64_F64 : RuntimeLibcall;
def SINTTOFP_I64_F80 : RuntimeLibcall;
def SINTTOFP_I64_F128 : RuntimeLibcall;
def SINTTOFP_I64_PPCF128 : RuntimeLibcall;
def SINTTOFP_I128_F16 : RuntimeLibcall;
def SINTTOFP_I128_F32 : RuntimeLibcall;
def SINTTOFP_I128_F64 : RuntimeLibcall;
def SINTTOFP_I128_F80 : RuntimeLibcall;
def SINTTOFP_I128_F128 : RuntimeLibcall;
def SINTTOFP_I128_PPCF128 : RuntimeLibcall;
def UINTTOFP_I32_F16 : RuntimeLibcall;
def UINTTOFP_I32_F32 : RuntimeLibcall;
def UINTTOFP_I32_F64 : RuntimeLibcall;
def UINTTOFP_I32_F80 : RuntimeLibcall;
def UINTTOFP_I32_F128 : RuntimeLibcall;
def UINTTOFP_I32_PPCF128 : RuntimeLibcall;
def UINTTOFP_I64_BF16 : RuntimeLibcall;
def UINTTOFP_I64_F16 : RuntimeLibcall;
def UINTTOFP_I64_F32 : RuntimeLibcall;
def UINTTOFP_I64_F64 : RuntimeLibcall;
def UINTTOFP_I64_F80 : RuntimeLibcall;
def UINTTOFP_I64_F128 : RuntimeLibcall;
def UINTTOFP_I64_PPCF128 : RuntimeLibcall;
def UINTTOFP_I128_F16 : RuntimeLibcall;
def UINTTOFP_I128_F32 : RuntimeLibcall;
def UINTTOFP_I128_F64 : RuntimeLibcall;
def UINTTOFP_I128_F80 : RuntimeLibcall;
def UINTTOFP_I128_F128 : RuntimeLibcall;
def UINTTOFP_I128_PPCF128 : RuntimeLibcall;
def CONVERT_F128_PPCF128 : RuntimeLibcall;
def CONVERT_PPCF128_F128 : RuntimeLibcall;

// Comparisons
foreach FPTy = ["F32", "F64", "F128", "PPCF128"] in {
  def OEQ_#FPTy : RuntimeLibcall;
  def UNE_#FPTy : RuntimeLibcall;
  def OGE_#FPTy : RuntimeLibcall;
  def OLT_#FPTy : RuntimeLibcall;
  def OLE_#FPTy : RuntimeLibcall;
  def OGT_#FPTy : RuntimeLibcall;
  def UO_#FPTy : RuntimeLibcall;
}

// Memory
def MEMCPY : RuntimeLibcall;
def MEMMOVE : RuntimeLibcall;
def MEMSET : RuntimeLibcall;
def CALLOC : RuntimeLibcall;
def BZERO : RuntimeLibcall;

// Element-wise unordered-atomic memory of different sizes
foreach MemSize = [1, 2, 4, 8, 16] in {
  def MEMCPY_ELEMENT_UNORDERED_ATOMIC_#MemSize : RuntimeLibcall;
  def MEMMOVE_ELEMENT_UNORDERED_ATOMIC_#MemSize : RuntimeLibcall;
  def MEMSET_ELEMENT_UNORDERED_ATOMIC_#MemSize : RuntimeLibcall;
}

// Exception handling
def UNWIND_RESUME : RuntimeLibcall;
def CXA_END_CLEANUP : RuntimeLibcall;

// Note: there are two sets of atomics libcalls; see
// <https://llvm.org/docs/Atomics.html> for more info on the
// difference between them.

// Atomic '__sync_*' libcalls.
defset list<RuntimeLibcall> LibCalls__sync = {
  foreach MemSize = [1, 2, 4, 8, 16] in {
    def SYNC_VAL_COMPARE_AND_SWAP_#MemSize : RuntimeLibcall;
    def SYNC_LOCK_TEST_AND_SET_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_ADD_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_SUB_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_AND_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_OR_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_XOR_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_NAND_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_MAX_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_UMAX_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_MIN_#MemSize : RuntimeLibcall;
    def SYNC_FETCH_AND_UMIN_#MemSize : RuntimeLibcall;
  }
}

defset list<RuntimeLibcall> LibCalls__atomic = {
  // Atomic `__atomic_*' libcalls.
  foreach MemSize = ["", "_1", "_2", "_4", "_8", "_16"] in {
    def ATOMIC_LOAD#MemSize : RuntimeLibcall;
    def ATOMIC_STORE#MemSize : RuntimeLibcall;
    def ATOMIC_EXCHANGE#MemSize : RuntimeLibcall;
    def ATOMIC_COMPARE_EXCHANGE#MemSize : RuntimeLibcall;
  }

  foreach MemSize = [1, 2, 4, 8, 16] in {
    def ATOMIC_FETCH_ADD_#MemSize : RuntimeLibcall;
    def ATOMIC_FETCH_SUB_#MemSize : RuntimeLibcall;
    def ATOMIC_FETCH_AND_#MemSize : RuntimeLibcall;
    def ATOMIC_FETCH_OR_#MemSize : RuntimeLibcall;
    def ATOMIC_FETCH_XOR_#MemSize : RuntimeLibcall;
    def ATOMIC_FETCH_NAND_#MemSize : RuntimeLibcall;
  }
}

multiclass AtomicOrderSizeLibcall {
  def _RELAX : RuntimeLibcall;
  def _ACQ : RuntimeLibcall;
  def _REL : RuntimeLibcall;
  def _ACQ_REL : RuntimeLibcall;
}

// Out-of-line atomics libcalls
defset list<RuntimeLibcall> LibCalls__OutOfLineAtomic = {
  foreach MemSize = [1, 2, 4, 8, 16] in {
    defm OUTLINE_ATOMIC_CAS#MemSize : AtomicOrderSizeLibcall;
    defm OUTLINE_ATOMIC_SWP#MemSize : AtomicOrderSizeLibcall;
    defm OUTLINE_ATOMIC_LDADD#MemSize : AtomicOrderSizeLibcall;
    defm OUTLINE_ATOMIC_LDSET#MemSize : AtomicOrderSizeLibcall;
    defm OUTLINE_ATOMIC_LDCLR#MemSize : AtomicOrderSizeLibcall;
    defm OUTLINE_ATOMIC_LDEOR#MemSize : AtomicOrderSizeLibcall;
  }
}

// Stack Protector Fail
def STACKPROTECTOR_CHECK_FAIL : RuntimeLibcall;

// Deoptimization
def DEOPTIMIZE : RuntimeLibcall;

// Return address
def RETURN_ADDRESS : RuntimeLibcall;

// Clear cache
def CLEAR_CACHE : RuntimeLibcall;
def RISCV_FLUSH_ICACHE : RuntimeLibcall;

multiclass LibmLongDoubleLibCall<string libcall_basename = !toupper(NAME),
                                 string rtbasename = NAME> {
  def NAME#"_f128"
      : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_F128"),
                           !strconcat(rtbasename, "l")>;
  def NAME#"_ppcf128"
      : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_PPCF128"),
                           !strconcat(rtbasename, "l")>;
  def NAME#"_f80"
      : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_F80"),
                           !strconcat(rtbasename, "l")>;
}

//--------------------------------------------------------------------
// Define implementation default libcalls
//--------------------------------------------------------------------

defset list<RuntimeLibcallImpl> DefaultRuntimeLibcallImpls = {
let IsDefault = true in {

//--------------------------------------------------------------------
// compiler-rt / libgcc
//--------------------------------------------------------------------

def __ashlhi3 : RuntimeLibcallImpl<SHL_I16>;
def __ashlsi3 : RuntimeLibcallImpl<SHL_I32>;
def __ashldi3 : RuntimeLibcallImpl<SHL_I64>;
def __ashlti3 : RuntimeLibcallImpl<SHL_I128>;

def __lshrhi3 : RuntimeLibcallImpl<SRL_I16>;
def __lshrsi3 : RuntimeLibcallImpl<SRL_I32>;
def __lshrdi3 : RuntimeLibcallImpl<SRL_I64>;
def __lshrti3 : RuntimeLibcallImpl<SRL_I128>;

def __ashrhi3 : RuntimeLibcallImpl<SRA_I16>;
def __ashrsi3 : RuntimeLibcallImpl<SRA_I32>;
def __ashrdi3 : RuntimeLibcallImpl<SRA_I64>;
def __ashrti3 : RuntimeLibcallImpl<SRA_I128>;

def __mulqi3 : RuntimeLibcallImpl<MUL_I8>;
def __mulhi3 : RuntimeLibcallImpl<MUL_I16>;
def __mulsi3 : RuntimeLibcallImpl<MUL_I32>;
def __muldi3 : RuntimeLibcallImpl<MUL_I64>;
def __multi3 : RuntimeLibcallImpl<MUL_I128>;

def __mulosi4 : RuntimeLibcallImpl<MULO_I32>;
def __mulodi4 : RuntimeLibcallImpl<MULO_I64>;
def __muloti4 : RuntimeLibcallImpl<MULO_I128>;

def __divqi3 : RuntimeLibcallImpl<SDIV_I8>;
def __divhi3 : RuntimeLibcallImpl<SDIV_I16>;
def __divsi3 : RuntimeLibcallImpl<SDIV_I32>;
def __divdi3 : RuntimeLibcallImpl<SDIV_I64>;
def __divti3 : RuntimeLibcallImpl<SDIV_I128>;

def __udivqi3 : RuntimeLibcallImpl<UDIV_I8>;
def __udivhi3 : RuntimeLibcallImpl<UDIV_I16>;
def __udivsi3 : RuntimeLibcallImpl<UDIV_I32>;
def __udivdi3 : RuntimeLibcallImpl<UDIV_I64>;
def __udivti3 : RuntimeLibcallImpl<UDIV_I128>;

def __modqi3 : RuntimeLibcallImpl<SREM_I8>;
def __modhi3 : RuntimeLibcallImpl<SREM_I16>;
def __modsi3 : RuntimeLibcallImpl<SREM_I32>;
def __moddi3 : RuntimeLibcallImpl<SREM_I64>;
def __modti3 : RuntimeLibcallImpl<SREM_I128>;

def __umodqi3 : RuntimeLibcallImpl<UREM_I8>;
def __umodhi3 : RuntimeLibcallImpl<UREM_I16>;
def __umodsi3 : RuntimeLibcallImpl<UREM_I32>;
def __umoddi3 : RuntimeLibcallImpl<UREM_I64>;
def __umodti3 : RuntimeLibcallImpl<UREM_I128>;

def __negsi2 : RuntimeLibcallImpl<NEG_I32>;
def __negdi2 : RuntimeLibcallImpl<NEG_I64>;

def __clzsi2 : RuntimeLibcallImpl<CTLZ_I32>;
def __clzdi2 : RuntimeLibcallImpl<CTLZ_I64>;
def __clzti2 : RuntimeLibcallImpl<CTLZ_I128>;

def __popcountsi2 : RuntimeLibcallImpl<CTPOP_I32>;
def __popcountdi2 : RuntimeLibcallImpl<CTPOP_I64>;
def __popcountti2 : RuntimeLibcallImpl<CTPOP_I128>;

def __addsf3 : RuntimeLibcallImpl<ADD_F32>;
def __adddf3 : RuntimeLibcallImpl<ADD_F64>;
def __addxf3 : RuntimeLibcallImpl<ADD_F80>;
def __addtf3 : RuntimeLibcallImpl<ADD_F128>;
def __gcc_qadd : RuntimeLibcallImpl<ADD_PPCF128>;

def __subsf3 : RuntimeLibcallImpl<SUB_F32>;
def __subdf3 : RuntimeLibcallImpl<SUB_F64>;
def __subxf3 : RuntimeLibcallImpl<SUB_F80>;
def __subtf3 : RuntimeLibcallImpl<SUB_F128>;
def __gcc_qsub : RuntimeLibcallImpl<SUB_PPCF128>;

def __mulsf3 : RuntimeLibcallImpl<MUL_F32>;
def __muldf3 : RuntimeLibcallImpl<MUL_F64>;
def __mulxf3 : RuntimeLibcallImpl<MUL_F80>;
def __multf3 : RuntimeLibcallImpl<MUL_F128>;
def __gcc_qmul : RuntimeLibcallImpl<MUL_PPCF128>;

def __divsf3 : RuntimeLibcallImpl<DIV_F32>;
def __divdf3 : RuntimeLibcallImpl<DIV_F64>;
def __divxf3 : RuntimeLibcallImpl<DIV_F80>;
def __divtf3 : RuntimeLibcallImpl<DIV_F128>;
def __gcc_qdiv : RuntimeLibcallImpl<DIV_PPCF128>;

def __powisf2 : RuntimeLibcallImpl<POWI_F32>;
def __powidf2 : RuntimeLibcallImpl<POWI_F64>;
def __powixf2 : RuntimeLibcallImpl<POWI_F80>;
def __powitf2_f128 : RuntimeLibcallImpl<POWI_F128, "__powitf2">;
def __powitf2_ppc128 : RuntimeLibcallImpl<POWI_PPCF128, "__powitf2">;

// Conversion
def __extendbfsf2 : RuntimeLibcallImpl<FPEXT_BF16_F32>;
def __gcc_stoq : RuntimeLibcallImpl<FPEXT_F32_PPCF128>;
def __gcc_dtoq : RuntimeLibcallImpl<FPEXT_F64_PPCF128>;
def __extendxftf2 : RuntimeLibcallImpl<FPEXT_F80_F128>;
def __extenddftf2 : RuntimeLibcallImpl<FPEXT_F64_F128>;
def __extendsftf2 : RuntimeLibcallImpl<FPEXT_F32_F128>;
def __extendhftf2 : RuntimeLibcallImpl<FPEXT_F16_F128>;
def __extendhfxf2 : RuntimeLibcallImpl<FPEXT_F16_F80>;
def __extendsfdf2 : RuntimeLibcallImpl<FPEXT_F32_F64>;
def __extendhfdf2 : RuntimeLibcallImpl<FPEXT_F16_F64>;
def __extendhfsf2 : RuntimeLibcallImpl<FPEXT_F16_F32>;
def __truncsfhf2 : RuntimeLibcallImpl<FPROUND_F32_F16>;
def __truncdfhf2 : RuntimeLibcallImpl<FPROUND_F64_F16>;
def __truncxfhf2 : RuntimeLibcallImpl<FPROUND_F80_F16>;
def __trunctfhf2_f128 : RuntimeLibcallImpl<FPROUND_F128_F16, "__trunctfhf2">;
def __trunctfhf2_ppcf128 : RuntimeLibcallImpl<FPROUND_PPCF128_F16, "__trunctfhf2">;
def __truncsfbf2 : RuntimeLibcallImpl<FPROUND_F32_BF16>;
def __truncdfbf2 : RuntimeLibcallImpl<FPROUND_F64_BF16>;
def __truncxfbf2 : RuntimeLibcallImpl<FPROUND_F80_BF16>;
def __trunctfbf2 : RuntimeLibcallImpl<FPROUND_F128_BF16>;
def __truncdfsf2 : RuntimeLibcallImpl<FPROUND_F64_F32>;
def __truncxfsf2 : RuntimeLibcallImpl<FPROUND_F80_F32>;
def __trunctfsf2 : RuntimeLibcallImpl<FPROUND_F128_F32>;
def __gcc_qtos : RuntimeLibcallImpl<FPROUND_PPCF128_F32>;
def __truncxfdf2 : RuntimeLibcallImpl<FPROUND_F80_F64>;
def __trunctfdf2 : RuntimeLibcallImpl<FPROUND_F128_F64>;
def __gcc_qtod : RuntimeLibcallImpl<FPROUND_PPCF128_F64>;
def __trunctfxf2 : RuntimeLibcallImpl<FPROUND_F128_F80>;
def __fixhfsi : RuntimeLibcallImpl<FPTOSINT_F16_I32>;
def __fixhfdi : RuntimeLibcallImpl<FPTOSINT_F16_I64>;
def __fixhfti : RuntimeLibcallImpl<FPTOSINT_F16_I128>;
def __fixsfsi : RuntimeLibcallImpl<FPTOSINT_F32_I32>;
def __fixsfdi : RuntimeLibcallImpl<FPTOSINT_F32_I64>;
def __fixsfti : RuntimeLibcallImpl<FPTOSINT_F32_I128>;
def __fixdfsi : RuntimeLibcallImpl<FPTOSINT_F64_I32>;
def __fixdfdi : RuntimeLibcallImpl<FPTOSINT_F64_I64>;
def __fixdfti : RuntimeLibcallImpl<FPTOSINT_F64_I128>;
def __fixxfsi : RuntimeLibcallImpl<FPTOSINT_F80_I32>;
def __fixxfdi : RuntimeLibcallImpl<FPTOSINT_F80_I64>;
def __fixxfti : RuntimeLibcallImpl<FPTOSINT_F80_I128>;
def __fixtfsi : RuntimeLibcallImpl<FPTOSINT_F128_I32>;
def __fixtfdi_f128 : RuntimeLibcallImpl<FPTOSINT_F128_I64, "__fixtfdi">;
def __fixtfti_f128 : RuntimeLibcallImpl<FPTOSINT_F128_I128, "__fixtfti">;
def __gcc_qtou : RuntimeLibcallImpl<FPTOSINT_PPCF128_I32>;
def __fixtfdi_ppcf128 : RuntimeLibcallImpl<FPTOSINT_PPCF128_I64, "__fixtfdi">;
def __fixtfti_ppcf128 : RuntimeLibcallImpl<FPTOSINT_PPCF128_I128, "__fixtfti">;
def __fixunshfsi : RuntimeLibcallImpl<FPTOUINT_F16_I32>;
def __fixunshfdi : RuntimeLibcallImpl<FPTOUINT_F16_I64>;
def __fixunshfti : RuntimeLibcallImpl<FPTOUINT_F16_I128>;
def __fixunssfsi : RuntimeLibcallImpl<FPTOUINT_F32_I32>;
def __fixunssfdi : RuntimeLibcallImpl<FPTOUINT_F32_I64>;
def __fixunssfti : RuntimeLibcallImpl<FPTOUINT_F32_I128>;
def __fixunsdfsi : RuntimeLibcallImpl<FPTOUINT_F64_I32>;
def __fixunsdfdi : RuntimeLibcallImpl<FPTOUINT_F64_I64>;
def __fixunsdfti : RuntimeLibcallImpl<FPTOUINT_F64_I128>;
def __fixunsxfsi : RuntimeLibcallImpl<FPTOUINT_F80_I32>;
def __fixunsxfdi : RuntimeLibcallImpl<FPTOUINT_F80_I64>;
def __fixunsxfti : RuntimeLibcallImpl<FPTOUINT_F80_I128>;
def __fixunstfsi_f128 : RuntimeLibcallImpl<FPTOUINT_F128_I32, "__fixunstfsi">;
def __fixunstfdi_f128 : RuntimeLibcallImpl<FPTOUINT_F128_I64, "__fixunstfdi">;
def __fixunstfti_f128 : RuntimeLibcallImpl<FPTOUINT_F128_I128, "__fixunstfti">;
def __fixunstfsi_ppcf128 : RuntimeLibcallImpl<FPTOUINT_PPCF128_I32, "__fixunstfsi">;
def __fixunstfdi_ppcf128 : RuntimeLibcallImpl<FPTOUINT_PPCF128_I64, "__fixunstfdi">;
def __fixunstfti_ppcf128 : RuntimeLibcallImpl<FPTOUINT_PPCF128_I128, "__fixunstfti">;
def __floatsihf : RuntimeLibcallImpl<SINTTOFP_I32_F16>;
def __floatsisf : RuntimeLibcallImpl<SINTTOFP_I32_F32>;
def __floatsidf : RuntimeLibcallImpl<SINTTOFP_I32_F64>;
def __floatsixf : RuntimeLibcallImpl<SINTTOFP_I32_F80>;
def __floatsitf : RuntimeLibcallImpl<SINTTOFP_I32_F128>;
def __gcc_itoq : RuntimeLibcallImpl<SINTTOFP_I32_PPCF128>;
def __floatdibf : RuntimeLibcallImpl<SINTTOFP_I64_BF16>;
def __floatdihf : RuntimeLibcallImpl<SINTTOFP_I64_F16>;
def __floatdisf : RuntimeLibcallImpl<SINTTOFP_I64_F32>;
def __floatdidf : RuntimeLibcallImpl<SINTTOFP_I64_F64>;
def __floatdixf : RuntimeLibcallImpl<SINTTOFP_I64_F80>;
def __floatditf_f128 : RuntimeLibcallImpl<SINTTOFP_I64_F128, "__floatditf">;
def __floatditf_ppcf128 : RuntimeLibcallImpl<SINTTOFP_I64_PPCF128, "__floatditf">;
def __floattihf : RuntimeLibcallImpl<SINTTOFP_I128_F16>;
def __floattisf : RuntimeLibcallImpl<SINTTOFP_I128_F32>;
def __floattidf : RuntimeLibcallImpl<SINTTOFP_I128_F64>;
def __floattixf : RuntimeLibcallImpl<SINTTOFP_I128_F80>;
def __floattitf_f128 : RuntimeLibcallImpl<SINTTOFP_I128_F128, "__floattitf">;
def __floattitf_ppcf128 : RuntimeLibcallImpl<SINTTOFP_I128_PPCF128, "__floattitf">;
def __floatunsihf : RuntimeLibcallImpl<UINTTOFP_I32_F16>;
def __floatunsisf : RuntimeLibcallImpl<UINTTOFP_I32_F32>;
def __floatunsidf : RuntimeLibcallImpl<UINTTOFP_I32_F64>;
def __floatunsixf : RuntimeLibcallImpl<UINTTOFP_I32_F80>;
def __floatunsitf : RuntimeLibcallImpl<UINTTOFP_I32_F128>;
def __gcc_utoq : RuntimeLibcallImpl<UINTTOFP_I32_PPCF128>;
def __floatundibf : RuntimeLibcallImpl<UINTTOFP_I64_BF16>;
def __floatundihf : RuntimeLibcallImpl<UINTTOFP_I64_F16>;
def __floatundisf : RuntimeLibcallImpl<UINTTOFP_I64_F32>;
def __floatundidf : RuntimeLibcallImpl<UINTTOFP_I64_F64>;
def __floatundixf : RuntimeLibcallImpl<UINTTOFP_I64_F80>;
def __floatunditf_f128 : RuntimeLibcallImpl<UINTTOFP_I64_F128, "__floatunditf">;
def __floatunditf_ppcf128 : RuntimeLibcallImpl<UINTTOFP_I64_PPCF128, "__floatunditf">;
def __floatuntihf : RuntimeLibcallImpl<UINTTOFP_I128_F16>;
def __floatuntisf : RuntimeLibcallImpl<UINTTOFP_I128_F32>;
def __floatuntidf : RuntimeLibcallImpl<UINTTOFP_I128_F64>;
def __floatuntixf : RuntimeLibcallImpl<UINTTOFP_I128_F80>;
def __floatuntitf_f128 : RuntimeLibcallImpl<UINTTOFP_I128_F128, "__floatuntitf">;
def __floatuntitf_ppcf128 : RuntimeLibcallImpl<UINTTOFP_I128_PPCF128, "__floatuntitf">;
def __extendkftf2 : RuntimeLibcallImpl<CONVERT_F128_PPCF128>;
def __trunctfkf2 : RuntimeLibcallImpl<CONVERT_PPCF128_F128>;

// Comparison
def __eqsf2 : RuntimeLibcallImpl<OEQ_F32>;
def __eqdf2 : RuntimeLibcallImpl<OEQ_F64>;
def __eqtf2 : RuntimeLibcallImpl<OEQ_F128>;
def __gcc_qeq : RuntimeLibcallImpl<OEQ_PPCF128>;
def __nesf2 : RuntimeLibcallImpl<UNE_F32>;
def __nedf2 : RuntimeLibcallImpl<UNE_F64>;
def __netf2 : RuntimeLibcallImpl<UNE_F128>;
def __gcc_qne : RuntimeLibcallImpl<UNE_PPCF128>;
def __gesf2 : RuntimeLibcallImpl<OGE_F32>;
def __gedf2 : RuntimeLibcallImpl<OGE_F64>;
def __getf2 : RuntimeLibcallImpl<OGE_F128>;
def __gcc_qge : RuntimeLibcallImpl<OGE_PPCF128>;
def __ltsf2 : RuntimeLibcallImpl<OLT_F32>;
def __ltdf2 : RuntimeLibcallImpl<OLT_F64>;
def __lttf2 : RuntimeLibcallImpl<OLT_F128>;
def __gcc_qlt : RuntimeLibcallImpl<OLT_PPCF128>;
def __lesf2 : RuntimeLibcallImpl<OLE_F32>;
def __ledf2 : RuntimeLibcallImpl<OLE_F64>;
def __letf2 : RuntimeLibcallImpl<OLE_F128>;
def __gcc_qle : RuntimeLibcallImpl<OLE_PPCF128>;
def __gtsf2 : RuntimeLibcallImpl<OGT_F32>;
def __gtdf2 : RuntimeLibcallImpl<OGT_F64>;
def __gttf2 : RuntimeLibcallImpl<OGT_F128>;
def __gcc_qgt : RuntimeLibcallImpl<OGT_PPCF128>;
def __unordsf2 : RuntimeLibcallImpl<UO_F32>;
def __unorddf2 : RuntimeLibcallImpl<UO_F64>;
def __unordtf2 : RuntimeLibcallImpl<UO_F128>;
def __gcc_qunord : RuntimeLibcallImpl<UO_PPCF128>;

// Element-wise unordered-atomic memory of different sizes
foreach MemSize = [ 1, 2, 4, 8, 16 ] in {
  def __llvm_memcpy_element_unordered_atomic_#MemSize : RuntimeLibcallImpl<
    !cast<RuntimeLibcall>("MEMCPY_ELEMENT_UNORDERED_ATOMIC_"#MemSize)>;

  def __llvm_memmove_element_unordered_atomic_#MemSize : RuntimeLibcallImpl<
    !cast<RuntimeLibcall>("MEMMOVE_ELEMENT_UNORDERED_ATOMIC_"#MemSize)>;

  def __llvm_memset_element_unordered_atomic_#MemSize : RuntimeLibcallImpl<
    !cast<RuntimeLibcall>("MEMSET_ELEMENT_UNORDERED_ATOMIC_"#MemSize)>;
}

// Exception handling
def _Unwind_Resume : RuntimeLibcallImpl<UNWIND_RESUME>;
def __cxa_end_cleanup : RuntimeLibcallImpl<CXA_END_CLEANUP>;

// Atomic '__sync_*' libcalls.
foreach lc = LibCalls__sync in {
  def __#!tolower(!cast<string>(lc)) : RuntimeLibcallImpl<lc>;
}

// Atomic `__atomic_*' libcalls.
foreach lc = LibCalls__atomic in {
  def __#!tolower(!cast<string>(lc)) : RuntimeLibcallImpl<lc>;
}

// Stack Protector Fail
def __stack_chk_fail : RuntimeLibcallImpl<STACKPROTECTOR_CHECK_FAIL>;

// Deoptimization
def __llvm_deoptimize : RuntimeLibcallImpl<DEOPTIMIZE>;

// Clear cache
def __clear_cache : RuntimeLibcallImpl<CLEAR_CACHE>;
def __riscv_flush_icache : RuntimeLibcallImpl<RISCV_FLUSH_ICACHE>;

//--------------------------------------------------------------------
// libm
//--------------------------------------------------------------------

def fmodf : RuntimeLibcallImpl<REM_F32>;
def fmod : RuntimeLibcallImpl<REM_F64>;
def fmodl_f128 : RuntimeLibcallImpl<REM_F128, "fmodl">;
def fmodl_f80 : RuntimeLibcallImpl<REM_F80, "fmodl">;
def fmodl_ppc128 : RuntimeLibcallImpl<REM_PPCF128, "fmodl">;

def fmaf : RuntimeLibcallImpl<FMA_F32>;
def fma : RuntimeLibcallImpl<FMA_F64>;
defm fma : LibmLongDoubleLibCall;

def sqrtf : RuntimeLibcallImpl<SQRT_F32>;
def sqrt : RuntimeLibcallImpl<SQRT_F64>;
defm sqrt : LibmLongDoubleLibCall;

def cbrtf : RuntimeLibcallImpl<CBRT_F32>;
def cbrt : RuntimeLibcallImpl<CBRT_F64>;
defm cbrt : LibmLongDoubleLibCall;

def logf : RuntimeLibcallImpl<LOG_F32>;
def log : RuntimeLibcallImpl<LOG_F64>;
defm log : LibmLongDoubleLibCall;

def __logf_finite : RuntimeLibcallImpl<LOG_FINITE_F32>;
def __log_finite : RuntimeLibcallImpl<LOG_FINITE_F64>;
def __logl_finite_f80 : RuntimeLibcallImpl<LOG_FINITE_F80, "__logl_finite">;
def __logl_finite_f128 : RuntimeLibcallImpl<LOG_FINITE_F128, "__logl_finite">;
def __logl_finite_ppcf128 : RuntimeLibcallImpl<LOG_FINITE_PPCF128, "__logl_finite">;

def log2f : RuntimeLibcallImpl<LOG2_F32>;
def log2 : RuntimeLibcallImpl<LOG2_F64>;
defm log2 : LibmLongDoubleLibCall;

def __log2f_finite : RuntimeLibcallImpl<LOG2_FINITE_F32>;
def __log2_finite : RuntimeLibcallImpl<LOG2_FINITE_F64>;
def __log2l_finite_f80 : RuntimeLibcallImpl<LOG2_FINITE_F80, "__log2l_finite">;
def __log2l_finite_f128 : RuntimeLibcallImpl<LOG2_FINITE_F128, "__log2l_finite">;
def __log2l_finite_ppcf128 : RuntimeLibcallImpl<LOG2_FINITE_PPCF128, "__log2l_finite">;

def log10f : RuntimeLibcallImpl<LOG10_F32>;
def log10 : RuntimeLibcallImpl<LOG10_F64>;
defm log10 : LibmLongDoubleLibCall;

def __log10f_finite : RuntimeLibcallImpl<LOG10_FINITE_F32>;
def __log10_finite : RuntimeLibcallImpl<LOG10_FINITE_F64>;
def __log10l_finite_f80 : RuntimeLibcallImpl<LOG10_FINITE_F80, "__log10l_finite">;
def __log10l_finite_f128 : RuntimeLibcallImpl<LOG10_FINITE_F128, "__log10l_finite">;
def __log10l_finite_ppcf128 : RuntimeLibcallImpl<LOG10_FINITE_PPCF128, "__log10l_finite">;

def expf : RuntimeLibcallImpl<EXP_F32>;
def exp : RuntimeLibcallImpl<EXP_F64>;
defm exp : LibmLongDoubleLibCall<"EXP", "exp">;

def __expf_finite : RuntimeLibcallImpl<EXP_FINITE_F32>;
def __exp_finite : RuntimeLibcallImpl<EXP_FINITE_F64>;
def __expl_finite_f80 : RuntimeLibcallImpl<EXP_FINITE_F80, "__expl_finite">;
def __expl_finite_f128 : RuntimeLibcallImpl<EXP_FINITE_F128, "__expl_finite">;
def __expl_finite_ppcf128 : RuntimeLibcallImpl<EXP_FINITE_PPCF128, "__expl_finite">;

def exp2f : RuntimeLibcallImpl<EXP2_F32>;
def exp2 : RuntimeLibcallImpl<EXP2_F64>;
defm exp2 : LibmLongDoubleLibCall<"EXP2", "exp2">;

def __exp2f_finite : RuntimeLibcallImpl<EXP2_FINITE_F32>;
def __exp2_finite : RuntimeLibcallImpl<EXP2_FINITE_F64>;
def __exp2l_finite_f80 : RuntimeLibcallImpl<EXP2_FINITE_F80, "__exp2l_finite">;
def __exp2l_finite_f128 : RuntimeLibcallImpl<EXP2_FINITE_F128, "__exp2l_finite">;
def __exp2l_finite_ppcf128 : RuntimeLibcallImpl<EXP2_FINITE_PPCF128, "__exp2l_finite">;

def exp10f : RuntimeLibcallImpl<EXP10_F32>;
def exp10 : RuntimeLibcallImpl<EXP10_F64>;
def exp10l_f80 : RuntimeLibcallImpl<EXP10_F80, "exp10l">;
def exp10l_f128 : RuntimeLibcallImpl<EXP10_F128, "exp10l">;
def exp10l_ppcf128 : RuntimeLibcallImpl<EXP10_PPCF128, "exp10l">;

def sinf : RuntimeLibcallImpl<SIN_F32>;
def sin : RuntimeLibcallImpl<SIN_F64>;
defm sin : LibmLongDoubleLibCall;

def cosf : RuntimeLibcallImpl<COS_F32>;
def cos : RuntimeLibcallImpl<COS_F64>;
defm cos : LibmLongDoubleLibCall;

def tanf : RuntimeLibcallImpl<TAN_F32>;
def tan : RuntimeLibcallImpl<TAN_F64>;
defm tan : LibmLongDoubleLibCall;

def sinhf : RuntimeLibcallImpl<SINH_F32>;
def sinh : RuntimeLibcallImpl<SINH_F64>;
defm sinh : LibmLongDoubleLibCall;

def coshf : RuntimeLibcallImpl<COSH_F32>;
def cosh : RuntimeLibcallImpl<COSH_F64>;
defm cosh : LibmLongDoubleLibCall;

def tanhf : RuntimeLibcallImpl<TANH_F32>;
def tanh : RuntimeLibcallImpl<TANH_F64>;
defm tanh : LibmLongDoubleLibCall;

def asinf : RuntimeLibcallImpl<ASIN_F32>;
def asin : RuntimeLibcallImpl<ASIN_F64>;
defm asin : LibmLongDoubleLibCall;

def acosf : RuntimeLibcallImpl<ACOS_F32>;
def acos : RuntimeLibcallImpl<ACOS_F64>;
defm acos : LibmLongDoubleLibCall;

def atanf : RuntimeLibcallImpl<ATAN_F32>;
def atan : RuntimeLibcallImpl<ATAN_F64>;
defm atan : LibmLongDoubleLibCall;

def atan2f : RuntimeLibcallImpl<ATAN2_F32>;
def atan2 : RuntimeLibcallImpl<ATAN2_F64>;
defm atan2 : LibmLongDoubleLibCall;

def powf : RuntimeLibcallImpl<POW_F32>;
def pow : RuntimeLibcallImpl<POW_F64>;
defm pow : LibmLongDoubleLibCall;

def __powf_finite : RuntimeLibcallImpl<POW_FINITE_F32>;
def __pow_finite : RuntimeLibcallImpl<POW_FINITE_F64>;
def __powl_finite_f80 : RuntimeLibcallImpl<POW_FINITE_F80, "__powl_finite">;
def __powl_finite_f128 : RuntimeLibcallImpl<POW_FINITE_F128, "__powl_finite">;
def __powl_finite_ppcf128 : RuntimeLibcallImpl<POW_FINITE_PPCF128, "__powl_finite">;

def ceilf : RuntimeLibcallImpl<CEIL_F32>;
def ceil : RuntimeLibcallImpl<CEIL_F64>;
defm ceil : LibmLongDoubleLibCall;

def truncf : RuntimeLibcallImpl<TRUNC_F32>;
def trunc : RuntimeLibcallImpl<TRUNC_F64>;
defm trunc : LibmLongDoubleLibCall;

def rintf : RuntimeLibcallImpl<RINT_F32>;
def rint : RuntimeLibcallImpl<RINT_F64>;
defm rint : LibmLongDoubleLibCall;

def nearbyintf : RuntimeLibcallImpl<NEARBYINT_F32>;
def nearbyint : RuntimeLibcallImpl<NEARBYINT_F64>;
defm nearbyint : LibmLongDoubleLibCall;

def roundf : RuntimeLibcallImpl<ROUND_F32>;
def round : RuntimeLibcallImpl<ROUND_F64>;
defm round : LibmLongDoubleLibCall;

def roundevenf : RuntimeLibcallImpl<ROUNDEVEN_F32>;
def roundeven : RuntimeLibcallImpl<ROUNDEVEN_F64>;
defm roundeven : LibmLongDoubleLibCall;

def floorf : RuntimeLibcallImpl<FLOOR_F32>;
def floor : RuntimeLibcallImpl<FLOOR_F64>;
defm floor : LibmLongDoubleLibCall;

def copysignf : RuntimeLibcallImpl<COPYSIGN_F32>;
def copysign : RuntimeLibcallImpl<COPYSIGN_F64>;
defm copysign : LibmLongDoubleLibCall;

def fminf : RuntimeLibcallImpl<FMIN_F32>;
def fmin : RuntimeLibcallImpl<FMIN_F64>;
defm fmin : LibmLongDoubleLibCall;

def fmaxf : RuntimeLibcallImpl<FMAX_F32>;
def fmax : RuntimeLibcallImpl<FMAX_F64>;
defm fmax : LibmLongDoubleLibCall;

def fminimumf : RuntimeLibcallImpl<FMINIMUM_F32>;
def fminimum : RuntimeLibcallImpl<FMINIMUM_F64>;
defm fminimum : LibmLongDoubleLibCall;

def fmaximumf : RuntimeLibcallImpl<FMAXIMUM_F32>;
def fmaximum : RuntimeLibcallImpl<FMAXIMUM_F64>;
defm fmaximum : LibmLongDoubleLibCall;

def fminimum_numf : RuntimeLibcallImpl<FMINIMUM_NUM_F32>;
def fminimum_num : RuntimeLibcallImpl<FMINIMUM_NUM_F64>;
defm fminimum_num : LibmLongDoubleLibCall;

def fmaximum_numf : RuntimeLibcallImpl<FMAXIMUM_NUM_F32>;
def fmaximum_num : RuntimeLibcallImpl<FMAXIMUM_NUM_F64>;
defm fmaximum_num : LibmLongDoubleLibCall;

def lroundf : RuntimeLibcallImpl<LROUND_F32>;
def lround : RuntimeLibcallImpl<LROUND_F64>;
defm lround : LibmLongDoubleLibCall;

def llroundf : RuntimeLibcallImpl<LLROUND_F32>;
def llround : RuntimeLibcallImpl<LLROUND_F64>;
defm llround : LibmLongDoubleLibCall;

def lrintf : RuntimeLibcallImpl<LRINT_F32>;
def lrint : RuntimeLibcallImpl<LRINT_F64>;
defm lrint : LibmLongDoubleLibCall;

def llrintf : RuntimeLibcallImpl<LLRINT_F32>;
def llrint : RuntimeLibcallImpl<LLRINT_F64>;
defm llrint : LibmLongDoubleLibCall;

def ldexpf : RuntimeLibcallImpl<LDEXP_F32>;
def ldexp : RuntimeLibcallImpl<LDEXP_F64>;
defm ldexp : LibmLongDoubleLibCall;

def frexpf : RuntimeLibcallImpl<FREXP_F32>;
def frexp : RuntimeLibcallImpl<FREXP_F64>;
defm frexp : LibmLongDoubleLibCall;

def sincospif : RuntimeLibcallImpl<SINCOSPI_F32>;
def sincospi : RuntimeLibcallImpl<SINCOSPI_F64>;
defm sincospi : LibmLongDoubleLibCall;

def modff : RuntimeLibcallImpl<MODF_F32>;
def modf : RuntimeLibcallImpl<MODF_F64>;
defm modf : LibmLongDoubleLibCall;

// Floating point environment
def fegetenv : RuntimeLibcallImpl<FEGETENV>;
def fesetenv : RuntimeLibcallImpl<FESETENV>;

// Floating point control modes
def fegetmode : RuntimeLibcallImpl<FEGETMODE>;
def fesetmode : RuntimeLibcallImpl<FESETMODE>;

//--------------------------------------------------------------------
// libc
//--------------------------------------------------------------------

// Memory
def memcpy : RuntimeLibcallImpl<MEMCPY>;
def memmove : RuntimeLibcallImpl<MEMMOVE>;
def memset : RuntimeLibcallImpl<MEMSET>;

// DSEPass can emit calloc if it finds a pair of malloc/memset
def calloc : RuntimeLibcallImpl<CALLOC>;

} // End let IsDefault = true
} // End defset DefaultRuntimeLibcallImpls

//--------------------------------------------------------------------
// Define implementation other libcalls
//--------------------------------------------------------------------

// TODO: Define other custom names targets use to override

def __exp10f : RuntimeLibcallImpl<EXP10_F32>;
def __exp10 : RuntimeLibcallImpl<EXP10_F64>;

def __sincosf_stret : RuntimeLibcallImpl<SINCOS_STRET_F32>;
def __sincos_stret : RuntimeLibcallImpl<SINCOS_STRET_F64>;

def sincosf : RuntimeLibcallImpl<SINCOS_F32>;
def sincos : RuntimeLibcallImpl<SINCOS_F64>;
defm sincos : LibmLongDoubleLibCall;

def bzero : RuntimeLibcallImpl<BZERO>;
def __bzero : RuntimeLibcallImpl<BZERO>;
def _Unwind_SjLj_Resume : RuntimeLibcallImpl<UNWIND_RESUME>;

//===----------------------------------------------------------------------===//
// F128 libm Runtime Libcalls
//===----------------------------------------------------------------------===//

def logf128 : RuntimeLibcallImpl<LOG_F128>;
def log2f128 : RuntimeLibcallImpl<LOG2_F128>;
def log10f128 : RuntimeLibcallImpl<LOG10_F128>;
def expf128 : RuntimeLibcallImpl<EXP_F128>;
def exp2f128 : RuntimeLibcallImpl<EXP2_F128>;
def exp10f128 : RuntimeLibcallImpl<EXP10_F128>;
def sinf128 : RuntimeLibcallImpl<SIN_F128>;
def cosf128 : RuntimeLibcallImpl<COS_F128>;
def tanf128 : RuntimeLibcallImpl<TAN_F128>;
def tanhf128 : RuntimeLibcallImpl<TANH_F128>;
def sincosf128 : RuntimeLibcallImpl<SINCOS_F128>;
def powf128 : RuntimeLibcallImpl<POW_F128>;
def fminf128 : RuntimeLibcallImpl<FMIN_F128>;
def fmaxf128 : RuntimeLibcallImpl<FMAX_F128>;
def fmodf128 : RuntimeLibcallImpl<REM_F128>;
def sqrtf128 : RuntimeLibcallImpl<SQRT_F128>;
def ceilf128 : RuntimeLibcallImpl<CEIL_F128>;
def floorf128 : RuntimeLibcallImpl<FLOOR_F128>;
def truncf128 : RuntimeLibcallImpl<TRUNC_F128>;
def roundf128 : RuntimeLibcallImpl<ROUND_F128>;
def lroundf128 : RuntimeLibcallImpl<LROUND_F128>;
def llroundf128 : RuntimeLibcallImpl<LLROUND_F128>;
def rintf128 : RuntimeLibcallImpl<RINT_F128>;
def lrintf128 : RuntimeLibcallImpl<LRINT_F128>;
def llrintf128 : RuntimeLibcallImpl<LLRINT_F128>;
def nearbyintf128 : RuntimeLibcallImpl<NEARBYINT_F128>;
def fmaf128 : RuntimeLibcallImpl<FMA_F128>;
def frexpf128 : RuntimeLibcallImpl<FREXP_F128>;

def cbrtf128 : RuntimeLibcallImpl<CBRT_F128>;
def fminimumf128 : RuntimeLibcallImpl<FMINIMUM_F128>;
def fmaximumf128 : RuntimeLibcallImpl<FMAXIMUM_F128>;
def fminimum_numf128 : RuntimeLibcallImpl<FMINIMUM_NUM_F128>;
def fmaximum_numf128 : RuntimeLibcallImpl<FMAXIMUM_NUM_F128>;
def asinf128 : RuntimeLibcallImpl<ASIN_F128>;
def acosf128 : RuntimeLibcallImpl<ACOS_F128>;
def atanf128 : RuntimeLibcallImpl<ATAN_F128>;
def atan2f128 : RuntimeLibcallImpl<ATAN2_F128>;
def ldexpf128 : RuntimeLibcallImpl<LDEXP_F128>;
def roundevenf128 : RuntimeLibcallImpl<ROUNDEVEN_F128>;
def modff128 : RuntimeLibcallImpl<MODF_F128>;
def sinhf128 : RuntimeLibcallImpl<SINH_F128>;
def coshf128 : RuntimeLibcallImpl<COSH_F128>;
def copysignf128 : RuntimeLibcallImpl<COPYSIGN_F128>;

def __logf128_finite : RuntimeLibcallImpl<LOG_FINITE_F128>;
def __log2f128_finite : RuntimeLibcallImpl<LOG2_FINITE_F128>;
def __log10f128_finite : RuntimeLibcallImpl<LOG10_FINITE_F128>;
def __expf128_finite : RuntimeLibcallImpl<EXP_FINITE_F128>;
def __exp2f128_finite : RuntimeLibcallImpl<EXP2_FINITE_F128>;
def __exp10f128_finite : RuntimeLibcallImpl<EXP10_FINITE_F128>;
def __powf128_finite : RuntimeLibcallImpl<POW_FINITE_F128>;

//===----------------------------------------------------------------------===//
// PPC Runtime Libcalls
//===----------------------------------------------------------------------===//

class PPCRuntimeLibcallImpl<RuntimeLibcall P, string Name = NAME>
    : RuntimeLibcallImpl<P, Name>;

// For IEEE quad-precision libcall names, PPC uses "kf" instead of "tf".
def __addkf3 : PPCRuntimeLibcallImpl<ADD_F128>;
def __subkf3 : PPCRuntimeLibcallImpl<SUB_F128>;
def __mulkf3 : PPCRuntimeLibcallImpl<MUL_F128>;
def __divkf3 : PPCRuntimeLibcallImpl<DIV_F128>;
def __powikf2 : PPCRuntimeLibcallImpl<POWI_F128>;
def __extendsfkf2 : PPCRuntimeLibcallImpl<FPEXT_F32_F128>;
def __extenddfkf2 : PPCRuntimeLibcallImpl<FPEXT_F64_F128>;
def __trunckfhf2 : PPCRuntimeLibcallImpl<FPROUND_F128_F16>;
def __trunckfsf2 : PPCRuntimeLibcallImpl<FPROUND_F128_F32>;
def __trunckfdf2 : PPCRuntimeLibcallImpl<FPROUND_F128_F64>;
def __fixkfsi : PPCRuntimeLibcallImpl<FPTOSINT_F128_I32>;
def __fixkfdi : PPCRuntimeLibcallImpl<FPTOSINT_F128_I64>;
def __fixkfti : PPCRuntimeLibcallImpl<FPTOSINT_F128_I128>;
def __fixunskfsi : PPCRuntimeLibcallImpl<FPTOUINT_F128_I32>;
def __fixunskfdi : PPCRuntimeLibcallImpl<FPTOUINT_F128_I64>;
def __fixunskfti : PPCRuntimeLibcallImpl<FPTOUINT_F128_I128>;
def __floatsikf : PPCRuntimeLibcallImpl<SINTTOFP_I32_F128>;
def __floatdikf : PPCRuntimeLibcallImpl<SINTTOFP_I64_F128>;
def __floattikf : PPCRuntimeLibcallImpl<SINTTOFP_I128_F128>;
def __floatunsikf : PPCRuntimeLibcallImpl<UINTTOFP_I32_F128>;
def __floatundikf : PPCRuntimeLibcallImpl<UINTTOFP_I64_F128>;
def __floatuntikf : PPCRuntimeLibcallImpl<UINTTOFP_I128_F128>;
def __eqkf2 : PPCRuntimeLibcallImpl<OEQ_F128>;
def __nekf2 : PPCRuntimeLibcallImpl<UNE_F128>;
def __gekf2 : PPCRuntimeLibcallImpl<OGE_F128>;
def __ltkf2 : PPCRuntimeLibcallImpl<OLT_F128>;
def __lekf2 : PPCRuntimeLibcallImpl<OLE_F128>;
def __gtkf2 : PPCRuntimeLibcallImpl<OGT_F128>;
def __unordkf2 : PPCRuntimeLibcallImpl<UO_F128>;

//===----------------------------------------------------------------------===//
// ZOS Runtime Libcalls
//===----------------------------------------------------------------------===//

class ZOSRuntimeLibcallImpl<RuntimeLibcall P, string Name = NAME>
    : RuntimeLibcallImpl<P, Name>;

def zos___TRNC_B : ZOSRuntimeLibcallImpl<TRUNC_F64, "@@TRNC@B">;
def zos___FTRC_B : ZOSRuntimeLibcallImpl<TRUNC_F32, "@@FTRC@B">;
def zos___LTRC_B : ZOSRuntimeLibcallImpl<TRUNC_F128, "@@LTRC@B">;
def zos___WSQT_B : ZOSRuntimeLibcallImpl<SQRT_F64, "@@WSQT@B">;
def zos___FSQT_B : ZOSRuntimeLibcallImpl<SQRT_F32, "@@FSQT@B">;
def zos___LSQT_B : ZOSRuntimeLibcallImpl<SQRT_F128, "@@LSQT@B">;
def zos___SSIN_B : ZOSRuntimeLibcallImpl<SIN_F64, "@@SSIN@B">;
def zos___FSIN_B : ZOSRuntimeLibcallImpl<SIN_F32, "@@FSIN@B">;
def zos___LSIN_B : ZOSRuntimeLibcallImpl<SIN_F128, "@@LSIN@B">;
def zos___ROUN_B : ZOSRuntimeLibcallImpl<ROUND_F64, "@@ROUN@B">;
def zos___ROUNFB : ZOSRuntimeLibcallImpl<ROUND_F32, "@@ROUNFB">;
def zos___ROUNLB : ZOSRuntimeLibcallImpl<ROUND_F128, "@@ROUNLB">;
def zos___SRNT_B : ZOSRuntimeLibcallImpl<RINT_F64, "@@SRNT@B">;
def zos___RINTFB : ZOSRuntimeLibcallImpl<RINT_F32, "@@RINTFB">;
def zos___RINTLB : ZOSRuntimeLibcallImpl<RINT_F128, "@@RINTLB">;
def zos___WFMD_B : ZOSRuntimeLibcallImpl<REM_F64, "@@WFMD@B">;
def zos___FFMD_B : ZOSRuntimeLibcallImpl<REM_F32, "@@FFMD@B">;
def zos___LFMD_B : ZOSRuntimeLibcallImpl<REM_F128, "@@LFMD@B">;
def zos___WPOW_B : ZOSRuntimeLibcallImpl<POW_F64, "@@WPOW@B">;
def zos___FPOW_B : ZOSRuntimeLibcallImpl<POW_F32, "@@FPOW@B">;
def zos___LPOW_B : ZOSRuntimeLibcallImpl<POW_F128, "@@LPOW@B">;
def zos___NBYI_B : ZOSRuntimeLibcallImpl<NEARBYINT_F64, "@@NBYI@B">;
def zos___NBYIFB : ZOSRuntimeLibcallImpl<NEARBYINT_F32, "@@NBYIFB">;
def zos___NBYILB : ZOSRuntimeLibcallImpl<NEARBYINT_F128, "@@NBYILB">;
def zos___ROND_B : ZOSRuntimeLibcallImpl<LROUND_F64, "@@ROND@B">;
def zos___FRND_B : ZOSRuntimeLibcallImpl<LROUND_F32, "@@FRND@B">;
def zos___LRND_B : ZOSRuntimeLibcallImpl<LROUND_F128, "@@LRND@B">;
def zos___LRNT_B : ZOSRuntimeLibcallImpl<LRINT_F64, "@@LRNT@B">;
def zos___LRNTFB : ZOSRuntimeLibcallImpl<LRINT_F32, "@@LRNTFB">;
def zos___LRNTLB : ZOSRuntimeLibcallImpl<LRINT_F128, "@@LRNTLB">;
def zos___WLOG_B : ZOSRuntimeLibcallImpl<LOG_F64, "@@WLOG@B">;
def zos___FLOG_B : ZOSRuntimeLibcallImpl<LOG_F32, "@@FLOG@B">;
def zos___LLOG_B : ZOSRuntimeLibcallImpl<LOG_F128, "@@LLOG@B">;
def zos___LOG2_B : ZOSRuntimeLibcallImpl<LOG2_F64, "@@LOG2@B">;
def zos___FLG2_B : ZOSRuntimeLibcallImpl<LOG2_F32, "@@FLG2@B">;
def zos___LLG2_B : ZOSRuntimeLibcallImpl<LOG2_F128, "@@LLG2@B">;
def zos___WLG1_B : ZOSRuntimeLibcallImpl<LOG10_F64, "@@WLG1@B">;
def zos___FLG1_B : ZOSRuntimeLibcallImpl<LOG10_F32, "@@FLG1@B">;
def zos___LLG1_B : ZOSRuntimeLibcallImpl<LOG10_F128, "@@LLG1@B">;
def zos___LLRD_B : ZOSRuntimeLibcallImpl<LLROUND_F64, "@@LLRD@B">;
def zos___LLRDFB : ZOSRuntimeLibcallImpl<LLROUND_F32, "@@LLRDFB">;
def zos___LLRDLB : ZOSRuntimeLibcallImpl<LLROUND_F128, "@@LLRDLB">;
def zos___LLRT_B : ZOSRuntimeLibcallImpl<LLRINT_F64, "@@LLRT@B">;
def zos___LLRTFB : ZOSRuntimeLibcallImpl<LLRINT_F32, "@@LLRTFB">;
def zos___LLRTLB : ZOSRuntimeLibcallImpl<LLRINT_F128, "@@LLRTLB">;
def zos___SLXP_B : ZOSRuntimeLibcallImpl<LDEXP_F64, "@@SLXP@B">;
def zos___FLXP_B : ZOSRuntimeLibcallImpl<LDEXP_F32, "@@FLXP@B">;
def zos___LLXP_B : ZOSRuntimeLibcallImpl<LDEXP_F128, "@@LLXP@B">;
def zos___SFXP_B : ZOSRuntimeLibcallImpl<FREXP_F64, "@@SFXP@B">;
def zos___FFXP_B : ZOSRuntimeLibcallImpl<FREXP_F32, "@@FFXP@B">;
def zos___LFXP_B : ZOSRuntimeLibcallImpl<FREXP_F128, "@@LFXP@B">;
def zos___FMIN_B : ZOSRuntimeLibcallImpl<FMIN_F64, "@@FMIN@B">;
def zos___FMINFB : ZOSRuntimeLibcallImpl<FMIN_F32, "@@FMINFB">;
def zos___FMINLB : ZOSRuntimeLibcallImpl<FMIN_F128, "@@FMINLB">;
def zos___FMA_B : ZOSRuntimeLibcallImpl<FMA_F64, "@@FMA@B">;
def zos___FMAFB : ZOSRuntimeLibcallImpl<FMA_F32, "@@FMAFB">;
def zos___FMALB : ZOSRuntimeLibcallImpl<FMA_F128, "@@FMALB">;
def zos___FMAX_B : ZOSRuntimeLibcallImpl<FMAX_F64, "@@FMAX@B">;
def zos___FMAXFB : ZOSRuntimeLibcallImpl<FMAX_F32, "@@FMAXFB">;
def zos___FMAXLB : ZOSRuntimeLibcallImpl<FMAX_F128, "@@FMAXLB">;
def zos___SFLR_B : ZOSRuntimeLibcallImpl<FLOOR_F64, "@@SFLR@B">;
def zos___FFLR_B : ZOSRuntimeLibcallImpl<FLOOR_F32, "@@FFLR@B">;
def zos___LFLR_B : ZOSRuntimeLibcallImpl<FLOOR_F128, "@@LFLR@B">;
def zos___WEXP_B : ZOSRuntimeLibcallImpl<EXP_F64, "@@WEXP@B">;
def zos___FEXP_B : ZOSRuntimeLibcallImpl<EXP_F32, "@@FEXP@B">;
def zos___LEXP_B : ZOSRuntimeLibcallImpl<EXP_F128, "@@LEXP@B">;
def zos___EXP2_B : ZOSRuntimeLibcallImpl<EXP2_F64, "@@EXP2@B">;
def zos___FXP2_B : ZOSRuntimeLibcallImpl<EXP2_F32, "@@FXP2@B">;
def zos___LXP2_B : ZOSRuntimeLibcallImpl<EXP2_F128, "@@LXP2@B">;
def zos___SCOS_B : ZOSRuntimeLibcallImpl<COS_F64, "@@SCOS@B">;
def zos___FCOS_B : ZOSRuntimeLibcallImpl<COS_F32, "@@FCOS@B">;
def zos___LCOS_B : ZOSRuntimeLibcallImpl<COS_F128, "@@LCOS@B">;
def zos___DCPY_B : ZOSRuntimeLibcallImpl<COPYSIGN_F64, "@@DCPY@B">;
def zos___FCPY_B : ZOSRuntimeLibcallImpl<COPYSIGN_F32, "@@FCPY@B">;
def zos___LCPY_B : ZOSRuntimeLibcallImpl<COPYSIGN_F128, "@@LCPY@B">;
def zos___SCEL_B : ZOSRuntimeLibcallImpl<CEIL_F64, "@@SCEL@B">;
def zos___FCEL_B : ZOSRuntimeLibcallImpl<CEIL_F32, "@@FCEL@B">;
def zos___LCEL_B : ZOSRuntimeLibcallImpl<CEIL_F128, "@@LCEL@B">;
def zos___SCRT_B : ZOSRuntimeLibcallImpl<CBRT_F64, "@@SCRT@B">;
def zos___FCBT_B : ZOSRuntimeLibcallImpl<CBRT_F32, "@@FCBT@B">;
def zos___LCBT_B : ZOSRuntimeLibcallImpl<CBRT_F128, "@@LCBT@B">;
