//===-- llvm/RuntimeLibcallList.td - Describe libcalls --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/IR/RuntimeLibcallsImpl.td"

//--------------------------------------------------------------------
// Utility classes
//--------------------------------------------------------------------

class DuplicateLibcallImplWithPrefix<RuntimeLibcallImpl Impl, string prefix>
    : RuntimeLibcallImpl<Impl.Provides, prefix#Impl.LibCallFuncName>;

/// Libcall Predicates
def isOSDarwin : RuntimeLibcallPredicate<"TT.isOSDarwin()">;
def isOSOpenBSD : RuntimeLibcallPredicate<"TT.isOSOpenBSD()">;
def isNotOSOpenBSD : RuntimeLibcallPredicate<"!TT.isOSOpenBSD()">;
def isOSWindows : RuntimeLibcallPredicate<"TT.isOSWindows()">;
def isNotOSWindows : RuntimeLibcallPredicate<"!TT.isOSWindows()">;
def isNotOSMSVCRT : RuntimeLibcallPredicate<"!TT.isOSMSVCRT()">;
def isPS : RuntimeLibcallPredicate<"TT.isPS()">;
def isNotOSWindowsOrIsCygwinMinGW
  : RuntimeLibcallPredicate<"!TT.isOSWindows() || TT.isOSCygMing()">;
def isWindowsMSVCEnvironment : RuntimeLibcallPredicate<
  [{TT.isWindowsMSVCEnvironment()}]>;

def isGNUEnvironment : RuntimeLibcallPredicate<"TT.isGNUEnvironment()">;
def darwinHasSinCosStret : RuntimeLibcallPredicate<"darwinHasSinCosStret(TT)">;
def darwinHasExp10 : RuntimeLibcallPredicate<"darwinHasExp10(TT)">;

def hasExp10 : RuntimeLibcallPredicate<[{!TT.isOSDarwin()}]>;

def hasSinCos : RuntimeLibcallPredicate<"hasSinCos(TT)">;

// FIXME: Way to combine predicates
def hasSinCos_f32_f64 : RuntimeLibcallPredicate<"hasSinCos_f32_f64(TT)">;

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
  def FAST_ADD_#FPTy : RuntimeLibcall;

  def SUB_#FPTy : RuntimeLibcall;
  def FAST_SUB_#FPTy : RuntimeLibcall;

  def MUL_#FPTy : RuntimeLibcall;
  def FAST_MUL_#FPTy : RuntimeLibcall;

  def DIV_#FPTy : RuntimeLibcall;
  def FAST_DIV_#FPTy : RuntimeLibcall;

  def REM_#FPTy : RuntimeLibcall;
  def FMA_#FPTy : RuntimeLibcall;
  def POWI_#FPTy : RuntimeLibcall;

  def SQRT_#FPTy : RuntimeLibcall;
  def FAST_SQRT_#FPTy : RuntimeLibcall;

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
def MEMCMP : RuntimeLibcall;
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
def UNWIND_REGISTER : RuntimeLibcall;
def UNWIND_UNREGISTER : RuntimeLibcall;
def UNWIND_CALL_PERSONALITY : RuntimeLibcall;
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

multiclass AtomicOrderSizeLibcallImpl<string ProvidesBase> {
  def _relax : RuntimeLibcallImpl<!cast<RuntimeLibcall>(ProvidesBase#"_RELAX")>;
  def _acq : RuntimeLibcallImpl<!cast<RuntimeLibcall>(ProvidesBase#"_ACQ")>;
  def _rel : RuntimeLibcallImpl<!cast<RuntimeLibcall>(ProvidesBase#"_REL")>;
  def _acq_rel : RuntimeLibcallImpl<!cast<RuntimeLibcall>(ProvidesBase#"_ACQ_REL")>;
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
def STACK_SMASH_HANDLER : RuntimeLibcall;

// Safe stack
def SAFESTACK_POINTER_ADDRESS : RuntimeLibcall;

def SECURITY_CHECK_COOKIE : RuntimeLibcall;

// Deoptimization
def DEOPTIMIZE : RuntimeLibcall;

// Return address
def RETURN_ADDRESS : RuntimeLibcall;

// Clear cache
def CLEAR_CACHE : RuntimeLibcall;
def RISCV_FLUSH_ICACHE : RuntimeLibcall;

// Mips16 calls
def MIPS16_RET_DC : RuntimeLibcall;
def MIPS16_RET_DF : RuntimeLibcall;
def MIPS16_RET_SC : RuntimeLibcall;
def MIPS16_RET_SF : RuntimeLibcall;

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

// AArch64 calls
def SC_MEMCPY : RuntimeLibcall;
def SC_MEMMOVE : RuntimeLibcall;
def SC_MEMSET : RuntimeLibcall;

// ARM EABI calls
def AEABI_MEMCPY4 : RuntimeLibcall; // Align 4
def AEABI_MEMCPY8 : RuntimeLibcall; // Align 8
def AEABI_MEMMOVE4 : RuntimeLibcall;
def AEABI_MEMMOVE8 : RuntimeLibcall;
def AEABI_MEMSET4 : RuntimeLibcall;
def AEABI_MEMSET8 : RuntimeLibcall;
def AEABI_MEMCLR : RuntimeLibcall;
def AEABI_MEMCLR4 : RuntimeLibcall;
def AEABI_MEMCLR8 : RuntimeLibcall;

// Hexagon calls
def HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES : RuntimeLibcall;

// XCore calls
def MEMCPY_ALIGN_4 : RuntimeLibcall;

// Objective-C calls
def OBJC_AUTORELEASE : RuntimeLibcall;
def OBJC_AUTORELEASEPOOLPOP : RuntimeLibcall;
def OBJC_AUTORELEASEPOOLPUSH : RuntimeLibcall;
def OBJC_AUTORELEASERETURNVALUE : RuntimeLibcall;
def OBJC_COPYWEAK : RuntimeLibcall;
def OBJC_DESTROYWEAK : RuntimeLibcall;
def OBJC_INITWEAK : RuntimeLibcall;
def OBJC_LOADWEAK : RuntimeLibcall;
def OBJC_LOADWEAKRETAINED : RuntimeLibcall;
def OBJC_MOVEWEAK : RuntimeLibcall;
def OBJC_RELEASE : RuntimeLibcall;
def OBJC_RETAIN : RuntimeLibcall;
def OBJC_RETAINAUTORELEASE : RuntimeLibcall;
def OBJC_RETAINAUTORELEASERETURNVALUE : RuntimeLibcall;
def OBJC_RETAINAUTORELEASEDRETURNVALUE : RuntimeLibcall;
def OBJC_CLAIMAUTORELEASEDRETURNVALUE : RuntimeLibcall;
def OBJC_RETAINBLOCK : RuntimeLibcall;
def OBJC_STORESTRONG : RuntimeLibcall;
def OBJC_STOREWEAK : RuntimeLibcall;
def OBJC_UNSAFECLAIMAUTORELEASEDRETURNVALUE : RuntimeLibcall;
def OBJC_RETAINEDOBJECT : RuntimeLibcall;
def OBJC_UNRETAINEDOBJECT : RuntimeLibcall;
def OBJC_UNRETAINEDPOINTER : RuntimeLibcall;
def OBJC_RETAIN_AUTORELEASE : RuntimeLibcall;
def OBJC_SYNC_ENTER : RuntimeLibcall;
def OBJC_SYNC_EXIT : RuntimeLibcall;

//--------------------------------------------------------------------
// Define implementation default libcalls
//--------------------------------------------------------------------

let IsDefault = true in {

//--------------------------------------------------------------------
// compiler-rt / libgcc
//--------------------------------------------------------------------

def __ashlhi3 : RuntimeLibcallImpl<SHL_I16>;
def __ashlsi3 : RuntimeLibcallImpl<SHL_I32>;
def __ashldi3 : RuntimeLibcallImpl<SHL_I64>;

def __lshrhi3 : RuntimeLibcallImpl<SRL_I16>;
def __lshrsi3 : RuntimeLibcallImpl<SRL_I32>;
def __lshrdi3 : RuntimeLibcallImpl<SRL_I64>;

def __ashrhi3 : RuntimeLibcallImpl<SRA_I16>;
def __ashrsi3 : RuntimeLibcallImpl<SRA_I32>;
def __ashrdi3 : RuntimeLibcallImpl<SRA_I64>;

def __mulqi3 : RuntimeLibcallImpl<MUL_I8>;
def __mulhi3 : RuntimeLibcallImpl<MUL_I16>;
def __mulsi3 : RuntimeLibcallImpl<MUL_I32>;
def __muldi3 : RuntimeLibcallImpl<MUL_I64>;

def __mulosi4 : RuntimeLibcallImpl<MULO_I32>;

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

defset list<RuntimeLibcallImpl> PowiLibcallImpls = {
def __powisf2 : RuntimeLibcallImpl<POWI_F32>;
def __powidf2 : RuntimeLibcallImpl<POWI_F64>;
def __powixf2 : RuntimeLibcallImpl<POWI_F80>;
def __powitf2_f128 : RuntimeLibcallImpl<POWI_F128, "__powitf2">;
def __powitf2_ppc128 : RuntimeLibcallImpl<POWI_PPCF128, "__powitf2">;
}

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

// Safe stack.
def __safestack_pointer_address : RuntimeLibcallImpl<SAFESTACK_POINTER_ADDRESS>;

// Deoptimization
def __llvm_deoptimize : RuntimeLibcallImpl<DEOPTIMIZE>;

// Clear cache
def __clear_cache : RuntimeLibcallImpl<CLEAR_CACHE>;

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

def exp10f : RuntimeLibcallImpl<EXP10_F32>;
def exp10 : RuntimeLibcallImpl<EXP10_F64>;
def exp10l_f80 : RuntimeLibcallImpl<EXP10_F80, "exp10l">;
def exp10l_f128 : RuntimeLibcallImpl<EXP10_F128, "exp10l">;
def exp10l_ppcf128 : RuntimeLibcallImpl<EXP10_PPCF128, "exp10l">;

// Stack Protector Fail
def __stack_chk_fail : RuntimeLibcallImpl<STACKPROTECTOR_CHECK_FAIL>;

//--------------------------------------------------------------------
// compiler-rt/libgcc but 64-bit only, not available by default
//--------------------------------------------------------------------

// Exist in libgcc and compiler-rt for 64-bit targets, or if
// COMPILER_RT_ENABLE_SOFTWARE_INT128.
defset list<RuntimeLibcallImpl> Int128RTLibcalls = {
  def __ashlti3 : RuntimeLibcallImpl<SHL_I128>;
  def __lshrti3 : RuntimeLibcallImpl<SRL_I128>;
  def __ashrti3 : RuntimeLibcallImpl<SRA_I128>;
  def __multi3 : RuntimeLibcallImpl<MUL_I128>;
}

//--------------------------------------------------------------------
// compiler-rt only, not available by default
//--------------------------------------------------------------------

defset list<RuntimeLibcallImpl> CompilerRTOnlyInt64Libcalls = {
  def __mulodi4 : RuntimeLibcallImpl<MULO_I64>;
}

defset list<RuntimeLibcallImpl> CompilerRTOnlyInt128Libcalls = {
  def __muloti4 : RuntimeLibcallImpl<MULO_I128>;
}

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
def _Unwind_SjLj_Register : RuntimeLibcallImpl<UNWIND_REGISTER>;
def _Unwind_SjLj_Unregister : RuntimeLibcallImpl<UNWIND_UNREGISTER>;
def _Unwind_CallPersonality : RuntimeLibcallImpl<UNWIND_CALL_PERSONALITY>;

// Used on OpenBSD
def __stack_smash_handler : RuntimeLibcallImpl<STACK_SMASH_HANDLER>;

def __riscv_flush_icache : RuntimeLibcallImpl<RISCV_FLUSH_ICACHE>;

def __security_check_cookie : RuntimeLibcallImpl<SECURITY_CHECK_COOKIE>;
def __security_check_cookie_arm64ec : RuntimeLibcallImpl<SECURITY_CHECK_COOKIE,
  "#__security_check_cookie_arm64ec">;

//===----------------------------------------------------------------------===//
// F128 libm Runtime Libcalls
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> LibmF128Libcalls = {
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
}

defset list<RuntimeLibcallImpl> LibmF128FiniteLibcalls = {
  def __logf128_finite : RuntimeLibcallImpl<LOG_FINITE_F128>;
  def __log2f128_finite : RuntimeLibcallImpl<LOG2_FINITE_F128>;
  def __log10f128_finite : RuntimeLibcallImpl<LOG10_FINITE_F128>;
  def __expf128_finite : RuntimeLibcallImpl<EXP_FINITE_F128>;
  def __exp2f128_finite : RuntimeLibcallImpl<EXP2_FINITE_F128>;
  def __exp10f128_finite : RuntimeLibcallImpl<EXP10_FINITE_F128>;
  def __powf128_finite : RuntimeLibcallImpl<POW_FINITE_F128>;
}

//===----------------------------------------------------------------------===//
// Common Libcall Sets
//===----------------------------------------------------------------------===//

// FIXME: Should move to explicit opt-in to different sets of libcalls
// instead of trying to remove from a default set. We have
// unreasonable defaults like reporting f80 calls on most targets when
// they are relevant to only one.

defvar AllDefaultRuntimeLibcallImpls
  = !filter(entry, !instances<RuntimeLibcallImpl>(), entry.IsDefault);

defvar DefaultRuntimeLibcallImpls_f80 =
    !filter(entry, AllDefaultRuntimeLibcallImpls,
            !match(!cast<string>(entry.Provides), "F80"));

defvar DefaultRuntimeLibcallImpls_ppcf128 =
    !filter(entry, AllDefaultRuntimeLibcallImpls,
            !match(!cast<string>(entry.Provides), "PPCF128"));

defvar DefaultRuntimeLibcallImpls_f128 =
    !filter(entry, AllDefaultRuntimeLibcallImpls,
            !match(!cast<string>(entry.Provides), "_F128"));

defvar DefaultRuntimeLibcallImpls =
!listremove(
  !listremove(
    !listremove(AllDefaultRuntimeLibcallImpls, Int128RTLibcalls),
                DefaultRuntimeLibcallImpls_f80),
                DefaultRuntimeLibcallImpls_ppcf128);

/// Default set of libcall impls for 32-bit architectures.
defvar DefaultLibcallImpls32 = DefaultRuntimeLibcallImpls;

/// Default set of libcall impls for 64-bit architectures.
defvar DefaultLibcallImpls64 = !listconcat(DefaultRuntimeLibcallImpls,
                                           Int128RTLibcalls);

defvar DarwinSinCosStret = LibcallImpls<(add __sincosf_stret, __sincos_stret),
                                        darwinHasSinCosStret>;
defvar DarwinExp10 = LibcallImpls<(add __exp10f, __exp10), darwinHasExp10>;

defvar SecurityCheckCookieIfWinMSVC =
    LibcallImpls<(add __security_check_cookie), isWindowsMSVCEnvironment>;

defvar LibmHasSinCosF32 = LibcallImpls<(add sincosf), hasSinCos>;
defvar LibmHasSinCosF64 =  LibcallImpls<(add sincos), hasSinCos>;
defvar LibmHasSinCosF80 = LibcallImpls<(add sincos_f80), hasSinCos>;
defvar LibmHasSinCosF128 = LibcallImpls<(add sincos_f128), hasSinCos>;
defvar LibmHasSinCosPPCF128 = LibcallImpls<(add sincos_ppcf128), hasSinCos>;

defvar LibmHasExp10F32 = LibcallImpls<(add exp10f), hasExp10>;
defvar LibmHasExp10F64 = LibcallImpls<(add exp10), hasExp10>;
defvar LibmHasExp10F80 = LibcallImpls<(add exp10l_f80), hasExp10>;
defvar LibmHasExp10F128 = LibcallImpls<(add exp10l_f128), hasExp10>;
defvar LibmHasExp10PPCF128 = LibcallImpls<(add exp10l_ppcf128), hasExp10>;

defvar DefaultLibmExp10 = [
  LibmHasExp10F32, LibmHasExp10F64, LibmHasExp10F128
];


defvar WindowsMathRemovals = [
  ldexpf, ldexp_f80, ldexp_f128, ldexp_ppcf128,
  frexpf, frexp_f80, frexp_f128, frexp_ppcf128
];

defvar MostPowI = !listremove(PowiLibcallImpls, [__powitf2_f128, __powitf2_ppc128]);
defvar WindowsExclusions = !listconcat(WindowsMathRemovals, MostPowI);

// Targets which support windows should start with these as a base and
// add in calls for other OSes
defvar WinDefaultLibcallImpls = !listremove(DefaultRuntimeLibcallImpls, WindowsExclusions);

defvar LibmHasFrexpF32 = LibcallImpls<(add frexpf), isNotOSWindowsOrIsCygwinMinGW>;
defvar LibmHasLdexpF32 = LibcallImpls<(add ldexpf), isNotOSWindowsOrIsCygwinMinGW>;

defvar LibmHasFrexpF80 = LibcallImpls<(add frexp_f80), isNotOSWindowsOrIsCygwinMinGW>;
defvar LibmHasLdexpF80 = LibcallImpls<(add ldexp_f80), isNotOSWindowsOrIsCygwinMinGW>;

defvar LibmHasFrexpF128 = LibcallImpls<(add frexp_f128), isNotOSWindowsOrIsCygwinMinGW>;
defvar LibmHasLdexpF128 = LibcallImpls<(add ldexp_f128), isNotOSWindowsOrIsCygwinMinGW>;

defvar has__stack_chk_fail = LibcallImpls<(add __stack_chk_fail), isNotOSOpenBSD>;
defvar has__stack_smash_handler = LibcallImpls<(add __stack_smash_handler), isOSOpenBSD>;

defvar DefaultStackProtector = (add has__stack_chk_fail,
                                    has__stack_smash_handler);

//===----------------------------------------------------------------------===//
// Objective-C Runtime Libcalls
//===----------------------------------------------------------------------===//

def objc_autorelease : RuntimeLibcallImpl<OBJC_AUTORELEASE>;
def objc_autoreleasePoolPop : RuntimeLibcallImpl<OBJC_AUTORELEASEPOOLPOP>;
def objc_autoreleasePoolPush : RuntimeLibcallImpl<OBJC_AUTORELEASEPOOLPUSH>;
def objc_autoreleaseReturnValue : RuntimeLibcallImpl<OBJC_AUTORELEASERETURNVALUE>;
def objc_copyWeak : RuntimeLibcallImpl<OBJC_COPYWEAK>;
def objc_destroyWeak : RuntimeLibcallImpl<OBJC_DESTROYWEAK>;
def objc_initWeak : RuntimeLibcallImpl<OBJC_INITWEAK>;
def objc_loadWeak : RuntimeLibcallImpl<OBJC_LOADWEAK>;
def objc_loadWeakRetained : RuntimeLibcallImpl<OBJC_LOADWEAKRETAINED>;
def objc_moveWeak : RuntimeLibcallImpl<OBJC_MOVEWEAK>;
def objc_release : RuntimeLibcallImpl<OBJC_RELEASE>;
def objc_retain : RuntimeLibcallImpl<OBJC_RETAIN>;
def objc_retainAutorelease : RuntimeLibcallImpl<OBJC_RETAINAUTORELEASE>;
def objc_retainAutoreleaseReturnValue : RuntimeLibcallImpl<OBJC_RETAINAUTORELEASERETURNVALUE>;
def objc_retainAutoreleasedReturnValue : RuntimeLibcallImpl<OBJC_RETAINAUTORELEASEDRETURNVALUE>;
def objc_claimAutoreleasedReturnValue : RuntimeLibcallImpl<OBJC_CLAIMAUTORELEASEDRETURNVALUE>;
def objc_retainBlock : RuntimeLibcallImpl<OBJC_RETAINBLOCK>;
def objc_storeStrong : RuntimeLibcallImpl<OBJC_STORESTRONG>;
def objc_storeWeak : RuntimeLibcallImpl<OBJC_STOREWEAK>;
def objc_unsafeClaimAutoreleasedReturnValue : RuntimeLibcallImpl<OBJC_UNSAFECLAIMAUTORELEASEDRETURNVALUE>;
def objc_retainedObject : RuntimeLibcallImpl<OBJC_RETAINEDOBJECT>;
def objc_unretainedObject : RuntimeLibcallImpl<OBJC_UNRETAINEDOBJECT>;
def objc_unretainedPointer : RuntimeLibcallImpl<OBJC_UNRETAINEDPOINTER>;
def objc_retain_autorelease : RuntimeLibcallImpl<OBJC_RETAIN_AUTORELEASE>;
def objc_sync_enter : RuntimeLibcallImpl<OBJC_SYNC_ENTER>;
def objc_sync_exit : RuntimeLibcallImpl<OBJC_SYNC_EXIT>;

//===----------------------------------------------------------------------===//
// AArch64 Runtime Libcalls
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> AArch64LibcallImpls = {
  foreach MemSize = [1, 2, 4, 8, 16] in {
    defm __aarch64_cas#MemSize
        : AtomicOrderSizeLibcallImpl<"OUTLINE_ATOMIC_CAS"#MemSize>;
  }

  foreach MemSize = [1, 2, 4, 8] in {
    defm __aarch64_swp#MemSize
        : AtomicOrderSizeLibcallImpl<"OUTLINE_ATOMIC_SWP"#MemSize>;
    defm __aarch64_ldadd#MemSize
        : AtomicOrderSizeLibcallImpl<"OUTLINE_ATOMIC_LDADD"#MemSize>;
    defm __aarch64_ldset#MemSize
        : AtomicOrderSizeLibcallImpl<"OUTLINE_ATOMIC_LDSET"#MemSize>;
    defm __aarch64_ldclr#MemSize
        : AtomicOrderSizeLibcallImpl<"OUTLINE_ATOMIC_LDCLR"#MemSize>;
    defm __aarch64_ldeor#MemSize
        : AtomicOrderSizeLibcallImpl<"OUTLINE_ATOMIC_LDEOR"#MemSize>;
  }

  def __arm_sc_memcpy : RuntimeLibcallImpl<SC_MEMCPY>;
  def __arm_sc_memmove : RuntimeLibcallImpl<SC_MEMMOVE>;
  def __arm_sc_memset : RuntimeLibcallImpl<SC_MEMSET>;
} // End AArch64LibcallImpls

def isAArch64_ExceptArm64EC
    : RuntimeLibcallPredicate<"(TT.isAArch64() && !TT.isWindowsArm64EC())">;
def isWindowsArm64EC : RuntimeLibcallPredicate<"TT.isWindowsArm64EC()">;
def isAArch64_ILP64  : RuntimeLibcallPredicate<"TT.isAArch64(64)">;


def AArch64SystemLibrary : SystemRuntimeLibrary<
  isAArch64_ExceptArm64EC,
  (add WinDefaultLibcallImpls,
       LibcallImpls<(add __powisf2, __powidf2), isNotOSMSVCRT>,
       LibmHasFrexpF32, LibmHasLdexpF32,
       LibmHasFrexpF128, LibmHasLdexpF128,
       AArch64LibcallImpls,
       LibcallImpls<(add Int128RTLibcalls), isAArch64_ILP64>,
       LibcallImpls<(add bzero), isOSDarwin>,
       DarwinExp10, DarwinSinCosStret,
       LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
       DefaultLibmExp10,
       DefaultStackProtector,
       SecurityCheckCookieIfWinMSVC)
>;

// Prepend a # to every name
defset list<RuntimeLibcallImpl> WinArm64ECDefaultRuntimeLibcallImpls = {
  foreach libcall = WinDefaultLibcallImpls in {
    def arm64ec_#libcall : DuplicateLibcallImplWithPrefix<libcall, "#">;
  }

  foreach libcall = AArch64LibcallImpls in {
    def arm64ec_#libcall : DuplicateLibcallImplWithPrefix<libcall, "#">;
  }
}

def arm64ec___stack_chk_fail : DuplicateLibcallImplWithPrefix<__stack_chk_fail, "#">;

def WindowsARM64ECSystemLibrary
    : SystemRuntimeLibrary<isWindowsArm64EC,
                           (add WinArm64ECDefaultRuntimeLibcallImpls,
                                arm64ec___stack_chk_fail,
                                LibcallImpls<(add __security_check_cookie_arm64ec),
                                              isWindowsMSVCEnvironment>)>;

//===----------------------------------------------------------------------===//
// AMDGPU Runtime Libcalls
//===----------------------------------------------------------------------===//

def isAMDGPU : RuntimeLibcallPredicate<"TT.isAMDGPU()">;

// No calls.
def AMDGPUSystemLibrary : SystemRuntimeLibrary<isAMDGPU, (add)>;

//===----------------------------------------------------------------------===//
// ARM Runtime Libcalls
//===----------------------------------------------------------------------===//

// if (isTargetMachO()) {
// if (Subtarget->isThumb() && Subtarget->hasVFP2Base() &&
//  Subtarget->hasARMOps() && !Subtarget->useSoftFloat()) {

def __addsf3vfp : RuntimeLibcallImpl<ADD_F32>;
def __subsf3vfp : RuntimeLibcallImpl<SUB_F32>;
def __mulsf3vfp : RuntimeLibcallImpl<MUL_F32>;
def __divsf3vfp : RuntimeLibcallImpl<DIV_F32>;

// Double-precision floating-point arithmetic.
def __adddf3vfp : RuntimeLibcallImpl<ADD_F64>;
def __subdf3vfp : RuntimeLibcallImpl<SUB_F64>;
def __muldf3vfp : RuntimeLibcallImpl<MUL_F64>;
def __divdf3vfp : RuntimeLibcallImpl<DIV_F64>;

// Single-precision comparisons.

// TODO: Track setcc type
def __eqsf2vfp : RuntimeLibcallImpl<OEQ_F32>; // CmpInst::ICMP_NE
def __nesf2vfp : RuntimeLibcallImpl<UNE_F32>; // CmpInst::ICMP_NE
def __ltsf2vfp : RuntimeLibcallImpl<OLT_F32>; // CmpInst::ICMP_NE
def __lesf2vfp : RuntimeLibcallImpl<OLE_F32>; // CmpInst::ICMP_NE
def __gesf2vfp : RuntimeLibcallImpl<OGE_F32>; // CmpInst::ICMP_NE
def __gtsf2vfp : RuntimeLibcallImpl<OGT_F32>; // CmpInst::ICMP_NE
def __unordsf2vfp : RuntimeLibcallImpl<UO_F32>; // CmpInst::ICMP_NE

// Double-precision comparisons.
def __eqdf2vfp : RuntimeLibcallImpl<OEQ_F64>; // CmpInst::ICMP_NE
def __nedf2vfp : RuntimeLibcallImpl<UNE_F64>; // CmpInst::ICMP_NE
def __ltdf2vfp : RuntimeLibcallImpl<OLT_F64>; // CmpInst::ICMP_NE
def __ledf2vfp : RuntimeLibcallImpl<OLE_F64>; // CmpInst::ICMP_NE
def __gedf2vfp : RuntimeLibcallImpl<OGE_F64>; // CmpInst::ICMP_NE
def __gtdf2vfp : RuntimeLibcallImpl<OGT_F64>; // CmpInst::ICMP_NE
def __unorddf2vfp : RuntimeLibcallImpl<UO_F64>; // CmpInst::ICMP_NE

// Floating-point to integer conversions.
// i64 conversions are done via library routines even when generating VFP
// instructions, so use the same ones.
def __fixdfsivfp : RuntimeLibcallImpl<FPTOSINT_F64_I32>;
def __fixunsdfsivfp : RuntimeLibcallImpl<FPTOUINT_F64_I32>;
def __fixsfsivfp : RuntimeLibcallImpl<FPTOSINT_F32_I32>;
def __fixunssfsivfp : RuntimeLibcallImpl<FPTOUINT_F32_I32>;

// Conversions between floating types.
def __truncdfsf2vfp : RuntimeLibcallImpl<FPROUND_F64_F32>;
def __extendsfdf2vfp : RuntimeLibcallImpl<FPEXT_F32_F64>;

// Integer to floating-point conversions.
// i64 conversions are done via library routines even when generating VFP
// instructions, so use the same ones.
// FIXME: There appears to be some naming inconsistency in ARM libgcc:
// e.g., __floatunsidf vs. __floatunssidfvfp.
def __floatsidfvfp : RuntimeLibcallImpl<SINTTOFP_I32_F64>;
def __floatunssidfvfp : RuntimeLibcallImpl<UINTTOFP_I32_F64>;
def __floatsisfvfp : RuntimeLibcallImpl<SINTTOFP_I32_F32>;
def __floatunssisfvfp : RuntimeLibcallImpl<UINTTOFP_I32_F32>;

//  // RTLIB
//  if (isAAPCS_ABI() &&
//      (isTargetAEABI() || isTargetGNUAEABI() ||
//       isTargetMuslAEABI() || isTargetAndroid())) {

// TODO: Set CallingConv = ARM_AAPCS

// Double-precision floating-point arithmetic helper functions
// RTABI chapter 4.1.2, Table 2
def __aeabi_dadd : RuntimeLibcallImpl<ADD_F64>; // CallingConv::ARM_AAPCS
def __aeabi_ddiv : RuntimeLibcallImpl<DIV_F64>; // CallingConv::ARM_AAPCS
def __aeabi_dmul : RuntimeLibcallImpl<MUL_F64>; // CallingConv::ARM_AAPCS
def __aeabi_dsub : RuntimeLibcallImpl<SUB_F64>; // CallingConv::ARM_AAPCS

// Double-precision floating-point comparison helper functions
// RTABI chapter 4.1.2, Table 3
def __aeabi_dcmpeq__oeq : RuntimeLibcallImpl<OEQ_F64, "__aeabi_dcmpeq">; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_dcmpeq__une : RuntimeLibcallImpl<UNE_F64, "__aeabi_dcmpeq">; // CallingConv::ARM_AAPCS, CmpInst::ICMP_EQ
def __aeabi_dcmplt : RuntimeLibcallImpl<OLT_F64>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_dcmple : RuntimeLibcallImpl<OLE_F64>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_dcmpge : RuntimeLibcallImpl<OGE_F64>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_dcmpgt : RuntimeLibcallImpl<OGT_F64>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_dcmpun : RuntimeLibcallImpl<UO_F64>; // CallingConv::ARM_AAPCS

// Single-precision floating-point arithmetic helper functions
// RTABI chapter 4.1.2, Table 4
def __aeabi_fadd : RuntimeLibcallImpl<ADD_F32>; // CallingConv::ARM_AAPCS
def __aeabi_fdiv : RuntimeLibcallImpl<DIV_F32>; // CallingConv::ARM_AAPCS
def __aeabi_fmul : RuntimeLibcallImpl<MUL_F32>; // CallingConv::ARM_AAPCS
def __aeabi_fsub : RuntimeLibcallImpl<SUB_F32>; // CallingConv::ARM_AAPCS

// Single-precision floating-point comparison helper functions
// RTABI chapter 4.1.2, Table 5
def __aeabi_fcmpeq__oeq : RuntimeLibcallImpl<OEQ_F32, "__aeabi_fcmpeq">; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_fcmpeq__une : RuntimeLibcallImpl<UNE_F32, "__aeabi_fcmpeq">; // CallingConv::ARM_AAPCS, CmpInst::ICMP_EQ
def __aeabi_fcmplt : RuntimeLibcallImpl<OLT_F32>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_fcmple : RuntimeLibcallImpl<OLE_F32>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_fcmpge : RuntimeLibcallImpl<OGE_F32>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_fcmpgt : RuntimeLibcallImpl<OGT_F32>; // CallingConv::ARM_AAPCS, CmpInst::ICMP_NE
def __aeabi_fcmpun : RuntimeLibcallImpl<UO_F32>; // CallingConv::ARM_AAPCS

// Floating-point to integer conversions.
// RTABI chapter 4.1.2, Table 6
def __aeabi_d2iz : RuntimeLibcallImpl<FPTOSINT_F64_I32>; //  CallingConv::ARM_AAPCS
def __aeabi_d2uiz : RuntimeLibcallImpl<FPTOUINT_F64_I32>; // CallingConv::ARM_AAPCS
def __aeabi_d2lz : RuntimeLibcallImpl<FPTOSINT_F64_I64>; //  CallingConv::ARM_AAPCS
def __aeabi_d2ulz : RuntimeLibcallImpl<FPTOUINT_F64_I64>; // CallingConv::ARM_AAPCS
def __aeabi_f2iz : RuntimeLibcallImpl<FPTOSINT_F32_I32>; //  CallingConv::ARM_AAPCS
def __aeabi_f2uiz : RuntimeLibcallImpl<FPTOUINT_F32_I32>; // CallingConv::ARM_AAPCS
def __aeabi_f2lz : RuntimeLibcallImpl<FPTOSINT_F32_I64>; //  CallingConv::ARM_AAPCS
def __aeabi_f2ulz : RuntimeLibcallImpl<FPTOUINT_F32_I64>; // CallingConv::ARM_AAPCS

// Conversions between floating types.
// RTABI chapter 4.1.2, Table 7
def __aeabi_d2f : RuntimeLibcallImpl<FPROUND_F64_F32>; // CallingConv::ARM_AAPCS
def __aeabi_d2h : RuntimeLibcallImpl<FPROUND_F64_F16>; // CallingConv::ARM_AAPCS
def  __aeabi_f2d : RuntimeLibcallImpl<FPEXT_F32_F64>; // CallingConv::ARM_AAPCS

// Integer to floating-point conversions.
// RTABI chapter 4.1.2, Table 8
def __aeabi_i2d : RuntimeLibcallImpl<SINTTOFP_I32_F64>; //  CallingConv::ARM_AAPCS
def __aeabi_ui2d : RuntimeLibcallImpl<UINTTOFP_I32_F64>; // CallingConv::ARM_AAPCS
def __aeabi_l2d : RuntimeLibcallImpl<SINTTOFP_I64_F64>; //  CallingConv::ARM_AAPCS
def __aeabi_ul2d : RuntimeLibcallImpl<UINTTOFP_I64_F64>; // CallingConv::ARM_AAPCS
def __aeabi_i2f : RuntimeLibcallImpl<SINTTOFP_I32_F32>; //  CallingConv::ARM_AAPCS
def __aeabi_ui2f : RuntimeLibcallImpl<UINTTOFP_I32_F32>; // CallingConv::ARM_AAPCS
def __aeabi_l2f : RuntimeLibcallImpl<SINTTOFP_I64_F32>; //  CallingConv::ARM_AAPCS
def __aeabi_ul2f : RuntimeLibcallImpl<UINTTOFP_I64_F32>; // CallingConv::ARM_AAPCS

// Long long helper functions
// RTABI chapter 4.2, Table 9
def __aeabi_lmul : RuntimeLibcallImpl<MUL_I64>; // CallingConv::ARM_AAPCS
def __aeabi_llsl : RuntimeLibcallImpl<SHL_I64>; // CallingConv::ARM_AAPCS
def __aeabi_llsr : RuntimeLibcallImpl<SRL_I64>; // CallingConv::ARM_AAPCS
def __aeabi_lasr : RuntimeLibcallImpl<SRA_I64>; // CallingConv::ARM_AAPCS

// Integer division functions
// RTABI chapter 4.3.1
def __aeabi_idiv : RuntimeLibcallImpl<SDIV_I32>; // CallingConv::ARM_AAPCS
def __aeabi_ldivmod : RuntimeLibcallImpl<SDIVREM_I64>; //  CallingConv::ARM_AAPCS
def __aeabi_uidiv : RuntimeLibcallImpl<UDIV_I32>; // CallingConv::ARM_AAPCS
def __aeabi_uldivmod : RuntimeLibcallImpl<UDIVREM_I64>; // CallingConv::ARM_AAPCS

def __aeabi_idivmod : RuntimeLibcallImpl<SDIVREM_I32>; //  CallingConv::ARM_AAPCS
def __aeabi_uidivmod : RuntimeLibcallImpl<UDIVREM_I32>; //  CallingConv::ARM_AAPCS

// EABI dependent RTLIB

// Memory operations
// RTABI chapter 4.3.4
def __aeabi_memcpy : RuntimeLibcallImpl<MEMCPY>; // CallingConv::ARM_AAPCS
def __aeabi_memcpy4 : RuntimeLibcallImpl<AEABI_MEMCPY4>;
def __aeabi_memcpy8 : RuntimeLibcallImpl<AEABI_MEMCPY8>;

def __aeabi_memmove : RuntimeLibcallImpl<MEMMOVE>; // CallingConv::ARM_AAPCS
def __aeabi_memmove4 : RuntimeLibcallImpl<AEABI_MEMMOVE4>;
def __aeabi_memmove8 : RuntimeLibcallImpl<AEABI_MEMMOVE8>;

def __aeabi_memset : RuntimeLibcallImpl<MEMSET>; // CallingConv::ARM_AAPCS
def __aeabi_memset4 : RuntimeLibcallImpl<AEABI_MEMSET4>;
def __aeabi_memset8 : RuntimeLibcallImpl<AEABI_MEMSET8>;

def __aeabi_memclr : RuntimeLibcallImpl<AEABI_MEMCLR>;
def __aeabi_memclr4 : RuntimeLibcallImpl<AEABI_MEMCLR4>;
def __aeabi_memclr8 : RuntimeLibcallImpl<AEABI_MEMCLR8>;

// isTargetWindows()
defset list<RuntimeLibcallImpl> WindowsFPIntCastLibcalls = {
def __stoi64 : RuntimeLibcallImpl<FPTOSINT_F32_I64>; // CallingConv::ARM_AAPCS_VFP
def __dtoi64 : RuntimeLibcallImpl<FPTOSINT_F64_I64>; // CallingConv::ARM_AAPCS_VFP
def __stou64 : RuntimeLibcallImpl<FPTOUINT_F32_I64>; // CallingConv::ARM_AAPCS_VFP
def __dtou64 : RuntimeLibcallImpl<FPTOUINT_F64_I64>; // CallingConv::ARM_AAPCS_VFP
def __i64tos : RuntimeLibcallImpl<SINTTOFP_I64_F32>; // CallingConv::ARM_AAPCS_VFP
def __i64tod : RuntimeLibcallImpl<SINTTOFP_I64_F64>; // CallingConv::ARM_AAPCS_VFP
def __u64tos : RuntimeLibcallImpl<UINTTOFP_I64_F32>; // CallingConv::ARM_AAPCS_VFP
def __u64tod : RuntimeLibcallImpl<UINTTOFP_I64_F64>; // CallingConv::ARM_AAPCS_VFP
}

def __rt_sdiv : RuntimeLibcallImpl<SDIVREM_I32>; // CallingConv::ARM_AAPCS
def __rt_sdiv64 : RuntimeLibcallImpl<SDIVREM_I64>; // CallingConv::ARM_AAPCS
def __rt_udiv : RuntimeLibcallImpl<UDIVREM_I32>; // CallingConv::ARM_AAPCS
def __rt_udiv64 : RuntimeLibcallImpl<UDIVREM_I64>; // CallingConv::ARM_AAPCS

// Use divmod compiler-rt calls for iOS 5.0 and later.
//  isTargetMachO() &&
// !(isTargetIOS() && isOSVersionLT(5, 0))
def __divmodsi4 : RuntimeLibcallImpl<SDIVREM_I32>;
def __udivmodsi4 : RuntimeLibcallImpl<UDIVREM_I32>;

// FIXME: Change calling convention of FPROUND_F32_F16,
// RTLIB::FPROUND_F64_F16, FPEXT_F16_F32 for !Subtarget->isTargetWatchABI())

// In EABI, these functions have an __aeabi_ prefix, but in GNUEABI they have
// a __gnu_ prefix (which is the default).
// isTargetAEABI()
def __aeabi_f2h : RuntimeLibcallImpl<FPROUND_F32_F16>; // CallingConv::ARM_AAPCS
//def __aeabi_d2h : RuntimeLibcallImpl<FPROUND_F64_F16>; // CallingConv::ARM_AAPCS
def __aeabi_h2f : RuntimeLibcallImpl<FPEXT_F16_F32>; // CallingConv::ARM_AAPCS

// !isTargetMachO()
def __gnu_f2h_ieee : RuntimeLibcallImpl<FPROUND_F32_F16>;
def __gnu_h2f_ieee : RuntimeLibcallImpl<FPEXT_F16_F32>;


def WindowARMDivRemCalls : LibcallImpls<
  (add __rt_sdiv, __rt_sdiv64, __rt_udiv, __rt_udiv64),
  isOSWindows> {
  let CallingConv = ARM_AAPCS;
}

def WindowARMFPIntCasts : LibcallImpls<
  (add WindowsFPIntCastLibcalls),
  isOSWindows> {
  let CallingConv = ARM_AAPCS_VFP;
}


// Register based DivRem for AEABI (RTABI 4.2)
def AEABIDivRemCalls : LibcallImpls<
  (add __aeabi_idivmod, __aeabi_ldivmod,
       __aeabi_uidivmod, __aeabi_uldivmod),
  RuntimeLibcallPredicate<[{TT.isTargetAEABI() || TT.isAndroid() || TT.isTargetGNUAEABI() ||
    TT.isTargetMuslAEABI()}]>> {
  let CallingConv = ARM_AAPCS;
}

def isARMOrThumb : RuntimeLibcallPredicate<"TT.isARM() || TT.isThumb()">;

def ARMSystemLibrary
    : SystemRuntimeLibrary<isARMOrThumb,
      (add WinDefaultLibcallImpls,
           LibcallImpls<(add __powisf2, __powidf2), isNotOSMSVCRT>,
           LibmHasFrexpF32, LibmHasLdexpF32,
           LibmHasFrexpF128, LibmHasLdexpF128,
           WindowARMDivRemCalls,
           WindowARMFPIntCasts,
           SecurityCheckCookieIfWinMSVC,
           AEABIDivRemCalls,
           DarwinSinCosStret, DarwinExp10,
           LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
           DefaultLibmExp10,

           // Use divmod compiler-rt calls for iOS 5.0 and later.
           LibcallImpls<(add __divmodsi4, __udivmodsi4),
                        RuntimeLibcallPredicate<[{TT.isOSBinFormatMachO() &&
                                                  (!TT.isiOS() || !TT.isOSVersionLT(5, 0))}]>>,
           DefaultStackProtector)> {
  let DefaultLibcallCallingConv = LibcallCallingConv<[{
     TT.isOSDarwin() ? CallingConv::C :
        (FloatABI == FloatABI::Hard ? CallingConv::ARM_AAPCS_VFP
                                    : CallingConv::ARM_AAPCS)
  }]>;
}

//===----------------------------------------------------------------------===//
// AVR Runtime Libcalls
//===----------------------------------------------------------------------===//

// Several of the runtime library functions use a special calling conv
def __divmodqi4 : RuntimeLibcallImpl<SDIVREM_I8>;
def __divmodhi4 : RuntimeLibcallImpl<SDIVREM_I16>;
def __udivmodqi4 : RuntimeLibcallImpl<UDIVREM_I8>;
def __udivmodhi4 : RuntimeLibcallImpl<UDIVREM_I16>;

//def __divmodsi4 : RuntimeLibcallImpl<SDIVREM_I32>;
//def __udivmodsi4 : RuntimeLibcallImpl<UDIVREM_I32>;

// Standard sinf/cosf name replaced with "sin" and "cos". Define a
// separate Impl so we can set a different type signature.
def avr_sin : RuntimeLibcallImpl<SIN_F32, "sin">;
def avr_cos : RuntimeLibcallImpl<COS_F32, "cos">;

def isAVR : RuntimeLibcallPredicate<"TT.getArch() == Triple::avr">;

def AVRSystemLibrary
    : SystemRuntimeLibrary<
          isAVR,
          (add (sub DefaultRuntimeLibcallImpls,

               // Division rtlib functions (not supported), use divmod
               // functions instead
               __divqi3, __divhi3, __divsi3, __udivqi3, __udivhi3, __udivsi3,

               // Modulus rtlib functions (not supported), use divmod functions
               // instead
               __modqi3, __modhi3, __modsi3, __umodqi3, __umodhi3, __umodsi3,

               // Standard f64 names are replaced
               sin, cos, sinf, cosf),

              // Several of the runtime library functions use a special calling
              // conv
              LibcallsWithCC<(add __divmodqi4, __divmodhi4, __udivmodqi4,
                                 __udivmodhi4),
                             AVR_BUILTIN>,
              __divmodsi4, __udivmodsi4,
              // Trigonometric rtlib functions
              avr_sin, avr_cos)>;

//===----------------------------------------------------------------------===//
// DXIL Runtime Libcalls
//===----------------------------------------------------------------------===//

def isDXIL : RuntimeLibcallPredicate<"TT.isDXIL()">;

// No calls
def DXILSystemLibrary : SystemRuntimeLibrary<isDXIL, (add)>;

//===----------------------------------------------------------------------===//
// Hexagon Runtime Libcalls
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> HexagonLibcalls = {
def __hexagon_divsi3 : RuntimeLibcallImpl<SDIV_I32>;
def __hexagon_divdi3 : RuntimeLibcallImpl<SDIV_I64>;
def __hexagon_udivsi3 : RuntimeLibcallImpl<UDIV_I32>;
def __hexagon_udivdi3 : RuntimeLibcallImpl<UDIV_I64>;
def __hexagon_modsi3 : RuntimeLibcallImpl<SREM_I32>;
def __hexagon_moddi3 : RuntimeLibcallImpl<SREM_I64>;
def __hexagon_umodsi3 : RuntimeLibcallImpl<UREM_I32>;
def __hexagon_umoddi3 : RuntimeLibcallImpl<UREM_I64>;

def __hexagon_adddf3 : RuntimeLibcallImpl<ADD_F64>;
def __hexagon_fast_adddf3 : RuntimeLibcallImpl<FAST_ADD_F64>;

def __hexagon_subdf3 : RuntimeLibcallImpl<SUB_F64>;
def __hexagon_fast_subdf3 : RuntimeLibcallImpl<FAST_SUB_F64>;

def __hexagon_muldf3 : RuntimeLibcallImpl<MUL_F64>;
def __hexagon_fast_muldf3 : RuntimeLibcallImpl<FAST_MUL_F64>;

def __hexagon_divdf3 : RuntimeLibcallImpl<DIV_F64>;
def __hexagon_fast_divdf3 : RuntimeLibcallImpl<FAST_DIV_F64>;

def __hexagon_divsf3 : RuntimeLibcallImpl<DIV_F32>;
def __hexagon_fast_divsf3 : RuntimeLibcallImpl<FAST_DIV_F32>;

def __hexagon_sqrtf : RuntimeLibcallImpl<SQRT_F32>;
def __hexagon_fast2_sqrtf : RuntimeLibcallImpl<FAST_SQRT_F32>;

// This is the only fast library function for sqrtd.
def __hexagon_fast2_sqrtdf2 : RuntimeLibcallImpl<FAST_SQRT_F64>;

def __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes
    : RuntimeLibcallImpl<HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES>;
}

def isHexagon : RuntimeLibcallPredicate<"TT.getArch() == Triple::hexagon">;

def HexagonSystemLibrary
    : SystemRuntimeLibrary<isHexagon,
    (add (sub DefaultLibcallImpls32,
    __adddf3, __divsf3, __udivsi3, __udivdi3,
    __umoddi3, __divdf3, __muldf3, __divsi3, __subdf3, sqrtf,
    __divdi3, __umodsi3, __moddi3, __modsi3), HexagonLibcalls,
    LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
    exp10f, exp10, exp10l_f128, __stack_chk_fail)>;

//===----------------------------------------------------------------------===//
// Lanai Runtime Libcalls
//===----------------------------------------------------------------------===//

def isLanai : RuntimeLibcallPredicate<"TT.getArch() == Triple::lanai">;

// Use fast calling convention for library functions.
def LanaiSystemLibrary
    : SystemRuntimeLibrary<isLanai, (add DefaultRuntimeLibcallImpls,
                                         __stack_chk_fail)> {
  let DefaultLibcallCallingConv = FASTCC;
}

//===----------------------------------------------------------------------===//
// Mips16 Runtime Libcalls
//===----------------------------------------------------------------------===//

// Hard float libcalls
def __mips16_adddf3 : RuntimeLibcallImpl<ADD_F64>;
def __mips16_addsf3 : RuntimeLibcallImpl<ADD_F32>;
def __mips16_divdf3 : RuntimeLibcallImpl<DIV_F64>;
def __mips16_divsf3 : RuntimeLibcallImpl<DIV_F32>;
def __mips16_eqdf2 : RuntimeLibcallImpl<OEQ_F64>;
def __mips16_eqsf2 : RuntimeLibcallImpl<OEQ_F32>;
def __mips16_extendsfdf2 : RuntimeLibcallImpl<FPEXT_F32_F64>;
def __mips16_fix_truncdfsi : RuntimeLibcallImpl<FPTOSINT_F64_I32>;
def __mips16_fix_truncsfsi : RuntimeLibcallImpl<FPTOSINT_F32_I32>;
def __mips16_floatsidf : RuntimeLibcallImpl<SINTTOFP_I32_F64>;
def __mips16_floatsisf : RuntimeLibcallImpl<SINTTOFP_I32_F32>;
def __mips16_floatunsidf : RuntimeLibcallImpl<UINTTOFP_I32_F64>;
def __mips16_floatunsisf : RuntimeLibcallImpl<UINTTOFP_I32_F32>;
def __mips16_gedf2 : RuntimeLibcallImpl<OGE_F64>;
def __mips16_gesf2 : RuntimeLibcallImpl<OGE_F32>;
def __mips16_gtdf2 : RuntimeLibcallImpl<OGT_F64>;
def __mips16_gtsf2 : RuntimeLibcallImpl<OGT_F32>;
def __mips16_ledf2 : RuntimeLibcallImpl<OLE_F64>;
def __mips16_lesf2 : RuntimeLibcallImpl<OLE_F32>;
def __mips16_ltdf2 : RuntimeLibcallImpl<OLT_F64>;
def __mips16_ltsf2 : RuntimeLibcallImpl<OLT_F32>;
def __mips16_muldf3 : RuntimeLibcallImpl<MUL_F64>;
def __mips16_mulsf3 : RuntimeLibcallImpl<MUL_F32>;
def __mips16_nedf2 : RuntimeLibcallImpl<UNE_F64>;
def __mips16_nesf2 : RuntimeLibcallImpl<UNE_F32>;
def __mips16_ret_dc : RuntimeLibcallImpl<MIPS16_RET_DC>;
def __mips16_ret_df : RuntimeLibcallImpl<MIPS16_RET_DF>;
def __mips16_ret_sc : RuntimeLibcallImpl<MIPS16_RET_SC>;
def __mips16_ret_sf : RuntimeLibcallImpl<MIPS16_RET_SF>;
def __mips16_subdf3 : RuntimeLibcallImpl<SUB_F64>;
def __mips16_subsf3 : RuntimeLibcallImpl<SUB_F32>;
def __mips16_truncdfsf2 : RuntimeLibcallImpl<FPROUND_F64_F32>;
def __mips16_unorddf2 : RuntimeLibcallImpl<UO_F64>;
def __mips16_unordsf2 : RuntimeLibcallImpl<UO_F32>;

//===----------------------------------------------------------------------===//
// MSP430 Runtime Libcalls
//===----------------------------------------------------------------------===//

// EABI Libcalls - EABI Section 6.2

// Floating point conversions - EABI Table 6
def __mspabi_cvtdf : RuntimeLibcallImpl<FPROUND_F64_F32>;
def __mspabi_cvtfd : RuntimeLibcallImpl<FPEXT_F32_F64>;

// The following is NOT implemented in libgcc
//def __mspabi_fixdi : RuntimeLibcallImpl<FPTOSINT_F64_I16>;
def __mspabi_fixdli : RuntimeLibcallImpl<FPTOSINT_F64_I32>;
def __mspabi_fixdlli : RuntimeLibcallImpl<FPTOSINT_F64_I64>;

// The following is NOT implemented in libgcc
//def __mspabi_fixdu : RuntimeLibcallImpl<FPTOUINT_F64_I16>;
def __mspabi_fixdul : RuntimeLibcallImpl<FPTOUINT_F64_I32>;
def __mspabi_fixdull : RuntimeLibcallImpl<FPTOUINT_F64_I64>;

// The following is NOT implemented in libgcc
//def __mspabi_fixfi : RuntimeLibcallImpl<FPTOSINT_F32_I16>;
def __mspabi_fixfli : RuntimeLibcallImpl<FPTOSINT_F32_I32>;
def __mspabi_fixflli : RuntimeLibcallImpl<FPTOSINT_F32_I64>;

// The following is NOT implemented in libgcc
//def __mspabi_fixfu : RuntimeLibcallImpl<FPTOUINT_F32_I16>;
def __mspabi_fixful : RuntimeLibcallImpl<FPTOUINT_F32_I32>;
def __mspabi_fixfull : RuntimeLibcallImpl<FPTOUINT_F32_I64>;

// TODO The following IS implemented in libgcc
//def __mspabi_fltid : RuntimeLibcallImpl<SINTTOFP_I16_F64>;
def __mspabi_fltlid : RuntimeLibcallImpl<SINTTOFP_I32_F64>;

// TODO The following IS implemented in libgcc but is not in the EABI
def __mspabi_fltllid : RuntimeLibcallImpl<SINTTOFP_I64_F64>;

// TODO The following IS implemented in libgcc
//def __mspabi_fltud : RuntimeLibcallImpl<UINTTOFP_I16_F64>;
def __mspabi_fltuld : RuntimeLibcallImpl<UINTTOFP_I32_F64>;

// The following IS implemented in libgcc but is not in the EABI
def __mspabi_fltulld : RuntimeLibcallImpl<UINTTOFP_I64_F64>;

// TODO The following IS implemented in libgcc
//def __mspabi_fltif : RuntimeLibcallImpl<SINTTOFP_I16_F32>;
def __mspabi_fltlif : RuntimeLibcallImpl<SINTTOFP_I32_F32>;

// TODO The following IS implemented in libgcc but is not in the EABI
def __mspabi_fltllif : RuntimeLibcallImpl<SINTTOFP_I64_F32>;

// TODO The following IS implemented in libgcc
//def __mspabi_fltuf : RuntimeLibcallImpl<UINTTOFP_I16_F32>;
def __mspabi_fltulf : RuntimeLibcallImpl<UINTTOFP_I32_F32>;

// The following IS implemented in libgcc but is not in the EABI
def __mspabi_fltullf : RuntimeLibcallImpl<UINTTOFP_I64_F32>;

// Floating point comparisons - EABI Table 7
def __mspabi_cmpd__oeq : RuntimeLibcallImpl<OEQ_F64, "__mspabi_cmpd">;
def __mspabi_cmpd__une : RuntimeLibcallImpl<UNE_F64, "__mspabi_cmpd">;
def __mspabi_cmpd__oge : RuntimeLibcallImpl<OGE_F64, "__mspabi_cmpd">;
def __mspabi_cmpd__olt : RuntimeLibcallImpl<OLT_F64, "__mspabi_cmpd">;
def __mspabi_cmpd__ole : RuntimeLibcallImpl<OLE_F64, "__mspabi_cmpd">;
def __mspabi_cmpd__ogt : RuntimeLibcallImpl<OGT_F64, "__mspabi_cmpd">;
def __mspabi_cmpf__oeq : RuntimeLibcallImpl<OEQ_F32, "__mspabi_cmpf">;
def __mspabi_cmpf__une : RuntimeLibcallImpl<UNE_F32, "__mspabi_cmpf">;
def __mspabi_cmpf__oge : RuntimeLibcallImpl<OGE_F32, "__mspabi_cmpf">;
def __mspabi_cmpf__olt : RuntimeLibcallImpl<OLT_F32, "__mspabi_cmpf">;
def __mspabi_cmpf__ole : RuntimeLibcallImpl<OLE_F32, "__mspabi_cmpf">;
def __mspabi_cmpf__ogt : RuntimeLibcallImpl<OGT_F32, "__mspabi_cmpf">;

// Floating point arithmetic - EABI Table 8
def __mspabi_addd : RuntimeLibcallImpl<ADD_F64>;
def __mspabi_addf : RuntimeLibcallImpl<ADD_F32>;
def __mspabi_divd : RuntimeLibcallImpl<DIV_F64>;
def __mspabi_divf : RuntimeLibcallImpl<DIV_F32>;
def __mspabi_mpyd : RuntimeLibcallImpl<MUL_F64>;
def __mspabi_mpyf : RuntimeLibcallImpl<MUL_F32>;
def __mspabi_subd : RuntimeLibcallImpl<SUB_F64>;
def __mspabi_subf : RuntimeLibcallImpl<SUB_F32>;

// The following are NOT implemented in libgcc
// def __mspabi_negd : RuntimeLibcallImpl<NEG_F64>;
// def __mspabi_negf : RuntimeLibcallImpl<NEG_F32>;

// Universal Integer Operations - EABI Table 9
def __mspabi_divi : RuntimeLibcallImpl<SDIV_I16>;
def __mspabi_divli : RuntimeLibcallImpl<SDIV_I32>;
def __mspabi_divlli : RuntimeLibcallImpl<SDIV_I64>;
def __mspabi_divu : RuntimeLibcallImpl<UDIV_I16>;
def __mspabi_divul : RuntimeLibcallImpl<UDIV_I32>;
def __mspabi_divull : RuntimeLibcallImpl<UDIV_I64>;
def __mspabi_remi : RuntimeLibcallImpl<SREM_I16>;
def __mspabi_remli : RuntimeLibcallImpl<SREM_I32>;
def __mspabi_remlli : RuntimeLibcallImpl<SREM_I64>;
def __mspabi_remu : RuntimeLibcallImpl<UREM_I16>;
def __mspabi_remul : RuntimeLibcallImpl<UREM_I32>;
def __mspabi_remull : RuntimeLibcallImpl<UREM_I64>;

// Bitwise Operations - EABI Table 10
// TODO: __mspabi_[srli/srai/slli] ARE implemented in libgcc
def __mspabi_srll : RuntimeLibcallImpl<SRL_I32>;
def __mspabi_sral : RuntimeLibcallImpl<SRA_I32>;
def __mspabi_slll : RuntimeLibcallImpl<SHL_I32>;
// __mspabi_[srlll/srall/sllll/rlli/rlll] are NOT implemented in libgcc

// hasHWMult16()
// Integer Multiply - EABI Table 9
//def __mspabi_mpyi_hw : RuntimeLibcallImpl<MUL_I16>;
def __mspabi_mpyl_hw : RuntimeLibcallImpl<MUL_I32>;
def __mspabi_mpyll_hw : RuntimeLibcallImpl<MUL_I64>;
// TODO The __mspabi_mpysl*_hw functions ARE implemented in libgcc
// TODO The __mspabi_mpyul*_hw functions ARE implemented in libgcc

// hasHWMult32()
// Integer Multiply - EABI Table 9
def __mspabi_mpyi_hw : RuntimeLibcallImpl<MUL_I16>;
def __mspabi_mpyl_hw32 : RuntimeLibcallImpl<MUL_I32>;
def __mspabi_mpyll_hw32 : RuntimeLibcallImpl<MUL_I64>;
// TODO The __mspabi_mpysl*_hw32 functions ARE implemented in libgcc
// TODO The __mspabi_mpyul*_hw32 functions ARE implemented in libgcc

// hasHWMultF5()
// Integer Multiply - EABI Table 9
def __mspabi_mpyi_f5hw : RuntimeLibcallImpl<MUL_I16>;
def __mspabi_mpyl_f5hw : RuntimeLibcallImpl<MUL_I32>;
def __mspabi_mpyll_f5hw : RuntimeLibcallImpl<MUL_I64>;
// TODO The __mspabi_mpysl*_f5hw functions ARE implemented in libgcc
// TODO The __mspabi_mpyul*_f5hw functions ARE implemented in libgcc

// NoHWMult
// Integer Multiply - EABI Table 9
def __mspabi_mpyi : RuntimeLibcallImpl<MUL_I16>;
def __mspabi_mpyl : RuntimeLibcallImpl<MUL_I32>;
def __mspabi_mpyll : RuntimeLibcallImpl<MUL_I64>;
// The __mspabi_mpysl* functions are NOT implemented in libgcc
// The __mspabi_mpyul* functions are NOT implemented in libgcc

// setLibcallCallingConv(MUL_I64, CallingConv::MSP430_BUILTIN);

def isMSP430 : RuntimeLibcallPredicate<"TT.getArch() == Triple::msp430">;

defvar MSP430DefaultOptOut = [
  __addsf3, __divsf3, __extendsfdf2, __truncdfsf2, __fixsfsi,
  __fixsfdi, __fixunssfsi, __mulsf3, __eqsf2, __gesf2, __gtsf2,
  __divhi3, __divsi3, __ashlsi3, __floatsidf, __floatsisf,
  __ashrsi3, __modhi3, __udivsi3, __fixdfsi, __fixunssfdi,
  __udivhi3, __umodsi3, __nesf2, __lesf2, __floatundisf,
  __fixdfdi, __fixunsdfsi, __modsi3, __floatunsisf,
  __fixunsdfdi, __ltsf2, __floatdisf, __floatdidf,
  __lshrsi3, __subsf3, __umodhi3, __floatunsidf,
  __floatundidf
];

// EABI Libcalls - EABI Section 6.2
def MSP430SystemLibrary
    : SystemRuntimeLibrary<isMSP430,
    (add (sub DefaultRuntimeLibcallImpls, MSP430DefaultOptOut),
      exp10f, exp10, exp10l_f128,

      // Floating point conversions - EABI Table 6
      __mspabi_cvtdf,
      __mspabi_cvtfd,
      // The following is NOT implemented in libgcc
      //__mspabi_fixdi,
      __mspabi_fixdli,
      __mspabi_fixdlli,
      // The following is NOT implemented in libgcc
      // __mspabi_fixdu
      __mspabi_fixdul,
      __mspabi_fixdull,
      // The following is NOT implemented in libgcc
      //__mspabi_fixfi,
      __mspabi_fixfli,
      __mspabi_fixflli,
      // The following is NOT implemented in libgcc
      //__mspabi_fixfu,
      __mspabi_fixful,
      __mspabi_fixfull,
      // TODO The following IS implemented in libgcc
      //__mspabi_fltid
      __mspabi_fltlid,
      // TODO The following IS implemented in libgcc but is not in the EABI
      __mspabi_fltllid,
      // TODO The following IS implemented in libgcc
      //__mspabi_fltud,
      __mspabi_fltuld,
      // The following IS implemented in libgcc but is not in the EABI
      __mspabi_fltulld,
      // TODO The following IS implemented in libgcc
      //__mspabi_fltif,
      __mspabi_fltlif,
      // TODO The following IS implemented in libgcc but is not in the EABI
      __mspabi_fltllif,
      // TODO The following IS implemented in libgcc
      //__mspabi_fltuf,
      __mspabi_fltulf,
      // The following IS implemented in libgcc but is not in the EABI
      __mspabi_fltullf,

      // Floating point comparisons - EABI Table 7
      LibcallsWithCC<(add __mspabi_cmpd__oeq,
                          __mspabi_cmpd__une,
                          __mspabi_cmpd__oge,
                          __mspabi_cmpd__olt,
                          __mspabi_cmpd__ole,
                          __mspabi_cmpd__ogt), MSP430_BUILTIN>,
      __mspabi_cmpf__oeq,
      __mspabi_cmpf__une,
      __mspabi_cmpf__oge,
      __mspabi_cmpf__olt,
      __mspabi_cmpf__ole,
      __mspabi_cmpf__ogt,

      // Floating point arithmetic - EABI Table 8
      LibcallsWithCC<(add __mspabi_addd,
      __mspabi_subd,
      __mspabi_mpyd,
      __mspabi_divd), MSP430_BUILTIN>,

      __mspabi_addf,
      __mspabi_subf,
      __mspabi_mpyf,
      __mspabi_divf,

      // The following are NOT implemented in libgcc
      // __mspabi_negd,
      // __mspabi_negf,

      // Universal Integer Operations - EABI Table 9
      __mspabi_divi,
      __mspabi_divli,
      LibcallsWithCC<(add __mspabi_divlli), MSP430_BUILTIN>,
      __mspabi_divu,
      __mspabi_divul,
      LibcallsWithCC<(add __mspabi_divull), MSP430_BUILTIN>,
      __mspabi_remi,
      __mspabi_remli,
      LibcallsWithCC<(add __mspabi_remlli), MSP430_BUILTIN>,
      __mspabi_remu,
      __mspabi_remul,
      LibcallsWithCC<(add __mspabi_remull), MSP430_BUILTIN>,

      // Bitwise Operations - EABI Table 10
      // TODO: __mspabi_[srli/srai/slli] ARE implemented in libgcc
      __mspabi_srll,
      __mspabi_sral,
      __mspabi_slll,
      // __mspabi_[srlll/srall/sllll/rlli/rlll] are NOT implemented in libgcc

      __stack_chk_fail
  )
>;

//===----------------------------------------------------------------------===//
// NVPTX Runtime Libcalls
//===----------------------------------------------------------------------===//

def isNVPTX : RuntimeLibcallPredicate<"TT.isNVPTX()">;

// No calls.
def NVPTXSystemLibrary : SystemRuntimeLibrary<isNVPTX, (add)>;

//===----------------------------------------------------------------------===//
// PPC Runtime Libcalls
//===----------------------------------------------------------------------===//

// For IEEE quad-precision libcall names, PPC uses "kf" instead of "tf".
defset list<RuntimeLibcallImpl> PPCRuntimeLibcalls = {
  def __addkf3 : RuntimeLibcallImpl<ADD_F128>;
  def __subkf3 : RuntimeLibcallImpl<SUB_F128>;
  def __mulkf3 : RuntimeLibcallImpl<MUL_F128>;
  def __divkf3 : RuntimeLibcallImpl<DIV_F128>;
  def __powikf2 : RuntimeLibcallImpl<POWI_F128>;
  def __extendsfkf2 : RuntimeLibcallImpl<FPEXT_F32_F128>;
  def __extenddfkf2 : RuntimeLibcallImpl<FPEXT_F64_F128>;
  def __trunckfhf2 : RuntimeLibcallImpl<FPROUND_F128_F16>;
  def __trunckfsf2 : RuntimeLibcallImpl<FPROUND_F128_F32>;
  def __trunckfdf2 : RuntimeLibcallImpl<FPROUND_F128_F64>;
  def __fixkfsi : RuntimeLibcallImpl<FPTOSINT_F128_I32>;
  def __fixkfdi : RuntimeLibcallImpl<FPTOSINT_F128_I64>;
  def __fixkfti : RuntimeLibcallImpl<FPTOSINT_F128_I128>;
  def __fixunskfsi : RuntimeLibcallImpl<FPTOUINT_F128_I32>;
  def __fixunskfdi : RuntimeLibcallImpl<FPTOUINT_F128_I64>;
  def __fixunskfti : RuntimeLibcallImpl<FPTOUINT_F128_I128>;
  def __floatsikf : RuntimeLibcallImpl<SINTTOFP_I32_F128>;
  def __floatdikf : RuntimeLibcallImpl<SINTTOFP_I64_F128>;
  def __floattikf : RuntimeLibcallImpl<SINTTOFP_I128_F128>;
  def __floatunsikf : RuntimeLibcallImpl<UINTTOFP_I32_F128>;
  def __floatundikf : RuntimeLibcallImpl<UINTTOFP_I64_F128>;
  def __floatuntikf : RuntimeLibcallImpl<UINTTOFP_I128_F128>;
  def __eqkf2 : RuntimeLibcallImpl<OEQ_F128>;
  def __nekf2 : RuntimeLibcallImpl<UNE_F128>;
  def __gekf2 : RuntimeLibcallImpl<OGE_F128>;
  def __ltkf2 : RuntimeLibcallImpl<OLT_F128>;
  def __lekf2 : RuntimeLibcallImpl<OLE_F128>;
  def __gtkf2 : RuntimeLibcallImpl<OGT_F128>;
  def __unordkf2 : RuntimeLibcallImpl<UO_F128>;
}

defset list<RuntimeLibcallImpl> PPC64AIXCallList = {
  def ___memcmp64 : RuntimeLibcallImpl<MEMCMP>;
  def ___memmove64 : RuntimeLibcallImpl<MEMCPY>;
  def ___memset64 : RuntimeLibcallImpl<MEMSET>;
  def ___bzero64 : RuntimeLibcallImpl<BZERO>;
}

defset list<RuntimeLibcallImpl> PPC32AIXCallList = {
  def ___memcmp : RuntimeLibcallImpl<MEMCMP>;
  def ___memmove : RuntimeLibcallImpl<MEMMOVE>;
  def ___memset : RuntimeLibcallImpl<MEMSET>;
  def ___bzero : RuntimeLibcallImpl<BZERO>;
}

defvar PPCOverrides = !foreach(entry, PPCRuntimeLibcalls, entry.Provides);

def isPPC : RuntimeLibcallPredicate<"TT.isPPC()">;
def isPPC32 : RuntimeLibcallPredicate<"TT.isPPC32()">;
def isPPC64 : RuntimeLibcallPredicate<"TT.isPPC64()">;
def isAIX : RuntimeLibcallPredicate<"TT.isOSAIX()">;
def isNotAIX : RuntimeLibcallPredicate<"!TT.isOSAIX()">;
def isPPC32_AIX : RuntimeLibcallPredicate<"(TT.isPPC32() && TT.isOSAIX())">;
def isPPC64_AIX : RuntimeLibcallPredicate<"(TT.isPPC64() && TT.isOSAIX())">;

def AIX32Calls : LibcallImpls<(add PPC32AIXCallList), isPPC32_AIX>;
def AIX64Calls : LibcallImpls<(add PPC64AIXCallList), isPPC64_AIX>;

// FIXME: Current emission behavior with multiple implementations is
// janky. We need to filter out the conflicting cases with different
// f128 names, and then add the overrides. We should switch to
// explicitly adding subsets of the default calls.
def PPCSystemLibrary
    : SystemRuntimeLibrary<isPPC,
      (add PPCRuntimeLibcalls,
           (sub DefaultRuntimeLibcallImpls, memcpy,
                DefaultRuntimeLibcallImpls_f128),
           __extendkftf2, __trunctfkf2,
           DefaultRuntimeLibcallImpls_ppcf128,
           exp10f, exp10, exp10l_ppcf128,
           LibmF128Libcalls, AIX32Calls, AIX64Calls,
           LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
           LibmHasSinCosPPCF128,
           AvailableIf<memcpy, isNotAIX>,
           LibcallImpls<(add Int128RTLibcalls), isPPC64>,
           DefaultStackProtector)>;

//===----------------------------------------------------------------------===//
// RISCV Runtime Libcalls
//===----------------------------------------------------------------------===//

def isRISCV : RuntimeLibcallPredicate<"TT.isRISCV()">;
def isRISCV64 : RuntimeLibcallPredicate<"TT.isRISCV64()">;

def RISCVSystemLibrary
    : SystemRuntimeLibrary<isRISCV,
      (add DefaultRuntimeLibcallImpls,
           exp10f, exp10, exp10l_f128,
           __riscv_flush_icache,
           LibcallImpls<(add Int128RTLibcalls), isRISCV64>,
           DefaultStackProtector)>;

//===----------------------------------------------------------------------===//
// SPARC Runtime Libcalls
//===----------------------------------------------------------------------===//

// Subtarget->useSoftMulDiv()
def sparc_umul : RuntimeLibcallImpl<MUL_I32, ".umul">;
def sparc_div : RuntimeLibcallImpl<SDIV_I32, ".div">;
def sparc_udiv : RuntimeLibcallImpl<UDIV_I32, ".udiv">;
def sparc_rem : RuntimeLibcallImpl<SREM_I32, ".rem">;
def sparc_urem : RuntimeLibcallImpl<UREM_I32, ".urem">;

// SPARC64 && !Subtarget->useSoftFloat()
def _Qp_add : RuntimeLibcallImpl<ADD_F128>;
def _Qp_sub : RuntimeLibcallImpl<SUB_F128>;
def _Qp_mul : RuntimeLibcallImpl<MUL_F128>;
def _Qp_div : RuntimeLibcallImpl<DIV_F128>;
def _Qp_sqrt : RuntimeLibcallImpl<SQRT_F128>;
def _Qp_qtoi : RuntimeLibcallImpl<FPTOSINT_F128_I32>;
def _Qp_qtoui : RuntimeLibcallImpl<FPTOUINT_F128_I32>;
def _Qp_itoq : RuntimeLibcallImpl<SINTTOFP_I32_F128>;
def _Qp_uitoq : RuntimeLibcallImpl<UINTTOFP_I32_F128>;
def _Qp_qtox : RuntimeLibcallImpl<FPTOSINT_F128_I64>;
def _Qp_qtoux : RuntimeLibcallImpl<FPTOUINT_F128_I64>;
def _Qp_xtoq : RuntimeLibcallImpl<SINTTOFP_I64_F128>;
def _Qp_uxtoq : RuntimeLibcallImpl<UINTTOFP_I64_F128>;
def _Qp_stoq : RuntimeLibcallImpl<FPEXT_F32_F128>;
def _Qp_dtoq : RuntimeLibcallImpl<FPEXT_F64_F128>;
def _Qp_qtos : RuntimeLibcallImpl<FPROUND_F128_F32>;
def _Qp_qtod : RuntimeLibcallImpl<FPROUND_F128_F64>;

// SPARC32 && !Subtarget->useSoftFloat()
def _Q_add : RuntimeLibcallImpl<ADD_F128>;
def _Q_sub : RuntimeLibcallImpl<SUB_F128>;
def _Q_mul : RuntimeLibcallImpl<MUL_F128>;
def _Q_div : RuntimeLibcallImpl<DIV_F128>;
def _Q_sqrt : RuntimeLibcallImpl<SQRT_F128>;
def _Q_qtoi : RuntimeLibcallImpl<FPTOSINT_F128_I32>;
def _Q_qtou : RuntimeLibcallImpl<FPTOUINT_F128_I32>;
def _Q_itoq : RuntimeLibcallImpl<SINTTOFP_I32_F128>;
def _Q_utoq : RuntimeLibcallImpl<UINTTOFP_I32_F128>;
def _Q_stoq : RuntimeLibcallImpl<FPEXT_F32_F128>;
def _Q_dtoq : RuntimeLibcallImpl<FPEXT_F64_F128>;
def _Q_qtos : RuntimeLibcallImpl<FPROUND_F128_F32>;
def _Q_qtod : RuntimeLibcallImpl<FPROUND_F128_F64>;

// SPARC32 && (Subtarget->hasHardQuad() || !Subtarget->useSoftFloat())
def _Q_qtoll : RuntimeLibcallImpl<FPTOSINT_F128_I64>;
def _Q_qtoull : RuntimeLibcallImpl<FPTOUINT_F128_I64>;
def _Q_lltoq : RuntimeLibcallImpl<SINTTOFP_I64_F128>;
def _Q_ulltoq : RuntimeLibcallImpl<UINTTOFP_I64_F128>;

def isSPARC : RuntimeLibcallPredicate<"TT.isSPARC()">;
def isSPARC32 : RuntimeLibcallPredicate<"TT.isSPARC32()">;
def isSPARC64 : RuntimeLibcallPredicate<"TT.isSPARC64()">;

defvar SPARC64_MulDivCalls = [
  __mulsi3, __divsi3, __modsi3, __udivsi3, __umodsi3
];

def SPARCSystemLibrary
    : SystemRuntimeLibrary<isSPARC,
      (add (sub DefaultLibcallImpls32, SPARC64_MulDivCalls),
            sparc_umul, sparc_div, sparc_udiv, sparc_rem, sparc_urem,
       LibcallImpls<(add _Q_qtoll, _Q_qtoull, _Q_lltoq, _Q_ulltoq), isSPARC32>,
       LibcallImpls<(add SPARC64_MulDivCalls, Int128RTLibcalls), isSPARC64>,
       LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
       DefaultStackProtector)
>;

//===----------------------------------------------------------------------===//
// SPIRV Runtime Libcalls
//===----------------------------------------------------------------------===//

def isSPIRV : RuntimeLibcallPredicate<"TT.isSPIRV()">;

// No calls FIXME: Add memcpy/memset is a hack to skip
// PreISelIntrinsicLowering in favor of SPIRVPrepareFunctions;
// probably should remove that and just use the default one.
def SPIRVSystemLibrary : SystemRuntimeLibrary<isSPIRV, (add memset, memcpy)>;

//===----------------------------------------------------------------------===//
// Windows Runtime Libcalls
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> WindowsDivRemMulLibcalls = {
def _alldiv : RuntimeLibcallImpl<SDIV_I64>;
def _aulldiv : RuntimeLibcallImpl<UDIV_I64>;
def _allrem : RuntimeLibcallImpl<SREM_I64>;
def _aullrem : RuntimeLibcallImpl<UREM_I64>;
def _allmul : RuntimeLibcallImpl<MUL_I64>;
}

//===----------------------------------------------------------------------===//
// X86 Runtime Libcalls
//===----------------------------------------------------------------------===//

def isX86_32 : RuntimeLibcallPredicate<"TT.getArch() == Triple::x86">;
def isX86_64 : RuntimeLibcallPredicate<"TT.getArch() == Triple::x86_64">;
def isX86 : RuntimeLibcallPredicate<"TT.isX86()">;

// Some darwins have an optimized __bzero/bzero function.
def darwinHas__bzero : RuntimeLibcallPredicate<"TT.isMacOSX() && !TT.isMacOSXVersionLT(10, 6)">;

// FIXME: This is has ldexpl/frexpl plus use f128 for long double.
def hasExpFrexplLdexplF128
  : RuntimeLibcallPredicate<[{(!TT.isOSWindows() || TT.isOSCygMing()) && !TT.isGNUEnvironment()}]>;

// Use the f128 variants of math functions on x86
defvar X86_F128_Libcalls = LibcallImpls<(add LibmF128Libcalls, LibmF128FiniteLibcalls), isGNUEnvironment>;

defvar SinCosF32F64Libcalls = LibcallImpls<(add sincosf, sincos), hasSinCos_f32_f64>;

defvar X86CommonLibcalls =
  (add WinDefaultLibcallImpls,
       DarwinSinCosStret, DarwinExp10,
       X86_F128_Libcalls,
       LibmHasSinCosF80, // FIXME: Depends on long double
       SinCosF32F64Libcalls,
       LibcallImpls<(add __bzero), darwinHas__bzero>,
       LibmHasFrexpF32, LibmHasLdexpF32,
       LibmHasFrexpF80, LibmHasLdexpF80,
       LibcallImpls<(add frexp_f128, ldexp_f128, exp10l_f128), hasExpFrexplLdexplF128>,
       DefaultRuntimeLibcallImpls_f80,
       LibmHasExp10F32, LibmHasExp10F64, LibmHasExp10F80,
       LibcallImpls<(add MostPowI), isNotOSMSVCRT>,
       SecurityCheckCookieIfWinMSVC,
       // FIXME: MSVCRT doesn't have powi. The f128 case is added as a
       // hack for one test relying on it.
       __powitf2_f128,
       DefaultStackProtector
);

defvar Windows32DivRemMulCalls =
  LibcallsWithCC<(add WindowsDivRemMulLibcalls), X86_STDCALL,
  RuntimeLibcallPredicate<"TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment()">>;

def X86_32SystemLibrary
    : SystemRuntimeLibrary<isX86_32,
      (add X86CommonLibcalls,
           Windows32DivRemMulCalls)>;

def X86_64SystemLibrary
    : SystemRuntimeLibrary<isX86_64,
      (add X86CommonLibcalls, Int128RTLibcalls)>;

//===----------------------------------------------------------------------===//
// XCore Runtime Libcalls
//===----------------------------------------------------------------------===//

def __memcpy_4 : RuntimeLibcallImpl<MEMCPY_ALIGN_4>;

def isXCore : RuntimeLibcallPredicate<"TT.getArch() == Triple::xcore">;
def XCoreSystemLibrary
    : SystemRuntimeLibrary<isXCore,
    (add DefaultRuntimeLibcallImpls,
         exp10f, exp10, exp10l_f128,
         __memcpy_4,
         LibcallImpls<(add LibmF128Libcalls, LibmF128FiniteLibcalls), isGNUEnvironment>
)>;

//===----------------------------------------------------------------------===//
// ZOS Runtime Libcalls
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> ZOSRuntimeLibcalls = {
  def zos___TRNC_B : RuntimeLibcallImpl<TRUNC_F64, "@@TRNC@B">;
  def zos___FTRC_B : RuntimeLibcallImpl<TRUNC_F32, "@@FTRC@B">;
  def zos___LTRC_B : RuntimeLibcallImpl<TRUNC_F128, "@@LTRC@B">;
  def zos___WSQT_B : RuntimeLibcallImpl<SQRT_F64, "@@WSQT@B">;
  def zos___FSQT_B : RuntimeLibcallImpl<SQRT_F32, "@@FSQT@B">;
  def zos___LSQT_B : RuntimeLibcallImpl<SQRT_F128, "@@LSQT@B">;
  def zos___SSIN_B : RuntimeLibcallImpl<SIN_F64, "@@SSIN@B">;
  def zos___FSIN_B : RuntimeLibcallImpl<SIN_F32, "@@FSIN@B">;
  def zos___LSIN_B : RuntimeLibcallImpl<SIN_F128, "@@LSIN@B">;
  def zos___ROUN_B : RuntimeLibcallImpl<ROUND_F64, "@@ROUN@B">;
  def zos___ROUNFB : RuntimeLibcallImpl<ROUND_F32, "@@ROUNFB">;
  def zos___ROUNLB : RuntimeLibcallImpl<ROUND_F128, "@@ROUNLB">;
  def zos___SRNT_B : RuntimeLibcallImpl<RINT_F64, "@@SRNT@B">;
  def zos___RINTFB : RuntimeLibcallImpl<RINT_F32, "@@RINTFB">;
  def zos___RINTLB : RuntimeLibcallImpl<RINT_F128, "@@RINTLB">;
  def zos___WFMD_B : RuntimeLibcallImpl<REM_F64, "@@WFMD@B">;
  def zos___FFMD_B : RuntimeLibcallImpl<REM_F32, "@@FFMD@B">;
  def zos___LFMD_B : RuntimeLibcallImpl<REM_F128, "@@LFMD@B">;
  def zos___WPOW_B : RuntimeLibcallImpl<POW_F64, "@@WPOW@B">;
  def zos___FPOW_B : RuntimeLibcallImpl<POW_F32, "@@FPOW@B">;
  def zos___LPOW_B : RuntimeLibcallImpl<POW_F128, "@@LPOW@B">;
  def zos___NBYI_B : RuntimeLibcallImpl<NEARBYINT_F64, "@@NBYI@B">;
  def zos___NBYIFB : RuntimeLibcallImpl<NEARBYINT_F32, "@@NBYIFB">;
  def zos___NBYILB : RuntimeLibcallImpl<NEARBYINT_F128, "@@NBYILB">;
  def zos___ROND_B : RuntimeLibcallImpl<LROUND_F64, "@@ROND@B">;
  def zos___FRND_B : RuntimeLibcallImpl<LROUND_F32, "@@FRND@B">;
  def zos___LRND_B : RuntimeLibcallImpl<LROUND_F128, "@@LRND@B">;
  def zos___LRNT_B : RuntimeLibcallImpl<LRINT_F64, "@@LRNT@B">;
  def zos___LRNTFB : RuntimeLibcallImpl<LRINT_F32, "@@LRNTFB">;
  def zos___LRNTLB : RuntimeLibcallImpl<LRINT_F128, "@@LRNTLB">;
  def zos___WLOG_B : RuntimeLibcallImpl<LOG_F64, "@@WLOG@B">;
  def zos___FLOG_B : RuntimeLibcallImpl<LOG_F32, "@@FLOG@B">;
  def zos___LLOG_B : RuntimeLibcallImpl<LOG_F128, "@@LLOG@B">;
  def zos___LOG2_B : RuntimeLibcallImpl<LOG2_F64, "@@LOG2@B">;
  def zos___FLG2_B : RuntimeLibcallImpl<LOG2_F32, "@@FLG2@B">;
  def zos___LLG2_B : RuntimeLibcallImpl<LOG2_F128, "@@LLG2@B">;
  def zos___WLG1_B : RuntimeLibcallImpl<LOG10_F64, "@@WLG1@B">;
  def zos___FLG1_B : RuntimeLibcallImpl<LOG10_F32, "@@FLG1@B">;
  def zos___LLG1_B : RuntimeLibcallImpl<LOG10_F128, "@@LLG1@B">;
  def zos___LLRD_B : RuntimeLibcallImpl<LLROUND_F64, "@@LLRD@B">;
  def zos___LLRDFB : RuntimeLibcallImpl<LLROUND_F32, "@@LLRDFB">;
  def zos___LLRDLB : RuntimeLibcallImpl<LLROUND_F128, "@@LLRDLB">;
  def zos___LLRT_B : RuntimeLibcallImpl<LLRINT_F64, "@@LLRT@B">;
  def zos___LLRTFB : RuntimeLibcallImpl<LLRINT_F32, "@@LLRTFB">;
  def zos___LLRTLB : RuntimeLibcallImpl<LLRINT_F128, "@@LLRTLB">;
  def zos___SLXP_B : RuntimeLibcallImpl<LDEXP_F64, "@@SLXP@B">;
  def zos___FLXP_B : RuntimeLibcallImpl<LDEXP_F32, "@@FLXP@B">;
  def zos___LLXP_B : RuntimeLibcallImpl<LDEXP_F128, "@@LLXP@B">;
  def zos___SFXP_B : RuntimeLibcallImpl<FREXP_F64, "@@SFXP@B">;
  def zos___FFXP_B : RuntimeLibcallImpl<FREXP_F32, "@@FFXP@B">;
  def zos___LFXP_B : RuntimeLibcallImpl<FREXP_F128, "@@LFXP@B">;
  def zos___FMIN_B : RuntimeLibcallImpl<FMIN_F64, "@@FMIN@B">;
  def zos___FMINFB : RuntimeLibcallImpl<FMIN_F32, "@@FMINFB">;
  def zos___FMINLB : RuntimeLibcallImpl<FMIN_F128, "@@FMINLB">;
  def zos___FMA_B : RuntimeLibcallImpl<FMA_F64, "@@FMA@B">;
  def zos___FMAFB : RuntimeLibcallImpl<FMA_F32, "@@FMAFB">;
  def zos___FMALB : RuntimeLibcallImpl<FMA_F128, "@@FMALB">;
  def zos___FMAX_B : RuntimeLibcallImpl<FMAX_F64, "@@FMAX@B">;
  def zos___FMAXFB : RuntimeLibcallImpl<FMAX_F32, "@@FMAXFB">;
  def zos___FMAXLB : RuntimeLibcallImpl<FMAX_F128, "@@FMAXLB">;
  def zos___SFLR_B : RuntimeLibcallImpl<FLOOR_F64, "@@SFLR@B">;
  def zos___FFLR_B : RuntimeLibcallImpl<FLOOR_F32, "@@FFLR@B">;
  def zos___LFLR_B : RuntimeLibcallImpl<FLOOR_F128, "@@LFLR@B">;
  def zos___WEXP_B : RuntimeLibcallImpl<EXP_F64, "@@WEXP@B">;
  def zos___FEXP_B : RuntimeLibcallImpl<EXP_F32, "@@FEXP@B">;
  def zos___LEXP_B : RuntimeLibcallImpl<EXP_F128, "@@LEXP@B">;
  def zos___EXP2_B : RuntimeLibcallImpl<EXP2_F64, "@@EXP2@B">;
  def zos___FXP2_B : RuntimeLibcallImpl<EXP2_F32, "@@FXP2@B">;
  def zos___LXP2_B : RuntimeLibcallImpl<EXP2_F128, "@@LXP2@B">;
  def zos___SCOS_B : RuntimeLibcallImpl<COS_F64, "@@SCOS@B">;
  def zos___FCOS_B : RuntimeLibcallImpl<COS_F32, "@@FCOS@B">;
  def zos___LCOS_B : RuntimeLibcallImpl<COS_F128, "@@LCOS@B">;
  def zos___DCPY_B : RuntimeLibcallImpl<COPYSIGN_F64, "@@DCPY@B">;
  def zos___FCPY_B : RuntimeLibcallImpl<COPYSIGN_F32, "@@FCPY@B">;
  def zos___LCPY_B : RuntimeLibcallImpl<COPYSIGN_F128, "@@LCPY@B">;
  def zos___SCEL_B : RuntimeLibcallImpl<CEIL_F64, "@@SCEL@B">;
  def zos___FCEL_B : RuntimeLibcallImpl<CEIL_F32, "@@FCEL@B">;
  def zos___LCEL_B : RuntimeLibcallImpl<CEIL_F128, "@@LCEL@B">;
  def zos___SCRT_B : RuntimeLibcallImpl<CBRT_F64, "@@SCRT@B">;
  def zos___FCBT_B : RuntimeLibcallImpl<CBRT_F32, "@@FCBT@B">;
  def zos___LCBT_B : RuntimeLibcallImpl<CBRT_F128, "@@LCBT@B">;
}

def isSystemZZOS : RuntimeLibcallPredicate<"(TT.isSystemZ() && TT.isOSzOS())">;
def isZOS : RuntimeLibcallPredicate<"TT.isOSzOS()">;
def SystemZZOSSystemLibrary
    : SystemRuntimeLibrary<
          isSystemZZOS, (add DefaultLibcallImpls64,
                            LibcallImpls<(add ZOSRuntimeLibcalls), isZOS>)>;

//===----------------------------------------------------------------------===//
// WebAssembly Runtime Libcalls
//===----------------------------------------------------------------------===//

// Define the emscripten name for return address helper.
// TODO: when implementing other Wasm backends, make this generic or only do
// this on emscripten depending on what they end up doing.
def emscripten_return_address : RuntimeLibcallImpl<RETURN_ADDRESS>;

def isWasm : RuntimeLibcallPredicate<"TT.isWasm()">;

// Define the emscripten name for return address helper.
// TODO: when implementing other Wasm backends, make this generic or only do
// this on emscripten depending on what they end up doing.
def WasmSystemLibrary
    : SystemRuntimeLibrary<isWasm,
      (add DefaultRuntimeLibcallImpls, Int128RTLibcalls,
           CompilerRTOnlyInt64Libcalls, CompilerRTOnlyInt128Libcalls,
           exp10f, exp10,
           emscripten_return_address,
           __stack_chk_fail)>;

//===----------------------------------------------------------------------===//
// Legacy Default Runtime Libcalls
//===----------------------------------------------------------------------===//

// TODO: Should make every target explicit.
def isDefaultLibcallArch : RuntimeLibcallPredicate<[{
  TT.isMIPS() || TT.isLoongArch() || TT.isVE() || TT.isBPF() ||
  TT.getArch() == Triple::csky || TT.getArch() == Triple::arc ||
  TT.getArch() == Triple::m68k ||  TT.getArch() == Triple::xtensa ||
  (TT.isSystemZ() && !TT.isOSzOS())
}]>;


def isArch64Bit : RuntimeLibcallPredicate<[{TT.isArch64Bit()}]>;
def LegacyDefaultSystemLibrary
    : SystemRuntimeLibrary<isDefaultLibcallArch,
      (add DefaultRuntimeLibcallImpls,
         LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
         exp10f, exp10, exp10l_f128,
         __powisf2, __powidf2, __powitf2_f128,
         LibcallImpls<(add Int128RTLibcalls), isArch64Bit>,
         DefaultStackProtector
)>;
