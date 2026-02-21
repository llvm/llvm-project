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
def isNotOSLinux : RuntimeLibcallPredicate<[{!TT.isOSLinux()}]>;
def isNotOSMSVCRT : RuntimeLibcallPredicate<"!TT.isOSMSVCRT()">;
def isPS : RuntimeLibcallPredicate<"TT.isPS()">;
def isMacOSX : RuntimeLibcallPredicate<[{TT.isMacOSX()}]>;
def isNotOSWindowsOrIsCygwinMinGW
  : RuntimeLibcallPredicate<"!TT.isOSWindows() || TT.isOSCygMing()">;

def isWindowsMSVCEnvironment : RuntimeLibcallPredicate<
  [{TT.isWindowsMSVCEnvironment()}]>;

def isNotOSLinuxAndNotOSOpenBSD : RuntimeLibcallPredicate<
  [{!TT.isOSLinux() && !TT.isOSOpenBSD()}]>;

def isNotOSAIXAndNotOSOpenBSD : RuntimeLibcallPredicate<
  [{!TT.isOSAIX() && !TT.isOSOpenBSD()}]>;

def isNotPS : RuntimeLibcallPredicate<
  [{!TT.isPS()}]>;

// OpenBSD uses __guard_local. AIX uses __ssp_canary_word, MSVC/Windows
// Itanium uses __security_cookie
def hasStackChkFail : RuntimeLibcallPredicate<
  [{ !TT.isOSOpenBSD() && !TT.isWindowsMSVCEnvironment() &&
     !TT.isWindowsItaniumEnvironment()}]>;

def isWindowsMSVCOrItaniumEnvironment : RuntimeLibcallPredicate<
  [{TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment()}]>;

def isAndroid : RuntimeLibcallPredicate<"TT.isAndroid()">;

def isGNUEnvironment : RuntimeLibcallPredicate<"TT.isGNUEnvironment()">;
def darwinHasSinCosStret : RuntimeLibcallPredicate<"darwinHasSinCosStret(TT)">;
def darwinHasExp10 : RuntimeLibcallPredicate<"darwinHasExp10(TT)">;
def darwinHasMemsetPattern : RuntimeLibcallPredicate<[{darwinHasMemsetPattern(TT)}]>;

def hasExp10 : RuntimeLibcallPredicate<[{!TT.isOSDarwin()}]>;

def hasSinCos : RuntimeLibcallPredicate<"hasSinCos(TT)">;

// FIXME: Way to combine predicates
def hasSinCos_f32_f64 : RuntimeLibcallPredicate<"hasSinCos_f32_f64(TT)">;

def ExceptionModelIsNotNone : RuntimeLibcallPredicate<
  [{ExceptionModel != ExceptionHandling::None}]
>;

def ExceptionModelHasUnwindResume : RuntimeLibcallPredicate<
  [{ExceptionModel != ExceptionHandling::None &&
    ExceptionModel != ExceptionHandling::SjLj}]
>;

def ExceptionModelIsSjLj : RuntimeLibcallPredicate<
  [{ExceptionModel == ExceptionHandling::SjLj}]
>;

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
  def REMQUO_#FPTy : RuntimeLibcall;
  def FDIM_#FPTy : RuntimeLibcall;

  def CABS_#FPTy : RuntimeLibcall;
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
  def NAN_#FPTy : RuntimeLibcall;
  def NEXTTOWARD_#FPTy : RuntimeLibcall;
  def REMAINDER_#FPTy : RuntimeLibcall;
  def SCALBLN_#FPTy : RuntimeLibcall;
  def SCALBN_#FPTy : RuntimeLibcall;
  def TGAMMA_#FPTy : RuntimeLibcall;
}

defvar F32VectorSuffixes = ["V2F32", "V4F32", "V8F32", "V16F32", "NXV4F32"];
defvar F64VectorSuffixes = ["V2F64", "V4F64", "V8F64", "NXV2F64"];

foreach S = !listconcat(F32VectorSuffixes, F64VectorSuffixes) in {
  def ACOS_#S : RuntimeLibcall;
  def ACOSH_#S : RuntimeLibcall;
  def ASIN_#S : RuntimeLibcall;
  def ASINH_#S : RuntimeLibcall;
  def ATAN_#S : RuntimeLibcall;
  def ATAN2_#S : RuntimeLibcall;
  def ATANH_#S : RuntimeLibcall;
  def CBRT_#S : RuntimeLibcall;
  def CEIL_#S : RuntimeLibcall;
  def COPYSIGN_#S : RuntimeLibcall;
  def COS_#S : RuntimeLibcall;
  def COSH_#S : RuntimeLibcall;
  def COSPI_#S : RuntimeLibcall;
  def ERFC_#S : RuntimeLibcall;
  def ERF_#S : RuntimeLibcall;
  def EXP_#S : RuntimeLibcall;
  def EXP_FINITE_#S : RuntimeLibcall;
  def EXP10_#S : RuntimeLibcall;
  def EXP2_#S : RuntimeLibcall;
  def EXPM1_#S : RuntimeLibcall;
  def FABS_#S : RuntimeLibcall;
  def FDIM_#S : RuntimeLibcall;
  def FLOOR_#S : RuntimeLibcall;
  def FMA_#S : RuntimeLibcall;
  def FMAX_#S : RuntimeLibcall;
  def FMIN_#S : RuntimeLibcall;
  def REM_#S : RuntimeLibcall; // "fmod"
  def HYPOT_#S : RuntimeLibcall;
  def ILOGB_#S : RuntimeLibcall;
  def LDEXP_#S : RuntimeLibcall;
  def LGAMMA_#S : RuntimeLibcall;
  def LOG_#S : RuntimeLibcall;
  def LOG10_#S : RuntimeLibcall;
  def LOG1P_#S : RuntimeLibcall;
  def LOG2_#S : RuntimeLibcall;
  def LOGB_#S : RuntimeLibcall;
  def MODF_#S : RuntimeLibcall;
  def NEXTAFTER_#S : RuntimeLibcall;
  def POW_#S : RuntimeLibcall;
  def SINCOS_#S : RuntimeLibcall;
  def SINCOSPI_#S : RuntimeLibcall;
  def SIN_#S : RuntimeLibcall;
  def SINH_#S : RuntimeLibcall;
  def SINPI_#S : RuntimeLibcall;
  def SQRT_#S : RuntimeLibcall;
  def TAN_#S : RuntimeLibcall;
  def TANH_#S : RuntimeLibcall;
  def TGAMMA_#S : RuntimeLibcall;
}

foreach S = F64VectorSuffixes in {
  def LOG_FINITE_#S : RuntimeLibcall;
  def POW_FINITE_#S : RuntimeLibcall;
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
def MEMCPY_CHK : RuntimeLibcall;
def MEMMOVE : RuntimeLibcall;
def MEMMOVE_CHK : RuntimeLibcall;
def MEMSET : RuntimeLibcall;
def MEMSET_CHK : RuntimeLibcall;
def MALLOC : RuntimeLibcall;
def CALLOC : RuntimeLibcall;
def FREE : RuntimeLibcall;
def BZERO : RuntimeLibcall;
def STRLEN : RuntimeLibcall;

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

// Really a global variable, not a function
def SAFESTACK_UNSAFE_STACK_PTR : RuntimeLibcall;

def STACK_PROBE : RuntimeLibcall;
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

multiclass LibmLongDoubleLibCall<string libcall_basename = !toupper(!substr(NAME, 0, !sub(!size(NAME), 1))),
                                 string rtname = NAME> {


  def NAME#"_f128"
      : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_F128"),
                           rtname>;
  def NAME#"_ppcf128"
      : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_PPCF128"),
                           rtname>;
  def NAME#"_f80"
      : RuntimeLibcallImpl<!cast<RuntimeLibcall>(libcall_basename#"_F80"),
                           rtname>;
}

// AArch64 calls
def SC_MEMCPY : RuntimeLibcall;
def SC_MEMMOVE : RuntimeLibcall;
def SC_MEMSET : RuntimeLibcall;
def SC_MEMCHR: RuntimeLibcall;

// AArch64 SME ABI calls
def SMEABI_SME_STATE : RuntimeLibcall;
def SMEABI_TPIDR2_SAVE : RuntimeLibcall;
def SMEABI_ZA_DISABLE : RuntimeLibcall;
def SMEABI_TPIDR2_RESTORE : RuntimeLibcall;
def SMEABI_GET_CURRENT_VG : RuntimeLibcall;
def SMEABI_SME_STATE_SIZE : RuntimeLibcall;
def SMEABI_SME_SAVE : RuntimeLibcall;
def SMEABI_SME_RESTORE : RuntimeLibcall;

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
def AEABI_UREAD4 : RuntimeLibcall;
def AEABI_UREAD8 : RuntimeLibcall;
def AEABI_UWRITE4 : RuntimeLibcall;
def AEABI_UWRITE8 : RuntimeLibcall;

// Hexagon calls
def HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES : RuntimeLibcall;
def HEXAGON_VOLATILE_MEMCPY : RuntimeLibcall;

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

def ABORT : RuntimeLibcall;
def ABS : RuntimeLibcall;
def ACCESS : RuntimeLibcall;
def ALIGNED_ALLOC : RuntimeLibcall;
def ATEXIT : RuntimeLibcall;
def ATOF : RuntimeLibcall;
def ATOI : RuntimeLibcall;
def ATOL : RuntimeLibcall;
def ATOLL : RuntimeLibcall;
def BCMP : RuntimeLibcall;
def BCOPY : RuntimeLibcall;
def CHMOD : RuntimeLibcall;
def CHOWN : RuntimeLibcall;
def CLEARERR : RuntimeLibcall;
def CLOSEDIR : RuntimeLibcall;
def CTERMID : RuntimeLibcall;
def CXA_ATEXIT : RuntimeLibcall;
def CXA_GUARD_ABORT : RuntimeLibcall;
def CXA_GUARD_ACQUIRE : RuntimeLibcall;
def CXA_GUARD_RELEASE : RuntimeLibcall;
def CXA_THROW : RuntimeLibcall;
def DUNDER_ISOC99_SCANF : RuntimeLibcall;
def DUNDER_ISOC99_SSCANF : RuntimeLibcall;
def DUNDER_STRDUP : RuntimeLibcall;
def DUNDER_STRNDUP : RuntimeLibcall;
def DUNDER_STRTOK_R : RuntimeLibcall;
def ENUM_VARIANT : RuntimeLibcall;
def EXECL : RuntimeLibcall;
def EXECLE : RuntimeLibcall;
def EXECLP : RuntimeLibcall;
def EXECV : RuntimeLibcall;
def EXECVE : RuntimeLibcall;
def EXECVP : RuntimeLibcall;
def EXECVPE : RuntimeLibcall;
def EXIT : RuntimeLibcall;
def FCLOSE : RuntimeLibcall;
def FDOPEN : RuntimeLibcall;
def FEOF : RuntimeLibcall;
def FERROR : RuntimeLibcall;
def FFLUSH : RuntimeLibcall;
def FFS : RuntimeLibcall;
def FFSL : RuntimeLibcall;
def FFSLL : RuntimeLibcall;
def FGETC : RuntimeLibcall;
def FGETC_UNLOCKED : RuntimeLibcall;
def FGETPOS : RuntimeLibcall;
def FGETS : RuntimeLibcall;
def FGETS_UNLOCKED : RuntimeLibcall;
def FILENO : RuntimeLibcall;
def FIPRINTF : RuntimeLibcall;
def FLOCKFILE : RuntimeLibcall;
def FLS : RuntimeLibcall;
def FLSL : RuntimeLibcall;
def FLSLL : RuntimeLibcall;
def FOPEN : RuntimeLibcall;
def FOPEN64 : RuntimeLibcall;
def FORK : RuntimeLibcall;
def FPRINTF : RuntimeLibcall;
def FPUTC : RuntimeLibcall;
def FPUTC_UNLOCKED : RuntimeLibcall;
def FPUTS : RuntimeLibcall;
def FPUTS_UNLOCKED : RuntimeLibcall;
def FREAD : RuntimeLibcall;
def FREAD_UNLOCKED : RuntimeLibcall;
def FSCANF : RuntimeLibcall;
def FSEEK : RuntimeLibcall;
def FSEEKO : RuntimeLibcall;
def FSEEKO64 : RuntimeLibcall;
def FSETPOS : RuntimeLibcall;
def FSTAT : RuntimeLibcall;
def FSTAT64 : RuntimeLibcall;
def FSTATVFS : RuntimeLibcall;
def FSTATVFS64 : RuntimeLibcall;
def FTELL : RuntimeLibcall;
def FTELLO : RuntimeLibcall;
def FTELLO64 : RuntimeLibcall;
def FTRYLOCKFILE : RuntimeLibcall;
def FUNLOCKFILE : RuntimeLibcall;
def FWRITE : RuntimeLibcall;
def FWRITE_UNLOCKED : RuntimeLibcall;
def GETC : RuntimeLibcall;
def GETCHAR : RuntimeLibcall;
def GETCHAR_UNLOCKED : RuntimeLibcall;
def GETC_UNLOCKED : RuntimeLibcall;
def GETENV : RuntimeLibcall;
def GETITIMER : RuntimeLibcall;
def GETLOGIN_R : RuntimeLibcall;
def GETPWNAM : RuntimeLibcall;
def GETS : RuntimeLibcall;
def GETTIMEOFDAY : RuntimeLibcall;
def HTONL : RuntimeLibcall;
def HTONS : RuntimeLibcall;
def IPRINTF : RuntimeLibcall;
def ISASCII : RuntimeLibcall;
def ISDIGIT : RuntimeLibcall;
def LABS : RuntimeLibcall;
def LCHOWN : RuntimeLibcall;
def LLABS : RuntimeLibcall;
def LSTAT : RuntimeLibcall;
def LSTAT64 : RuntimeLibcall;
def MEMALIGN : RuntimeLibcall;
def MEMCCPY : RuntimeLibcall;
def MEMCCPY_CHK : RuntimeLibcall;
def MEMCHR : RuntimeLibcall;
def MEMPCPY : RuntimeLibcall;
def MEMPCPY_CHK : RuntimeLibcall;
def MEMRCHR : RuntimeLibcall;
def MEMSET_PATTERN16 : RuntimeLibcall;
def MEMSET_PATTERN4 : RuntimeLibcall;
def MEMSET_PATTERN8 : RuntimeLibcall;
def MKDIR : RuntimeLibcall;
def MKTIME : RuntimeLibcall;
def MSVC_DELETE_ARRAY_PTR32 : RuntimeLibcall;
def MSVC_DELETE_ARRAY_PTR32_INT : RuntimeLibcall;
def MSVC_DELETE_ARRAY_PTR32_NOTHROW : RuntimeLibcall;
def MSVC_DELETE_ARRAY_PTR64 : RuntimeLibcall;
def MSVC_DELETE_ARRAY_PTR64_LONGLONG : RuntimeLibcall;
def MSVC_DELETE_ARRAY_PTR64_NOTHROW : RuntimeLibcall;
def MSVC_DELETE_PTR32 : RuntimeLibcall;
def MSVC_DELETE_PTR32_INT : RuntimeLibcall;
def MSVC_DELETE_PTR32_NOTHROW : RuntimeLibcall;
def MSVC_DELETE_PTR64 : RuntimeLibcall;
def MSVC_DELETE_PTR64_LONGLONG : RuntimeLibcall;
def MSVC_DELETE_PTR64_NOTHROW : RuntimeLibcall;
def MSVC_NEW_ARRAY_INT : RuntimeLibcall;
def MSVC_NEW_ARRAY_INT_NOTHROW : RuntimeLibcall;
def MSVC_NEW_ARRAY_LONGLONG : RuntimeLibcall;
def MSVC_NEW_ARRAY_LONGLONG_NOTHROW : RuntimeLibcall;
def MSVC_NEW_INT : RuntimeLibcall;
def MSVC_NEW_INT_NOTHROW : RuntimeLibcall;
def MSVC_NEW_LONGLONG : RuntimeLibcall;
def MSVC_NEW_LONGLONG_NOTHROW : RuntimeLibcall;
def NTOHL : RuntimeLibcall;
def NTOHS : RuntimeLibcall;
def OPEN : RuntimeLibcall;
def OPEN64 : RuntimeLibcall;
def OPENDIR : RuntimeLibcall;
def PCLOSE : RuntimeLibcall;
def PERROR : RuntimeLibcall;
def POPEN : RuntimeLibcall;
def POSIX_MEMALIGN : RuntimeLibcall;
def PREAD : RuntimeLibcall;
def PRINTF : RuntimeLibcall;
def PUTC : RuntimeLibcall;
def PUTCHAR : RuntimeLibcall;
def PUTCHAR_UNLOCKED : RuntimeLibcall;
def PUTC_UNLOCKED : RuntimeLibcall;
def PUTS : RuntimeLibcall;
def PVALLOC : RuntimeLibcall;
def PWRITE : RuntimeLibcall;
def QSORT : RuntimeLibcall;
def READ : RuntimeLibcall;
def READLINK : RuntimeLibcall;
def REALLOC : RuntimeLibcall;
def REALLOCARRAY : RuntimeLibcall;
def REALLOCF : RuntimeLibcall;
def REALPATH : RuntimeLibcall;
def REMOVE : RuntimeLibcall;
def RENAME : RuntimeLibcall;
def REWIND : RuntimeLibcall;
def RMDIR : RuntimeLibcall;
def SCANF : RuntimeLibcall;
def SETBUF : RuntimeLibcall;
def SETITIMER : RuntimeLibcall;
def SETVBUF : RuntimeLibcall;
def SIPRINTF : RuntimeLibcall;
def SIZE_RETURNING_NEW : RuntimeLibcall;
def SIZE_RETURNING_NEW_ALIGNED : RuntimeLibcall;
def SIZE_RETURNING_NEW_ALIGNED_HOT_COLD : RuntimeLibcall;
def SIZE_RETURNING_NEW_HOT_COLD : RuntimeLibcall;
def SMALL_FPRINTF : RuntimeLibcall;
def SMALL_PRINTF : RuntimeLibcall;
def SMALL_SPRINTF : RuntimeLibcall;
def SNPRINTF : RuntimeLibcall;
def SNPRINTF_CHK : RuntimeLibcall;
def SPRINTF : RuntimeLibcall;
def SPRINTF_CHK : RuntimeLibcall;
def SSCANF : RuntimeLibcall;
def STAT : RuntimeLibcall;
def STAT64 : RuntimeLibcall;
def STATVFS : RuntimeLibcall;
def STATVFS64 : RuntimeLibcall;
def STPCPY : RuntimeLibcall;
def STPCPY_CHK : RuntimeLibcall;
def STPNCPY : RuntimeLibcall;
def STPNCPY_CHK : RuntimeLibcall;
def STRCASECMP : RuntimeLibcall;
def STRCAT : RuntimeLibcall;
def STRCAT_CHK : RuntimeLibcall;
def STRCHR : RuntimeLibcall;
def STRCMP : RuntimeLibcall;
def STRCOLL : RuntimeLibcall;
def STRCPY : RuntimeLibcall;
def STRCPY_CHK : RuntimeLibcall;
def STRCSPN : RuntimeLibcall;
def STRDUP : RuntimeLibcall;
def STRLCAT : RuntimeLibcall;
def STRLCAT_CHK : RuntimeLibcall;
def STRLCPY : RuntimeLibcall;
def STRLCPY_CHK : RuntimeLibcall;
def STRLEN_CHK : RuntimeLibcall;
def STRNCASECMP : RuntimeLibcall;
def STRNCAT : RuntimeLibcall;
def STRNCAT_CHK : RuntimeLibcall;
def STRNCMP : RuntimeLibcall;
def STRNCPY : RuntimeLibcall;
def STRNCPY_CHK : RuntimeLibcall;
def STRNDUP : RuntimeLibcall;
def STRNLEN : RuntimeLibcall;
def STRPBRK : RuntimeLibcall;
def STRRCHR : RuntimeLibcall;
def STRSPN : RuntimeLibcall;
def STRSTR : RuntimeLibcall;
def STRTOD : RuntimeLibcall;
def STRTOF : RuntimeLibcall;
def STRTOK : RuntimeLibcall;
def STRTOK_R : RuntimeLibcall;
def STRTOL : RuntimeLibcall;
def STRTOLD : RuntimeLibcall;
def STRTOLL : RuntimeLibcall;
def STRTOUL : RuntimeLibcall;
def STRTOULL : RuntimeLibcall;
def STRXFRM : RuntimeLibcall;
def SYSTEM : RuntimeLibcall;
def TERMINATE : RuntimeLibcall;
def TIMES : RuntimeLibcall;
def TMPFILE : RuntimeLibcall;
def TMPFILE64 : RuntimeLibcall;
def TOASCII : RuntimeLibcall;
def UNAME : RuntimeLibcall;
def UNDER_IO_GETC : RuntimeLibcall;
def UNDER_IO_PUTC : RuntimeLibcall;
def UNGETC : RuntimeLibcall;
def UNLINK : RuntimeLibcall;
def UNSETENV : RuntimeLibcall;
def UTIME : RuntimeLibcall;
def UTIMES : RuntimeLibcall;
def VALLOC : RuntimeLibcall;
def VEC_CALLOC : RuntimeLibcall;
def VEC_FREE : RuntimeLibcall;
def VEC_MALLOC : RuntimeLibcall;
def VEC_REALLOC : RuntimeLibcall;
def VFPRINTF : RuntimeLibcall;
def VFSCANF : RuntimeLibcall;
def VPRINTF : RuntimeLibcall;
def VSCANF : RuntimeLibcall;
def VSNPRINTF : RuntimeLibcall;
def VSNPRINTF_CHK : RuntimeLibcall;
def VSPRINTF : RuntimeLibcall;
def VSPRINTF_CHK : RuntimeLibcall;
def VSSCANF : RuntimeLibcall;
def WCSLEN : RuntimeLibcall;
def WRITE : RuntimeLibcall;
def ZDAPV : RuntimeLibcall;
def ZDAPVJ : RuntimeLibcall;
def ZDAPVJST11ALIGN_VAL_T : RuntimeLibcall;
def ZDAPVM : RuntimeLibcall;
def ZDAPVMST11ALIGN_VAL_T : RuntimeLibcall;
def ZDAPVRKST9NOTHROW_T : RuntimeLibcall;
def ZDAPVST11ALIGN_VAL_T : RuntimeLibcall;
def ZDAPVST11ALIGN_VAL_TRKST9NOTHROW_T : RuntimeLibcall;
def ZDLPV : RuntimeLibcall;
def ZDLPVJ : RuntimeLibcall;
def ZDLPVJST11ALIGN_VAL_T : RuntimeLibcall;
def ZDLPVM : RuntimeLibcall;
def ZDLPVMST11ALIGN_VAL_T : RuntimeLibcall;
def ZDLPVRKST9NOTHROW_T : RuntimeLibcall;
def ZDLPVST11ALIGN_VAL_T : RuntimeLibcall;
def ZDLPVST11ALIGN_VAL_TRKST9NOTHROW_T : RuntimeLibcall;
def ZNAJ : RuntimeLibcall;
def ZNAJRKST9NOTHROW_T : RuntimeLibcall;
def ZNAJST11ALIGN_VAL_T : RuntimeLibcall;
def ZNAJST11ALIGN_VAL_TRKST9NOTHROW_T : RuntimeLibcall;
def ZNAM : RuntimeLibcall;
def ZNAM12__HOT_COLD_T : RuntimeLibcall;
def ZNAMRKST9NOTHROW_T : RuntimeLibcall;
def ZNAMRKST9NOTHROW_T12__HOT_COLD_T : RuntimeLibcall;
def ZNAMST11ALIGN_VAL_T : RuntimeLibcall;
def ZNAMST11ALIGN_VAL_T12__HOT_COLD_T : RuntimeLibcall;
def ZNAMST11ALIGN_VAL_TRKST9NOTHROW_T : RuntimeLibcall;
def ZNAMST11ALIGN_VAL_TRKST9NOTHROW_T12__HOT_COLD_T : RuntimeLibcall;
def ZNWJ : RuntimeLibcall;
def ZNWJRKST9NOTHROW_T : RuntimeLibcall;
def ZNWJST11ALIGN_VAL_T : RuntimeLibcall;
def ZNWJST11ALIGN_VAL_TRKST9NOTHROW_T : RuntimeLibcall;
def ZNWM : RuntimeLibcall;
def ZNWM12__HOT_COLD_T : RuntimeLibcall;
def ZNWMRKST9NOTHROW_T : RuntimeLibcall;
def ZNWMRKST9NOTHROW_T12__HOT_COLD_T : RuntimeLibcall;
def ZNWMST11ALIGN_VAL_T : RuntimeLibcall;
def ZNWMST11ALIGN_VAL_T12__HOT_COLD_T : RuntimeLibcall;
def ZNWMST11ALIGN_VAL_TRKST9NOTHROW_T : RuntimeLibcall;
def ZNWMST11ALIGN_VAL_TRKST9NOTHROW_T12__HOT_COLD_T : RuntimeLibcall;
def KMPC_ALLOC_SHARED : RuntimeLibcall;
def KMPC_FREE_SHARED : RuntimeLibcall;

//--------------------------------------------------------------------
// Global variable references
//--------------------------------------------------------------------
//
// TODO: These are not libcalls and probably should be distinguished
// in some way from callable functions.
// --------------------------------------------------------------------

def STACK_CHECK_GUARD : RuntimeLibcall;

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
def __safestack_unsafe_stack_ptr : RuntimeLibcallImpl<SAFESTACK_UNSAFE_STACK_PTR>;

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
defm fmal : LibmLongDoubleLibCall;

def sqrtf : RuntimeLibcallImpl<SQRT_F32>;
def sqrt : RuntimeLibcallImpl<SQRT_F64>;
defm sqrtl : LibmLongDoubleLibCall;

def cbrtf : RuntimeLibcallImpl<CBRT_F32>;
def cbrt : RuntimeLibcallImpl<CBRT_F64>;
defm cbrtl : LibmLongDoubleLibCall;

def logf : RuntimeLibcallImpl<LOG_F32>;
def log : RuntimeLibcallImpl<LOG_F64>;
defm logl : LibmLongDoubleLibCall;

def __logf_finite : RuntimeLibcallImpl<LOG_FINITE_F32>;
def __log_finite : RuntimeLibcallImpl<LOG_FINITE_F64>;
def __logl_finite_f80 : RuntimeLibcallImpl<LOG_FINITE_F80, "__logl_finite">;
def __logl_finite_f128 : RuntimeLibcallImpl<LOG_FINITE_F128, "__logl_finite">;
def __logl_finite_ppcf128 : RuntimeLibcallImpl<LOG_FINITE_PPCF128, "__logl_finite">;

def log2f : RuntimeLibcallImpl<LOG2_F32>;
def log2 : RuntimeLibcallImpl<LOG2_F64>;
defm log2l : LibmLongDoubleLibCall;

def __log2f_finite : RuntimeLibcallImpl<LOG2_FINITE_F32>;
def __log2_finite : RuntimeLibcallImpl<LOG2_FINITE_F64>;
def __log2l_finite_f80 : RuntimeLibcallImpl<LOG2_FINITE_F80, "__log2l_finite">;
def __log2l_finite_f128 : RuntimeLibcallImpl<LOG2_FINITE_F128, "__log2l_finite">;
def __log2l_finite_ppcf128 : RuntimeLibcallImpl<LOG2_FINITE_PPCF128, "__log2l_finite">;

def log10f : RuntimeLibcallImpl<LOG10_F32>;
def log10 : RuntimeLibcallImpl<LOG10_F64>;
defm log10l : LibmLongDoubleLibCall;

def __log10f_finite : RuntimeLibcallImpl<LOG10_FINITE_F32>;
def __log10_finite : RuntimeLibcallImpl<LOG10_FINITE_F64>;
def __log10l_finite_f80 : RuntimeLibcallImpl<LOG10_FINITE_F80, "__log10l_finite">;
def __log10l_finite_f128 : RuntimeLibcallImpl<LOG10_FINITE_F128, "__log10l_finite">;
def __log10l_finite_ppcf128 : RuntimeLibcallImpl<LOG10_FINITE_PPCF128, "__log10l_finite">;

def expf : RuntimeLibcallImpl<EXP_F32>;
def exp : RuntimeLibcallImpl<EXP_F64>;
defm expl : LibmLongDoubleLibCall<"EXP">;

def __expf_finite : RuntimeLibcallImpl<EXP_FINITE_F32>;
def __exp_finite : RuntimeLibcallImpl<EXP_FINITE_F64>;
def __expl_finite_f80 : RuntimeLibcallImpl<EXP_FINITE_F80, "__expl_finite">;
def __expl_finite_f128 : RuntimeLibcallImpl<EXP_FINITE_F128, "__expl_finite">;
def __expl_finite_ppcf128 : RuntimeLibcallImpl<EXP_FINITE_PPCF128, "__expl_finite">;

def exp2f : RuntimeLibcallImpl<EXP2_F32>;
def exp2 : RuntimeLibcallImpl<EXP2_F64>;
defm exp2l : LibmLongDoubleLibCall<"EXP2">;

def __exp2f_finite : RuntimeLibcallImpl<EXP2_FINITE_F32>;
def __exp2_finite : RuntimeLibcallImpl<EXP2_FINITE_F64>;
def __exp2l_finite_f80 : RuntimeLibcallImpl<EXP2_FINITE_F80, "__exp2l_finite">;
def __exp2l_finite_f128 : RuntimeLibcallImpl<EXP2_FINITE_F128, "__exp2l_finite">;
def __exp2l_finite_ppcf128 : RuntimeLibcallImpl<EXP2_FINITE_PPCF128, "__exp2l_finite">;

def sinf : RuntimeLibcallImpl<SIN_F32>;
def sin : RuntimeLibcallImpl<SIN_F64>;
defm sinl : LibmLongDoubleLibCall;

def cosf : RuntimeLibcallImpl<COS_F32>;
def cos : RuntimeLibcallImpl<COS_F64>;
defm cosl : LibmLongDoubleLibCall;

def tanf : RuntimeLibcallImpl<TAN_F32>;
def tan : RuntimeLibcallImpl<TAN_F64>;
defm tanl : LibmLongDoubleLibCall;

def sinhf : RuntimeLibcallImpl<SINH_F32>;
def sinh : RuntimeLibcallImpl<SINH_F64>;
defm sinhl : LibmLongDoubleLibCall;

def coshf : RuntimeLibcallImpl<COSH_F32>;
def cosh : RuntimeLibcallImpl<COSH_F64>;
defm coshl : LibmLongDoubleLibCall;

def tanhf : RuntimeLibcallImpl<TANH_F32>;
def tanh : RuntimeLibcallImpl<TANH_F64>;
defm tanhl : LibmLongDoubleLibCall;

def asinf : RuntimeLibcallImpl<ASIN_F32>;
def asin : RuntimeLibcallImpl<ASIN_F64>;
defm asinl : LibmLongDoubleLibCall;

def acosf : RuntimeLibcallImpl<ACOS_F32>;
def acos : RuntimeLibcallImpl<ACOS_F64>;
defm acosl : LibmLongDoubleLibCall;

def atanf : RuntimeLibcallImpl<ATAN_F32>;
def atan : RuntimeLibcallImpl<ATAN_F64>;
defm atanl : LibmLongDoubleLibCall;

def atan2f : RuntimeLibcallImpl<ATAN2_F32>;
def atan2 : RuntimeLibcallImpl<ATAN2_F64>;
defm atan2l : LibmLongDoubleLibCall;

def powf : RuntimeLibcallImpl<POW_F32>;
def pow : RuntimeLibcallImpl<POW_F64>;
defm powl : LibmLongDoubleLibCall;

def __powf_finite : RuntimeLibcallImpl<POW_FINITE_F32>;
def __pow_finite : RuntimeLibcallImpl<POW_FINITE_F64>;
def __powl_finite_f80 : RuntimeLibcallImpl<POW_FINITE_F80, "__powl_finite">;
def __powl_finite_f128 : RuntimeLibcallImpl<POW_FINITE_F128, "__powl_finite">;
def __powl_finite_ppcf128 : RuntimeLibcallImpl<POW_FINITE_PPCF128, "__powl_finite">;

def ceilf : RuntimeLibcallImpl<CEIL_F32>;
def ceil : RuntimeLibcallImpl<CEIL_F64>;
defm ceill : LibmLongDoubleLibCall;

def truncf : RuntimeLibcallImpl<TRUNC_F32>;
def trunc : RuntimeLibcallImpl<TRUNC_F64>;
defm truncl : LibmLongDoubleLibCall;

def rintf : RuntimeLibcallImpl<RINT_F32>;
def rint : RuntimeLibcallImpl<RINT_F64>;
defm rintl : LibmLongDoubleLibCall;

def nearbyintf : RuntimeLibcallImpl<NEARBYINT_F32>;
def nearbyint : RuntimeLibcallImpl<NEARBYINT_F64>;
defm nearbyintl : LibmLongDoubleLibCall;

def roundf : RuntimeLibcallImpl<ROUND_F32>;
def round : RuntimeLibcallImpl<ROUND_F64>;
defm roundl : LibmLongDoubleLibCall;

def roundevenf : RuntimeLibcallImpl<ROUNDEVEN_F32>;
def roundeven : RuntimeLibcallImpl<ROUNDEVEN_F64>;
defm roundevenl : LibmLongDoubleLibCall;

def floorf : RuntimeLibcallImpl<FLOOR_F32>;
def floor : RuntimeLibcallImpl<FLOOR_F64>;
defm floorl : LibmLongDoubleLibCall;

def copysignf : RuntimeLibcallImpl<COPYSIGN_F32>;
def copysign : RuntimeLibcallImpl<COPYSIGN_F64>;
defm copysignl : LibmLongDoubleLibCall;

def fminf : RuntimeLibcallImpl<FMIN_F32>;
def fmin : RuntimeLibcallImpl<FMIN_F64>;
defm fminl : LibmLongDoubleLibCall;

def fmaxf : RuntimeLibcallImpl<FMAX_F32>;
def fmax : RuntimeLibcallImpl<FMAX_F64>;
defm fmaxl : LibmLongDoubleLibCall;

def fminimumf : RuntimeLibcallImpl<FMINIMUM_F32>;
def fminimum : RuntimeLibcallImpl<FMINIMUM_F64>;
defm fminimuml : LibmLongDoubleLibCall;

def fmaximumf : RuntimeLibcallImpl<FMAXIMUM_F32>;
def fmaximum : RuntimeLibcallImpl<FMAXIMUM_F64>;
defm fmaximuml : LibmLongDoubleLibCall;

def fminimum_numf : RuntimeLibcallImpl<FMINIMUM_NUM_F32>;
def fminimum_num : RuntimeLibcallImpl<FMINIMUM_NUM_F64>;
defm fminimum_numl : LibmLongDoubleLibCall;

def fmaximum_numf : RuntimeLibcallImpl<FMAXIMUM_NUM_F32>;
def fmaximum_num : RuntimeLibcallImpl<FMAXIMUM_NUM_F64>;
defm fmaximum_numl : LibmLongDoubleLibCall;

def lroundf : RuntimeLibcallImpl<LROUND_F32>;
def lround : RuntimeLibcallImpl<LROUND_F64>;
defm lroundl : LibmLongDoubleLibCall;

def llroundf : RuntimeLibcallImpl<LLROUND_F32>;
def llround : RuntimeLibcallImpl<LLROUND_F64>;
defm llroundl : LibmLongDoubleLibCall;

def lrintf : RuntimeLibcallImpl<LRINT_F32>;
def lrint : RuntimeLibcallImpl<LRINT_F64>;
defm lrintl : LibmLongDoubleLibCall;

def llrintf : RuntimeLibcallImpl<LLRINT_F32>;
def llrint : RuntimeLibcallImpl<LLRINT_F64>;
defm llrintl : LibmLongDoubleLibCall;

def ldexpf : RuntimeLibcallImpl<LDEXP_F32>;
def ldexp : RuntimeLibcallImpl<LDEXP_F64>;
defm ldexpl : LibmLongDoubleLibCall;

def frexpf : RuntimeLibcallImpl<FREXP_F32>;
def frexp : RuntimeLibcallImpl<FREXP_F64>;
defm frexpl : LibmLongDoubleLibCall;

def modff : RuntimeLibcallImpl<MODF_F32>;
def modf : RuntimeLibcallImpl<MODF_F64>;
defm modfl : LibmLongDoubleLibCall;

def nanf : RuntimeLibcallImpl<NAN_F32>;
def nan : RuntimeLibcallImpl<NAN_F64>;
defm nanl : LibmLongDoubleLibCall;

def nexttowardf : RuntimeLibcallImpl<NEXTTOWARD_F32>;
def nexttoward : RuntimeLibcallImpl<NEXTTOWARD_F64>;
defm nexttowardl : LibmLongDoubleLibCall;

def remainderf : RuntimeLibcallImpl<REMAINDER_F32>;
def remainder : RuntimeLibcallImpl<REMAINDER_F64>;
defm remainderl : LibmLongDoubleLibCall;

def remquof : RuntimeLibcallImpl<REMQUO_F32>;
def remquo : RuntimeLibcallImpl<REMQUO_F64>;
defm remquol : LibmLongDoubleLibCall;

def fdimf : RuntimeLibcallImpl<FDIM_F32>;
def fdim : RuntimeLibcallImpl<FDIM_F64>;
defm fdiml : LibmLongDoubleLibCall;

def scalbnf : RuntimeLibcallImpl<SCALBN_F32>;
def scalbn : RuntimeLibcallImpl<SCALBN_F64>;
defm scalbnl : LibmLongDoubleLibCall;

def scalblnf : RuntimeLibcallImpl<SCALBLN_F32>;
def scalbln : RuntimeLibcallImpl<SCALBLN_F64>;
defm scalblnl : LibmLongDoubleLibCall;

def tgammaf : RuntimeLibcallImpl<TGAMMA_F32>;
def tgamma : RuntimeLibcallImpl<TGAMMA_F64>;
defm tgammal : LibmLongDoubleLibCall;

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

def __memcpy_chk : RuntimeLibcallImpl<MEMCPY_CHK>;
def __memmove_chk : RuntimeLibcallImpl<MEMMOVE_CHK>;
def __memset_chk : RuntimeLibcallImpl<MEMSET_CHK>;

def malloc : RuntimeLibcallImpl<MALLOC>;

// DSEPass can emit calloc if it finds a pair of malloc/memset
def calloc : RuntimeLibcallImpl<CALLOC>;
def free : RuntimeLibcallImpl<FREE>;

} // End let IsDefault = true

def exp10f : RuntimeLibcallImpl<EXP10_F32>;
def exp10 : RuntimeLibcallImpl<EXP10_F64>;
def exp10l_f80 : RuntimeLibcallImpl<EXP10_F80, "exp10l">;
def exp10l_f128 : RuntimeLibcallImpl<EXP10_F128, "exp10l">;
def exp10l_ppcf128 : RuntimeLibcallImpl<EXP10_PPCF128, "exp10l">;

// Stack Protector Fail
def __stack_chk_fail : RuntimeLibcallImpl<STACKPROTECTOR_CHECK_FAIL>;

//--------------------------------------------------------------------
// Other functions from TargetLibraryInfo
//
// TODO: These need to be organized by library and added to relevant
// systems.
///
// --------------------------------------------------------------------

def __2_YAPAXI_Z : RuntimeLibcallImpl<MSVC_NEW_INT, "??2@YAPAXI@Z">;
def __2_YAPAXIABUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_NEW_INT_NOTHROW,
                         "??2@YAPAXIABUnothrow_t@std@@@Z">;
def __2_YAPEAX_K_Z : RuntimeLibcallImpl<MSVC_NEW_LONGLONG, "??2@YAPEAX_K@Z">;
def __2_YAPEAX_KAEBUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_NEW_LONGLONG_NOTHROW,
                         "??2@YAPEAX_KAEBUnothrow_t@std@@@Z">;
def __3_YAXPAX_Z : RuntimeLibcallImpl<MSVC_DELETE_PTR32, "??3@YAXPAX@Z">;
def __3_YAXPAXABUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_DELETE_PTR32_NOTHROW,
                         "??3@YAXPAXABUnothrow_t@std@@@Z">;
def __3_YAXPAXI_Z : RuntimeLibcallImpl<MSVC_DELETE_PTR32_INT, "??3@YAXPAXI@Z">;
def __3_YAXPEAX_Z : RuntimeLibcallImpl<MSVC_DELETE_PTR64, "??3@YAXPEAX@Z">;
def __3_YAXPEAXAEBUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_DELETE_PTR64_NOTHROW,
                         "??3@YAXPEAXAEBUnothrow_t@std@@@Z">;
def __3_YAXPEAX_K_Z
    : RuntimeLibcallImpl<MSVC_DELETE_PTR64_LONGLONG, "??3@YAXPEAX_K@Z">;
def ___U_YAPAXI_Z : RuntimeLibcallImpl<MSVC_NEW_ARRAY_INT, "??_U@YAPAXI@Z">;
def ___U_YAPAXIABUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_NEW_ARRAY_INT_NOTHROW,
                         "??_U@YAPAXIABUnothrow_t@std@@@Z">;
def ___U_YAPEAX_K_Z
    : RuntimeLibcallImpl<MSVC_NEW_ARRAY_LONGLONG, "??_U@YAPEAX_K@Z">;
def ___U_YAPEAX_KAEBUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_NEW_ARRAY_LONGLONG_NOTHROW,
                         "??_U@YAPEAX_KAEBUnothrow_t@std@@@Z">;
def ___V_YAXPAX_Z
    : RuntimeLibcallImpl<MSVC_DELETE_ARRAY_PTR32, "??_V@YAXPAX@Z">;
def ___V_YAXPAXABUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_DELETE_ARRAY_PTR32_NOTHROW,
                         "??_V@YAXPAXABUnothrow_t@std@@@Z">;
def ___V_YAXPAXI_Z
    : RuntimeLibcallImpl<MSVC_DELETE_ARRAY_PTR32_INT, "??_V@YAXPAXI@Z">;
def ___V_YAXPEAX_Z
    : RuntimeLibcallImpl<MSVC_DELETE_ARRAY_PTR64, "??_V@YAXPEAX@Z">;
def ___V_YAXPEAXAEBUnothrow_t_std___Z
    : RuntimeLibcallImpl<MSVC_DELETE_ARRAY_PTR64_NOTHROW,
                         "??_V@YAXPEAXAEBUnothrow_t@std@@@Z">;
def ___V_YAXPEAX_K_Z
    : RuntimeLibcallImpl<MSVC_DELETE_ARRAY_PTR64_LONGLONG, "??_V@YAXPEAX_K@Z">;
def _IO_getc : RuntimeLibcallImpl<UNDER_IO_GETC>;
def _IO_putc : RuntimeLibcallImpl<UNDER_IO_PUTC>;
def _ZdaPv : RuntimeLibcallImpl<ZDAPV>;
def _ZdaPvRKSt9nothrow_t : RuntimeLibcallImpl<ZDAPVRKST9NOTHROW_T>;
def _ZdaPvSt11align_val_t : RuntimeLibcallImpl<ZDAPVST11ALIGN_VAL_T>;
def _ZdaPvSt11align_val_tRKSt9nothrow_t
    : RuntimeLibcallImpl<ZDAPVST11ALIGN_VAL_TRKST9NOTHROW_T>;
def _ZdaPvj : RuntimeLibcallImpl<ZDAPVJ>;
def _ZdaPvjSt11align_val_t : RuntimeLibcallImpl<ZDAPVJST11ALIGN_VAL_T>;
def _ZdaPvm : RuntimeLibcallImpl<ZDAPVM>;
def _ZdaPvmSt11align_val_t : RuntimeLibcallImpl<ZDAPVMST11ALIGN_VAL_T>;
def _ZdlPv : RuntimeLibcallImpl<ZDLPV>;
def _ZdlPvRKSt9nothrow_t : RuntimeLibcallImpl<ZDLPVRKST9NOTHROW_T>;
def _ZdlPvSt11align_val_t : RuntimeLibcallImpl<ZDLPVST11ALIGN_VAL_T>;
def _ZdlPvSt11align_val_tRKSt9nothrow_t
    : RuntimeLibcallImpl<ZDLPVST11ALIGN_VAL_TRKST9NOTHROW_T>;
def _ZdlPvj : RuntimeLibcallImpl<ZDLPVJ>;
def _ZdlPvjSt11align_val_t : RuntimeLibcallImpl<ZDLPVJST11ALIGN_VAL_T>;
def _ZdlPvm : RuntimeLibcallImpl<ZDLPVM>;
def _ZdlPvmSt11align_val_t : RuntimeLibcallImpl<ZDLPVMST11ALIGN_VAL_T>;
def _Znaj : RuntimeLibcallImpl<ZNAJ>;
def _ZnajRKSt9nothrow_t : RuntimeLibcallImpl<ZNAJRKST9NOTHROW_T>;
def _ZnajSt11align_val_t : RuntimeLibcallImpl<ZNAJST11ALIGN_VAL_T>;
def _ZnajSt11align_val_tRKSt9nothrow_t
    : RuntimeLibcallImpl<ZNAJST11ALIGN_VAL_TRKST9NOTHROW_T>;
def _Znam : RuntimeLibcallImpl<ZNAM>;
def _Znam12__hot_cold_t : RuntimeLibcallImpl<ZNAM12__HOT_COLD_T>;
def _ZnamRKSt9nothrow_t : RuntimeLibcallImpl<ZNAMRKST9NOTHROW_T>;
def _ZnamRKSt9nothrow_t12__hot_cold_t
    : RuntimeLibcallImpl<ZNAMRKST9NOTHROW_T12__HOT_COLD_T>;
def _ZnamSt11align_val_t : RuntimeLibcallImpl<ZNAMST11ALIGN_VAL_T>;
def _ZnamSt11align_val_t12__hot_cold_t
    : RuntimeLibcallImpl<ZNAMST11ALIGN_VAL_T12__HOT_COLD_T>;
def _ZnamSt11align_val_tRKSt9nothrow_t
    : RuntimeLibcallImpl<ZNAMST11ALIGN_VAL_TRKST9NOTHROW_T>;
def _ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : RuntimeLibcallImpl<ZNAMST11ALIGN_VAL_TRKST9NOTHROW_T12__HOT_COLD_T>;
def _Znwj : RuntimeLibcallImpl<ZNWJ>;
def _ZnwjRKSt9nothrow_t : RuntimeLibcallImpl<ZNWJRKST9NOTHROW_T>;
def _ZnwjSt11align_val_t : RuntimeLibcallImpl<ZNWJST11ALIGN_VAL_T>;
def _ZnwjSt11align_val_tRKSt9nothrow_t
    : RuntimeLibcallImpl<ZNWJST11ALIGN_VAL_TRKST9NOTHROW_T>;
def _Znwm : RuntimeLibcallImpl<ZNWM>;
def _Znwm12__hot_cold_t : RuntimeLibcallImpl<ZNWM12__HOT_COLD_T>;
def _ZnwmRKSt9nothrow_t : RuntimeLibcallImpl<ZNWMRKST9NOTHROW_T>;
def _ZnwmRKSt9nothrow_t12__hot_cold_t
    : RuntimeLibcallImpl<ZNWMRKST9NOTHROW_T12__HOT_COLD_T>;
def _ZnwmSt11align_val_t : RuntimeLibcallImpl<ZNWMST11ALIGN_VAL_T>;
def _ZnwmSt11align_val_t12__hot_cold_t
    : RuntimeLibcallImpl<ZNWMST11ALIGN_VAL_T12__HOT_COLD_T>;
def _ZnwmSt11align_val_tRKSt9nothrow_t
    : RuntimeLibcallImpl<ZNWMST11ALIGN_VAL_TRKST9NOTHROW_T>;
def _ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t
    : RuntimeLibcallImpl<ZNWMST11ALIGN_VAL_TRKST9NOTHROW_T12__HOT_COLD_T>;
def __size_returning_new : RuntimeLibcallImpl<SIZE_RETURNING_NEW>;
def __size_returning_new_hot_cold
    : RuntimeLibcallImpl<SIZE_RETURNING_NEW_HOT_COLD>;
def __size_returning_new_aligned
    : RuntimeLibcallImpl<SIZE_RETURNING_NEW_ALIGNED>;
def __size_returning_new_aligned_hot_cold
    : RuntimeLibcallImpl<SIZE_RETURNING_NEW_ALIGNED_HOT_COLD>;
def __cxa_atexit : RuntimeLibcallImpl<CXA_ATEXIT>;
def atexit : RuntimeLibcallImpl<ATEXIT>;
def abort : RuntimeLibcallImpl<ABORT>;
def exit : RuntimeLibcallImpl<EXIT>;
def _Exit : RuntimeLibcallImpl<EXIT>;
def _ZSt9terminatev : RuntimeLibcallImpl<TERMINATE>;
def __cxa_throw : RuntimeLibcallImpl<CXA_THROW>;
def __cxa_guard_abort : RuntimeLibcallImpl<CXA_GUARD_ABORT>;
def __cxa_guard_acquire : RuntimeLibcallImpl<CXA_GUARD_ACQUIRE>;
def __cxa_guard_release : RuntimeLibcallImpl<CXA_GUARD_RELEASE>;
def __isoc99_scanf : RuntimeLibcallImpl<DUNDER_ISOC99_SCANF>;
def __isoc99_sscanf : RuntimeLibcallImpl<DUNDER_ISOC99_SSCANF>;
def __kmpc_alloc_shared : RuntimeLibcallImpl<KMPC_ALLOC_SHARED>;
def __kmpc_free_shared : RuntimeLibcallImpl<KMPC_FREE_SHARED>;
def __memccpy_chk : RuntimeLibcallImpl<MEMCCPY_CHK>;
def __mempcpy_chk : RuntimeLibcallImpl<MEMPCPY_CHK>;
def __small_fprintf : RuntimeLibcallImpl<SMALL_FPRINTF>;
def __small_printf : RuntimeLibcallImpl<SMALL_PRINTF>;
def __small_sprintf : RuntimeLibcallImpl<SMALL_SPRINTF>;
def __snprintf_chk : RuntimeLibcallImpl<SNPRINTF_CHK>;
def __sprintf_chk : RuntimeLibcallImpl<SPRINTF_CHK>;
def __stpcpy_chk : RuntimeLibcallImpl<STPCPY_CHK>;
def __stpncpy_chk : RuntimeLibcallImpl<STPNCPY_CHK>;
def __strcat_chk : RuntimeLibcallImpl<STRCAT_CHK>;
def __strcpy_chk : RuntimeLibcallImpl<STRCPY_CHK>;
def __strdup : RuntimeLibcallImpl<DUNDER_STRDUP>;
def __strlcat_chk : RuntimeLibcallImpl<STRLCAT_CHK>;
def __strlcpy_chk : RuntimeLibcallImpl<STRLCPY_CHK>;
def __strlen_chk : RuntimeLibcallImpl<STRLEN_CHK>;
def __strncat_chk : RuntimeLibcallImpl<STRNCAT_CHK>;
def __strncpy_chk : RuntimeLibcallImpl<STRNCPY_CHK>;
def __strndup : RuntimeLibcallImpl<DUNDER_STRNDUP>;
def __strtok_r : RuntimeLibcallImpl<DUNDER_STRTOK_R>;
def __vsnprintf_chk : RuntimeLibcallImpl<VSNPRINTF_CHK>;
def __vsprintf_chk : RuntimeLibcallImpl<VSPRINTF_CHK>;
def abs : RuntimeLibcallImpl<ABS>;
def access : RuntimeLibcallImpl<ACCESS>;
def aligned_alloc : RuntimeLibcallImpl<ALIGNED_ALLOC>;
def atof : RuntimeLibcallImpl<ATOF>;
def atoi : RuntimeLibcallImpl<ATOI>;
def atol : RuntimeLibcallImpl<ATOL>;
def atoll : RuntimeLibcallImpl<ATOLL>;
def bcmp : RuntimeLibcallImpl<BCMP>;
def bcopy : RuntimeLibcallImpl<BCOPY>;
def cabs : RuntimeLibcallImpl<CABS_F64>;
def cabsf : RuntimeLibcallImpl<CABS_F32>;
defm cabsl : LibmLongDoubleLibCall;
def chmod : RuntimeLibcallImpl<CHMOD>;
def chown : RuntimeLibcallImpl<CHOWN>;
def clearerr : RuntimeLibcallImpl<CLEARERR>;
def closedir : RuntimeLibcallImpl<CLOSEDIR>;
def ctermid : RuntimeLibcallImpl<CTERMID>;
def execl : RuntimeLibcallImpl<EXECL>;
def execle : RuntimeLibcallImpl<EXECLE>;
def execlp : RuntimeLibcallImpl<EXECLP>;
def execv : RuntimeLibcallImpl<EXECV>;
def execvP : RuntimeLibcallImpl<EXECVP>;
def execve : RuntimeLibcallImpl<EXECVE>;
def execvp : RuntimeLibcallImpl<EXECVP>;
def execvpe : RuntimeLibcallImpl<EXECVPE>;
def fclose : RuntimeLibcallImpl<FCLOSE>;
def fdopen : RuntimeLibcallImpl<FDOPEN>;
def feof : RuntimeLibcallImpl<FEOF>;
def ferror : RuntimeLibcallImpl<FERROR>;
def fflush : RuntimeLibcallImpl<FFLUSH>;
def ffs : RuntimeLibcallImpl<FFS>;
def ffsl : RuntimeLibcallImpl<FFSL>;
def ffsll : RuntimeLibcallImpl<FFSLL>;
def fgetc : RuntimeLibcallImpl<FGETC>;
def fgetc_unlocked : RuntimeLibcallImpl<FGETC_UNLOCKED>;
def fgetpos : RuntimeLibcallImpl<FGETPOS>;
def fgets : RuntimeLibcallImpl<FGETS>;
def fgets_unlocked : RuntimeLibcallImpl<FGETS_UNLOCKED>;
def fileno : RuntimeLibcallImpl<FILENO>;
def fiprintf : RuntimeLibcallImpl<FIPRINTF>;
def flockfile : RuntimeLibcallImpl<FLOCKFILE>;
def fls : RuntimeLibcallImpl<FLS>;
def flsl : RuntimeLibcallImpl<FLSL>;
def flsll : RuntimeLibcallImpl<FLSLL>;
def fopen : RuntimeLibcallImpl<FOPEN>;
def fopen64 : RuntimeLibcallImpl<FOPEN64>;
def fork : RuntimeLibcallImpl<FORK>;
def fprintf : RuntimeLibcallImpl<FPRINTF>;
def fputc : RuntimeLibcallImpl<FPUTC>;
def fputc_unlocked : RuntimeLibcallImpl<FPUTC_UNLOCKED>;
def fputs : RuntimeLibcallImpl<FPUTS>;
def fputs_unlocked : RuntimeLibcallImpl<FPUTS_UNLOCKED>;
def fread : RuntimeLibcallImpl<FREAD>;
def fread_unlocked : RuntimeLibcallImpl<FREAD_UNLOCKED>;
def fscanf : RuntimeLibcallImpl<FSCANF>;
def fseek : RuntimeLibcallImpl<FSEEK>;
def fseeko : RuntimeLibcallImpl<FSEEKO>;
def fseeko64 : RuntimeLibcallImpl<FSEEKO64>;
def fsetpos : RuntimeLibcallImpl<FSETPOS>;
def fstat : RuntimeLibcallImpl<FSTAT>;
def fstat64 : RuntimeLibcallImpl<FSTAT64>;
def fstatvfs : RuntimeLibcallImpl<FSTATVFS>;
def fstatvfs64 : RuntimeLibcallImpl<FSTATVFS64>;
def ftell : RuntimeLibcallImpl<FTELL>;
def ftello : RuntimeLibcallImpl<FTELLO>;
def ftello64 : RuntimeLibcallImpl<FTELLO64>;
def ftrylockfile : RuntimeLibcallImpl<FTRYLOCKFILE>;
def funlockfile : RuntimeLibcallImpl<FUNLOCKFILE>;
def fwrite : RuntimeLibcallImpl<FWRITE>;
def fwrite_unlocked : RuntimeLibcallImpl<FWRITE_UNLOCKED>;
def getc : RuntimeLibcallImpl<GETC>;
def getc_unlocked : RuntimeLibcallImpl<GETC_UNLOCKED>;
def getchar : RuntimeLibcallImpl<GETCHAR>;
def getchar_unlocked : RuntimeLibcallImpl<GETCHAR_UNLOCKED>;
def getenv : RuntimeLibcallImpl<GETENV>;
def getitimer : RuntimeLibcallImpl<GETITIMER>;
def getlogin_r : RuntimeLibcallImpl<GETLOGIN_R>;
def getpwnam : RuntimeLibcallImpl<GETPWNAM>;
def gets : RuntimeLibcallImpl<GETS>;
def gettimeofday : RuntimeLibcallImpl<GETTIMEOFDAY>;
def htonl : RuntimeLibcallImpl<HTONL>;
def htons : RuntimeLibcallImpl<HTONS>;
def iprintf : RuntimeLibcallImpl<IPRINTF>;
def isascii : RuntimeLibcallImpl<ISASCII>;
def isdigit : RuntimeLibcallImpl<ISDIGIT>;
def labs : RuntimeLibcallImpl<LABS>;
def lchown : RuntimeLibcallImpl<LCHOWN>;
def llabs : RuntimeLibcallImpl<LLABS>;
def lstat : RuntimeLibcallImpl<LSTAT>;
def lstat64 : RuntimeLibcallImpl<LSTAT64>;
def memalign : RuntimeLibcallImpl<MEMALIGN>;
def memccpy : RuntimeLibcallImpl<MEMCCPY>;
def memchr : RuntimeLibcallImpl<MEMCHR>;
def memcmp : RuntimeLibcallImpl<MEMCMP>;
def mempcpy : RuntimeLibcallImpl<MEMPCPY>;
def memrchr : RuntimeLibcallImpl<MEMRCHR>;
def memset_pattern16 : RuntimeLibcallImpl<MEMSET_PATTERN16>;
def memset_pattern4 : RuntimeLibcallImpl<MEMSET_PATTERN4>;
def memset_pattern8 : RuntimeLibcallImpl<MEMSET_PATTERN8>;
def mkdir : RuntimeLibcallImpl<MKDIR>;
def mktime : RuntimeLibcallImpl<MKTIME>;
def ntohl : RuntimeLibcallImpl<NTOHL>;
def ntohs : RuntimeLibcallImpl<NTOHS>;
def open : RuntimeLibcallImpl<OPEN>;
def open64 : RuntimeLibcallImpl<OPEN64>;
def opendir : RuntimeLibcallImpl<OPENDIR>;
def pclose : RuntimeLibcallImpl<PCLOSE>;
def perror : RuntimeLibcallImpl<PERROR>;
def popen : RuntimeLibcallImpl<POPEN>;
def posix_memalign : RuntimeLibcallImpl<POSIX_MEMALIGN>;
def pread : RuntimeLibcallImpl<PREAD>;
def printf : RuntimeLibcallImpl<PRINTF>;
def putc : RuntimeLibcallImpl<PUTC>;
def putc_unlocked : RuntimeLibcallImpl<PUTC_UNLOCKED>;
def putchar : RuntimeLibcallImpl<PUTCHAR>;
def putchar_unlocked : RuntimeLibcallImpl<PUTCHAR_UNLOCKED>;
def puts : RuntimeLibcallImpl<PUTS>;
def pvalloc : RuntimeLibcallImpl<PVALLOC>;
def pwrite : RuntimeLibcallImpl<PWRITE>;
def qsort : RuntimeLibcallImpl<QSORT>;
def read : RuntimeLibcallImpl<READ>;
def readlink : RuntimeLibcallImpl<READLINK>;
def realloc : RuntimeLibcallImpl<REALLOC>;
def reallocf : RuntimeLibcallImpl<REALLOCF>;
def reallocarray : RuntimeLibcallImpl<REALLOCARRAY>;
def realpath : RuntimeLibcallImpl<REALPATH>;
def remove : RuntimeLibcallImpl<REMOVE>;
def rename : RuntimeLibcallImpl<RENAME>;
def rewind : RuntimeLibcallImpl<REWIND>;
def rmdir : RuntimeLibcallImpl<RMDIR>;
def scanf : RuntimeLibcallImpl<SCANF>;
def setbuf : RuntimeLibcallImpl<SETBUF>;
def setitimer : RuntimeLibcallImpl<SETITIMER>;
def setvbuf : RuntimeLibcallImpl<SETVBUF>;
def siprintf : RuntimeLibcallImpl<SIPRINTF>;
def snprintf : RuntimeLibcallImpl<SNPRINTF>;
def sprintf : RuntimeLibcallImpl<SPRINTF>;
def sscanf : RuntimeLibcallImpl<SSCANF>;
def stat : RuntimeLibcallImpl<STAT>;
def stat64 : RuntimeLibcallImpl<STAT64>;
def statvfs : RuntimeLibcallImpl<STATVFS>;
def statvfs64 : RuntimeLibcallImpl<STATVFS64>;
def stpcpy : RuntimeLibcallImpl<STPCPY>;
def stpncpy : RuntimeLibcallImpl<STPNCPY>;
def strcasecmp : RuntimeLibcallImpl<STRCASECMP>;
def strcat : RuntimeLibcallImpl<STRCAT>;
def strchr : RuntimeLibcallImpl<STRCHR>;
def strcmp : RuntimeLibcallImpl<STRCMP>;
def strcoll : RuntimeLibcallImpl<STRCOLL>;
def strcpy : RuntimeLibcallImpl<STRCPY>;
def strcspn : RuntimeLibcallImpl<STRCSPN>;
def strdup : RuntimeLibcallImpl<STRDUP>;
def strlcat : RuntimeLibcallImpl<STRLCAT>;
def strlcpy : RuntimeLibcallImpl<STRLCPY>;
def strlen : RuntimeLibcallImpl<STRLEN>;
def strncasecmp : RuntimeLibcallImpl<STRNCASECMP>;
def strncat : RuntimeLibcallImpl<STRNCAT>;
def strncmp : RuntimeLibcallImpl<STRNCMP>;
def strncpy : RuntimeLibcallImpl<STRNCPY>;
def strndup : RuntimeLibcallImpl<STRNDUP>;
def strnlen : RuntimeLibcallImpl<STRNLEN>;
def strpbrk : RuntimeLibcallImpl<STRPBRK>;
def strrchr : RuntimeLibcallImpl<STRRCHR>;
def strspn : RuntimeLibcallImpl<STRSPN>;
def strstr : RuntimeLibcallImpl<STRSTR>;
def strtod : RuntimeLibcallImpl<STRTOD>;
def strtof : RuntimeLibcallImpl<STRTOF>;
def strtok : RuntimeLibcallImpl<STRTOK>;
def strtok_r : RuntimeLibcallImpl<STRTOK_R>;
def strtol : RuntimeLibcallImpl<STRTOL>;
def strtold : RuntimeLibcallImpl<STRTOLD>;
def strtoll : RuntimeLibcallImpl<STRTOLL>;
def strtoul : RuntimeLibcallImpl<STRTOUL>;
def strtoull : RuntimeLibcallImpl<STRTOULL>;
def strxfrm : RuntimeLibcallImpl<STRXFRM>;
def system : RuntimeLibcallImpl<SYSTEM>;
def times : RuntimeLibcallImpl<TIMES>;
def tmpfile : RuntimeLibcallImpl<TMPFILE>;
def tmpfile64 : RuntimeLibcallImpl<TMPFILE64>;
def toascii : RuntimeLibcallImpl<TOASCII>;
def uname : RuntimeLibcallImpl<UNAME>;
def ungetc : RuntimeLibcallImpl<UNGETC>;
def unlink : RuntimeLibcallImpl<UNLINK>;
def unsetenv : RuntimeLibcallImpl<UNSETENV>;
def utime : RuntimeLibcallImpl<UTIME>;
def utimes : RuntimeLibcallImpl<UTIMES>;
def valloc : RuntimeLibcallImpl<VALLOC>;
def vec_calloc : RuntimeLibcallImpl<VEC_CALLOC>;
def vec_free : RuntimeLibcallImpl<VEC_FREE>;
def vec_malloc : RuntimeLibcallImpl<VEC_MALLOC>;
def vec_realloc : RuntimeLibcallImpl<VEC_REALLOC>;
def vfprintf : RuntimeLibcallImpl<VFPRINTF>;
def vfscanf : RuntimeLibcallImpl<VFSCANF>;
def vprintf : RuntimeLibcallImpl<VPRINTF>;
def vscanf : RuntimeLibcallImpl<VSCANF>;
def vsnprintf : RuntimeLibcallImpl<VSNPRINTF>;
def vsprintf : RuntimeLibcallImpl<VSPRINTF>;
def vsscanf : RuntimeLibcallImpl<VSSCANF>;
def wcslen : RuntimeLibcallImpl<WCSLEN>;
def write : RuntimeLibcallImpl<WRITE>;

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
defm sincosl : LibmLongDoubleLibCall;

// Exists in sun math library
def sincospif : RuntimeLibcallImpl<SINCOSPI_F32>;
def sincospi : RuntimeLibcallImpl<SINCOSPI_F64>;
defm sincospil : LibmLongDoubleLibCall;

// Exists on macOS
def __sincospif : RuntimeLibcallImpl<SINCOSPI_F32>;
def __sincospi : RuntimeLibcallImpl<SINCOSPI_F64>;

def bzero : RuntimeLibcallImpl<BZERO>;
def __bzero : RuntimeLibcallImpl<BZERO>;

// Exception handling
defset list<RuntimeLibcallImpl> DefaultExceptionHandlingLibcalls = {
  def _Unwind_Resume : RuntimeLibcallImpl<UNWIND_RESUME>;
  def __cxa_end_cleanup : RuntimeLibcallImpl<CXA_END_CLEANUP>;
}

defset list<RuntimeLibcallImpl> SjLjExceptionHandlingLibcalls = {
  def _Unwind_SjLj_Resume : RuntimeLibcallImpl<UNWIND_RESUME>;
  def _Unwind_SjLj_Register : RuntimeLibcallImpl<UNWIND_REGISTER>;
  def _Unwind_SjLj_Unregister : RuntimeLibcallImpl<UNWIND_UNREGISTER>;
}

// Only used on wasm?
def _Unwind_CallPersonality : RuntimeLibcallImpl<UNWIND_CALL_PERSONALITY>;

// Used on OpenBSD
def __stack_smash_handler : RuntimeLibcallImpl<STACK_SMASH_HANDLER>;

def __riscv_flush_icache : RuntimeLibcallImpl<RISCV_FLUSH_ICACHE>;

def _chkstk : RuntimeLibcallImpl<STACK_PROBE>;
def _alloca : RuntimeLibcallImpl<STACK_PROBE>;

def __chkstk : RuntimeLibcallImpl<STACK_PROBE>;
def ___chkstk_ms : RuntimeLibcallImpl<STACK_PROBE>;
def __chkstk_arm64ec : RuntimeLibcallImpl<STACK_PROBE, "#__chkstk_arm64ec">;

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

//--------------------------------------------------------------------
// Global variable references
//--------------------------------------------------------------------

def __stack_chk_guard : RuntimeLibcallImpl<STACK_CHECK_GUARD>;

// Name used on OpenBSD
def __guard_local : RuntimeLibcallImpl<STACK_CHECK_GUARD>;

// Name used with Windows MSVC
def __security_cookie : RuntimeLibcallImpl<STACK_CHECK_GUARD>;

// Name used on AIX
def __ssp_canary_word : RuntimeLibcallImpl<STACK_CHECK_GUARD>;

//===----------------------------------------------------------------------===//
// Common Libcall Sets
//===----------------------------------------------------------------------===//

defvar ExceptionModelCalls = (add
  LibcallImpls<(add __cxa_end_cleanup),
               ExceptionModelIsNotNone>,
  LibcallImpls<(add _Unwind_Resume),
               ExceptionModelHasUnwindResume>,
  LibcallImpls<(add SjLjExceptionHandlingLibcalls),
               ExceptionModelIsSjLj>
);

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

// FIXME: Ideally we would just use dags everywhere, but for the
// arm64ec case we need iterable lists so we can add the # prefix
defvar DefaultRuntimeLibcallImplsBaseList =
!listremove(
  !listremove(
    !listremove(AllDefaultRuntimeLibcallImpls, Int128RTLibcalls),
                DefaultRuntimeLibcallImpls_f80),
                DefaultRuntimeLibcallImpls_ppcf128);

defvar DefaultRuntimeLibcallImpls =
  (add DefaultRuntimeLibcallImplsBaseList,
       ExceptionModelCalls);

/// Default set of libcall impls for 32-bit architectures.
defvar DefaultLibcallImpls32 = (add DefaultRuntimeLibcallImpls);

/// Default set of libcall impls for 64-bit architectures.
defvar DefaultLibcallImpls64 = (add DefaultRuntimeLibcallImpls,
                                    Int128RTLibcalls);

// TODO: Guessing sincospi added at same time as sincos_stret
defvar DarwinSinCosStret = LibcallImpls<(add __sincosf_stret, __sincos_stret,
                                             __sincospif, __sincospi),
                                        darwinHasSinCosStret>;
defvar DarwinExp10 = LibcallImpls<(add __exp10f, __exp10), darwinHasExp10>;

defvar DarwinMemsetPattern = LibcallImpls<(add memset_pattern4,
                                               memset_pattern8,
                                               memset_pattern16),
                                               darwinHasMemsetPattern>;

defvar MacOSUnlockedIO = LibcallImpls<(add
  getc_unlocked, getchar_unlocked, putc_unlocked, putchar_unlocked),
  isMacOSX>;

defvar SecurityCheckCookieIfWinMSVC =
    LibcallImpls<(add __security_check_cookie, __security_cookie),
                 isWindowsMSVCOrItaniumEnvironment>;

defvar LibmHasSinCosF32 = LibcallImpls<(add sincosf), hasSinCos>;
defvar LibmHasSinCosF64 =  LibcallImpls<(add sincos), hasSinCos>;
defvar LibmHasSinCosF80 = LibcallImpls<(add sincosl_f80), hasSinCos>;
defvar LibmHasSinCosF128 = LibcallImpls<(add sincosl_f128), hasSinCos>;
defvar LibmHasSinCosPPCF128 = LibcallImpls<(add sincosl_ppcf128), hasSinCos>;

defvar LibmHasExp10F32 = LibcallImpls<(add exp10f), hasExp10>;
defvar LibmHasExp10F64 = LibcallImpls<(add exp10), hasExp10>;
defvar LibmHasExp10F80 = LibcallImpls<(add exp10l_f80), hasExp10>;
defvar LibmHasExp10F128 = LibcallImpls<(add exp10l_f128), hasExp10>;
defvar LibmHasExp10PPCF128 = LibcallImpls<(add exp10l_ppcf128), hasExp10>;

defvar DefaultLibmExp10 = [
  LibmHasExp10F32, LibmHasExp10F64, LibmHasExp10F128
];


defvar WindowsMathRemovals = [
  ldexpf, ldexpl_f80, ldexpl_f128, ldexpl_ppcf128,
  frexpf, frexpl_f80, frexpl_f128, frexpl_ppcf128
];

defvar MostPowI = !listremove(PowiLibcallImpls, [__powitf2_f128, __powitf2_ppc128]);
defvar WindowsExclusions = !listconcat(WindowsMathRemovals, MostPowI);

// Targets which support windows should start with these as a base and
// add in calls for other OSes
defvar WinDefaultLibcallImplsBaseList =
  !listremove(DefaultRuntimeLibcallImplsBaseList,
              WindowsExclusions);

defvar WinDefaultLibcallImpls = (add WinDefaultLibcallImplsBaseList,
                                     ExceptionModelCalls);

defvar LibmHasFrexpF32 = LibcallImpls<(add frexpf), isNotOSWindowsOrIsCygwinMinGW>;
defvar LibmHasLdexpF32 = LibcallImpls<(add ldexpf), isNotOSWindowsOrIsCygwinMinGW>;

defvar LibmHasFrexpF80 = LibcallImpls<(add frexpl_f80), isNotOSWindowsOrIsCygwinMinGW>;
defvar LibmHasLdexpF80 = LibcallImpls<(add ldexpl_f80), isNotOSWindowsOrIsCygwinMinGW>;

defvar LibmHasFrexpF128 = LibcallImpls<(add frexpl_f128), isNotOSWindowsOrIsCygwinMinGW>;
defvar LibmHasLdexpF128 = LibcallImpls<(add ldexpl_f128), isNotOSWindowsOrIsCygwinMinGW>;

defvar has__stack_chk_fail = LibcallImpls<(add __stack_chk_fail), hasStackChkFail>;
defvar has__stack_chk_guard =
    LibcallImpls<(add __stack_chk_guard), hasStackChkFail>;
defvar has__stack_smash_handler = LibcallImpls<(add __stack_smash_handler), isOSOpenBSD>;
defvar has___guard_local = LibcallImpls<(add __guard_local), isOSOpenBSD>;

defvar DefaultStackProtector = (add has__stack_chk_fail, has__stack_chk_guard,
    has__stack_smash_handler, has___guard_local);

defvar LibcSafestackPointerAddress =
    LibcallImpls<(add __safestack_pointer_address), isAndroid>;

// compiler-rt sources suggest this is only available if
// (TT.isOSLinux() || TT.isOSFreeBSD() || TT.isOSNetBSD() ||
// TT.isOSSolaris()), but tests exist for other os targets.
defvar DefaultSafeStackGlobals = (add __safestack_unsafe_stack_ptr);

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
  def __arm_sc_memchr : RuntimeLibcallImpl<SC_MEMCHR>;
} // End AArch64LibcallImpls

def __arm_sme_state : RuntimeLibcallImpl<SMEABI_SME_STATE>;
def __arm_tpidr2_save : RuntimeLibcallImpl<SMEABI_TPIDR2_SAVE>;
def __arm_za_disable : RuntimeLibcallImpl<SMEABI_ZA_DISABLE>;
def __arm_tpidr2_restore : RuntimeLibcallImpl<SMEABI_TPIDR2_RESTORE>;
def __arm_get_current_vg : RuntimeLibcallImpl<SMEABI_GET_CURRENT_VG>;
def __arm_sme_state_size : RuntimeLibcallImpl<SMEABI_SME_STATE_SIZE>;
def __arm_sme_save : RuntimeLibcallImpl<SMEABI_SME_SAVE>;
def __arm_sme_restore : RuntimeLibcallImpl<SMEABI_SME_RESTORE>;

def SMEABI_LibCalls_PreserveMost_From_X0 : LibcallsWithCC<(add
  __arm_tpidr2_save,
  __arm_za_disable,
  __arm_tpidr2_restore),
  SMEABI_PreserveMost_From_X0>;

def SMEABI_LibCalls_PreserveMost_From_X1 : LibcallsWithCC<(add
  __arm_get_current_vg,
  __arm_sme_state_size,
  __arm_sme_save,
  __arm_sme_restore),
  SMEABI_PreserveMost_From_X1>;

def SMEABI_LibCalls_PreserveMost_From_X2 : LibcallsWithCC<(add
  __arm_sme_state),
  SMEABI_PreserveMost_From_X2>;

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
       DarwinExp10, DarwinSinCosStret, DarwinMemsetPattern,
       MacOSUnlockedIO,
       LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
       DefaultLibmExp10,
       DefaultStackProtector,
       SecurityCheckCookieIfWinMSVC,
       LibcallImpls<(add __chkstk), isOSWindows>,
       SMEABI_LibCalls_PreserveMost_From_X0,
       SMEABI_LibCalls_PreserveMost_From_X1,
       SMEABI_LibCalls_PreserveMost_From_X2,
       LibcSafestackPointerAddress)
>;

// Prepend a # to every name
defset list<RuntimeLibcallImpl> WinArm64ECDefaultRuntimeLibcallImpls = {
  foreach libcall = WinDefaultLibcallImplsBaseList in {
    def arm64ec_#libcall : DuplicateLibcallImplWithPrefix<libcall, "#">;
  }

  foreach libcall = AArch64LibcallImpls in {
    def arm64ec_#libcall : DuplicateLibcallImplWithPrefix<libcall, "#">;
  }
}

def arm64ec___stack_chk_fail : DuplicateLibcallImplWithPrefix<__stack_chk_fail, "#">;

defset list<RuntimeLibcallImpl> WinArm64ECDefaultExceptionHandlingLibcalls = {
  foreach libcall = DefaultExceptionHandlingLibcalls in {
    def arm64ec_#libcall : DuplicateLibcallImplWithPrefix<libcall, "#">;
  }
}

defset list<RuntimeLibcallImpl> WinArm64ECSjLjExceptionHandlingLibcalls = {
  foreach libcall = SjLjExceptionHandlingLibcalls in {
    def arm64ec_#libcall : DuplicateLibcallImplWithPrefix<libcall, "#">;
  }
}

defvar ExceptionModelCallsArm64EC = (add
  LibcallImpls<(add arm64ec___cxa_end_cleanup),
               ExceptionModelIsNotNone>,
  LibcallImpls<(add arm64ec__Unwind_Resume),
               ExceptionModelHasUnwindResume>,
  LibcallImpls<(add WinArm64ECSjLjExceptionHandlingLibcalls),
               ExceptionModelIsSjLj>
);

def WindowsARM64ECSystemLibrary
    : SystemRuntimeLibrary<isWindowsArm64EC,
                           (add WinArm64ECDefaultRuntimeLibcallImpls,
                                AvailableIf<arm64ec___stack_chk_fail, hasStackChkFail>,
                                AvailableIf<__stack_chk_guard, hasStackChkFail>,
                                LibcallImpls<(add __security_check_cookie_arm64ec,
                                                  __security_cookie),
                                              isWindowsMSVCEnvironment>,
                                __chkstk_arm64ec,
                                ExceptionModelCallsArm64EC)>;

//===----------------------------------------------------------------------===//
// AMDGPU Runtime Libcalls
//===----------------------------------------------------------------------===//

def isAMDGPU : RuntimeLibcallPredicate<"TT.isAMDGPU()">;

// No calls.
def AMDGPUSystemLibrary : SystemRuntimeLibrary<isAMDGPU, (add)>;

//===----------------------------------------------------------------------===//
// ARM Runtime Libcalls
//===----------------------------------------------------------------------===//

def isTargetAEABIAndAAPCS_ABI : RuntimeLibcallPredicate<
  [{TT.isTargetAEABI() && isAAPCS_ABI(TT, ABIName)}]>;

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

// Unaligned memory access helper functions
// RTABI chapter 5.3.3
def __aeabi_uread4  : RuntimeLibcallImpl<AEABI_UREAD4>;  // CallingConv::ARM_AAPCS
def __aeabi_uread8  : RuntimeLibcallImpl<AEABI_UREAD8>;  // CallingConv::ARM_AAPCS
def __aeabi_uwrite4 : RuntimeLibcallImpl<AEABI_UWRITE4>; // CallingConv::ARM_AAPCS
def __aeabi_uwrite8 : RuntimeLibcallImpl<AEABI_UWRITE8>; // CallingConv::ARM_AAPCS

defvar AEABIOverrides = [
  __eqsf2, __eqdf2,
  __nesf2, __nedf2,
  __ltsf2, __ltdf2,
  __lesf2, __ledf2,
  __gesf2, __gedf2,
  __gtsf2, __gtdf2,
  __unordsf2, __unorddf2,

  __addsf3, __adddf3,
  __divsf3, __divdf3,
  __mulsf3, __muldf3,
  __subsf3, __subdf3,

  __fixdfsi, __fixunsdfsi,
  __fixdfdi, __fixunsdfdi,
  __fixsfsi, __fixunssfsi,
  __fixsfdi, __fixunssfdi,

  __floatsidf, __floatunsidf,
  __floatdidf, __floatundidf,
  __floatsisf, __floatunsisf,
  __floatdisf, __floatundisf,

  __muldi3, __ashldi3,
  __lshrdi3, __ashrdi3,

  __divsi3, __udivsi3

  // Half conversion cases are a mess and handled separately.
  //  __truncdfsf2, __truncdfhf2,
  //  __extendsfdf2,
  //  __truncsfhf2, __extendhfsf2
];

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
def __aeabi_f2d : RuntimeLibcallImpl<FPEXT_F32_F64>; // CallingConv::ARM_AAPCS

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
def __aeabi_h2f : RuntimeLibcallImpl<FPEXT_F16_F32>; // CallingConv::ARM_AAPCS

// !isTargetMachO()
def __gnu_f2h_ieee : RuntimeLibcallImpl<FPROUND_F32_F16>;
def __gnu_h2f_ieee : RuntimeLibcallImpl<FPEXT_F16_F32>;

// The half <-> float conversion functions are always soft-float on
// non-watchos platforms, but are needed for some targets which use a
// hard-float calling convention by default.
def ARMHalfConvertLibcallCallingConv : LibcallCallingConv<
  [{TT.isWatchABI() ? static_cast<CallingConv::ID>(DefaultCC) :
    static_cast<CallingConv::ID>(isAAPCS_ABI(TT, ABIName) ? CallingConv::ARM_AAPCS : CallingConv::ARM_APCS)}]
>;

def ARMLibgccHalfConvertCalls :
  LibcallImpls<(add __truncsfhf2, __extendhfsf2),
    RuntimeLibcallPredicate<[{!TT.isTargetAEABI() && TT.isOSBinFormatMachO()}]>> {
  let CallingConv = ARMHalfConvertLibcallCallingConv;
}

// FIXME: These conditions are probably bugged. We're using the
// default libgcc call when the other cases are replaced.
def ARMDoubleToHalfCalls :
  LibcallImpls<(add __truncdfhf2),
    RuntimeLibcallPredicate<[{!TT.isTargetAEABI()}]>> {
  let CallingConv = ARMHalfConvertLibcallCallingConv;
}

// In EABI, these functions have an __aeabi_ prefix, but in GNUEABI
// they have a __gnu_ prefix (which is the default).
def EABIHalfConvertCalls : LibcallImpls<(add __aeabi_f2h, __aeabi_h2f),
                                        isTargetAEABIAndAAPCS_ABI> {
  let CallingConv = ARM_AAPCS;
}

def GNUEABIHalfConvertCalls :
  LibcallImpls<(add __gnu_f2h_ieee, __gnu_h2f_ieee),
    RuntimeLibcallPredicate<[{!TT.isOSBinFormatMachO() &&
                              !TT.isTargetAEABI()}]>> {
  let CallingConv = ARMHalfConvertLibcallCallingConv;
}

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
  RuntimeLibcallPredicate<[{hasAEABILibcalls(TT)}]>> {
  let CallingConv = ARM_AAPCS;
}

def AEABICalls : LibcallImpls<
  (add
      // Double-precision floating-point arithmetic helper functions
      // RTABI chapter 4.1.2, Table 2
      __aeabi_dadd,
      __aeabi_ddiv,
      __aeabi_dmul,
      __aeabi_dsub,

      // Double-precision floating-point comparison helper functions
      // RTABI chapter 4.1.2, Table 3
      __aeabi_dcmpeq__oeq,
      __aeabi_dcmpeq__une,
      __aeabi_dcmplt,
      __aeabi_dcmple,
      __aeabi_dcmpge,
      __aeabi_dcmpgt,
       __aeabi_dcmpun,

      // Single-precision floating-point arithmetic helper functions
      // RTABI chapter 4.1.2, Table 4
      __aeabi_fadd,
      __aeabi_fdiv,
      __aeabi_fmul,
      __aeabi_fsub,

      // Single-precision floating-point comparison helper functions
      // RTABI chapter 4.1.2, Table 5
      __aeabi_fcmpeq__oeq,
      __aeabi_fcmpeq__une,
      __aeabi_fcmplt,
      __aeabi_fcmple,
      __aeabi_fcmpge,
      __aeabi_fcmpgt,
      __aeabi_fcmpun,

      // Floating-point to integer conversions.
      // RTABI chapter 4.1.2, Table 6
      __aeabi_d2iz,
      __aeabi_d2uiz,
      __aeabi_d2lz,
      __aeabi_d2ulz,
      __aeabi_f2iz,
      __aeabi_f2uiz,
      __aeabi_f2lz,
      __aeabi_f2ulz,

      // Conversions between floating types.
      // RTABI chapter 4.1.2, Table 7
      __aeabi_d2f,
      __aeabi_f2d,
      //__aeabi_h2f added separately
      //__aeabi_f2h added separately
      __aeabi_d2h,

      // Integer to floating-point conversions.
      // RTABI chapter 4.1.2, Table 8
      __aeabi_i2d,
      __aeabi_ui2d,
      __aeabi_l2d,
      __aeabi_ul2d,
      __aeabi_i2f,
      __aeabi_ui2f,
      __aeabi_l2f,
      __aeabi_ul2f,

      // Long long helper functions
      // RTABI chapter 4.2, Table 9
      __aeabi_lmul,
      __aeabi_llsl,
      __aeabi_llsr,
      __aeabi_lasr,

      // Integer division functions
      // RTABI chapter 4.3.1
      __aeabi_idiv,
      __aeabi_uidiv),
  RuntimeLibcallPredicate<[{hasAEABILibcalls(TT) && isAAPCS_ABI(TT, ABIName)}]>> {
  let CallingConv = ARM_AAPCS;
}

// EABI dependent RTLIB, Memory operations
// RTABI chapter 4.3.4
def AEABI45MemCalls : LibcallImpls<
  (add __aeabi_memcpy,  __aeabi_memcpy4, __aeabi_memcpy8,
       __aeabi_memmove, __aeabi_memmove4, __aeabi_memmove8,
       __aeabi_memset, __aeabi_memset4, __aeabi_memset8,
       __aeabi_memclr, __aeabi_memclr4, __aeabi_memclr8),
  RuntimeLibcallPredicate<[{(EABIVersion == EABI::EABI4 ||
                             EABIVersion == EABI::EABI5) &&
                             hasAEABILibcalls(TT) && isAAPCS_ABI(TT, ABIName)}]>> {
  let CallingConv = ARM_AAPCS;
}

// Unaligned Memory read/write operations
// RTABI chapter 5.3.3
def AEABIUnalignedMemCalls : LibcallImpls<
  (add __aeabi_uread4,  __aeabi_uread8,
       __aeabi_uwrite4, __aeabi_uwrite8),
  RuntimeLibcallPredicate<[{hasAEABILibcalls(TT) && isAAPCS_ABI(TT, ABIName)}]>> {
  let CallingConv = ARM_AAPCS;
}

def isARMOrThumb : RuntimeLibcallPredicate<"TT.isARM() || TT.isThumb()">;

def ARMSystemLibrary
    : SystemRuntimeLibrary<isARMOrThumb,
      (add (sub WinDefaultLibcallImpls, ARMLibgccHalfConvertCalls,
                                        GNUEABIHalfConvertCalls,
                                        ARMDoubleToHalfCalls,
                                        AEABIOverrides),
           LibcallImpls<(add __powisf2, __powidf2), isNotOSMSVCRT>,
           LibmHasFrexpF32, LibmHasLdexpF32,
           LibmHasFrexpF128, LibmHasLdexpF128,
           WindowARMDivRemCalls,
           WindowARMFPIntCasts,
           SecurityCheckCookieIfWinMSVC,
           LibcallImpls<(add __chkstk), isOSWindows>,
           AEABIDivRemCalls,
           DarwinSinCosStret, DarwinExp10, DarwinMemsetPattern,
           LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
           DefaultLibmExp10,
           LibcSafestackPointerAddress,

           AEABICalls,
           AEABI45MemCalls,
           AEABIUnalignedMemCalls,
           ARMLibgccHalfConvertCalls,
           EABIHalfConvertCalls,
           GNUEABIHalfConvertCalls,
           ARMDoubleToHalfCalls,

           LibcallImpls<(add AEABIOverrides),
             RuntimeLibcallPredicate<[{
               (!hasAEABILibcalls(TT) || !isAAPCS_ABI(TT, ABIName)) &&
               !TT.isOSWindows()
            }]>>,
           // Use divmod compiler-rt calls for iOS 5.0 and later.
           LibcallImpls<(add __divmodsi4, __udivmodsi4),
                        RuntimeLibcallPredicate<[{TT.isOSBinFormatMachO() &&
                                                  (!TT.isiOS() || !TT.isOSVersionLT(5, 0))}]>>,
           DefaultStackProtector,
           DefaultSafeStackGlobals)> {
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

def hexagon_memcpy_forward_vp4cp4n2 : RuntimeLibcallImpl<HEXAGON_VOLATILE_MEMCPY>;
}

def isHexagon : RuntimeLibcallPredicate<"TT.getArch() == Triple::hexagon">;

def HexagonSystemLibrary
    : SystemRuntimeLibrary<isHexagon,
    (add (sub DefaultLibcallImpls32,
    __adddf3, __divsf3, __udivsi3, __udivdi3,
    __umoddi3, __divdf3, __muldf3, __divsi3, __subdf3, sqrtf,
    __divdi3, __umodsi3, __moddi3, __modsi3), HexagonLibcalls,
    LibmHasSinCosF32, LibmHasSinCosF64, LibmHasSinCosF128,
    exp10f, exp10, exp10l_f128, __stack_chk_fail, __stack_chk_guard)>;

//===----------------------------------------------------------------------===//
// Lanai Runtime Libcalls
//===----------------------------------------------------------------------===//

def isLanai : RuntimeLibcallPredicate<"TT.getArch() == Triple::lanai">;

// Use fast calling convention for library functions.
def LanaiSystemLibrary
    : SystemRuntimeLibrary<isLanai, (add DefaultRuntimeLibcallImpls,
                                        __stack_chk_fail, __stack_chk_guard)> {
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
  __floatundidf, __gtdf2, __eqdf2, __gedf2, __ltdf2, __ledf2,
  __adddf3, __divdf3, __divdi3, __moddi3,
  __muldf3, __subdf3, __udivdi3, __umoddi3
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

      // May not be used depending on subtarget
      LibcallsWithCC<(add __mspabi_mpyll), MSP430_BUILTIN>,

      // Bitwise Operations - EABI Table 10
      // TODO: __mspabi_[srli/srai/slli] ARE implemented in libgcc
      __mspabi_srll,
      __mspabi_sral,
      __mspabi_slll,
      // __mspabi_[srlll/srall/sllll/rlli/rlll] are NOT implemented in libgcc

      __stack_chk_fail,
      __stack_chk_guard
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
  def ___memccpy64 : RuntimeLibcallImpl<MEMCCPY>;
  def ___memcmp64 : RuntimeLibcallImpl<MEMCMP>;
  def ___memmove64 : RuntimeLibcallImpl<MEMMOVE>;
  def ___memset64 : RuntimeLibcallImpl<MEMSET>;
  def ___bzero64 : RuntimeLibcallImpl<BZERO>;
  def ___strcmp64 : RuntimeLibcallImpl<STRCMP>;
  def ___strcpy64 : RuntimeLibcallImpl<STRCPY>;
  def ___strlen64 : RuntimeLibcallImpl<STRLEN>;
  def ___strstr64 : RuntimeLibcallImpl<STRSTR>;
}

defset list<RuntimeLibcallImpl> PPC32AIXCallList = {
  def ___memccpy : RuntimeLibcallImpl<MEMCCPY>;
  def ___memcmp : RuntimeLibcallImpl<MEMCMP>;
  def ___memmove : RuntimeLibcallImpl<MEMMOVE>;
  def ___memset : RuntimeLibcallImpl<MEMSET>;
  def ___bzero : RuntimeLibcallImpl<BZERO>;
  def ___strcmp : RuntimeLibcallImpl<STRCMP>;
  def ___strcpy : RuntimeLibcallImpl<STRCPY>;
  def ___strlen : RuntimeLibcallImpl<STRLEN>;
  def ___strstr : RuntimeLibcallImpl<STRSTR>;
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
           has__stack_smash_handler,
           has___guard_local,
           AvailableIf<__ssp_canary_word, isAIX>,
           AvailableIf<vec_calloc, isAIX>,
           AvailableIf<vec_malloc, isAIX>,
           AvailableIf<vec_realloc, isAIX>,
           AvailableIf<vec_free, isAIX>,
           AvailableIf<__stack_chk_fail, isNotOSOpenBSD>,
           AvailableIf<__stack_chk_guard, isNotOSAIXAndNotOSOpenBSD>)>;

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
           DefaultStackProtector,
           DefaultSafeStackGlobals)>;

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
       has__stack_chk_fail,
       has__stack_smash_handler,
       has___guard_local,
       AvailableIf<__stack_chk_guard, isNotOSLinuxAndNotOSOpenBSD>)
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

// FIXME: Should have utility function to filter by known provider.
defvar WindowsDivRemMulLibcallOverrides = [
  __divdi3, __udivdi3, __moddi3, __umoddi3, __muldi3
];

//===----------------------------------------------------------------------===//
// X86 Runtime Libcalls
//===----------------------------------------------------------------------===//

def isX86_32 : RuntimeLibcallPredicate<"TT.getArch() == Triple::x86">;
def isX86_64 : RuntimeLibcallPredicate<"TT.getArch() == Triple::x86_64">;
def isX86 : RuntimeLibcallPredicate<"TT.isX86()">;

def isCygwinMinGW64 : RuntimeLibcallPredicate<
  [{TT.isOSCygMing() && TT.getArch() == Triple::x86_64}]>;
def isCygwinMinGW32 : RuntimeLibcallPredicate<
  [{TT.isOSCygMing() && TT.getArch() == Triple::x86}]>;

def isWin32NotCygMing : RuntimeLibcallPredicate<
  [{TT.getArch() == Triple::x86 &&
     (TT.isOSWindows() || TT.isUEFI()) && !TT.isOSCygMing()}]>;
def isWin64NotCygMing : RuntimeLibcallPredicate<
  [{TT.getArch() == Triple::x86_64 &&
    (TT.isOSWindows() || TT.isUEFI()) && !TT.isOSCygMing()}]>;


// Some darwins have an optimized __bzero/bzero function.
def darwinHas__bzero : RuntimeLibcallPredicate<"TT.isMacOSX() && !TT.isMacOSXVersionLT(10, 6)">;

// FIXME: This is has ldexpl/frexpl plus use f128 for long double.
def hasExpFrexplLdexplF128
  : RuntimeLibcallPredicate<[{(!TT.isOSWindows() || TT.isOSCygMing()) && !TT.isGNUEnvironment()}]>;

// Use the f128 variants of math functions on x86
defvar X86_F128_Libcalls = LibcallImpls<(add LibmF128Libcalls, LibmF128FiniteLibcalls), isGNUEnvironment>;

defvar SinCosF32F64Libcalls = LibcallImpls<(add sincosf, sincos), hasSinCos_f32_f64>;

defvar MemChkLibcalls = [__memcpy_chk, __memset_chk, __memmove_chk];

defvar X86CommonLibcalls =
  (add (sub WinDefaultLibcallImpls, WindowsDivRemMulLibcallOverrides, MemChkLibcalls),
       DarwinSinCosStret, DarwinExp10, DarwinMemsetPattern, MacOSUnlockedIO,
       X86_F128_Libcalls,
       LibmHasSinCosF80, // FIXME: Depends on long double
       SinCosF32F64Libcalls,
       LibcallImpls<(add __bzero), darwinHas__bzero>,
       LibmHasFrexpF32, LibmHasLdexpF32,
       LibmHasFrexpF80, LibmHasLdexpF80,
       LibcallImpls<(add frexpl_f128, ldexpl_f128, exp10l_f128), hasExpFrexplLdexplF128>,
       DefaultRuntimeLibcallImpls_f80,
       LibmHasExp10F32, LibmHasExp10F64, LibmHasExp10F80,
       LibcallImpls<(add MostPowI), isNotOSMSVCRT>,
       SecurityCheckCookieIfWinMSVC,
       // FIXME: MSVCRT doesn't have powi. The f128 case is added as a
       // hack for one test relying on it.
       __powitf2_f128,
       DefaultStackProtector,
       DefaultSafeStackGlobals,
       LibcallImpls<(add MemChkLibcalls), isNotPS>,
       LibcallImpls<(add ___chkstk_ms), isCygwinMinGW64>,
       LibcallImpls<(add __chkstk), isWin64NotCygMing>,
       LibcallImpls<(add _alloca), isCygwinMinGW32>,
       LibcallImpls<(add _chkstk), isWin32NotCygMing>,
       LibcSafestackPointerAddress
);

defvar Windows32DivRemMulCalls =
  LibcallsWithCC<(add WindowsDivRemMulLibcalls), X86_STDCALL,
  RuntimeLibcallPredicate<"TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment()">>;

defvar NotWindows32DivRemMulCalls =
  LibcallImpls<(add WindowsDivRemMulLibcallOverrides),
    RuntimeLibcallPredicate<"!TT.isWindowsMSVCEnvironment() && !TT.isWindowsItaniumEnvironment()">>;

def X86_32SystemLibrary
    : SystemRuntimeLibrary<isX86_32,
      (add X86CommonLibcalls,
             NotWindows32DivRemMulCalls,
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
         iprintf, siprintf, fiprintf,
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
def isOSEmscripten : RuntimeLibcallPredicate<[{TT.isOSEmscripten()}]>;

// Define the emscripten name for return address helper.
// TODO: when implementing other Wasm backends, make this generic or only do
// this on emscripten depending on what they end up doing.
def WasmSystemLibrary
    : SystemRuntimeLibrary<isWasm,
      (add DefaultRuntimeLibcallImpls, Int128RTLibcalls,
           CompilerRTOnlyInt64Libcalls, CompilerRTOnlyInt128Libcalls,
           exp10f, exp10,
           _Unwind_CallPersonality,
           emscripten_return_address,
           LibcallImpls<(add __small_printf,
                             __small_sprintf,
                             __small_fprintf), isOSEmscripten>,
           __stack_chk_fail, __stack_chk_guard)>;

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
         DefaultStackProtector, DefaultSafeStackGlobals
)>;

//===----------------------------------------------------------------------===//
// Vector math libraries
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Accelerate framework functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> ACCELERATE_VECFUNCS = {
  def vacosf : RuntimeLibcallImpl<ACOS_V4F32>;
  def vacoshf : RuntimeLibcallImpl<ACOSH_V4F32>;
  def vasinf : RuntimeLibcallImpl<ASIN_V4F32>;
  def vasinhf : RuntimeLibcallImpl<ASINH_V4F32>;
  def vatan2f : RuntimeLibcallImpl<ATAN2_V4F32>;
  def vatanf : RuntimeLibcallImpl<ATAN_V4F32>;
  def vatanhf : RuntimeLibcallImpl<ATANH_V4F32>;
  def vceilf : RuntimeLibcallImpl<CEIL_V4F32>;
  def vcosf : RuntimeLibcallImpl<COS_V4F32>;
  def vcoshf : RuntimeLibcallImpl<COSH_V4F32>;
  def vexpf : RuntimeLibcallImpl<EXP_V4F32>;
  def vexpm1f : RuntimeLibcallImpl<EXPM1_V4F32>;
  def vfabsf : RuntimeLibcallImpl<FABS_V4F32>;
  def vfloorf : RuntimeLibcallImpl<FLOOR_V4F32>;
  def vlog10f : RuntimeLibcallImpl<LOG10_V4F32>;
  def vlog1pf : RuntimeLibcallImpl<LOG1P_V4F32>;
  def vlogbf : RuntimeLibcallImpl<LOGB_V4F32>;
  def vlogf : RuntimeLibcallImpl<LOG_V4F32>;
  def vsinf : RuntimeLibcallImpl<SIN_V4F32>;
  def vsinhf : RuntimeLibcallImpl<SINH_V4F32>;
  def vsqrtf : RuntimeLibcallImpl<SQRT_V4F32>;
  def vtanf : RuntimeLibcallImpl<TAN_V4F32>;
  def vtanhf : RuntimeLibcallImpl<TANH_V4F32>;
}

//===----------------------------------------------------------------------===//
// Darwin_libsystem_m vector functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> DARWIN_LIBSYSTEM_M_VECFUNCS = {
  def _simd_acos_d2 : RuntimeLibcallImpl<ACOS_V2F64>;
  def _simd_acos_f4 : RuntimeLibcallImpl<ACOS_V4F32>;
  def _simd_acosh_d2 : RuntimeLibcallImpl<ACOSH_V2F64>;
  def _simd_acosh_f4 : RuntimeLibcallImpl<ACOSH_V4F32>;
  def _simd_asin_d2 : RuntimeLibcallImpl<ASIN_V2F64>;
  def _simd_asin_f4 : RuntimeLibcallImpl<ASIN_V4F32>;
  def _simd_asinh_d2 : RuntimeLibcallImpl<ASINH_V2F64>;
  def _simd_asinh_f4 : RuntimeLibcallImpl<ASINH_V4F32>;
  def _simd_atan2_d2 : RuntimeLibcallImpl<ATAN2_V2F64>;
  def _simd_atan2_f4 : RuntimeLibcallImpl<ATAN2_V4F32>;
  def _simd_atan_d2 : RuntimeLibcallImpl<ATAN_V2F64>;
  def _simd_atan_f4 : RuntimeLibcallImpl<ATAN_V4F32>;
  def _simd_atanh_d2 : RuntimeLibcallImpl<ATANH_V2F64>;
  def _simd_atanh_f4 : RuntimeLibcallImpl<ATANH_V4F32>;
  def _simd_cbrt_d2 : RuntimeLibcallImpl<CBRT_V2F64>;
  def _simd_cbrt_f4 : RuntimeLibcallImpl<CBRT_V4F32>;
  def _simd_cos_d2 : RuntimeLibcallImpl<COS_V2F64>;
  def _simd_cos_f4 : RuntimeLibcallImpl<COS_V4F32>;
  def _simd_cosh_d2 : RuntimeLibcallImpl<COSH_V2F64>;
  def _simd_cosh_f4 : RuntimeLibcallImpl<COSH_V4F32>;
  def _simd_erf_d2 : RuntimeLibcallImpl<ERF_V2F64>;
  def _simd_erf_f4 : RuntimeLibcallImpl<ERF_V4F32>;
  def _simd_exp_d2 : RuntimeLibcallImpl<EXP_V2F64>;
  def _simd_exp_f4 : RuntimeLibcallImpl<EXP_V4F32>;
  def _simd_pow_d2 : RuntimeLibcallImpl<POW_V2F64>;
  def _simd_pow_f4 : RuntimeLibcallImpl<POW_V4F32>;
  def _simd_sin_d2 : RuntimeLibcallImpl<SIN_V2F64>;
  def _simd_sin_f4 : RuntimeLibcallImpl<SIN_V4F32>;
  def _simd_sinh_d2 : RuntimeLibcallImpl<SINH_V2F64>;
  def _simd_sinh_f4 : RuntimeLibcallImpl<SINH_V4F32>;
  def _simd_tan_d2 : RuntimeLibcallImpl<TAN_V2F64>;
  def _simd_tan_f4 : RuntimeLibcallImpl<TAN_V4F32>;
  def _simd_tanh_d2 : RuntimeLibcallImpl<TANH_V2F64>;
  def _simd_tanh_f4 : RuntimeLibcallImpl<TANH_V4F32>;
}

//===----------------------------------------------------------------------===//
// GLIBC Vector Math library LIBMVEC functions
//===----------------------------------------------------------------------===//

defvar LIBMVECPrefix = "LIBMVEC_";

class LibmvecLibcall<RuntimeLibcall P>
    : RuntimeLibcallImpl<P, !substr(NAME, !size(LIBMVECPrefix))>;

defset list<RuntimeLibcallImpl> LIBMVEC_X86_VECFUNCS = {
  def LIBMVEC__ZGVbN2v___exp_finite : LibmvecLibcall<EXP_FINITE_V2F64>;
  def LIBMVEC__ZGVbN2v___log_finite : LibmvecLibcall<LOG_FINITE_V2F64>;
  def LIBMVEC__ZGVbN2v_cos : LibmvecLibcall<COS_V2F64>;
  def LIBMVEC__ZGVbN2v_exp : LibmvecLibcall<EXP_V2F64>;
  def LIBMVEC__ZGVbN2v_log : LibmvecLibcall<LOG_V2F64>;
  def LIBMVEC__ZGVbN2v_sin : LibmvecLibcall<SIN_V2F64>;
  def LIBMVEC__ZGVbN2v_tan : LibmvecLibcall<TAN_V2F64>;
  def LIBMVEC__ZGVbN2vv___pow_finite : LibmvecLibcall<POW_FINITE_V2F64>;
  def LIBMVEC__ZGVbN2vv_pow : LibmvecLibcall<POW_V2F64>;
  def LIBMVEC__ZGVbN4v___expf_finite : LibmvecLibcall<EXP_FINITE_V4F64>;
  def LIBMVEC__ZGVbN4v___logf_finite : LibmvecLibcall<LOG_FINITE_V4F64>;
  def LIBMVEC__ZGVbN4v_cosf : LibmvecLibcall<COS_V4F32>;
  def LIBMVEC__ZGVbN4v_expf : LibmvecLibcall<EXP_V4F32>;
  def LIBMVEC__ZGVbN4v_logf : LibmvecLibcall<LOG_V4F32>;
  def LIBMVEC__ZGVbN4v_sinf : LibmvecLibcall<SIN_V4F32>;
  def LIBMVEC__ZGVbN4v_tanf : LibmvecLibcall<TAN_V4F32>;
  def LIBMVEC__ZGVbN4vv___powf_finite : LibmvecLibcall<POW_FINITE_V4F64>;
  def LIBMVEC__ZGVbN4vv_powf : LibmvecLibcall<POW_V4F32>;
  def LIBMVEC__ZGVdN4v___exp_finite : LibmvecLibcall<EXP_FINITE_V4F64>;
  def LIBMVEC__ZGVdN4v___log_finite : LibmvecLibcall<LOG_FINITE_V4F64>;
  def LIBMVEC__ZGVdN4v_cos : LibmvecLibcall<COS_V4F64>;
  def LIBMVEC__ZGVdN4v_exp : LibmvecLibcall<EXP_V4F64>;
  def LIBMVEC__ZGVdN4v_log : LibmvecLibcall<LOG_V4F64>;
  def LIBMVEC__ZGVdN4v_sin : LibmvecLibcall<SIN_V4F64>;
  def LIBMVEC__ZGVdN4v_tan : LibmvecLibcall<TAN_V4F64>;
  def LIBMVEC__ZGVdN4vv___pow_finite : LibmvecLibcall<POW_FINITE_V4F64>;
  def LIBMVEC__ZGVdN4vv_pow : LibmvecLibcall<POW_V4F64>;
  def LIBMVEC__ZGVdN8v___expf_finite : LibmvecLibcall<EXP_FINITE_V8F64>;
  def LIBMVEC__ZGVdN8v___logf_finite : LibmvecLibcall<LOG_FINITE_V8F64>;
  def LIBMVEC__ZGVdN8v_cosf : LibmvecLibcall<COS_V8F32>;
  def LIBMVEC__ZGVdN8v_expf : LibmvecLibcall<EXP_V8F32>;
  def LIBMVEC__ZGVdN8v_logf : LibmvecLibcall<LOG_V8F32>;
  def LIBMVEC__ZGVdN8v_sinf : LibmvecLibcall<SIN_V8F32>;
  def LIBMVEC__ZGVdN8v_tanf : LibmvecLibcall<TAN_V8F32>;
  def LIBMVEC__ZGVdN8vv___powf_finite : LibmvecLibcall<POW_FINITE_V8F64>;
  def LIBMVEC__ZGVdN8vv_powf : LibmvecLibcall<POW_V8F32>;
}

defset list<RuntimeLibcallImpl> LIBMVEC_AARCH64_VECFUNCS = {
  def LIBMVEC__ZGVnN2v_acos : LibmvecLibcall<ACOS_V2F64>;
  def LIBMVEC__ZGVnN2v_acosf : LibmvecLibcall<ACOS_V2F32>;
  def LIBMVEC__ZGVnN2v_acosh : LibmvecLibcall<ACOSH_V2F64>;
  def LIBMVEC__ZGVnN2v_acoshf : LibmvecLibcall<ACOSH_V2F32>;
  def LIBMVEC__ZGVnN2v_asin : LibmvecLibcall<ASIN_V2F64>;
  def LIBMVEC__ZGVnN2v_asinf : LibmvecLibcall<ASIN_V2F32>;
  def LIBMVEC__ZGVnN2v_asinh : LibmvecLibcall<ASINH_V2F64>;
  def LIBMVEC__ZGVnN2v_asinhf : LibmvecLibcall<ASINH_V2F32>;
  def LIBMVEC__ZGVnN2v_atan : LibmvecLibcall<ATAN_V2F64>;
  def LIBMVEC__ZGVnN2v_atanf : LibmvecLibcall<ATAN_V2F32>;
  def LIBMVEC__ZGVnN2v_atanh : LibmvecLibcall<ATANH_V2F64>;
  def LIBMVEC__ZGVnN2v_atanhf : LibmvecLibcall<ATANH_V2F32>;
  def LIBMVEC__ZGVnN2v_cbrt : LibmvecLibcall<CBRT_V2F64>;
  def LIBMVEC__ZGVnN2v_cbrtf : LibmvecLibcall<CBRT_V2F32>;
  def LIBMVEC__ZGVnN2v_cos : LibmvecLibcall<COS_V2F64>;
  def LIBMVEC__ZGVnN2v_cosf : LibmvecLibcall<COS_V2F32>;
  def LIBMVEC__ZGVnN2v_cosh : LibmvecLibcall<COSH_V2F64>;
  def LIBMVEC__ZGVnN2v_coshf : LibmvecLibcall<COSH_V2F32>;
  def LIBMVEC__ZGVnN2v_erf : LibmvecLibcall<ERF_V2F64>;
  def LIBMVEC__ZGVnN2v_erfc : LibmvecLibcall<ERFC_V2F64>;
  def LIBMVEC__ZGVnN2v_erfcf : LibmvecLibcall<ERFC_V2F32>;
  def LIBMVEC__ZGVnN2v_erff : LibmvecLibcall<ERF_V2F32>;
  def LIBMVEC__ZGVnN2v_exp : LibmvecLibcall<EXP_V2F64>;
  def LIBMVEC__ZGVnN2v_exp10 : LibmvecLibcall<EXP10_V2F64>;
  def LIBMVEC__ZGVnN2v_exp10f : LibmvecLibcall<EXP10_V2F32>;
  def LIBMVEC__ZGVnN2v_exp2 : LibmvecLibcall<EXP2_V2F64>;
  def LIBMVEC__ZGVnN2v_exp2f : LibmvecLibcall<EXP2_V2F32>;
  def LIBMVEC__ZGVnN2v_expf : LibmvecLibcall<EXP_V2F32>;
  def LIBMVEC__ZGVnN2v_expm1 : LibmvecLibcall<EXPM1_V2F64>;
  def LIBMVEC__ZGVnN2v_expm1f : LibmvecLibcall<EXPM1_V2F32>;
  def LIBMVEC__ZGVnN2v_log : LibmvecLibcall<LOG_V2F64>;
  def LIBMVEC__ZGVnN2v_log10 : LibmvecLibcall<LOG10_V2F64>;
  def LIBMVEC__ZGVnN2v_log10f : LibmvecLibcall<LOG10_V2F32>;
  def LIBMVEC__ZGVnN2v_log1p : LibmvecLibcall<LOG1P_V2F64>;
  def LIBMVEC__ZGVnN2v_log1pf : LibmvecLibcall<LOG1P_V2F32>;
  def LIBMVEC__ZGVnN2v_log2 : LibmvecLibcall<LOG2_V2F64>;
  def LIBMVEC__ZGVnN2v_log2f : LibmvecLibcall<LOG2_V2F32>;
  def LIBMVEC__ZGVnN2v_logf : LibmvecLibcall<LOG_V2F32>;
  def LIBMVEC__ZGVnN2v_sin : LibmvecLibcall<SIN_V2F64>;
  def LIBMVEC__ZGVnN2v_sinf : LibmvecLibcall<SIN_V2F32>;
  def LIBMVEC__ZGVnN2v_sinh : LibmvecLibcall<SINH_V2F64>;
  def LIBMVEC__ZGVnN2v_sinhf : LibmvecLibcall<SINH_V2F32>;
  def LIBMVEC__ZGVnN2v_tan : LibmvecLibcall<TAN_V2F64>;
  def LIBMVEC__ZGVnN2v_tanf : LibmvecLibcall<TAN_V2F32>;
  def LIBMVEC__ZGVnN2v_tanh : LibmvecLibcall<TANH_V2F64>;
  def LIBMVEC__ZGVnN2v_tanhf : LibmvecLibcall<TANH_V2F32>;
  def LIBMVEC__ZGVnN2vv_atan2 : LibmvecLibcall<ATAN2_V2F64>;
  def LIBMVEC__ZGVnN2vv_atan2f : LibmvecLibcall<ATAN2_V2F32>;
  def LIBMVEC__ZGVnN2vv_hypot : LibmvecLibcall<HYPOT_V2F64>;
  def LIBMVEC__ZGVnN2vv_hypotf : LibmvecLibcall<HYPOT_V2F32>;
  def LIBMVEC__ZGVnN2vv_pow : LibmvecLibcall<POW_V2F64>;
  def LIBMVEC__ZGVnN2vv_powf : LibmvecLibcall<POW_V2F32>;
  def LIBMVEC__ZGVnN4v_acosf : LibmvecLibcall<ACOS_V4F32>;
  def LIBMVEC__ZGVnN4v_acoshf : LibmvecLibcall<ACOSH_V4F32>;
  def LIBMVEC__ZGVnN4v_asinf : LibmvecLibcall<ASIN_V4F32>;
  def LIBMVEC__ZGVnN4v_asinhf : LibmvecLibcall<ASINH_V4F32>;
  def LIBMVEC__ZGVnN4v_atanf : LibmvecLibcall<ATAN_V4F32>;
  def LIBMVEC__ZGVnN4v_atanhf : LibmvecLibcall<ATANH_V4F32>;
  def LIBMVEC__ZGVnN4v_cbrtf : LibmvecLibcall<CBRT_V4F32>;
  def LIBMVEC__ZGVnN4v_cosf : LibmvecLibcall<COS_V4F32>;
  def LIBMVEC__ZGVnN4v_coshf : LibmvecLibcall<COSH_V4F32>;
  def LIBMVEC__ZGVnN4v_erfcf : LibmvecLibcall<ERFC_V4F32>;
  def LIBMVEC__ZGVnN4v_erff : LibmvecLibcall<ERF_V4F32>;
  def LIBMVEC__ZGVnN4v_exp10f : LibmvecLibcall<EXP10_V4F32>;
  def LIBMVEC__ZGVnN4v_exp2f : LibmvecLibcall<EXP2_V4F32>;
  def LIBMVEC__ZGVnN4v_expf : LibmvecLibcall<EXP_V4F32>;
  def LIBMVEC__ZGVnN4v_expm1f : LibmvecLibcall<EXPM1_V4F32>;
  def LIBMVEC__ZGVnN4v_log10f : LibmvecLibcall<LOG10_V4F32>;
  def LIBMVEC__ZGVnN4v_log1pf : LibmvecLibcall<LOG1P_V4F32>;
  def LIBMVEC__ZGVnN4v_log2f : LibmvecLibcall<LOG2_V4F32>;
  def LIBMVEC__ZGVnN4v_logf : LibmvecLibcall<LOG_V4F32>;
  def LIBMVEC__ZGVnN4v_sinf : LibmvecLibcall<SIN_V4F32>;
  def LIBMVEC__ZGVnN4v_sinhf : LibmvecLibcall<SINH_V4F32>;
  def LIBMVEC__ZGVnN4v_tanf : LibmvecLibcall<TAN_V4F32>;
  def LIBMVEC__ZGVnN4v_tanhf : LibmvecLibcall<TANH_V4F32>;
  def LIBMVEC__ZGVnN4vv_atan2f : LibmvecLibcall<ATAN2_V4F32>;
  def LIBMVEC__ZGVnN4vv_hypotf : LibmvecLibcall<HYPOT_V4F32>;
  def LIBMVEC__ZGVnN4vv_powf : LibmvecLibcall<POW_V4F32>;
  def LIBMVEC__ZGVsMxv_acos : LibmvecLibcall<ACOS_NXV2F64>;
  def LIBMVEC__ZGVsMxv_acosf : LibmvecLibcall<ACOS_NXV4F32>;
  def LIBMVEC__ZGVsMxv_acosh : LibmvecLibcall<ACOSH_NXV2F64>;
  def LIBMVEC__ZGVsMxv_acoshf : LibmvecLibcall<ACOSH_NXV4F32>;
  def LIBMVEC__ZGVsMxv_asin : LibmvecLibcall<ASIN_NXV2F64>;
  def LIBMVEC__ZGVsMxv_asinf : LibmvecLibcall<ASIN_NXV4F32>;
  def LIBMVEC__ZGVsMxv_asinh : LibmvecLibcall<ASINH_NXV2F64>;
  def LIBMVEC__ZGVsMxv_asinhf : LibmvecLibcall<ASINH_NXV4F32>;
  def LIBMVEC__ZGVsMxv_atan : LibmvecLibcall<ATAN_NXV2F64>;
  def LIBMVEC__ZGVsMxv_atanf : LibmvecLibcall<ATAN_NXV4F32>;
  def LIBMVEC__ZGVsMxv_atanh : LibmvecLibcall<ATANH_NXV2F64>;
  def LIBMVEC__ZGVsMxv_atanhf : LibmvecLibcall<ATANH_NXV4F32>;
  def LIBMVEC__ZGVsMxv_cbrt : LibmvecLibcall<CBRT_NXV2F64>;
  def LIBMVEC__ZGVsMxv_cbrtf : LibmvecLibcall<CBRT_NXV4F32>;
  def LIBMVEC__ZGVsMxv_cos : LibmvecLibcall<COS_NXV2F64>;
  def LIBMVEC__ZGVsMxv_cosf : LibmvecLibcall<COS_NXV4F32>;
  def LIBMVEC__ZGVsMxv_cosh : LibmvecLibcall<COSH_NXV2F64>;
  def LIBMVEC__ZGVsMxv_coshf : LibmvecLibcall<COSH_NXV4F32>;
  def LIBMVEC__ZGVsMxv_erf : LibmvecLibcall<ERF_NXV2F64>;
  def LIBMVEC__ZGVsMxv_erfc : LibmvecLibcall<ERFC_NXV2F64>;
  def LIBMVEC__ZGVsMxv_erfcf : LibmvecLibcall<ERFC_NXV4F32>;
  def LIBMVEC__ZGVsMxv_erff : LibmvecLibcall<ERF_NXV4F32>;
  def LIBMVEC__ZGVsMxv_exp : LibmvecLibcall<EXP_NXV2F64>;
  def LIBMVEC__ZGVsMxv_exp10 : LibmvecLibcall<EXP10_NXV2F64>;
  def LIBMVEC__ZGVsMxv_exp10f : LibmvecLibcall<EXP10_NXV4F32>;
  def LIBMVEC__ZGVsMxv_exp2 : LibmvecLibcall<EXP2_NXV2F64>;
  def LIBMVEC__ZGVsMxv_exp2f : LibmvecLibcall<EXP2_NXV4F32>;
  def LIBMVEC__ZGVsMxv_expf : LibmvecLibcall<EXP_NXV4F32>;
  def LIBMVEC__ZGVsMxv_expm1 : LibmvecLibcall<EXPM1_NXV2F64>;
  def LIBMVEC__ZGVsMxv_expm1f : LibmvecLibcall<EXPM1_NXV4F32>;
  def LIBMVEC__ZGVsMxv_log : LibmvecLibcall<LOG_NXV2F64>;
  def LIBMVEC__ZGVsMxv_log10 : LibmvecLibcall<LOG10_NXV2F64>;
  def LIBMVEC__ZGVsMxv_log10f : LibmvecLibcall<LOG10_NXV4F32>;
  def LIBMVEC__ZGVsMxv_log1p : LibmvecLibcall<LOG1P_NXV2F64>;
  def LIBMVEC__ZGVsMxv_log1pf : LibmvecLibcall<LOG1P_NXV4F32>;
  def LIBMVEC__ZGVsMxv_log2 : LibmvecLibcall<LOG2_NXV2F64>;
  def LIBMVEC__ZGVsMxv_log2f : LibmvecLibcall<LOG2_NXV4F32>;
  def LIBMVEC__ZGVsMxv_logf : LibmvecLibcall<LOG_NXV4F32>;
  def LIBMVEC__ZGVsMxv_sin : LibmvecLibcall<SIN_NXV2F64>;
  def LIBMVEC__ZGVsMxv_sinf : LibmvecLibcall<SIN_NXV4F32>;
  def LIBMVEC__ZGVsMxv_sinh : LibmvecLibcall<SINH_NXV2F64>;
  def LIBMVEC__ZGVsMxv_sinhf : LibmvecLibcall<SINH_NXV4F32>;
  def LIBMVEC__ZGVsMxv_tan : LibmvecLibcall<TAN_NXV2F64>;
  def LIBMVEC__ZGVsMxv_tanf : LibmvecLibcall<TAN_NXV4F32>;
  def LIBMVEC__ZGVsMxv_tanh : LibmvecLibcall<TANH_NXV2F64>;
  def LIBMVEC__ZGVsMxv_tanhf : LibmvecLibcall<TANH_NXV4F32>;
  def LIBMVEC__ZGVsMxvv_atan2 : LibmvecLibcall<ATAN2_NXV2F64>;
  def LIBMVEC__ZGVsMxvv_atan2f : LibmvecLibcall<ATAN2_NXV4F32>;
  def LIBMVEC__ZGVsMxvv_hypot : LibmvecLibcall<HYPOT_NXV2F64>;
  def LIBMVEC__ZGVsMxvv_hypotf : LibmvecLibcall<HYPOT_NXV4F32>;
  def LIBMVEC__ZGVsMxvv_pow : LibmvecLibcall<POW_NXV2F64>;
  def LIBMVEC__ZGVsMxvv_powf : LibmvecLibcall<POW_NXV4F32>;
}

//===----------------------------------------------------------------------===//
// IBM MASS vector library (MASSV) functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> MASSV_VECFUNCS = {
  def __acosd2 : RuntimeLibcallImpl<ACOS_V2F64>;
  def __acosf4 : RuntimeLibcallImpl<ACOS_V4F32>;
  def __acoshd2 : RuntimeLibcallImpl<ACOSH_V2F64>;
  def __acoshf4 : RuntimeLibcallImpl<ACOSH_V4F32>;
  def __asind2 : RuntimeLibcallImpl<ASIN_V2F64>;
  def __asinf4 : RuntimeLibcallImpl<ASIN_V4F32>;
  def __asinhd2 : RuntimeLibcallImpl<ASINH_V2F64>;
  def __asinhf4 : RuntimeLibcallImpl<ASINH_V4F32>;
  def __atan2d2 : RuntimeLibcallImpl<ATAN2_V2F64>;
  def __atan2f4 : RuntimeLibcallImpl<ATAN2_V4F32>;
  def __atand2 : RuntimeLibcallImpl<ATAN_V2F64>;
  def __atanf4 : RuntimeLibcallImpl<ATAN_V4F32>;
  def __atanhd2 : RuntimeLibcallImpl<ATANH_V2F64>;
  def __atanhf4 : RuntimeLibcallImpl<ATANH_V4F32>;
  def __cbrtd2 : RuntimeLibcallImpl<CBRT_V2F64>;
  def __cbrtf4 : RuntimeLibcallImpl<CBRT_V4F32>;
  def __cosd2 : RuntimeLibcallImpl<COS_V2F64>;
  def __cosf4 : RuntimeLibcallImpl<COS_V4F32>;
  def __coshd2 : RuntimeLibcallImpl<COSH_V2F64>;
  def __coshf4 : RuntimeLibcallImpl<COSH_V4F32>;
  def __exp2d2 : RuntimeLibcallImpl<EXP2_V2F64>;
  def __exp2f4 : RuntimeLibcallImpl<EXP2_V4F32>;
  def __expd2 : RuntimeLibcallImpl<EXP_V2F64>;
  def __expf4 : RuntimeLibcallImpl<EXP_V4F32>;
  def __expm1d2 : RuntimeLibcallImpl<EXPM1_V2F64>;
  def __expm1f4 : RuntimeLibcallImpl<EXPM1_V4F32>;
  def __log10d2 : RuntimeLibcallImpl<LOG10_V2F64>;
  def __log10f4 : RuntimeLibcallImpl<LOG10_V4F32>;
  def __log1pd2 : RuntimeLibcallImpl<LOG1P_V2F64>;
  def __log1pf4 : RuntimeLibcallImpl<LOG1P_V4F32>;
  def __log2d2 : RuntimeLibcallImpl<LOG2_V2F64>;
  def __log2f4 : RuntimeLibcallImpl<LOG2_V4F32>;
  def __logd2 : RuntimeLibcallImpl<LOG_V2F64>;
  def __logf4 : RuntimeLibcallImpl<LOG_V4F32>;
  def __powd2 : RuntimeLibcallImpl<POW_V2F64>;
  def __powf4 : RuntimeLibcallImpl<POW_V4F32>;
  def __sind2 : RuntimeLibcallImpl<SIN_V2F64>;
  def __sinf4 : RuntimeLibcallImpl<SIN_V4F32>;
  def __sinhd2 : RuntimeLibcallImpl<SINH_V2F64>;
  def __sinhf4 : RuntimeLibcallImpl<SINH_V4F32>;
  def __tand2 : RuntimeLibcallImpl<TAN_V2F64>;
  def __tanf4 : RuntimeLibcallImpl<TAN_V4F32>;
  def __tanhd2 : RuntimeLibcallImpl<TANH_V2F64>;
  def __tanhf4 : RuntimeLibcallImpl<TANH_V4F32>;
}

//===----------------------------------------------------------------------===//
// Intel SVML library functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> SVML_VECFUNCS = {
  def __svml_cos2 : RuntimeLibcallImpl<COS_V2F64>;
  def __svml_cos4 : RuntimeLibcallImpl<COS_V4F64>;
  def __svml_cos8 : RuntimeLibcallImpl<COS_V8F64>;
  def __svml_cosf16 : RuntimeLibcallImpl<COS_V16F32>;
  def __svml_cosf4 : RuntimeLibcallImpl<COS_V4F32>;
  def __svml_cosf8 : RuntimeLibcallImpl<COS_V8F32>;
  def __svml_exp2 : RuntimeLibcallImpl<EXP_V2F64>;
  def __svml_exp22 : RuntimeLibcallImpl<EXP2_V2F64>;
  def __svml_exp24 : RuntimeLibcallImpl<EXP2_V4F64>;
  def __svml_exp28 : RuntimeLibcallImpl<EXP2_V8F64>;
  def __svml_exp2f16 : RuntimeLibcallImpl<EXP2_V16F32>;
  def __svml_exp2f4 : RuntimeLibcallImpl<EXP2_V4F32>;
  def __svml_exp2f8 : RuntimeLibcallImpl<EXP2_V8F32>;
  def __svml_exp4 : RuntimeLibcallImpl<EXP_V4F64>;
  def __svml_exp8 : RuntimeLibcallImpl<EXP_V8F64>;
  def __svml_expf16 : RuntimeLibcallImpl<EXP_V16F32>;
  def __svml_expf4 : RuntimeLibcallImpl<EXP_V4F32>;
  def __svml_expf8 : RuntimeLibcallImpl<EXP_V8F32>;
  def __svml_log102 : RuntimeLibcallImpl<LOG10_V2F64>;
  def __svml_log104 : RuntimeLibcallImpl<LOG10_V4F64>;
  def __svml_log108 : RuntimeLibcallImpl<LOG10_V8F64>;
  def __svml_log10f16 : RuntimeLibcallImpl<LOG10_V16F32>;
  def __svml_log10f4 : RuntimeLibcallImpl<LOG10_V4F32>;
  def __svml_log10f8 : RuntimeLibcallImpl<LOG10_V8F32>;
  def __svml_log2 : RuntimeLibcallImpl<LOG_V2F64>;
  def __svml_log22 : RuntimeLibcallImpl<LOG2_V2F64>;
  def __svml_log24 : RuntimeLibcallImpl<LOG2_V4F64>;
  def __svml_log28 : RuntimeLibcallImpl<LOG2_V8F64>;
  def __svml_log2f16 : RuntimeLibcallImpl<LOG2_V16F32>;
  def __svml_log2f4 : RuntimeLibcallImpl<LOG2_V4F32>;
  def __svml_log2f8 : RuntimeLibcallImpl<LOG2_V8F32>;
  def __svml_log4 : RuntimeLibcallImpl<LOG_V4F64>;
  def __svml_log8 : RuntimeLibcallImpl<LOG_V8F64>;
  def __svml_logf16 : RuntimeLibcallImpl<LOG_V16F32>;
  def __svml_logf4 : RuntimeLibcallImpl<LOG_V4F32>;
  def __svml_logf8 : RuntimeLibcallImpl<LOG_V8F32>;
  def __svml_pow2 : RuntimeLibcallImpl<POW_V2F64>;
  def __svml_pow4 : RuntimeLibcallImpl<POW_V4F64>;
  def __svml_pow8 : RuntimeLibcallImpl<POW_V8F64>;
  def __svml_powf16 : RuntimeLibcallImpl<POW_V16F32>;
  def __svml_powf4 : RuntimeLibcallImpl<POW_V4F32>;
  def __svml_powf8 : RuntimeLibcallImpl<POW_V8F32>;
  def __svml_sin2 : RuntimeLibcallImpl<SIN_V2F64>;
  def __svml_sin4 : RuntimeLibcallImpl<SIN_V4F64>;
  def __svml_sin8 : RuntimeLibcallImpl<SIN_V8F64>;
  def __svml_sinf16 : RuntimeLibcallImpl<SIN_V16F32>;
  def __svml_sinf4 : RuntimeLibcallImpl<SIN_V4F32>;
  def __svml_sinf8 : RuntimeLibcallImpl<SIN_V8F32>;
  def __svml_sqrt2 : RuntimeLibcallImpl<SQRT_V2F64>;
  def __svml_sqrt4 : RuntimeLibcallImpl<SQRT_V4F64>;
  def __svml_sqrt8 : RuntimeLibcallImpl<SQRT_V8F64>;
  def __svml_sqrtf16 : RuntimeLibcallImpl<SQRT_V16F32>;
  def __svml_sqrtf4 : RuntimeLibcallImpl<SQRT_V4F32>;
  def __svml_sqrtf8 : RuntimeLibcallImpl<SQRT_V8F32>;
  def __svml_tan2 : RuntimeLibcallImpl<TAN_V2F64>;
  def __svml_tan4 : RuntimeLibcallImpl<TAN_V4F64>;
  def __svml_tan8 : RuntimeLibcallImpl<TAN_V8F64>;
  def __svml_tanf16 : RuntimeLibcallImpl<TAN_V16F32>;
  def __svml_tanf4 : RuntimeLibcallImpl<TAN_V4F32>;
  def __svml_tanf8 : RuntimeLibcallImpl<TAN_V8F32>;
}

//===----------------------------------------------------------------------===//
// SIMD Library for Evaluating Elementary Functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> SLEEFGNUABI_VF2_VECFUNCS = {
  def _ZGVnN2v_acos : RuntimeLibcallImpl<ACOS_V2F64>;
  def _ZGVnN2v_acosh : RuntimeLibcallImpl<ACOSH_V2F64>;
  def _ZGVnN2v_asin : RuntimeLibcallImpl<ASIN_V2F64>;
  def _ZGVnN2v_asinh : RuntimeLibcallImpl<ASINH_V2F64>;
  def _ZGVnN2v_atan : RuntimeLibcallImpl<ATAN_V2F64>;
  def _ZGVnN2v_atanh : RuntimeLibcallImpl<ATANH_V2F64>;
  def _ZGVnN2v_cbrt : RuntimeLibcallImpl<CBRT_V2F64>;
  def _ZGVnN2v_cos : RuntimeLibcallImpl<COS_V2F64>;
  def _ZGVnN2v_cosh : RuntimeLibcallImpl<COSH_V2F64>;
  def _ZGVnN2v_cospi : RuntimeLibcallImpl<COSPI_V2F64>;
  def _ZGVnN2v_erf : RuntimeLibcallImpl<ERF_V2F64>;
  def _ZGVnN2v_erfc : RuntimeLibcallImpl<ERFC_V2F64>;
  def _ZGVnN2v_exp : RuntimeLibcallImpl<EXP_V2F64>;
  def _ZGVnN2v_exp10 : RuntimeLibcallImpl<EXP10_V2F64>;
  def _ZGVnN2v_exp2 : RuntimeLibcallImpl<EXP2_V2F64>;
  def _ZGVnN2v_expm1 : RuntimeLibcallImpl<EXPM1_V2F64>;
  def _ZGVnN2v_ilogb : RuntimeLibcallImpl<ILOGB_V2F64>;
  def _ZGVnN2v_lgamma : RuntimeLibcallImpl<LGAMMA_V2F64>;
  def _ZGVnN2v_log : RuntimeLibcallImpl<LOG_V2F64>;
  def _ZGVnN2v_log10 : RuntimeLibcallImpl<LOG10_V2F64>;
  def _ZGVnN2v_log1p : RuntimeLibcallImpl<LOG1P_V2F64>;
  def _ZGVnN2v_log2 : RuntimeLibcallImpl<LOG2_V2F64>;
  def _ZGVnN2v_sin : RuntimeLibcallImpl<SIN_V2F64>;
  def _ZGVnN2v_sinh : RuntimeLibcallImpl<SINH_V2F64>;
  def _ZGVnN2v_sinpi : RuntimeLibcallImpl<SINPI_V2F64>;
  def _ZGVnN2v_sqrt : RuntimeLibcallImpl<SQRT_V2F64>;
  def _ZGVnN2v_tan : RuntimeLibcallImpl<TAN_V2F64>;
  def _ZGVnN2v_tanh : RuntimeLibcallImpl<TANH_V2F64>;
  def _ZGVnN2v_tgamma : RuntimeLibcallImpl<TGAMMA_V2F64>;
  def _ZGVnN2vl8_modf : RuntimeLibcallImpl<MODF_V2F64>;
  def _ZGVnN2vl8l8_sincos : RuntimeLibcallImpl<SINCOS_V2F64>;
  def _ZGVnN2vl8l8_sincospi : RuntimeLibcallImpl<SINCOSPI_V2F64>;
  def _ZGVnN2vv_atan2 : RuntimeLibcallImpl<ATAN2_V2F64>;
  def _ZGVnN2vv_copysign : RuntimeLibcallImpl<COPYSIGN_V2F64>;
  def _ZGVnN2vv_fdim : RuntimeLibcallImpl<FDIM_V2F64>;
  def _ZGVnN2vv_fmax : RuntimeLibcallImpl<FMAX_V2F64>;
  def _ZGVnN2vv_fmin : RuntimeLibcallImpl<FMIN_V2F64>;
  def _ZGVnN2vv_fmod : RuntimeLibcallImpl<REM_V2F64>;
  def _ZGVnN2vv_hypot : RuntimeLibcallImpl<HYPOT_V2F64>;
  def _ZGVnN2vv_ldexp : RuntimeLibcallImpl<LDEXP_V2F64>;
  def _ZGVnN2vv_nextafter : RuntimeLibcallImpl<NEXTAFTER_V2F64>;
  def _ZGVnN2vv_pow : RuntimeLibcallImpl<POW_V2F64>;
  def _ZGVnN2vvv_fma : RuntimeLibcallImpl<FMA_V2F64>;
}

defset list<RuntimeLibcallImpl> SLEEFGNUABI_VF4_VECFUNCS = {
  def _ZGVnN4v_acosf : RuntimeLibcallImpl<ACOS_V4F32>;
  def _ZGVnN4v_acoshf : RuntimeLibcallImpl<ACOSH_V4F32>;
  def _ZGVnN4v_asinf : RuntimeLibcallImpl<ASIN_V4F32>;
  def _ZGVnN4v_asinhf : RuntimeLibcallImpl<ASINH_V4F32>;
  def _ZGVnN4v_atanf : RuntimeLibcallImpl<ATAN_V4F32>;
  def _ZGVnN4v_atanhf : RuntimeLibcallImpl<ATANH_V4F32>;
  def _ZGVnN4v_cbrtf : RuntimeLibcallImpl<CBRT_V4F32>;
  def _ZGVnN4v_cosf : RuntimeLibcallImpl<COS_V4F32>;
  def _ZGVnN4v_coshf : RuntimeLibcallImpl<COSH_V4F32>;
  def _ZGVnN4v_cospif : RuntimeLibcallImpl<COSPI_V4F32>;
  def _ZGVnN4v_erfcf : RuntimeLibcallImpl<ERFC_V4F32>;
  def _ZGVnN4v_erff : RuntimeLibcallImpl<ERF_V4F32>;
  def _ZGVnN4v_exp10f : RuntimeLibcallImpl<EXP10_V4F32>;
  def _ZGVnN4v_exp2f : RuntimeLibcallImpl<EXP2_V4F32>;
  def _ZGVnN4v_expf : RuntimeLibcallImpl<EXP_V4F32>;
  def _ZGVnN4v_expm1f : RuntimeLibcallImpl<EXPM1_V4F32>;
  def _ZGVnN4v_ilogbf : RuntimeLibcallImpl<ILOGB_V4F32>;
  def _ZGVnN4v_lgammaf : RuntimeLibcallImpl<LGAMMA_V4F32>;
  def _ZGVnN4v_log10f : RuntimeLibcallImpl<LOG10_V4F32>;
  def _ZGVnN4v_log1pf : RuntimeLibcallImpl<LOG1P_V4F32>;
  def _ZGVnN4v_log2f : RuntimeLibcallImpl<LOG2_V4F32>;
  def _ZGVnN4v_logf : RuntimeLibcallImpl<LOG_V4F32>;
  def _ZGVnN4v_sinf : RuntimeLibcallImpl<SIN_V4F32>;
  def _ZGVnN4v_sinhf : RuntimeLibcallImpl<SINH_V4F32>;
  def _ZGVnN4v_sinpif : RuntimeLibcallImpl<SINPI_V4F32>;
  def _ZGVnN4v_sqrtf : RuntimeLibcallImpl<SQRT_V4F32>;
  def _ZGVnN4v_tanf : RuntimeLibcallImpl<TAN_V4F32>;
  def _ZGVnN4v_tanhf : RuntimeLibcallImpl<TANH_V4F32>;
  def _ZGVnN4v_tgammaf : RuntimeLibcallImpl<TGAMMA_V4F32>;
  def _ZGVnN4vl4_modff : RuntimeLibcallImpl<MODF_V4F32>;
  def _ZGVnN4vl4l4_sincosf : RuntimeLibcallImpl<SINCOS_V4F32>;
  def _ZGVnN4vl4l4_sincospif : RuntimeLibcallImpl<SINCOSPI_V4F32>;
  def _ZGVnN4vv_atan2f : RuntimeLibcallImpl<ATAN2_V4F32>;
  def _ZGVnN4vv_copysignf : RuntimeLibcallImpl<COPYSIGN_V4F32>;
  def _ZGVnN4vv_fdimf : RuntimeLibcallImpl<FDIM_V4F32>;
  def _ZGVnN4vv_fmaxf : RuntimeLibcallImpl<FMAX_V4F32>;
  def _ZGVnN4vv_fminf : RuntimeLibcallImpl<FMIN_V4F32>;
  def _ZGVnN4vv_fmodf : RuntimeLibcallImpl<REM_V4F32>;
  def _ZGVnN4vv_hypotf : RuntimeLibcallImpl<HYPOT_V4F32>;
  def _ZGVnN4vv_ldexpf : RuntimeLibcallImpl<LDEXP_V4F32>;
  def _ZGVnN4vv_nextafterf : RuntimeLibcallImpl<NEXTAFTER_V4F32>;
  def _ZGVnN4vv_powf : RuntimeLibcallImpl<POW_V4F32>;
  def _ZGVnN4vvv_fmaf : RuntimeLibcallImpl<FMA_V4F32>;
}

defset list<RuntimeLibcallImpl> SLEEFGNUABI_SCALABLE_VECFUNCS = {
  def _ZGVsMxv_acos : RuntimeLibcallImpl<ACOS_NXV2F64>;
  def _ZGVsMxv_acosf : RuntimeLibcallImpl<ACOS_NXV4F32>;
  def _ZGVsMxv_acosh : RuntimeLibcallImpl<ACOSH_NXV2F64>;
  def _ZGVsMxv_acoshf : RuntimeLibcallImpl<ACOSH_NXV4F32>;
  def _ZGVsMxv_asin : RuntimeLibcallImpl<ASIN_NXV2F64>;
  def _ZGVsMxv_asinf : RuntimeLibcallImpl<ASIN_NXV4F32>;
  def _ZGVsMxv_asinh : RuntimeLibcallImpl<ASINH_NXV2F64>;
  def _ZGVsMxv_asinhf : RuntimeLibcallImpl<ASINH_NXV4F32>;
  def _ZGVsMxv_atan : RuntimeLibcallImpl<ATAN_NXV2F64>;
  def _ZGVsMxv_atanf : RuntimeLibcallImpl<ATAN_NXV4F32>;
  def _ZGVsMxv_atanh : RuntimeLibcallImpl<ATANH_NXV2F64>;
  def _ZGVsMxv_atanhf : RuntimeLibcallImpl<ATANH_NXV4F32>;
  def _ZGVsMxv_cbrt : RuntimeLibcallImpl<CBRT_NXV2F64>;
  def _ZGVsMxv_cbrtf : RuntimeLibcallImpl<CBRT_NXV4F32>;
  def _ZGVsMxv_cos : RuntimeLibcallImpl<COS_NXV2F64>;
  def _ZGVsMxv_cosf : RuntimeLibcallImpl<COS_NXV4F32>;
  def _ZGVsMxv_cosh : RuntimeLibcallImpl<COSH_NXV2F64>;
  def _ZGVsMxv_coshf : RuntimeLibcallImpl<COSH_NXV4F32>;
  def _ZGVsMxv_cospi : RuntimeLibcallImpl<COSPI_NXV2F64>;
  def _ZGVsMxv_cospif : RuntimeLibcallImpl<COSPI_NXV4F32>;
  def _ZGVsMxv_erf : RuntimeLibcallImpl<ERF_NXV2F64>;
  def _ZGVsMxv_erfc : RuntimeLibcallImpl<ERFC_NXV2F64>;
  def _ZGVsMxv_erfcf : RuntimeLibcallImpl<ERFC_NXV4F32>;
  def _ZGVsMxv_erff : RuntimeLibcallImpl<ERF_NXV4F32>;
  def _ZGVsMxv_exp : RuntimeLibcallImpl<EXP_NXV2F64>;
  def _ZGVsMxv_exp10 : RuntimeLibcallImpl<EXP10_NXV2F64>;
  def _ZGVsMxv_exp10f : RuntimeLibcallImpl<EXP10_NXV4F32>;
  def _ZGVsMxv_exp2 : RuntimeLibcallImpl<EXP2_NXV2F64>;
  def _ZGVsMxv_exp2f : RuntimeLibcallImpl<EXP2_NXV4F32>;
  def _ZGVsMxv_expf : RuntimeLibcallImpl<EXP_NXV4F32>;
  def _ZGVsMxv_expm1 : RuntimeLibcallImpl<EXPM1_NXV2F64>;
  def _ZGVsMxv_expm1f : RuntimeLibcallImpl<EXPM1_NXV4F32>;
  def _ZGVsMxv_ilogb : RuntimeLibcallImpl<ILOGB_NXV2F64>;
  def _ZGVsMxv_ilogbf : RuntimeLibcallImpl<ILOGB_NXV4F32>;
  def _ZGVsMxv_lgamma : RuntimeLibcallImpl<LGAMMA_NXV2F64>;
  def _ZGVsMxv_lgammaf : RuntimeLibcallImpl<LGAMMA_NXV4F32>;
  def _ZGVsMxv_log : RuntimeLibcallImpl<LOG_NXV2F64>;
  def _ZGVsMxv_log10 : RuntimeLibcallImpl<LOG10_NXV2F64>;
  def _ZGVsMxv_log10f : RuntimeLibcallImpl<LOG10_NXV4F32>;
  def _ZGVsMxv_log1p : RuntimeLibcallImpl<LOG1P_NXV2F64>;
  def _ZGVsMxv_log1pf : RuntimeLibcallImpl<LOG1P_NXV4F32>;
  def _ZGVsMxv_log2 : RuntimeLibcallImpl<LOG2_NXV2F64>;
  def _ZGVsMxv_log2f : RuntimeLibcallImpl<LOG2_NXV4F32>;
  def _ZGVsMxv_logf : RuntimeLibcallImpl<LOG_NXV4F32>;
  def _ZGVsMxv_sin : RuntimeLibcallImpl<SIN_NXV2F64>;
  def _ZGVsMxv_sinf : RuntimeLibcallImpl<SIN_NXV4F32>;
  def _ZGVsMxv_sinh : RuntimeLibcallImpl<SINH_NXV2F64>;
  def _ZGVsMxv_sinhf : RuntimeLibcallImpl<SINH_NXV4F32>;
  def _ZGVsMxv_sinpi : RuntimeLibcallImpl<SINPI_NXV2F64>;
  def _ZGVsMxv_sinpif : RuntimeLibcallImpl<SINPI_NXV4F32>;
  def _ZGVsMxv_sqrt : RuntimeLibcallImpl<SQRT_NXV2F64>;
  def _ZGVsMxv_sqrtf : RuntimeLibcallImpl<SQRT_NXV4F32>;
  def _ZGVsMxv_tan : RuntimeLibcallImpl<TAN_NXV2F64>;
  def _ZGVsMxv_tanf : RuntimeLibcallImpl<TAN_NXV4F32>;
  def _ZGVsMxv_tanh : RuntimeLibcallImpl<TANH_NXV2F64>;
  def _ZGVsMxv_tanhf : RuntimeLibcallImpl<TANH_NXV4F32>;
  def _ZGVsMxv_tgamma : RuntimeLibcallImpl<TGAMMA_NXV2F64>;
  def _ZGVsMxv_tgammaf : RuntimeLibcallImpl<TGAMMA_NXV4F32>;
  def _ZGVsMxvv_atan2 : RuntimeLibcallImpl<ATAN2_NXV2F64>;
  def _ZGVsMxvv_atan2f : RuntimeLibcallImpl<ATAN2_NXV4F32>;
  def _ZGVsMxvv_copysign : RuntimeLibcallImpl<COPYSIGN_NXV2F64>;
  def _ZGVsMxvv_copysignf : RuntimeLibcallImpl<COPYSIGN_NXV4F32>;
  def _ZGVsMxvv_fdim : RuntimeLibcallImpl<FDIM_NXV2F64>;
  def _ZGVsMxvv_fdimf : RuntimeLibcallImpl<FDIM_NXV4F32>;
  def _ZGVsMxvv_fmax : RuntimeLibcallImpl<FMAX_NXV2F64>;
  def _ZGVsMxvv_fmaxf : RuntimeLibcallImpl<FMAX_NXV4F32>;
  def _ZGVsMxvv_fmin : RuntimeLibcallImpl<FMIN_NXV2F64>;
  def _ZGVsMxvv_fminf : RuntimeLibcallImpl<FMIN_NXV4F32>;
  def _ZGVsMxvv_fmod : RuntimeLibcallImpl<REM_NXV2F64>;
  def _ZGVsMxvv_fmodf : RuntimeLibcallImpl<REM_NXV4F32>;
  def _ZGVsMxvv_hypot : RuntimeLibcallImpl<HYPOT_NXV2F64>;
  def _ZGVsMxvv_hypotf : RuntimeLibcallImpl<HYPOT_NXV4F32>;
  def _ZGVsMxvv_ldexp : RuntimeLibcallImpl<LDEXP_NXV2F64>;
  def _ZGVsMxvv_ldexpf : RuntimeLibcallImpl<LDEXP_NXV4F32>;
  def _ZGVsMxvv_nextafter : RuntimeLibcallImpl<NEXTAFTER_NXV2F64>;
  def _ZGVsMxvv_nextafterf : RuntimeLibcallImpl<NEXTAFTER_NXV4F32>;
  def _ZGVsMxvv_pow : RuntimeLibcallImpl<POW_NXV2F64>;
  def _ZGVsMxvv_powf : RuntimeLibcallImpl<POW_NXV4F32>;
  def _ZGVsMxvvv_fma : RuntimeLibcallImpl<FMA_NXV2F64>;
  def _ZGVsMxvvv_fmaf : RuntimeLibcallImpl<FMA_NXV4F32>;
  def _ZGVsNxvl8_modf : RuntimeLibcallImpl<MODF_NXV2F64>;
  def _ZGVsNxvl4_modff : RuntimeLibcallImpl<MODF_NXV4F32>;
  def _ZGVsNxvl4l4_sincosf : RuntimeLibcallImpl<SINCOS_NXV4F32>;
  def _ZGVsNxvl4l4_sincospif : RuntimeLibcallImpl<SINCOSPI_NXV4F32>;
  def _ZGVsNxvl8l8_sincos : RuntimeLibcallImpl<SINCOS_NXV2F64>;
  def _ZGVsNxvl8l8_sincospi : RuntimeLibcallImpl<SINCOSPI_NXV2F64>;
}

defset list<RuntimeLibcallImpl> SLEEFGNUABI_SCALABLE_VECFUNCS_RISCV = {
  def Sleef_acosdx_u10rvvm2 : RuntimeLibcallImpl<ACOS_NXV2F64>;
  def Sleef_acosfx_u10rvvm2 : RuntimeLibcallImpl<ACOS_NXV4F32>;
  def Sleef_acoshdx_u10rvvm2 : RuntimeLibcallImpl<ACOSH_NXV2F64>;
  def Sleef_acoshfx_u10rvvm2 : RuntimeLibcallImpl<ACOSH_NXV4F32>;
  def Sleef_asindx_u10rvvm2 : RuntimeLibcallImpl<ASIN_NXV2F64>;
  def Sleef_asinfx_u10rvvm2 : RuntimeLibcallImpl<ASIN_NXV4F32>;
  def Sleef_asinhdx_u10rvvm2 : RuntimeLibcallImpl<ASINH_NXV2F64>;
  def Sleef_asinhfx_u10rvvm2 : RuntimeLibcallImpl<ASINH_NXV4F32>;
  def Sleef_atan2dx_u10rvvm2 : RuntimeLibcallImpl<ATAN2_NXV2F64>;
  def Sleef_atan2fx_u10rvvm2 : RuntimeLibcallImpl<ATAN2_NXV4F32>;
  def Sleef_atandx_u10rvvm2 : RuntimeLibcallImpl<ATAN_NXV2F64>;
  def Sleef_atanfx_u10rvvm2 : RuntimeLibcallImpl<ATAN_NXV4F32>;
  def Sleef_atanhdx_u10rvvm2 : RuntimeLibcallImpl<ATANH_NXV2F64>;
  def Sleef_atanhfx_u10rvvm2 : RuntimeLibcallImpl<ATANH_NXV4F32>;
  def Sleef_cbrtdx_u10rvvm2 : RuntimeLibcallImpl<CBRT_NXV2F64>;
  def Sleef_cbrtfx_u10rvvm2 : RuntimeLibcallImpl<CBRT_NXV4F32>;
  def Sleef_copysigndx_rvvm2 : RuntimeLibcallImpl<COPYSIGN_NXV2F64>;
  def Sleef_copysignfx_rvvm2 : RuntimeLibcallImpl<COPYSIGN_NXV4F32>;
  def Sleef_cosdx_u10rvvm2 : RuntimeLibcallImpl<COS_NXV2F64>;
  def Sleef_cosfx_u10rvvm2 : RuntimeLibcallImpl<COS_NXV4F32>;
  def Sleef_coshdx_u10rvvm2 : RuntimeLibcallImpl<COSH_NXV2F64>;
  def Sleef_coshfx_u10rvvm2 : RuntimeLibcallImpl<COSH_NXV4F32>;
  def Sleef_cospidx_u05rvvm2 : RuntimeLibcallImpl<COSPI_NXV2F64>;
  def Sleef_cospifx_u05rvvm2 : RuntimeLibcallImpl<COSPI_NXV4F32>;
  def Sleef_erfcdx_u15rvvm2 : RuntimeLibcallImpl<ERFC_NXV2F64>;
  def Sleef_erfcfx_u15rvvm2 : RuntimeLibcallImpl<ERFC_NXV4F32>;
  def Sleef_erfdx_u10rvvm2 : RuntimeLibcallImpl<ERF_NXV2F64>;
  def Sleef_erffx_u10rvvm2 : RuntimeLibcallImpl<ERF_NXV4F32>;
  def Sleef_exp10dx_u10rvvm2 : RuntimeLibcallImpl<EXP10_NXV2F64>;
  def Sleef_exp10fx_u10rvvm2 : RuntimeLibcallImpl<EXP10_NXV4F32>;
  def Sleef_exp2dx_u10rvvm2 : RuntimeLibcallImpl<EXP2_NXV2F64>;
  def Sleef_exp2fx_u10rvvm2 : RuntimeLibcallImpl<EXP2_NXV4F32>;
  def Sleef_expdx_u10rvvm2 : RuntimeLibcallImpl<EXP_NXV2F64>;
  def Sleef_expfx_u10rvvm2 : RuntimeLibcallImpl<EXP_NXV4F32>;
  def Sleef_expm1dx_u10rvvm2 : RuntimeLibcallImpl<EXPM1_NXV2F64>;
  def Sleef_expm1fx_u10rvvm2 : RuntimeLibcallImpl<EXPM1_NXV4F32>;
  def Sleef_fdimdx_rvvm2 : RuntimeLibcallImpl<FDIM_NXV2F64>;
  def Sleef_fdimfx_rvvm2 : RuntimeLibcallImpl<FDIM_NXV4F32>;
  def Sleef_fmadx_rvvm2 : RuntimeLibcallImpl<FMA_NXV2F64>;
  def Sleef_fmafx_rvvm2 : RuntimeLibcallImpl<FMA_NXV4F32>;
  def Sleef_fmaxdx_rvvm2 : RuntimeLibcallImpl<FMAX_NXV2F64>;
  def Sleef_fmaxfx_rvvm2 : RuntimeLibcallImpl<FMAX_NXV4F32>;
  def Sleef_fmindx_u10rvvm2 : RuntimeLibcallImpl<FMIN_NXV2F64>;
  def Sleef_fminfx_u10rvvm2 : RuntimeLibcallImpl<FMIN_NXV4F32>;
  def Sleef_fmoddx_rvvm2 : RuntimeLibcallImpl<REM_NXV2F64>;
  def Sleef_fmodfx_rvvm2 : RuntimeLibcallImpl<REM_NXV4F32>;
  def Sleef_hypotdx_u05rvvm2 : RuntimeLibcallImpl<HYPOT_NXV2F64>;
  def Sleef_hypotfx_u05rvvm2 : RuntimeLibcallImpl<HYPOT_NXV4F32>;
  def Sleef_ilogbdx_rvvm2 : RuntimeLibcallImpl<ILOGB_NXV2F64>;
  def Sleef_ilogbfx_rvvm2 : RuntimeLibcallImpl<ILOGB_NXV4F32>;
  def Sleef_ldexpdx_rvvm2 : RuntimeLibcallImpl<LDEXP_NXV2F64>;
  def Sleef_ldexpfx_rvvm2 : RuntimeLibcallImpl<LDEXP_NXV4F32>;
  def Sleef_lgammadx_u10rvvm2 : RuntimeLibcallImpl<LGAMMA_NXV2F64>;
  def Sleef_lgammafx_u10rvvm2 : RuntimeLibcallImpl<LGAMMA_NXV4F32>;
  def Sleef_log10dx_u10rvvm2 : RuntimeLibcallImpl<LOG10_NXV2F64>;
  def Sleef_log10fx_u10rvvm2 : RuntimeLibcallImpl<LOG10_NXV4F32>;
  def Sleef_log1pdx_u10rvvm2 : RuntimeLibcallImpl<LOG1P_NXV2F64>;
  def Sleef_log1pfx_u10rvvm2 : RuntimeLibcallImpl<LOG1P_NXV4F32>;
  def Sleef_log2dx_u10rvvm2 : RuntimeLibcallImpl<LOG2_NXV2F64>;
  def Sleef_log2fx_u10rvvm2 : RuntimeLibcallImpl<LOG2_NXV4F32>;
  def Sleef_logdx_u10rvvm2 : RuntimeLibcallImpl<LOG_NXV2F64>;
  def Sleef_logfx_u10rvvm2 : RuntimeLibcallImpl<LOG_NXV4F32>;
  def Sleef_modfdx_rvvm2 : RuntimeLibcallImpl<MODF_NXV2F64>;
  def Sleef_modffx_rvvm2 : RuntimeLibcallImpl<MODF_NXV4F32>;
  def Sleef_nextafterdx_rvvm2 : RuntimeLibcallImpl<NEXTAFTER_NXV2F64>;
  def Sleef_nextafterfx_rvvm2 : RuntimeLibcallImpl<NEXTAFTER_NXV4F32>;
  def Sleef_powdx_u10rvvm2 : RuntimeLibcallImpl<POW_NXV2F64>;
  def Sleef_powfx_u10rvvm2 : RuntimeLibcallImpl<POW_NXV4F32>;
  def Sleef_sincosdx_u10rvvm2 : RuntimeLibcallImpl<SINCOS_NXV2F64>;
  def Sleef_sincosfx_u10rvvm2 : RuntimeLibcallImpl<SINCOS_NXV4F32>;
  def Sleef_sincospidx_u10rvvm2 : RuntimeLibcallImpl<SINCOSPI_NXV2F64>;
  def Sleef_sincospifx_u10rvvm2 : RuntimeLibcallImpl<SINCOSPI_NXV4F32>;
  def Sleef_sindx_u10rvvm2 : RuntimeLibcallImpl<SIN_NXV2F64>;
  def Sleef_sinfx_u10rvvm2 : RuntimeLibcallImpl<SIN_NXV4F32>;
  def Sleef_sinhdx_u10rvvm2 : RuntimeLibcallImpl<SINH_NXV2F64>;
  def Sleef_sinhfx_u10rvvm2 : RuntimeLibcallImpl<SINH_NXV4F32>;
  def Sleef_sinpidx_u05rvvm2 : RuntimeLibcallImpl<SINPI_NXV2F64>;
  def Sleef_sinpifx_u05rvvm2 : RuntimeLibcallImpl<SINPI_NXV4F32>;
  def Sleef_sqrtdx_u05rvvm2 : RuntimeLibcallImpl<SQRT_NXV2F64>;
  def Sleef_sqrtfx_u05rvvm2 : RuntimeLibcallImpl<SQRT_NXV4F32>;
  def Sleef_tandx_u10rvvm2 : RuntimeLibcallImpl<TAN_NXV2F64>;
  def Sleef_tanfx_u10rvvm2 : RuntimeLibcallImpl<TAN_NXV4F32>;
  def Sleef_tanhdx_u10rvvm2 : RuntimeLibcallImpl<TANH_NXV2F64>;
  def Sleef_tanhfx_u10rvvm2 : RuntimeLibcallImpl<TANH_NXV4F32>;
  def Sleef_tgammadx_u10rvvm2 : RuntimeLibcallImpl<TGAMMA_NXV2F64>;
  def Sleef_tgammafx_u10rvvm2 : RuntimeLibcallImpl<TGAMMA_NXV4F32>;
}

//===----------------------------------------------------------------------===//
// Arm Performance Libraries (ARMPL) functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> ARMPL_VECFUNCS = {
  def armpl_svacos_f32_x : RuntimeLibcallImpl<ACOS_NXV4F32>;
  def armpl_svacos_f64_x : RuntimeLibcallImpl<ACOS_NXV2F64>;
  def armpl_svacosh_f32_x : RuntimeLibcallImpl<ACOSH_NXV4F32>;
  def armpl_svacosh_f64_x : RuntimeLibcallImpl<ACOSH_NXV2F64>;
  def armpl_svasin_f32_x : RuntimeLibcallImpl<ASIN_NXV4F32>;
  def armpl_svasin_f64_x : RuntimeLibcallImpl<ASIN_NXV2F64>;
  def armpl_svasinh_f32_x : RuntimeLibcallImpl<ASINH_NXV4F32>;
  def armpl_svasinh_f64_x : RuntimeLibcallImpl<ASINH_NXV2F64>;
  def armpl_svatan2_f32_x : RuntimeLibcallImpl<ATAN2_NXV4F32>;
  def armpl_svatan2_f64_x : RuntimeLibcallImpl<ATAN2_NXV2F64>;
  def armpl_svatan_f32_x : RuntimeLibcallImpl<ATAN_NXV4F32>;
  def armpl_svatan_f64_x : RuntimeLibcallImpl<ATAN_NXV2F64>;
  def armpl_svatanh_f32_x : RuntimeLibcallImpl<ATANH_NXV4F32>;
  def armpl_svatanh_f64_x : RuntimeLibcallImpl<ATANH_NXV2F64>;
  def armpl_svcbrt_f32_x : RuntimeLibcallImpl<CBRT_NXV4F32>;
  def armpl_svcbrt_f64_x : RuntimeLibcallImpl<CBRT_NXV2F64>;
  def armpl_svcopysign_f32_x : RuntimeLibcallImpl<COPYSIGN_NXV4F32>;
  def armpl_svcopysign_f64_x : RuntimeLibcallImpl<COPYSIGN_NXV2F64>;
  def armpl_svcos_f32_x : RuntimeLibcallImpl<COS_NXV4F32>;
  def armpl_svcos_f64_x : RuntimeLibcallImpl<COS_NXV2F64>;
  def armpl_svcosh_f32_x : RuntimeLibcallImpl<COSH_NXV4F32>;
  def armpl_svcosh_f64_x : RuntimeLibcallImpl<COSH_NXV2F64>;
  def armpl_svcospi_f32_x : RuntimeLibcallImpl<COSPI_NXV4F32>;
  def armpl_svcospi_f64_x : RuntimeLibcallImpl<COSPI_NXV2F64>;
  def armpl_sverf_f32_x : RuntimeLibcallImpl<ERF_NXV4F32>;
  def armpl_sverf_f64_x : RuntimeLibcallImpl<ERF_NXV2F64>;
  def armpl_sverfc_f32_x : RuntimeLibcallImpl<ERFC_NXV4F32>;
  def armpl_sverfc_f64_x : RuntimeLibcallImpl<ERFC_NXV2F64>;
  def armpl_svexp10_f32_x : RuntimeLibcallImpl<EXP10_NXV4F32>;
  def armpl_svexp10_f64_x : RuntimeLibcallImpl<EXP10_NXV2F64>;
  def armpl_svexp2_f32_x : RuntimeLibcallImpl<EXP2_NXV4F32>;
  def armpl_svexp2_f64_x : RuntimeLibcallImpl<EXP2_NXV2F64>;
  def armpl_svexp_f32_x : RuntimeLibcallImpl<EXP_NXV4F32>;
  def armpl_svexp_f64_x : RuntimeLibcallImpl<EXP_NXV2F64>;
  def armpl_svexpm1_f32_x : RuntimeLibcallImpl<EXPM1_NXV4F32>;
  def armpl_svexpm1_f64_x : RuntimeLibcallImpl<EXPM1_NXV2F64>;
  def armpl_svfdim_f32_x : RuntimeLibcallImpl<FDIM_NXV4F32>;
  def armpl_svfdim_f64_x : RuntimeLibcallImpl<FDIM_NXV2F64>;
  def armpl_svfma_f32_x : RuntimeLibcallImpl<FMA_NXV4F32>;
  def armpl_svfma_f64_x : RuntimeLibcallImpl<FMA_NXV2F64>;
  def armpl_svfmax_f32_x : RuntimeLibcallImpl<FMAX_NXV4F32>;
  def armpl_svfmax_f64_x : RuntimeLibcallImpl<FMAX_NXV2F64>;
  def armpl_svfmin_f32_x : RuntimeLibcallImpl<FMIN_NXV4F32>;
  def armpl_svfmin_f64_x : RuntimeLibcallImpl<FMIN_NXV2F64>;
  def armpl_svfmod_f32_x : RuntimeLibcallImpl<REM_NXV4F32>;
  def armpl_svfmod_f64_x : RuntimeLibcallImpl<REM_NXV2F64>;
  def armpl_svhypot_f32_x : RuntimeLibcallImpl<HYPOT_NXV4F32>;
  def armpl_svhypot_f64_x : RuntimeLibcallImpl<HYPOT_NXV2F64>;
  def armpl_svilogb_f32_x : RuntimeLibcallImpl<ILOGB_NXV4F32>;
  def armpl_svilogb_f64_x : RuntimeLibcallImpl<ILOGB_NXV2F64>;
  def armpl_svldexp_f32_x : RuntimeLibcallImpl<LDEXP_NXV4F32>;
  def armpl_svldexp_f64_x : RuntimeLibcallImpl<LDEXP_NXV2F64>;
  def armpl_svlgamma_f32_x : RuntimeLibcallImpl<LGAMMA_NXV4F32>;
  def armpl_svlgamma_f64_x : RuntimeLibcallImpl<LGAMMA_NXV2F64>;
  def armpl_svlog10_f32_x : RuntimeLibcallImpl<LOG10_NXV4F32>;
  def armpl_svlog10_f64_x : RuntimeLibcallImpl<LOG10_NXV2F64>;
  def armpl_svlog1p_f32_x : RuntimeLibcallImpl<LOG1P_NXV4F32>;
  def armpl_svlog1p_f64_x : RuntimeLibcallImpl<LOG1P_NXV2F64>;
  def armpl_svlog2_f32_x : RuntimeLibcallImpl<LOG2_NXV4F32>;
  def armpl_svlog2_f64_x : RuntimeLibcallImpl<LOG2_NXV2F64>;
  def armpl_svlog_f32_x : RuntimeLibcallImpl<LOG_NXV4F32>;
  def armpl_svlog_f64_x : RuntimeLibcallImpl<LOG_NXV2F64>;
  def armpl_svmodf_f32_x : RuntimeLibcallImpl<MODF_NXV4F32>;
  def armpl_svmodf_f64_x : RuntimeLibcallImpl<MODF_NXV2F64>;
  def armpl_svnextafter_f32_x : RuntimeLibcallImpl<NEXTAFTER_NXV4F32>;
  def armpl_svnextafter_f64_x : RuntimeLibcallImpl<NEXTAFTER_NXV2F64>;
  def armpl_svpow_f32_x : RuntimeLibcallImpl<POW_NXV4F32>;
  def armpl_svpow_f64_x : RuntimeLibcallImpl<POW_NXV2F64>;
  def armpl_svsin_f32_x : RuntimeLibcallImpl<SIN_NXV4F32>;
  def armpl_svsin_f64_x : RuntimeLibcallImpl<SIN_NXV2F64>;
  def armpl_svsincos_f32_x : RuntimeLibcallImpl<SINCOS_NXV4F32>;
  def armpl_svsincos_f64_x : RuntimeLibcallImpl<SINCOS_NXV2F64>;
  def armpl_svsincospi_f32_x : RuntimeLibcallImpl<SINCOSPI_NXV4F32>;
  def armpl_svsincospi_f64_x : RuntimeLibcallImpl<SINCOSPI_NXV2F64>;
  def armpl_svsinh_f32_x : RuntimeLibcallImpl<SINH_NXV4F32>;
  def armpl_svsinh_f64_x : RuntimeLibcallImpl<SINH_NXV2F64>;
  def armpl_svsinpi_f32_x : RuntimeLibcallImpl<SINPI_NXV4F32>;
  def armpl_svsinpi_f64_x : RuntimeLibcallImpl<SINPI_NXV2F64>;
  def armpl_svsqrt_f32_x : RuntimeLibcallImpl<SQRT_NXV4F32>;
  def armpl_svsqrt_f64_x : RuntimeLibcallImpl<SQRT_NXV2F64>;
  def armpl_svtan_f32_x : RuntimeLibcallImpl<TAN_NXV4F32>;
  def armpl_svtan_f64_x : RuntimeLibcallImpl<TAN_NXV2F64>;
  def armpl_svtanh_f32_x : RuntimeLibcallImpl<TANH_NXV4F32>;
  def armpl_svtanh_f64_x : RuntimeLibcallImpl<TANH_NXV2F64>;
  def armpl_svtgamma_f32_x : RuntimeLibcallImpl<TGAMMA_NXV4F32>;
  def armpl_svtgamma_f64_x : RuntimeLibcallImpl<TGAMMA_NXV2F64>;
  def armpl_vacoshq_f32 : RuntimeLibcallImpl<ACOSH_V4F32>;
  def armpl_vacoshq_f64 : RuntimeLibcallImpl<ACOSH_V2F64>;
  def armpl_vacosq_f32 : RuntimeLibcallImpl<ACOS_V4F32>;
  def armpl_vacosq_f64 : RuntimeLibcallImpl<ACOS_V2F64>;
  def armpl_vasinhq_f32 : RuntimeLibcallImpl<ASINH_V4F32>;
  def armpl_vasinhq_f64 : RuntimeLibcallImpl<ASINH_V2F64>;
  def armpl_vasinq_f32 : RuntimeLibcallImpl<ASIN_V4F32>;
  def armpl_vasinq_f64 : RuntimeLibcallImpl<ASIN_V2F64>;
  def armpl_vatan2q_f32 : RuntimeLibcallImpl<ATAN2_V4F32>;
  def armpl_vatan2q_f64 : RuntimeLibcallImpl<ATAN2_V2F64>;
  def armpl_vatanhq_f32 : RuntimeLibcallImpl<ATANH_V4F32>;
  def armpl_vatanhq_f64 : RuntimeLibcallImpl<ATANH_V2F64>;
  def armpl_vatanq_f32 : RuntimeLibcallImpl<ATAN_V4F32>;
  def armpl_vatanq_f64 : RuntimeLibcallImpl<ATAN_V2F64>;
  def armpl_vcbrtq_f32 : RuntimeLibcallImpl<CBRT_V4F32>;
  def armpl_vcbrtq_f64 : RuntimeLibcallImpl<CBRT_V2F64>;
  def armpl_vcopysignq_f32 : RuntimeLibcallImpl<COPYSIGN_V4F32>;
  def armpl_vcopysignq_f64 : RuntimeLibcallImpl<COPYSIGN_V2F64>;
  def armpl_vcoshq_f32 : RuntimeLibcallImpl<COSH_V4F32>;
  def armpl_vcoshq_f64 : RuntimeLibcallImpl<COSH_V2F64>;
  def armpl_vcospiq_f32 : RuntimeLibcallImpl<COSPI_V4F32>;
  def armpl_vcospiq_f64 : RuntimeLibcallImpl<COSPI_V2F64>;
  def armpl_vcosq_f32 : RuntimeLibcallImpl<COS_V4F32>;
  def armpl_vcosq_f64 : RuntimeLibcallImpl<COS_V2F64>;
  def armpl_verfcq_f32 : RuntimeLibcallImpl<ERFC_V4F32>;
  def armpl_verfcq_f64 : RuntimeLibcallImpl<ERFC_V2F64>;
  def armpl_verfq_f32 : RuntimeLibcallImpl<ERF_V4F32>;
  def armpl_verfq_f64 : RuntimeLibcallImpl<ERF_V2F64>;
  def armpl_vexp10q_f32 : RuntimeLibcallImpl<EXP10_V4F32>;
  def armpl_vexp10q_f64 : RuntimeLibcallImpl<EXP10_V2F64>;
  def armpl_vexp2q_f32 : RuntimeLibcallImpl<EXP2_V4F32>;
  def armpl_vexp2q_f64 : RuntimeLibcallImpl<EXP2_V2F64>;
  def armpl_vexpm1q_f32 : RuntimeLibcallImpl<EXPM1_V4F32>;
  def armpl_vexpm1q_f64 : RuntimeLibcallImpl<EXPM1_V2F64>;
  def armpl_vexpq_f32 : RuntimeLibcallImpl<EXP_V4F32>;
  def armpl_vexpq_f64 : RuntimeLibcallImpl<EXP_V2F64>;
  def armpl_vfdimq_f32 : RuntimeLibcallImpl<FDIM_V4F32>;
  def armpl_vfdimq_f64 : RuntimeLibcallImpl<FDIM_V2F64>;
  def armpl_vfmaq_f32 : RuntimeLibcallImpl<FMA_V4F32>;
  def armpl_vfmaq_f64 : RuntimeLibcallImpl<FMA_V2F64>;
  def armpl_vfmaxq_f32 : RuntimeLibcallImpl<FMAX_V4F32>;
  def armpl_vfmaxq_f64 : RuntimeLibcallImpl<FMAX_V2F64>;
  def armpl_vfminq_f32 : RuntimeLibcallImpl<FMIN_V4F32>;
  def armpl_vfminq_f64 : RuntimeLibcallImpl<FMIN_V2F64>;
  def armpl_vfmodq_f32 : RuntimeLibcallImpl<REM_V4F32>;
  def armpl_vfmodq_f64 : RuntimeLibcallImpl<REM_V2F64>;
  def armpl_vhypotq_f32 : RuntimeLibcallImpl<HYPOT_V4F32>;
  def armpl_vhypotq_f64 : RuntimeLibcallImpl<HYPOT_V2F64>;
  def armpl_vilogbq_f32 : RuntimeLibcallImpl<ILOGB_V4F32>;
  def armpl_vilogbq_f64 : RuntimeLibcallImpl<ILOGB_V2F64>;
  def armpl_vldexpq_f32 : RuntimeLibcallImpl<LDEXP_V4F32>;
  def armpl_vldexpq_f64 : RuntimeLibcallImpl<LDEXP_V2F64>;
  def armpl_vlgammaq_f32 : RuntimeLibcallImpl<LGAMMA_V4F32>;
  def armpl_vlgammaq_f64 : RuntimeLibcallImpl<LGAMMA_V2F64>;
  def armpl_vlog10q_f32 : RuntimeLibcallImpl<LOG10_V4F32>;
  def armpl_vlog10q_f64 : RuntimeLibcallImpl<LOG10_V2F64>;
  def armpl_vlog1pq_f32 : RuntimeLibcallImpl<LOG1P_V4F32>;
  def armpl_vlog1pq_f64 : RuntimeLibcallImpl<LOG1P_V2F64>;
  def armpl_vlog2q_f32 : RuntimeLibcallImpl<LOG2_V4F32>;
  def armpl_vlog2q_f64 : RuntimeLibcallImpl<LOG2_V2F64>;
  def armpl_vlogq_f32 : RuntimeLibcallImpl<LOG_V4F32>;
  def armpl_vlogq_f64 : RuntimeLibcallImpl<LOG_V2F64>;
  def armpl_vmodfq_f32 : RuntimeLibcallImpl<MODF_V4F32>;
  def armpl_vmodfq_f64 : RuntimeLibcallImpl<MODF_V2F64>;
  def armpl_vnextafterq_f32 : RuntimeLibcallImpl<NEXTAFTER_V4F32>;
  def armpl_vnextafterq_f64 : RuntimeLibcallImpl<NEXTAFTER_V2F64>;
  def armpl_vpowq_f32 : RuntimeLibcallImpl<POW_V4F32>;
  def armpl_vpowq_f64 : RuntimeLibcallImpl<POW_V2F64>;
  def armpl_vsincospiq_f32 : RuntimeLibcallImpl<SINCOSPI_V4F32>;
  def armpl_vsincospiq_f64 : RuntimeLibcallImpl<SINCOSPI_V2F64>;
  def armpl_vsincosq_f32 : RuntimeLibcallImpl<SINCOS_V4F32>;
  def armpl_vsincosq_f64 : RuntimeLibcallImpl<SINCOS_V2F64>;
  def armpl_vsinhq_f32 : RuntimeLibcallImpl<SINH_V4F32>;
  def armpl_vsinhq_f64 : RuntimeLibcallImpl<SINH_V2F64>;
  def armpl_vsinpiq_f32 : RuntimeLibcallImpl<SINPI_V4F32>;
  def armpl_vsinpiq_f64 : RuntimeLibcallImpl<SINPI_V2F64>;
  def armpl_vsinq_f32 : RuntimeLibcallImpl<SIN_V4F32>;
  def armpl_vsinq_f64 : RuntimeLibcallImpl<SIN_V2F64>;
  def armpl_vsqrtq_f32 : RuntimeLibcallImpl<SQRT_V4F32>;
  def armpl_vsqrtq_f64 : RuntimeLibcallImpl<SQRT_V2F64>;
  def armpl_vtanhq_f32 : RuntimeLibcallImpl<TANH_V4F32>;
  def armpl_vtanhq_f64 : RuntimeLibcallImpl<TANH_V2F64>;
  def armpl_vtanq_f32 : RuntimeLibcallImpl<TAN_V4F32>;
  def armpl_vtanq_f64 : RuntimeLibcallImpl<TAN_V2F64>;
  def armpl_vtgammaq_f32 : RuntimeLibcallImpl<TGAMMA_V4F32>;
  def armpl_vtgammaq_f64 : RuntimeLibcallImpl<TGAMMA_V2F64>;
}

//===----------------------------------------------------------------------===//
// AMD vector math library (AMDLIBM) functions
//===----------------------------------------------------------------------===//

defset list<RuntimeLibcallImpl> AMDLIBM_VECFUNCS = {
  def amd_vrd2_atan : RuntimeLibcallImpl<ATAN_V2F64>;
  def amd_vrd2_cbrt : RuntimeLibcallImpl<CBRT_V2F64>;
  def amd_vrd2_cos : RuntimeLibcallImpl<COS_V2F64>;
  def amd_vrd2_erf : RuntimeLibcallImpl<ERF_V2F64>;
  def amd_vrd2_exp : RuntimeLibcallImpl<EXP_V2F64>;
  def amd_vrd2_exp10 : RuntimeLibcallImpl<EXP10_V2F64>;
  def amd_vrd2_exp2 : RuntimeLibcallImpl<EXP2_V2F64>;
  def amd_vrd2_expm1 : RuntimeLibcallImpl<EXPM1_V2F64>;
  def amd_vrd2_log : RuntimeLibcallImpl<LOG_V2F64>;
  def amd_vrd2_log10 : RuntimeLibcallImpl<LOG10_V2F64>;
  def amd_vrd2_log1p : RuntimeLibcallImpl<LOG1P_V2F64>;
  def amd_vrd2_log2 : RuntimeLibcallImpl<LOG2_V2F64>;
  def amd_vrd2_pow : RuntimeLibcallImpl<POW_V2F64>;
  def amd_vrd2_sin : RuntimeLibcallImpl<SIN_V2F64>;
  def amd_vrd2_tan : RuntimeLibcallImpl<TAN_V2F64>;
  def amd_vrd4_atan : RuntimeLibcallImpl<ATAN_V4F64>;
  def amd_vrd4_cos : RuntimeLibcallImpl<COS_V4F64>;
  def amd_vrd4_erf : RuntimeLibcallImpl<ERF_V4F32>;
  def amd_vrd4_exp : RuntimeLibcallImpl<EXP_V4F64>;
  def amd_vrd4_exp2 : RuntimeLibcallImpl<EXP2_V4F64>;
  def amd_vrd4_log : RuntimeLibcallImpl<LOG_V4F64>;
  def amd_vrd4_log2 : RuntimeLibcallImpl<LOG2_V4F64>;
  def amd_vrd4_pow : RuntimeLibcallImpl<POW_V4F64>;
  def amd_vrd4_sin : RuntimeLibcallImpl<SIN_V4F64>;
  def amd_vrd4_sincos : RuntimeLibcallImpl<SINCOS_V4F64>;
  def amd_vrd4_tan : RuntimeLibcallImpl<TAN_V4F64>;
  def amd_vrd8_asin : RuntimeLibcallImpl<ASIN_V8F64>;
  def amd_vrd8_atan : RuntimeLibcallImpl<ATAN_V8F64>;
  def amd_vrd8_cos : RuntimeLibcallImpl<COS_V8F64>;
  def amd_vrd8_erf : RuntimeLibcallImpl<ERF_V8F32>;
  def amd_vrd8_exp : RuntimeLibcallImpl<EXP_V8F64>;
  def amd_vrd8_exp2 : RuntimeLibcallImpl<EXP2_V8F64>;
  def amd_vrd8_log : RuntimeLibcallImpl<LOG_V8F64>;
  def amd_vrd8_log2 : RuntimeLibcallImpl<LOG2_V8F64>;
  def amd_vrd8_pow : RuntimeLibcallImpl<POW_V8F64>;
  def amd_vrd8_sin : RuntimeLibcallImpl<SIN_V8F64>;
  def amd_vrd8_sincos : RuntimeLibcallImpl<SINCOS_V8F64>;
  def amd_vrd8_tan : RuntimeLibcallImpl<TAN_V8F64>;
  def amd_vrs16_acosf : RuntimeLibcallImpl<ACOS_V16F32>;
  def amd_vrs16_asinf : RuntimeLibcallImpl<ASIN_V16F32>;
  def amd_vrs16_atanf : RuntimeLibcallImpl<ATAN_V16F32>;
  def amd_vrs16_cosf : RuntimeLibcallImpl<COS_V16F32>;
  def amd_vrs16_erff : RuntimeLibcallImpl<ERF_V16F32>;
  def amd_vrs16_exp2f : RuntimeLibcallImpl<EXP2_V16F32>;
  def amd_vrs16_expf : RuntimeLibcallImpl<EXP_V16F32>;
  def amd_vrs16_log10f : RuntimeLibcallImpl<LOG10_V16F32>;
  def amd_vrs16_log2f : RuntimeLibcallImpl<LOG2_V16F32>;
  def amd_vrs16_logf : RuntimeLibcallImpl<LOG_V16F32>;
  def amd_vrs16_powf : RuntimeLibcallImpl<POW_V16F32>;
  def amd_vrs16_sincosf : RuntimeLibcallImpl<SINCOS_V16F32>;
  def amd_vrs16_sinf : RuntimeLibcallImpl<SIN_V16F32>;
  def amd_vrs16_tanf : RuntimeLibcallImpl<TAN_V16F32>;
  def amd_vrs16_tanhf : RuntimeLibcallImpl<TANH_V16F32>;
  def amd_vrs4_acosf : RuntimeLibcallImpl<ACOS_V4F32>;
  def amd_vrs4_asinf : RuntimeLibcallImpl<ASIN_V4F32>;
  def amd_vrs4_atanf : RuntimeLibcallImpl<ATAN_V4F32>;
  def amd_vrs4_cbrtf : RuntimeLibcallImpl<CBRT_V4F32>;
  def amd_vrs4_cosf : RuntimeLibcallImpl<COS_V4F32>;
  def amd_vrs4_coshf : RuntimeLibcallImpl<COSH_V4F32>;
  def amd_vrs4_erff : RuntimeLibcallImpl<ERF_V4F32>;
  def amd_vrs4_exp10f : RuntimeLibcallImpl<EXP10_V4F32>;
  def amd_vrs4_exp2f : RuntimeLibcallImpl<EXP2_V4F32>;
  def amd_vrs4_expf : RuntimeLibcallImpl<EXP_V4F32>;
  def amd_vrs4_expm1f : RuntimeLibcallImpl<EXPM1_V4F32>;
  def amd_vrs4_log10f : RuntimeLibcallImpl<LOG10_V4F32>;
  def amd_vrs4_log1pf : RuntimeLibcallImpl<LOG1P_V4F32>;
  def amd_vrs4_log2f : RuntimeLibcallImpl<LOG2_V4F32>;
  def amd_vrs4_logf : RuntimeLibcallImpl<LOG_V4F32>;
  def amd_vrs4_powf : RuntimeLibcallImpl<POW_V4F32>;
  def amd_vrs4_sincosf : RuntimeLibcallImpl<SINCOS_V4F32>;
  def amd_vrs4_sinf : RuntimeLibcallImpl<SIN_V4F32>;
  def amd_vrs4_tanf : RuntimeLibcallImpl<TAN_V4F32>;
  def amd_vrs4_tanhf : RuntimeLibcallImpl<TANH_V4F32>;
  def amd_vrs8_acosf : RuntimeLibcallImpl<ACOS_V8F32>;
  def amd_vrs8_asinf : RuntimeLibcallImpl<ASIN_V8F32>;
  def amd_vrs8_atanf : RuntimeLibcallImpl<ATAN_V8F32>;
  def amd_vrs8_cosf : RuntimeLibcallImpl<COS_V8F32>;
  def amd_vrs8_coshf : RuntimeLibcallImpl<COSH_V8F32>;
  def amd_vrs8_erff : RuntimeLibcallImpl<ERF_V8F32>;
  def amd_vrs8_exp2f : RuntimeLibcallImpl<EXP2_V8F32>;
  def amd_vrs8_expf : RuntimeLibcallImpl<EXP_V8F32>;
  def amd_vrs8_log10f : RuntimeLibcallImpl<LOG10_V8F32>;
  def amd_vrs8_log2f : RuntimeLibcallImpl<LOG2_V8F32>;
  def amd_vrs8_logf : RuntimeLibcallImpl<LOG_V8F32>;
  def amd_vrs8_powf : RuntimeLibcallImpl<POW_V8F32>;
  def amd_vrs8_sincosf : RuntimeLibcallImpl<SINCOS_V8F32>;
  def amd_vrs8_sinf : RuntimeLibcallImpl<SIN_V8F32>;
  def amd_vrs8_tanf : RuntimeLibcallImpl<TAN_V8F32>;
  def amd_vrs8_tanhf : RuntimeLibcallImpl<TANH_V8F32>;
}
