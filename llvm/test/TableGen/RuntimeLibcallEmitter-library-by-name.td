// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Each LibraryName gets an enumerator and a setAvailableLibraryLibcalls case.
// Systems dispatch through it, and an isolated variant is selected by its own
// predicate.

include "llvm/IR/RuntimeLibcallsImpl.td"

def SIN_V2F64 : RuntimeLibcall;
def COS_V2F64 : RuntimeLibcall;
def SIN_F64 : RuntimeLibcall;

def sin : RuntimeLibcallImpl<SIN_F64>;
def vsin2 : RuntimeLibcallImpl<SIN_V2F64>;
def vcos2 : RuntimeLibcallImpl<COS_V2F64>;
def other_vsin2 : RuntimeLibcallImpl<SIN_V2F64>;

def IsX86 : LibcallPredicate<[{TT.getArch() == Triple::x86_64}]>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;
def IsMSP430 : LibcallPredicate<[{TT.getArch() == Triple::msp430}]>;
def isMSP430 : RuntimeLibcallAvailability<(all_of IsMSP430)>;

def VECTORCC : LibcallCallingConv<[{CallingConv::AArch64_VectorCall}]>;
def MSP430CC : LibcallCallingConv<[{CallingConv::MSP430_BUILTIN}]>;

def Libm : LibcallLibrary<"libm", (add sin)>;
def X86Libm : LibcallLibrary<"libm", (add sin), isX86> {
  let Isolated = true;
}
def X86SystemLibrary : SystemRuntimeLibrary<isX86, ["libm"]>;
def MSP430SystemLibrary : SystemRuntimeLibrary<isMSP430, ["libm"]> {
  let DefaultLibcallCallingConv = MSP430CC;
}

def VecLibA : LibcallLibrary<"veclib-a",
  (add vsin2, LibcallsWithCC<(add vcos2), VECTORCC>)>;
def VecLibB : LibcallLibrary<"veclib-b", (add other_vsin2)>;

// CHECK-LABEL: enum class RuntimeLibrary : unsigned char {
// CHECK-NEXT:   UnknownLibrary = 0,
// CHECK-NEXT:   libm,
// CHECK-NEXT:   veclib_a,
// CHECK-NEXT:   veclib_b,
// CHECK-NEXT: };

// CHECK: bool setAvailableLibraryLibcalls(RTLIB::RuntimeLibrary Library, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat);

// CHECK-LABEL: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_veclib_a(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK-NEXT:   {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_vsin2, // vsin2
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls_AlwaysAvailable_VECTORCC[] = {
// CHECK-NEXT:         RTLIB::impl_vcos2, // vcos2
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls_AlwaysAvailable_VECTORCC) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:       setLibcallImplCallingConv(Impl, CallingConv::AArch64_VectorCall);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: CallingConv::ID llvm::RTLIB::RuntimeLibcallsInfo::getDefaultLibcallCallingConv(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   if (TT.getArch() == Triple::msp430) {
// CHECK-NEXT:     return CallingConv::MSP430_BUILTIN;
// CHECK-NEXT:   }
// CHECK-NEXT:   return CallingConv::C;
// CHECK-NEXT: }

// CHECK-LABEL: ArrayRef<RTLIB::RuntimeLibrary> llvm::RTLIB::RuntimeLibcallsInfo::getDefaultRuntimeLibraries(const llvm::Triple &TT) {
// CHECK-NEXT:   if (TT.getArch() == Triple::msp430) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libm,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   if (TT.getArch() == Triple::x86_64) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libm,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   return {};
// CHECK-NEXT: }

// CHECK-LABEL: bool llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibraryLibcalls(RTLIB::RuntimeLibrary Library, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   const CallingConv::ID DefaultCC = getDefaultLibcallCallingConv(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat);
// CHECK-NEXT:   switch (Library) {
// CHECK-NEXT:   case RTLIB::RuntimeLibrary::libm:
// CHECK-NEXT:     if (TT.getArch() == Triple::x86_64) {
// CHECK-NEXT:       setAvailableLibFuncs_X86Libm(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:       return true;
// CHECK-NEXT:     }
// CHECK-NEXT:     setAvailableLibFuncs_libm(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   case RTLIB::RuntimeLibrary::veclib_a:
// CHECK-NEXT:     setAvailableLibFuncs_veclib_a(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   case RTLIB::RuntimeLibrary::veclib_b:
// CHECK-NEXT:     setAvailableLibFuncs_veclib_b(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   default:
// CHECK-NEXT:     return false;
// CHECK-NEXT:   }
// CHECK-NEXT: }
