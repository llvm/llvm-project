// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Each LibraryName gets an enumerator. Only libraries no SystemRuntimeLibrary
// references get a setAvailableLibraryLibcalls case.

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
def X86SystemLibrary : SystemRuntimeLibrary<isX86, (add X86Libm)>;
def MSP430SystemLibrary : SystemRuntimeLibrary<isMSP430, (add Libm)> {
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

// CHECK-LABEL: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(
// CHECK-NOT: veclib
// CHECK: setAvailableLibFuncs_X86Libm(
// CHECK-NOT: veclib
// CHECK: }

// CHECK-LABEL: bool llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibraryLibcalls(RTLIB::RuntimeLibrary Library, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   switch (Library) {
// CHECK-NEXT:   case RTLIB::RuntimeLibrary::veclib_a:
// CHECK-NEXT:     setAvailableLibFuncs_veclib_a(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   case RTLIB::RuntimeLibrary::veclib_b:
// CHECK-NEXT:     setAvailableLibFuncs_veclib_b(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   default:
// CHECK-NEXT:     return false;
// CHECK-NEXT:   }
// CHECK-NEXT: }
