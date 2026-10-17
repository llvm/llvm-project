// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check that a SystemRuntimeLibrary naming LibcallLibrary defs dispatches to
// their setAvailableLibFuncs_<name> functions under an isLibraryAvailable guard,
// inside the target's triple block.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def SQRT_F64 : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;

def IsX86 : LibcallPredicate<[{TT.isX86()}]>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;

def Libc : LibcallLibrary<"libc", (add memcpy)>;
def Libm : LibcallLibrary<"libm", (add sqrt)>;

def X86System : SystemRuntimeLibrary<isX86, (add Libc, Libm)>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(
// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libm(

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK:   if (TT.isX86()) {
// CHECK-NEXT:     if (isLibraryAvailable(RTLIB::RuntimeLibrary::libc))
// CHECK-NEXT:       setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-NEXT:     if (isLibraryAvailable(RTLIB::RuntimeLibrary::libm))
// CHECK-NEXT:       setAvailableLibFuncs_libm(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-EMPTY:
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
