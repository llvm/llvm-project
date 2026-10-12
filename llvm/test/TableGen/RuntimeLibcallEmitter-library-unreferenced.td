// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// A LibcallLibrary is always emitted, even if no SystemRuntimeLibrary
// references it.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def SQRT_F64 : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;

def IsX86 : LibcallPredicate<[{TT.isX86()}]>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;

def ExtDefaultCC : LibcallCallingConv<
  [{TT.isX86() ? static_cast<CallingConv::ID>(CallingConv::Fast) : static_cast<CallingConv::ID>(DefaultCC)}]
>;

def Libc : LibcallLibrary<"libc", (add memcpy)>;
def ExtLib : LibcallLibrary<"ext", (add LibcallsWithCC<(add sqrt), ExtDefaultCC>)>;

def X86System : SystemRuntimeLibrary<isX86, (add Libc)>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_ext(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK: setLibcallImplCallingConv(Impl, TT.isX86() ? static_cast<CallingConv::ID>(CallingConv::Fast) : static_cast<CallingConv::ID>(DefaultCC));

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(

// Only the referenced library is dispatched.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(
// CHECK: if (isLibraryAvailable(RTLIB::RuntimeLibrary::libc))
// CHECK-NEXT: setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-NOT: setAvailableLibFuncs_ext(
