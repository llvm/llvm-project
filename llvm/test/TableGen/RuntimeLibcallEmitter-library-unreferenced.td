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

def X86System : SystemRuntimeLibrary<isX86, ["libc"]>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_ext(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK: setLibcallImplCallingConv(Impl, TT.isX86() ? static_cast<CallingConv::ID>(CallingConv::Fast) : static_cast<CallingConv::ID>(DefaultCC));

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(

// Only the referenced library is a default library.

// CHECK-LABEL: ArrayRef<RTLIB::RuntimeLibrary> llvm::RTLIB::RuntimeLibcallsInfo::getDefaultRuntimeLibraries(const llvm::Triple &TT) {
// CHECK-NEXT:   if (TT.isX86()) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libc,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   return {};
// CHECK-NEXT: }
