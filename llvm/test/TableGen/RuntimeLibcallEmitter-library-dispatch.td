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

def X86System : SystemRuntimeLibrary<isX86, ["libc", "libm"]>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(
// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libm(

// CHECK-LABEL: ArrayRef<RTLIB::RuntimeLibrary> llvm::RTLIB::RuntimeLibcallsInfo::getDefaultRuntimeLibraries(const llvm::Triple &TT) {
// CHECK-NEXT:   if (TT.isX86()) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libc,
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libm,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   return {};
// CHECK-NEXT: }
