// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check that a SystemRuntimeLibrary naming LibcallLibrary defs dispatches to
// their setAvailableLibFuncs_<name> functions under an isLibraryAvailable guard,
// inside the target's triple block, while bare (non-library) members are still
// emitted inline via the flat path.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def SQRT_F64 : RuntimeLibcall;
def EXTRA : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;
def extra : RuntimeLibcallImpl<EXTRA>;

def IsX86 : LibcallPredicate<[{TT.isX86()}]>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;

def Libc : LibcallLibrary<"libc", (add memcpy)>;
def Libm : LibcallLibrary<"libm", (add sqrt)>;

// Pulls two named libraries plus one bare inline member.
def X86System : SystemRuntimeLibrary<isX86, (add Libc, Libm, extra)>;

// The named libraries still get their standalone implementation functions.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(
// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libm(

// The driver dispatches to each named library under its presence guard, then
// emits the bare member (extra) inline.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK:   if (TT.isX86()) {
// CHECK-NEXT:     static constexpr LibcallImplBitset SystemAvailableImpls({
// CHECK: AvailableLibcallImpls = SystemAvailableImpls;
// CHECK-EMPTY:
// CHECK-NEXT:     if (isLibraryAvailable("libc"))
// CHECK-NEXT:       setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, EABIVersion, ABIName, LongDoubleFormat);
// CHECK-NEXT:     if (isLibraryAvailable("libm"))
// CHECK-NEXT:       setAvailableLibFuncs_libm(TT, ExceptionModel, FloatABI, EABIVersion, ABIName, LongDoubleFormat);
// CHECK-EMPTY:
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_extra, // extra
// CHECK-NEXT:     };
// CHECK: for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK:     return;
// CHECK-NEXT:   }
