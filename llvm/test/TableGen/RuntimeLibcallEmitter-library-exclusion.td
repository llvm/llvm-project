// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check that a LibraryExclusion emits the excluded impls as setUnavailable
// inside the named library function, guarded on the exclusion's predicate, so
// the system dispatch stays a plain call.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def DIV_I32 : RuntimeLibcall;
def MOD_I32 : RuntimeLibcall;
def SQRT_F64 : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def __divsi3 : RuntimeLibcallImpl<DIV_I32>;
def __modsi3 : RuntimeLibcallImpl<MOD_I32>;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;

def IsX86 : LibcallPredicate<[{TT.isX86()}]>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;

def CompilerRt : LibcallLibrary<"compiler-rt", (add memcpy, __divsi3, __modsi3)>;
def Libm : LibcallLibrary<"libm", (add sqrt)>;

def X86System : SystemRuntimeLibrary<isX86, ["compiler-rt", "libm"]>;

// Drop the two div/mod entries this target overrides.
def : LibraryExclusion<"compiler-rt", isX86, [__divsi3, __modsi3]>;

// The compiler-rt library function sets its members available, then emits the
// X86 opt-out under a triple guard at the end of the same function.
// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_compiler_rt(
// CHECK: for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT: setAvailable(Impl);
// CHECK: if (TT.isX86()) {
// CHECK-NEXT: setUnavailable(RTLIB::impl___divsi3); // __divsi3
// CHECK-NEXT: setUnavailable(RTLIB::impl___modsi3); // __modsi3
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libm(

// The system enables both libraries by name; no setUnavailable here.
// CHECK-LABEL: ArrayRef<RTLIB::RuntimeLibrary> llvm::RTLIB::RuntimeLibcallsInfo::getDefaultRuntimeLibraries(const llvm::Triple &TT) {
// CHECK-NEXT:   if (TT.isX86()) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::compiler_rt,
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libm,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   return {};
// CHECK-NEXT: }
