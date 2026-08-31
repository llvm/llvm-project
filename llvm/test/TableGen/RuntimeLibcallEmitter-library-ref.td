// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check that a LibraryRef<Lib, [impls]> member dispatches to the library like a
// plain reference, with the excluded impls emitted as setUnavailable inside the
// library function (guarded on the consumer's triple), so the dispatcher stays a
// plain call.

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

// Dispatch compiler-rt but drop the two div/mod entries this target overrides;
// dispatch libm plainly.
def X86System : SystemRuntimeLibrary<isX86,
    (add LibraryRef<CompilerRt, [__divsi3, __modsi3]>, Libm)>;

// The compiler-rt library function sets its members available, then emits the
// X86 consumer's opt-out under a triple guard at the end of the same function.
// CHECK: static void setAvailableLibFuncs_compiler_rt(
// CHECK: for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT: Info.setAvailable(Impl);
// CHECK: if (TT.isX86()) {
// CHECK-NEXT: Info.setUnavailable(RTLIB::impl___divsi3); // __divsi3
// CHECK-NEXT: Info.setUnavailable(RTLIB::impl___modsi3); // __modsi3
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: static void setAvailableLibFuncs_libm(

// The dispatcher is a plain call for both libraries; no setUnavailable here.
// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(
// CHECK:   if (TT.isX86()) {
// CHECK: AvailableLibcallImpls = SystemAvailableImpls;
// CHECK-EMPTY:
// CHECK-NEXT:     if (isLibraryAvailable("compiler-rt"))
// CHECK-NEXT:       setAvailableLibFuncs_compiler_rt(*this, TT, ExceptionModel, FloatABI, EABIVersion, ABIName, LongDoubleFormat);
// CHECK-NEXT:     if (isLibraryAvailable("libm"))
// CHECK-NEXT:       setAvailableLibFuncs_libm(*this, TT, ExceptionModel, FloatABI, EABIVersion, ABIName, LongDoubleFormat);
// CHECK-NOT:   setUnavailable
