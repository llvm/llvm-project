// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check that when a target excludes some impls from a shared library via
// LibraryRef AND re-adds them through its own same-name library variant, the
// variant's adds are emitted AFTER the exclusion's setUnavailable calls. This
// lets a target's own re-adds win over its own opt-out, while the exclusion
// still suppresses those impls for any other variant's contribution.

include "llvm/IR/RuntimeLibcallsImpl.td"

def ADD_F32 : RuntimeLibcall;
def SUB_F32 : RuntimeLibcall;
def SQRT_F64 : RuntimeLibcall;

def __addsf3 : RuntimeLibcallImpl<ADD_F32>;
def __subsf3 : RuntimeLibcallImpl<SUB_F32>;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;

def IsARM : LibcallPredicate<[{TT.isARM()}]>;
def IsOSWindows : LibcallPredicate<[{TT.isOSWindows()}]>;
def isARM : RuntimeLibcallAvailability<(all_of IsARM)>;
def isNotOSWindows : RuntimeLibcallAvailability<(not IsOSWindows)>;

// The shared compiler-rt core provides __addsf3/__subsf3 unconditionally.
def CompilerRt : LibcallLibrary<"compiler-rt", (add __addsf3, __subsf3)>;
def Libm : LibcallLibrary<"libm", (add sqrt)>;

// ARM's own compiler-rt variant re-adds __addsf3 on non-Windows ARM
// only. The per-member predicate keeps it out of the shared core so
// it stays a distinct re-add.
def ARMCompilerRt : LibcallLibrary<"compiler-rt",
    (add LibcallImpls<(add __addsf3), isNotOSWindows>), isARM>;

// ARM excludes __addsf3 from the shared core (all ARM), then re-adds
// it via ARMCompilerRt on non-Windows ARM.
def ARMSystem : SystemRuntimeLibrary<isARM,
    (add LibraryRef<CompilerRt, [__addsf3]>, Libm)>;

// In the merged compiler-rt function: the shared core runs first,
// then the exclusion's setUnavailable, then ARM's own variant re-add
// (deferred past the exclusion because it re-adds an excluded impl on
// the same consumer).
//
// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_compiler_rt(
// CHECK: RTLIB::impl___addsf3, // __addsf3
// CHECK-NEXT: RTLIB::impl___subsf3, // __subsf3
// CHECK: setAvailable(Impl);
// CHECK: if (TT.isARM()) {
// CHECK-NEXT: setUnavailable(RTLIB::impl___addsf3); // __addsf3
// CHECK-NEXT: }
// CHECK: if (TT.isARM()) {
// CHECK-NEXT: if (!(TT.isOSWindows())) {
// CHECK-NEXT: static const RTLIB::LibcallImpl LibraryCalls_isNotOSWindows[] = {
// CHECK-NEXT: RTLIB::impl___addsf3, // __addsf3
// CHECK: setAvailable(Impl);
