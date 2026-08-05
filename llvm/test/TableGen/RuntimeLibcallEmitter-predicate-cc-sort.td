// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Two groups share the same availability predicate but differ in calling
// convention. The emitter must order them deterministically (by calling
// convention name) so the generated output is stable.

include "llvm/IR/RuntimeLibcallsImpl.td"

def FUNC_AAPCS : RuntimeLibcall;
def FUNC_DEFAULT : RuntimeLibcall;

def impl_aapcs : RuntimeLibcallImpl<FUNC_AAPCS>;
def impl_default : RuntimeLibcallImpl<FUNC_DEFAULT>;

def isWin : RuntimeLibcallPredicate<[{TT.isOSWindows()}]>;

def WinLibrary : SystemRuntimeLibrary<isWin,
  (add LibcallsWithCC<(add impl_aapcs), ARM_AAPCS, isWin>,
       LibcallImpls<(add impl_default), isWin>)
>;

// The ARM_AAPCS group sorts after the default-CC group (empty calling
// convention name orders before "ARM_AAPCS").
// CHECK: if (TT.isOSWindows()) {
// CHECK:   static const RTLIB::LibcallImpl LibraryCalls_isWin[] = {
// CHECK-NEXT:       RTLIB::impl_impl_default, // impl_default
// CHECK-NEXT:   };
// CHECK: for (const RTLIB::LibcallImpl Impl : LibraryCalls_isWin) {
// CHECK-NEXT:   setAvailable(Impl);
// CHECK-NEXT: }
// CHECK: if (TT.isOSWindows()) {
// CHECK:   static const RTLIB::LibcallImpl LibraryCalls_isWin_ARM_AAPCS[] = {
// CHECK-NEXT:       RTLIB::impl_impl_aapcs, // impl_aapcs
// CHECK-NEXT:   };
// CHECK: for (const RTLIB::LibcallImpl Impl : LibraryCalls_isWin_ARM_AAPCS) {
// CHECK-NEXT:   setAvailable(Impl);
// CHECK-NEXT:   setLibcallImplCallingConv(Impl, CallingConv::ARM_AAPCS);
// CHECK-NEXT: }
