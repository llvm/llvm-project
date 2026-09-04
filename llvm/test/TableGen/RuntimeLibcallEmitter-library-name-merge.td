// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Several libraries sharing one linker name (a GNU and a Darwin "libc") merge
// into a single setAvailableLibFuncs_libc. The shared core is emitted once,
// unguarded; each variant's guarded block carries only its own extras.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def MEMSET : RuntimeLibcall;
def BZERO : RuntimeLibcall;
def STPCPY : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def memset : RuntimeLibcallImpl<MEMSET>;
def bzero : RuntimeLibcallImpl<BZERO>;
def stpcpy : RuntimeLibcallImpl<STPCPY>;

def IsDarwin : LibcallPredicate<[{TT.isOSDarwin()}]>;
def IsLinux : LibcallPredicate<[{TT.isOSLinux()}]>;
def isDarwin : RuntimeLibcallAvailability<(all_of IsDarwin)>;
def isLinux : RuntimeLibcallAvailability<(all_of IsLinux)>;

def SQRT_F64 : RuntimeLibcall;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;

// The core libc functions shared by every libc variant.
defvar CoreLibc = (add memcpy, memset);

// Each system's libc = the common core + its own extras, selected by Pred.
def GnuLibc : LibcallLibrary<"libc", (add CoreLibc, stpcpy)> { let Pred = isLinux; }
def DarwinLibc : LibcallLibrary<"libc", (add CoreLibc, bzero)> { let Pred = isDarwin; }

// sqrt appears in both variants but with a different calling convention
// (ARM_AAPCS on GNU, default on Darwin), so it must not be hoisted into the
// shared core (that would drop the per-target CC).
def GnuLibm : LibcallLibrary<"libm", (add LibcallsWithCC<(add sqrt), ARM_AAPCS>)> {
  let Pred = isLinux;
}
def DarwinLibm : LibcallLibrary<"libm", (add sqrt)> { let Pred = isDarwin; }

// The shared core (memcpy, memset) is emitted once and unguarded; each variant's
// guarded block carries only its own extras (bzero for Darwin, stpcpy for GNU).

// CHECK: static void setAvailableLibFuncs_libc(llvm::RTLIB::RuntimeLibcallsInfo &Info, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:       RTLIB::impl_memcpy, // memcpy
// CHECK-NEXT:       RTLIB::impl_memset, // memset
// CHECK-NEXT:   };
// CHECK-EMPTY:
// CHECK-NEXT:   for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:     Info.setAvailable(Impl);
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   if (TT.isOSDarwin()) {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_bzero, // bzero
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       Info.setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT:   if (TT.isOSLinux()) {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_stpcpy, // stpcpy
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       Info.setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// sqrt differs in calling convention between the two libm variants, so it stays
// per-variant: no shared-core hoist, and the GNU variant keeps its ARM_AAPCS.

// CHECK: static void setAvailableLibFuncs_libm(llvm::RTLIB::RuntimeLibcallsInfo &Info, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   if (TT.isOSDarwin()) {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_sqrt, // sqrt
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       Info.setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT:   if (TT.isOSLinux()) {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls_AlwaysAvailable_ARM_AAPCS[] = {
// CHECK-NEXT:         RTLIB::impl_sqrt, // sqrt
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls_AlwaysAvailable_ARM_AAPCS) {
// CHECK-NEXT:       Info.setAvailable(Impl);
// CHECK-NEXT:       Info.setLibcallImplCallingConv(Impl, CallingConv::ARM_AAPCS);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }
