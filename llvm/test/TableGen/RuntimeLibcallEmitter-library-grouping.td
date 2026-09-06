// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check that each LibcallLibrary emits a standalone implementation function
// (suffixed by the library name) that makes the library's members available,
// with per-member triple predicates nested inside.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def MEMSET : RuntimeLibcall;
def SQRT_F64 : RuntimeLibcall;
def SINCOS_F64 : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def memset_pattern : RuntimeLibcallImpl<MEMSET, "memset_pattern">;
def sqrt : RuntimeLibcallImpl<SQRT_F64, "sqrt">;
def sincos : RuntimeLibcallImpl<SINCOS_F64, "sincos">;

def IsDarwin : LibcallPredicate<[{TT.isOSDarwin()}]>;
def IsX86 : LibcallPredicate<[{TT.isX86()}]>;
def isDarwin : RuntimeLibcallAvailability<(all_of IsDarwin)>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;

// libc: memcpy is universal; memset_pattern only on Darwin.
def Libc : LibcallLibrary<"libc", (add memcpy,
                                       LibcallImpls<(add memset_pattern), isDarwin>)>;

// libm: sqrt is universal; sincos only on x86.
def Libm : LibcallLibrary<"libm", (add sqrt,
                                       LibcallImpls<(add sincos), isX86>)>;

// One standalone implementation function is emitted per LibcallLibrary.

// CHECK: static void setAvailableLibFuncs_libc(llvm::RTLIB::RuntimeLibcallsInfo &Info, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_memcpy, // memcpy
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       Info.setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:     if (TT.isOSDarwin()) {
// CHECK-NEXT:       static const RTLIB::LibcallImpl LibraryCalls_isDarwin[] = {
// CHECK-NEXT:           RTLIB::impl_memset_pattern, // memset_pattern
// CHECK-NEXT:       };
// CHECK-EMPTY:
// CHECK-NEXT:       for (const RTLIB::LibcallImpl Impl : LibraryCalls_isDarwin) {
// CHECK-NEXT:         Info.setAvailable(Impl);
// CHECK-NEXT:       }
// CHECK-EMPTY:
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: static void setAvailableLibFuncs_libm(llvm::RTLIB::RuntimeLibcallsInfo &Info, const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-NEXT:   {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_sqrt, // sqrt
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       Info.setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:     if (TT.isX86()) {
// CHECK-NEXT:       static const RTLIB::LibcallImpl LibraryCalls_isX86[] = {
// CHECK-NEXT:           RTLIB::impl_sincos, // sincos
// CHECK-NEXT:       };
// CHECK-EMPTY:
// CHECK-NEXT:       for (const RTLIB::LibcallImpl Impl : LibraryCalls_isX86) {
// CHECK-NEXT:         Info.setAvailable(Impl);
// CHECK-NEXT:       }
// CHECK-EMPTY:
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }
