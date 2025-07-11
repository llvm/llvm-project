// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"


def SHL_I32 : RuntimeLibcall;
def SRL_I64 : RuntimeLibcall;

def SQRT_F128 : RuntimeLibcall;
def SQRT_F80 : RuntimeLibcall;
def BZERO : RuntimeLibcall;

// Test default names.
let IsDefault = true in {
  def __ashlsi3 : RuntimeLibcallImpl<SHL_I32>;
  def __lshrdi3 : RuntimeLibcallImpl<SRL_I64>;

  def sqrtl_f128 : RuntimeLibcallImpl<SQRT_F128, "sqrtl">;
  def sqrtl_f80 : RuntimeLibcallImpl<SQRT_F80, "sqrtl">;
}

// Ignore non-default in initDefaultLibCallNames.
def bzero : RuntimeLibcallImpl<BZERO>;

// All entries should be emitted in Libcall enum.
// CHECK: #ifdef GET_RUNTIME_LIBCALL_ENUM
// CHECK-NEXT: namespace llvm {
// CHECK-NEXT: namespace RTLIB {
// CHECK-NEXT: enum Libcall : unsigned short {
// CHECK-NEXT: BZERO = 0,
// CHECK-NEXT: SHL_I32 = 1,
// CHECK-NEXT: SQRT_F80 = 2,
// CHECK-NEXT: SQRT_F128 = 3,
// CHECK-NEXT: SRL_I64 = 4,
// CHECK-NEXT: UNKNOWN_LIBCALL = 5
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT:enum LibcallImpl : unsigned short {
// CHECK-NEXT:  Unsupported = 0,
// CHECK-NEXT:  __ashlsi3 = 1, // __ashlsi3
// CHECK-NEXT:  __lshrdi3 = 2, // __lshrdi3
// CHECK-NEXT:  bzero = 3, // bzero
// CHECK-NEXT:  sqrtl_f80 = 4, // sqrtl
// CHECK-NEXT:  sqrtl_f128 = 5, // sqrtl
// CHECK-NEXT:  NumLibcallImpls = 6
// CHECK-NEXT: };
// CHECK-NEXT: } // End namespace RTLIB
// CHECK-NEXT: } // End namespace llvm
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_INIT_RUNTIME_LIBCALL_NAMES
// CHECK-NEXT: const RTLIB::LibcallImpl llvm::RTLIB::RuntimeLibcallsInfo::DefaultLibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {
// CHECK-NEXT:   RTLIB::Unsupported, // RTLIB::BZERO
// CHECK-NEXT:   RTLIB::__ashlsi3, // RTLIB::SHL_I32
// CHECK-NEXT:   RTLIB::sqrtl_f80, // RTLIB::SQRT_F80
// CHECK-NEXT:   RTLIB::sqrtl_f128, // RTLIB::SQRT_F128
// CHECK-NEXT:   RTLIB::__lshrdi3, // RTLIB::SRL_I64
// CHECK-NEXT:   RTLIB::Unsupported
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: const char *const llvm::RTLIB::RuntimeLibcallsInfo::LibCallImplNames[RTLIB::NumLibcallImpls] = {
// CHECK-NEXT: nullptr, // RTLIB::Unsupported
// CHECK-NEXT: "__ashlsi3", // RTLIB::__ashlsi3
// CHECK-NEXT: "__lshrdi3", // RTLIB::__lshrdi3
// CHECK-NEXT: "bzero", // RTLIB::bzero
// CHECK-NEXT: "sqrtl", // RTLIB::sqrtl_f80
// CHECK-NEXT: "sqrtl", // RTLIB::sqrtl_f128
// CHECK-NEXT: };

// CHECK: const RTLIB::Libcall llvm::RTLIB::RuntimeLibcallsInfo::ImplToLibcall[RTLIB::NumLibcallImpls] = {
// CHECK-NEXT: RTLIB::UNKNOWN_LIBCALL, // RTLIB::Unsupported
// CHECK-NEXT: RTLIB::SHL_I32, // RTLIB::__ashlsi3
// CHECK-NEXT: RTLIB::SRL_I64, // RTLIB::__lshrdi3
// CHECK-NEXT: RTLIB::BZERO, // RTLIB::bzero
// CHECK-NEXT: RTLIB::SQRT_F80, // RTLIB::sqrtl_f80
// CHECK-NEXT: RTLIB::SQRT_F128, // RTLIB::sqrtl_f128
// CHECK-NEXT: };
