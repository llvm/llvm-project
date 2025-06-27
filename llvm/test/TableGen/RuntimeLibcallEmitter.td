// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"


def SHL_I32 : RuntimeLibcall;
def SRL_I64 : RuntimeLibcall;
def BZERO : RuntimeLibcall;

// Test default names.
let IsDefault = true in {
  def __ashlsi3 : RuntimeLibcallImpl<SHL_I32>;
  def __lshrdi3 : RuntimeLibcallImpl<SRL_I64>;
}

// Ignore non-default in initDefaultLibCallNames.
def bzero : RuntimeLibcallImpl<BZERO>;

// All entries should be emitted in Libcall enum.
// CHECK: #ifdef GET_RUNTIME_LIBCALL_ENUM
// CHECK-NEXT: namespace llvm {
// CHECK-NEXT: namespace RTLIB {
// CHECK-NEXT: enum Libcall {
// CHECK-NEXT: BZERO = 0,
// CHECK-NEXT: SHL_I32 = 1,
// CHECK-NEXT: SRL_I64 = 2,
// CHECK-NEXT: UNKNOWN_LIBCALL = 3
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: } // End namespace RTLIB
// CHECK-NEXT: } // End namespace llvm
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_INIT_RUNTIME_LIBCALL_NAMES
// CHECK-NEXT: const char *const llvm::RTLIB::RuntimeLibcallsInfo::DefaultLibcallRoutineNames[RTLIB::UNKNOWN_LIBCALL + 1] = {
// CHECK-NEXT: nullptr, // RTLIB::BZERO
// CHECK-NEXT: "__ashlsi3", // RTLIB::SHL_I32
// CHECK-NEXT: "__lshrdi3", // RTLIB::SRL_I64
// CHECK-NEXT: nullptr // RTLIB::UNKNOWN_LIBCALL
// CHECK-NEXT: };
