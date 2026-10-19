// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s 2> %t.err | FileCheck %s
// RUN: count 0 < %t.err

// Check behavior of libcall emission when multiple RuntimeLibcallImpl
// implementations provide the same RuntimeLibcall. This table records
// all available implementations, not which one is selected.

include "llvm/IR/RuntimeLibcallsImpl.td"

def SOME_FUNC : RuntimeLibcall;
def OTHER_FUNC : RuntimeLibcall;
def ANOTHER_DUP : RuntimeLibcall;

def IsTargetArchA : LibcallPredicate<[{isTargetArchA()}]>;
def IsTargetArchB : LibcallPredicate<[{isTargetArchB()}]>;
def IsTargetArchC : LibcallPredicate<[{isTargetArchC()}]>;

def isTargetArchA : RuntimeLibcallAvailability<(all_of IsTargetArchA)>;
def isTargetArchB : RuntimeLibcallAvailability<(all_of IsTargetArchB)>;
def isTargetArchC : RuntimeLibcallAvailability<(all_of IsTargetArchC)>;

def func_a : RuntimeLibcallImpl<SOME_FUNC>;
def func_b : RuntimeLibcallImpl<SOME_FUNC>;
def func_c : RuntimeLibcallImpl<SOME_FUNC>;
def other_func : RuntimeLibcallImpl<OTHER_FUNC>;

def dup0 : RuntimeLibcallImpl<ANOTHER_DUP>;
def dup1 : RuntimeLibcallImpl<ANOTHER_DUP>;

// func_a and func_b both provide SOME_FUNC.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_liba(
// CHECK:    static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:        RTLIB::impl_func_a, // func_a
// CHECK-NEXT:        RTLIB::impl_func_b, // func_b
// CHECK-NEXT:    };

def LibA : LibcallLibrary<"liba", (add func_b, func_a)>;
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA, (add LibA)>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libb(
// CHECK:    static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:        RTLIB::impl_other_func, // other_func
// CHECK-NEXT:        RTLIB::impl_func_a, // func_a
// CHECK-NEXT:        RTLIB::impl_func_b, // func_b
// CHECK-NEXT:    };

def LibB : LibcallLibrary<"libb", (add func_a, other_func, func_b)>;
def TheSystemLibraryB : SystemRuntimeLibrary<isTargetArchB, (add LibB)>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(
// CHECK:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_dup0, // dup0
// CHECK-NEXT:         RTLIB::impl_dup1, // dup1
// CHECK-NEXT:         RTLIB::impl_other_func, // other_func
// CHECK-NEXT:         RTLIB::impl_func_a, // func_a
// CHECK-NEXT:         RTLIB::impl_func_b, // func_b
// CHECK-NEXT:         RTLIB::impl_func_c, // func_c
// CHECK-NEXT:     };

def LibC : LibcallLibrary<"libc",
  (add func_a, dup1, other_func, func_b, func_c, dup0)
>;
def TheSystemLibraryC : SystemRuntimeLibrary<isTargetArchC, (add LibC)>;
