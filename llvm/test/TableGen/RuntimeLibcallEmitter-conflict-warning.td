// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s 2> %t.err | FileCheck %s
// RUN: FileCheck -check-prefix=ERR %s < %t.err

// Check behavior of libcall emission when multiple RuntimeLibcallImpl
// implementations provide the same RuntimeLibcall

include "llvm/IR/RuntimeLibcallsImpl.td"

def SOME_FUNC : RuntimeLibcall;
def OTHER_FUNC : RuntimeLibcall;
def ANOTHER_DUP : RuntimeLibcall;

def isTargetArchA : RuntimeLibcallPredicate<[{isTargetArchA()}]>;
def isTargetArchB : RuntimeLibcallPredicate<[{isTargetArchB()}]>;
def isTargetArchC : RuntimeLibcallPredicate<[{isTargetArchC()}]>;

def func_a : RuntimeLibcallImpl<SOME_FUNC>;
def func_b : RuntimeLibcallImpl<SOME_FUNC>;
def func_c : RuntimeLibcallImpl<SOME_FUNC>;
def other_func : RuntimeLibcallImpl<OTHER_FUNC>;

def dup0 : RuntimeLibcallImpl<ANOTHER_DUP>;
def dup1 : RuntimeLibcallImpl<ANOTHER_DUP>;

// func_a and func_b both provide SOME_FUNC.

// CHECK: if (isTargetArchA()) {
// CHECK-NEXT: setLibcallImpl(RTLIB::SOME_FUNC, RTLIB::impl_func_b); // func_b

// ERR: :[[@LINE+1]]:5: warning: conflicting implementations for libcall SOME_FUNC: func_b, func_a
def TheSystemLibraryA : SystemRuntimeLibrary<isTargetArchA,
  (add func_b, func_a)
>;

// CHECK: if (isTargetArchB()) {
// CHECK-NEXT: setLibcallImpl(RTLIB::OTHER_FUNC, RTLIB::impl_other_func); // other_func
// CHECK-NEXT: setLibcallImpl(RTLIB::SOME_FUNC, RTLIB::impl_func_a); // func_a

// ERR: :[[@LINE+1]]:5: warning: conflicting implementations for libcall SOME_FUNC: func_a, func_b
def TheSystemLibraryB : SystemRuntimeLibrary<isTargetArchB,
  (add func_a, other_func, func_b)
>;

// CHECK: if (isTargetArchC()) {
// CHECK-NEXT: setLibcallImpl(RTLIB::ANOTHER_DUP, RTLIB::impl_dup1); // dup1
// CHECK-NEXT: setLibcallImpl(RTLIB::OTHER_FUNC, RTLIB::impl_other_func); // other_func
// CHECK-NEXT: setLibcallImpl(RTLIB::SOME_FUNC, RTLIB::impl_func_a); // func_a

// ERR: :[[@LINE+3]]:5: warning: conflicting implementations for libcall ANOTHER_DUP: dup1, dup0
// ERR: :[[@LINE+2]]:5: warning: conflicting implementations for libcall SOME_FUNC: func_a, func_b
// ERR: :[[@LINE+1]]:5: warning: conflicting implementations for libcall SOME_FUNC: func_a, func_c
def TheSystemLibraryC : SystemRuntimeLibrary<isTargetArchC,
  (add func_a, dup1, other_func, func_b, func_c, dup0)
>;
