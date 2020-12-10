; RUN: not --crash llc < %s -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:   -vec-extabi -mtriple powerpc-ibm-aix-xcoff 2>&1 | \
; RUN: FileCheck %s --check-prefix=AIX-ERROR

; RUN: not --crash llc < %s -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:   -vec-extabi -mtriple powerpc64-ibm-aix-xcoff 2>&1 | \
; RUN: FileCheck %s --check-prefix=AIX-ERROR

define dso_local i32 @vec_caller() {
entry:
  %call = call i32 bitcast (i32 (...)* @vec_callee_stack to i32 (<4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>)*)(<4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> <i32 5, i32 6, i32 7, i32 8>, <4 x i32> <i32 9, i32 10, i32 11, i32 12>, <4 x i32> <i32 13, i32 14, i32 15, i32 16>, <4 x i32> <i32 17, i32 18, i32 19, i32 20>, <4 x i32> <i32 21, i32 22, i32 23, i32 24>, <4 x i32> <i32 25, i32 26, i32 27, i32 28>, <4 x i32> <i32 29, i32 30, i32 31, i32 32>, <4 x i32> <i32 33, i32 34, i32 35, i32 36>, <4 x i32> <i32 37, i32 38, i32 39, i32 40>, <4 x i32> <i32 41, i32 42, i32 43, i32 44>, <4 x i32> <i32 45, i32 46, i32 47, i32 48>, <4 x i32> <i32 49, i32 50, i32 51, i32 52>, <4 x i32> <i32 53, i32 54, i32 55, i32 56>)
  ret i32 0
}

declare i32 @vec_callee_stack(...)

; AIX-ERROR:  LLVM ERROR: passing vector parameters to the stack is unimplemented for AIX
