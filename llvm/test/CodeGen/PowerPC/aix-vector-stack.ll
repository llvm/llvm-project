; RUN: not --crash llc < %s -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:   -vec-extabi -mtriple powerpc-ibm-aix-xcoff 2>&1 | \
; RUN: FileCheck %s --check-prefix=AIX-ERROR

; RUN: not --crash llc < %s -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:   -vec-extabi -mtriple powerpc64-ibm-aix-xcoff 2>&1 | \
; RUN: FileCheck %s --check-prefix=AIX-ERROR

define dso_local <4 x i32> @vec_callee_stack(<4 x i32> %vec1, <4 x i32> %vec2, <4 x i32> %vec3, <4 x i32> %vec4, <4 x i32> %vec5, <4 x i32> %vec6, <4 x i32> %vec7, <4 x i32> %vec8, <4 x i32> %vec9, <4 x i32> %vec10, <4 x i32> %vec11, <4 x i32> %vec12, <4 x i32> %vec13, <4 x i32> %vec14) {
entry:
  %add = add <4 x i32> %vec1, %vec2
  %add1 = add <4 x i32> %add, %vec3
  %add2 = add <4 x i32> %add1, %vec4
  %add3 = add <4 x i32> %add2, %vec5
  %add4 = add <4 x i32> %add3, %vec6
  %add5 = add <4 x i32> %add4, %vec7
  %add6 = add <4 x i32> %add5, %vec8
  %add7 = add <4 x i32> %add6, %vec9
  %add8 = add <4 x i32> %add7, %vec10
  %add9 = add <4 x i32> %add8, %vec11
  %add10 = add <4 x i32> %add9, %vec12
  %add11 = add <4 x i32> %add10, %vec13
  %add12 = add <4 x i32> %add11, %vec14
  ret <4 x i32> %add12
}

; AIX-ERROR:  LLVM ERROR: passing vector parameters to the stack is unimplemented for AIX
