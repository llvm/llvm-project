; This testcase consists of alias relations on 128-bit pointers that
; should be completely resolvable by basicaa.

; RUN: opt < %s -basicaa -aa-eval -print-no-aliases -print-may-aliases -print-must-aliases -disable-output 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128-p100:128:64:64-p101:128:64:64"


; test0 is similar to SimpleCases.ll

%T = type { i32, [10 x i8] }

; CHECK:     Function: test0
; CHECK-NOT:   MayAlias:
define void @test0(%T addrspace(100)* %P) {
  %A = getelementptr %T, %T addrspace(100)* %P, i64 0
  %B = getelementptr %T, %T addrspace(100)* %P, i64 0, i32 0
  %C = getelementptr %T, %T addrspace(100)* %P, i64 0, i32 1
  %D = getelementptr %T, %T addrspace(100)* %P, i64 0, i32 1, i64 0
  %E = getelementptr %T, %T addrspace(100)* %P, i64 0, i32 1, i64 5
  ret void
}

; test1 checks that >64 bits of index can be considered.
; If BasicAA is truncating the arithmetic, it will conclude
; that %A and %B must alias when in fact they must not.

; CHECK:     Function: test1
; CHECK-NOT:   MustAlias:
; CHECK:       NoAlias:
; CHECK-SAME:  %A
; CHECK-SAME:  %B
define void @test1(double addrspace(100)* %P, i128 %i) {
  ; 1180591620717411303424 is 2**70
  ;  590295810358705651712 is 2**69
  %i70 = add i128 %i, 1180591620717411303424 
  %i69 = add i128 %i, 590295810358705651712
  %A = getelementptr double, double addrspace(100)* %P, i128 %i70
  %B = getelementptr double, double addrspace(100)* %P, i128 %i69
  ret void
}

; test2 checks that >64 bits of index can be considered
; and computes the same address in two ways to ensure that
; they are considered equivalent.

; CHECK: Function: test2
; CHECK: MustAlias:
; CHECK-SAME: %A
; CHECK-SAME: %C
define void @test2(double addrspace(100)* %P, i128 %i) {
  ; 1180591620717411303424 is 2**70
  ;  590295810358705651712 is 2**69
  %i70 = add i128 %i, 1180591620717411303424 
  %i69 = add i128 %i, 590295810358705651712
  %j70 = add i128 %i69, 590295810358705651712 
  %A = getelementptr double, double addrspace(100)* %P, i128 %i70
  %C = getelementptr double, double addrspace(100)* %P, i128 %j70
  ret void
}
