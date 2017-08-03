; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i8:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%union.anon = type { <2 x i8> }

@i = global <2 x i8> <i8 150, i8 100>, align 8
@j = global <2 x i8> <i8 10, i8 13>, align 8
@res = common global %union.anon zeroinitializer, align 8

; Make sure we load the constants i and j starting offset zero.
; Also make sure that we sign-extend it.
; Based on /gcc-4_2-testsuite/src/gcc.c-torture/execute/pr23135.c

; CHECK: main
define i32 @main() nounwind uwtable {
entry:
; CHECK: pmovsxbq  i(%rip), %
; CHECK: pmovsxbq  j(%rip), %
  %0 = load <2 x i8>, <2 x i8>* @i, align 8
  %1 = load <2 x i8>, <2 x i8>* @j, align 8
  %div = sdiv <2 x i8> %1, %0
  store <2 x i8> %div, <2 x i8>* getelementptr inbounds (%union.anon, %union.anon* @res, i32 0, i32 0), align 8
  ret i32 0
; CHECK: ret
}
