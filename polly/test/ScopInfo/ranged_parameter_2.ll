; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output -polly-allow-nonaffine -polly-invariant-load-hoisting=true < %s \
; RUN:  -debug 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK: Region: %bb1---%bb16
; CHECK:   [n] -> {  : false }

; This test case at some point caused an assertion when modeling a scop, due
; to use constructing an invalid lower and upper bound for the range of
; non-affine accesses.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @zot(ptr noalias %arg, ptr %D, i32 %n) {
bb:
  br label %bb1

bb1:
  %tmp4 = load ptr, ptr %D
  %tmp5 = add i64 undef, 3
  %tmp6 = add i64 %tmp5, undef
  %tmp7 = add i64 %tmp6, undef
  %tmp8 = getelementptr double, ptr %tmp4, i64 %tmp7
  store i64 42, ptr %tmp8
  br label %bb11

bb11:
  store i64 43, ptr %arg
  br label %bb14

bb14:
  %tmp15 = icmp eq i32 0, %n
  br i1 %tmp15, label %bb16, label %bb1

bb16:
  ret void
}
