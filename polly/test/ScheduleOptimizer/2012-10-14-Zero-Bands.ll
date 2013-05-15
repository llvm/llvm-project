; RUN: opt %loadPolly -polly-opt-isl -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1536 x float] zeroinitializer

define void @read_nres() {
entry:
  br label %if.cond

if.cond:
  br i1 false, label %if.then, label %if.end

if.then:
  %ptr = getelementptr [1536 x float]* @A, i64 0, i32 23
  store float undef, float* %ptr
  br label %if.end

if.end:
  br label %return

return:
  ret void
}

; CHECK: Calculated schedule:
; CHECK: { Stmt_if_then[] -> [] }
