; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' \
; RUN:   -polly-print-ast -polly-process-unprofitable -disable-output < %s \
; RUN:   | FileCheck %s
;
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl>' \
; RUN:   -polly-process-unprofitable -debug-only=polly-opt-isl -disable-output < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEBUG
; REQUIRES: asserts
;
; The AST should show a simple two-level loop without tiling:
; CHECK:      for (int c0 = 0; c0 <= 1023; c0 += 1)
; CHECK-NEXT:   for (int c1 = 0; c1 <= 511; c1 += 1)
; CHECK-NEXT:     Stmt_body(c0, c1);
;
; Verify no tiling artifacts appear:
; CHECK-NOT: floor
; CHECK-NOT: 1st level tiling
;
; DEBUG: Tiling skipped: no read access depends on multiple band dimensions

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

define void @streaming_no_reuse(ptr noalias %A, ptr noalias %B) {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.i.inc ]
  br label %body

body:
  %j = phi i64 [ 0, %for.i ], [ %j.next, %body ]
  %B.addr = getelementptr inbounds i32, ptr %B, i64 %j
  %val = load i32, ptr %B.addr, align 4
  %A.addr = getelementptr inbounds [512 x i32], ptr %A, i64 %i, i64 %j
  store i32 %val, ptr %A.addr, align 4
  %j.next = add nuw nsw i64 %j, 1
  %j.cmp = icmp slt i64 %j.next, 512
  br i1 %j.cmp, label %body, label %for.i.inc

for.i.inc:
  %i.next = add nuw nsw i64 %i, 1
  %i.cmp = icmp slt i64 %i.next, 1024
  br i1 %i.cmp, label %for.i, label %exit

exit:
  ret void
}
