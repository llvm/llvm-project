; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"

; GVN shouldn't PRE the load around the loop backedge because it's
; not actually redundant around the loop backedge, despite appearances
; if phi-translation is ignored.

; CHECK: define void @test0(i8* %begin)
; CHECK: loop:
; CHECK:   %l0 = load i8* %phi
; CHECK:   call void @bar(i8 %l0)
; CHECK:   %l1 = load i8* %phi
define void @test0(i8* %begin) {
entry:
  br label %loop

loop:
  %phi = phi i8* [ %begin, %entry ], [ %next, %loop ]
  %l0 = load i8* %phi
  call void @bar(i8 %l0)
  %l1 = load i8* %phi
  %next = getelementptr inbounds i8* %phi, i8 %l1
  br label %loop
}

declare void @bar(i8)
