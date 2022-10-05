; RUN: opt -S -basic-aa -licm < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' -S %s| FileCheck %s
;
; Manually validate LCSSA form is preserved even after SSAUpdater is used to
; promote things in the loop bodies.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i32 0, align 4
@y = common global i32 0, align 4

define void @PR18688() {
; CHECK-LABEL: @PR18688(

entry:
  br i1 undef, label %return, label %outer.preheader

outer.preheader:
  br label %outer.header
; CHECK: outer.preheader:
; CHECK: br label %outer.header

outer.header:
  store i32 0, ptr @x, align 4
  br i1 undef, label %outer.latch, label %inner.preheader
; CHECK: outer.header:
; CHECK-NEXT: br i1 undef, label %outer.latch, label %inner.preheader

inner.preheader:
  br label %inner.header
; CHECK: inner.preheader:
; CHECK-NEXT: br label %inner.header

inner.header:
  br i1 undef, label %inner.body.rhs, label %inner.latch
; CHECK: inner.header:
; CHECK-NEXT: %[[PHI0:[^,]+]] = phi i32 [ %{{[^,]+}}, %inner.latch ], [ 0, %inner.preheader ]
; CHECK-NEXT: br i1 undef, label %inner.body.rhs, label %inner.latch

inner.body.rhs:
  store i32 0, ptr @x, align 4
  br label %inner.latch
; CHECK: inner.body.rhs:
; CHECK-NEXT: br label %inner.latch

inner.latch:
  %y_val = load i32, ptr @y, align 4
  %icmp = icmp eq i32 %y_val, 0
  br i1 %icmp, label %inner.exit, label %inner.header
; CHECK: inner.latch:
; CHECK-NEXT: %[[PHI1:[^,]+]] = phi i32 [ 0, %inner.body.rhs ], [ %[[PHI0]], %inner.header ]
; CHECK-NEXT: br i1 %{{[^,]+}}, label %inner.exit, label %inner.header

inner.exit:
  br label %outer.latch
; CHECK: inner.exit:
; CHECK-NEXT: %[[INNER_LCSSA:[^,]+]] = phi i32 [ %[[PHI1]], %inner.latch ]
; CHECK-NEXT: br label %outer.latch

outer.latch:
  br i1 undef, label %outer.exit, label %outer.header
; CHECK: outer.latch:
; CHECK-NEXT: %[[PHI2:[^,]+]] = phi i32 [ %[[INNER_LCSSA]], %inner.exit ], [ 0, %outer.header ]
; CHECK-NEXT: br i1 {{.*}}, label %outer.exit, label %outer.header

outer.exit:
  br label %return
; CHECK: outer.exit:
; CHECK-NEXT: %[[OUTER_LCSSA:[^,]+]] = phi i32 [ %[[PHI2]], %outer.latch ]
; CHECK-NEXT: store i32 %[[OUTER_LCSSA]]
; CHECK-NEXT: br label %return

return:
  ret void
}

