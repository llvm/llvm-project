; RUN: opt < %s -indvars -replexitval=always -S | FileCheck %s
; Make sure IndVars preserves LCSSA form, especially across loop nests. 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @PR18642(i32 %x) {
; CHECK-LABEL: @PR18642(
entry:
  br label %outer.header
; CHECK:   br label %outer.header

outer.header:
; CHECK: outer.header:
  %outer.iv = phi i32 [ 0, %entry ], [ %x, %outer.latch ]
  br label %inner.header
; CHECK:   %[[SCEV_EXPANDED:.*]] = add i32
; CHECK:   br label %inner.header

inner.header:
; CHECK: inner.header:
  %inner.iv = phi i32 [ undef, %outer.header ], [ %inc, %inner.latch ]
  %cmp1 = icmp slt i32 %inner.iv, %outer.iv
  br i1 %cmp1, label %inner.latch, label %outer.latch
; CHECK:   br i1 {{.*}}, label %inner.latch, label %outer.latch

inner.latch:
; CHECK: inner.latch:
  %inc = add nsw i32 %inner.iv, 1
  %cmp2 = icmp slt i32 %inner.iv, %outer.iv
  br i1 %cmp2, label %inner.header, label %exit
; CHECK:   br i1 {{.*}}, label %inner.header, label %[[EXIT_FROM_INNER:.*]]

outer.latch:
; CHECK: outer.latch:
  br i1 undef, label %outer.header, label %exit
; CHECK:   br i1 {{.*}}, label %outer.header, label %[[EXIT_FROM_OUTER:.*]]

; CHECK: [[EXIT_FROM_INNER]]:
; CHECK-NEXT: %[[LCSSA:.*]] = phi i32 [ %[[SCEV_EXPANDED]], %inner.latch ]
; CHECK-NEXT: br label %exit

; CHECK: [[EXIT_FROM_OUTER]]:
; CHECK-NEXT: br label %exit

exit:
; CHECK: exit:
  %exit.phi = phi i32 [ %inc, %inner.latch ], [ undef, %outer.latch ]
; CHECK-NEXT: phi i32 [ %[[LCSSA]], %[[EXIT_FROM_INNER]] ], [ undef, %[[EXIT_FROM_OUTER]] ]
  ret void
}
