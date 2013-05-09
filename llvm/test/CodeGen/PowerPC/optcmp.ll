; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -disable-ppc-cmp-opt=0 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @foo(i32 signext %a, i32 signext %b, i32* nocapture %c) #0 {
entry:
  %sub = sub nsw i32 %a, %b
  store i32 %sub, i32* %c, align 4, !tbaa !0
  %cmp = icmp sgt i32 %a, %b
  %cond = select i1 %cmp, i32 %a, i32 %b
  ret i32 %cond

; CHECK: @foo
; CHECK-NOT: subf.
}

define signext i32 @foo2(i32 signext %a, i32 signext %b, i32* nocapture %c) #0 {
entry:
  %shl = shl i32 %a, %b
  store i32 %shl, i32* %c, align 4, !tbaa !0
  %cmp = icmp sgt i32 %shl, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv

; CHECK: @foo2
; CHECK-NOT: slw.
}

define i64 @fool(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  store i64 %sub, i64* %c, align 8, !tbaa !3
  %cmp = icmp sgt i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK: @fool
; CHECK: subf. [[REG:[0-9]+]], 4, 3
; CHECK: isel 3, 3, 4, 1
; CHECK: std [[REG]], 0(5)
}

define i64 @foolb(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  store i64 %sub, i64* %c, align 8, !tbaa !3
  %cmp = icmp sle i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK: @foolb
; CHECK: subf. [[REG:[0-9]+]], 4, 3
; CHECK: isel 3, 4, 3, 1
; CHECK: std [[REG]], 0(5)
}

define i64 @foolc(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %b, %a
  store i64 %sub, i64* %c, align 8, !tbaa !3
  %cmp = icmp sgt i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK: @foolc
; CHECK: subf. [[REG:[0-9]+]], 3, 4
; CHECK: isel 3, 3, 4, 0
; CHECK: std [[REG]], 0(5)
}

define i64 @foold(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %b, %a
  store i64 %sub, i64* %c, align 8, !tbaa !3
  %cmp = icmp eq i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK: @foold
; CHECK: subf. [[REG:[0-9]+]], 3, 4
; CHECK: isel 3, 3, 4, 2
; CHECK: std [[REG]], 0(5)
}

define i64 @foold2(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  store i64 %sub, i64* %c, align 8, !tbaa !3
  %cmp = icmp eq i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK: @foold2
; CHECK: subf. [[REG:[0-9]+]], 4, 3
; CHECK: isel 3, 3, 4, 2
; CHECK: std [[REG]], 0(5)
}

define i64 @foo2l(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %shl = shl i64 %a, %b
  store i64 %shl, i64* %c, align 8, !tbaa !3
  %cmp = icmp sgt i64 %shl, 0
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1

; CHECK: @foo2l
; CHECK: sld. 4, 3, 4
; CHECK: std 4, 0(5)
}

define double @food(double %a, double %b, double* nocapture %c) #0 {
entry:
  %sub = fsub double %a, %b
  store double %sub, double* %c, align 8, !tbaa !3
  %cmp = fcmp ogt double %a, %b
  %cond = select i1 %cmp, double %a, double %b
  ret double %cond

; CHECK: @food
; CHECK-NOT: fsub. 0, 1, 2
; CHECK: stfd 0, 0(5)
}

define float @foof(float %a, float %b, float* nocapture %c) #0 {
entry:
  %sub = fsub float %a, %b
  store float %sub, float* %c, align 4, !tbaa !3
  %cmp = fcmp ogt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond

; CHECK: @foof
; CHECK-NOT: fsubs. 0, 1, 2
; CHECK: stfs 0, 0(5)
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"long", metadata !1}
!4 = metadata !{metadata !"any pointer", metadata !1}

