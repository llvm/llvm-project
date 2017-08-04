; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -mattr=-crbits -disable-ppc-cmp-opt=0 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -mattr=-crbits -disable-ppc-cmp-opt=0 -ppc-gen-isel=false | FileCheck --check-prefix=CHECK-NO-ISEL %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @foo(i32 signext %a, i32 signext %b, i32* nocapture %c) #0 {
entry:
  %sub = sub nsw i32 %a, %b
  store i32 %sub, i32* %c, align 4
  %cmp = icmp sgt i32 %a, %b
  %cond = select i1 %cmp, i32 %a, i32 %b
  ret i32 %cond

; CHECK: @foo
; CHECK-NOT: subf.
}

define signext i32 @foo2(i32 signext %a, i32 signext %b, i32* nocapture %c) #0 {
entry:
  %shl = shl i32 %a, %b
  store i32 %shl, i32* %c, align 4
  %cmp = icmp sgt i32 %shl, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv

; CHECK: @foo2
; CHECK-NOT: slw.
}

define i64 @fool(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  store i64 %sub, i64* %c, align 8
  %cmp = icmp sgt i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK-LABEL: @fool
; CHECK-NO-ISEL-LABEL: @fool
; CHECK: subf. [[REG:[0-9]+]], 4, 3
; CHECK: isel 3, 3, 4, 1
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 3, 4, 0
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]

; CHECK: std [[REG]], 0(5)
}

define i64 @foolb(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  store i64 %sub, i64* %c, align 8
  %cmp = icmp sle i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK-LABEL: @foolb
; CHECK-NO-ISEL-LABEL: @foolb
; CHECK: subf. [[REG:[0-9]+]], 4, 3
; CHECK: isel 3, 4, 3, 1
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL-NEXT: b .LBB
; CHECK-NO-ISEL addi: 3, 4, 0
; CHECK: std [[REG]], 0(5)
}

define i64 @foolc(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %b, %a
  store i64 %sub, i64* %c, align 8
  %cmp = icmp sgt i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK-LABEL: @foolc
; CHECK-NO-ISEL-LABEL: @foolc
; CHECK: subf. [[REG:[0-9]+]], 3, 4
; CHECK: isel 3, 3, 4, 0
; CHECK-NO-ISEL: bc 12, 0, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 3, 4, 0
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK: std [[REG]], 0(5)
}

define i64 @foold(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %b, %a
  store i64 %sub, i64* %c, align 8
  %cmp = icmp slt i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK-LABEL: @foold
; CHECK-NO-ISEL-LABEL: @foold
; CHECK: subf. [[REG:[0-9]+]], 3, 4
; CHECK: isel 3, 3, 4, 1
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 3, 4, 0
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK: std [[REG]], 0(5)
}

define i64 @foold2(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  store i64 %sub, i64* %c, align 8
  %cmp = icmp slt i64 %a, %b
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK-LABEL: @foold2
; CHECK-NO-ISEL-LABEL: @foold2
; CHECK: subf. [[REG:[0-9]+]], 4, 3
; CHECK: isel 3, 3, 4, 0
; CHECK-NO-ISEL: bc 12, 0, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: ori 3, 4, 0
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK: std [[REG]], 0(5)
}

define i64 @foo2l(i64 %a, i64 %b, i64* nocapture %c) #0 {
entry:
  %shl = shl i64 %a, %b
  store i64 %shl, i64* %c, align 8
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
  store double %sub, double* %c, align 8
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
  store float %sub, float* %c, align 4
  %cmp = fcmp ogt float %a, %b
  %cond = select i1 %cmp, float %a, float %b
  ret float %cond

; CHECK: @foof
; CHECK-NOT: fsubs. 0, 1, 2
; CHECK: stfs 0, 0(5)
}

declare i64 @llvm.ctpop.i64(i64);

define signext i64 @fooct(i64 signext %a, i64 signext %b, i64* nocapture %c) #0 {
entry:
  %sub = sub nsw i64 %a, %b
  %subc = call i64 @llvm.ctpop.i64(i64 %sub)
  store i64 %subc, i64* %c, align 4
  %cmp = icmp sgt i64 %subc, 0
  %cond = select i1 %cmp, i64 %a, i64 %b
  ret i64 %cond

; CHECK: @fooct
; CHECK-NOT: popcntd.
}

