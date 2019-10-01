; RUN: llc < %s -asm-verbose=false -O3 -mtriple=armv6-apple-darwin -relocation-model=pic  -mcpu=arm1136jf-s -arm-atomic-cfg-tidy=0 | FileCheck %s
; rdar://8959122 illegal register operands for UMULL instruction
;   in cfrac nightly test.
; Armv6 generates a umull that must write to two distinct destination regs.

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:64-n32"
target triple = "armv6-apple-darwin10"

define void @ptoa(i1 %tst, i8* %p8, i8 %val8) nounwind {
entry:
  br i1 false, label %bb3, label %bb

bb:                                               ; preds = %entry
  br label %bb3

bb3:                                              ; preds = %bb, %entry
  %0 = call noalias i8* @malloc() nounwind
  br i1 %tst, label %bb46, label %bb8

bb8:                                              ; preds = %bb3
  %1 = getelementptr inbounds i8, i8* %0, i32 0
  store volatile i8 0, i8* %1, align 1
  %2 = call i32 @ptou() nounwind
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %3 = udiv i32 %2, 10
  %4 = urem i32 %3, 10
  %5 = icmp ult i32 %4, 10
  %6 = trunc i32 %4 to i8
  %7 = or i8 %6, 48
  %8 = add i8 %6, 87
  %iftmp.5.0.1 = select i1 %5, i8 %7, i8 %8
  store volatile i8 %iftmp.5.0.1, i8* %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %9 = udiv i32 %2, 100
  %10 = urem i32 %9, 10
  %11 = icmp ult i32 %10, 10
  %12 = trunc i32 %10 to i8
  %13 = or i8 %12, 48
  %14 = add i8 %12, 87
  %iftmp.5.0.2 = select i1 %11, i8 %13, i8 %14
  store volatile i8 %iftmp.5.0.2, i8* %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %15 = udiv i32 %2, 10000
  %16 = urem i32 %15, 10
  %17 = icmp ult i32 %16, 10
  %18 = trunc i32 %16 to i8
  %19 = or i8 %18, 48
  %20 = add i8 %18, 87
  %iftmp.5.0.4 = select i1 %17, i8 %19, i8 %20
  store volatile i8 %iftmp.5.0.4, i8* null, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %21 = udiv i32 %2, 100000
  %22 = urem i32 %21, 10
  %23 = icmp ult i32 %22, 10
  %iftmp.5.0.5 = select i1 %23, i8 0, i8 %val8
  store volatile i8 %iftmp.5.0.5, i8* %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %24 = udiv i32 %2, 1000000
  %25 = urem i32 %24, 10
  %26 = icmp ult i32 %25, 10
  %27 = trunc i32 %25 to i8
  %28 = or i8 %27, 48
  %29 = add i8 %27, 87
  %iftmp.5.0.6 = select i1 %26, i8 %28, i8 %29
  store volatile i8 %iftmp.5.0.6, i8* %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %30 = udiv i32 %2, 10000000
  %31 = urem i32 %30, 10
  %32 = icmp ult i32 %31, 10
  %33 = trunc i32 %31 to i8
  %34 = or i8 %33, 48
  %35 = add i8 %33, 87
  %iftmp.5.0.7 = select i1 %32, i8 %34, i8 %35
  store volatile i8 %iftmp.5.0.7, i8* %p8, align 1
  %36 = udiv i32 %2, 100000000
  %37 = urem i32 %36, 10
  %38 = icmp ult i32 %37, 10
  %39 = trunc i32 %37 to i8
  %40 = or i8 %39, 48
  %41 = add i8 %39, 87
  %iftmp.5.0.8 = select i1 %38, i8 %40, i8 %41
  store volatile i8 %iftmp.5.0.8, i8* null, align 1
  br label %bb46

bb46:                                             ; preds = %bb3
  ret void
}

declare noalias i8* @malloc() nounwind

declare i32 @ptou()
