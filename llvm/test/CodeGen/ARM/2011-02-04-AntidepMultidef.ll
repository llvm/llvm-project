; RUN: llc < %s -asm-verbose=false -O3 -mtriple=armv6-apple-darwin -relocation-model=pic  -mcpu=arm1136jf-s -arm-atomic-cfg-tidy=0 | FileCheck %s
; rdar://8959122 illegal register operands for UMULL instruction
;   in cfrac nightly test.
; Armv6 generates a umull that must write to two distinct destination regs.

; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:64-n32"
target triple = "armv6-apple-darwin10"

define void @ptoa(i1 %tst, ptr %p8, i8 %val8) nounwind {
entry:
  br i1 false, label %bb3, label %bb

bb:                                               ; preds = %entry
  br label %bb3

bb3:                                              ; preds = %bb, %entry
  %0 = call noalias ptr @malloc() nounwind
  br i1 %tst, label %bb46, label %bb8

bb8:                                              ; preds = %bb3
  store volatile i8 0, ptr %0, align 1
  %1 = call i32 @ptou() nounwind
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %2 = udiv i32 %1, 10
  %3 = urem i32 %2, 10
  %4 = icmp ult i32 %3, 10
  %5 = trunc i32 %3 to i8
  %6 = or i8 %5, 48
  %7 = add i8 %5, 87
  %iftmp.5.0.1 = select i1 %4, i8 %6, i8 %7
  store volatile i8 %iftmp.5.0.1, ptr %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %8 = udiv i32 %1, 100
  %9 = urem i32 %8, 10
  %10 = icmp ult i32 %9, 10
  %11 = trunc i32 %9 to i8
  %12 = or i8 %11, 48
  %13 = add i8 %11, 87
  %iftmp.5.0.2 = select i1 %10, i8 %12, i8 %13
  store volatile i8 %iftmp.5.0.2, ptr %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %14 = udiv i32 %1, 10000
  %15 = urem i32 %14, 10
  %16 = icmp ult i32 %15, 10
  %17 = trunc i32 %15 to i8
  %18 = or i8 %17, 48
  %19 = add i8 %17, 87
  %iftmp.5.0.4 = select i1 %16, i8 %18, i8 %19
  store volatile i8 %iftmp.5.0.4, ptr null, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %20 = udiv i32 %1, 100000
  %21 = urem i32 %20, 10
  %22 = icmp ult i32 %21, 10
  %iftmp.5.0.5 = select i1 %22, i8 0, i8 %val8
  store volatile i8 %iftmp.5.0.5, ptr %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %23 = udiv i32 %1, 1000000
  %24 = urem i32 %23, 10
  %25 = icmp ult i32 %24, 10
  %26 = trunc i32 %24 to i8
  %27 = or i8 %26, 48
  %28 = add i8 %26, 87
  %iftmp.5.0.6 = select i1 %25, i8 %27, i8 %28
  store volatile i8 %iftmp.5.0.6, ptr %p8, align 1
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  ; CHECK: umull [[REGISTER:lr|r[0-9]+]],
  ; CHECK-NOT: [[REGISTER]],
  ; CHECK: {{lr|r[0-9]+}}, {{lr|r[0-9]+$}}
  %29 = udiv i32 %1, 10000000
  %30 = urem i32 %29, 10
  %31 = icmp ult i32 %30, 10
  %32 = trunc i32 %30 to i8
  %33 = or i8 %32, 48
  %34 = add i8 %32, 87
  %iftmp.5.0.7 = select i1 %31, i8 %33, i8 %34
  store volatile i8 %iftmp.5.0.7, ptr %p8, align 1
  %35 = udiv i32 %1, 100000000
  %36 = urem i32 %35, 10
  %37 = icmp ult i32 %36, 10
  %38 = trunc i32 %36 to i8
  %39 = or i8 %38, 48
  %40 = add i8 %38, 87
  %iftmp.5.0.8 = select i1 %37, i8 %39, i8 %40
  store volatile i8 %iftmp.5.0.8, ptr null, align 1
  br label %bb46

bb46:                                             ; preds = %bb3
  ret void
}

declare noalias ptr @malloc() nounwind

declare i32 @ptou()
