; RUN: opt -S -gvn -enable-load-pre < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; These tests exercise situations when instructions that were first instructions
; with implicit control flow get removed. We make sure that after that we don't
; face crashes and are still able to perform PRE correctly.

declare i32 @foo(i32 %arg) #0

define hidden void @test_01(i32 %x, i32 %y) {

; c2 only throws if c1 throws, so it can be safely removed and then PRE can
; hoist the load out of loop.

; CHECK-LABEL: @test_01
; CHECK:       entry:
; CHECK-NEXT:    %c1 = call i32 @foo(i32 %x)
; CHECK-NEXT:    %val.pre = load i32, i32* null, align 8
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %c3 = call i32 @foo(i32 %val.pre)
; CHECK-NEXT:    br label %loop

entry:
  %c1 = call i32 @foo(i32 %x)
  br label %loop

loop:
  %c2 = call i32 @foo(i32 %x)
  %val = load i32, i32* null, align 8
  %c3 = call i32 @foo(i32 %val)
  br label %loop
}

define hidden void @test_02(i32 %x, i32 %y) {

; PRE is not allowed because c2 may throw.

; CHECK-LABEL: @test_02
; CHECK:       entry:
; CHECK-NEXT:    %c1 = call i32 @foo(i32 %x)
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %c2 = call i32 @foo(i32 %y)
; CHECK-NEXT:    %val = load i32, i32* null, align 8
; CHECK-NEXT:    %c3 = call i32 @foo(i32 %val)
; CHECK-NEXT:    br label %loop

entry:
  %c1 = call i32 @foo(i32 %x)
  br label %loop

loop:
  %c2 = call i32 @foo(i32 %y)
  %val = load i32, i32* null, align 8
  %c3 = call i32 @foo(i32 %val)
  br label %loop
}

define hidden void @test_03(i32 %x, i32 %y) {

; PRE of load is allowed because c2 only throws if c1 throws. c3 should
; not be eliminated. c4 is eliminated because it only throws if c3 throws.

; CHECK-LABEL: @test_03
; CHECK:       entry:
; CHECK-NEXT:    %c1 = call i32 @foo(i32 %x)
; CHECK-NEXT:    %val.pre = load i32, i32* null, align 8
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %c3 = call i32 @foo(i32 %y)
; CHECK-NEXT:    %c5 = call i32 @foo(i32 %val.pre)
; CHECK-NEXT:    br label %loop

entry:
  %c1 = call i32 @foo(i32 %x)
  br label %loop

loop:
  %c2 = call i32 @foo(i32 %x)
  %val = load i32, i32* null, align 8
  %c3 = call i32 @foo(i32 %y)
  %val2 = load i32, i32* null, align 8
  %c4 = call i32 @foo(i32 %y)
  %c5 = call i32 @foo(i32 %val)
  br label %loop
}

define hidden void @test_04(i32 %x, i32 %y) {

; PRE is not allowed even after we remove c2 because now c3 prevents us from it.

; CHECK-LABEL: @test_04
; CHECK:       entry:
; CHECK-NEXT:    %c1 = call i32 @foo(i32 %x)
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %c3 = call i32 @foo(i32 %y)
; CHECK-NEXT:    %val = load i32, i32* null, align 8
; CHECK-NEXT:    %c5 = call i32 @foo(i32 %val)
; CHECK-NEXT:    br label %loop

entry:
  %c1 = call i32 @foo(i32 %x)
  br label %loop

loop:
  %c2 = call i32 @foo(i32 %x)
  %c3 = call i32 @foo(i32 %y)
  %val = load i32, i32* null, align 8
  %c4 = call i32 @foo(i32 %y)
  %c5 = call i32 @foo(i32 %val)
  br label %loop
}

attributes #0 = { readnone }
