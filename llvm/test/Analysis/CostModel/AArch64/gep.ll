; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

define i8 @test1(i8* %p) {
; CHECK-LABEL: test1
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 0
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test2(i16* %p) {
; CHECK-LABEL: test2
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 0
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test3(i32* %p) {
; CHECK-LABEL: test3
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 0
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test4(i64* %p) {
; CHECK-LABEL: test4
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 0
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test5(i8* %p) {
; CHECK-LABEL: test5
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 1024
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test6(i16* %p) {
; CHECK-LABEL: test6
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 1024
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test7(i32* %p) {
; CHECK-LABEL: test7
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 1024
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test8(i64* %p) {
; CHECK-LABEL: test8
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 1024
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test9(i8* %p) {
; CHECK-LABEL: test9
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 4096
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test10(i16* %p) {
; CHECK-LABEL: test10
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 4096
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test11(i32* %p) {
; CHECK-LABEL: test11
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 4096
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test12(i64* %p) {
; CHECK-LABEL: test12
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 4096
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test13(i8* %p) {
; CHECK-LABEL: test13
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 -64
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test14(i16* %p) {
; CHECK-LABEL: test14
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 -64
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test15(i32* %p) {
; CHECK-LABEL: test15
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 -64
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test16(i64* %p) {
; CHECK-LABEL: test16
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 -64
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test17(i8* %p) {
; CHECK-LABEL: test17
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 -1024
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test18(i16* %p) {
; CHECK-LABEL: test18
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 -1024
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test19(i32* %p) {
; CHECK-LABEL: test19
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 -1024
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test20(i64* %p) {
; CHECK-LABEL: test20
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 -1024
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test21(i8* %p, i32 %i) {
; CHECK-LABEL: test21
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 %i
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test22(i16* %p, i32 %i) {
; CHECK-LABEL: test22
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 %i
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test23(i32* %p, i32 %i) {
; CHECK-LABEL: test23
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 %i
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test24(i64* %p, i32 %i) {
; CHECK-LABEL: test24
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 %i
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test25(i8* %p) {
; CHECK-LABEL: test25
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 -128
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test26(i16* %p) {
; CHECK-LABEL: test26
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 -128
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test27(i32* %p) {
; CHECK-LABEL: test27
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 -128
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test28(i64* %p) {
; CHECK-LABEL: test28
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 -128
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test29(i8* %p) {
; CHECK-LABEL: test29
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 -256
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test30(i16* %p) {
; CHECK-LABEL: test30
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 -256
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test31(i32* %p) {
; CHECK-LABEL: test31
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 -256
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test32(i64* %p) {
; CHECK-LABEL: test32
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 -256
  %v = load i64, i64* %a
  ret i64 %v
}

define i8 @test33(i8* %p) {
; CHECK-LABEL: test33
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, i8*
  %a = getelementptr inbounds i8, i8* %p, i32 -512
  %v = load i8, i8* %a
  ret i8 %v
}

define i16 @test34(i16* %p) {
; CHECK-LABEL: test34
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, i16*
  %a = getelementptr inbounds i16, i16* %p, i32 -512
  %v = load i16, i16* %a
  ret i16 %v
}

define i32 @test35(i32* %p) {
; CHECK-LABEL: test35
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, i32*
  %a = getelementptr inbounds i32, i32* %p, i32 -512
  %v = load i32, i32* %a
  ret i32 %v
}

define i64 @test36(i64* %p) {
; CHECK-LABEL: test36
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, i64*
  %a = getelementptr inbounds i64, i64* %p, i32 -512
  %v = load i64, i64* %a
  ret i64 %v
}

; CHECK-LABEL: test37
; CHECK: cost of 1 for instruction:  {{.*}} = getelementptr inbounds i8*, i8**
define i8 @test37(i64 %j, i8** readonly %P) {
entry:
  %arrayidx0 = getelementptr inbounds i8*, i8** %P, i64 %j
  %l1 = call i8* @func(i8** %arrayidx0)
  ret i8 0
}

; CHECK-LABEL: test38
; CHECK: cost of 1 for instruction: {{.*}} = getelementptr inbounds i8*, i8**
define i8 @test38(i8** readonly %P) {
entry:
  %arrayidx0 = getelementptr inbounds i8*, i8** %P, i64 10
  %l1 = call i8* @func(i8** %arrayidx0)
  ret i8 0
}

; CHECK-LABEL:test39
; CHECK: cost of 0 for instruction: {{.*}} = getelementptr inbounds i8*, i8**
define i8 @test39(i8** readonly %P) {
entry:
  %arrayidx0 = getelementptr inbounds i8*, i8** %P, i64 0
  %l1 = call i8* @func(i8** %arrayidx0)
  ret i8 0
}

; CHECK-LABEL:test40
; CHECK: cost of 1 for instruction: {{.*}} = getelementptr inbounds i8*, i8**
define i8** @test40(i8** readonly %P) {
entry:
  %arrayidx0 = getelementptr inbounds i8*, i8** %P, i64 10
  ret i8** %arrayidx0
}

; CHECK-LABEL:test41
; CHECK: cost of 1 for instruction: {{.*}} = getelementptr inbounds i8, i8*
define i8 @test41(i8* %V, i8** readonly %P) {
entry:
  %arrayidx0 = getelementptr inbounds i8, i8* %V, i64 10
  store i8* %arrayidx0, i8** %P
  ret i8 0
}

declare i8* @func(i8**)
