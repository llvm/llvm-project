; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64--linux-gnu < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

define i8 @test1(ptr %p) {
; CHECK-LABEL: test1
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 1
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test2(ptr %p) {
; CHECK-LABEL: test2
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 1
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test3(ptr %p) {
; CHECK-LABEL: test3
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 1
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test4(ptr %p) {
; CHECK-LABEL: test4
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 1
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test5(ptr %p) {
; CHECK-LABEL: test5
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 1024
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test6(ptr %p) {
; CHECK-LABEL: test6
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 1024
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test7(ptr %p) {
; CHECK-LABEL: test7
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 1024
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test8(ptr %p) {
; CHECK-LABEL: test8
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 1024
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test9(ptr %p) {
; CHECK-LABEL: test9
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 4096
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test10(ptr %p) {
; CHECK-LABEL: test10
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 4096
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test11(ptr %p) {
; CHECK-LABEL: test11
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 4096
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test12(ptr %p) {
; CHECK-LABEL: test12
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 4096
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test13(ptr %p) {
; CHECK-LABEL: test13
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 -64
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test14(ptr %p) {
; CHECK-LABEL: test14
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 -64
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test15(ptr %p) {
; CHECK-LABEL: test15
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 -64
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test16(ptr %p) {
; CHECK-LABEL: test16
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 -64
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test17(ptr %p) {
; CHECK-LABEL: test17
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 -1024
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test18(ptr %p) {
; CHECK-LABEL: test18
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 -1024
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test19(ptr %p) {
; CHECK-LABEL: test19
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 -1024
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test20(ptr %p) {
; CHECK-LABEL: test20
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 -1024
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test21(ptr %p, i32 %i) {
; CHECK-LABEL: test21
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 %i
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test22(ptr %p, i32 %i) {
; CHECK-LABEL: test22
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 %i
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test23(ptr %p, i32 %i) {
; CHECK-LABEL: test23
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 %i
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test24(ptr %p, i32 %i) {
; CHECK-LABEL: test24
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 %i
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test25(ptr %p) {
; CHECK-LABEL: test25
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 -128
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test26(ptr %p) {
; CHECK-LABEL: test26
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 -128
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test27(ptr %p) {
; CHECK-LABEL: test27
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 -128
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test28(ptr %p) {
; CHECK-LABEL: test28
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 -128
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test29(ptr %p) {
; CHECK-LABEL: test29
; CHECK: cost of 0 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 -256
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test30(ptr %p) {
; CHECK-LABEL: test30
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 -256
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test31(ptr %p) {
; CHECK-LABEL: test31
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 -256
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test32(ptr %p) {
; CHECK-LABEL: test32
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 -256
  %v = load i64, ptr %a
  ret i64 %v
}

define i8 @test33(ptr %p) {
; CHECK-LABEL: test33
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i8, ptr
  %a = getelementptr inbounds i8, ptr %p, i32 -512
  %v = load i8, ptr %a
  ret i8 %v
}

define i16 @test34(ptr %p) {
; CHECK-LABEL: test34
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i16, ptr
  %a = getelementptr inbounds i16, ptr %p, i32 -512
  %v = load i16, ptr %a
  ret i16 %v
}

define i32 @test35(ptr %p) {
; CHECK-LABEL: test35
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i32, ptr
  %a = getelementptr inbounds i32, ptr %p, i32 -512
  %v = load i32, ptr %a
  ret i32 %v
}

define i64 @test36(ptr %p) {
; CHECK-LABEL: test36
; CHECK: cost of 1 for instruction: {{.*}} getelementptr inbounds i64, ptr
  %a = getelementptr inbounds i64, ptr %p, i32 -512
  %v = load i64, ptr %a
  ret i64 %v
}
