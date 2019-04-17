; RUN: opt < %s -constprop -S | FileCheck %s

%struct = type { i32, [4 x i8] }

define i32 @test1() {
  %A = extractvalue %struct { i32 2, [4 x i8] c"foo\00" }, 0
  ret i32 %A
; CHECK-LABEL: @test1(
; CHECK: ret i32 2
}

define i8 @test2() {
  %A = extractvalue %struct { i32 2, [4 x i8] c"foo\00" }, 1, 2
  ret i8 %A
; CHECK-LABEL: @test2(
; CHECK: ret i8 111
}

define i32 @test3() {
  %A = extractvalue [3 x %struct] [ %struct { i32 0, [4 x i8] c"aaaa" }, %struct { i32 1, [4 x i8] c"bbbb" }, %struct { i32 2, [4 x i8] c"cccc" } ], 1, 0
  ret i32 %A
; CHECK-LABEL: @test3(
; CHECK: ret i32 1
}

define i32 @zeroinitializer-test1() {
  %A = extractvalue %struct zeroinitializer, 0
  ret i32 %A
; CHECK: @zeroinitializer-test1
; CHECK: ret i32 0
}

define i8 @zeroinitializer-test2() {
  %A = extractvalue %struct zeroinitializer, 1, 2
  ret i8 %A
; CHECK: @zeroinitializer-test2
; CHECK: ret i8 0
}

define i32 @zeroinitializer-test3() {
  %A = extractvalue [3 x %struct] zeroinitializer, 1, 0
  ret i32 %A
; CHECK: @zeroinitializer-test3
; CHECK: ret i32 0
}

define i32 @undef-test1() {
  %A = extractvalue %struct undef, 0
  ret i32 %A
; CHECK: @undef-test1
; CHECK: ret i32 undef
}

define i8 @undef-test2() {
  %A = extractvalue %struct undef, 1, 2
  ret i8 %A
; CHECK: @undef-test2
; CHECK: ret i8 undef
}

define i32 @undef-test3() {
  %A = extractvalue [3 x %struct] undef, 1, 0
  ret i32 %A
; CHECK: @undef-test3
; CHECK: ret i32 undef
}

