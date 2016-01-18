; RUN: opt < %s -constprop -S | FileCheck %s

%struct = type { i32, [4 x i8] }

define %struct @test1() {
  %A = insertvalue %struct { i32 2, [4 x i8] c"foo\00" }, i32 1, 0
  ret %struct %A
; CHECK-LABEL: @test1(
; CHECK: ret %struct { i32 1, [4 x i8] c"foo\00" }
}

define %struct @test2() {
  %A = insertvalue %struct { i32 2, [4 x i8] c"foo\00" }, i8 1, 1, 2
  ret %struct %A
; CHECK-LABEL: @test2(
; CHECK: ret %struct { i32 2, [4 x i8] c"fo\01\00" }
}

define [3 x %struct] @test3() {
  %A = insertvalue [3 x %struct] [ %struct { i32 0, [4 x i8] c"aaaa" }, %struct { i32 1, [4 x i8] c"bbbb" }, %struct { i32 2, [4 x i8] c"cccc" } ], i32 -1, 1, 0
  ret [3 x %struct] %A
; CHECK-LABEL: @test3(
; CHECK:ret [3 x %struct] [%struct { i32 0, [4 x i8] c"aaaa" }, %struct { i32 -1, [4 x i8] c"bbbb" }, %struct { i32 2, [4 x i8] c"cccc" }]
}

define %struct @zeroinitializer-test1() {
  %A = insertvalue %struct zeroinitializer, i32 1, 0
  ret %struct %A
; CHECK: @zeroinitializer-test1
; CHECK: ret %struct { i32 1, [4 x i8] zeroinitializer }
}

define %struct @zeroinitializer-test2() {
  %A = insertvalue %struct zeroinitializer, i8 1, 1, 2
  ret %struct %A
; CHECK: @zeroinitializer-test2
; CHECK: ret %struct { i32 0, [4 x i8] c"\00\00\01\00" }
}

define [3 x %struct] @zeroinitializer-test3() {
  %A = insertvalue [3 x %struct] zeroinitializer, i32 1, 1, 0
  ret [3 x %struct] %A
; CHECK: @zeroinitializer-test3
; CHECK: ret [3 x %struct] [%struct zeroinitializer, %struct { i32 1, [4 x i8] zeroinitializer }, %struct zeroinitializer]
}

define %struct @undef-test1() {
  %A = insertvalue %struct undef, i32 1, 0
  ret %struct %A
; CHECK: @undef-test1
; CHECK: ret %struct { i32 1, [4 x i8] undef }
}

define %struct @undef-test2() {
  %A = insertvalue %struct undef, i8 0, 1, 2
  ret %struct %A
; CHECK: @undef-test2
; CHECK: ret %struct { i32 undef, [4 x i8] [i8 undef, i8 undef, i8 0, i8 undef] }
}

define [3 x %struct] @undef-test3() {
  %A = insertvalue [3 x %struct] undef, i32 0, 1, 0
  ret [3 x %struct] %A
; CHECK: @undef-test3
; CHECK: ret [3 x %struct] [%struct undef, %struct { i32 0, [4 x i8] undef }, %struct undef]
}

define i32 @test-float-Nan() {
  %A = bitcast i32 2139171423 to float
  %B = insertvalue [1 x float] undef, float %A, 0
  %C = extractvalue [1 x float] %B, 0
  %D = bitcast float %C to i32
  ret i32 %D
; CHECK: @test-float-Nan
; CHECK: ret i32 2139171423
}
