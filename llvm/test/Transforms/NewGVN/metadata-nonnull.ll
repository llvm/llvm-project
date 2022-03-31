; RUN: opt %s -passes=newgvn -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @test1(i8** %v0, i8** %v1) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  top:
; CHECK-NEXT:    [[V2:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]], align 8, !nonnull !0
; CHECK-NEXT:    store i8* [[V2]], i8** [[V1:%.*]]
; CHECK-NEXT:    ret i8* [[V2]]
;
top:
  %v2 = load i8*, i8** %v0, !nonnull !0
  store i8* %v2, i8** %v1
  %v3 = load i8*, i8** %v1
  ret i8* %v3
}

; FIXME: could propagate nonnull to first load?
define i8* @test2(i8** %v0, i8** %v1) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  top:
; CHECK-NEXT:    [[V2:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    store i8* [[V2]], i8** [[V1:%.*]]
; CHECK-NEXT:    ret i8* [[V2]]
;
top:
  %v2 = load i8*, i8** %v0
  store i8* %v2, i8** %v1
  %v3 = load i8*, i8** %v1, !nonnull !0
  ret i8* %v3
}

declare void @use1(i8* %a) readonly

define i8* @test3(i8** %v0) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:  top:
; CHECK-NEXT:    [[V1:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    call void @use1(i8* [[V1]])
; CHECK-NEXT:    br i1 undef, label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    ret i8* [[V1]]
; CHECK:       bb2:
; CHECK-NEXT:    ret i8* [[V1]]
;
top:
  %v1 = load i8*, i8** %v0
  call void @use1(i8* %v1)
  br i1 undef, label %bb1, label %bb2

bb1:
  %v2 = load i8*, i8** %v0, !nonnull !0
  ret i8* %v2

bb2:
  %v3 = load i8*, i8** %v0
  ret i8* %v3
}

define i8* @test4(i8** %v0) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:  top:
; CHECK-NEXT:    [[V1:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    call void @use1(i8* [[V1]])
; CHECK-NEXT:    br i1 undef, label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    ret i8* [[V1]]
; CHECK:       bb2:
; CHECK-NEXT:    ret i8* [[V1]]
;
top:
  %v1 = load i8*, i8** %v0
  call void @use1(i8* %v1)
  br i1 undef, label %bb1, label %bb2

bb1:
  %v2 = load i8*, i8** %v0
  ret i8* %v2

bb2:
  %v3 = load i8*, i8** %v0, !nonnull !0
  ret i8* %v3
}

define i8* @test5(i8** %v0) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:  top:
; CHECK-NEXT:    [[V1:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]], align 8, !nonnull !0
; CHECK-NEXT:    call void @use1(i8* [[V1]])
; CHECK-NEXT:    br i1 undef, label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    ret i8* [[V1]]
; CHECK:       bb2:
; CHECK-NEXT:    ret i8* [[V1]]
;
top:
  %v1 = load i8*, i8** %v0, !nonnull !0
  call void @use1(i8* %v1)
  br i1 undef, label %bb1, label %bb2

bb1:
  %v2 = load i8*, i8** %v0
  ret i8* %v2

bb2:
  %v3 = load i8*, i8** %v0
  ret i8* %v3
}

define i8* @test6(i8** %v0, i8** %v1) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:  top:
; CHECK-NEXT:    br i1 undef, label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[V2:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]], align 8, !nonnull !0
; CHECK-NEXT:    store i8* [[V2]], i8** [[V1:%.*]]
; CHECK-NEXT:    ret i8* [[V2]]
; CHECK:       bb2:
; CHECK-NEXT:    [[V4:%.*]] = load i8*, i8** [[V0]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    store i8* [[V4]], i8** [[V1]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    ret i8* [[V4]]
;
top:
  br i1 undef, label %bb1, label %bb2

bb1:
  %v2 = load i8*, i8** %v0, !nonnull !0
  store i8* %v2, i8** %v1
  %v3 = load i8*, i8** %v1
  ret i8* %v3

bb2:
  %v4 = load i8*, i8** %v0
  store i8* %v4, i8** %v1
  %v5 = load i8*, i8** %v1, !nonnull !0
  ret i8* %v5
}

declare void @use2(i8* %a)

define i8* @test7(i8** %v0) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:  top:
; CHECK-NEXT:    [[V1:%.*]] = load i8*, i8** [[V0:%[a-z0-9]+]], align 8, !nonnull !0
; CHECK-NEXT:    call void @use2(i8* [[V1]])
; CHECK-NEXT:    br i1 undef, label [[BB1:%.*]], label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[V2:%.*]] = load i8*, i8** [[V0]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    ret i8* [[V2]]
; CHECK:       bb2:
; CHECK-NEXT:    [[V3:%.*]] = load i8*, i8** [[V0]]
; CHECK-NOT:     !nonnull
; CHECK-NEXT:    ret i8* [[V3]]
;
top:
  %v1 = load i8*, i8** %v0, !nonnull !0
  call void @use2(i8* %v1)
  br i1 undef, label %bb1, label %bb2

bb1:
  %v2 = load i8*, i8** %v0
  ret i8* %v2

bb2:
  %v3 = load i8*, i8** %v0
  ret i8* %v3
}

!0 = !{}
