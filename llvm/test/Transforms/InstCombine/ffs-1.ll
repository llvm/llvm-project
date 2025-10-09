; Test that the ffs* library call simplifier works correctly.
;
; RUN: opt < %s -passes=instcombine -S                                    | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC
; RUN: opt < %s -passes=instcombine -mtriple i386-pc-linux -S             | FileCheck %s --check-prefix=ALL --check-prefix=TARGET
; RUN: opt < %s -passes=instcombine -mtriple=arm64-apple-ios9.0 -S        | FileCheck %s --check-prefix=ALL --check-prefix=TARGET
; RUN: opt < %s -passes=instcombine -mtriple=arm64-apple-tvos9.0 -S       | FileCheck %s --check-prefix=ALL --check-prefix=TARGET
; RUN: opt < %s -passes=instcombine -mtriple=thumbv7k-apple-watchos2.0 -S | FileCheck %s --check-prefix=ALL --check-prefix=TARGET
; RUN: opt < %s -passes=instcombine -mtriple=x86_64-apple-macosx10.11 -S  | FileCheck %s --check-prefix=ALL --check-prefix=TARGET
; RUN: opt < %s -passes=instcombine -mtriple=x86_64-freebsd-gnu -S        | FileCheck %s --check-prefix=ALL --check-prefix=TARGET

declare i32 @ffs(i32)
declare i32 @ffsl(i32)
declare i32 @ffsll(i64)

; Check ffs(0) -> 0.

define i32 @test_simplify1() {
; ALL-LABEL: @test_simplify1(
; ALL-NEXT:    ret i32 0
;
  %ret = call i32 @ffs(i32 0)
  ret i32 %ret
}

define i32 @test_simplify2() {
; GENERIC-LABEL: @test_simplify2(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsl(i32 0)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify2(
; TARGET-NEXT:    ret i32 0
;
  %ret = call i32 @ffsl(i32 0)
  ret i32 %ret
}

define i32 @test_simplify3() {
; GENERIC-LABEL: @test_simplify3(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 0)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify3(
; TARGET-NEXT:    ret i32 0
;
  %ret = call i32 @ffsll(i64 0)
  ret i32 %ret
}

; Check ffs(c) -> cttz(c) + 1, where 'c' is a constant.

define i32 @test_simplify4() {
; ALL-LABEL: @test_simplify4(
; ALL-NEXT:    ret i32 1
;
  %ret = call i32 @ffs(i32 1)
  ret i32 %ret
}

define i32 @test_simplify5() {
; ALL-LABEL: @test_simplify5(
; ALL-NEXT:    ret i32 12
;
  %ret = call i32 @ffs(i32 2048)
  ret i32 %ret
}

define i32 @test_simplify6() {
; ALL-LABEL: @test_simplify6(
; ALL-NEXT:    ret i32 17
;
  %ret = call i32 @ffs(i32 65536)
  ret i32 %ret
}

define i32 @test_simplify7() {
; GENERIC-LABEL: @test_simplify7(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsl(i32 65536)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify7(
; TARGET-NEXT:    ret i32 17
;
  %ret = call i32 @ffsl(i32 65536)
  ret i32 %ret
}

define i32 @test_simplify8() {
; GENERIC-LABEL: @test_simplify8(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 1024)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify8(
; TARGET-NEXT:    ret i32 11
;
  %ret = call i32 @ffsll(i64 1024)
  ret i32 %ret
}

define i32 @test_simplify9() {
; GENERIC-LABEL: @test_simplify9(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 65536)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify9(
; TARGET-NEXT:    ret i32 17
;
  %ret = call i32 @ffsll(i64 65536)
  ret i32 %ret
}

define i32 @test_simplify10() {
; GENERIC-LABEL: @test_simplify10(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 17179869184)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify10(
; TARGET-NEXT:    ret i32 35
;
  %ret = call i32 @ffsll(i64 17179869184)
  ret i32 %ret
}

define i32 @test_simplify11() {
; GENERIC-LABEL: @test_simplify11(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 281474976710656)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify11(
; TARGET-NEXT:    ret i32 49
;
  %ret = call i32 @ffsll(i64 281474976710656)
  ret i32 %ret
}

define i32 @test_simplify12() {
; GENERIC-LABEL: @test_simplify12(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 1152921504606846976)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify12(
; TARGET-NEXT:    ret i32 61
;
  %ret = call i32 @ffsll(i64 1152921504606846976)
  ret i32 %ret
}

; Check ffs(x) -> x != 0 ? (i32)llvm.cttz(x) + 1 : 0.

define i32 @test_simplify13(i32 %x) {
; ALL-LABEL: @test_simplify13(
; ALL-NEXT:    [[CTTZ:%.*]] = call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %x, i1 true)
; ALL-NEXT:    [[TMP1:%.*]] = add nuw nsw i32 [[CTTZ]], 1
; ALL-NEXT:    [[TMP2:%.*]] = icmp eq i32 %x, 0
; ALL-NEXT:    [[TMP3:%.*]] = select i1 [[TMP2]], i32 0, i32 [[TMP1]]
; ALL-NEXT:    ret i32 [[TMP3]]
;
  %ret = call i32 @ffs(i32 %x)
  ret i32 %ret
}

define i32 @test_simplify14(i32 %x) {
; GENERIC-LABEL: @test_simplify14(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsl(i32 %x)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify14(
; TARGET-NEXT:    [[CTTZ:%.*]] = call range(i32 0, 33) i32 @llvm.cttz.i32(i32 %x, i1 true)
; TARGET-NEXT:    [[TMP1:%.*]] = add nuw nsw i32 [[CTTZ]], 1
; TARGET-NEXT:    [[TMP2:%.*]] = icmp eq i32 %x, 0
; TARGET-NEXT:    [[TMP3:%.*]] = select i1 [[TMP2]], i32 0, i32 [[TMP1]]
; TARGET-NEXT:    ret i32 [[TMP3]]
;
  %ret = call i32 @ffsl(i32 %x)
  ret i32 %ret
}

define i32 @test_simplify15(i64 %x) {
; GENERIC-LABEL: @test_simplify15(
; GENERIC-NEXT:    [[RET:%.*]] = call i32 @ffsll(i64 %x)
; GENERIC-NEXT:    ret i32 [[RET]]
;
; TARGET-LABEL: @test_simplify15(
; TARGET-NEXT:    [[CTTZ:%.*]] = call range(i64 0, 65) i64 @llvm.cttz.i64(i64 %x, i1 true)
; TARGET-NEXT:    [[TMP1:%.*]] = trunc nuw nsw i64 [[CTTZ]] to i32
; TARGET-NEXT:    [[TMP2:%.*]] = add nuw nsw i32 [[TMP1]], 1
; TARGET-NEXT:    [[TMP3:%.*]] = icmp eq i64 %x, 0
; TARGET-NEXT:    [[TMP4:%.*]] = select i1 [[TMP3]], i32 0, i32 [[TMP2]]
; TARGET-NEXT:    ret i32 [[TMP4]]
;
  %ret = call i32 @ffsll(i64 %x)
  ret i32 %ret
}

