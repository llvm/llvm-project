; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target triple = "x86_64-unknown-freebsd11.0"

define i32 @myfls() {
; CHECK-LABEL: @myfls(
; CHECK-NEXT:    ret i32 6
;
  %call = call i32 @fls(i32 42)
  ret i32 %call
}

define i32 @myflsl() {
; CHECK-LABEL: @myflsl(
; CHECK-NEXT:    ret i32 6
;
  %patatino = call i32 @flsl(i64 42)
  ret i32 %patatino
}

define i32 @myflsll() {
; CHECK-LABEL: @myflsll(
; CHECK-NEXT:    ret i32 6
;
  %whatever = call i32 @flsll(i64 42)
  ret i32 %whatever
}

; Lower to llvm.ctlz() if the argument is not a constant

define i32 @flsnotconst(i64 %z) {
; CHECK-LABEL: @flsnotconst(
; CHECK-NEXT:    [[CTLZ:%.*]] = call i64 @llvm.ctlz.i64(i64 %z, i1 false), !range !0
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i64 [[CTLZ]] to i32
; CHECK-NEXT:    [[TMP2:%.*]] = sub nsw i32 64, [[TMP1]]
; CHECK-NEXT:    ret i32 [[TMP2]]
;
  %goo = call i32 @flsl(i64 %z)
  ret i32 %goo
}

; Make sure we lower fls(0) to 0 and not to `undef`.

define i32 @flszero() {
; CHECK-LABEL: @flszero(
; CHECK-NEXT:    ret i32 0
;
  %zero = call i32 @fls(i32 0)
  ret i32 %zero
}

declare i32 @fls(i32)
declare i32 @flsl(i64)
declare i32 @flsll(i64)
