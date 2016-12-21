; RUN: opt < %s -instcombine -S | FileCheck %s

target triple = "x86_64-unknown-freebsd11.0"


; CHECK-LABEL: define i32 @myfls(
; CHECK: ret i32 6
; CHECK: }

define i32 @myfls() {
entry:
  %call = call i32 @fls(i32 42)
  ret i32 %call
}

; CHECK-LABEL: define i32 @myflsl(
; CHECK: ret i32 6
; CHECK: }

define i32 @myflsl() {
  %patatino = call i32 @flsl(i64 42)
  ret i32 %patatino
}

; CHECK-LABEL: define i32 @myflsll(
; CHECK: ret i32 6
; CHECK: }

define i32 @myflsll() {
  %whatever = call i32 @flsll(i64 42)
  ret i32 %whatever
}

; Lower to llvm.ctlz() if the argument is not a constant
; CHECK-LABEL: define i32 @flsnotconst(
; CHECK-NEXT:  %ctlz = call i64 @llvm.ctlz.i64(i64 %z, i1 false)
; CHECK-NEXT:  %1 = sub nsw i64 64, %ctlz
; CHECK-NEXT:  %2 = trunc i64 %1 to i32
; CHECK-NEXT:  ret i32 %2

define i32 @flsnotconst(i64 %z) {
  %goo = call i32 @flsl(i64 %z)
  ret i32 %goo
}

; Make sure we lower fls(0) to 0 and not to `undef`.
; CHECK-LABEL: define i32 @flszero(
; CHECK: ret i32 0
; CHECK: }
define i32 @flszero() {
  %zero = call i32 @fls(i32 0)
  ret i32 %zero
}

declare i32 @fls(i32)
declare i32 @flsl(i64)
declare i32 @flsll(i64)
