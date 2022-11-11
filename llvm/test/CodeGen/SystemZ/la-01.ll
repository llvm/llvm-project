; Test loads of symbolic addresses when generating small-model non-PIC.
; All addresses can be treated as PC
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@e4 = external dso_local global i32
@d4 = dso_local global i32 1
@e2 = external dso_local global i32, align 2
@d2 = dso_local global i32 1, align 2
@e1 = external dso_local global i32, align 1
@d1 = dso_local global i32 1, align 1

declare dso_local void @ef()
define dso_local void @df() {
  ret void
}

declare dso_local void @foo(ptr)

; Test a load of a fully-aligned external variable.
define dso_local ptr@f1() {
; CHECK-LABEL: f1:
; CHECK: larl %r2, e4
; CHECK-NEXT: br %r14
  ret ptr@e4
}

; Test a load of a fully-aligned local variable.
define dso_local ptr@f2() {
; CHECK-LABEL: f2:
; CHECK: larl %r2, d4
; CHECK-NEXT: br %r14
  ret ptr@d4
}

; Test a load of a 2-byte-aligned external variable.
define dso_local ptr@f3() {
; CHECK-LABEL: f3:
; CHECK: larl %r2, e2
; CHECK-NEXT: br %r14
  ret ptr@e2
}

; Test a load of a 2-byte-aligned local variable.
define dso_local ptr@f4() {
; CHECK-LABEL: f4:
; CHECK: larl %r2, d2
; CHECK-NEXT: br %r14
  ret ptr@d2
}

; Test a load of an unaligned external variable, which must go via the GOT.
define dso_local ptr@f5() {
; CHECK-LABEL: f5:
; CHECK: lgrl %r2, e1@GOT
; CHECK-NEXT: br %r14
  ret ptr@e1
}

; Test a load of an unaligned local variable, which must go via the GOT.
define dso_local ptr@f6() {
; CHECK-LABEL: f6:
; CHECK: lgrl %r2, d1@GOT
; CHECK-NEXT: br %r14
  ret ptr@d1
}

; Test a load of an external function.
define dso_local ptr@f7() {
; CHECK-LABEL: f7:
; CHECK: larl %r2, ef
; CHECK-NEXT: br %r14
  ret ptr@ef
}

; Test a load of a local function.
define dso_local ptr@f8() {
; CHECK-LABEL: f8:
; CHECK: larl %r2, df
; CHECK-NEXT: br %r14
  ret ptr@df
}

; Test that LARL can be rematerialized.
define dso_local i32 @f9() {
; CHECK-LABEL: f9:
; CHECK: larl %r2, d2
; CHECK: brasl %r14, foo@PLT
; CHECK: larl %r2, d2
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  call void @foo(ptr@d2)
  call void @foo(ptr@d2)
  ret i32 0
}
