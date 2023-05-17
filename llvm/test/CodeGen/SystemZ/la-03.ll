; Test loads of symbolic addresses in PIC code.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -relocation-model=pic | FileCheck %s

@ev = external global i32
@dv = global i32 0
@pv = protected global i32 0
@hv = hidden global i32 0

declare void @ef()
define void @df() {
  ret void
}
define protected void @pf() {
  ret void
}
define hidden void @hf() {
  ret void
}

; Test loads of external variables, which must go via the GOT.
define ptr@f1() {
; CHECK-LABEL: f1:
; CHECK: lgrl %r2, ev@GOT
; CHECK: br %r14
  ret ptr@ev
}

; Check loads of locally-defined normal-visibility variables, which might
; be overridden.  The load must go via the GOT.
define ptr@f2() {
; CHECK-LABEL: f2:
; CHECK: lgrl %r2, dv@GOT
; CHECK: br %r14
  ret ptr@dv
}

; Check loads of protected variables, which in the small code model
; must be in range of LARL.
define ptr@f3() {
; CHECK-LABEL: f3:
; CHECK: larl %r2, pv
; CHECK: br %r14
  ret ptr@pv
}

; ...likewise hidden variables.
define ptr@f4() {
; CHECK-LABEL: f4:
; CHECK: larl %r2, hv
; CHECK: br %r14
  ret ptr@hv
}

; Like f1, but for functions.
define ptr@f5() {
; CHECK-LABEL: f5:
; CHECK: lgrl %r2, ef@GOT
; CHECK: br %r14
  ret ptr@ef
}

; Like f2, but for functions.
define ptr@f6() {
; CHECK-LABEL: f6:
; CHECK: lgrl %r2, df@GOT
; CHECK: br %r14
  ret ptr@df
}

; Like f3, but for functions.
define ptr@f7() {
; CHECK-LABEL: f7:
; CHECK: larl %r2, pf
; CHECK: br %r14
  ret ptr@pf
}

; Like f4, but for functions.
define ptr@f8() {
; CHECK-LABEL: f8:
; CHECK: larl %r2, hf
; CHECK: br %r14
  ret ptr@hf
}
