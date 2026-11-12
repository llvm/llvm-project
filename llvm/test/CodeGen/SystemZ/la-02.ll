; Test loads of symbolic addresses when generating medium- and
; large-model non-PIC.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -code-model=medium | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -code-model=large | FileCheck %s

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

; Test loads of external variables.  There is no guarantee that the
; variable will be in range of LARL.
define ptr@f1() {
; CHECK-LABEL: f1:
; CHECK: lgrl %r2, ev@GOT
; CHECK: br %r14
  ret ptr@ev
}

; ...likewise locally-defined normal-visibility variables.
define ptr@f2() {
; CHECK-LABEL: f2:
; CHECK: lgrl %r2, dv@GOT
; CHECK: br %r14
  ret ptr@dv
}

; ...likewise protected variables.
define ptr@f3() {
; CHECK-LABEL: f3:
; CHECK: lgrl %r2, pv@GOT
; CHECK: br %r14
  ret ptr@pv
}

; ...likewise hidden variables.
define ptr@f4() {
; CHECK-LABEL: f4:
; CHECK: lgrl %r2, hv@GOT
; CHECK: br %r14
  ret ptr@hv
}

; Check loads of external functions.  This could use LARL, but we don't have
; code to detect that yet.
define ptr@f5() {
; CHECK-LABEL: f5:
; CHECK: lgrl %r2, ef@GOT
; CHECK: br %r14
  ret ptr@ef
}

; ...likewise locally-defined normal-visibility functions.
define ptr@f6() {
; CHECK-LABEL: f6:
; CHECK: lgrl %r2, df@GOT
; CHECK: br %r14
  ret ptr@df
}

; ...likewise protected functions.
define ptr@f7() {
; CHECK-LABEL: f7:
; CHECK: lgrl %r2, pf@GOT
; CHECK: br %r14
  ret ptr@pf
}

; ...likewise hidden functions.
define ptr@f8() {
; CHECK-LABEL: f8:
; CHECK: lgrl %r2, hf@GOT
; CHECK: br %r14
  ret ptr@hf
}
