; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

;CHECK: @func30
;CHECK: movi.4h v1, #1
;CHECK: and.8b v0, v0, v1
;CHECK: ushll.4s  v0, v0, #0
;CHECK: str  q0, [x0]
;CHECK: ret

%T0_30 = type <4 x i1>
%T1_30 = type <4 x i32>
define void @func30(%T0_30 %v0, %T1_30* %p1) {
  %r = zext %T0_30 %v0 to %T1_30
  store %T1_30 %r, %T1_30* %p1
  ret void
}

; Extend from v1i1 was crashing things (PR20791). Make sure we do something
; sensible instead.
define <1 x i32> @autogen_SD7918() {
; CHECK-LABEL: autogen_SD7918
; CHECK: movi.2d v0, #0000000000000000
; CHECK-NEXT: ret
  %I29 = insertelement <1 x i1> zeroinitializer, i1 false, i32 0
  %ZE = zext <1 x i1> %I29 to <1 x i32>
  ret <1 x i32> %ZE
}
