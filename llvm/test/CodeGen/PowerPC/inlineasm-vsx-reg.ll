; RUN: llc -verify-machineinstrs -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names  -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s
define signext i32 @foo(<4 x float> %__A) {
entry:
  %0 = tail call { i32, <4 x float> } asm "xxsldwi ${1:x},${2:x},${2:x},3;\0Axscvspdp ${1:x},${1:x};\0Afctiw  $1,$1;\0Amfvsrd  $0,${1:x};\0A", "=r,=&^wa,^wa"(<4 x float> %__A)
  %asmresult = extractvalue { i32, <4 x float> } %0, 0
  ret i32 %asmresult
; CHECK-LABEL: foo:
; CHECK: #APP
; CHECK: xxsldwi vs0, v2, v2, 3
; CHECK: xscvspdp f0, f0
; CHECK: fctiw f0, f0
; CHECK: mffprd r3, f0
; CHECK: #NO_APP
}

define signext i32 @foo1(<4 x float> %__A) {
entry:
  %0 = tail call { i32, <4 x float> } asm "xxsldwi ${1:x},${2:x},${2:x},3;\0Axscvspdp ${1:x},${1:x};\0Afctiw  $1,$1;\0Amfvsrd  $0,${1:x};\0A", "=r,=&^wi,^wa"(<4 x float> %__A)
  %asmresult = extractvalue { i32, <4 x float> } %0, 0
  ret i32 %asmresult

; CHECK-LABEL: foo1:
; CHECK: #APP
; CHECK: xxsldwi vs0, v2, v2, 3
; CHECK: xscvspdp f0, f0
; CHECK: fctiw f0, f0
; CHECK: mffprd r3, f0
; CHECK: #NO_APP
; CHECK: extsw r3, r3
}

define double @test() {
  entry:
    %0 = tail call double asm "mtvsrd ${0:x}, 1", "=^ws,~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14}"()
    ret double %0
; CHECK-LABEL: test:
; CHECK: #APP
; CHECK: mtvsrd v2, r1
; CHECK: #NO_APP
}

define float @test_ww(float %x, float %y) {
  %1 = tail call float asm "xsmaxdp ${0:x}, ${1:x}, ${2:x}", "=^ww,^ww,^ww"(float %x, float %y)
  ret float %1
; CHECK-LABEL: test_ww:
; CHECK: #APP
; CHECK: xsmaxdp f1, f1, f2
; CHECK: #NO_APP
}
